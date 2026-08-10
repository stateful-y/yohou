"""Tests for DecompositionPipeline meta-forecaster."""

from datetime import datetime, timedelta

import polars as pl
import pytest
from sklearn.base import clone

from conftest import run_checks
from yohou.compose import DecompositionPipeline
from yohou.point import SeasonalNaive
from yohou.stationarity import LogTransformer, PatternSeasonalityForecaster, PolynomialTrendForecaster
from yohou.testing import _yield_yohou_forecaster_checks


class TestSystematicChecks:
    """Systematic validation of DecompositionPipeline via check generators."""

    @pytest.mark.parametrize(
        "forecaster",
        [
            DecompositionPipeline([
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("seasonality", SeasonalNaive(seasonality=7)),
            ]),
            DecompositionPipeline([
                ("trend", PolynomialTrendForecaster(degree=2)),
                ("seasonality", PatternSeasonalityForecaster(seasonality=12, method="average")),
            ]),
            DecompositionPipeline(
                [
                    ("trend", PolynomialTrendForecaster(degree=1)),
                    ("seasonality", SeasonalNaive(seasonality=7)),
                ],
                target_transformer=LogTransformer(),
            ),
            DecompositionPipeline([
                ("trend", PolynomialTrendForecaster(degree=1)),
            ]),
        ],
    )
    @pytest.mark.slow
    def test_checks(self, forecaster, y_X_factory):
        """Run systematic checks on DecompositionPipeline.

        Note: check_observe_extends_observations and check_rewind_replaces_observations
        are not yielded because DecompositionPipeline sets tracks_observations=False.
        """
        y, X_actual = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        y = y.with_columns([(pl.col(col).abs() + 1).alias(col) for col in y.columns if col != "time"])

        y_train, y_test = y[:80], y[80:]
        X_actual_train, X_actual_test = (X_actual[:80], X_actual[80:]) if X_actual is not None else (None, None)

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_actual_train, forecasting_horizon=3)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, X_actual_train, y_test, X_actual_test),
        )


class TestPanelObserveRewind:
    """Panel-mode observe/rewind must not crash on the scalar-transformer assumption.

    In panel mode the base class stores ``target_transformer_``/
    ``actual_transformer_`` as dicts keyed by group, so ``observe``/``rewind``
    must branch on ``groups_`` instead of asserting a single ``BaseActualTransformer``
    (which would raise ``AssertionError``).
    """

    @pytest.fixture
    def panel_data(self):
        """Two-group strictly-positive linear panel (LogTransformer-safe)."""
        n = 60
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=n - 1),
            interval="1d",
            eager=True,
        )
        data = {"time": time}
        for g in range(2):
            slope = g + 1
            data[f"g{g}__value"] = [slope * t + 10.0 * (g + 1) for t in range(n)]
        return pl.DataFrame(data)

    def test_panel_observe_then_rewind_with_target_transformer(self, panel_data):
        y = panel_data
        y_train, y_new = y[:50], y[50:]
        forecaster = DecompositionPipeline(
            [("trend", PolynomialTrendForecaster(degree=1))],
            target_transformer=LogTransformer(),
        )
        forecaster.fit(y_train, forecasting_horizon=3)

        forecaster.observe(y_new)
        y_pred = forecaster.predict()
        assert "g0__value" in y_pred.columns
        assert "g1__value" in y_pred.columns

        forecaster.rewind(y_train)
        y_pred_rewound = forecaster.predict()
        assert "g0__value" in y_pred_rewound.columns
        assert "g1__value" in y_pred_rewound.columns

    @pytest.mark.parametrize("with_shared", [False, True])
    def test_panel_observe_then_rewind_with_actual_transformer(self, panel_data, with_shared):
        """Panel observe/rewind must apply the per-group feature transformer.

        Covers the ``actual_transformer_`` panel branch (and the shared
        ``_panel_X_actual_schema`` helper): with X_actual present in panel
        mode, ``observe``/``rewind`` build the per-group transform dict instead
        of asserting a single ``BaseActualTransformer``. The ``with_shared`` parameter
        exercises both the local-only and local-plus-shared X_actual schemas.
        """
        from yohou.preprocessing import FunctionTransformer

        y = panel_data
        X_actual = panel_data.rename({c: c.replace("__value", "__feat") if c != "time" else c for c in y.columns})
        if with_shared:
            # A shared (global, no ``__``) column makes _panel_X_actual_schema
            # merge both the local and shared schemas.
            X_actual = X_actual.with_columns(shared_feat=pl.col("g0__feat") + pl.col("g1__feat"))
        y_train, y_new = y[:50], y[50:]
        X_train, X_new = X_actual[:50], X_actual[50:]

        forecaster = DecompositionPipeline(
            [("trend", PolynomialTrendForecaster(degree=1))],
            actual_transformer=FunctionTransformer(),
        )
        forecaster.fit(y_train, X_actual=X_train, forecasting_horizon=3)

        forecaster.observe(y_new, X_actual=X_new)
        y_pred = forecaster.predict()
        assert "g0__value" in y_pred.columns
        assert "g1__value" in y_pred.columns

        forecaster.rewind(y_train, X_actual=X_train)
        y_pred_rewound = forecaster.predict()
        assert "g0__value" in y_pred_rewound.columns
        assert "g1__value" in y_pred_rewound.columns

    def test_panel_observe_with_both_transformers_active(self, panel_data):
        """Panel observe must dispatch target and feature transformers together.

        With a panel target_transformer (dict) and a panel actual_transformer
        (dict) both fitted, observe(y, X_actual) builds y_t_dict via the target
        branch and X_t_dict via the feature branch in the same call. This
        exercises the simultaneous-dict path that the target-only and
        feature-only tests each miss.
        """
        from yohou.preprocessing import FunctionTransformer

        y = panel_data
        X_actual = panel_data.rename({c: c.replace("__value", "__feat") if c != "time" else c for c in y.columns})
        y_train, y_new = y[:50], y[50:]
        X_train, X_new = X_actual[:50], X_actual[50:]

        forecaster = DecompositionPipeline(
            [("trend", PolynomialTrendForecaster(degree=1))],
            target_transformer=LogTransformer(),
            actual_transformer=FunctionTransformer(),
        )
        forecaster.fit(y_train, X_actual=X_train, forecasting_horizon=3)
        assert isinstance(forecaster.target_transformer_, dict)
        assert isinstance(forecaster.actual_transformer_, dict)

        forecaster.observe(y_new, X_actual=X_new)
        y_pred = forecaster.predict()
        assert "g0__value" in y_pred.columns
        assert "g1__value" in y_pred.columns


class TestObservationBufferBounded:
    """observe() must keep _y_observed bounded to observation_horizon."""

    def test_y_observed_bounded_after_many_observes(self):
        """Repeated observe() does not grow _y_observed without bound."""
        from yohou.stationarity import SeasonalDifferencing

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=199),
            interval="1d",
            eager=True,
        )
        data = pl.DataFrame({"time": time, "value": [float(i) for i in range(200)]})

        forecaster = DecompositionPipeline(
            [("trend", PolynomialTrendForecaster(degree=1)), ("seas", SeasonalNaive(seasonality=7))],
            target_transformer=SeasonalDifferencing(seasonality=1),
        )
        forecaster.fit(data[:30], forecasting_horizon=5)

        for start in range(30, 110, 10):
            forecaster.observe(data[start : start + 10])

        # Buffer is trimmed to observation_horizon, not the full observed history.
        assert forecaster._y_observed.height <= forecaster.observation_horizon
        # predict still round-trips through the bounded inverse-transform context.
        y_pred = forecaster.predict(forecasting_horizon=3)
        assert "value" in y_pred.columns

    def test_panel_y_observed_bounded(self):
        """In panel mode each group's _y_observed buffer is trimmed to observation_horizon."""
        from yohou.stationarity import SeasonalDifferencing

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=99),
            interval="1d",
            eager=True,
        )
        data = pl.DataFrame({
            "time": time,
            "g0__value": [float(i) for i in range(100)],
            "g1__value": [float(2 * i) for i in range(100)],
        })

        forecaster = DecompositionPipeline(
            [("trend", PolynomialTrendForecaster(degree=1))],
            target_transformer=SeasonalDifferencing(seasonality=1),
        )
        forecaster.fit(data[:30], forecasting_horizon=3)
        for start in range(30, 90, 10):
            forecaster.observe(data[start : start + 10])

        assert isinstance(forecaster._y_observed, dict)
        for group_df in forecaster._y_observed.values():
            assert group_df.height <= forecaster.observation_horizon


class TestResidualAlignmentGuard:
    """_check_residual_alignment raises when the residual/prediction join loses rows."""

    def test_row_loss_raises(self):
        """Fewer aligned rows than expected signals a stride/horizon misalignment."""
        with pytest.raises(ValueError, match="alignment for component 'trend' lost 2 row"):
            DecompositionPipeline._check_residual_alignment("trend", aligned_height=3, expected_height=5)

    def test_exact_match_passes(self):
        """Equal aligned and expected heights raise nothing."""
        DecompositionPipeline._check_residual_alignment("trend", aligned_height=5, expected_height=5)


class TestBasicFitPredict:
    """Tests for basic fit/predict workflow."""

    @pytest.fixture
    def daily_data(self):
        """Create 50-day daily time series."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        return pl.DataFrame({"time": time, "value": range(50)})

    def test_basic_workflow(self, daily_data):
        """Trend+seasonality decomposition extrapolates a pure linear signal.

        ``daily_data`` is ``value = range(50)``, a perfect slope-1 line with no
        seasonal component. A degree-1 trend captures the line exactly and the
        seasonal residual is ~0, so forecasts must continue 30, 31, 32, ...
        (the column/value contract itself is covered by the systematic suite).
        """
        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ])
        forecaster.fit(daily_data[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        # Fitted on days 0-29 (values 0-29); the additive decomposition
        # reassembles into a linear continuation with unit step.
        values = y_pred["value"].to_list()
        assert values == pytest.approx([29, 30, 31, 32, 33])

    def test_different_horizons(self, daily_data):
        """Test prediction with different horizons."""
        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ])
        forecaster.fit(daily_data[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=10)
        assert len(y_pred) == 10

    def test_predict_rejects_non_positive_horizon(self, daily_data):
        """An explicitly invalid horizon is validated, not forwarded to inner forecasters."""
        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ])
        forecaster.fit(daily_data[:30], forecasting_horizon=5)

        with pytest.raises(ValueError, match="forecasting_horizon must be >= 1"):
            forecaster.predict(forecasting_horizon=0)

    def test_prediction_types(self):
        """Test correct forecaster_type in tags."""
        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ])

        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.forecaster_type == frozenset({"point"})

    def test_prediction_types_propagates_interval(self):
        """If last forecaster supports point+interval, both propagate."""
        from yohou.utils.tags import POINT_INTERVAL

        class _DualForecaster(SeasonalNaive):
            def __sklearn_tags__(self):
                tags = super().__sklearn_tags__()
                tags.forecaster_tags.forecaster_type = POINT_INTERVAL
                return tags

        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("dual", _DualForecaster(seasonality=7)),
        ])

        tags = forecaster.__sklearn_tags__()
        assert "point" in tags.forecaster_tags.forecaster_type
        assert "interval" in tags.forecaster_tags.forecaster_type


class TestResiduals:
    """Tests for residual storage functionality."""

    @pytest.fixture
    def daily_data(self):
        """Create 50-day daily time series."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        return pl.DataFrame({"time": time, "value": range(50)})

    def test_store_residuals(self, daily_data):
        """Test residuals are stored when enabled."""
        forecaster = DecompositionPipeline(
            [
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("seasonality", SeasonalNaive(seasonality=7)),
            ],
            store_residuals=True,
        )
        forecaster.fit(daily_data[:30], forecasting_horizon=5)

        assert hasattr(forecaster, "residuals_")
        assert "trend" in forecaster.residuals_
        assert "seasonality" in forecaster.residuals_

        trend_residuals = forecaster.residuals_["trend"]
        assert "time" in trend_residuals.columns
        assert "value" in trend_residuals.columns
        assert len(trend_residuals) == 30

    def test_store_residuals_then_observe_appends(self, daily_data):
        """observe() appends to residuals_ without KeyError after fit."""
        forecaster = DecompositionPipeline(
            [
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("seasonality", SeasonalNaive(seasonality=7)),
            ],
            store_residuals=True,
        )
        forecaster.fit(daily_data[:30], forecasting_horizon=5)
        len_before = len(forecaster.residuals_["trend"])

        forecaster.observe(daily_data[30:40])

        assert "trend" in forecaster.residuals_
        assert "seasonality" in forecaster.residuals_
        assert len(forecaster.residuals_["trend"]) > len_before


class TestTargetTransformer:
    """Tests for target transformer (e.g., multiplicative decomposition)."""

    def test_log_transform(self):
        """Test with LogTransformer target transformer."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": [float(i + 1) for i in range(50)]})

        forecaster = DecompositionPipeline(
            [
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("seasonality", SeasonalNaive(seasonality=7)),
            ],
            target_transformer=LogTransformer(),
        )
        forecaster.fit(y[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        assert all(y_pred["value"] > 0)


class TestUpdateReset:
    """Tests for observe, rewind, and observe_predict methods."""

    @pytest.fixture
    def daily_data(self):
        """Create 50-day daily time series."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        return pl.DataFrame({"time": time, "value": range(50)})

    def test_observe_predict(self, daily_data):
        """Test observe_predict method."""
        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ])
        fit_forecasting_horizon = 5
        forecaster.fit(daily_data[:30], forecasting_horizon=fit_forecasting_horizon)

        n_new = 10
        predict_forecasting_horizon = 5
        y_pred = forecaster.observe_predict(
            daily_data[30 : 30 + n_new], forecasting_horizon=predict_forecasting_horizon
        )

        assert len(y_pred) == predict_forecasting_horizon * (1 + n_new // fit_forecasting_horizon)
        assert "vintage_time" in y_pred.columns

    def test_rewind(self, daily_data):
        """Test rewind method restores state."""
        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ])
        forecaster.fit(daily_data[:30], forecasting_horizon=5)

        forecaster.observe(daily_data[30:35])
        forecaster.rewind(daily_data[20:30])

        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5

    def test_rewind_threads_residuals_through_components(self, daily_data):
        """Rewind seeds each component with its residual stage, not full y.

        Under the residual-threading contract, the second component is rewound
        on (target - first component's prediction), so its observed buffer must
        differ from the first component's buffer, which is rewound on the full
        target. A regression that seeds every component with the full target
        would make these buffers identical.
        """
        forecaster = DecompositionPipeline([
            ("level", SeasonalNaive(seasonality=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ])
        forecaster.fit(daily_data[:30], forecasting_horizon=5)
        forecaster.observe(daily_data[30:40])
        forecaster.rewind(daily_data[20:40])

        named = dict(forecaster.forecasters_)
        level_observed = named["level"]._y_observed
        seasonality_observed = named["seasonality"]._y_observed
        assert level_observed is not None
        assert seasonality_observed is not None

        level_vals = level_observed.select(pl.exclude("time")).to_numpy()
        seasonality_vals = seasonality_observed.select(pl.exclude("time")).to_numpy()

        # The seasonality component sees residuals, not the raw target.
        assert not (level_vals == seasonality_vals).all()

    def test_observation_horizon(self, daily_data):
        """Test observation_horizon property."""
        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ])

        # Before fitting, observation_horizon returns 0 (no fitted transformers)
        assert forecaster.observation_horizon == 0

        forecaster.fit(daily_data[:30], forecasting_horizon=5)
        horizon = forecaster.observation_horizon
        assert horizon >= 0


class TestValidation:
    """Tests for input validation and error handling."""

    def test_duplicate_forecaster_names(self):
        """Test that duplicate forecaster names are rejected."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": range(50)})

        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("trend", SeasonalNaive(seasonality=7)),
        ])

        with pytest.raises(ValueError, match="Names provided are not unique"):
            forecaster.fit(y[:30], forecasting_horizon=5)


class TestPanelData:
    """Tests for panel data support."""

    def test_panel_data(self, panel_time_series_factory):
        """Test with panel data."""
        y_panel = panel_time_series_factory(length=50, n_series=3, seed=42)

        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ])
        forecaster.fit(y_panel[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        assert "vintage_time" in y_pred.columns
        assert "time" in y_pred.columns


class TestMultiComponent:
    """Tests for multi-component and single-component decomposition."""

    def test_three_components(self):
        """Test with three components (trend + seasonality + residual)."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=99),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": range(100)})

        forecaster = DecompositionPipeline(
            [
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("seasonality", SeasonalNaive(seasonality=7)),
                ("residual", SeasonalNaive(seasonality=1)),
            ],
            store_residuals=True,
        )
        forecaster.fit(y[:60], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        assert len(forecaster.residuals_) == 3
        assert "trend" in forecaster.residuals_
        assert "seasonality" in forecaster.residuals_
        assert "residual" in forecaster.residuals_

    def test_single_component(self):
        """Test with single component (edge case)."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": range(50)})

        forecaster = DecompositionPipeline([("trend", PolynomialTrendForecaster(degree=1))])
        forecaster.fit(y[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5


class TestExogenousFeatures:
    """Tests for exogenous feature support."""

    def test_with_exogenous_features(self):
        """Test with exogenous features."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": range(50)})
        X_actual = pl.DataFrame({"time": time, "feature": range(50, 100)})

        from sklearn.linear_model import Ridge

        from yohou.point import PointReductionForecaster
        from yohou.preprocessing import LagTransformer

        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            (
                "ml",
                PointReductionForecaster(estimator=Ridge(), actual_transformer=LagTransformer(lag=[1, 2])),
            ),
        ])
        fit_forecasting_horizon = 5
        forecaster.fit(y[:30], X_actual=X_actual[:30], forecasting_horizon=fit_forecasting_horizon)

        predict_forecasting_horizon = 5
        y_pred = forecaster.predict(forecasting_horizon=predict_forecasting_horizon)

        assert len(y_pred) == predict_forecasting_horizon


class TestDecompositionPipelineWithoutExogenous:
    """Tests for DecompositionPipeline with X_actual=None throughout the lifecycle."""

    @pytest.fixture
    def daily_y(self):
        """Create 50-day daily time series with positive values."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        return pl.DataFrame({"time": time, "value": [float(i + 1) for i in range(50)]})

    def test_fit_predict_without_exogenous(self, daily_y):
        """DecompositionPipeline should work without exogenous features."""
        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ])
        forecaster.fit(daily_y[:40], X_actual=None, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns
        assert len(y_pred) == 5

    def test_observe_predict_without_exogenous(self, daily_y):
        """DecompositionPipeline observe_predict should work without exogenous features."""
        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ])
        forecaster.fit(daily_y[:40], X_actual=None, forecasting_horizon=5)
        y_pred = forecaster.observe_predict(daily_y[40:43], X_actual=None)

        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns

    def test_requires_exogenous_tag(self):
        """DecompositionPipeline always reports requires_exogenous=False.

        The pipeline accepts but does not require X_actual. Inner forecasters
        manage their own features internally.
        """
        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ])
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.requires_exogenous is False

        from sklearn.linear_model import Ridge

        from yohou.point import PointReductionForecaster

        forecaster_mixed = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("ml", PointReductionForecaster(estimator=Ridge())),
        ])
        tags_mixed = forecaster_mixed.__sklearn_tags__()
        assert tags_mixed.forecaster_tags.requires_exogenous is False


class TestFeatureTransformerParam:
    """Tests for actual_transformer parameter on DecompositionPipeline."""

    def test_actual_transformer_in_get_params(self):
        """actual_transformer should appear in get_params()."""
        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
        ])
        params = forecaster.get_params()
        assert "actual_transformer" in params
        assert params["actual_transformer"] is None

    def test_actual_transformer_set_params(self):
        """actual_transformer should be settable via set_params()."""
        from yohou.preprocessing import LagTransformer

        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
        ])
        lag = LagTransformer(lag=[1])
        forecaster.set_params(actual_transformer=lag)
        assert forecaster.actual_transformer is lag

    def test_actual_transformer_clone(self):
        """actual_transformer should survive clone()."""
        from yohou.preprocessing import LagTransformer

        forecaster = DecompositionPipeline(
            [("trend", PolynomialTrendForecaster(degree=1))],
            actual_transformer=LagTransformer(lag=[1]),
        )
        cloned = clone(forecaster)
        assert cloned.actual_transformer is not None
        assert isinstance(cloned.actual_transformer, LagTransformer)


class TestDecompositionPipelinePanelInverseTransform:
    """Tests for predict panel inverse transform path."""

    @pytest.fixture
    def panel_y(self, panel_time_series_factory):
        """Panel data with 2 groups, positive values for LogTransformer."""
        y = panel_time_series_factory(length=80, n_series=1, n_groups=2)
        return y.with_columns([pl.col(c).cast(pl.Float64).add(1.0) for c in y.columns if c != "time"])

    def test_panel_predict_with_target_transformer(self, panel_y):
        """Panel predict with target_transformer exercises inverse transform path."""
        forecaster = DecompositionPipeline(
            [
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("season", SeasonalNaive(seasonality=7)),
            ],
            target_transformer=LogTransformer(),
        )
        forecaster.fit(panel_y[:60], forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        assert "vintage_time" in y_pred.columns
        target_cols = [c for c in y_pred.columns if c not in {"time", "vintage_time"}]
        assert len(target_cols) == 2
        assert all("__" in c for c in target_cols)

    def test_panel_predict_values_are_exp_space(self, panel_y):
        """Inverse transform should undo log, producing original-scale values."""
        forecaster = DecompositionPipeline(
            [
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("season", SeasonalNaive(seasonality=7)),
            ],
            target_transformer=LogTransformer(),
        )
        forecaster.fit(panel_y[:60], forecasting_horizon=5)

        y_pred_transformed = forecaster.predict(forecasting_horizon=5, predict_transformed=True)
        y_pred_original = forecaster.predict(forecasting_horizon=5)

        val_cols_t = [c for c in y_pred_transformed.columns if c not in {"time", "vintage_time"}]
        val_cols_o = [c for c in y_pred_original.columns if c not in {"time", "vintage_time"}]

        for ct, co in zip(val_cols_t, val_cols_o, strict=False):
            assert y_pred_transformed[ct].to_list() != y_pred_original[co].to_list()


class TestDecompositionPipelineObserveRewind:
    """Tests for observe and rewind with transformers."""

    def test_observe_with_target_transformer(self, time_series_factory):
        """Observe with target_transformer exercises observe+transform path."""
        y = time_series_factory(length=80, n_components=1)
        y = y.with_columns([pl.col(c).add(1.0) for c in y.columns if c != "time"])

        forecaster = DecompositionPipeline(
            [
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("season", SeasonalNaive(seasonality=7)),
            ],
            target_transformer=LogTransformer(),
        )
        forecaster.fit(y[:60], forecasting_horizon=5)

        forecaster.observe(y=y[60:70])
        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5

    def test_observe_with_actual_transformer(self, y_X_factory):
        """Observe with actual_transformer exercises feature transform path."""
        from yohou.preprocessing import LagTransformer

        y, X_actual = y_X_factory(length=80, n_targets=1, n_features=2)

        forecaster = DecompositionPipeline(
            [("trend", PolynomialTrendForecaster(degree=1))],
            actual_transformer=LagTransformer(lag=[1]),
        )
        forecaster.fit(y[:60], X_actual[:60], forecasting_horizon=5)

        forecaster.observe(y=y[60:70], X_actual=X_actual[60:70])
        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5

    def test_observe_with_store_residuals(self, time_series_factory):
        """Observe with store_residuals=True accumulates residuals."""
        y = time_series_factory(length=80, n_components=1)

        forecaster = DecompositionPipeline(
            [
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("season", SeasonalNaive(seasonality=7)),
            ],
            store_residuals=True,
        )
        forecaster.fit(y[:60], forecasting_horizon=5)

        initial_residuals_len = len(forecaster.residuals_["trend"])
        forecaster.observe(y=y[60:70])
        assert len(forecaster.residuals_["trend"]) > initial_residuals_len

    def test_rewind_with_target_transformer(self, time_series_factory):
        """Rewind with target_transformer exercises rewind_transform path."""
        y = time_series_factory(length=80, n_components=1)
        y = y.with_columns([pl.col(c).add(1.0) for c in y.columns if c != "time"])

        forecaster = DecompositionPipeline(
            [
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("season", SeasonalNaive(seasonality=7)),
            ],
            target_transformer=LogTransformer(),
        )
        forecaster.fit(y[:60], forecasting_horizon=5)

        forecaster.rewind(y=y[:60])
        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5

    def test_observe_advances_observed_time(self, time_series_factory):
        """The pipeline's own ``observed_time_``, not only its components'.

        Nothing inside the pipeline reads it -- predict builds its vintage from the
        components -- so it stayed at the fit end through every observe without
        changing a prediction. It is read from outside: a conformal fit compares it
        against the calibration block to decide whether the replay left the
        forecaster predictable, and a pipeline that never advanced looked to that
        check like a replay that had rewound.
        """
        y = time_series_factory(length=80, n_components=1)

        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("season", SeasonalNaive(seasonality=7)),
        ])
        forecaster.fit(y[:60], forecasting_horizon=5)
        assert forecaster.observed_time_ == y[:60]["time"][-1]

        forecaster.observe(y=y[60:70])
        assert forecaster.observed_time_ == y[60:70]["time"][-1]

        # Rewind resets the window rather than extending it, so it moves backwards.
        forecaster.rewind(y=y[:60])
        assert forecaster.observed_time_ == y[:60]["time"][-1]

    def test_observe_advances_observed_time_panel(self, panel_time_series_factory):
        """Panel keys it by group, the shape the base class stores."""
        y = panel_time_series_factory(length=80, n_series=1, n_groups=2)

        forecaster = DecompositionPipeline([("trend", PolynomialTrendForecaster(degree=1))])
        forecaster.fit(y[:60], forecasting_horizon=5)
        forecaster.observe(y=y[60:70])

        assert isinstance(forecaster.observed_time_, dict)
        assert set(forecaster.observed_time_) == set(forecaster.groups_)
        assert all(stamp == y[60:70]["time"][-1] for stamp in forecaster.observed_time_.values())


class TestDecompositionPipelineFeatureTransformer:
    """Tests for DecompositionPipeline with actual_transformer."""

    def test_rewind_with_actual_transformer(self, time_series_factory):
        """Rewind with actual_transformer exercises feature rewind_transform path."""
        from yohou.preprocessing import FunctionTransformer

        y = time_series_factory(length=80, n_components=1)
        X_actual = time_series_factory(length=80, n_components=1, seed=99)
        X_actual = X_actual.rename({c: f"feat_{c}" if c != "time" else c for c in X_actual.columns})

        forecaster = DecompositionPipeline(
            [
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("season", SeasonalNaive(seasonality=7)),
            ],
            actual_transformer=FunctionTransformer(),
        )
        forecaster.fit(y[:60], X_actual=X_actual[:60], forecasting_horizon=5)

        forecaster.rewind(y=y[:60], X_actual=X_actual[:60])
        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5

    def test_predict_with_target_transformer_inverse(self, time_series_factory):
        """Predict with target_transformer and predict_transformed=False does inverse transform."""
        y = time_series_factory(length=80, n_components=1)
        y = y.with_columns([pl.col(c).abs().add(1.0) for c in y.columns if c != "time"])

        forecaster = DecompositionPipeline(
            [
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("season", SeasonalNaive(seasonality=7)),
            ],
            target_transformer=LogTransformer(),
        )
        forecaster.fit(y[:60], forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5, predict_transformed=False)
        assert len(y_pred) == 5
        assert all(v > 0 for v in y_pred.select(pl.exclude("time", "vintage_time")).to_series().to_list())


class TestObservePredictWithExogenous:
    """Tests for observe_predict with exogenous features on DecompositionPipeline."""

    @pytest.fixture
    def exogenous_pipeline(self):
        """Create a pipeline with exogenous-aware inner forecaster."""
        from sklearn.linear_model import Ridge

        from yohou.point import PointReductionForecaster
        from yohou.preprocessing import LagTransformer

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": [float(i) for i in range(50)]})
        X_actual = pl.DataFrame({"time": time, "feature": [float(i) for i in range(50, 100)]})

        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            (
                "ml",
                PointReductionForecaster(estimator=Ridge(), actual_transformer=LagTransformer(lag=[1, 2])),
            ),
        ])
        forecaster.fit(y[:30], X_actual=X_actual[:30], forecasting_horizon=5)
        return forecaster, y, X_actual

    def test_observe_predict_with_exogenous(self, exogenous_pipeline):
        """observe_predict uses pipeline's custom observe for residual decomposition."""
        forecaster, y, X_actual = exogenous_pipeline
        y_pred = forecaster.observe_predict(
            y=y[30:40],
            X_actual=X_actual[30:40],
            forecasting_horizon=5,
        )
        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns
        assert "vintage_time" in y_pred.columns

    def test_observe_with_exogenous_decomposes_residuals(self, exogenous_pipeline):
        """observe() correctly decomposes new observations across inner forecasters."""
        forecaster, y, X_actual = exogenous_pipeline

        forecaster.observe(y=y[30:35], X_actual=X_actual[30:35])
        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        assert "value" in y_pred.columns

    def test_observe_predict_regression_without_exogenous(self):
        """observe_predict without exogenous works after observe_predict override."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": [float(i) for i in range(50)]})

        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        y_pred = forecaster.observe_predict(y=y[30:40])
        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns
        # Verify predictions exist for each vintage
        n_vintages = y_pred["vintage_time"].n_unique()
        assert n_vintages >= 2

    def test_observe_large_chunk_with_exogenous(self, exogenous_pipeline):
        """observe() handles chunks larger than fit_fh with exogenous inner forecasters."""
        forecaster, y, X_actual = exogenous_pipeline
        # Observe 15 rows (3x fit_fh=5), triggers recursive path
        forecaster.observe(y=y[30:45], X_actual=X_actual[30:45])
        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5


class TestPanelExogenous:
    """Tests for panel data with exogenous features on DecompositionPipeline."""

    def test_panel_with_exogenous_features(self):
        """DecompositionPipeline works with panel data and exogenous features."""
        from sklearn.linear_model import Ridge

        from yohou.point import PointReductionForecaster
        from yohou.preprocessing import LagTransformer

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "group_a__value": [float(i) for i in range(50)],
            "group_b__value": [float(i + 10) for i in range(50)],
        })
        X_actual = pl.DataFrame({
            "time": time,
            "group_a__feature": [float(i) for i in range(50, 100)],
            "group_b__feature": [float(i + 5) for i in range(50, 100)],
        })

        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            (
                "ml",
                PointReductionForecaster(estimator=Ridge(), actual_transformer=LagTransformer(lag=[1, 2])),
            ),
        ])
        forecaster.fit(y[:30], X_actual=X_actual[:30], forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        assert "group_a__value" in y_pred.columns
        assert "group_b__value" in y_pred.columns


class TestStepColumnStripping:
    """Pipeline-derived step columns are stripped before forwarding to components.

    Otherwise a component re-deriving the same step columns from X_future/X_forecast
    collides with the forwarded ones (DuplicateError).
    """

    @staticmethod
    def _pipeline():
        from sklearn.ensemble import HistGradientBoostingRegressor

        from yohou.point import PointReductionForecaster

        return DecompositionPipeline([
            ("a", PointReductionForecaster(estimator=HistGradientBoostingRegressor(max_iter=5))),
            ("b", PointReductionForecaster(estimator=HistGradientBoostingRegressor(max_iter=5))),
        ])

    @staticmethod
    def _time(n):
        return pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=n - 1),
            interval="1d",
            eager=True,
        )

    def test_panel_global_x_future_no_collision(self):
        """Panel data + a global (non-prefixed) X_future column does not raise DuplicateError."""
        n = 120
        time = self._time(n)
        y = pl.DataFrame({
            "time": time,
            "store_1__sales": [10 + i * 0.5 for i in range(n)],
            "store_2__sales": [20 + i * 0.3 for i in range(n)],
        })
        x_future = pl.DataFrame({"time": time, "is_holiday": [float(i % 7 == 0) for i in range(n)]})
        pipe = self._pipeline()
        pipe.fit(y, forecasting_horizon=1, X_future=x_future)
        pred = pipe.predict(forecasting_horizon=1)
        assert pred.height == 1
        assert {"store_1__sales", "store_2__sales"}.issubset(pred.columns)

    def test_panel_group_local_x_future_no_collision(self):
        """Panel data + group-local (prefixed) X_future columns are stripped without collision."""
        n = 120
        time = self._time(n)
        y = pl.DataFrame({
            "time": time,
            "store_1__sales": [10 + i * 0.5 for i in range(n)],
            "store_2__sales": [20 + i * 0.3 for i in range(n)],
        })
        x_future = pl.DataFrame({
            "time": time,
            "store_1__promo": [float(i % 5 == 0) for i in range(n)],
            "store_2__promo": [float(i % 3 == 0) for i in range(n)],
        })
        pipe = self._pipeline()
        pipe.fit(y, forecasting_horizon=1, X_future=x_future)
        assert pipe.predict(forecasting_horizon=1).height == 1

    def test_standard_x_future_no_collision(self):
        """Non-panel X_future step columns are stripped before forwarding (regression case)."""
        n = 120
        time = self._time(n)
        y = pl.DataFrame({"time": time, "sales": [10 + i * 0.5 for i in range(n)]})
        x_future = pl.DataFrame({"time": time, "is_holiday": [float(i % 7 == 0) for i in range(n)]})
        pipe = self._pipeline()
        pipe.fit(y, forecasting_horizon=1, X_future=x_future)
        assert pipe.predict(forecasting_horizon=1).height == 1

    def test_panel_global_local_step_column_clash_raises(self):
        """A global X_future column whose step name matches a per-group suffix is rejected.

        A global ``promo`` and group-local ``store_*__promo`` both expand to a
        ``promo_step_1`` column, so the global step name clashes with the per-group
        suffix and must raise rather than silently overwrite one of them.
        """
        n = 120
        time = self._time(n)
        y = pl.DataFrame({
            "time": time,
            "store_1__sales": [10 + i * 0.5 for i in range(n)],
            "store_2__sales": [20 + i * 0.3 for i in range(n)],
        })
        x_future = pl.DataFrame({
            "time": time,
            "promo": [float(i % 5 == 0) for i in range(n)],
            "store_1__promo": [float(i % 3 == 0) for i in range(n)],
            "store_2__promo": [float(i % 4 == 0) for i in range(n)],
        })
        pipe = self._pipeline()
        with pytest.raises(ValueError, match="Step column name clash"):
            pipe.fit(y, forecasting_horizon=1, X_future=x_future)


class TestForecastTransformerReachesComponents:
    """The forecast_transformer slot must affect component forecasts.

    The pipeline fits the slot and strips its transformed pipeline-level step
    columns before forwarding, so it must forward the transformed X_forecast to
    components; forwarding the raw frame renders the slot inert.
    """

    @staticmethod
    def _y(n):
        time = pl.datetime_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1) + timedelta(days=n - 1),
            interval="1d",
            eager=True,
        )
        return pl.DataFrame({"time": time, "y": [float(i % 7) + i * 0.13 for i in range(n)]})

    @staticmethod
    def _x_forecast(n, horizon):
        rows = []
        for i in range(n):
            vintage = datetime(2024, 1, 1) + timedelta(days=i)
            for h in range(1, horizon + 1):
                rows.append({
                    "vintage_time": vintage,
                    "time": vintage + timedelta(days=h),
                    "load": float(10 + i + h),
                    "wind": float(3 + (i % 5)),
                })
        return pl.DataFrame(rows)

    @staticmethod
    def _pipeline(forecast_transformer):
        from sklearn.linear_model import LinearRegression

        from yohou.point import PointReductionForecaster

        return DecompositionPipeline(
            [("t", PointReductionForecaster(estimator=LinearRegression(), nan_handling="drop"))],
            forecast_transformer=forecast_transformer,
        )

    def _net_transformer(self):
        from yohou.compose import PerVintageActualTransformer
        from yohou.preprocessing import FunctionTransformer

        def net(df):
            return df.with_columns((pl.col("load") - pl.col("wind")).alias("net")).drop("load", "wind")

        return PerVintageActualTransformer(transformer=FunctionTransformer(func=net))

    def test_slot_changes_forecasts_and_component_features(self):
        """A slot deriving a new feature changes predictions and the component's inputs."""
        n, horizon = 48, 3
        y = self._y(n)
        x_forecast = self._x_forecast(n, horizon)

        with_slot = self._pipeline(self._net_transformer())
        with_slot.fit(y, X_forecast=x_forecast, forecasting_horizon=horizon)
        without_slot = self._pipeline(None)
        without_slot.fit(y, X_forecast=x_forecast, forecasting_horizon=horizon)

        component = with_slot.forecasters_[0][1]
        assert any(name.startswith("net_step_") for name in component.feature_names_in_)
        assert not any(name.startswith("load_step_") for name in component.feature_names_in_)

        assert with_slot.predict()["y"].to_list() != without_slot.predict()["y"].to_list()

    def test_unset_slot_leaves_predictions_unchanged(self):
        """forecast_transformer=None forwards the raw frame, so components see load/wind."""
        n, horizon = 48, 3
        y = self._y(n)
        x_forecast = self._x_forecast(n, horizon)

        pipe = self._pipeline(None)
        pipe.fit(y, X_forecast=x_forecast, forecasting_horizon=horizon)

        component = pipe.forecasters_[0][1]
        assert any(name.startswith("load_step_") for name in component.feature_names_in_)
        assert pipe.predict().height == horizon


class TestNamedForecasters:
    """Fitted components are reachable by the name they were given."""

    @staticmethod
    def _y(n: int = 40) -> pl.DataFrame:
        time = pl.datetime_range(
            datetime(2024, 1, 1), datetime(2024, 1, 1) + timedelta(days=n - 1), interval="1d", eager=True
        )
        return pl.DataFrame({"time": time, "v": [float(i % 7) + 10 for i in range(n)]})

    def _pipeline(self):
        return DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ])

    def test_returns_the_fitted_component_under_its_name(self):
        fitted = self._pipeline().fit(self._y(), forecasting_horizon=3)
        assert set(fitted.named_forecasters_) == {"trend", "seasonality"}
        assert isinstance(fitted.named_forecasters_["trend"], PolynomialTrendForecaster)

    def test_requires_fit(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            _ = self._pipeline().named_forecasters_

    def test_agrees_with_the_positional_list(self):
        fitted = self._pipeline().fit(self._y(), forecasting_horizon=3)
        for name, forecaster in fitted.forecasters_:
            assert fitted.named_forecasters_[name] is forecaster
