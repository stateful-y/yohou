"""Tests for DecompositionPipeline meta-forecaster."""

from datetime import datetime, timedelta

import polars as pl
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

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
        """Test basic fit and predict workflow."""
        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ])
        forecaster.fit(daily_data[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        assert "vintage_time" in y_pred.columns
        assert "time" in y_pred.columns
        assert "value" in y_pred.columns

    def test_different_horizons(self, daily_data):
        """Test prediction with different horizons."""
        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ])
        forecaster.fit(daily_data[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=10)
        assert len(y_pred) == 10

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

    def test_observation_horizon(self, daily_data):
        """Test observation_horizon property."""
        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ])

        with pytest.raises(NotFittedError, match="fitted"):
            _ = forecaster.observation_horizon

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
                PointReductionForecaster(estimator=Ridge(), feature_transformer=LagTransformer(lag=[1, 2])),
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

    def test_ignores_exogenous_tag(self):
        """DecompositionPipeline always reports ignores_exogenous=True.

        The pipeline accepts but does not require X_actual. Inner forecasters
        manage their own features internally.
        """
        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ])
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.ignores_exogenous is True

        from sklearn.linear_model import Ridge

        from yohou.point import PointReductionForecaster

        forecaster_mixed = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("ml", PointReductionForecaster(estimator=Ridge())),
        ])
        tags_mixed = forecaster_mixed.__sklearn_tags__()
        assert tags_mixed.forecaster_tags.ignores_exogenous is True


class TestFeatureTransformerParam:
    """Tests for feature_transformer parameter on DecompositionPipeline."""

    def test_feature_transformer_in_get_params(self):
        """feature_transformer should appear in get_params()."""
        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
        ])
        params = forecaster.get_params()
        assert "feature_transformer" in params
        assert params["feature_transformer"] is None

    def test_feature_transformer_set_params(self):
        """feature_transformer should be settable via set_params()."""
        from yohou.preprocessing import LagTransformer

        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
        ])
        lag = LagTransformer(lag=[1])
        forecaster.set_params(feature_transformer=lag)
        assert forecaster.feature_transformer is lag

    def test_feature_transformer_clone(self):
        """feature_transformer should survive clone()."""
        from yohou.preprocessing import LagTransformer

        forecaster = DecompositionPipeline(
            [("trend", PolynomialTrendForecaster(degree=1))],
            feature_transformer=LagTransformer(lag=[1]),
        )
        cloned = clone(forecaster)
        assert cloned.feature_transformer is not None
        assert isinstance(cloned.feature_transformer, LagTransformer)


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

    def test_observe_with_feature_transformer(self, y_X_factory):
        """Observe with feature_transformer exercises feature transform path."""
        from yohou.preprocessing import LagTransformer

        y, X_actual = y_X_factory(length=80, n_targets=1, n_features=2)

        forecaster = DecompositionPipeline(
            [("trend", PolynomialTrendForecaster(degree=1))],
            feature_transformer=LagTransformer(lag=[1]),
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


class TestDecompositionPipelineFeatureTransformer:
    """Tests for DecompositionPipeline with feature_transformer."""

    def test_rewind_with_feature_transformer(self, time_series_factory):
        """Rewind with feature_transformer exercises feature rewind_transform path."""
        from yohou.preprocessing import FunctionTransformer

        y = time_series_factory(length=80, n_components=1)
        X_actual = time_series_factory(length=80, n_components=1, seed=99)
        X_actual = X_actual.rename({c: f"feat_{c}" if c != "time" else c for c in X_actual.columns})

        forecaster = DecompositionPipeline(
            [
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("season", SeasonalNaive(seasonality=7)),
            ],
            feature_transformer=FunctionTransformer(),
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
                PointReductionForecaster(estimator=Ridge(), feature_transformer=LagTransformer(lag=[1, 2])),
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
                PointReductionForecaster(estimator=Ridge(), feature_transformer=LagTransformer(lag=[1, 2])),
            ),
        ])
        forecaster.fit(y[:30], X_actual=X_actual[:30], forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        assert "group_a__value" in y_pred.columns
        assert "group_b__value" in y_pred.columns
