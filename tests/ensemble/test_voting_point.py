"""Tests for VotingPointForecaster ensemble.

Tests VotingPointForecaster using both the check generator pattern and
specific tests for aggregation behavior, error handling, and sklearn
compatibility.
"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from conftest import run_checks
from yohou.ensemble import VotingPointForecaster
from yohou.point import SeasonalNaive
from yohou.testing import _yield_yohou_forecaster_checks


class TestVotingPointForecasterSystematicChecks:
    """Systematic validation via check generators."""

    @pytest.mark.parametrize(
        "forecaster",
        [
            VotingPointForecaster(
                forecasters=[
                    ("naive_1", SeasonalNaive(seasonality=1)),
                    ("naive_7", SeasonalNaive(seasonality=7)),
                ],
                method="mean",
            ),
            VotingPointForecaster(
                forecasters=[
                    ("naive_1", SeasonalNaive(seasonality=1)),
                    ("naive_7", SeasonalNaive(seasonality=7)),
                ],
                method="median",
            ),
            VotingPointForecaster(
                forecasters=[
                    ("naive_1", SeasonalNaive(seasonality=1)),
                    ("naive_7", SeasonalNaive(seasonality=7)),
                ],
                method="mean",
                weights=[0.3, 0.7],
            ),
        ],
        ids=["mean", "median", "weighted_mean"],
    )
    def test_voting_point_systematic_checks(self, forecaster, y_X_factory):
        """Run systematic checks on VotingPointForecaster."""
        y, X_actual = y_X_factory(length=100, n_targets=2, n_features=0, seed=42)

        y_train, y_test = y[:80], y[80:]
        X_actual_train, X_actual_test = (X_actual[:80], X_actual[80:]) if X_actual is not None else (None, None)

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_actual_train, forecasting_horizon=5)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, X_actual_train, y_test, X_actual_test),
            expected_failures=set(),
        )


class TestVotingPointForecasterAggregation:
    """Tests for aggregation correctness."""

    def test_mean_aggregation(self, y_X_factory):
        """Test that mean aggregation produces the average of predictions."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        naive_1 = SeasonalNaive(seasonality=1)
        naive_7 = SeasonalNaive(seasonality=7)

        forecaster = VotingPointForecaster(
            forecasters=[("n1", naive_1), ("n7", naive_7)],
            method="mean",
        )
        forecaster.fit(y[:40], forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)

        n1_fitted = clone(naive_1).fit(y[:40], forecasting_horizon=3)
        n7_fitted = clone(naive_7).fit(y[:40], forecasting_horizon=3)
        y_pred_1 = n1_fitted.predict(forecasting_horizon=3)
        y_pred_7 = n7_fitted.predict(forecasting_horizon=3)

        target_col = [c for c in y_pred.columns if c not in ("vintage_time", "time")][0]
        expected = (y_pred_1[target_col].to_numpy() + y_pred_7[target_col].to_numpy()) / 2
        np.testing.assert_allclose(y_pred[target_col].to_numpy(), expected)

    def test_median_aggregation(self, y_X_factory):
        """Test that median aggregation produces the median of predictions."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        forecasters = [
            ("n1", SeasonalNaive(seasonality=1)),
            ("n2", SeasonalNaive(seasonality=2)),
            ("n7", SeasonalNaive(seasonality=7)),
        ]

        forecaster = VotingPointForecaster(forecasters=forecasters, method="median")
        forecaster.fit(y[:40], forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)

        individual_preds = []
        for _, naive in forecasters:
            fitted = clone(naive).fit(y[:40], forecasting_horizon=3)
            individual_preds.append(fitted.predict(forecasting_horizon=3))

        target_col = [c for c in y_pred.columns if c not in ("vintage_time", "time")][0]
        values = np.column_stack([p[target_col].to_numpy() for p in individual_preds])
        expected = np.median(values, axis=1)
        np.testing.assert_allclose(y_pred[target_col].to_numpy(), expected)

    def test_weighted_mean_aggregation(self, y_X_factory):
        """Test that weighted mean uses numpy.average with raw weights."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        forecasters = [
            ("n1", SeasonalNaive(seasonality=1)),
            ("n7", SeasonalNaive(seasonality=7)),
        ]
        weights = [1.0, 3.0]

        forecaster = VotingPointForecaster(forecasters=forecasters, method="mean", weights=weights)
        forecaster.fit(y[:40], forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)

        individual_preds = []
        for _, naive in forecasters:
            fitted = clone(naive).fit(y[:40], forecasting_horizon=3)
            individual_preds.append(fitted.predict(forecasting_horizon=3))

        target_col = [c for c in y_pred.columns if c not in ("vintage_time", "time")][0]
        values = np.column_stack([p[target_col].to_numpy() for p in individual_preds])
        expected = np.average(values, axis=1, weights=weights)
        np.testing.assert_allclose(y_pred[target_col].to_numpy(), expected)

    def test_weights_ignored_with_median(self, y_X_factory):
        """Test that weights are silently ignored when method='median'."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        forecaster_no_weights = VotingPointForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
            method="median",
        )
        forecaster_weights = VotingPointForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
            method="median",
            weights=[0.1, 0.9],
        )

        forecaster_no_weights.fit(y[:40], forecasting_horizon=3)
        forecaster_weights.fit(y[:40], forecasting_horizon=3)

        y_pred_no = forecaster_no_weights.predict(forecasting_horizon=3)
        y_pred_w = forecaster_weights.predict(forecasting_horizon=3)

        target_col = [c for c in y_pred_no.columns if c not in ("vintage_time", "time")][0]
        np.testing.assert_allclose(y_pred_no[target_col].to_numpy(), y_pred_w[target_col].to_numpy())


class TestVotingPointForecasterPanelData:
    """Tests for panel data support."""

    def test_panel_data_predictions(self, y_X_factory):
        """Test VotingPointForecaster works with panel data."""
        y, _ = y_X_factory(length=100, n_targets=2, n_features=0, seed=42, panel=True)

        forecaster = VotingPointForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
            method="mean",
        )
        forecaster.fit(y[:80], forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)

        assert forecaster.groups_ is not None
        assert len(y_pred) == 3
        target_cols = [c for c in y_pred.columns if c not in ("vintage_time", "time")]
        assert len(target_cols) > 0


class TestVotingPointForecasterErrorHandling:
    """Tests for error handling behavior."""

    def test_single_forecaster_failure_skipped(self, y_X_factory):
        """Test that a single failing forecaster is skipped with warning."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        forecaster = VotingPointForecaster(
            forecasters=[
                ("good", SeasonalNaive(seasonality=1)),
                ("bad", SeasonalNaive(seasonality=10000)),
            ],
            method="mean",
        )

        with pytest.warns(UserWarning, match="failed during fit"):
            forecaster.fit(y[:40], forecasting_horizon=3)

        assert len(forecaster.forecasters_) == 1
        y_pred = forecaster.predict(forecasting_horizon=3)
        assert len(y_pred) == 3

    def test_all_forecasters_fail_raises(self, y_X_factory):
        """Test that ensemble raises when all forecasters fail."""
        y, _ = y_X_factory(length=10, n_targets=1, n_features=0, seed=42)

        forecaster = VotingPointForecaster(
            forecasters=[
                ("bad1", SeasonalNaive(seasonality=10000)),
                ("bad2", SeasonalNaive(seasonality=10000)),
            ],
            method="mean",
        )

        with pytest.raises(RuntimeError, match="All base forecasters failed"):
            forecaster.fit(y[:5], forecasting_horizon=3)

    def test_weights_length_mismatch_raises(self, y_X_factory):
        """Test that mismatched weights length raises ValueError."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        forecaster = VotingPointForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
            weights=[0.5],
        )

        with pytest.raises(ValueError, match="Number of weights"):
            forecaster.fit(y[:40], forecasting_horizon=3)

    def test_duplicate_names_raises(self):
        """Test that duplicate forecaster names raise ValueError."""
        forecaster = VotingPointForecaster(
            forecasters=[
                ("name", SeasonalNaive(seasonality=1)),
                ("name", SeasonalNaive(seasonality=7)),
            ],
        )

        with pytest.raises(ValueError, match="Duplicate forecaster names"):
            forecaster.fit(
                pl.DataFrame({
                    "time": pl.datetime_range(
                        start=datetime(2020, 1, 1),
                        end=datetime(2020, 2, 19),
                        interval="1d",
                        eager=True,
                    ),
                    "value": range(50),
                }),
                forecasting_horizon=3,
            )

    def test_double_underscore_name_raises(self):
        """A forecaster name containing '__' is rejected.

        A name like 'foo__bar' must fail validation; otherwise it corrupts
        nested-parameter routing.
        """
        forecaster = VotingPointForecaster(
            forecasters=[
                ("foo__bar", SeasonalNaive(seasonality=1)),
                ("baz", SeasonalNaive(seasonality=7)),
            ],
        )
        with pytest.raises(ValueError, match="must not contain '__'"):
            forecaster.fit(
                pl.DataFrame({
                    "time": pl.datetime_range(
                        start=datetime(2020, 1, 1),
                        end=datetime(2020, 2, 19),
                        interval="1d",
                        eager=True,
                    ),
                    "value": range(50),
                }),
                forecasting_horizon=3,
            )

    def test_schema_mismatch_validates(self):
        """Test that matching target schemas work correctly."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 2, 19),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value_a": range(50), "value_b": range(50)})

        forecaster = VotingPointForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
        )
        forecaster.fit(y[:40], forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)
        assert "value_a" in y_pred.columns
        assert "value_b" in y_pred.columns

    def test_schema_mismatch_raises(self):
        """Test that mismatched target schemas raise ValueError."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 2, 19),
            interval="1d",
            eager=True,
        )
        y_a = pl.DataFrame({"time": time, "value_a": range(50)})
        y_b = pl.DataFrame({"time": time, "value_b": range(50)})

        n1 = SeasonalNaive(seasonality=1)
        n7 = SeasonalNaive(seasonality=7)
        n1.fit(y_a[:40], forecasting_horizon=3)
        n7.fit(y_b[:40], forecasting_horizon=3)

        forecaster = VotingPointForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
        )
        # Manually inject mismatched fitted forecasters
        forecaster.forecasters_ = [("n1", n1), ("n7", n7)]

        with pytest.raises(ValueError, match="All base forecasters must predict the same target columns"):
            forecaster._validate_schemas_match()

    @pytest.mark.skipif(
        not hasattr(__import__("sklearn.utils", fromlist=["_repr_html"]), "_repr_html"),
        reason="sklearn.utils._repr_html not available (sklearn < 1.8)",
    )
    def test_sk_visual_block(self):
        """Test _sk_visual_block_ returns a parallel visual block."""
        forecaster = VotingPointForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
        )
        block = forecaster._sk_visual_block_()
        assert block.kind == "parallel"
        assert list(block.names) == ["n1", "n7"]

    def test_aggregate_interval_envelope_fallback(self):
        """Envelope strategy falls back to mean for non-lower/upper columns."""
        from yohou.ensemble._base import _BaseEnsembleForecaster

        pred1 = pl.DataFrame({"time": [1, 2], "coverage_90": [10.0, 20.0]})
        pred2 = pl.DataFrame({"time": [1, 2], "coverage_90": [30.0, 40.0]})

        result = _BaseEnsembleForecaster._aggregate_interval_values(
            predictions=[pred1, pred2],
            interval_cols=["coverage_90"],
            strategy="envelope",
            weights=None,
        )
        assert result["coverage_90"].to_list() == [20.0, 30.0]


class TestVotingPointForecasterObserveRewind:
    """Tests for observe, rewind, and observe_predict."""

    def test_observe_and_predict(self, y_X_factory):
        """Test that observe delegates to all base forecasters."""
        y, _ = y_X_factory(length=60, n_targets=1, n_features=0, seed=42)

        forecaster = VotingPointForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
            method="mean",
        )
        forecaster.fit(y[:40], forecasting_horizon=3)

        forecaster.observe(y=y[40:50])
        y_pred = forecaster.predict(forecasting_horizon=3)
        assert len(y_pred) == 3

    def test_rewind_and_predict(self, y_X_factory):
        """Test that rewind delegates to all base forecasters."""
        y, _ = y_X_factory(length=60, n_targets=1, n_features=0, seed=42)

        forecaster = VotingPointForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
            method="mean",
        )
        forecaster.fit(y[:50], forecasting_horizon=3)
        forecaster.observe(y=y[50:55])

        forecaster.rewind(y=y[:50])
        y_pred = forecaster.predict(forecasting_horizon=3)
        assert len(y_pred) == 3

    def test_observe_predict_shortcut(self, y_X_factory):
        """Test observe_predict convenience method."""
        y, _ = y_X_factory(length=60, n_targets=1, n_features=0, seed=42)

        forecaster = VotingPointForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
            method="mean",
        )
        forecaster.fit(y[:40], forecasting_horizon=3)
        y_pred = forecaster.observe_predict(y=y[40:50], forecasting_horizon=3)

        # Rolling observe-predict produces multiple prediction windows
        assert len(y_pred) > 3


class TestVotingPointForecasterValidation:
    """Tests for input validation edge cases."""

    def test_malformed_tuple_raises(self, y_X_factory):
        """Test that non-tuple entries raise ValueError."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        forecaster = VotingPointForecaster(
            forecasters=[SeasonalNaive(seasonality=1)],
        )
        with pytest.raises(ValueError, match="must be a.*tuple"):
            forecaster.fit(y[:40], forecasting_horizon=3)

    def test_non_string_name_raises(self, y_X_factory):
        """Test that non-string name raises ValueError."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        forecaster = VotingPointForecaster(
            forecasters=[(42, SeasonalNaive(seasonality=1))],
        )
        with pytest.raises(ValueError, match="must be a string"):
            forecaster.fit(y[:40], forecasting_horizon=3)

    def test_non_forecaster_raises(self, y_X_factory):
        """Test that non-BaseForecaster raises ValueError."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        forecaster = VotingPointForecaster(
            forecasters=[("bad", "not_a_forecaster")],
        )
        with pytest.raises(ValueError, match="must be a BaseForecaster"):
            forecaster.fit(y[:40], forecasting_horizon=3)

    def test_named_forecasters_property(self, y_X_factory):
        """Test that named_forecasters_ returns a Bunch with correct keys."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        forecaster = VotingPointForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
        )
        forecaster.fit(y[:40], forecasting_horizon=3)

        named = forecaster.named_forecasters_
        assert hasattr(named, "n1")
        assert hasattr(named, "n7")

    def test_forecasters_setter(self):
        """Test that _forecasters setter updates forecasters attribute."""
        forecaster = VotingPointForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
            ],
        )
        new_forecasters = [
            ("a", SeasonalNaive(seasonality=2)),
            ("b", SeasonalNaive(seasonality=3)),
        ]
        forecaster._forecasters = new_forecasters
        assert forecaster.forecasters == new_forecasters


class TestVotingPointForecasterSklearn:
    """Tests for sklearn compatibility."""

    def test_clone(self, y_X_factory):
        """Test that VotingPointForecaster can be cloned."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        forecaster = VotingPointForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
            method="mean",
        )
        forecaster.fit(y[:40], forecasting_horizon=3)

        cloned = clone(forecaster)
        with pytest.raises(NotFittedError):
            cloned.predict(forecasting_horizon=3)

    def test_get_set_params(self):
        """Test get_params and set_params work with nested estimators."""
        forecaster = VotingPointForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
            method="mean",
        )

        params = forecaster.get_params(deep=True)
        assert params["method"] == "mean"
        assert params["n1__seasonality"] == 1
        assert params["n7__seasonality"] == 7

        forecaster.set_params(n1__seasonality=3)
        assert forecaster.get_params()["n1__seasonality"] == 3

    def test_not_fitted_error(self):
        """Test that predict raises NotFittedError before fit."""
        forecaster = VotingPointForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
            ],
        )

        with pytest.raises(NotFittedError):
            forecaster.predict(forecasting_horizon=3)


class TestVotingSchemaValidation:
    """Tests for _validate_schemas_match across base forecasters."""

    def _ensemble_with(self, schemas):
        """Build an (unfitted) ensemble and inject fake fitted children."""
        from types import SimpleNamespace

        ensemble = VotingPointForecaster(forecasters=[("a", SeasonalNaive()), ("b", SeasonalNaive())])
        ensemble.forecasters_ = [(name, SimpleNamespace(local_y_schema_=schema)) for name, schema in schemas]
        return ensemble

    def test_mismatched_target_columns_raises(self):
        """Different predicted column names raise a clear error."""
        ensemble = self._ensemble_with([
            ("a", {"sales": pl.Float64}),
            ("b", {"revenue": pl.Float64}),
        ])
        with pytest.raises(ValueError, match="must predict the same target columns"):
            ensemble._validate_schemas_match()

    def test_mismatched_target_dtypes_raises(self):
        """Same columns but differing dtypes raise a dtype-mismatch error."""
        ensemble = self._ensemble_with([
            ("a", {"sales": pl.Float64}),
            ("b", {"sales": pl.Float32}),
        ])
        with pytest.raises(ValueError, match="differ from"):
            ensemble._validate_schemas_match()

    def test_matching_schemas_pass(self):
        """Identical schemas validate without error."""
        ensemble = self._ensemble_with([
            ("a", {"sales": pl.Float64}),
            ("b", {"sales": pl.Float64}),
        ])
        ensemble._validate_schemas_match()
