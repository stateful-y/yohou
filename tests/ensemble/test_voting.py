"""Tests for VotingForecaster ensemble.

Tests VotingForecaster using both the check generator pattern and
specific tests for aggregation behavior, interval support, error handling,
and sklearn compatibility.
"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from conftest import run_checks
from yohou.ensemble import VotingForecaster
from yohou.interval import SplitConformalForecaster
from yohou.point import SeasonalNaive
from yohou.testing import _yield_yohou_forecaster_checks


class TestVotingForecasterSystematicChecks:
    """Systematic validation via check generators."""

    @pytest.mark.parametrize(
        "forecaster",
        [
            VotingForecaster(
                forecasters=[
                    ("naive_1", SeasonalNaive(seasonality=1)),
                    ("naive_7", SeasonalNaive(seasonality=7)),
                ],
                method="mean",
            ),
            VotingForecaster(
                forecasters=[
                    ("naive_1", SeasonalNaive(seasonality=1)),
                    ("naive_7", SeasonalNaive(seasonality=7)),
                ],
                method="median",
            ),
            VotingForecaster(
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
    def test_voting_forecaster_systematic_checks(self, forecaster, y_X_factory):
        """Run systematic checks on VotingForecaster."""
        y, X = y_X_factory(length=100, n_targets=2, n_features=0, seed=42)

        y_train, y_test = y[:80], y[80:]
        X_train, X_test = (X[:80], X[80:]) if X is not None else (None, None)

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_train, forecasting_horizon=5)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, X_train, y_test, X_test),
            expected_failures=set(),
        )


class TestVotingForecasterAggregation:
    """Tests for aggregation correctness."""

    def test_mean_aggregation(self, y_X_factory):
        """Test that mean aggregation produces the average of predictions."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        naive_1 = SeasonalNaive(seasonality=1)
        naive_7 = SeasonalNaive(seasonality=7)

        forecaster = VotingForecaster(
            forecasters=[("n1", naive_1), ("n7", naive_7)],
            method="mean",
        )
        forecaster.fit(y[:40], forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)

        # Compute expected mean from individual predictions
        n1_fitted = clone(naive_1).fit(y[:40], forecasting_horizon=3)
        n7_fitted = clone(naive_7).fit(y[:40], forecasting_horizon=3)
        y_pred_1 = n1_fitted.predict(forecasting_horizon=3)
        y_pred_7 = n7_fitted.predict(forecasting_horizon=3)

        target_col = [c for c in y_pred.columns if c not in ("observed_time", "time")][0]
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

        forecaster = VotingForecaster(forecasters=forecasters, method="median")
        forecaster.fit(y[:40], forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)

        # Compute expected median from individual predictions
        individual_preds = []
        for _, naive in forecasters:
            fitted = clone(naive).fit(y[:40], forecasting_horizon=3)
            individual_preds.append(fitted.predict(forecasting_horizon=3))

        target_col = [c for c in y_pred.columns if c not in ("observed_time", "time")][0]
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

        forecaster = VotingForecaster(forecasters=forecasters, method="mean", weights=weights)
        forecaster.fit(y[:40], forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)

        # Compute expected weighted average
        individual_preds = []
        for _, naive in forecasters:
            fitted = clone(naive).fit(y[:40], forecasting_horizon=3)
            individual_preds.append(fitted.predict(forecasting_horizon=3))

        target_col = [c for c in y_pred.columns if c not in ("observed_time", "time")][0]
        values = np.column_stack([p[target_col].to_numpy() for p in individual_preds])
        expected = np.average(values, axis=1, weights=weights)
        np.testing.assert_allclose(y_pred[target_col].to_numpy(), expected)

    def test_weights_ignored_with_median(self, y_X_factory):
        """Test that weights are silently ignored when method='median'."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        forecaster_no_weights = VotingForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
            method="median",
        )
        forecaster_weights = VotingForecaster(
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

        target_col = [c for c in y_pred_no.columns if c not in ("observed_time", "time")][0]
        np.testing.assert_allclose(y_pred_no[target_col].to_numpy(), y_pred_w[target_col].to_numpy())


class TestVotingForecasterPanelData:
    """Tests for panel data support."""

    def test_panel_data_predictions(self, y_X_factory):
        """Test VotingForecaster works with panel data."""
        y, _ = y_X_factory(length=100, n_targets=2, n_features=0, seed=42, panel=True)

        forecaster = VotingForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
            method="mean",
        )
        forecaster.fit(y[:80], forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)

        assert forecaster.panel_group_names_ is not None
        assert len(y_pred) == 3
        # Panel columns should be present in predictions
        target_cols = [c for c in y_pred.columns if c not in ("observed_time", "time")]
        assert len(target_cols) > 0


class TestVotingForecasterErrorHandling:
    """Tests for error handling behavior."""

    def test_single_forecaster_failure_skipped(self, y_X_factory):
        """Test that a single failing forecaster is skipped with warning."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        # SeasonalNaive with absurdly high seasonality should fail
        forecaster = VotingForecaster(
            forecasters=[
                ("good", SeasonalNaive(seasonality=1)),
                ("bad", SeasonalNaive(seasonality=10000)),
            ],
            method="mean",
        )

        with pytest.warns(UserWarning, match="failed during fit"):
            forecaster.fit(y[:40], forecasting_horizon=3)

        # Should still have one surviving forecaster
        assert len(forecaster.forecasters_) == 1
        y_pred = forecaster.predict(forecasting_horizon=3)
        assert len(y_pred) == 3

    def test_all_forecasters_fail_raises(self, y_X_factory):
        """Test that ensemble raises when all forecasters fail."""
        y, _ = y_X_factory(length=10, n_targets=1, n_features=0, seed=42)

        forecaster = VotingForecaster(
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

        forecaster = VotingForecaster(
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
        forecaster = VotingForecaster(
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

    def test_schema_mismatch_raises(self):
        """Test that mismatched target schemas raise ValueError."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 2, 19),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value_a": range(50), "value_b": range(50)})

        # This would require column forecasters that predict different columns;
        # since VotingForecaster feeds same y to all, schemas should match
        forecaster = VotingForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
        )
        forecaster.fit(y[:40], forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)
        assert "value_a" in y_pred.columns
        assert "value_b" in y_pred.columns


class TestVotingForecasterInterval:
    """Tests for predict_interval aggregation strategies."""

    @pytest.fixture()
    def _interval_ensemble(self, y_X_factory):
        """Fit interval forecasters and return (y_train, ensemble, base list)."""
        y, _ = y_X_factory(length=80, n_targets=1, n_features=0, seed=42)
        y_train = y[:60]

        base_forecasters = [
            (
                "conf_1",
                SplitConformalForecaster(
                    point_forecaster=SeasonalNaive(seasonality=1),
                    calibration_size=10,
                ),
            ),
            (
                "conf_7",
                SplitConformalForecaster(
                    point_forecaster=SeasonalNaive(seasonality=7),
                    calibration_size=10,
                ),
            ),
        ]

        ensemble = VotingForecaster(
            forecasters=base_forecasters,
            method="mean",
            interval_strategy="envelope",
        )
        ensemble.fit(y_train, forecasting_horizon=3)
        return y_train, ensemble, base_forecasters

    def test_predict_interval_available(self, _interval_ensemble):
        """Test that predict_interval is exposed when base forecasters support it."""
        _, ensemble, _ = _interval_ensemble
        assert hasattr(ensemble, "predict_interval")
        y_pred = ensemble.predict_interval(forecasting_horizon=3)
        assert len(y_pred) == 3

        lower_cols = [c for c in y_pred.columns if "_lower_" in c]
        upper_cols = [c for c in y_pred.columns if "_upper_" in c]
        assert len(lower_cols) > 0
        assert len(upper_cols) > 0

    def test_envelope_strategy(self, y_X_factory):
        """Test that envelope takes min(lower) and max(upper)."""
        y, _ = y_X_factory(length=80, n_targets=1, n_features=0, seed=42)
        y_train = y[:60]

        base_forecasters = [
            (
                "conf_1",
                SplitConformalForecaster(
                    point_forecaster=SeasonalNaive(seasonality=1),
                    calibration_size=10,
                ),
            ),
            (
                "conf_7",
                SplitConformalForecaster(
                    point_forecaster=SeasonalNaive(seasonality=7),
                    calibration_size=10,
                ),
            ),
        ]

        ensemble = VotingForecaster(
            forecasters=base_forecasters,
            method="mean",
            interval_strategy="envelope",
        )
        ensemble.fit(y_train, forecasting_horizon=3)
        y_pred = ensemble.predict_interval(forecasting_horizon=3)

        # Compute individual interval predictions
        preds = []
        for _, f in base_forecasters:
            fitted = clone(f).fit(y_train, forecasting_horizon=3)
            preds.append(fitted.predict_interval(forecasting_horizon=3))

        for col in y_pred.columns:
            if col in ("observed_time", "time"):
                continue
            vals = np.column_stack([p[col].to_numpy() for p in preds])
            if "_lower_" in col:
                expected = np.min(vals, axis=1)
            elif "_upper_" in col:
                expected = np.max(vals, axis=1)
            else:
                expected = np.mean(vals, axis=1)
            np.testing.assert_allclose(y_pred[col].to_numpy(), expected)

    def test_mean_strategy(self, y_X_factory):
        """Test that mean strategy averages all interval columns."""
        y, _ = y_X_factory(length=80, n_targets=1, n_features=0, seed=42)
        y_train = y[:60]

        base_forecasters = [
            (
                "conf_1",
                SplitConformalForecaster(
                    point_forecaster=SeasonalNaive(seasonality=1),
                    calibration_size=10,
                ),
            ),
            (
                "conf_7",
                SplitConformalForecaster(
                    point_forecaster=SeasonalNaive(seasonality=7),
                    calibration_size=10,
                ),
            ),
        ]

        ensemble = VotingForecaster(
            forecasters=base_forecasters,
            method="mean",
            interval_strategy="mean",
        )
        ensemble.fit(y_train, forecasting_horizon=3)
        y_pred = ensemble.predict_interval(forecasting_horizon=3)

        preds = []
        for _, f in base_forecasters:
            fitted = clone(f).fit(y_train, forecasting_horizon=3)
            preds.append(fitted.predict_interval(forecasting_horizon=3))

        for col in y_pred.columns:
            if col in ("observed_time", "time"):
                continue
            vals = np.column_stack([p[col].to_numpy() for p in preds])
            expected = np.mean(vals, axis=1)
            np.testing.assert_allclose(y_pred[col].to_numpy(), expected)

    def test_median_strategy(self, y_X_factory):
        """Test that median strategy takes median of all interval columns."""
        y, _ = y_X_factory(length=80, n_targets=1, n_features=0, seed=42)
        y_train = y[:60]

        base_forecasters = [
            (
                "conf_1",
                SplitConformalForecaster(
                    point_forecaster=SeasonalNaive(seasonality=1),
                    calibration_size=10,
                ),
            ),
            (
                "conf_2",
                SplitConformalForecaster(
                    point_forecaster=SeasonalNaive(seasonality=2),
                    calibration_size=10,
                ),
            ),
            (
                "conf_7",
                SplitConformalForecaster(
                    point_forecaster=SeasonalNaive(seasonality=7),
                    calibration_size=10,
                ),
            ),
        ]

        ensemble = VotingForecaster(
            forecasters=base_forecasters,
            method="median",
            interval_strategy="median",
        )
        ensemble.fit(y_train, forecasting_horizon=3)
        y_pred = ensemble.predict_interval(forecasting_horizon=3)

        preds = []
        for _, f in base_forecasters:
            fitted = clone(f).fit(y_train, forecasting_horizon=3)
            preds.append(fitted.predict_interval(forecasting_horizon=3))

        for col in y_pred.columns:
            if col in ("observed_time", "time"):
                continue
            vals = np.column_stack([p[col].to_numpy() for p in preds])
            expected = np.median(vals, axis=1)
            np.testing.assert_allclose(y_pred[col].to_numpy(), expected)

    def test_point_only_forecasters_no_predict_interval(self):
        """Test that predict_interval is absent with point-only forecasters."""
        forecaster = VotingForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
        )
        assert not hasattr(forecaster, "predict_interval")


class TestVotingForecasterObserveRewind:
    """Tests for observe, rewind, observe_predict, and observe_predict_interval."""

    def test_observe_and_predict(self, y_X_factory):
        """Test that observe delegates to all base forecasters."""
        y, _ = y_X_factory(length=60, n_targets=1, n_features=0, seed=42)

        forecaster = VotingForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
            method="mean",
        )
        forecaster.fit(y[:40], forecasting_horizon=3)

        # observe then predict should work
        forecaster.observe(y=y[40:50])
        y_pred = forecaster.predict(forecasting_horizon=3)
        assert len(y_pred) == 3

    def test_rewind_and_predict(self, y_X_factory):
        """Test that rewind delegates to all base forecasters."""
        y, _ = y_X_factory(length=60, n_targets=1, n_features=0, seed=42)

        forecaster = VotingForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
            method="mean",
        )
        forecaster.fit(y[:50], forecasting_horizon=3)
        forecaster.observe(y=y[50:55])

        # rewind back to fit data
        forecaster.rewind(y=y[:50])
        y_pred = forecaster.predict(forecasting_horizon=3)
        assert len(y_pred) == 3

    def test_observe_predict_shortcut(self, y_X_factory):
        """Test observe_predict convenience method."""
        y, _ = y_X_factory(length=60, n_targets=1, n_features=0, seed=42)

        forecaster = VotingForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
            method="mean",
        )
        forecaster.fit(y[:40], forecasting_horizon=3)
        y_pred = forecaster.observe_predict(y=y[40:50], forecasting_horizon=3)
        assert len(y_pred) == 3

    def test_observe_predict_interval_shortcut(self, y_X_factory):
        """Test observe_predict_interval convenience method."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        base_forecasters = [
            (
                "conf_1",
                SplitConformalForecaster(
                    point_forecaster=SeasonalNaive(seasonality=1),
                    calibration_size=10,
                ),
            ),
            (
                "conf_7",
                SplitConformalForecaster(
                    point_forecaster=SeasonalNaive(seasonality=7),
                    calibration_size=10,
                ),
            ),
        ]

        ensemble = VotingForecaster(
            forecasters=base_forecasters,
            method="mean",
            interval_strategy="envelope",
        )
        ensemble.fit(y[:70], forecasting_horizon=3)
        y_pred = ensemble.observe_predict_interval(y=y[70:80], forecasting_horizon=3)
        assert len(y_pred) == 3

        lower_cols = [c for c in y_pred.columns if "_lower_" in c]
        assert len(lower_cols) > 0


class TestVotingForecasterIntervalWeighted:
    """Tests for interval aggregation with weights (mean strategy + weights)."""

    def test_weighted_interval_mean_strategy(self, y_X_factory):
        """Test that mean strategy with weights uses np.average for intervals."""
        y, _ = y_X_factory(length=80, n_targets=1, n_features=0, seed=42)
        y_train = y[:60]

        base_forecasters = [
            (
                "conf_1",
                SplitConformalForecaster(
                    point_forecaster=SeasonalNaive(seasonality=1),
                    calibration_size=10,
                ),
            ),
            (
                "conf_7",
                SplitConformalForecaster(
                    point_forecaster=SeasonalNaive(seasonality=7),
                    calibration_size=10,
                ),
            ),
        ]

        weights = [1.0, 3.0]
        ensemble = VotingForecaster(
            forecasters=base_forecasters,
            method="mean",
            weights=weights,
            interval_strategy="mean",
        )
        ensemble.fit(y_train, forecasting_horizon=3)
        y_pred = ensemble.predict_interval(forecasting_horizon=3)

        # Compute expected weighted average
        preds = []
        for _, f in base_forecasters:
            fitted = clone(f).fit(y_train, forecasting_horizon=3)
            preds.append(fitted.predict_interval(forecasting_horizon=3))

        for col in y_pred.columns:
            if col in ("observed_time", "time"):
                continue
            vals = np.column_stack([p[col].to_numpy() for p in preds])
            expected = np.average(vals, axis=1, weights=weights)
            np.testing.assert_allclose(y_pred[col].to_numpy(), expected)


class TestVotingForecasterValidation:
    """Tests for input validation edge cases."""

    def test_malformed_tuple_raises(self, y_X_factory):
        """Test that non-tuple entries raise ValueError."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        forecaster = VotingForecaster(
            forecasters=[SeasonalNaive(seasonality=1)],
        )
        with pytest.raises(ValueError, match="must be a.*tuple"):
            forecaster.fit(y[:40], forecasting_horizon=3)

    def test_non_string_name_raises(self, y_X_factory):
        """Test that non-string name raises ValueError."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        forecaster = VotingForecaster(
            forecasters=[(42, SeasonalNaive(seasonality=1))],
        )
        with pytest.raises(ValueError, match="must be a string"):
            forecaster.fit(y[:40], forecasting_horizon=3)

    def test_non_forecaster_raises(self, y_X_factory):
        """Test that non-BaseForecaster raises ValueError."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        forecaster = VotingForecaster(
            forecasters=[("bad", "not_a_forecaster")],
        )
        with pytest.raises(ValueError, match="must be a BaseForecaster"):
            forecaster.fit(y[:40], forecasting_horizon=3)

    def test_named_forecasters_property(self, y_X_factory):
        """Test that named_forecasters_ returns a Bunch with correct keys."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        forecaster = VotingForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
                ("n7", SeasonalNaive(seasonality=7)),
            ],
        )
        forecaster.fit(y[:40], forecasting_horizon=3)

        named = forecaster.named_forecasters_
        assert hasattr(named, "n1")
        assert hasattr(named, "n7")

    def test_forecasters_setter(self, y_X_factory):
        """Test that _forecasters setter updates forecasters attribute."""
        forecaster = VotingForecaster(
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


class TestVotingForecasterSklearn:
    """Tests for sklearn compatibility."""

    def test_clone(self, y_X_factory):
        """Test that VotingForecaster can be cloned."""
        y, _ = y_X_factory(length=50, n_targets=1, n_features=0, seed=42)

        forecaster = VotingForecaster(
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
        forecaster = VotingForecaster(
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
        forecaster = VotingForecaster(
            forecasters=[
                ("n1", SeasonalNaive(seasonality=1)),
            ],
        )

        with pytest.raises(NotFittedError):
            forecaster.predict(forecasting_horizon=3)
