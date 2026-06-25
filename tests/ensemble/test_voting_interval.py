"""Tests for VotingIntervalForecaster ensemble.

Tests VotingIntervalForecaster using both the check generator pattern and
specific tests for interval aggregation strategies, point prediction support,
and error handling.
"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from conftest import run_checks
from yohou.ensemble import VotingIntervalForecaster
from yohou.interval import SplitConformalForecaster
from yohou.point import SeasonalNaive
from yohou.testing import _yield_yohou_forecaster_checks


class TestVotingIntervalForecasterSystematicChecks:
    """Systematic validation via check generators."""

    @pytest.mark.parametrize(
        "forecaster",
        [
            VotingIntervalForecaster(
                forecasters=[
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
                ],
                method="envelope",
            ),
            VotingIntervalForecaster(
                forecasters=[
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
                ],
                method="mean",
                weights=[0.3, 0.7],
            ),
        ],
        ids=["envelope", "weighted_mean"],
    )
    def test_voting_interval_systematic_checks(self, forecaster, y_X_factory):
        """Run systematic checks on VotingIntervalForecaster."""
        y, X_actual = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        y_train, y_test = y[:80], y[80:]
        X_actual_train, X_actual_test = (X_actual[:80], X_actual[80:]) if X_actual is not None else (None, None)

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_actual_train, forecasting_horizon=5)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, X_actual_train, y_test, X_actual_test),
            expected_failures=set(),
        )


class TestVotingIntervalForecasterStrategies:
    """Tests for interval aggregation strategies."""

    @pytest.fixture()
    def _interval_data(self, y_X_factory):
        """Provide training data and base forecasters for interval tests."""
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
        return y_train, base_forecasters

    def test_predict_interval_available(self, _interval_data):
        """Test that predict_interval works on fitted ensemble."""
        y_train, base_forecasters = _interval_data

        ensemble = VotingIntervalForecaster(
            forecasters=base_forecasters,
            method="envelope",
        )
        ensemble.fit(y_train, forecasting_horizon=3)
        y_pred = ensemble.predict_interval(forecasting_horizon=3)

        assert len(y_pred) == 3
        lower_cols = [c for c in y_pred.columns if "_lower_" in c]
        upper_cols = [c for c in y_pred.columns if "_upper_" in c]
        assert len(lower_cols) > 0
        assert len(upper_cols) > 0

    def test_predict_interval_forwards_strategy_to_children(self, _interval_data):
        """predict_interval forwards the ``strategy`` kwarg to each child."""
        y_train, base_forecasters = _interval_data

        ensemble = VotingIntervalForecaster(
            forecasters=base_forecasters,
            method="envelope",
        )
        ensemble.fit(y_train, forecasting_horizon=3)

        seen_strategies = []
        for _, child in ensemble.forecasters_:
            original = child.predict_interval

            def _spy(*args, _orig=original, **kwargs):
                seen_strategies.append(kwargs.get("strategy"))
                return _orig(*args, **kwargs)

            child.predict_interval = _spy

        ensemble.predict_interval(forecasting_horizon=3, strategy="median")

        assert seen_strategies == ["median"] * len(ensemble.forecasters_)

    def test_envelope_strategy(self, _interval_data):
        """Test that envelope takes min(lower) and max(upper)."""
        y_train, base_forecasters = _interval_data

        ensemble = VotingIntervalForecaster(
            forecasters=base_forecasters,
            method="envelope",
        )
        ensemble.fit(y_train, forecasting_horizon=3)
        y_pred = ensemble.predict_interval(forecasting_horizon=3)

        preds = []
        for _, f in base_forecasters:
            fitted = clone(f).fit(y_train, forecasting_horizon=3)
            preds.append(fitted.predict_interval(forecasting_horizon=3))

        for col in y_pred.columns:
            if col in ("vintage_time", "time"):
                continue
            vals = np.column_stack([p[col].to_numpy() for p in preds])
            if "_lower_" in col:
                expected = np.min(vals, axis=1)
            elif "_upper_" in col:
                expected = np.max(vals, axis=1)
            else:
                expected = np.mean(vals, axis=1)
            np.testing.assert_allclose(y_pred[col].to_numpy(), expected)

    def test_mean_strategy(self, _interval_data):
        """Test that mean strategy averages all interval columns."""
        y_train, base_forecasters = _interval_data

        ensemble = VotingIntervalForecaster(
            forecasters=base_forecasters,
            method="mean",
        )
        ensemble.fit(y_train, forecasting_horizon=3)
        y_pred = ensemble.predict_interval(forecasting_horizon=3)

        preds = []
        for _, f in base_forecasters:
            fitted = clone(f).fit(y_train, forecasting_horizon=3)
            preds.append(fitted.predict_interval(forecasting_horizon=3))

        for col in y_pred.columns:
            if col in ("vintage_time", "time"):
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

        ensemble = VotingIntervalForecaster(
            forecasters=base_forecasters,
            method="median",
        )
        ensemble.fit(y_train, forecasting_horizon=3)
        y_pred = ensemble.predict_interval(forecasting_horizon=3)

        preds = []
        for _, f in base_forecasters:
            fitted = clone(f).fit(y_train, forecasting_horizon=3)
            preds.append(fitted.predict_interval(forecasting_horizon=3))

        for col in y_pred.columns:
            if col in ("vintage_time", "time"):
                continue
            vals = np.column_stack([p[col].to_numpy() for p in preds])
            expected = np.median(vals, axis=1)
            np.testing.assert_allclose(y_pred[col].to_numpy(), expected)

    def test_weighted_interval_mean_strategy(self, _interval_data):
        """Test that mean strategy with weights uses np.average for intervals."""
        y_train, base_forecasters = _interval_data
        weights = [1.0, 3.0]

        ensemble = VotingIntervalForecaster(
            forecasters=base_forecasters,
            method="mean",
            weights=weights,
        )
        ensemble.fit(y_train, forecasting_horizon=3)
        y_pred = ensemble.predict_interval(forecasting_horizon=3)

        preds = []
        for _, f in base_forecasters:
            fitted = clone(f).fit(y_train, forecasting_horizon=3)
            preds.append(fitted.predict_interval(forecasting_horizon=3))

        for col in y_pred.columns:
            if col in ("vintage_time", "time"):
                continue
            vals = np.column_stack([p[col].to_numpy() for p in preds])
            expected = np.average(vals, axis=1, weights=weights)
            np.testing.assert_allclose(y_pred[col].to_numpy(), expected)


class TestVotingIntervalForecasterPointPrediction:
    """Tests for point predictions on interval ensemble."""

    def test_predict_available_with_point_children(self, y_X_factory):
        """Test that predict() is available when children support it."""
        y, _ = y_X_factory(length=80, n_targets=1, n_features=0, seed=42)

        ensemble = VotingIntervalForecaster(
            forecasters=[
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
            ],
            point_method="mean",
        )
        ensemble.fit(y[:60], forecasting_horizon=3)
        y_pred = ensemble.predict(forecasting_horizon=3)

        assert len(y_pred) == 3
        target_cols = [c for c in y_pred.columns if c not in ("vintage_time", "time")]
        assert len(target_cols) > 0

    def test_point_method_uses_correct_aggregation(self, y_X_factory):
        """Test that predict() uses point_method not method."""
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

        ensemble = VotingIntervalForecaster(
            forecasters=base_forecasters,
            method="envelope",
            point_method="median",
        )
        ensemble.fit(y_train, forecasting_horizon=3)
        y_pred = ensemble.predict(forecasting_horizon=3)

        # Verify it uses median (point_method), not envelope (method)
        preds = []
        for _, f in base_forecasters:
            fitted = clone(f).fit(y_train, forecasting_horizon=3)
            preds.append(fitted.predict(forecasting_horizon=3))

        target_col = [c for c in y_pred.columns if c not in ("vintage_time", "time")][0]
        vals = np.column_stack([p[target_col].to_numpy() for p in preds])
        expected = np.median(vals, axis=1)
        np.testing.assert_allclose(y_pred[target_col].to_numpy(), expected)


class TestVotingIntervalForecasterDynamicType:
    """Tests for dynamic forecaster_type tag."""

    def test_type_is_point_interval_with_split_conformal(self):
        """Test that type is POINT_INTERVAL when children support both."""
        ensemble = VotingIntervalForecaster(
            forecasters=[
                (
                    "conf_1",
                    SplitConformalForecaster(
                        point_forecaster=SeasonalNaive(seasonality=1),
                        calibration_size=10,
                    ),
                ),
            ],
        )
        tags = ensemble.__sklearn_tags__()
        assert tags.forecaster_tags is not None
        assert tags.forecaster_tags.forecaster_type == frozenset({"point", "interval"})


class TestVotingIntervalForecasterPanelData:
    """Tests for panel data support."""

    def test_panel_data_interval_predictions(self, y_X_factory):
        """Test VotingIntervalForecaster works with panel data."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42, panel=True)

        forecaster = VotingIntervalForecaster(
            forecasters=[
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
            ],
            method="envelope",
        )
        forecaster.fit(y[:80], forecasting_horizon=3)
        y_pred = forecaster.predict_interval(forecasting_horizon=3)

        assert forecaster.groups_ is not None
        assert len(y_pred) == 3


class TestVotingIntervalForecasterErrorHandling:
    """Tests for error handling behavior."""

    def test_weights_length_mismatch_raises(self, y_X_factory):
        """Test that mismatched weights length raises ValueError."""
        y, _ = y_X_factory(length=80, n_targets=1, n_features=0, seed=42)

        forecaster = VotingIntervalForecaster(
            forecasters=[
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
            ],
            weights=[0.5],
        )

        with pytest.raises(ValueError, match="Number of weights"):
            forecaster.fit(y[:60], forecasting_horizon=3)

    def test_duplicate_names_raises(self):
        """Test that duplicate forecaster names raise ValueError."""
        forecaster = VotingIntervalForecaster(
            forecasters=[
                (
                    "name",
                    SplitConformalForecaster(
                        point_forecaster=SeasonalNaive(seasonality=1),
                        calibration_size=10,
                    ),
                ),
                (
                    "name",
                    SplitConformalForecaster(
                        point_forecaster=SeasonalNaive(seasonality=7),
                        calibration_size=10,
                    ),
                ),
            ],
        )

        with pytest.raises(ValueError, match="Duplicate forecaster names"):
            forecaster.fit(
                pl.DataFrame({
                    "time": pl.datetime_range(
                        start=datetime(2020, 1, 1),
                        end=datetime(2020, 3, 20),
                        interval="1d",
                        eager=True,
                    ),
                    "value": range(80),
                }),
                forecasting_horizon=3,
            )


class TestVotingIntervalForecasterObserveRewind:
    """Tests for observe, rewind, and observe_predict methods."""

    def test_observe_and_predict_interval(self, y_X_factory):
        """Test that observe delegates to all base forecasters."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        ensemble = VotingIntervalForecaster(
            forecasters=[
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
            ],
            method="envelope",
        )
        ensemble.fit(y[:70], forecasting_horizon=3)
        ensemble.observe(y=y[70:80])
        y_pred = ensemble.predict_interval(forecasting_horizon=3)
        assert len(y_pred) == 3

    def test_observe_predict_interval_shortcut(self, y_X_factory):
        """Test observe_predict_interval convenience method."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        ensemble = VotingIntervalForecaster(
            forecasters=[
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
            ],
            method="envelope",
        )
        ensemble.fit(y[:70], forecasting_horizon=3)
        y_pred = ensemble.observe_predict_interval(y=y[70:80], forecasting_horizon=3)

        # Rolling observe-predict produces multiple prediction windows
        assert len(y_pred) > 3
        lower_cols = [c for c in y_pred.columns if "_lower_" in c]
        assert len(lower_cols) > 0

    def test_observe_predict_point_shortcut(self, y_X_factory):
        """Test observe_predict convenience method for point predictions."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        ensemble = VotingIntervalForecaster(
            forecasters=[
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
            ],
            point_method="mean",
        )
        ensemble.fit(y[:70], forecasting_horizon=3)
        y_pred = ensemble.observe_predict(y=y[70:80], forecasting_horizon=3)

        # Rolling observe-predict produces multiple prediction windows
        assert len(y_pred) > 3


class TestVotingIntervalForecasterSklearn:
    """Tests for sklearn compatibility."""

    def test_clone(self, y_X_factory):
        """Test that VotingIntervalForecaster can be cloned."""
        y, _ = y_X_factory(length=80, n_targets=1, n_features=0, seed=42)

        forecaster = VotingIntervalForecaster(
            forecasters=[
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
            ],
            method="envelope",
        )
        forecaster.fit(y[:60], forecasting_horizon=3)

        cloned = clone(forecaster)
        with pytest.raises(NotFittedError):
            cloned.predict_interval(forecasting_horizon=3)

    def test_get_set_params(self):
        """Test get_params and set_params work with nested estimators."""
        forecaster = VotingIntervalForecaster(
            forecasters=[
                (
                    "conf_1",
                    SplitConformalForecaster(
                        point_forecaster=SeasonalNaive(seasonality=1),
                        calibration_size=10,
                    ),
                ),
            ],
            method="envelope",
            point_method="median",
        )

        params = forecaster.get_params(deep=True)
        assert params["method"] == "envelope"
        assert params["point_method"] == "median"
        assert params["conf_1__calibration_size"] == 10

    def test_not_fitted_error(self):
        """Test that predict_interval raises NotFittedError before fit."""
        forecaster = VotingIntervalForecaster(
            forecasters=[
                (
                    "conf_1",
                    SplitConformalForecaster(
                        point_forecaster=SeasonalNaive(seasonality=1),
                        calibration_size=10,
                    ),
                ),
            ],
        )

        with pytest.raises(NotFittedError):
            forecaster.predict_interval(forecasting_horizon=3)


class TestVotingIntervalForecasterPredictTransformed:
    """predict/observe_predict validate transformed schemas when predict_transformed=True."""

    def _ensemble(self, y_X_factory):
        y, _ = y_X_factory(length=80, n_targets=1, n_features=0, seed=42)
        ensemble = VotingIntervalForecaster(
            forecasters=[
                ("conf_1", SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=1), calibration_size=10)),
                ("conf_7", SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=7), calibration_size=10)),
            ],
            point_method="mean",
        )
        ensemble.fit(y[:60], forecasting_horizon=3)
        return ensemble

    def test_predict_transformed_validates_and_predicts(self, y_X_factory):
        """predict(predict_transformed=True) runs the transformed-schema check and predicts."""
        ensemble = self._ensemble(y_X_factory)
        y_pred = ensemble.predict(forecasting_horizon=3, predict_transformed=True)
        assert len(y_pred) == 3

    def test_observe_predict_transformed(self, y_X_factory):
        """observe_predict(predict_transformed=True) also runs the check before predicting."""
        y, _ = y_X_factory(length=80, n_targets=1, n_features=0, seed=42)
        ensemble = self._ensemble(y_X_factory)
        y_pred = ensemble.observe_predict(y[60:63], forecasting_horizon=3, predict_transformed=True)
        target_cols = [c for c in y_pred.columns if c not in ("vintage_time", "time")]
        assert len(y_pred) > 0
        assert len(target_cols) > 0
