"""Tests for cross-learning functionality in point forecasters."""

from copy import deepcopy

import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from conftest import run_checks
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer
from yohou.testing import _yield_yohou_forecaster_checks


class TestPointReductionPanelChecks:
    @pytest.mark.parametrize(
        "forecaster,tags,expected_failures",
        [
            (
                PointReductionForecaster(estimator=LinearRegression(), feature_transformer=LagTransformer(lag=[1, 2])),
                {"forecaster_type": frozenset({"point"}), "uses_reduction": True, "supports_panel_data": True},
                [],
            ),
        ],
    )
    def test_point_reduction_panel_checks(self, forecaster, tags, expected_failures, panel_time_series_factory):
        """Run systematic cross-learning checks on PointReductionForecaster with panel data."""
        y = panel_time_series_factory(length=100, n_series=3)
        y_train, y_test = y[:80], y[80:]

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_actual=None, forecasting_horizon=3)

        run_checks(
            deepcopy(forecaster_fitted),
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, None, y_test, None),
            expected_failures=set(expected_failures),
        )


class TestPointReductionPanelBehavior:
    def test_panel_predict_all_groups_default(self, panel_time_series_factory):
        """Test that predict with panel_group=None predicts all groups."""
        y = panel_time_series_factory(length=50, n_series=3)
        y_train = y[:40]

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=[1, 2]),
        )

        forecaster.fit(y=y_train, X_actual=None, forecasting_horizon=3)

        # Predict with panel_group=None (default)
        y_pred = forecaster.predict(forecasting_horizon=3, panel_group=None)

        # Should have predictions for all 3 series (with __ separator)
        assert "panel__series_0" in y_pred.columns
        assert "panel__series_1" in y_pred.columns
        assert "panel__series_2" in y_pred.columns
        assert len(y_pred) == 3  # 3 forecast steps

    def test_panel_predict_single_group(self, panel_time_series_factory):
        """Test that predict with panel_group filters to a single group."""
        y = panel_time_series_factory(length=50, n_series=3)
        y_train = y[:40]

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=[1, 2]),
        )

        forecaster.fit(y=y_train, X_actual=None, forecasting_horizon=3)

        # Predict only for panel group (all series within the group)
        y_pred = forecaster.predict(forecasting_horizon=3, panel_group="panel")

        # Should have all series columns with __ separator
        assert "panel__series_0" in y_pred.columns
        assert "panel__series_1" in y_pred.columns
        assert "panel__series_2" in y_pred.columns
        assert len(y_pred) == 3

    def test_panel_invalid_group_raises_error(self, panel_time_series_factory):
        """Test that invalid panel_group raises ValueError."""
        y = panel_time_series_factory(length=50, n_series=3)
        y_train = y[:40]

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=[1, 2]),
        )

        forecaster.fit(y=y_train, X_actual=None, forecasting_horizon=3)

        # Try to predict with invalid group name
        with pytest.raises(ValueError, match="not found in fitted forecaster"):
            forecaster.predict(forecasting_horizon=3, groups=["invalid_group"])

    def test_panel_global_data_no_groups(self, time_series_factory):
        """Test that panel_group has no effect on global data."""
        y = time_series_factory(length=50, n_components=1)
        y_train = y[:40]

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=[1, 2]),
        )

        forecaster.fit(y=y_train, X_actual=None, forecasting_horizon=3)

        # Should work the same with or without panel_group
        y_pred_default = forecaster.predict(forecasting_horizon=3, panel_group=None)
        y_pred_explicit = forecaster.predict(forecasting_horizon=3, panel_group=None)

        assert y_pred_default.equals(y_pred_explicit)
        assert "feature_0" in y_pred_default.columns
        assert len(y_pred_default) == 3


def test_panel_fit_insufficient_data_for_group_raises():
    """A panel forecaster whose observation_horizon exceeds a group's rows raises, naming the group."""
    from datetime import datetime

    import polars as pl

    from yohou.point import SeasonalNaive

    time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True)
    y = pl.DataFrame({"time": time, "g0__v": [1.0, 2.0, 3.0, 4.0, 5.0], "g1__v": [2.0, 3.0, 4.0, 5.0, 6.0]})

    with pytest.raises(ValueError, match="Not enough data to set observed y for group 'g0'"):
        SeasonalNaive(seasonality=100).fit(y, forecasting_horizon=1)
