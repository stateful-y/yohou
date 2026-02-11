"""Tests for cross-learning functionality in interval forecasters."""

import pytest
from sklearn.base import clone

from conftest import run_checks
from yohou.interval import IntervalReductionForecaster
from yohou.testing import _yield_yohou_forecaster_checks


class TestIntervalReductionPanelChecks:
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "forecaster,tags,expected_failures",
        [
            (
                IntervalReductionForecaster(),
                {"forecaster_type": "interval", "uses_reduction": True, "supports_panel_data": True},
                [],
            ),
        ],
    )
    def test_interval_reduction_panel_checks(self, forecaster, tags, expected_failures, panel_time_series_factory):
        """Run systematic cross-learning checks on IntervalReductionForecaster with panel data."""
        y = panel_time_series_factory(length=100, n_series=3)
        y_train, y_test = y[:80], y[80:]

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X=None, forecasting_horizon=3, coverage_rates=[0.1, 0.5, 0.9])

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, None, y_test, None),
            expected_failures=set(expected_failures),
        )


class TestIntervalReductionPanelBehavior:
    def test_panel_interval_predict_all_groups_default(self, panel_time_series_factory):
        """Test that interval predict with panel_group_names=None predicts all groups."""
        y = panel_time_series_factory(length=50, n_series=3)
        y_train = y[:40]

        forecaster = IntervalReductionForecaster()

        forecaster.fit(y=y_train, X=None, forecasting_horizon=3, coverage_rates=[0.1, 0.5, 0.9])

        # Predict with panel_group_names=None (default)
        y_pred = forecaster.predict_interval(X=None, forecasting_horizon=3, panel_group_names=None)

        # Should have predictions for all 3 series with intervals (flat columns)
        assert "panel__series_0_lower_0.1" in y_pred.columns
        assert "panel__series_1_lower_0.1" in y_pred.columns
        assert "panel__series_2_lower_0.1" in y_pred.columns
        assert len(y_pred) == 3  # 3 forecast steps

    def test_panel_interval_predict_single_group(self, panel_time_series_factory):
        """Test that interval predict with panel_group_names filters to a single group."""
        y = panel_time_series_factory(length=50, n_series=3)
        y_train = y[:40]

        forecaster = IntervalReductionForecaster()

        forecaster.fit(y=y_train, X=None, forecasting_horizon=3, coverage_rates=[0.1, 0.5, 0.9])

        # Predict only for panel group
        y_pred = forecaster.predict_interval(forecasting_horizon=3, panel_group_names=["panel"])

        # Should still have all series since "panel" is the group name
        assert len(y_pred) == 3

    def test_panel_interval_invalid_group(self, panel_time_series_factory):
        """Test that invalid panel_group_names raises ValueError."""
        y = panel_time_series_factory(length=50, n_series=3)
        y_train = y[:40]

        forecaster = IntervalReductionForecaster()

        forecaster.fit(y=y_train, X=None, forecasting_horizon=3, coverage_rates=[0.1, 0.5, 0.9])

        # Try to predict with invalid group name
        with pytest.raises(ValueError, match="not found in fitted forecaster"):
            forecaster.predict_interval(X=None, forecasting_horizon=3, panel_group_names=["invalid_group"])

    def test_panel_interval_global_data(self, time_series_factory):
        """Test that panel_group_names has no effect on global data."""
        y = time_series_factory(length=50, n_components=1)
        y_train = y[:40]

        forecaster = IntervalReductionForecaster()

        forecaster.fit(y=y_train, X=None, forecasting_horizon=3, coverage_rates=[0.1, 0.5, 0.9])

        # Should work the same with or without panel_group_names
        y_pred_default = forecaster.predict_interval(X=None, forecasting_horizon=3, panel_group_names=None)
        y_pred_explicit = forecaster.predict_interval(X=None, forecasting_horizon=3, panel_group_names=None)

        assert y_pred_default.equals(y_pred_explicit)
        assert "feature_0_lower_0.1" in y_pred_default.columns
        assert len(y_pred_default) == 3
