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
                {"forecaster_type": frozenset({"interval"}), "uses_reduction": True, "supports_panel_data": True},
                [],
            ),
        ],
    )
    def test_interval_reduction_panel_checks(self, forecaster, tags, expected_failures, panel_time_series_factory):
        """Run systematic cross-learning checks on IntervalReductionForecaster with panel data."""
        y = panel_time_series_factory(length=100, n_series=3)
        y_train, y_test = y[:80], y[80:]

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_actual=None, forecasting_horizon=3, coverage_rates=[0.1, 0.5, 0.9])

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, None, y_test, None),
            expected_failures=set(expected_failures),
        )


class TestIntervalReductionPanelBehavior:
    def test_panel_interval_global_data(self, time_series_factory):
        """A global-fitted interval forecaster predicts with ``groups=None`` and rejects groups.

        The default ``groups=None`` path returns the flat interval columns, and
        passing any explicit ``groups`` on a globally-fitted forecaster must
        raise rather than silently ignore the argument.
        """
        y = time_series_factory(length=50, n_components=1)
        y_train = y[:40]

        forecaster = IntervalReductionForecaster()

        forecaster.fit(y=y_train, X_actual=None, forecasting_horizon=3, coverage_rates=[0.1, 0.5, 0.9])

        y_pred = forecaster.predict_interval(forecasting_horizon=3, groups=None)
        assert "feature_0_lower_0.1" in y_pred.columns
        assert len(y_pred) == 3

        with pytest.raises(ValueError, match="fitted on global data"):
            forecaster.predict_interval(forecasting_horizon=3, groups=["feature_0"])
