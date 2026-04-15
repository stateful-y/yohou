"""Tests for yohou.testing.interval check functions."""

import pytest

from yohou.interval.split_conformal import SplitConformalForecaster
from yohou.testing.interval import (
    check_coverage_rates_parameter,
    check_coverage_rates_validation,
    check_interval_bounds,
    check_interval_prediction_columns,
    check_interval_prediction_types,
)


class TestIntervalChecks:
    """Tests for interval prediction check functions."""

    def test_check_interval_prediction_columns(self, y_X_factory):
        """Test check_interval_prediction_columns passes for valid interval forecaster."""
        y, X = y_X_factory(length=100, n_targets=2, n_features=3, seed=42)
        forecaster = SplitConformalForecaster(calibration_size=50)
        forecaster.fit(y[:80], X[:80], forecasting_horizon=3)

        # Should not raise
        check_interval_prediction_columns(forecaster, y[:80], X[:80])

    def test_check_interval_prediction_columns_single_coverage(self, y_X_factory):
        """Test check validates single coverage rate."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)
        forecaster = SplitConformalForecaster(calibration_size=50)
        forecaster.fit(y[:80], X[:80], forecasting_horizon=3, coverage_rates=[0.95])

        # Should not raise
        check_interval_prediction_columns(forecaster, y[:80], X[:80])

    def test_check_interval_bounds(self, y_X_factory):
        """Test check_interval_bounds passes for valid interval forecaster."""
        y, X = y_X_factory(length=100, n_targets=2, n_features=3, seed=42)
        forecaster = SplitConformalForecaster(calibration_size=50)
        forecaster.fit(y[:80], X[:80], forecasting_horizon=3)

        # Should not raise
        check_interval_bounds(forecaster, y[:80], X[:80])

    def test_check_interval_bounds_single_target(self, y_X_factory):
        """Test check validates single target intervals."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)
        forecaster = SplitConformalForecaster(calibration_size=50)
        forecaster.fit(y[:80], X[:80], forecasting_horizon=3, coverage_rates=[0.9])

        # Should not raise
        check_interval_bounds(forecaster, y[:80], X[:80])

    def test_check_interval_prediction_types(self):
        """Test check_interval_prediction_types passes for valid interval forecaster."""
        forecaster = SplitConformalForecaster()

        # Should not raise
        check_interval_prediction_types(forecaster)

    def test_check_interval_prediction_types_fails_for_point_only(self):
        """Test check fails if prediction_types doesn't include interval."""
        from yohou.point.naive import SeasonalNaive

        # Point-only forecaster
        forecaster = SeasonalNaive(seasonality=12)

        # Should raise AssertionError
        with pytest.raises(AssertionError, match="interval"):
            check_interval_prediction_types(forecaster)

    def test_check_coverage_rates_parameter(self, y_X_factory):
        """Test check_coverage_rates_parameter passes for valid interval forecaster."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)
        forecaster = SplitConformalForecaster(calibration_size=50)
        forecaster.fit(y[:80], X[:80], forecasting_horizon=3)

        check_coverage_rates_parameter(forecaster)

    def test_check_coverage_rates_validation(self, y_X_factory):
        """Test check_coverage_rates_validation passes for valid interval forecaster."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)
        forecaster = SplitConformalForecaster(calibration_size=50)
        forecaster.fit(y[:80], X[:80], forecasting_horizon=3)

        # Should not raise
        check_coverage_rates_validation(forecaster, y[:80], X[:80])
