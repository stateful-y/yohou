"""Tests for parameter validation in forecasters."""


import pytest

from yohou.interval_forecaster import IntervalReductionForecaster, SplitConformalForecaster
from yohou.point_forecaster import PointReductionForecaster, SeasonalNaive

# Add parent directory to path for imports
from yohou.testing import check_coverage_rates_validation, check_forecasting_horizon_validation


def test_point_forecaster_horizon_validation(y_X_factory):
    """Test that point forecasters validate forecasting_horizon in fit and predict."""
    y, X = y_X_factory(length=50, seed=42)

    # Test PointReductionForecaster
    forecaster = PointReductionForecaster()
    check_forecasting_horizon_validation(forecaster, y, X)

    # Test SeasonalNaive
    forecaster = SeasonalNaive(seasonality=7)
    check_forecasting_horizon_validation(forecaster, y, X)


def test_interval_forecaster_horizon_validation(y_X_factory):
    """Test that interval forecasters validate forecasting_horizon in fit and predict."""
    y, X = y_X_factory(length=50, seed=42)

    # Test IntervalReductionForecaster
    forecaster = IntervalReductionForecaster()
    check_forecasting_horizon_validation(forecaster, y, X)

    # Test SplitConformalForecaster
    forecaster = SplitConformalForecaster()
    check_forecasting_horizon_validation(forecaster, y, X)


def test_interval_forecaster_coverage_rates_validation(y_X_factory):
    """Test that interval forecasters validate coverage_rates in fit and predict."""
    y, X = y_X_factory(length=200, seed=42)

    # Test IntervalReductionForecaster
    forecaster = IntervalReductionForecaster()
    check_coverage_rates_validation(forecaster, y, X)

    # Note: SplitConformalForecaster has implementation issues, tested separately


def test_predict_validates_horizon(y_X_factory):
    """Test that predict() validates forecasting_horizon parameter."""
    y, X = y_X_factory(length=50, seed=42)
    y_train, y_test = y[:40], y[40:]
    X_train, X_test = X[:40], X[40:]

    forecaster = PointReductionForecaster()
    forecaster.fit(y_train, X_train, forecasting_horizon=3)

    # Test invalid horizon in predict
    with pytest.raises(ValueError, match="forecasting_horizon|positive"):
        forecaster.predict(forecasting_horizon=0, X=X_test)

    with pytest.raises(ValueError, match="forecasting_horizon|positive"):
        forecaster.predict(forecasting_horizon=-1, X=X_test)


def test_predict_interval_validates_coverage_rates(y_X_factory):
    """Test that predict_interval() validates coverage_rates parameter."""
    y, X = y_X_factory(length=50, seed=42)
    y_train, y_test = y[:40], y[40:]
    X_train, X_test = X[:40], X[40:]

    forecaster = IntervalReductionForecaster()
    forecaster.fit(y_train, X_train, forecasting_horizon=3, coverage_rates=[0.95])

    # Test invalid coverage_rates in predict_interval
    with pytest.raises(ValueError, match="coverage"):
        forecaster.predict_interval(forecasting_horizon=3, coverage_rates=[0.0])

    with pytest.raises(ValueError, match="coverage"):
        forecaster.predict_interval(forecasting_horizon=3, coverage_rates=[1.5])

    with pytest.raises(ValueError, match="coverage"):
        forecaster.predict_interval(forecasting_horizon=3, coverage_rates=[-0.5])
