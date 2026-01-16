"""Tests for yohou.testing.reduction check functions."""

from sklearn.linear_model import LinearRegression

from yohou.interval_forecaster.reduction import IntervalReductionForecaster
from yohou.point_forecaster.reduction import PointReductionForecaster
from yohou.testing.reduction import (
    check_estimator_parameter,
    check_reduction_strategy,
)


def test_check_estimator_parameter_point():
    """Test check_estimator_parameter passes for point reduction forecaster."""
    forecaster = PointReductionForecaster(estimator=LinearRegression())

    # Should not raise
    check_estimator_parameter(forecaster)


def test_check_estimator_parameter_interval():
    """Test check_estimator_parameter passes for interval reduction forecaster."""
    forecaster = IntervalReductionForecaster(estimator=LinearRegression())

    # Should not raise
    check_estimator_parameter(forecaster)


def test_check_estimator_parameter_default():
    """Test check validates default estimator is set."""
    forecaster = PointReductionForecaster()

    # Should not raise - default estimator is valid
    check_estimator_parameter(forecaster)


def test_check_reduction_strategy_point():
    """Test check_reduction_strategy passes for point reduction forecaster."""
    forecaster = PointReductionForecaster(reduction_strategy="direct")

    # Should not raise
    check_reduction_strategy(forecaster)


def test_check_reduction_strategy_multi_output():
    """Test check validates multi-output strategy."""
    forecaster = PointReductionForecaster(reduction_strategy="multi-output")

    # Should not raise
    check_reduction_strategy(forecaster)


def test_check_reduction_strategy_interval():
    """Test check validates interval reduction forecaster."""
    forecaster = IntervalReductionForecaster(reduction_strategy="direct")

    # Should not raise
    check_reduction_strategy(forecaster)
