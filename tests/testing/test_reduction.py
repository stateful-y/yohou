"""Tests for yohou.testing.reduction check functions."""

from sklearn.linear_model import LinearRegression

from yohou.interval.reduction import IntervalReductionForecaster
from yohou.point.reduction import PointReductionForecaster
from yohou.testing.reduction import (
    check_estimator_parameter,
    check_reduction_strategy,
)


class TestReductionChecks:
    """Tests for reduction forecaster check functions."""

    def test_check_estimator_parameter_point(self):
        """Test check_estimator_parameter passes for point reduction forecaster."""
        forecaster = PointReductionForecaster(estimator=LinearRegression())

        # Should not raise
        check_estimator_parameter(forecaster)

    def test_check_estimator_parameter_interval(self):
        """Test check_estimator_parameter passes for interval reduction forecaster."""
        forecaster = IntervalReductionForecaster(estimator=LinearRegression())

        # Should not raise
        check_estimator_parameter(forecaster)

    def test_check_estimator_parameter_default(self):
        """Test check validates default estimator is set."""
        forecaster = PointReductionForecaster()

        # Should not raise - default estimator is valid
        check_estimator_parameter(forecaster)

    def test_check_reduction_strategy_point(self):
        """Test check_reduction_strategy passes for point reduction forecaster."""
        forecaster = PointReductionForecaster(reduction_strategy="direct")

        # Should not raise
        check_reduction_strategy(forecaster)

    def test_check_reduction_strategy_multi_output(self):
        """Test check validates multi-output strategy."""
        forecaster = PointReductionForecaster(reduction_strategy="multi-output")

        # Should not raise
        check_reduction_strategy(forecaster)

    def test_check_reduction_strategy_interval(self):
        """Test check validates interval reduction forecaster."""
        forecaster = IntervalReductionForecaster(reduction_strategy="direct")

        # Should not raise
        check_reduction_strategy(forecaster)

    def test_check_reduction_strategy_dir_rec(self):
        """Test check validates dir-rec strategy."""
        forecaster = PointReductionForecaster(reduction_strategy="dir-rec")

        # Should not raise
        check_reduction_strategy(forecaster)

    def test_check_reduction_strategy_dir_rec_interval(self):
        """Test check validates dir-rec strategy on interval forecaster."""
        forecaster = IntervalReductionForecaster(reduction_strategy="dir-rec")

        # Should not raise
        check_reduction_strategy(forecaster)

    def test_check_reduction_strategy_no_attribute(self):
        """Test check returns early when forecaster lacks reduction_strategy."""
        from sklearn.linear_model import LinearRegression

        estimator = LinearRegression()
        # LinearRegression has no reduction_strategy attribute, so check returns early
        check_reduction_strategy(estimator)
