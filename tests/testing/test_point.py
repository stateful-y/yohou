"""Tests for yohou.testing.point check functions."""

import pytest

from yohou.point.naive import SeasonalNaive
from yohou.testing.point import (
    check_point_prediction_structure,
    check_point_prediction_types,
)


class TestPointChecks:
    """Tests for point prediction check functions."""

    def test_check_point_prediction_structure(self, y_X_factory):
        """Test check_point_prediction_structure passes for valid point forecaster."""
        y, X = y_X_factory(length=50, n_targets=2, n_features=3, seed=42)
        forecaster = SeasonalNaive(seasonality=12)
        forecaster.fit(y[:40], X[:40], forecasting_horizon=3)

        # Should not raise
        check_point_prediction_structure(forecaster, y[:40])

    def test_check_point_prediction_structure_different_horizon(self, y_X_factory):
        """Test check validates prediction with different horizon."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        forecaster = SeasonalNaive(seasonality=12)
        forecaster.fit(y[:40], X[:40], forecasting_horizon=3)

        # Should not raise - check uses its own horizon internally
        check_point_prediction_structure(forecaster, y[:40])

    def test_check_point_prediction_types(self):
        """Test check_point_prediction_types passes for valid point forecaster."""
        forecaster = SeasonalNaive(seasonality=12)

        # Should not raise
        check_point_prediction_types(forecaster)

    def test_check_point_prediction_types_fails_for_wrong_type(self):
        """Test check fails if forecaster_type doesn't include 'point'."""
        from yohou.class_proba import ClassProbaReductionForecaster

        # Class-proba forecaster has no point capability
        forecaster = ClassProbaReductionForecaster()

        # Should raise AssertionError - no 'point' in forecaster_type
        with pytest.raises(AssertionError, match="point"):
            check_point_prediction_types(forecaster)
