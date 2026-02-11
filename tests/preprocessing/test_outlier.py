"""Tests for outlier handling transformers.

Tests OutlierThresholdHandler and OutlierPercentileHandler basic functionality.
"""

from conftest import run_checks
from yohou.preprocessing import (
    OutlierPercentileHandler,
    OutlierThresholdHandler,
)
from yohou.testing import _yield_yohou_transformer_checks


class TestOutlierThresholdHandler:
    """Test OutlierThresholdHandler functionality."""

    def test_systematic_checks(self, time_series_train_test_factory):
        """Run all systematic checks for OutlierThresholdHandler."""
        transformer = OutlierThresholdHandler(low=-3.0, high=3.0, strategy="clip")

        X_train, X_test = time_series_train_test_factory(train_length=100, test_length=50)

        transformer.fit(X_train)

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
        )

    def test_clip_strategy(self, time_series_with_outliers_factory):
        """Test OutlierThresholdHandler clip strategy."""
        X = time_series_with_outliers_factory(length=50, n_components=2, outlier_magnitude=10.0)
        handler = OutlierThresholdHandler(low=-5.0, high=5.0, strategy="clip")
        handler.fit(X)
        X_clipped = handler.transform(X)

        # Check all values are within bounds
        assert "time" in X_clipped.columns
        for col in X_clipped.columns:
            if col != "time":
                assert X_clipped[col].min() >= -5.0
                assert X_clipped[col].max() <= 5.0

    def test_nan_strategy(self, time_series_with_outliers_factory):
        """Test OutlierThresholdHandler nan strategy."""
        X = time_series_with_outliers_factory(length=50, n_components=2, outlier_magnitude=10.0)
        handler = OutlierThresholdHandler(low=-5.0, high=5.0, strategy="nan")
        handler.fit(X)
        X_nan = handler.transform(X)

        assert "time" in X_nan.columns

    def test_preserves_time_column(self, time_series_with_outliers_factory):
        """Test OutlierThresholdHandler preserves time column."""
        X = time_series_with_outliers_factory(length=50, n_components=2)
        handler = OutlierThresholdHandler(low=-5.0, high=5.0, strategy="clip")
        handler.fit(X)
        X_transformed = handler.transform(X)

        assert "time" in X_transformed.columns
        assert len(X_transformed) == len(X)


class TestOutlierPercentileHandler:
    """Test OutlierPercentileHandler functionality."""

    def test_systematic_checks(self, time_series_train_test_factory):
        """Run all systematic checks for OutlierPercentileHandler."""
        transformer = OutlierPercentileHandler(low=5, high=95, strategy="clip")

        X_train, X_test = time_series_train_test_factory(train_length=100, test_length=50)

        transformer.fit(X_train)

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
        )

    def test_clip_strategy(self, time_series_with_outliers_factory):
        """Test OutlierPercentileHandler clip strategy."""
        X = time_series_with_outliers_factory(length=50, n_components=2, outlier_magnitude=10.0)
        handler = OutlierPercentileHandler(low=5.0, high=95.0, strategy="clip")
        handler.fit(X)
        X_clipped = handler.transform(X)

        assert "time" in X_clipped.columns

    def test_nan_strategy(self, time_series_with_outliers_factory):
        """Test OutlierPercentileHandler nan strategy."""
        X = time_series_with_outliers_factory(length=50, n_components=2, outlier_magnitude=10.0)
        handler = OutlierPercentileHandler(low=5.0, high=95.0, strategy="nan")
        handler.fit(X)
        X_nan = handler.transform(X)

        assert "time" in X_nan.columns

    def test_preserves_time_column(self, time_series_with_outliers_factory):
        """Test OutlierPercentileHandler preserves time column."""
        X = time_series_with_outliers_factory(length=50, n_components=2)
        handler = OutlierPercentileHandler(low=5.0, high=95.0, strategy="clip")
        handler.fit(X)
        X_transformed = handler.transform(X)

        assert "time" in X_transformed.columns
        assert len(X_transformed) == len(X)
