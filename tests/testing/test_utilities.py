"""Test suite for new testing utilities (fixtures and assertion helpers)."""

from datetime import datetime

import polars as pl
import pytest

from testing.assertions import (
    assert_continuous_time,
    assert_forecaster_output_valid,
    assert_transformer_output_valid,
)
from yohou.point.naive import SeasonalNaive
from yohou.stationarity.transformers import SeasonalDifferencing


class TestDataFactories:
    """Tests for data factory fixtures."""

    def test_y_X_factory_panel(self, y_X_factory):
        """Test that y_X_factory generates panel data correctly."""
        y, X = y_X_factory(length=50, n_targets=2, n_features=3, panel=True, n_groups=3, seed=42)

        # Check y structure
        assert "time" in y.columns
        assert len(y) == 50
        # Should have 2 targets * 3 groups = 6 panel columns + time
        assert len(y.columns) == 7

        # Check panel column naming
        expected_y_cols = [
            "group_0__y_0",
            "group_0__y_1",
            "group_1__y_0",
            "group_1__y_1",
            "group_2__y_0",
            "group_2__y_1",
        ]
        for col in expected_y_cols:
            assert col in y.columns

        # Check X structure
        assert X is not None
        assert "time" in X.columns
        assert len(X) == 50
        # Should have 3 features * 3 groups = 9 panel columns + time
        assert len(X.columns) == 10

        # Check panel column naming
        expected_X_cols = [
            "group_0__X_0",
            "group_0__X_1",
            "group_0__X_2",
            "group_1__X_0",
            "group_1__X_1",
            "group_1__X_2",
            "group_2__X_0",
            "group_2__X_1",
            "group_2__X_2",
        ]
        for col in expected_X_cols:
            assert col in X.columns

    def test_X_factory_alias(self, X_factory):
        """Test that X_factory alias works correctly."""
        X = X_factory(length=30, n_components=2, seed=42)

        assert isinstance(X, pl.DataFrame)
        assert "time" in X.columns
        assert len(X) == 30
        assert "feature_0" in X.columns
        assert "feature_1" in X.columns


class TestPatternFactory:
    """Tests for pattern_factory fixture."""

    def test_linear_trend(self, pattern_factory):
        """Test pattern_factory generates linear trend correctly."""
        df = pattern_factory("linear_trend", length=100, amplitude=2.0, noise_level=0.0, seed=42)

        assert isinstance(df, pl.DataFrame)
        assert "time" in df.columns
        assert "target" in df.columns
        assert len(df) == 100

        # Check linear trend (no noise, so exact)
        values = df["target"].to_list()
        assert values[0] == 0.0
        assert values[10] == 20.0
        assert values[50] == 100.0

    def test_seasonal(self, pattern_factory):
        """Test pattern_factory generates seasonal pattern correctly."""
        df = pattern_factory("seasonal", length=24, amplitude=1.0, noise_level=0.0, seed=42)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 24

        # Check seasonality (should complete 2 full cycles with period=12)
        values = df["target"].to_list()
        # At t=0 and t=12, sin(0) and sin(2pi) ~= 0
        assert abs(values[0]) < 0.1
        assert abs(values[12]) < 0.1

    def test_all_patterns(self, pattern_factory):
        """Test that all pattern types can be generated without errors."""
        patterns = [
            "linear_trend",
            "exponential_trend",
            "seasonal",
            "trend_seasonal",
            "step_change",
            "random_walk",
        ]

        for pattern in patterns:
            df = pattern_factory(pattern, length=50, amplitude=1.0, noise_level=0.1, seed=42)
            assert isinstance(df, pl.DataFrame)
            assert len(df) == 50
            assert "time" in df.columns
            assert "target" in df.columns


class TestAssertionHelpers:
    """Tests for assertion helper functions."""

    def test_assert_forecaster_output_valid_basic(self):
        """Test assert_forecaster_output_valid with basic validation."""
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 5),
            interval="1d",
            eager=True,
        )
        y_pred = pl.DataFrame({
            "time": time,
            "observed_time": [datetime(2020, 12, 31)] * 5,
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        # Should not raise
        assert_forecaster_output_valid(y_pred, expected_length=5)

    def test_assert_forecaster_output_valid_with_columns(self):
        """Test assert_forecaster_output_valid with column checking."""
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 3),
            interval="1d",
            eager=True,
        )
        y_pred = pl.DataFrame({
            "time": time,
            "observed_time": [datetime(2020, 12, 31)] * 3,
            "y_0": [1.0, 2.0, 3.0],
            "y_1": [4.0, 5.0, 6.0],
        })

        # Should not raise
        assert_forecaster_output_valid(y_pred, expected_length=3, expected_columns=["y_0", "y_1"])

        # Should raise with wrong columns
        with pytest.raises(AssertionError):
            assert_forecaster_output_valid(y_pred, expected_length=3, expected_columns=["y_0"])

    def test_assert_continuous_time(self):
        """Test assert_continuous_time validates intervals."""
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 10),
            interval="1d",
            eager=True,
        )
        df = pl.DataFrame({"time": time, "value": list(range(10))})

        # Should not raise
        assert_continuous_time(df, expected_interval="1d")

        # Should raise with wrong interval
        with pytest.raises(AssertionError, match="Expected interval 1h, got 1d"):
            assert_continuous_time(df, expected_interval="1h")

    def test_assert_transformer_output_valid(self, X_factory):
        """Test assert_transformer_output_valid with transformer."""
        X = X_factory(length=50, n_components=2, seed=42)

        # Transform with SeasonalDifferencing (reduces length by seasonality)
        transformer = SeasonalDifferencing(seasonality=1)
        X_transformed = transformer.fit_transform(X)

        # Should not raise
        assert_transformer_output_valid(X_transformed, X, expected_length=49)

        # Should raise with wrong length
        with pytest.raises(AssertionError):
            assert_transformer_output_valid(X_transformed, X, expected_length=50)


class TestIntegration:
    """Integration tests for testing utilities."""

    def test_pattern_factory_with_forecaster(self, pattern_factory):
        """Test pattern_factory integration with actual forecaster."""
        df = pattern_factory("seasonal", length=100, amplitude=2.0, noise_level=0.1, seed=42)

        # Fit forecaster
        forecaster = SeasonalNaive(seasonality=12)
        forecaster.fit(df[:80], forecasting_horizon=5)

        # Predict
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Use assertion helpers
        assert_forecaster_output_valid(y_pred, expected_length=5, expected_columns=["target"])
        assert_continuous_time(y_pred, expected_interval="1d")
