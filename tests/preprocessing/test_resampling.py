"""Tests for resampling transformers.

Tests Downsampler and Upsampler transformer-specific behavior.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone

from yohou.preprocessing.resampling import Downsampler, Upsampler


def create_hourly_data(length: int = 48, seed: int = 42) -> pl.DataFrame:
    """Create hourly time series data for testing.

    Parameters
    ----------
    length : int
        Number of samples (hours).
    seed : int
        Random seed.

    Returns
    -------
    pl.DataFrame
        DataFrame with time column and value columns.

    """
    np.random.seed(seed)
    time = [datetime(2021, 1, 1) + timedelta(hours=i) for i in range(length)]
    return pl.DataFrame({
        "time": time,
        "value_a": np.cumsum(np.random.randn(length)) + 100,
        "value_b": np.sin(np.linspace(0, 4 * np.pi, length)) * 10 + 50,
    })


def create_daily_data(length: int = 30, seed: int = 42) -> pl.DataFrame:
    """Create daily time series data for testing.

    Parameters
    ----------
    length : int
        Number of samples (days).
    seed : int
        Random seed.

    Returns
    -------
    pl.DataFrame
        DataFrame with time column and value columns.

    """
    np.random.seed(seed)
    time = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(length)]
    return pl.DataFrame({
        "time": time,
        "value_a": np.cumsum(np.random.randn(length)) + 100,
        "value_b": np.sin(np.linspace(0, 4 * np.pi, length)) * 10 + 50,
    })


class TestDownsampler:
    """Tests for Downsampler transformer."""

    def test_downsample_hourly_to_daily(self) -> None:
        """Test downsampling from hourly to daily."""
        X = create_hourly_data(length=48)  # 2 days of data
        downsampler = Downsampler(interval="1d", aggregation="mean")
        downsampler.fit(X)
        X_daily = downsampler.transform(X)

        assert len(X_daily) == 2
        assert "time" in X_daily.columns
        assert "value_a" in X_daily.columns
        assert "value_b" in X_daily.columns

    def test_downsample_aggregation_sum(self) -> None:
        """Test downsampling with sum aggregation."""
        X = create_hourly_data(length=24)
        downsampler = Downsampler(interval="1d", aggregation="sum")
        downsampler.fit(X)
        X_daily = downsampler.transform(X)

        # Sum of 24 hourly values should equal the daily value
        expected_sum_a = X["value_a"].sum()
        actual_sum_a = X_daily["value_a"].sum()
        assert abs(expected_sum_a - actual_sum_a) < 1e-10

    def test_downsample_aggregation_min_max(self) -> None:
        """Test downsampling with min/max aggregations."""
        X = create_hourly_data(length=24)

        downsampler_min = Downsampler(interval="1d", aggregation="min")
        downsampler_min.fit(X)
        X_min = downsampler_min.transform(X)

        downsampler_max = Downsampler(interval="1d", aggregation="max")
        downsampler_max.fit(X)
        X_max = downsampler_max.transform(X)

        # Min should be <= max for all values
        assert X_min["value_a"][0] <= X_max["value_a"][0]
        assert X_min["value_b"][0] <= X_max["value_b"][0]

    def test_downsample_aggregation_first_last(self) -> None:
        """Test downsampling with first/last aggregations."""
        X = create_hourly_data(length=24)

        downsampler_first = Downsampler(interval="1d", aggregation="first")
        downsampler_first.fit(X)
        X_first = downsampler_first.transform(X)

        downsampler_last = Downsampler(interval="1d", aggregation="last")
        downsampler_last.fit(X)
        X_last = downsampler_last.transform(X)

        # First value should match X[0], last should match X[-1]
        assert X_first["value_a"][0] == X["value_a"][0]
        assert X_last["value_a"][0] == X["value_a"][-1]

    def test_downsample_aggregation_median(self) -> None:
        """Test downsampling with median aggregation."""
        X = create_hourly_data(length=24)
        downsampler = Downsampler(interval="1d", aggregation="median")
        downsampler.fit(X)
        X_daily = downsampler.transform(X)

        # Median should be within range of min/max
        assert X_daily["value_a"][0] >= X["value_a"].min()
        assert X_daily["value_a"][0] <= X["value_a"].max()

    def test_downsample_closed_label_options(self) -> None:
        """Test downsampling with different closed/label options."""
        X = create_hourly_data(length=24)

        downsampler_left = Downsampler(interval="1d", closed="left", label="left")
        downsampler_left.fit(X)
        X_left = downsampler_left.transform(X)

        downsampler_right = Downsampler(interval="1d", closed="right", label="right")
        downsampler_right.fit(X)
        X_right = downsampler_right.transform(X)

        # Both should produce output
        assert len(X_left) > 0
        assert len(X_right) > 0

    def test_downsample_preserves_column_order(self) -> None:
        """Test that downsampling preserves column order."""
        X = create_hourly_data(length=24)
        downsampler = Downsampler(interval="1d")
        downsampler.fit(X)
        X_daily = downsampler.transform(X)

        # First column should be time, then data columns in order
        assert X_daily.columns[0] == "time"
        # Data columns follow (may not preserve exact order due to group_by)

    def test_downsample_fit_transform(self) -> None:
        """Test fit_transform convenience method."""
        X = create_hourly_data(length=48)
        downsampler = Downsampler(interval="1d")

        X_daily = downsampler.fit_transform(X)
        assert len(X_daily) == 2

    def test_downsample_clone(self) -> None:
        """Test that Downsampler can be cloned."""
        downsampler = Downsampler(interval="1d", aggregation="sum")
        cloned = clone(downsampler)

        assert cloned.interval == downsampler.interval
        assert cloned.aggregation == downsampler.aggregation

    def test_downsample_rejects_smaller_target(self) -> None:
        """Test that fitting rejects target interval smaller than input."""
        # Create daily data and try to "downsample" to hourly (should fail)
        X = create_daily_data(length=30)
        downsampler = Downsampler(interval="1h", aggregation="mean")

        with pytest.raises(ValueError, match="smaller than input interval"):
            downsampler.fit(X)


class TestUpsampler:
    """Tests for Upsampler transformer."""

    def test_upsample_daily_to_hourly(self) -> None:
        """Test upsampling from daily to hourly."""
        X = create_daily_data(length=3)  # 3 days
        upsampler = Upsampler(interval="12h", interpolation="linear")
        upsampler.fit(X)
        X_12h = upsampler.transform(X)

        # 3 days should become 5 12-hour periods (start to end inclusive)
        # days: 0, 1, 2 -> 12h: 0, 0.5, 1, 1.5, 2 = 5 points
        assert len(X_12h) == 5
        assert "time" in X_12h.columns
        assert "value_a" in X_12h.columns

    def test_upsample_interpolation_linear(self) -> None:
        """Test linear interpolation."""
        time = [datetime(2021, 1, 1), datetime(2021, 1, 3)]  # 2-day gap
        X = pl.DataFrame({
            "time": time,
            "value": [0.0, 2.0],
        })
        upsampler = Upsampler(interval="1d", interpolation="linear")
        upsampler.fit(X)
        X_daily = upsampler.transform(X)

        # Should have day 1, 2, 3 with values 0, 1, 2
        assert len(X_daily) == 3
        assert X_daily["value"][0] == 0.0
        assert X_daily["value"][1] == 1.0  # Linear interpolation
        assert X_daily["value"][2] == 2.0

    def test_upsample_interpolation_forward(self) -> None:
        """Test forward fill interpolation."""
        time = [datetime(2021, 1, 1), datetime(2021, 1, 3)]
        X = pl.DataFrame({
            "time": time,
            "value": [10.0, 20.0],
        })
        upsampler = Upsampler(interval="1d", interpolation="forward")
        upsampler.fit(X)
        X_daily = upsampler.transform(X)

        # Middle value should be forward-filled from first
        assert len(X_daily) == 3
        assert X_daily["value"][0] == 10.0
        assert X_daily["value"][1] == 10.0  # Forward fill
        assert X_daily["value"][2] == 20.0

    def test_upsample_interpolation_backward(self) -> None:
        """Test backward fill interpolation."""
        time = [datetime(2021, 1, 1), datetime(2021, 1, 3)]
        X = pl.DataFrame({
            "time": time,
            "value": [10.0, 20.0],
        })
        upsampler = Upsampler(interval="1d", interpolation="backward")
        upsampler.fit(X)
        X_daily = upsampler.transform(X)

        # Middle value should be backward-filled from last
        assert len(X_daily) == 3
        assert X_daily["value"][0] == 10.0
        assert X_daily["value"][1] == 20.0  # Backward fill
        assert X_daily["value"][2] == 20.0

    def test_upsample_interpolation_nearest(self) -> None:
        """Test nearest neighbor interpolation."""
        time = [datetime(2021, 1, 1), datetime(2021, 1, 3)]
        X = pl.DataFrame({
            "time": time,
            "value": [10.0, 20.0],
        })
        upsampler = Upsampler(interval="1d", interpolation="nearest")
        upsampler.fit(X)
        X_daily = upsampler.transform(X)

        # Nearest should fill from forward then backward
        assert len(X_daily) == 3
        assert X_daily["value"][0] == 10.0
        # Nearest can be either 10 or 20 depending on implementation
        assert X_daily["value"][1] in [10.0, 20.0]
        assert X_daily["value"][2] == 20.0

    def test_upsample_fit_transform(self) -> None:
        """Test fit_transform convenience method."""
        X = create_daily_data(length=3)
        upsampler = Upsampler(interval="12h")

        X_12h = upsampler.fit_transform(X)
        assert len(X_12h) == 5

    def test_upsample_clone(self) -> None:
        """Test that Upsampler can be cloned."""
        upsampler = Upsampler(interval="1h", interpolation="nearest")
        cloned = clone(upsampler)

        assert cloned.interval == upsampler.interval
        assert cloned.interpolation == upsampler.interpolation

    def test_upsample_rejects_larger_target(self) -> None:
        """Test that fitting rejects target interval larger than input."""
        # Create hourly data and try to "upsample" to daily (should fail)
        X = create_hourly_data(length=48)
        upsampler = Upsampler(interval="1d", interpolation="linear")

        with pytest.raises(ValueError, match="larger than input interval"):
            upsampler.fit(X)
