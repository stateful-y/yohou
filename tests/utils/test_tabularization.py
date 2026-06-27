"""Tests for tabularization utilities."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from yohou.utils.tabularization import tabularize


@pytest.fixture
def simple_df():
    """Create a simple time series DataFrame for tabularization tests."""
    return pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(seconds=4),
            interval="1s",
            eager=True,
        ),
        "value": [10, 20, 30, 40, 50],
    })


@pytest.fixture
def multi_column_df():
    """Create a multi-column time series DataFrame."""
    return pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(seconds=5),
            interval="1s",
            eager=True,
        ),
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })


class TestTabularizeBasic:
    """Basic tabularization tests."""

    def test_single_lag(self, simple_df):
        """Test tabularization with a single lag."""
        result = tabularize(simple_df, lags=[1])
        # Should drop first 1 row (max lag)
        assert len(result) == 4
        # Should have time + lagged columns (no original "value" column)
        assert "time" not in [c for c in result.columns if c.startswith("value")]

    def test_multiple_lags(self, simple_df):
        """Test tabularization with multiple lags."""
        result = tabularize(simple_df, lags=[1, 2])
        # Should drop first max(lags)=2 rows
        assert len(result) == 3
        assert "value_lag_1" in result.columns
        assert "value_lag_2" in result.columns

    def test_lag_values_correct(self, simple_df):
        """Test that lagged values are shifted correctly."""
        result = tabularize(simple_df, lags=[1, 2])
        # Row 0 of result corresponds to time index 2 (value=30)
        # lag_1 should be value at time index 1 (=20)
        # lag_2 should be value at time index 0 (=10)
        assert result["value_lag_1"][0] == 20
        assert result["value_lag_2"][0] == 10

    def test_time_column_preserved(self, simple_df):
        """Test that the time column is preserved in output."""
        result = tabularize(simple_df, lags=[1])
        assert "time" in result.columns

    def test_time_column_not_lagged_as_numeric(self, simple_df):
        """Test that datetime time column is not lagged."""
        result = tabularize(simple_df, lags=[1])
        # time is Datetime type, should not produce time_lag_1
        time_lag_cols = [c for c in result.columns if c.startswith("time_lag")]
        assert len(time_lag_cols) == 0


class TestTabularizeMultiColumn:
    """Test tabularization with multiple columns."""

    def test_all_columns_lagged(self, multi_column_df):
        """Test that all non-time columns are lagged."""
        result = tabularize(multi_column_df, lags=[1])
        assert "a_lag_1" in result.columns
        assert "b_lag_1" in result.columns

    def test_output_shape(self, multi_column_df):
        """Test output shape with multiple columns and lags."""
        result = tabularize(multi_column_df, lags=[1, 2, 3])
        # 6 rows - 3 (max lag) = 3 rows
        assert len(result) == 3
        # time + 2 cols * 3 lags = 7 columns
        assert len(result.columns) == 1 + 2 * 3


class TestTabularizeEdgeCases:
    """Test edge cases for tabularization."""

    def test_large_lag(self, simple_df):
        """Test that large lags reduce output rows correctly."""
        result = tabularize(simple_df, lags=[4])
        assert len(result) == 1

    def test_non_sequential_lags(self, simple_df):
        """Test with non-sequential lag values."""
        result = tabularize(simple_df, lags=[1, 3])
        # max lag is 3, so 5 - 3 = 2 rows
        assert len(result) == 2
        assert "value_lag_1" in result.columns
        assert "value_lag_3" in result.columns

    def test_original_columns_excluded(self, simple_df):
        """Test that original non-time columns are excluded from output."""
        result = tabularize(simple_df, lags=[1])
        # "value" should not be in output, only "value_lag_1"
        assert "value" not in result.columns

    def test_extra_datetime_column_dropped(self):
        """A second Datetime column is excluded from lagging and dropped entirely.

        ``tabularize`` skips Datetime columns when building lag features
        (``dtype != pl.Datetime``) but also removes every non-"time" original
        column via ``pl.exclude``. A second Datetime column therefore receives
        no lag feature and silently disappears from the output.
        """
        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2021, 1, 1),
                end=datetime(2021, 1, 1) + timedelta(seconds=4),
                interval="1s",
                eager=True,
            ),
            "event_date": pl.datetime_range(
                start=datetime(2020, 6, 1),
                end=datetime(2020, 6, 1) + timedelta(seconds=4),
                interval="1s",
                eager=True,
            ),
            "value": [10, 20, 30, 40, 50],
        })
        result = tabularize(df, lags=[1])
        assert result.columns == ["time", "value_lag_1"]
        assert "event_date" not in result.columns
        assert "event_date_lag_1" not in result.columns


class TestTabularizeIntegration:
    """Integration-style tests for tabularize."""

    def test_lag_values_match_manual_shift(self):
        """Lag columns equal the manually shifted source values."""
        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2021, 1, 1),
                end=datetime(2021, 1, 1) + timedelta(seconds=9),
                interval="1s",
                eager=True,
            ),
            "x": list(range(10)),
        })
        result = tabularize(df, lags=[1, 2])
        # max(lags) == 2 rows dropped, so result row 0 is source index 2 (x == 2).
        assert result["x_lag_1"].to_list() == list(range(1, 9))
        assert result["x_lag_2"].to_list() == list(range(0, 8))
