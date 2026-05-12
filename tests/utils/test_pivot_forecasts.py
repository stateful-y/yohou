"""Tests for pivot_forecasts utility."""

from datetime import datetime

import polars as pl
import pytest

from yohou.utils.pivot import pivot_forecasts


class TestPivotForecasts:
    """Tests for the pivot_forecasts utility function."""

    def test_basic_single_column(self):
        """Pivot single value column with two vintages of three steps each."""
        df = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 3 + [datetime(2020, 1, 2)] * 3,
            "time": [
                datetime(2020, 1, 2),
                datetime(2020, 1, 3),
                datetime(2020, 1, 4),
                datetime(2020, 1, 3),
                datetime(2020, 1, 4),
                datetime(2020, 1, 5),
            ],
            "temp": [10.0, 11.0, 12.0, 15.0, 16.0, 17.0],
        })
        result = pivot_forecasts(df)

        assert result.columns == ["time", "temp_step_1", "temp_step_2", "temp_step_3"]
        assert result.shape == (2, 4)
        assert result["time"].to_list() == [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        assert result["temp_step_1"].to_list() == [10.0, 15.0]
        assert result["temp_step_2"].to_list() == [11.0, 16.0]
        assert result["temp_step_3"].to_list() == [12.0, 17.0]

    def test_multiple_value_columns(self):
        """Pivot with two value columns: temp and wind."""
        df = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 2,
            "time": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "temp": [10.0, 11.0],
            "wind": [5.0, 6.0],
        })
        result = pivot_forecasts(df)

        assert result.columns == [
            "time",
            "temp_step_1",
            "wind_step_1",
            "temp_step_2",
            "wind_step_2",
        ]
        assert result["temp_step_1"].to_list() == [10.0]
        assert result["wind_step_2"].to_list() == [6.0]

    def test_ragged_vintages(self):
        """Shorter vintage produces nulls for missing steps."""
        df = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 3 + [datetime(2020, 1, 2)] * 2,
            "time": [
                datetime(2020, 1, 2),
                datetime(2020, 1, 3),
                datetime(2020, 1, 4),
                datetime(2020, 1, 3),
                datetime(2020, 1, 4),
            ],
            "temp": [10.0, 11.0, 12.0, 20.0, 21.0],
        })
        result = pivot_forecasts(df)

        assert result.shape == (2, 4)
        assert result["temp_step_3"].to_list() == [12.0, None]

    def test_single_vintage(self):
        """One vintage produces one row."""
        df = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 2,
            "time": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "temp": [10.0, 11.0],
        })
        result = pivot_forecasts(df)

        assert result.shape == (1, 3)
        assert result["time"].to_list() == [datetime(2020, 1, 1)]

    def test_custom_column_names(self):
        """Configurable vintage_col and time_col."""
        df = pl.DataFrame({
            "issue_date": [datetime(2020, 1, 1)] * 2,
            "target_date": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "value": [1.0, 2.0],
        })
        result = pivot_forecasts(df, vintage_col="issue_date", time_col="target_date")

        assert "time" in result.columns
        assert result["time"].to_list() == [datetime(2020, 1, 1)]
        assert result["value_step_1"].to_list() == [1.0]
        assert result["value_step_2"].to_list() == [2.0]

    def test_missing_vintage_col_raises(self):
        """Missing vintage column raises ValueError."""
        df = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "temp": [10.0],
        })
        with pytest.raises(ValueError, match="'vintage_time' not found"):
            pivot_forecasts(df)

    def test_missing_time_col_raises(self):
        """Missing time column raises ValueError."""
        df = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)],
            "temp": [10.0],
        })
        with pytest.raises(ValueError, match="'target_date' not found"):
            pivot_forecasts(df, time_col="target_date")

    def test_no_value_columns_raises(self):
        """Only vintage and time columns, no values, raises ValueError."""
        df = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)],
            "time": [datetime(2020, 1, 2)],
        })
        with pytest.raises(ValueError, match="No value columns found"):
            pivot_forecasts(df)

    def test_step_naming_pattern(self):
        """Step columns follow <col>_step_<n> naming (1-based)."""
        df = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 4,
            "time": [datetime(2020, 1, d) for d in range(2, 6)],
            "x": [1.0, 2.0, 3.0, 4.0],
        })
        result = pivot_forecasts(df)

        step_cols = [c for c in result.columns if c != "time"]
        assert step_cols == ["x_step_1", "x_step_2", "x_step_3", "x_step_4"]

    def test_integer_values(self):
        """Integer value columns are preserved."""
        df = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 2,
            "time": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "is_holiday": [1, 0],
        })
        result = pivot_forecasts(df)

        assert result["is_holiday_step_1"].to_list() == [1]
        assert result["is_holiday_step_2"].to_list() == [0]

    def test_output_time_is_vintage(self):
        """Output 'time' column contains the vintage times, not target times."""
        df = pl.DataFrame({
            "vintage_time": [datetime(2020, 3, 1)] * 2 + [datetime(2020, 3, 2)] * 2,
            "time": [datetime(2020, 3, 2), datetime(2020, 3, 3)] * 2,
            "val": [1.0, 2.0, 3.0, 4.0],
        })
        result = pivot_forecasts(df)

        assert result["time"].to_list() == [datetime(2020, 3, 1), datetime(2020, 3, 2)]

    def test_preserves_vintage_order(self):
        """Output rows maintain the order of vintages."""
        df = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 3)] * 2 + [datetime(2020, 1, 1)] * 2,
            "time": [datetime(2020, 1, 4), datetime(2020, 1, 5)] * 2,
            "val": [30.0, 31.0, 10.0, 11.0],
        })
        result = pivot_forecasts(df)

        assert result["time"].to_list() == [datetime(2020, 1, 3), datetime(2020, 1, 1)]
        assert result["val_step_1"].to_list() == [30.0, 10.0]

    def test_null_values_in_data(self):
        """Null values in the input are preserved in step columns."""
        df = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 3,
            "time": [datetime(2020, 1, 2), datetime(2020, 1, 3), datetime(2020, 1, 4)],
            "temp": [10.0, None, 12.0],
        })
        result = pivot_forecasts(df)

        assert result["temp_step_1"].to_list() == [10.0]
        assert result["temp_step_2"].to_list() == [None]
        assert result["temp_step_3"].to_list() == [12.0]

    def test_empty_dataframe_returns_empty(self):
        """Empty DataFrame with correct schema returns empty pivoted result."""
        df = pl.DataFrame({
            "vintage_time": pl.Series([], dtype=pl.Datetime("us")),
            "time": pl.Series([], dtype=pl.Datetime("us")),
            "temp": pl.Series([], dtype=pl.Float64),
        })
        result = pivot_forecasts(df)

        assert result.is_empty()
        assert "time" in result.columns
        assert "temp" in result.columns

    def test_panel_prefixed_columns(self):
        """Columns with __ panel prefix produce prefixed step names."""
        df = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 2,
            "time": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "store_A__temp": [10.0, 11.0],
        })
        result = pivot_forecasts(df)

        assert "store_A__temp_step_1" in result.columns
        assert "store_A__temp_step_2" in result.columns
