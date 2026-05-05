"""Tests for window_futures utility."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from yohou.utils.pivot import window_futures


class TestWindowFuture:
    """Tests for the window_futures utility function."""

    def test_basic_daily(self):
        """Window daily holidays with H=3."""
        holidays = pl.DataFrame({
            "time": [datetime(2020, 1, d) for d in range(1, 8)],
            "is_holiday": [1, 0, 0, 1, 0, 0, 0],
        })
        obs_times = pl.Series([datetime(2020, 1, 1), datetime(2020, 1, 2)])
        result = window_futures(holidays, obs_times, forecasting_horizon=3, interval="1d")

        assert result.columns == ["time", "is_holiday_step_1", "is_holiday_step_2", "is_holiday_step_3"]
        assert result.shape == (2, 4)
        # obs=Jan 1: steps Jan 2, Jan 3, Jan 4 → [0, 0, 1]
        assert result["is_holiday_step_1"].to_list() == [0, 0]
        assert result["is_holiday_step_2"].to_list() == [0, 1]
        assert result["is_holiday_step_3"].to_list() == [1, 0]

    def test_hourly_interval(self):
        """Window with hourly interval."""
        df = pl.DataFrame({
            "time": [datetime(2020, 1, 1, h) for h in range(6)],
            "val": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        })
        obs_times = pl.Series([datetime(2020, 1, 1, 0)])
        result = window_futures(df, obs_times, forecasting_horizon=2, interval="1h")

        assert result["val_step_1"].to_list() == [11.0]
        assert result["val_step_2"].to_list() == [12.0]

    def test_timedelta_interval(self):
        """Accepts timedelta as interval."""
        df = pl.DataFrame({
            "time": [datetime(2020, 1, 1, h) for h in range(6)],
            "val": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        })
        obs_times = pl.Series([datetime(2020, 1, 1, 0)])
        result = window_futures(df, obs_times, forecasting_horizon=2, interval=timedelta(hours=1))

        assert result["val_step_1"].to_list() == [11.0]
        assert result["val_step_2"].to_list() == [12.0]

    def test_multiple_value_columns(self):
        """Window with two value columns."""
        df = pl.DataFrame({
            "time": [datetime(2020, 1, d) for d in range(1, 5)],
            "holiday": [1, 0, 0, 1],
            "day_type": [0, 1, 1, 0],
        })
        obs_times = pl.Series([datetime(2020, 1, 1)])
        result = window_futures(df, obs_times, forecasting_horizon=2, interval="1d")

        assert "holiday_step_1" in result.columns
        assert "holiday_step_2" in result.columns
        assert "day_type_step_1" in result.columns
        assert "day_type_step_2" in result.columns
        assert result["holiday_step_1"].to_list() == [0]
        assert result["day_type_step_1"].to_list() == [1]

    def test_partial_coverage_produces_nulls(self):
        """Steps beyond df coverage produce nulls."""
        df = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "val": [10.0, 20.0],
        })
        obs_times = pl.Series([datetime(2020, 1, 1)])
        result = window_futures(df, obs_times, forecasting_horizon=3, interval="1d")

        assert result["val_step_1"].to_list() == [20.0]
        assert result["val_step_2"].to_list() == [None]
        assert result["val_step_3"].to_list() == [None]

    def test_output_time_is_observation_time(self):
        """Output 'time' column contains observation times."""
        df = pl.DataFrame({
            "time": [datetime(2020, 1, d) for d in range(1, 6)],
            "val": list(range(5)),
        })
        obs_times = pl.Series([datetime(2020, 1, 1), datetime(2020, 1, 3)])
        result = window_futures(df, obs_times, forecasting_horizon=2, interval="1d")

        assert result["time"].to_list() == [datetime(2020, 1, 1), datetime(2020, 1, 3)]

    def test_missing_time_col_raises(self):
        """Missing time column raises ValueError."""
        df = pl.DataFrame({"val": [1.0]})
        obs = pl.Series([datetime(2020, 1, 1)])
        with pytest.raises(ValueError, match="'time' not found"):
            window_futures(df, obs, forecasting_horizon=1, interval="1d")

    def test_zero_horizon_raises(self):
        """Zero forecasting_horizon raises ValueError."""
        df = pl.DataFrame({"time": [datetime(2020, 1, 1)], "val": [1.0]})
        obs = pl.Series([datetime(2020, 1, 1)])
        with pytest.raises(ValueError, match="forecasting_horizon must be positive"):
            window_futures(df, obs, forecasting_horizon=0, interval="1d")

    def test_negative_horizon_raises(self):
        """Negative forecasting_horizon raises ValueError."""
        df = pl.DataFrame({"time": [datetime(2020, 1, 1)], "val": [1.0]})
        obs = pl.Series([datetime(2020, 1, 1)])
        with pytest.raises(ValueError, match="forecasting_horizon must be positive"):
            window_futures(df, obs, forecasting_horizon=-2, interval="1d")

    def test_no_value_columns_raises(self):
        """Only time column, no values, raises ValueError."""
        df = pl.DataFrame({"time": [datetime(2020, 1, 1)]})
        obs = pl.Series([datetime(2020, 1, 1)])
        with pytest.raises(ValueError, match="No value columns found"):
            window_futures(df, obs, forecasting_horizon=1, interval="1d")

    def test_empty_observation_times(self):
        """Empty observation_times produces empty result with correct schema."""
        df = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "val": [10.0, 20.0],
        })
        obs = pl.Series("obs", [], dtype=pl.Datetime)
        result = window_futures(df, obs, forecasting_horizon=2, interval="1d")

        assert result.shape[0] == 0
        assert "val_step_1" in result.columns
        assert "val_step_2" in result.columns

    def test_step_naming(self):
        """Step columns follow <col>_step_<n> pattern (1-based)."""
        df = pl.DataFrame({
            "time": [datetime(2020, 1, d) for d in range(1, 6)],
            "x": list(range(5)),
        })
        obs = pl.Series([datetime(2020, 1, 1)])
        result = window_futures(df, obs, forecasting_horizon=4, interval="1d")

        step_cols = [c for c in result.columns if c != "time"]
        assert step_cols == ["x_step_1", "x_step_2", "x_step_3", "x_step_4"]

    def test_custom_time_col(self):
        """Custom time_col parameter works."""
        df = pl.DataFrame({
            "target_date": [datetime(2020, 1, d) for d in range(1, 5)],
            "val": [1, 2, 3, 4],
        })
        obs = pl.Series([datetime(2020, 1, 1)])
        result = window_futures(df, obs, forecasting_horizon=2, interval="1d", time_col="target_date")

        assert "time" in result.columns
        assert result["val_step_1"].to_list() == [2]
        assert result["val_step_2"].to_list() == [3]

    def test_panel_prefixed_columns(self):
        """Columns with __ panel prefix get prefixed step names."""
        df = pl.DataFrame({
            "time": [datetime(2020, 1, d) for d in range(1, 5)],
            "store_A__temp": [10.0, 11.0, 12.0, 13.0],
        })
        obs = pl.Series([datetime(2020, 1, 1)])
        result = window_futures(df, obs, forecasting_horizon=2, interval="1d")

        assert "store_A__temp_step_1" in result.columns
        assert "store_A__temp_step_2" in result.columns

    def test_monthly_interval(self):
        """Monthly interval using add_interval."""
        df = pl.DataFrame({
            "time": [datetime(2020, m, 1) for m in range(1, 7)],
            "val": list(range(6)),
        })
        obs = pl.Series([datetime(2020, 1, 1)])
        result = window_futures(df, obs, forecasting_horizon=3, interval="1mo")

        assert result["val_step_1"].to_list() == [1]  # Feb
        assert result["val_step_2"].to_list() == [2]  # Mar
        assert result["val_step_3"].to_list() == [3]  # Apr
