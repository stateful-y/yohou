"""Tests for window_forecasts utility."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from yohou.utils.pivot import window_forecasts


def test_window_forecasts_exported_from_utils():
    """window_forecasts is re-exported from the yohou.utils package surface."""
    from yohou import utils

    assert "window_forecasts" in utils.__all__
    assert utils.window_forecasts is window_forecasts


class TestWindowForecasts:
    """Tests for the window_forecasts utility function."""

    def test_exact_vintage_match_backward_compat(self):
        """1:1 vintage alignment produces same result as exact match."""
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1, h) for h in range(3) for _ in range(2)],
            "time": [
                datetime(2020, 1, 1, 1),
                datetime(2020, 1, 1, 2),
                datetime(2020, 1, 1, 2),
                datetime(2020, 1, 1, 3),
                datetime(2020, 1, 1, 3),
                datetime(2020, 1, 1, 4),
            ],
            "temp": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        })
        obs = pl.Series([datetime(2020, 1, 1, h) for h in range(3)])
        result = window_forecasts(X_forecast, obs, forecasting_horizon=2, interval="1h")

        assert result.shape == (3, 3)
        assert result["temp_step_1"].to_list() == [10.0, 12.0, 14.0]
        assert result["temp_step_2"].to_list() == [11.0, 13.0, 15.0]

    def test_sparse_6h_vintages_with_hourly_observations(self):
        """6-hourly vintages with hourly observation times."""
        # Vintage at 00:00 covers hours 1..48, vintage at 06:00 covers hours 7..54
        X_forecast = pl.DataFrame({
            "vintage_time": ([datetime(2020, 1, 1, 0)] * 10 + [datetime(2020, 1, 1, 6)] * 10),
            "time": ([datetime(2020, 1, 1, h) for h in range(1, 11)] + [datetime(2020, 1, 1, h) for h in range(7, 17)]),
            "temp": list(range(20)),
        })
        obs = pl.Series([
            datetime(2020, 1, 1, 0),  # uses vintage 00:00
            datetime(2020, 1, 1, 3),  # uses vintage 00:00 (latest <= 03:00)
            datetime(2020, 1, 1, 6),  # uses vintage 06:00
            datetime(2020, 1, 1, 9),  # uses vintage 06:00
        ])
        result = window_forecasts(X_forecast, obs, forecasting_horizon=2, interval="1h")

        assert result.shape == (4, 3)
        # obs 00:00 → vintage 00:00 → step_1=value at 01:00, step_2=value at 02:00
        assert result["temp_step_1"][0] == 0.0  # hour 1 from vintage 00:00
        assert result["temp_step_2"][0] == 1.0  # hour 2 from vintage 00:00
        # obs 03:00 → vintage 00:00 → step_1=value at 04:00, step_2=value at 05:00
        assert result["temp_step_1"][1] == 3.0  # hour 4 from vintage 00:00
        assert result["temp_step_2"][1] == 4.0  # hour 5 from vintage 00:00
        # obs 06:00 → vintage 06:00 → step_1=value at 07:00, step_2=value at 08:00
        assert result["temp_step_1"][2] == 10.0  # hour 7 from vintage 06:00
        assert result["temp_step_2"][2] == 11.0  # hour 8 from vintage 06:00
        # obs 09:00 → vintage 06:00 → step_1=value at 10:00, step_2=value at 11:00
        assert result["temp_step_1"][3] == 13.0  # hour 10 from vintage 06:00
        assert result["temp_step_2"][3] == 14.0  # hour 11 from vintage 06:00

    def test_no_vintage_before_observation_produces_nulls(self):
        """Observation time before all vintages produces null steps."""
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1, 6)] * 2,
            "time": [datetime(2020, 1, 1, 7), datetime(2020, 1, 1, 8)],
            "temp": [10.0, 11.0],
        })
        obs = pl.Series([
            datetime(2020, 1, 1, 3),  # before vintage 06:00 → null
            datetime(2020, 1, 1, 6),  # at vintage 06:00 → has values
        ])
        result = window_forecasts(X_forecast, obs, forecasting_horizon=2, interval="1h")

        assert result.shape == (2, 3)
        assert result["temp_step_1"][0] is None
        assert result["temp_step_2"][0] is None
        assert result["temp_step_1"][1] == 10.0
        assert result["temp_step_2"][1] == 11.0

    def test_empty_x_forecast_produces_all_nulls(self):
        """Empty X_forecast produces all-null step columns."""
        X_forecast = pl.DataFrame({
            "vintage_time": pl.Series([], dtype=pl.Datetime("us")),
            "time": pl.Series([], dtype=pl.Datetime("us")),
            "temp": pl.Series([], dtype=pl.Float64),
        })
        obs = pl.Series([datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 1)])
        result = window_forecasts(X_forecast, obs, forecasting_horizon=2, interval="1h")

        assert result.shape == (2, 3)
        assert result["temp_step_1"][0] is None
        assert result["temp_step_1"][1] is None
        assert result["temp_step_2"][0] is None
        assert result["temp_step_2"][1] is None

    def test_partial_coverage_produces_partial_nulls(self):
        """Vintage with fewer steps than horizon produces nulls for missing steps."""
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1, 0)] * 2,
            "time": [datetime(2020, 1, 1, 1), datetime(2020, 1, 1, 2)],
            "temp": [10.0, 11.0],
        })
        obs = pl.Series([datetime(2020, 1, 1, 0)])
        result = window_forecasts(X_forecast, obs, forecasting_horizon=4, interval="1h")

        assert result.shape == (1, 5)
        assert result["temp_step_1"][0] == 10.0
        assert result["temp_step_2"][0] == 11.0
        assert result["temp_step_3"][0] is None
        assert result["temp_step_4"][0] is None

    def test_multiple_value_columns(self):
        """Multiple value columns produce separate step columns."""
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1, 0)] * 2,
            "time": [datetime(2020, 1, 1, 1), datetime(2020, 1, 1, 2)],
            "temp": [10.0, 11.0],
            "wind": [5.0, 6.0],
        })
        obs = pl.Series([datetime(2020, 1, 1, 0)])
        result = window_forecasts(X_forecast, obs, forecasting_horizon=2, interval="1h")

        assert result.shape == (1, 5)
        assert result["temp_step_1"][0] == 10.0
        assert result["wind_step_1"][0] == 5.0
        assert result["temp_step_2"][0] == 11.0
        assert result["wind_step_2"][0] == 6.0

    def test_timedelta_interval(self):
        """Accepts timedelta as interval."""
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1, 0)] * 2,
            "time": [datetime(2020, 1, 1, 1), datetime(2020, 1, 1, 2)],
            "temp": [10.0, 11.0],
        })
        obs = pl.Series([datetime(2020, 1, 1, 0)])
        result = window_forecasts(X_forecast, obs, forecasting_horizon=2, interval=timedelta(hours=1))

        assert result["temp_step_1"][0] == 10.0
        assert result["temp_step_2"][0] == 11.0

    def test_daily_interval(self):
        """Works with daily interval."""
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 3,
            "time": [datetime(2020, 1, 2), datetime(2020, 1, 3), datetime(2020, 1, 4)],
            "temp": [10.0, 11.0, 12.0],
        })
        obs = pl.Series([datetime(2020, 1, 1)])
        result = window_forecasts(X_forecast, obs, forecasting_horizon=3, interval="1d")

        assert result["temp_step_1"][0] == 10.0
        assert result["temp_step_2"][0] == 11.0
        assert result["temp_step_3"][0] == 12.0

    def test_observation_order_preserved(self):
        """Output row order matches observation_times order."""
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1, 0)] * 3,
            "time": [datetime(2020, 1, 1, h) for h in range(1, 4)],
            "temp": [10.0, 11.0, 12.0],
        })
        obs = pl.Series([datetime(2020, 1, 1, 2), datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 1)])
        result = window_forecasts(X_forecast, obs, forecasting_horizon=1, interval="1h")

        assert result["time"].to_list() == obs.to_list()

    def test_missing_vintage_col_raises(self):
        """Missing vintage_col raises ValueError."""
        X_forecast = pl.DataFrame({"time": [datetime(2020, 1, 1)], "temp": [10.0]})
        obs = pl.Series([datetime(2020, 1, 1)])
        with pytest.raises(ValueError, match="vintage_time"):
            window_forecasts(X_forecast, obs, forecasting_horizon=1, interval="1h")

    def test_missing_time_col_raises(self):
        """Missing time_col raises ValueError."""
        X_forecast = pl.DataFrame({"vintage_time": [datetime(2020, 1, 1)], "temp": [10.0]})
        obs = pl.Series([datetime(2020, 1, 1)])
        with pytest.raises(ValueError, match="time"):
            window_forecasts(X_forecast, obs, forecasting_horizon=1, interval="1h")

    def test_no_value_columns_raises(self):
        """No value columns raises ValueError."""
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)],
            "time": [datetime(2020, 1, 2)],
        })
        obs = pl.Series([datetime(2020, 1, 1)])
        with pytest.raises(ValueError, match="No value columns"):
            window_forecasts(X_forecast, obs, forecasting_horizon=1, interval="1h")

    def test_invalid_forecasting_horizon_raises(self):
        """Non-positive forecasting_horizon raises ValueError."""
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)],
            "time": [datetime(2020, 1, 2)],
            "temp": [10.0],
        })
        obs = pl.Series([datetime(2020, 1, 1)])
        with pytest.raises(ValueError, match="positive"):
            window_forecasts(X_forecast, obs, forecasting_horizon=0, interval="1h")

    def test_panel_prefixed_columns(self):
        """Panel-prefixed value columns produce prefixed step names."""
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1, 0)] * 2,
            "time": [datetime(2020, 1, 1, 1), datetime(2020, 1, 1, 2)],
            "store_A__temp": [10.0, 11.0],
        })
        obs = pl.Series([datetime(2020, 1, 1, 0)])
        result = window_forecasts(X_forecast, obs, forecasting_horizon=2, interval="1h")

        assert "store_A__temp_step_1" in result.columns
        assert "store_A__temp_step_2" in result.columns

    def test_all_obs_before_all_vintages_produces_all_nulls(self):
        """Non-empty X_forecast but all obs before all vintages returns all nulls."""
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1, 6)] * 2,
            "time": [datetime(2020, 1, 1, 7), datetime(2020, 1, 1, 8)],
            "temp": [10.0, 11.0],
        })
        obs = pl.Series([datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 3)])
        result = window_forecasts(X_forecast, obs, forecasting_horizon=2, interval="1h")

        assert result.shape == (2, 3)
        assert result["temp_step_1"][0] is None
        assert result["temp_step_1"][1] is None
        assert result["temp_step_2"][0] is None
        assert result["temp_step_2"][1] is None
