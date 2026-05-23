"""Tests for _derive_step_columns utility."""

from datetime import datetime

import polars as pl
import pytest

from yohou.base.utils import _derive_step_columns


class TestDeriveStepColumns:
    """Tests for the _derive_step_columns utility function."""

    # --- None inputs ---

    def test_both_none_returns_none(self):
        """Both X_future and X_forecast None → returns None."""
        obs = pl.Series([datetime(2020, 1, 1)])
        result = _derive_step_columns(None, None, obs, 3, "1d")
        assert result is None

    # --- X_future only ---

    def test_x_future_only(self):
        """X_future only → windowed step columns."""
        X_future = pl.DataFrame({
            "time": [datetime(2020, 1, d) for d in range(1, 6)],
            "is_holiday": [1, 0, 0, 1, 0],
        })
        obs = pl.Series([datetime(2020, 1, 1), datetime(2020, 1, 2)])
        result = _derive_step_columns(X_future, None, obs, 2, "1d")

        assert result is not None
        assert "time" in result.columns
        assert "is_holiday_step_1" in result.columns
        assert "is_holiday_step_2" in result.columns
        assert result.shape == (2, 3)

    # --- X_forecast only ---

    def test_x_forecast_only(self):
        """X_forecast only → pivoted step columns."""
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 2 + [datetime(2020, 1, 2)] * 2,
            "time": [datetime(2020, 1, 2), datetime(2020, 1, 3)] * 2,
            "temp": [10.0, 11.0, 15.0, 16.0],
        })
        obs = pl.Series([datetime(2020, 1, 1), datetime(2020, 1, 2)])
        result = _derive_step_columns(None, X_forecast, obs, 2, "1d")

        assert result is not None
        assert "temp_step_1" in result.columns
        assert "temp_step_2" in result.columns
        assert result["time"].to_list() == [datetime(2020, 1, 1), datetime(2020, 1, 2)]

    # --- Both X_future and X_forecast ---

    def test_both_sources(self):
        """Both X_future and X_forecast → combined step columns."""
        X_future = pl.DataFrame({
            "time": [datetime(2020, 1, d) for d in range(1, 5)],
            "is_holiday": [1, 0, 0, 1],
        })
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 2,
            "time": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "temp": [10.0, 11.0],
        })
        obs = pl.Series([datetime(2020, 1, 1)])
        result = _derive_step_columns(X_future, X_forecast, obs, 2, "1d")

        assert result is not None
        assert "is_holiday_step_1" in result.columns
        assert "temp_step_1" in result.columns
        assert result.shape == (1, 5)  # time + 2 holiday steps + 2 temp steps

    # --- Collision detection ---

    def test_collision_with_existing_columns(self):
        """Step column name collides with existing_columns → ValueError."""
        X_future = pl.DataFrame({
            "time": [datetime(2020, 1, d) for d in range(1, 5)],
            "temp": [10.0, 11.0, 12.0, 13.0],
        })
        obs = pl.Series([datetime(2020, 1, 1)])
        with pytest.raises(ValueError, match="collide with existing columns"):
            _derive_step_columns(
                X_future,
                None,
                obs,
                2,
                "1d",
                existing_columns={"temp_step_1"},
            )

    def test_collision_between_sources(self):
        """Same column name from X_future and X_forecast → ValueError."""
        X_future = pl.DataFrame({
            "time": [datetime(2020, 1, d) for d in range(1, 5)],
            "temp": [10.0, 11.0, 12.0, 13.0],
        })
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 2,
            "time": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "temp": [20.0, 21.0],
        })
        obs = pl.Series([datetime(2020, 1, 1)])
        with pytest.raises(ValueError, match="collision between X_future and X_forecast"):
            _derive_step_columns(X_future, X_forecast, obs, 2, "1d")

    def test_no_collision_different_columns(self):
        """Different column names across sources → no error."""
        X_future = pl.DataFrame({
            "time": [datetime(2020, 1, d) for d in range(1, 5)],
            "holiday": [1, 0, 0, 1],
        })
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 2,
            "time": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "temp": [10.0, 11.0],
        })
        obs = pl.Series([datetime(2020, 1, 1)])
        result = _derive_step_columns(
            X_future,
            X_forecast,
            obs,
            2,
            "1d",
            existing_columns={"sensor_a", "sensor_b"},
        )
        assert result is not None

    # --- Partial coverage / nulls ---

    def test_x_forecast_partial_coverage(self):
        """X_forecast missing some observation times → null step columns."""
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 2,
            "time": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "temp": [10.0, 11.0],
        })
        obs = pl.Series([datetime(2020, 1, 1), datetime(2020, 1, 5)])
        result = _derive_step_columns(None, X_forecast, obs, 2, "1d")

        assert result.shape == (2, 3)
        assert result["temp_step_1"][0] == 10.0
        assert result["temp_step_1"][1] is None  # Jan 5 not in X_forecast

    # --- Panel data ---

    def test_panel_prefixed_columns(self):
        """Columns with __ panel prefix produce prefixed step names."""
        X_future = pl.DataFrame({
            "time": [datetime(2020, 1, d) for d in range(1, 5)],
            "store_A__temp": [10.0, 11.0, 12.0, 13.0],
        })
        obs = pl.Series([datetime(2020, 1, 1)])
        result = _derive_step_columns(X_future, None, obs, 2, "1d")

        assert "store_A__temp_step_1" in result.columns
        assert "store_A__temp_step_2" in result.columns

    # --- Step column names extraction ---

    def test_step_column_names_extractable(self):
        """Step column names can be extracted by excluding 'time'."""
        X_future = pl.DataFrame({
            "time": [datetime(2020, 1, d) for d in range(1, 5)],
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7, 8],
        })
        obs = pl.Series([datetime(2020, 1, 1)])
        result = _derive_step_columns(X_future, None, obs, 2, "1d")

        step_names = {c for c in result.columns if c != "time"}
        assert step_names == {"a_step_1", "a_step_2", "b_step_1", "b_step_2"}

    # --- Single observation time ---

    def test_single_observation_time(self):
        """Single observation time produces one row."""
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 3,
            "time": [datetime(2020, 1, 2), datetime(2020, 1, 3), datetime(2020, 1, 4)],
            "wind": [5.0, 6.0, 7.0],
        })
        obs = pl.Series([datetime(2020, 1, 1)])
        result = _derive_step_columns(None, X_forecast, obs, 3, "1d")

        assert result.shape == (1, 4)
        assert result["wind_step_1"].to_list() == [5.0]
        assert result["wind_step_3"].to_list() == [7.0]

    # --- Forecast horizon filtering ---

    def test_forecast_timestamps_beyond_horizon_are_filtered(self):
        """Vintage with more timestamps than fh keeps only those within the window."""
        # 5 hourly forecasts but fh=3: only the first 3 hours after obs should survive
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 5,
            "time": [datetime(2020, 1, 1, h) for h in range(1, 6)],
            "temp": [10.0, 11.0, 12.0, 13.0, 14.0],
        })
        obs = pl.Series([datetime(2020, 1, 1)])
        result = _derive_step_columns(None, X_forecast, obs, 3, "1h")

        assert result is not None
        step_cols = [c for c in result.columns if c != "time"]
        assert step_cols == ["temp_step_1", "temp_step_2", "temp_step_3"]
        assert result["temp_step_1"].to_list() == [10.0]
        assert result["temp_step_3"].to_list() == [12.0]

    def test_forecast_exactly_h_timestamps_unchanged(self):
        """Vintage with exactly H timestamps within the window passes through."""
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 3,
            "time": [datetime(2020, 1, 1, h) for h in range(1, 4)],
            "temp": [10.0, 11.0, 12.0],
        })
        obs = pl.Series([datetime(2020, 1, 1)])
        result = _derive_step_columns(None, X_forecast, obs, 3, "1h")

        assert result is not None
        step_cols = [c for c in result.columns if c != "time"]
        assert step_cols == ["temp_step_1", "temp_step_2", "temp_step_3"]
        assert result["temp_step_1"].to_list() == [10.0]
        assert result["temp_step_3"].to_list() == [12.0]

    def test_short_vintage_emits_warning(self):
        """Vintage with fewer timestamps than fh emits UserWarning."""
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 2,
            "time": [datetime(2020, 1, 1, 1), datetime(2020, 1, 1, 2)],
            "temp": [10.0, 11.0],
        })
        obs = pl.Series([datetime(2020, 1, 1)])

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _derive_step_columns(None, X_forecast, obs, 5, "1h")

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        msg = str(user_warnings[0].message)
        assert "2" in msg
        assert "5" in msg
        assert result is not None

    def test_no_warning_when_steps_cover_horizon(self):
        """No warning emitted when vintages have >= H timestamps."""
        X_forecast = pl.DataFrame({
            "vintage_time": [datetime(2020, 1, 1)] * 3,
            "time": [datetime(2020, 1, 1, h) for h in range(1, 4)],
            "temp": [10.0, 11.0, 12.0],
        })
        obs = pl.Series([datetime(2020, 1, 1)])

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _derive_step_columns(None, X_forecast, obs, 3, "1h")

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0
