"""Tests for as-of vintage selection in observe() / _inject_step_columns_after_update."""

from datetime import datetime, timedelta

import polars as pl
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer


FREQ = "1h"
H = 3
N_TRAIN = 50
SEED = 42


def _make_y(times: pl.Series) -> pl.DataFrame:
    return pl.DataFrame({"time": times, "y": [float(i) for i in range(len(times))]})


def _make_x_forecast(
    vintage_times: list[datetime],
    forecast_start_offsets: list[int],
    n_steps: int,
) -> pl.DataFrame:
    """Build X_forecast with given vintage times and forecast target times.

    Each vintage covers n_steps hourly timestamps starting from the vintage time
    plus the corresponding offset.
    """
    rows = []
    for vt, offset in zip(vintage_times, forecast_start_offsets):
        for s in range(n_steps):
            rows.append({
                "vintage_time": vt,
                "time": vt + timedelta(hours=offset + s),
                "temp": float(100 * vt.hour + s),
            })
    return pl.DataFrame(rows)


def _make_x_forecast_aligned(times: pl.Series, h: int) -> pl.DataFrame:
    """Build X_forecast with one vintage per observation time, each covering h steps."""
    rows = []
    for i in range(len(times)):
        vt = times[i]
        for s in range(1, h + 1):
            rows.append({
                "vintage_time": vt,
                "time": vt + timedelta(hours=s),
                "temp": float(100 * i + s),
            })
    return pl.DataFrame(rows)


def _fit_forecaster() -> PointReductionForecaster:
    """Fit a minimal forecaster with X_forecast support."""
    train_times = pl.Series("time", [datetime(2020, 1, 1) + timedelta(hours=i) for i in range(N_TRAIN)])
    y_train = _make_y(train_times)
    x_fc = _make_x_forecast_aligned(train_times, H)

    f = PointReductionForecaster(
        estimator=HistGradientBoostingRegressor(max_iter=10, random_state=SEED),
        feature_transformer=LagTransformer([1, 2]),
        reduction_strategy="direct",
    )
    f.fit(y_train, forecasting_horizon=H, X_forecast=x_fc)
    return f


class TestObserveAsofVintageSelection:
    """Verify _X_forecast_raw_ stores the as-of vintage (latest V <= obs_time)."""

    def test_exact_match_vintage(self):
        """When vintage_time == observed_time_, exact vintage is selected."""
        f = _fit_forecaster()
        # Observe the next hour after training ends (hour 49 is last train)
        obs_time = datetime(2020, 1, 1) + timedelta(hours=N_TRAIN)
        y_obs = pl.DataFrame({"time": [obs_time], "y": [999.0]})
        x_fc = _make_x_forecast([obs_time], [1], H)

        f.observe(y_obs, X_forecast=x_fc)

        assert f._X_forecast_raw_ is not None
        assert len(f._X_forecast_raw_) == H
        assert f._X_forecast_raw_["vintage_time"][0] == obs_time

    def test_earlier_vintage_selected(self):
        """When no exact match, latest vintage before obs_time is selected."""
        f = _fit_forecaster()
        obs_time = datetime(2020, 1, 1) + timedelta(hours=N_TRAIN)
        # Vintage 2 hours earlier than observation
        vintage = obs_time - timedelta(hours=2)
        x_fc = _make_x_forecast([vintage], [1], H)

        f.observe(
            pl.DataFrame({"time": [obs_time], "y": [999.0]}),
            X_forecast=x_fc,
        )

        assert f._X_forecast_raw_ is not None
        assert len(f._X_forecast_raw_) == H
        assert f._X_forecast_raw_["vintage_time"][0] == vintage

    def test_multiple_vintages_picks_latest_before_obs(self):
        """With multiple vintages, the latest one at or before obs_time is used."""
        f = _fit_forecaster()
        obs_time = datetime(2020, 1, 1) + timedelta(hours=N_TRAIN)
        v1 = obs_time - timedelta(hours=6)
        v2 = obs_time - timedelta(hours=1)
        v3 = obs_time + timedelta(hours=6)  # future, should not be selected
        x_fc = _make_x_forecast([v1, v2, v3], [1, 1, 1], H)

        f.observe(
            pl.DataFrame({"time": [obs_time], "y": [999.0]}),
            X_forecast=x_fc,
        )

        assert f._X_forecast_raw_ is not None
        assert f._X_forecast_raw_["vintage_time"][0] == v2

    def test_no_vintage_before_obs_produces_empty(self):
        """When all vintages are after obs_time, _X_forecast_raw_ is empty."""
        f = _fit_forecaster()
        obs_time = datetime(2020, 1, 1) + timedelta(hours=N_TRAIN)
        future_vintage = obs_time + timedelta(hours=6)
        x_fc = _make_x_forecast([future_vintage], [1], H)

        f.observe(
            pl.DataFrame({"time": [obs_time], "y": [999.0]}),
            X_forecast=x_fc,
        )

        assert f._X_forecast_raw_ is not None
        assert len(f._X_forecast_raw_) == 0
