"""Tests for per-channel X_forecast vintage retention in observe and rewind.

Observe and rewind keep, per base column, the newest vintage still covering the
observation point, rather than collapsing the frame to a single vintage. That is
what lets channels issued on different schedules survive into the cache the
fallback path reads. These tests inspect the retained cache directly; the
end-to-end consequence (a slower channel still contributing step features at
predict) is exercised once densification lands, in the resolution tests.
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
from sklearn.dummy import DummyRegressor

from yohou.point import PointReductionForecaster

H = 3


def _d(day: int) -> datetime:
    return datetime(2021, 1, 1) + timedelta(days=day)


def _y(days: range) -> pl.DataFrame:
    return pl.DataFrame({
        "time": [_d(k) for k in days],
        "value": np.asarray([float(k) for k in days]),
    })


def _forecast(spec: dict[int, list[str]], columns: tuple[str, ...] = ("fast", "slow")) -> pl.DataFrame:
    """Build a forecast frame from ``{vintage_day: [columns issued at it]}``.

    Every listed vintage carries ``H`` contiguous daily steps for each of its
    issued columns; the other columns are null at that vintage.
    """
    rows = []
    for vintage_day, issued in spec.items():
        for step in range(1, H + 1):
            row = {"vintage_time": _d(vintage_day), "time": _d(vintage_day + step)}
            for col in columns:
                row[col] = float(vintage_day + step) if col in issued else None
            rows.append(row)
    schema = {"vintage_time": pl.Datetime, "time": pl.Datetime}
    schema.update(dict.fromkeys(columns, pl.Float64))
    return pl.DataFrame(rows, schema=schema)


def _fitted() -> PointReductionForecaster:
    """A forecaster fit with both channels present, so both step columns exist."""
    forecaster = PointReductionForecaster(estimator=DummyRegressor(), nan_handling="pass")
    fit_forecast = _forecast({v: ["fast", "slow"] for v in range(8)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster.fit(y=_y(range(8)), forecasting_horizon=H, X_forecast=fit_forecast)
    return forecaster


def _vintages(frame: pl.DataFrame) -> set:
    return set(frame["vintage_time"].to_list())


class TestPerChannelRetention:
    """A vintage carrying only the fast channel does not evict the slow one."""

    def test_slower_channel_survives_observe(self):
        """``slow`` is issued at an older vintage that still covers the point."""
        forecaster = _fitted()
        # fast issued daily through day 10; slow only at day 8, still covering
        # days 9..11, so it is live at the observation point (day 10).
        mixed = _forecast({8: ["fast", "slow"], 9: ["fast"], 10: ["fast"]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.observe(y=_y(range(8, 11)), X_forecast=mixed)

        raw = forecaster._X_forecast_raw_
        # Both channels retained: fast from its newest live vintage (day 10),
        # slow from its own newest live vintage (day 8).
        assert _vintages(raw) == {_d(8), _d(10)}
        assert raw.filter(pl.col("vintage_time") == _d(8))["slow"].drop_nulls().len() > 0
        assert raw.filter(pl.col("vintage_time") == _d(10))["fast"].drop_nulls().len() > 0
        # The old single-vintage rule would have kept only day 10 and dropped slow.
        assert raw.filter(pl.col("slow").is_not_null()).height > 0

    def test_stale_channel_is_evicted(self):
        """A channel whose only vintage no longer reaches the point is dropped."""
        forecaster = _fitted()
        # slow at day 6 covers days 7..9; at observation day 10 it reaches nothing
        # past the point (max target day 9 <= day 10), so it is evicted.
        mixed = _forecast({6: ["slow"], 8: ["fast"], 9: ["fast"], 10: ["fast"]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.observe(y=_y(range(8, 11)), X_forecast=mixed)

        raw = forecaster._X_forecast_raw_
        assert _d(6) not in _vintages(raw), "a stale vintage covering nothing must be evicted"
        assert raw.filter(pl.col("slow").is_not_null()).height == 0

    def test_retained_state_is_bounded_by_channel_count(self):
        """Many vintages per interval retain at most one vintage per channel."""
        forecaster = _fitted()
        # A dense burst of fast vintages plus one slow vintage.
        spec = {v: ["fast"] for v in range(8, 11)}
        spec[8] = ["fast", "slow"]
        mixed = _forecast(spec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.observe(y=_y(range(8, 11)), X_forecast=mixed)

        # Two channels -> at most two retained vintages, no matter how many arrived.
        assert len(_vintages(forecaster._X_forecast_raw_)) <= 2


class TestRewindRetention:
    """Rewind restores retention consistent with its target time."""

    def test_rewind_drops_vintages_newer_than_target(self):
        """After rewinding, no retained vintage is newer than the rewind point."""
        forecaster = _fitted()
        mixed = _forecast({8: ["fast", "slow"], 9: ["fast"], 10: ["fast"]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecaster.observe(y=_y(range(8, 11)), X_forecast=mixed)
            # Rewind to day 8 with the same frame available.
            forecaster.rewind(y=_y(range(8, 9)), X_forecast=mixed)

        raw = forecaster._X_forecast_raw_
        assert all(v <= _d(8) for v in _vintages(raw)), "rewind must not retain a vintage past its target"
