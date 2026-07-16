"""Synthetic dataset generators for exogenous feature examples.

Generate deterministic, parameterized time series with all three exogenous
feature types (``X_actual``, ``X_future``, ``X_forecast``) for use in
examples and tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
from sklearn.utils import Bunch

_HOLIDAY_MONTH_DAYS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 6),
    (1, 8),
    (1, 15),
    (2, 19),
    (3, 29),
    (4, 1),
    (5, 1),
    (5, 27),
    (7, 4),
    (9, 2),
    (11, 28),
    (12, 25),
    (12, 26),
)
"""Fixed civil holiday calendar as ``(month, day)`` pairs.

The dates are deliberately irregular with respect to weekday, so the resulting
indicator cannot be reconstructed from any cyclical encoding of the clock. That
is what makes it a genuine ``X_future`` feature: a forecaster must be told the
holiday status of each future step, because no transformer can derive it from
the observation point. A weekday predicate would instead be a clock feature and
would belong in a ``feature_transformer``.

The early-January entries are load-bearing. The default series is 200 hourly
rows, spanning only 2024-01-01 to 2024-01-09, so a calendar carrying nothing but
widely spaced real holidays would leave the tail of that window (the usual test
split) with no holiday at all, and the feature would never move an evaluated
prediction.
"""


def _holiday_indicator(times: pl.Series) -> np.ndarray:
    """Mark timestamps whose date appears in the fixed holiday calendar.

    Depends only on the calendar, never on ``random_state``, so the holidays are
    a property of the dataset rather than of the seed.
    """
    # dt.month() and dt.day() are Int8, so the 100 * month term overflows for
    # every month past January unless the operands are widened first.
    month = times.dt.month().cast(pl.Int32)
    day = times.dt.day().cast(pl.Int32)
    keys = [100 * m + d for m, d in _HOLIDAY_MONTH_DAYS]
    return (100 * month + day).is_in(keys).cast(pl.Float64).to_numpy()


def _forecast_index_grid(n_samples: int, forecasting_horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Build vintage and target index arrays for X_forecast construction.

    Reproduces, in vectorized form, the row order of the nested loop over
    vintage indices ``i in range(forecasting_horizon, n_samples)`` and steps
    ``step in range(1, forecasting_horizon + 1)``, keeping only pairs whose
    target index ``i + step`` stays within the sample range. The row order is
    preserved so downstream RNG draws stay numerically identical.
    """
    vintage = np.arange(forecasting_horizon, n_samples)
    steps = np.arange(1, forecasting_horizon + 1)
    vintage_grid = np.repeat(vintage, forecasting_horizon)
    target_grid = vintage_grid + np.tile(steps, vintage.shape[0])
    keep = target_grid < n_samples
    return vintage_grid[keep], target_grid[keep]


def make_exogenous_regression(
    *,
    n_samples: int = 200,
    forecasting_horizon: int = 6,
    noise: float = 0.1,
    forecast_bias: float = 0.5,
    random_state: int = 42,
) -> Bunch:
    """Generate a synthetic regression dataset with exogenous features.

    Creates hourly electricity prices driven by temperature and a holiday
    indicator with a known linear relationship:
    ``price = 50 + 2 * temperature + 10 * is_holiday + noise``.

    Three exogenous feature types are produced:

    - **X_actual** (observation features): realized temperature readings
      with a 24 hour sinusoidal cycle plus measurement noise (std=0.5).
    - **X_future** (known future): an ``is_holiday`` indicator drawn from a
      fixed civil holiday calendar, covering the full time range. The dates
      are irregular with respect to weekday, so the indicator cannot be
      derived from the timestamp and genuinely requires the ``X_future``
      channel.
    - **X_forecast** (external forecasts): weather temperature forecasts
      with one vintage per observation from index ``forecasting_horizon``
      up to (but not including) the last observation, each covering the
      next ``forecasting_horizon`` steps that remain within the sample.
      The final observation produces no vintage because all of its
      forecast steps fall outside the sample range. Forecasts carry a
      small systematic bias relative to actuals.

    Parameters
    ----------
    n_samples : int, default=200
        Number of hourly observations.
    forecasting_horizon : int, default=6
        Number of forward steps per X_forecast vintage.
    noise : float, default=0.1
        Standard deviation of the target noise term.
    forecast_bias : float, default=0.5
        Systematic bias added to weather forecasts relative to actuals.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    Bunch
        Dictionary-like object with the following attributes:

        y : pl.DataFrame
            Target with columns ``["time", "price"]``.
        X_actual : pl.DataFrame
            Observation features with columns ``["time", "temperature"]``
            (24 hour sinusoidal cycle plus measurement noise, std=0.5).
        X_future : pl.DataFrame
            Known future features with columns ``["time", "is_holiday"]``.
        X_forecast : pl.DataFrame
            External forecasts with columns
            ``["vintage_time", "time", "wx_temp"]``. One vintage per
            observation from index ``forecasting_horizon`` up to (but not
            including) the last observation; the final observation
            produces no vintage.
        frame : pl.DataFrame
            ``y``, ``X_actual``, and ``X_future`` joined on ``"time"``.
            ``X_forecast`` is excluded because it has a different schema.
        feature_names : list of str
            ``["temperature", "is_holiday", "wx_temp"]``.
        target_names : list of str
            ``["price"]``.
        frequency : str
            ``"1h"``.
        DESCR : str
            Human readable description.

    See Also
    --------
    - [`make_exogenous_classification`][yohou.datasets._generators.make_exogenous_classification] : Classification variant with categorical target.
    - [`fetch_tourism_monthly`][yohou.datasets._fetchers.fetch_tourism_monthly] : Real monthly tourism dataset (univariate).

    Examples
    --------
    >>> from yohou.datasets import make_exogenous_regression
    >>> data = make_exogenous_regression(n_samples=100)
    >>> data.y.columns
    ['time', 'price']
    >>> data.X_forecast.columns
    ['vintage_time', 'time', 'wx_temp']

    """
    rng = np.random.default_rng(random_state)
    times = pl.datetime_range(
        datetime(2024, 1, 1),
        datetime(2024, 1, 1) + timedelta(hours=n_samples - 1),
        interval="1h",
        eager=True,
    ).alias("time")
    t = np.arange(n_samples, dtype=float)

    actual_temp = 15.0 + 5.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n_samples)
    holidays = _holiday_indicator(times)
    price = 50.0 + 2.0 * actual_temp + 10.0 * holidays + rng.normal(0, noise, n_samples)

    y = pl.DataFrame({"time": times, "price": price})
    X_actual = pl.DataFrame({"time": times, "temperature": actual_temp})
    X_future = pl.DataFrame({"time": times, "is_holiday": holidays})

    vintage_idx, target_idx = _forecast_index_grid(n_samples, forecasting_horizon)
    wx_noise = rng.normal(0, 0.3, vintage_idx.shape[0])
    X_forecast = pl.DataFrame(
        {
            "vintage_time": times.gather(vintage_idx),
            "time": times.gather(target_idx),
            "wx_temp": actual_temp[target_idx] + forecast_bias + wx_noise,
        },
        schema={"vintage_time": times.dtype, "time": times.dtype, "wx_temp": pl.Float64},
    )

    frame = y.join(X_actual, on="time").join(X_future, on="time")

    return Bunch(
        y=y,
        X_actual=X_actual,
        X_future=X_future,
        X_forecast=X_forecast,
        frame=frame,
        feature_names=["temperature", "is_holiday", "wx_temp"],
        target_names=["price"],
        frequency="1h",
        DESCR=(
            "Synthetic hourly electricity prices with exogenous features.\n"
            "Target: price = 50 + 2 * temperature + 10 * is_holiday + noise.\n"
            "X_actual: realized temperature (sinusoidal 24h cycle + noise).\n"
            "X_future: is_holiday indicator from a fixed civil holiday calendar\n"
            "  (irregular dates, not derivable from the timestamp).\n"
            "X_forecast: weather temperature forecasts with systematic bias."
        ),
    )


def make_exogenous_classification(
    *,
    n_samples: int = 400,
    forecasting_horizon: int = 6,
    noise: float = 0.1,
    forecast_bias: float = 0.3,
    random_state: int = 42,
) -> Bunch:
    """Generate a synthetic classification dataset with exogenous features.

    Creates hourly air quality readings classified into three categories
    based on pollutant concentration thresholds.

    Three exogenous feature types are produced:

    - **X_actual** (observation features): realized pollutant readings
      with a 24 hour sinusoidal cycle.
    - **X_future** (known future): an ``is_holiday`` indicator drawn from a
      fixed civil holiday calendar, covering the full time range. Holidays
      reduce traffic and therefore pollutant levels. The dates are irregular
      with respect to weekday, so the indicator cannot be derived from the
      timestamp and genuinely requires the ``X_future`` channel.
    - **X_forecast** (external forecasts): pollutant concentration
      forecasts with one vintage per observation from index
      ``forecasting_horizon`` up to (but not including) the last
      observation, each covering the next ``forecasting_horizon`` steps
      that remain within the sample. The final observation produces no
      vintage because all of its forecast steps fall outside the sample
      range.

    Classification thresholds are applied to the *effective* pollutant signal
    (the stored ``X_actual`` ``pollutant`` minus a 5-unit weekend offset, plus
    jitter), not to the raw ``X_actual["pollutant"]`` feature. As a result a
    small fraction of rows near the 40/60 boundaries do not satisfy these
    thresholds when read off the stored feature:

    - ``"good"``: effective pollutant < 40
    - ``"moderate"``: 40 <= effective pollutant < 60
    - ``"poor"``: effective pollutant >= 60

    Parameters
    ----------
    n_samples : int, default=400
        Number of hourly observations.
    forecasting_horizon : int, default=6
        Number of forward steps per X_forecast vintage.
    noise : float, default=0.1
        Standard deviation of an additional jitter layered on top of the
        pollutant signal just before classification. The underlying pollutant
        signal already carries a fixed ``std=5.0`` noise term, so at the
        default ``noise=0.1`` this jitter only marginally perturbs the
        decision boundary; raise it to inject more label noise.
    forecast_bias : float, default=0.3
        Systematic bias added to pollutant forecasts relative to actuals.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    Bunch
        Dictionary-like object with the following attributes:

        y : pl.DataFrame
            Target with columns ``["time", "air_quality"]``. Values are
            one of ``"good"``, ``"moderate"``, ``"poor"``.
        X_actual : pl.DataFrame
            Observation features with columns ``["time", "pollutant"]``.
        X_future : pl.DataFrame
            Known future features with columns ``["time", "is_holiday"]``.
        X_forecast : pl.DataFrame
            External forecasts with columns
            ``["vintage_time", "time", "pollutant_forecast"]``.
        frame : pl.DataFrame
            ``y``, ``X_actual``, and ``X_future`` joined on ``"time"``.
        feature_names : list of str
            ``["pollutant", "is_holiday", "pollutant_forecast"]``.
        target_names : list of str
            ``["air_quality"]``.
        classes : list of str
            ``["good", "moderate", "poor"]``.
        frequency : str
            ``"1h"``.
        DESCR : str
            Human readable description.

    See Also
    --------
    - [`make_exogenous_regression`][yohou.datasets._generators.make_exogenous_regression] : Regression variant with continuous target.
    - [`fetch_air_quality_classification`][yohou.datasets._fetchers.fetch_air_quality_classification] : Real air quality classification dataset.

    Examples
    --------
    >>> from yohou.datasets import make_exogenous_classification
    >>> data = make_exogenous_classification(n_samples=200)
    >>> data.y.columns
    ['time', 'air_quality']
    >>> data.classes
    ['good', 'moderate', 'poor']

    """
    rng = np.random.default_rng(random_state)
    times = pl.datetime_range(
        datetime(2024, 1, 1),
        datetime(2024, 1, 1) + timedelta(hours=n_samples - 1),
        interval="1h",
        eager=True,
    ).alias("time")
    t = np.arange(n_samples, dtype=float)

    # Pollutant with 24h cycle centered at 50, amplitude 20
    pollutant = 50.0 + 20.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 5.0, n_samples)

    is_holiday = _holiday_indicator(times)

    # Holiday effect: less traffic, lower pollutant readings
    effective_pollutant = pollutant - 5.0 * is_holiday + rng.normal(0, noise, n_samples)

    # Classify
    classes = ["good", "moderate", "poor"]
    labels = np.where(
        effective_pollutant < 40,
        "good",
        np.where(effective_pollutant < 60, "moderate", "poor"),
    )

    y = pl.DataFrame({"time": times, "air_quality": labels})
    X_actual = pl.DataFrame({"time": times, "pollutant": pollutant})
    X_future = pl.DataFrame({"time": times, "is_holiday": is_holiday})

    vintage_idx, target_idx = _forecast_index_grid(n_samples, forecasting_horizon)
    pollutant_noise = rng.normal(0, 2.0, vintage_idx.shape[0])
    X_forecast = pl.DataFrame(
        {
            "vintage_time": times.gather(vintage_idx),
            "time": times.gather(target_idx),
            "pollutant_forecast": pollutant[target_idx] + forecast_bias + pollutant_noise,
        },
        schema={"vintage_time": times.dtype, "time": times.dtype, "pollutant_forecast": pl.Float64},
    )

    frame = y.join(X_actual, on="time").join(X_future, on="time")

    return Bunch(
        y=y,
        X_actual=X_actual,
        X_future=X_future,
        X_forecast=X_forecast,
        frame=frame,
        feature_names=["pollutant", "is_holiday", "pollutant_forecast"],
        target_names=["air_quality"],
        classes=classes,
        frequency="1h",
        DESCR=(
            "Synthetic hourly air quality classification with exogenous features.\n"
            "Target: air_quality in {good, moderate, poor} based on pollutant thresholds.\n"
            "X_actual: realized pollutant readings (sinusoidal 24h cycle + noise).\n"
            "X_future: is_holiday indicator from a fixed civil holiday calendar\n"
            "  (irregular dates, not derivable from the timestamp).\n"
            "X_forecast: pollutant concentration forecasts with systematic bias."
        ),
    )
