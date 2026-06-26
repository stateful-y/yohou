"""Utilities for pivoting forecasts and windowing known-future features."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from yohou.utils.validation import parse_interval

__all__ = ["window_forecasts", "window_futures"]

_POLARS_OFFSET_UNITS: dict[str, tuple[int, str]] = {
    "d": (1, "d"),
    "h": (1, "h"),
    "m": (1, "m"),
    "s": (1, "s"),
    "w": (1, "w"),
    "mo": (1, "mo"),
    "q": (3, "mo"),
    "y": (1, "y"),
    "ms": (1, "ms"),
    "us": (1, "us"),
    "ns": (1, "ns"),
}


def _target_time_expr(base_col: str, step_col: str, interval: str | timedelta) -> pl.Expr:
    """Return an expression for ``base + step * interval`` over a DataFrame.

    Mirrors :func:`yohou.utils.validation.add_interval` in vectorized form,
    including the calendar-aware semantics of ``offset_by`` for month, quarter,
    and year units (day-of-month clamping and leap-year handling).
    """
    if isinstance(interval, timedelta):
        return pl.col(base_col) + pl.duration(microseconds=int(interval.total_seconds() * 1_000_000)) * pl.col(step_col)

    multiplier, unit = parse_interval(interval)
    scale, polars_unit = _POLARS_OFFSET_UNITS[unit]
    total = (pl.col(step_col) * (multiplier * scale)).cast(pl.String) + pl.lit(polars_unit)
    return pl.col(base_col).dt.offset_by(total)


def window_futures(
    X_future: pl.DataFrame,
    observation_times: pl.Series,
    forecasting_horizon: int,
    interval: str | timedelta,
    *,
    time_col: str = "time",
) -> pl.DataFrame:
    """Window known-future features into step-indexed columns.

    For each observation time T and forecast horizon H, extracts values at
    ``T + 1*interval`` through ``T + H*interval`` from ``X_future``, producing
    step-indexed columns ``<col>_step_1`` through ``<col>_step_H``.

    The output has one row per observation time and uses ``"time"`` as the
    time column (set to the observation time).

    Parameters
    ----------
    X_future : pl.DataFrame
        Known-future data with a ``time_col`` column and one or more value
        columns. Values are deterministic (e.g., holidays, day-of-week).
    observation_times : pl.Series
        Series of observation timestamps to window from.
    forecasting_horizon : int
        Number of forward steps (H) to extract per observation time.
    interval : str or timedelta
        Time frequency between steps (e.g., ``"1d"``, ``"1h"``).
    time_col : str, default="time"
        Name of the time column in ``X_future``.

    Returns
    -------
    pl.DataFrame
        Wide DataFrame with ``[time, <col>_step_1, ..., <col>_step_H]``.
        One row per unique observation time. Duplicate timestamps in
        ``observation_times`` are deduplicated, so the output contains at
        most one row per distinct observation time. This diverges from
        ``window_forecasts``, which preserves duplicate observation times.

    Raises
    ------
    ValueError
        If ``time_col`` is not in ``X_future``.
    ValueError
        If ``forecasting_horizon`` is not positive.
    ValueError
        If no value columns remain after removing ``time_col``.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> holidays = pl.DataFrame({
    ...     "time": [
    ...         datetime(2020, 1, 1),
    ...         datetime(2020, 1, 2),
    ...         datetime(2020, 1, 3),
    ...         datetime(2020, 1, 4),
    ...         datetime(2020, 1, 5),
    ...     ],
    ...     "is_holiday": [1, 0, 0, 1, 0],
    ... })
    >>> obs_times = pl.Series([datetime(2020, 1, 1), datetime(2020, 1, 2)])
    >>> window_futures(holidays, obs_times, forecasting_horizon=3, interval="1d")
    shape: (2, 4)
    ┌─────────────────────┬───────────────────┬───────────────────┬───────────────────┐
    │ time                ┆ is_holiday_step_1 ┆ is_holiday_step_2 ┆ is_holiday_step_3 │
    │ ---                 ┆ ---               ┆ ---               ┆ ---               │
    │ datetime[μs]        ┆ i64               ┆ i64               ┆ i64               │
    ╞═════════════════════╪═══════════════════╪═══════════════════╪═══════════════════╡
    │ 2020-01-01 00:00:00 ┆ 0                 ┆ 0                 ┆ 1                 │
    │ 2020-01-02 00:00:00 ┆ 0                 ┆ 1                 ┆ 0                 │
    └─────────────────────┴───────────────────┴───────────────────┴───────────────────┘

    """
    if time_col not in X_future.columns:
        msg = f"Column '{time_col}' not found in DataFrame. Available columns: {X_future.columns}"
        raise ValueError(msg)

    if forecasting_horizon < 1:
        msg = f"forecasting_horizon must be positive, got {forecasting_horizon}."
        raise ValueError(msg)

    value_cols = [c for c in X_future.columns if c != time_col]
    if not value_cols:
        msg = f"No value columns found. DataFrame has only '{time_col}'. At least one value column is required."
        raise ValueError(msg)

    if observation_times.is_empty():
        # No observation times: return empty frame with correct schema
        cols = {"time": pl.Series([], dtype=observation_times.dtype)}
        for col in value_cols:
            for step in range(1, forecasting_horizon + 1):
                cols[f"{col}_step_{step}"] = pl.Series([], dtype=X_future[col].dtype)
        return pl.DataFrame(cols)

    # Build a lookup table as the cross product of observation times and steps,
    # then compute each target time with a vectorized calendar-aware expression.
    steps = pl.DataFrame({"_step": range(1, forecasting_horizon + 1)})
    lookup = (
        pl
        .DataFrame({"vintage_time": observation_times})
        .join(steps, how="cross")
        .with_columns(_target_time_expr("vintage_time", "_step", interval).alias("time"))
    )

    # Join lookup with X_future on time to get values at each target time
    joined = lookup.join(
        X_future.select([time_col] + value_cols),
        left_on="time",
        right_on=time_col,
        how="left",
    )

    # Pivot to wide format
    pivot_exprs: list[pl.Expr] = []
    for step in range(1, forecasting_horizon + 1):
        for col in value_cols:
            pivot_exprs.append(
                pl.when(pl.col("_step") == step).then(pl.col(col)).otherwise(None).max().alias(f"{col}_step_{step}")
            )

    result = joined.group_by("vintage_time", maintain_order=True).agg(pivot_exprs)
    result = result.rename({"vintage_time": "time"})

    return result


def window_forecasts(
    X_forecast: pl.DataFrame,
    observation_times: pl.Series,
    forecasting_horizon: int,
    interval: str | timedelta,
    *,
    vintage_col: str = "vintage_time",
    time_col: str = "time",
) -> pl.DataFrame:
    """Window forecast data into step-indexed columns using as-of vintage selection.

    For each observation time T, selects the latest vintage V where V <= T
    (as-of / backward match), then extracts forecast values at
    ``T + 1*interval`` through ``T + H*interval`` from that vintage's rows.
    The result is a wide DataFrame with step columns aligned to observation
    times, not vintage times.

    Steps are aligned to the observation time, making this function suitable
    for training with sparse vintage schedules (e.g., 6-hourly weather
    forecasts with hourly observations).

    Parameters
    ----------
    X_forecast : pl.DataFrame
        Tidy forecast DataFrame with ``vintage_col``, ``time_col``, and one
        or more value columns. Each row is a single forecast value at a
        specific target time from a specific vintage.
    observation_times : pl.Series
        Series of observation timestamps. One output row per observation time.
    forecasting_horizon : int
        Number of forward steps (H) to extract per observation time.
    interval : str or timedelta
        Time frequency between steps (e.g., ``"1h"``, ``"1d"``).
    vintage_col : str, default="vintage_time"
        Name of the column identifying when the forecast was issued.
    time_col : str, default="time"
        Name of the column identifying the target time of each forecast value.

    Returns
    -------
    pl.DataFrame
        Wide DataFrame with ``[time, <col>_step_1, ..., <col>_step_H]``.
        One row per observation time. Step columns are null when no vintage
        is available at or before the observation time, or when the matched
        vintage does not cover that step's target time.

    Raises
    ------
    ValueError
        If ``vintage_col`` or ``time_col`` is not in ``X_forecast``.
    ValueError
        If ``forecasting_horizon`` is not positive.
    ValueError
        If no value columns remain after removing ``vintage_col`` and ``time_col``.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> X_forecast = pl.DataFrame({
    ...     "vintage_time": [datetime(2020, 1, 1, 0)] * 3 + [datetime(2020, 1, 1, 6)] * 3,
    ...     "time": [datetime(2020, 1, 1, 1), datetime(2020, 1, 1, 2), datetime(2020, 1, 1, 3)] * 2,
    ...     "temp": [10.0, 11.0, 12.0, 15.0, 16.0, 17.0],
    ... })
    >>> obs = pl.Series([datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 3)])
    >>> window_forecasts(X_forecast, obs, forecasting_horizon=2, interval="1h")
    shape: (2, 3)
    ┌─────────────────────┬─────────────┬─────────────┐
    │ time                ┆ temp_step_1 ┆ temp_step_2 │
    │ ---                 ┆ ---         ┆ ---         │
    │ datetime[μs]        ┆ f64         ┆ f64         │
    ╞═════════════════════╪═════════════╪═════════════╡
    │ 2020-01-01 00:00:00 ┆ 10.0        ┆ 11.0        │
    │ 2020-01-01 03:00:00 ┆ null        ┆ null        │
    └─────────────────────┴─────────────┴─────────────┘

    """
    if vintage_col not in X_forecast.columns:
        msg = f"Column '{vintage_col}' not found in DataFrame. Available columns: {X_forecast.columns}"
        raise ValueError(msg)
    if time_col not in X_forecast.columns:
        msg = f"Column '{time_col}' not found in DataFrame. Available columns: {X_forecast.columns}"
        raise ValueError(msg)
    if forecasting_horizon < 1:
        msg = f"forecasting_horizon must be positive, got {forecasting_horizon}."
        raise ValueError(msg)

    value_cols = [c for c in X_forecast.columns if c not in (vintage_col, time_col)]
    if not value_cols:
        msg = (
            f"No value columns found. DataFrame has only '{vintage_col}' and "
            f"'{time_col}'. At least one value column is required."
        )
        raise ValueError(msg)

    # Build null result schema for empty / no-match cases
    def _null_result() -> pl.DataFrame:
        """Return an all-null DataFrame with the expected step column schema."""
        cols: dict[str, pl.Series] = {"time": observation_times}
        for col in value_cols:
            for step in range(1, forecasting_horizon + 1):
                cols[f"{col}_step_{step}"] = pl.Series(
                    [None] * len(observation_times),
                    dtype=X_forecast[col].dtype,
                )
        return pl.DataFrame(cols)

    if X_forecast.is_empty() or observation_times.is_empty():
        return _null_result()

    # As-of join: for each observation time, find the latest vintage <= T
    obs_df = pl.DataFrame({"time": observation_times}).sort("time")
    vintage_keys = X_forecast.select(vintage_col).unique().sort(vintage_col).rename({vintage_col: "_vintage"})

    matched = obs_df.join_asof(
        vintage_keys,
        left_on="time",
        right_on="_vintage",
        strategy="backward",
    )
    # matched has columns: time, _vintage (the matched vintage or null)

    # Build lookup table: for each (obs_time, matched_vintage), target times
    # T+1..T+H. Drop observation times with no matching vintage, then take the
    # cross product with steps and compute target times in a vectorized pass.
    matched_with_vintage = matched.filter(pl.col("_vintage").is_not_null())
    if matched_with_vintage.is_empty():
        return _null_result()

    steps = pl.DataFrame({"_step": range(1, forecasting_horizon + 1)})
    lookup = (
        matched_with_vintage
        .rename({"time": "_obs_time", "_vintage": vintage_col})
        .join(steps, how="cross")
        .with_columns(_target_time_expr("_obs_time", "_step", interval).alias(time_col))
    )

    # Join lookup with X_forecast to get values at each target time from the matched vintage
    joined = lookup.join(
        X_forecast,
        on=[vintage_col, time_col],
        how="left",
    )

    # Pivot to wide format
    pivot_exprs: list[pl.Expr] = []
    for step in range(1, forecasting_horizon + 1):
        for col in value_cols:
            pivot_exprs.append(
                pl.when(pl.col("_step") == step).then(pl.col(col)).otherwise(None).max().alias(f"{col}_step_{step}")
            )

    result = joined.group_by("_obs_time", maintain_order=True).agg(pivot_exprs)
    result = result.rename({"_obs_time": "time"})

    # Left join back to observation_times to ensure all obs times are present
    # (some may have had no matching vintage)
    obs_all = pl.DataFrame({"time": observation_times})
    result = obs_all.join(result, on="time", how="left")

    return result
