"""Utilities for pivoting forecasts and windowing known-future features."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from yohou.utils.validation import add_interval

__all__ = ["pivot_forecasts", "window_forecasts", "window_futures"]


def pivot_forecasts(
    X_forecast: pl.DataFrame,
    *,
    vintage_col: str = "vintage_time",
    time_col: str = "time",
) -> pl.DataFrame:
    """Pivot tidy forecast data to step-indexed columns.

    Converts a tidy DataFrame with ``[vintage_col, time_col, col1, col2, ...]``
    into a wide DataFrame with ``[time, col1_step_1, col1_step_2, ..., col2_step_1, ...]``.
    The step index is the ordinal rank of the ``time_col`` within each vintage
    group (1-based). The ``vintage_col`` is renamed to ``"time"`` in the output,
    and the original ``time_col`` is dropped.

    Parameters
    ----------
    X_forecast : pl.DataFrame
        Tidy forecast DataFrame. Must contain ``vintage_col`` and ``time_col``
        columns plus one or more value columns.
    vintage_col : str, default="vintage_time"
        Name of the column identifying when the forecast was issued.
    time_col : str, default="time"
        Name of the column identifying the target time of each forecast value.

    Returns
    -------
    pl.DataFrame
        Wide DataFrame with ``[time, <col>_step_1, <col>_step_2, ...]``.
        Each row corresponds to one vintage. Missing steps in ragged
        vintages are filled with ``null``.

    Raises
    ------
    ValueError
        If ``vintage_col`` or ``time_col`` is not in ``X_forecast``.
    ValueError
        If no value columns remain after removing ``vintage_col`` and ``time_col``.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> X_forecast = pl.DataFrame({
    ...     "vintage_time": [datetime(2020, 1, 1)] * 3 + [datetime(2020, 1, 2)] * 3,
    ...     "time": [datetime(2020, 1, 2), datetime(2020, 1, 3), datetime(2020, 1, 4)] * 2,
    ...     "temp": [10.0, 11.0, 12.0, 15.0, 16.0, 17.0],
    ... })
    >>> pivot_forecasts(X_forecast)
    shape: (2, 4)
    ┌─────────────────────┬─────────────┬─────────────┬─────────────┐
    │ time                ┆ temp_step_1 ┆ temp_step_2 ┆ temp_step_3 │
    │ ---                 ┆ ---         ┆ ---         ┆ ---         │
    │ datetime[μs]        ┆ f64         ┆ f64         ┆ f64         │
    ╞═════════════════════╪═════════════╪═════════════╪═════════════╡
    │ 2020-01-01 00:00:00 ┆ 10.0        ┆ 11.0        ┆ 12.0        │
    │ 2020-01-02 00:00:00 ┆ 15.0        ┆ 16.0        ┆ 17.0        │
    └─────────────────────┴─────────────┴─────────────┴─────────────┘

    """
    if vintage_col not in X_forecast.columns:
        msg = f"Column '{vintage_col}' not found in DataFrame. Available columns: {X_forecast.columns}"
        raise ValueError(msg)
    if time_col not in X_forecast.columns:
        msg = f"Column '{time_col}' not found in DataFrame. Available columns: {X_forecast.columns}"
        raise ValueError(msg)

    value_cols = [c for c in X_forecast.columns if c not in (vintage_col, time_col)]
    if not value_cols:
        msg = (
            f"No value columns found. DataFrame has only '{vintage_col}' and "
            f"'{time_col}'. At least one value column is required."
        )
        raise ValueError(msg)

    if X_forecast.is_empty():
        return (
            X_forecast
            .drop(time_col)
            .rename({vintage_col: "time"})
            .select("time", *[c for c in X_forecast.columns if c not in {vintage_col, time_col}])
        )

    # Assign ordinal step index within each vintage group (1-based).
    X_forecast_ranked = X_forecast.with_columns(
        pl.col(time_col).rank("ordinal").over(vintage_col).cast(pl.Int32).alias("_step")
    )

    max_step = int(X_forecast_ranked["_step"].max())  # ty: ignore[invalid-argument-type]

    # Build pivot expressions: one column per (value_col, step) pair.
    pivot_exprs: list[pl.Expr] = []
    for step in range(1, max_step + 1):
        for col in value_cols:
            pivot_exprs.append(
                pl
                .when(pl.col("_step") == step)
                .then(pl.col(col))
                .otherwise(None)
                .max()  # one non-null value per group per step
                .alias(f"{col}_step_{step}")
            )

    result = X_forecast_ranked.group_by(vintage_col, maintain_order=True).agg(pivot_exprs)

    # Rename vintage_col → "time"
    result = result.rename({vintage_col: "time"})

    return result


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
        One row per observation time.

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

    # Build a lookup mapping from time → row values
    # Join approach: create a tidy representation with vintage_time, then pivot.
    rows: list[dict] = []
    for obs_time in observation_times.to_list():
        for step in range(1, forecasting_horizon + 1):
            target_time = add_interval(obs_time, interval, n=step)
            rows.append({"vintage_time": obs_time, "time": target_time, "_step": step})

    if not rows:
        # No observation times: return empty frame with correct schema
        cols = {"time": pl.Series([], dtype=observation_times.dtype)}
        for col in value_cols:
            for step in range(1, forecasting_horizon + 1):
                cols[f"{col}_step_{step}"] = pl.Series([], dtype=X_future[col].dtype)
        return pl.DataFrame(cols)

    lookup = pl.DataFrame(rows)

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

    This differs from `pivot_forecasts` which assigns steps by ordinal rank
    within each vintage group. ``window_forecasts`` aligns steps to the
    observation time, making it suitable for training with sparse vintage
    schedules (e.g., 6-hourly weather forecasts with hourly observations).

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

    # Build lookup rows: for each (obs_time, matched_vintage), target times T+1..T+H
    rows: list[dict] = []
    for obs_time, vintage in zip(
        matched["time"].to_list(),
        matched["_vintage"].to_list(),
        strict=True,
    ):
        if vintage is None:
            continue
        for step in range(1, forecasting_horizon + 1):
            target_time = add_interval(obs_time, interval, n=step)
            rows.append({
                "_obs_time": obs_time,
                vintage_col: vintage,
                time_col: target_time,
                "_step": step,
            })

    if not rows:
        return _null_result()

    lookup = pl.DataFrame(rows)

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
