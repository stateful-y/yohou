"""Input validation utilities for time series data."""

import calendar
import re
from datetime import datetime, timedelta

import polars as pl
import polars.selectors as cs


def check_interval_consistency(df: pl.DataFrame) -> str:
    """Validate that a time series has uniform time spacing.

    Checks that all consecutive time steps in the DataFrame have the same interval.
    Supports both fixed intervals (daily, hourly) and variable-length intervals
    (monthly, quarterly, yearly).

    Parameters
    ----------
    df : pl.DataFrame
        Time series DataFrame with a "time" column containing datetime values.

    Returns
    -------
    str
        String representation of the interval.
        Examples: "1d", "1h", "1w", "1mo", "3mo", "1q", "1y"

    Raises
    ------
    ValueError
        If the time intervals are not consistent throughout the DataFrame.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> # Valid: uniform 1-day intervals
    >>> df = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         start=datetime(2020, 1, 1),
    ...         end=datetime(2020, 1, 5),
    ...         interval="1d",
    ...         eager=True
    ...     ),
    ...     "value": [10, 20, 30, 40, 50]
    ... })
    >>> interval = check_interval_consistency(df)
    >>> interval
    '1d'

    >>> # Invalid: inconsistent intervals
    >>> df_bad = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 4)],
    ...     "value": [10, 20, 30]
    ... })
    >>> check_interval_consistency(df_bad)  # doctest: +SKIP
    ValueError

    See Also
    --------
    check_inputs : Validates multiple DataFrames have matching intervals
    check_continuity : Validates temporal continuity between DataFrames
    add_interval : Add intervals to datetime values

    """
    time_series = df["time"].to_list()

    if len(time_series) < 2:
        raise ValueError("Need at least 2 time points to infer interval")

    # Calculate deltas
    deltas = [time_series[i + 1] - time_series[i] for i in range(len(time_series) - 1)]
    unique_deltas = sorted(set(deltas))

    # Fast path: exact timedelta match - convert to string
    if len(unique_deltas) == 1:
        return _timedelta_to_string(unique_deltas[0])

    # Check if deltas are all similar (within small tolerance for rounding)
    delta_days = [d.days for d in unique_deltas]
    max_delta = max(delta_days)

    # Sub-day intervals with small variation (e.g., hourly with DST)
    if max_delta == 0:
        # All deltas are sub-day
        delta_seconds = [d.total_seconds() for d in unique_deltas]
        if max(delta_seconds) - min(delta_seconds) <= 3600:  # ±1 hour tolerance
            median_seconds = sorted(delta_seconds)[len(delta_seconds) // 2]
            return _timedelta_to_string(timedelta(seconds=median_seconds))

    # Infer based on delta distribution
    freq = _infer_freq_from_deltas(time_series, unique_deltas)
    if freq is not None:
        return freq

    # Could not infer - raise detailed error
    raise ValueError(
        f"Time series has inconsistent intervals. "
        f"Found {len(unique_deltas)} different intervals: {unique_deltas}. "
        f"Cannot infer a regular frequency pattern."
    )


def check_inputs(y: pl.DataFrame, X_ante: pl.DataFrame | None, X_post: pl.DataFrame | None) -> str:
    """Validate that target and feature DataFrames have consistent time intervals.

    Ensures all input DataFrames (target y, ex-ante features X_ante, ex-post features
    X_post) have the same uniform time interval. This is required for proper alignment
    in forecasting operations.

    Parameters
    ----------
    y : pl.DataFrame
        Target time series with "time" column.

    X_ante : pl.DataFrame or None
        Ex-ante (known in advance) feature time series with "time" column, or None.

    X_post : pl.DataFrame or None
        Ex-post (observed after the fact) feature time series with "time" column, or None.

    Returns
    -------
    str
        The common time interval shared by all provided DataFrames (e.g., "1d", "1mo").

    Raises
    ------
    ValueError
        If any DataFrame has inconsistent intervals internally, or if the intervals
        don't match across DataFrames.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> time_index = pl.datetime_range(
    ...     start=datetime(2020, 1, 1),
    ...     end=datetime(2020, 1, 5),
    ...     interval="1d",
    ...     eager=True
    ... )
    >>> y = pl.DataFrame({"time": time_index, "sales": [100, 110, 120, 130, 140]})
    >>> X_ante = pl.DataFrame({"time": time_index, "holiday": [0, 0, 1, 0, 0]})
    >>> interval = check_inputs(y, X_ante, None)
    >>> interval
    '1d'

    See Also
    --------
    check_interval_consistency : Validates single DataFrame intervals

    """
    y_interval = check_interval_consistency(y)
    if X_ante is not None:
        X_ante_interval = check_interval_consistency(X_ante)

        if X_ante_interval != y_interval:
            raise ValueError(
                f"Time interval mismatch: y has interval {y_interval},  but X_ante has interval "
                f"{X_ante_interval}. All inputs must have the same time interval."
            )

    if X_post is not None:
        X_post_interval = check_interval_consistency(X_post)

        if X_post_interval != y_interval:
            raise ValueError(
                f"Time interval mismatch: y has interval {y_interval}, but X_post has interval "
                f"{X_post_interval}. All inputs must have the same time interval."
            )

    return y_interval


def check_continuity(
    df_p: pl.DataFrame, df_n: pl.DataFrame, expected_interval: str, check_intervals: bool = True
) -> None:
    """Validate temporal continuity between consecutive DataFrames.

    Ensures that two DataFrames representing consecutive time periods have no gaps
    or overlaps in their time indices. Used when appending new data to existing
    time series.

    Parameters
    ----------
    df_p : pl.DataFrame
        Previous (earlier) time series DataFrame with "time" column.

    df_n : pl.DataFrame
        Next (later) time series DataFrame with "time" column.

    expected_interval : str
        Expected time interval between consecutive observations.
        Examples: "1d", "1h", "1mo", "3mo", "1y"

    check_intervals : bool, default=True
        If True, validates that both DataFrames have consistent internal intervals.

    Raises
    ------
    ValueError
        If there is a gap or overlap between df_p and df_n, or if internal
        intervals don't match expected_interval.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> # Continuous time series
    >>> df1 = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         datetime(2020, 1, 1), datetime(2020, 1, 3), "1d", eager=True
    ...     ),
    ...     "value": [10, 20, 30]
    ... })
    >>> df2 = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         datetime(2020, 1, 4), datetime(2020, 1, 6), "1d", eager=True
    ...     ),
    ...     "value": [40, 50, 60]
    ... })
    >>> check_continuity(df1, df2, "1d")

    See Also
    --------
    check_interval_consistency : Validates uniform time spacing

    """
    if check_intervals:
        if len(df_p) > 1:
            interval_p = check_interval_consistency(df_p)

            if interval_p != expected_interval:
                raise ValueError(
                    f"Previous DataFrame has interval {interval_p}, but expected interval is "
                    f"{expected_interval}."
                )

        if len(df_n) > 1:
            interval_n = check_interval_consistency(df_n)

            if len(df_p) > 1 and interval_p != interval_n:
                raise ValueError(
                    "Interval mismatch between DataFrames: previous DataFrame has interval "
                    f"{interval_p}, but next DataFrame has interval {interval_n}."
                )

            if interval_n != expected_interval:
                raise ValueError(
                    f"Next DataFrame has interval {interval_n}, "
                    f"but expected interval is {expected_interval}."
                )

    time_p = df_p.select(cs.by_name("time")).tail(1)
    time_n = df_n.select(cs.by_name("time")).head(1)

    time = pl.concat([time_p, time_n])

    time_change = time.select(pl.col("time").diff()).fill_null(strategy="backward")
    interval_td = time_change[0, 0]

    # Convert expected_interval string to timedelta for comparison
    expected_interval_td = interval_to_timedelta(expected_interval)

    # For variable-length intervals (monthly, quarterly, yearly), we need to compute
    # the expected timedelta using calendar arithmetic
    if expected_interval_td is None:
        # Use add_interval to compute what the expected next time should be
        last_time = time_p["time"].item()  # Extract scalar from single-row DataFrame
        expected_next_time = add_interval(last_time, expected_interval, 1)
        first_time = time_n["time"].item()  # Extract scalar from single-row DataFrame

        if first_time != expected_next_time:
            if first_time > expected_next_time:
                raise ValueError(
                    f"Gap detected between DataFrames: previous DataFrame ends at {last_time}, "
                    f"next DataFrame starts at {first_time}, expected {expected_next_time} "
                    f"(interval {expected_interval})."
                )
            else:
                raise ValueError(
                    f"Overlap detected between DataFrames: previous DataFrame ends at {last_time}, "
                    f"next DataFrame starts at {first_time}, expected {expected_next_time} "
                    f"(interval {expected_interval})."
                )
    else:
        # Fixed interval - can use timedelta comparison
        if interval_td != expected_interval_td:
            last_time_p = time_p[0, 0]
            first_time_n = time_n[0, 0]
            if interval_td > expected_interval_td:
                raise ValueError(
                    f"Gap detected between DataFrames: previous DataFrame ends at {last_time_p}, "
                    f"next DataFrame starts at {first_time_n}, creating a gap of {interval_td} "
                    f"(expected {expected_interval})."
                )
            else:
                raise ValueError(
                    f"Overlap detected between DataFrames: previous DataFrame ends at "
                    f"{last_time_p}, next DataFrame starts at {first_time_n}, with interval "
                    f"{interval_td} (expected {expected_interval})."
                )


def check_inverse_transform(
    X_t: pl.DataFrame, X_p: pl.DataFrame | None, observation_horizon: int
) -> None:
    """Validate inputs for inverse_transform operations.

    Ensures that the transformed time series (X_t) and previous untransformed
    time series (X_p) meet requirements for inverse transformation:
    - Both have consistent time intervals
    - X_p is provided when observation_horizon > 0 and is at least that long
    - X_p and X_t are temporally continuous (no gaps or overlaps)

    Parameters
    ----------
    X_t : pl.DataFrame
        Transformed time series with "time" column to be inverted.

    X_p : pl.DataFrame or None
        Untransformed time series corresponding to at least `observation_horizon`
        immediately previous time stamps. Required when observation_horizon > 0.

    observation_horizon : int
        Number of previous observations required for inverse transformation.
        If 0, X_p can be None. If > 0, X_p must be provided.

    Raises
    ------
    ValueError
        If X_p is None when observation_horizon > 0, if time intervals are
        inconsistent, or if X_p and X_t are not temporally continuous.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> # Valid: observation_horizon = 0, X_p can be None
    >>> X_t = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         datetime(2020, 1, 1), datetime(2020, 1, 3), "1d", eager=True
    ...     ),
    ...     "value": [1.0, 2.0, 3.0]
    ... })
    >>> check_inverse_transform(X_t, None, observation_horizon=0)  # No error

    >>> # Valid: observation_horizon > 0 with continuous X_p
    >>> X_p = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         datetime(2019, 12, 30), datetime(2019, 12, 31), "1d", eager=True
    ...     ),
    ...     "value": [8.0, 9.0]
    ... })
    >>> X_t = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         datetime(2020, 1, 1), datetime(2020, 1, 3), "1d", eager=True
    ...     ),
    ...     "value": [1.0, 2.0, 3.0]
    ... })
    >>> check_inverse_transform(X_t, X_p, observation_horizon=2)  # No error

    See Also
    --------
    check_interval_consistency : Validates uniform time spacing
    check_continuity : Validates temporal continuity between DataFrames

    """
    # Validate observation_horizon requirement
    if observation_horizon > 0 and X_p is None:
        raise ValueError(
            "X_p cannot be None to invert a transform that has observation_horizon > 0. "
            "Provide the necessary previous untransformed data."
        )

    # Check interval consistency for X_t
    X_t_interval = check_interval_consistency(X_t)

    # If X_p is provided, validate it and check continuity with X_t
    if X_p is not None and len(X_p) > 0:
        # Validate X_p has sufficient length
        if len(X_p) < observation_horizon:
            raise ValueError(
                f"X_p must have at least {observation_horizon} rows (observation_horizon), "
                f"but has only {len(X_p)} rows."
            )

        # Check interval consistency for X_p (only if it has more than 1 row)
        if len(X_p) > 1:
            X_p_interval = check_interval_consistency(X_p)

            # Ensure intervals match
            if X_p_interval != X_t_interval:
                raise ValueError(
                    f"Time intervals do not match: X_p has interval {X_p_interval}, "
                    f"but X_t has interval {X_t_interval}."
                )

        # Check temporal continuity: X_p should end right before X_t begins
        check_continuity(X_p, X_t, expected_interval=X_t_interval, check_intervals=True)


def _timedelta_to_string(td: timedelta) -> str:
    """Convert a timedelta to string interval format.

    Parameters
    ----------
    td : timedelta
        Timedelta to convert.

    Returns
    -------
    str
        String representation like \"1d\", \"1h\", \"1w\", \"2w\", etc.

    Examples
    --------
    >>> _timedelta_to_string(timedelta(days=1))
    '1d'
    >>> _timedelta_to_string(timedelta(hours=1))
    '1h'
    >>> _timedelta_to_string(timedelta(days=7))
    '7d'

    """
    total_seconds = td.total_seconds()

    # Try common patterns from largest to smallest
    # Prefer day-based representation for consistency
    if total_seconds % 86400 == 0:
        days = int(total_seconds // 86400)
        return f"{days}d"
    elif total_seconds % 3600 == 0:
        hours = int(total_seconds // 3600)
        return f"{hours}h"
    elif total_seconds % 60 == 0:
        minutes = int(total_seconds // 60)
        return f"{minutes}min"
    else:
        seconds = int(total_seconds)
        return f"{seconds}s"


def _infer_freq_from_deltas(
    time_series: list[datetime], unique_deltas: list[timedelta]
) -> str | None:
    """Infer frequency pattern from delta distribution.

    Parameters
    ----------
    time_series : list[datetime]
        List of datetime values.

    unique_deltas : list[timedelta]
        Unique time deltas between consecutive dates.

    Returns
    -------
    str or None
        Inferred frequency string or None if cannot infer.

    """
    delta_days = [d.days for d in unique_deltas]
    min_delta, max_delta = min(delta_days), max(delta_days)

    # Daily pattern: uniform day intervals
    if len(unique_deltas) == 1:
        return _timedelta_to_string(unique_deltas[0])

    # Try monthly inference first (handles 1mo, 2mo, 3mo, 6mo, etc.)
    # Monthly patterns: 28-31 days (1mo), 59-62 days (2mo), 89-92 days (3mo), 181-184 days (6mo)
    if (
        28 <= min_delta <= 31
        or 59 <= min_delta <= 62
        or 89 <= min_delta <= 92
        or 181 <= min_delta <= 184
    ):
        freq = _infer_monthly_freq(time_series)
        if freq is not None:
            return freq

    # Yearly patterns: 365-366 days
    if 365 <= min_delta <= 366 and 365 <= max_delta <= 366:
        return "1y"

    return None


def _infer_monthly_freq(time_series: list[datetime]) -> str | None:
    """Infer monthly frequency (1mo, 2mo, 3mo, etc.) by checking month differences.

    Parameters
    ----------
    time_series : list[datetime]
        List of datetime values.

    Returns
    -------
    str or None
        Frequency string like \"1mo\", \"2mo\", \"3mo\", etc., or None if not monthly.

    """
    month_diffs = []
    for i in range(len(time_series) - 1):
        d1, d2 = time_series[i], time_series[i + 1]
        month_diff = (d2.year - d1.year) * 12 + (d2.month - d1.month)
        month_diffs.append(month_diff)

    unique_month_diffs = set(month_diffs)
    if len(unique_month_diffs) == 1:
        n_months = month_diffs[0]
        if _check_day_of_month_consistency(time_series):
            return f"{n_months}mo" if n_months > 1 else "1mo"
    return None


def _check_day_of_month_consistency(time_series: list[datetime]) -> bool:
    """Check if day-of-month is consistent (allowing for month-end edge cases).

    Parameters
    ----------
    time_series : list[datetime]
        List of datetime values.

    Returns
    -------
    bool
        True if day-of-month is consistent.

    """
    if not time_series:
        return False

    # The target day is the day from the first date in the series
    target_day = time_series[0].day

    for dt in time_series:
        days_in_month = calendar.monthrange(dt.year, dt.month)[1]
        actual_day = dt.day

        # If target day exceeds days in month, should be capped at month end
        if target_day > days_in_month:
            if actual_day != days_in_month:
                return False
        elif target_day != actual_day:
            return False

    return True


def parse_interval(interval: str) -> tuple[int, str]:
    """Parse interval string into (multiplier, unit).

    Parameters
    ----------
    interval : str
        Interval string like \"1d\", \"3mo\", \"2w\".

    Returns
    -------
    tuple[int, str]
        Tuple of (multiplier, unit).

    Examples
    --------
    >>> parse_interval(\"1d\")
    (1, 'd')
    >>> parse_interval(\"3mo\")
    (3, 'mo')
    >>> parse_interval(\"2w\")
    (2, 'w')

    """
    match = re.match(r"(\d+)(mo|q|y|w|d|h|min|s)", interval)
    if not match:
        raise ValueError(f"Invalid interval format: {interval}")
    return int(match.group(1)), match.group(2)


def interval_to_timedelta(interval: str) -> timedelta | None:
    """Convert fixed interval to timedelta, or None for variable intervals.

    Parameters
    ----------
    interval : str
        Interval string.

    Returns
    -------
    timedelta or None
        Timedelta for fixed intervals, None for variable intervals.

    Examples
    --------
    >>> interval_to_timedelta(\"1d\")
    datetime.timedelta(days=1)
    >>> interval_to_timedelta(\"2h\")
    datetime.timedelta(seconds=7200)
    >>> interval_to_timedelta(\"1mo\") is None
    True

    """
    multiplier, unit = parse_interval(interval)

    if unit == "d":
        return timedelta(days=multiplier)
    elif unit == "h":
        return timedelta(hours=multiplier)
    elif unit == "min":
        return timedelta(minutes=multiplier)
    elif unit == "s":
        return timedelta(seconds=multiplier)
    elif unit == "w":
        return timedelta(weeks=multiplier)
    else:
        # Variable-length interval, cannot convert to timedelta
        return None


def add_interval(dt: datetime, interval: str, n: int = 1) -> datetime:
    """Add n intervals to a datetime (handles variable-length intervals).

    Supports multi-period intervals like \"2mo\", \"3mo\", \"6mo\", etc.

    Parameters
    ----------
    dt : datetime
        Starting datetime.

    interval : str
        Interval string like \"1d\", \"1mo\", \"3mo\", \"1q\", \"1y\".

    n : int, default=1
        Number of intervals to add.

    Returns
    -------
    datetime
        Result datetime.

    Examples
    --------
    >>> from datetime import datetime
    >>> add_interval(datetime(2020, 1, 15), \"1d\", 5)
    datetime.datetime(2020, 1, 20, 0, 0)
    >>> add_interval(datetime(2020, 1, 31), \"1mo\", 1)
    datetime.datetime(2020, 2, 29, 0, 0)
    >>> add_interval(datetime(2020, 1, 31), \"2mo\", 2)
    datetime.datetime(2020, 5, 31, 0, 0)

    """
    multiplier, unit = parse_interval(interval)
    total_units = multiplier * n

    if unit == "d":
        return dt + timedelta(days=total_units)
    elif unit == "h":
        return dt + timedelta(hours=total_units)
    elif unit == "min":
        return dt + timedelta(minutes=total_units)
    elif unit == "s":
        return dt + timedelta(seconds=total_units)
    elif unit == "w":
        return dt + timedelta(weeks=total_units)
    elif unit == "mo":
        # Add months with day-of-month preservation
        month = dt.month - 1 + total_units
        year = dt.year + month // 12
        month = month % 12 + 1
        day = min(dt.day, calendar.monthrange(year, month)[1])
        return dt.replace(year=year, month=month, day=day)
    elif unit == "q":
        # Quarters are 3 months
        return add_interval(dt, "3mo", n)
    elif unit == "y":
        # Add years (handles leap years)
        return dt.replace(year=dt.year + total_units)
    else:
        raise ValueError(f"Unsupported interval unit: {unit}")
