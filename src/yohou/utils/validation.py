"""Input validation utilities for time series data."""

from datetime import timedelta
from typing import Any

import polars as pl
import polars.selectors as cs


def check_interval_consistency(df: pl.DataFrame) -> timedelta:
    """Validate that a time series has uniform time spacing.

    Checks that all consecutive time steps in the DataFrame have the same interval.
    This is required for yohou forecasters to properly handle temporal data.

    Parameters
    ----------
    df : pl.DataFrame
        Time series DataFrame with a "time" column containing datetime values.

    Returns
    -------
    timedelta
        The uniform time interval between consecutive observations.

    Raises
    ------
    ValueError
        If the time intervals are not consistent throughout the DataFrame.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
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
    datetime.timedelta(days=1)

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

    """
    time_change = df.select(cs.by_name("time").diff()).fill_null(strategy="backward")

    interval: timedelta = time_change[0, 0]

    if len(time_change.filter(pl.col("time") == interval)) != len(time_change):
        raise ValueError()

    return interval


def check_inputs(
    y: pl.DataFrame, X_ante: pl.DataFrame | None, X_post: pl.DataFrame | None
) -> timedelta:
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
    timedelta
        The common time interval shared by all provided DataFrames.

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
    >>> interval.days
    1

    See Also
    --------
    check_interval_consistency : Validates single DataFrame intervals
    :meth:`yohou.base.BaseForecaster._pre_fit` : Calls this during fit

    """
    y_interval = check_interval_consistency(y)
    if X_ante is not None:
        X_ante_interval = check_interval_consistency(X_ante)

        if X_ante_interval != y_interval:
            raise ValueError()

    if X_post is not None:
        X_post_interval = check_interval_consistency(X_post)

        if X_post_interval != y_interval:
            raise ValueError()

    return y_interval


def check_continuity(
    df_p: pl.DataFrame, df_n: pl.DataFrame, expected_interval: Any, check_intervals: bool = True
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

    expected_interval : timedelta or Any
        Expected time interval between consecutive observations.

    check_intervals : bool, default=True
        If True, validates that both DataFrames have consistent internal intervals.

    Returns
    -------
    None

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
    >>> check_continuity(df1, df2, timedelta(days=1))  # No error - continuous

    See Also
    --------
    check_interval_consistency : Validates uniform time spacing
    :meth:`yohou.base.BaseForecaster.update` : Uses this when adding new observations

    """
    if check_intervals:
        if len(df_p) > 1:
            interval_p = check_interval_consistency(df_p)

            if interval_p != expected_interval:
                raise ValueError()

        if len(df_n) > 1:
            interval_n = check_interval_consistency(df_n)

            if len(df_p) > 1 and interval_p != interval_n:
                raise ValueError()

            if interval_n != expected_interval:
                raise ValueError()

            interval = interval_p

    time_p = df_p.select(cs.by_name("time"))[[-1]]
    time_n = df_n.select(cs.by_name("time"))[[0]]

    time = pl.concat([time_p, time_n])

    time_change = time.select(pl.col("time").diff()).fill_null(strategy="backward")
    interval = time_change[0, 0]

    if interval != expected_interval:
        raise ValueError()
