from datetime import timedelta
from typing import Any

import polars as pl
import polars.selectors as cs


def check_interval_consistency(df: pl.DataFrame) -> timedelta:
    time_change = df.select(cs.by_name("time").diff()).fill_null(strategy="backward")

    interval: timedelta = time_change[0, 0]

    if len(time_change.filter(pl.col("time") == interval)) != len(time_change):
        raise ValueError()

    return interval


def check_inputs(
    y: pl.DataFrame, X_ante: pl.DataFrame | None, X_post: pl.DataFrame | None
) -> timedelta:
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
    interval = time_change[0, 0].seconds

    if interval != expected_interval:
        raise ValueError()
