"""Calendar feature extraction shared by the calendar transformers and segmenters.

Lives under ``yohou.utils`` rather than ``yohou.preprocessing`` because
``yohou.segmenting`` needs it and is imported by ``yohou.weighting``, which
``yohou.base`` imports at module level. Importing ``yohou.preprocessing`` from
there would be circular, since ``yohou.preprocessing`` imports ``yohou.base``.
"""

import numbers
from collections.abc import Iterable
from typing import Any

import polars as pl

ALL_FEATURES = (
    "year",
    "month",
    "week",
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "hour",
    "minute",
    "quarter",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
)

BOOLEAN_FEATURES = frozenset({
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
})


def _extract_feature(feature: str, time_zone: str | None = None) -> pl.Expr:
    """Build a polars expression that extracts a calendar feature from the time column.

    Parameters
    ----------
    feature : str
        Calendar feature name.
    time_zone : str or None, default=None
        If set, the ``"time"`` column is converted into this zone before the feature is
        read, so wall-clock features are local. This conversion is ephemeral: it affects
        only the extracted value, never the output ``"time"`` column.

    Returns
    -------
    pl.Expr
        Expression that produces the feature column.

    """
    col = pl.col("time")
    if time_zone is not None:
        col = col.dt.convert_time_zone(time_zone)
    if feature == "year":
        return col.dt.year().cast(pl.Int32).alias("cal_year")
    if feature == "month":
        return col.dt.month().cast(pl.Int32).alias("cal_month")
    if feature == "week":
        return col.dt.week().cast(pl.Int32).alias("cal_week")
    if feature == "day_of_week":
        return col.dt.weekday().cast(pl.Int32).alias("cal_day_of_week")
    if feature == "day_of_month":
        return col.dt.day().cast(pl.Int32).alias("cal_day_of_month")
    if feature == "day_of_year":
        return col.dt.ordinal_day().cast(pl.Int32).alias("cal_day_of_year")
    if feature == "hour":
        return col.dt.hour().cast(pl.Int32).alias("cal_hour")
    if feature == "minute":
        return col.dt.minute().cast(pl.Int32).alias("cal_minute")
    if feature == "quarter":
        return col.dt.quarter().cast(pl.Int32).alias("cal_quarter")
    if feature == "is_weekend":
        return (col.dt.weekday() >= 6).cast(pl.Int32).alias("cal_is_weekend")
    if feature == "is_month_start":
        return (col.dt.day() == 1).cast(pl.Int32).alias("cal_is_month_start")
    if feature == "is_month_end":
        return (col.dt.day() == col.dt.month_end().dt.day()).cast(pl.Int32).alias("cal_is_month_end")
    if feature == "is_quarter_start":
        return ((col.dt.month() - 1) % 3 == 0).and_(col.dt.day() == 1).cast(pl.Int32).alias("cal_is_quarter_start")
    if feature == "is_quarter_end":
        return (
            ((col.dt.month() % 3 == 0).and_(col.dt.day() == col.dt.month_end().dt.day()))
            .cast(pl.Int32)
            .alias("cal_is_quarter_end")
        )
    if feature == "is_year_start":
        return (col.dt.ordinal_day() == 1).cast(pl.Int32).alias("cal_is_year_start")
    if feature == "is_year_end":
        return (col.dt.month().eq(12).and_(col.dt.day().eq(31))).cast(pl.Int32).alias("cal_is_year_end")
    msg = f"Unknown feature: {feature}"
    raise ValueError(msg)


#: Inclusive ``(low, high)`` range of the integer each calendar feature yields, or None
#: when the feature is unbounded. Weekdays follow polars: Monday is 1, Sunday is 7.
_FEATURE_DOMAINS: dict[str, tuple[int, int] | None] = {
    "year": None,
    "month": (1, 12),
    "week": (1, 53),
    "day_of_week": (1, 7),
    "day_of_month": (1, 31),
    "day_of_year": (1, 366),
    "hour": (0, 23),
    "minute": (0, 59),
    "quarter": (1, 4),
    **dict.fromkeys(sorted(BOOLEAN_FEATURES), (0, 1)),
}

#: Features that need a time of day, so a ``Date`` series cannot provide them.
_TIME_OF_DAY_FEATURES = frozenset({"hour", "minute"})


def _check_feature_values(feature: str, values: Iterable[Any]) -> list[int]:
    """Validate candidate values of a calendar feature and return them as ints.

    Parameters
    ----------
    feature : str
        Calendar feature name, one of ``ALL_FEATURES``.
    values : iterable
        Candidate feature values. Each must be an integer inside the feature's range.

    Returns
    -------
    list of int
        The values as Python ints, in input order.

    Raises
    ------
    ValueError
        If ``feature`` is unknown, or if any value is not an integer or lies outside
        the feature's range. The message names the offending values and the range.

    """
    if feature not in _FEATURE_DOMAINS:
        msg = f"Unknown feature: {feature!r}. Valid features: {list(ALL_FEATURES)}"
        raise ValueError(msg)
    values = list(values)
    non_integers = [v for v in values if not isinstance(v, numbers.Integral)]
    if non_integers:
        msg = f"Values for feature {feature!r} must be integers, got {non_integers}"
        raise ValueError(msg)
    domain = _FEATURE_DOMAINS[feature]
    if domain is not None:
        low, high = domain
        out_of_range = [v for v in values if not low <= v <= high]
        if out_of_range:
            msg = f"Values {out_of_range} are outside the range of feature {feature!r} ({low} to {high})"
            raise ValueError(msg)
    return [int(v) for v in values]
