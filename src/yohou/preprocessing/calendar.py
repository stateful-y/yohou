"""Calendar and holiday feature transformers for time series."""

from datetime import timedelta

import numpy as np
import polars as pl
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseActualTransformer
from yohou.utils.validation import interval_to_timedelta

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

_FEATURE_MIN_GRANULARITY: dict[str, set[str]] = {
    "hour": {"1s", "1m", "5m", "10m", "15m", "30m", "1h"},
    "minute": {"1s", "1m", "5m", "10m", "15m", "30m"},
}


def _interval_supports_feature(interval: str, feature: str) -> bool:
    """Check whether a time interval supports extracting a given feature.

    Parameters
    ----------
    interval : str
        Detected time interval string (e.g., "1d", "1h").
    feature : str
        Calendar feature name.

    Returns
    -------
    bool
        True if the feature is applicable to the interval.

    """
    required = _FEATURE_MIN_GRANULARITY.get(feature)
    if required is None:
        return True
    return interval in required


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


class CalendarFeatureTransformer(BaseActualTransformer):
    r"""Extract calendar-based features from the time column.

    Creates new integer feature columns derived from the datetime index,
    useful for capturing seasonal and calendar effects in reduction
    forecasters. Output columns are prefixed with ``cal_``.

    Each feature is a deterministic function of the timestamp $t$:

    $$f_j(t) = \text{calendar}_{j}(t) \quad \text{for } j \in \{\text{month}, \text{day_of_week}, \ldots\}$$

    For example, $f_{\text{month}}(t) \in \{1, \ldots, 12\}$ and
    $f_{\text{is_weekend}}(t) \in \{0, 1\}$.

    Parameters
    ----------
    features : list of str or None, default=None
        Calendar features to extract. If ``None``, extracts all features
        applicable to the detected time interval. Valid options:
        ``"year"``, ``"month"``, ``"week"``, ``"day_of_week"``,
        ``"day_of_month"``, ``"day_of_year"``, ``"hour"``, ``"minute"``,
        ``"quarter"``, ``"is_weekend"``, ``"is_month_start"``,
        ``"is_month_end"``, ``"is_quarter_start"``, ``"is_quarter_end"``,
        ``"is_year_start"``, ``"is_year_end"``.
    time_zone : str or None, default=None
        If set to an IANA zone, features are computed from the ``"time"`` column
        converted into that zone, so wall-clock features (``cal_hour``,
        ``cal_day_of_week``, ...) are local. The conversion is ephemeral: the output
        ``"time"`` column is returned unchanged, so it stays a valid join key against
        frames in its original zone. Requires a timezone-aware ``"time"`` column. If
        ``None`` (the default), features are read from ``"time"`` as-is.

    Attributes
    ----------
    applicable_features_ : list of str
        Calendar features that will be extracted during transform.

    Raises
    ------
    ValueError
        At fit time if any requested feature name is not a valid calendar
        feature; if a requested feature is not applicable to the detected time
        interval (e.g. ``"hour"`` on daily data); if any generated ``cal_*``
        column name conflicts with an existing column in ``X``; or if
        ``time_zone`` is set and the ``"time"`` column is not timezone-aware.

    See Also
    --------
    - [`HolidayFeatureTransformer`][yohou.preprocessing.calendar.HolidayFeatureTransformer] : Binary holiday indicator from user-provided dates.
    - [`DaylightSavingFeatureTransformer`][yohou.preprocessing.calendar.DaylightSavingFeatureTransformer] : Daylight-saving offset and transition-day features.
    - [`FourierFeatureTransformer`][yohou.preprocessing.time_features.FourierFeatureTransformer] : Sin/cos harmonics for cyclical encoding.
    - [`TimeIndexTransformer`][yohou.preprocessing.time_features.TimeIndexTransformer] : Numeric time index for trend features.
    - [`FunctionTransformer`][yohou.preprocessing.function.FunctionTransformer] : Custom function-based transforms.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> time = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 3, 1), interval="1d", eager=True
    ... )
    >>> X = pl.DataFrame({"time": time, "value": range(len(time))})
    >>> transformer = CalendarFeatureTransformer(features=["month", "day_of_week"])
    >>> transformer.fit(X)
    CalendarFeatureTransformer(features=['month', 'day_of_week'])
    >>> X_t = transformer.transform(X)
    >>> "cal_month" in X_t.columns
    True

    With ``time_zone`` set, ``cal_hour`` is local while the output ``"time"`` stays put:

    >>> from datetime import timezone
    >>> t = pl.datetime_range(
    ...     datetime(2026, 7, 1, 17, tzinfo=timezone.utc),
    ...     datetime(2026, 7, 1, 18, tzinfo=timezone.utc),
    ...     interval="1h",
    ...     eager=True,
    ... )
    >>> Xz = pl.DataFrame({"time": t})
    >>> out = CalendarFeatureTransformer(features=["hour"], time_zone="America/Chicago").fit_transform(Xz)
    >>> out["cal_hour"].to_list()  # UTC 17, 18 -> Central 12, 13
    [12, 13]
    >>> out["time"].to_list() == Xz["time"].to_list()  # output time unchanged
    True

    """

    _parameter_constraints: dict = {
        "features": [list, None],
        "time_zone": [str, None],
    }

    def __init__(
        self,
        features: list[str] | None = None,
        time_zone: str | None = None,
    ):
        self.features = features
        self.time_zone = time_zone

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit the internal model."""
        if self.time_zone is not None:
            dtype = X.schema["time"]
            source_zone = dtype.time_zone if isinstance(dtype, pl.Datetime) else None
            if source_zone is None:
                raise ValueError(
                    "CalendarFeatureTransformer with time_zone set requires a timezone-aware "
                    "'time' column (a zone conversion is undefined without an instant and a "
                    f"source zone); got dtype {dtype}. Localize the source first, e.g. "
                    "dt.replace_time_zone('UTC')."
                )
        if self.features is not None:
            invalid = set(self.features) - set(ALL_FEATURES)
            if invalid:
                raise ValueError(f"Unknown features: {sorted(invalid)}. Valid features: {list(ALL_FEATURES)}")
            inapplicable = [f for f in self.features if not _interval_supports_feature(self.interval_, f)]
            if inapplicable:
                raise ValueError(
                    f"Features {inapplicable} are not applicable to data with "
                    f"interval '{self.interval_}'. These features require "
                    f"sub-daily data."
                )
            self.applicable_features_ = list(self.features)
        else:
            self.applicable_features_ = [f for f in ALL_FEATURES if _interval_supports_feature(self.interval_, f)]

        generated_names = [f"cal_{f}" for f in self.applicable_features_]
        existing = set(X.columns) - {"time"}
        conflicts = set(generated_names) & existing
        if conflicts:
            raise ValueError(
                f"Generated column names {sorted(conflicts)} conflict with "
                f"existing columns in X. Rename input columns or select "
                f"different features."
            )

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Extract calendar features from the time column.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            DataFrame with ``"time"`` column and extracted calendar features.

        """
        feature_exprs = [_extract_feature(f, self.time_zone) for f in self.applicable_features_]

        # The output "time" is the original column, never rezoned: only the feature values
        # above are computed in ``time_zone``, so ``time`` stays a valid join key.
        return X.select(pl.col("time"), *feature_exprs)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names (unused, for API compatibility).

        Returns
        -------
        list of str
            All non-time output column names.

        """
        check_is_fitted(self, ["applicable_features_"])
        return [f"cal_{f}" for f in self.applicable_features_]


class HolidayFeatureTransformer(BaseActualTransformer):
    r"""Extract holiday indicator features from a user-provided holiday calendar.

    Produces a binary ``holiday_indicator`` column (and optional proximity
    features) by matching the ``"time"`` column against a provided DataFrame
    of holiday dates. Output columns are prefixed with ``holiday_``.

    Given a set of holiday dates $H = \{h_1, h_2, \ldots\}$, the indicator
    for each timestamp $t$ is:

    $$\text{is_holiday}(t) = \mathbb{1}[\text{date}(t) \in H]$$

    When ``days_to_next=True`` or ``days_since_last=True``, the transformer also computes:

    $$\text{days_to_next}(t) = \min_{h \in H, h \geq t} (h - t) \quad \text{(null if no future holiday)}$$

    $$\text{days_since_last}(t) = \min_{h \in H, h \leq t} (t - h) \quad \text{(null if no past holiday)}$$

    Parameters
    ----------
    holidays : pl.DataFrame or None, default=None
        DataFrame with a ``"date"`` column containing holiday dates.
        The column must be of ``Date`` or ``Datetime`` type. ``None`` is only
        a signature default; a DataFrame must be supplied before calling
        ``fit`` (passing ``None`` raises ``ValueError`` at fit time).
    days_to_next : bool, default=False
        If ``True``, produce a ``holiday_days_to_next`` integer column
        with the number of days until the next holiday (null if none).
    days_since_last : bool, default=False
        If ``True``, produce a ``holiday_days_since_last`` integer column
        with the number of days since the last holiday (null if none).

    Attributes
    ----------
    holiday_dates_ : list of date
        Sorted list of holiday dates used for matching.

    Raises
    ------
    ValueError
        At fit time if ``holidays`` is ``None``, is not a polars DataFrame,
        lacks a ``"date"`` column, or that column is not a Date/Datetime type.

    See Also
    --------
    - [`CalendarFeatureTransformer`][yohou.preprocessing.calendar.CalendarFeatureTransformer] : Calendar features (month, day of week, etc.).
    - [`DaylightSavingFeatureTransformer`][yohou.preprocessing.calendar.DaylightSavingFeatureTransformer] : Daylight-saving offset and transition-day features.
    - [`FourierFeatureTransformer`][yohou.preprocessing.time_features.FourierFeatureTransformer] : Sin/cos harmonics for cyclical encoding.
    - [`TimeIndexTransformer`][yohou.preprocessing.time_features.TimeIndexTransformer] : Numeric time index for trend features.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, date
    >>> time = pl.datetime_range(
    ...     start=datetime(2020, 12, 23), end=datetime(2020, 12, 27), interval="1d", eager=True
    ... )
    >>> X = pl.DataFrame({"time": time, "value": range(len(time))})
    >>> holidays = pl.DataFrame({"date": [date(2020, 12, 25)]})
    >>> transformer = HolidayFeatureTransformer(holidays=holidays)
    >>> transformer.fit(X)  # doctest: +ELLIPSIS
    HolidayFeatureTransformer(holidays=shape: (1, 1)...)
    >>> X_t = transformer.transform(X)
    >>> X_t["holiday_indicator"].to_list()
    [0, 0, 1, 0, 0]

    """

    _parameter_constraints: dict = {
        "holidays": "no_validation",
        "days_to_next": ["boolean"],
        "days_since_last": ["boolean"],
    }

    _PREFIX = "holiday"

    def __init__(
        self,
        holidays: pl.DataFrame | None = None,
        days_to_next: bool = False,
        days_since_last: bool = False,
    ):
        self.holidays = holidays
        self.days_to_next = days_to_next
        self.days_since_last = days_since_last

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit the internal model."""
        if self.holidays is None:
            raise ValueError("holidays must be provided as a polars DataFrame, got None")
        if not isinstance(self.holidays, pl.DataFrame):
            raise ValueError(f"holidays must be a polars DataFrame, got {type(self.holidays).__name__}")
        if "date" not in self.holidays.columns:
            raise ValueError("holidays DataFrame must have a 'date' column")
        date_dtype = self.holidays["date"].dtype
        if date_dtype not in (pl.Date, pl.Datetime, pl.Datetime("us"), pl.Datetime("ns"), pl.Datetime("ms")):
            raise ValueError(f"holidays 'date' column must be Date or Datetime type, got {date_dtype}")

        dates_series = self.holidays["date"].cast(pl.Date)
        self.holiday_dates_ = sorted(dates_series.to_list())

        generated_names = [f"{self._PREFIX}_indicator"]
        if self.days_to_next:
            generated_names.append(f"{self._PREFIX}_days_to_next")
        if self.days_since_last:
            generated_names.append(f"{self._PREFIX}_days_since_last")
        existing = set(X.columns) - {"time"}
        conflicts = set(generated_names) & existing
        if conflicts:
            raise ValueError(f"Generated column names {sorted(conflicts)} conflict with existing columns in X.")

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Extract holiday features from the time column.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            DataFrame with ``"time"`` column and holiday indicator features.

        """
        dates = X["time"].cast(pl.Date)
        is_holiday = dates.is_in(self.holiday_dates_).cast(pl.Int32).alias(f"{self._PREFIX}_indicator")

        new_cols = [is_holiday]

        if self.days_to_next or self.days_since_last:
            holiday_arr = np.array(self.holiday_dates_, dtype="datetime64[D]")
            dates_arr = dates.to_numpy().astype("datetime64[D]")

            if len(holiday_arr) == 0:
                if self.days_to_next:
                    new_cols.append(
                        pl.Series(
                            f"{self._PREFIX}_days_to_next",
                            [None] * len(X),
                            dtype=pl.Int32,
                        )
                    )
                if self.days_since_last:
                    new_cols.append(
                        pl.Series(
                            f"{self._PREFIX}_days_since_last",
                            [None] * len(X),
                            dtype=pl.Int32,
                        )
                    )
            else:
                if self.days_to_next:
                    # Smallest holiday h with h >= t (side="left" includes exact matches).
                    idx_next = np.searchsorted(holiday_arr, dates_arr, side="left")
                    valid = idx_next < len(holiday_arr)
                    deltas = (holiday_arr[np.clip(idx_next, 0, len(holiday_arr) - 1)] - dates_arr) // np.timedelta64(
                        1, "D"
                    )
                    to_next = [int(d) if v else None for d, v in zip(deltas, valid, strict=True)]
                    new_cols.append(pl.Series(f"{self._PREFIX}_days_to_next", to_next, dtype=pl.Int32))

                if self.days_since_last:
                    # Largest holiday h with h <= t (side="right" minus 1 includes exact matches,
                    # so a date that is itself a holiday yields 0).
                    idx_prev = np.searchsorted(holiday_arr, dates_arr, side="right") - 1
                    valid = idx_prev >= 0
                    deltas = (dates_arr - holiday_arr[np.clip(idx_prev, 0, len(holiday_arr) - 1)]) // np.timedelta64(
                        1, "D"
                    )
                    since_last = [int(d) if v else None for d, v in zip(deltas, valid, strict=True)]
                    new_cols.append(pl.Series(f"{self._PREFIX}_days_since_last", since_last, dtype=pl.Int32))

        return X.select(pl.col("time")).with_columns(*new_cols)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names (unused, for API compatibility).

        Returns
        -------
        list of str
            Generated holiday feature column names.

        """
        check_is_fitted(self, ["holiday_dates_"])
        generated = [f"{self._PREFIX}_indicator"]
        if self.days_to_next:
            generated.append(f"{self._PREFIX}_days_to_next")
        if self.days_since_last:
            generated.append(f"{self._PREFIX}_days_since_last")
        return generated


DST_ALL_FEATURES = ("in_effect", "transition_day", "transition_type")
DST_DEFAULT_FEATURES = ("in_effect", "transition_day")

# ``in_effect`` reads the offset at an instant; on daily-or-coarser data there is no
# single instant (it would be sampled at an arbitrary local midnight), so it is gated to
# sub-daily data. ``transition_day``/``transition_type`` are date-level and apply at any
# interval.
DST_SUBDAILY_FEATURES = frozenset({"in_effect"})


def _interval_is_subdaily(interval: str) -> bool:
    """Return whether a detected time interval is finer than one day.

    Parameters
    ----------
    interval : str
        Detected time interval string (e.g., "1h", "1d", "1mo").

    Returns
    -------
    bool
        True for fixed sub-daily intervals; False for daily-or-coarser and for
        variable intervals (e.g. "1mo", which has no fixed timedelta).

    """
    delta = interval_to_timedelta(interval)
    return delta is not None and delta < timedelta(days=1)


def _extract_dst_feature(feature: str, time_zone: str) -> pl.Expr:
    """Build a polars expression for a daylight-saving feature in ``time_zone``.

    Parameters
    ----------
    feature : str
        Daylight-saving feature name.
    time_zone : str
        IANA zone in which the daylight-saving regime is evaluated. The ``"time"``
        column is converted into it before the offset is read.

    Returns
    -------
    pl.Expr
        Expression that produces the ``dst_<feature>`` column.

    """
    local = pl.col("time").dt.convert_time_zone(time_zone)
    if feature == "in_effect":
        # Minutes of daylight-saving offset in effect: 0 in standard time, > 0 in DST.
        return (local.dt.dst_offset().dt.total_minutes() != 0).cast(pl.Int32).alias("dst_in_effect")
    # A local date is a transition date when its offset differs from the next date's,
    # compared at each date's local midnight.
    midnight = local.dt.truncate("1d")
    dst_today = midnight.dt.dst_offset().dt.total_minutes()
    dst_tomorrow = midnight.dt.offset_by("1d").dt.dst_offset().dt.total_minutes()
    if feature == "transition_day":
        return (dst_today != dst_tomorrow).cast(pl.Int32).alias("dst_transition_day")
    if feature == "transition_type":
        # Spring-forward gains the offset the next day (+1); fall-back loses it (-1).
        return (dst_tomorrow - dst_today).sign().cast(pl.Int32).alias("dst_transition_type")
    msg = f"Unknown feature: {feature}"
    raise ValueError(msg)


class DaylightSavingFeatureTransformer(BaseActualTransformer):
    r"""Deterministic daylight-saving-time features from the time column.

    A series whose local time observes daylight saving crosses two kinds of boundary that a
    UTC-indexed frame cannot see directly: the summer/winter clock offset, which shifts the
    UTC hour of the local diurnal cycle by one hour, and the two transition days a year,
    which are 23- or 25-hour local days with a skipped or repeated hour. Both are
    deterministic and known for any future date, so they enter a model as unlagged,
    known-future features alongside the calendar terms. Output columns are prefixed with
    ``dst_``.

    Daylight saving is evaluated in ``time_zone``: the ``"time"`` column (which must be
    timezone-aware) is converted into that zone, then the offset is read there. The
    parameter names the zone whose daylight-saving regime is evaluated, not an assertion
    about the input's zone, so a UTC-indexed frame and a Central-indexed frame produce the
    same features for the same instants.

    For a timestamp $t$ with local daylight-saving offset $\delta(t)$ (in minutes, $0$ in
    standard time), and $\delta_d$ the offset at the local midnight of date $d$:

    $$\text{in\_effect}(t) = \mathbb{1}[\delta(t) \neq 0], \quad
      \text{transition\_day}(d) = \mathbb{1}[\delta_d \neq \delta_{d+1}], \quad
      \text{transition\_type}(d) = \operatorname{sign}(\delta_{d+1} - \delta_d)$$

    Note that ``dst_transition_day`` is exactly ``dst_transition_type != 0``; both are
    provided because a tree splits on them differently (the binary is a cleaner root split,
    the signed carries the spring/fall direction).

    Parameters
    ----------
    time_zone : str, default="America/Chicago"
        IANA time zone in which daylight saving time is evaluated.
    features : list of str or None, default=None
        Which daylight-saving features to emit. If ``None``, emits every default feature
        applicable to the detected time interval (``["in_effect", "transition_day"]``).
        Valid options: ``"in_effect"``, ``"transition_day"``, ``"transition_type"``.

    Attributes
    ----------
    applicable_features_ : list of str
        The resolved feature list emitted during ``transform`` (after interval gating).

    Raises
    ------
    ValueError
        At fit time if the ``"time"`` column is not a timezone-aware ``pl.Datetime`` (a
        timezone-naive datetime or a ``pl.Date`` has no well-defined offset); if any
        requested feature name is unknown; if ``"in_effect"`` is explicitly requested on
        data whose detected interval is daily or coarser; or if a generated ``dst_*``
        column name conflicts with an existing column in ``X``.

    See Also
    --------
    - [`CalendarFeatureTransformer`][yohou.preprocessing.calendar.CalendarFeatureTransformer] : Calendar features (month, day of week, etc.).
    - [`HolidayFeatureTransformer`][yohou.preprocessing.calendar.HolidayFeatureTransformer] : Binary holiday indicator from user-provided dates.
    - [`FourierFeatureTransformer`][yohou.preprocessing.time_features.FourierFeatureTransformer] : Sin/cos harmonics for cyclical encoding.

    Examples
    --------
    >>> from datetime import datetime, timezone
    >>> import polars as pl
    >>> from yohou.preprocessing import DaylightSavingFeatureTransformer
    >>> time = pl.datetime_range(
    ...     datetime(2026, 3, 8, 6, tzinfo=timezone.utc),
    ...     datetime(2026, 3, 8, 10, tzinfo=timezone.utc),
    ...     interval="1h",
    ...     eager=True,
    ... )
    >>> X = pl.DataFrame({"time": time})
    >>> DaylightSavingFeatureTransformer().fit_transform(X)["dst_in_effect"].to_list()
    [0, 0, 1, 1, 1]

    """

    _parameter_constraints: dict = {
        "time_zone": [str],
        "features": [list, None],
    }

    def __init__(self, time_zone: str = "America/Chicago", features: list[str] | None = None):
        self.time_zone = time_zone
        self.features = features

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Validate the input contract, resolve features (with interval gating), and check conflicts."""
        dtype = X.schema["time"]
        time_zone = dtype.time_zone if isinstance(dtype, pl.Datetime) else None
        if time_zone is None:
            raise ValueError(
                "DaylightSavingFeatureTransformer requires a timezone-aware 'time' column "
                "(a daylight-saving offset is undefined without an instant and a zone); "
                f"got dtype {dtype}. Localize it first, e.g. dt.replace_time_zone('UTC')."
            )

        requested = list(self.features) if self.features is not None else list(DST_DEFAULT_FEATURES)
        invalid = set(requested) - set(DST_ALL_FEATURES)
        if invalid:
            raise ValueError(f"Unknown DST features: {sorted(invalid)}. Valid features: {list(DST_ALL_FEATURES)}")

        subdaily = _interval_is_subdaily(self.interval_)
        gated = [f for f in requested if f in DST_SUBDAILY_FEATURES and not subdaily]
        if self.features is not None and gated:
            raise ValueError(
                f"Features {gated} require sub-daily data, but the detected interval is "
                f"'{self.interval_}'. These features need an instant, not just a date."
            )
        self.applicable_features_ = [f for f in requested if f not in gated]

        generated = [f"dst_{f}" for f in self.applicable_features_]
        conflicts = set(generated) & (set(X.columns) - {"time"})
        if conflicts:
            raise ValueError(f"Generated column names {sorted(conflicts)} conflict with existing columns in X.")

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Emit ``time`` plus the selected ``dst_*`` feature columns."""
        exprs = [_extract_dst_feature(f, self.time_zone) for f in self.applicable_features_]
        return X.select(pl.col("time"), *exprs)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Return the selected ``dst_*`` feature-column names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names (unused, for API compatibility).

        Returns
        -------
        list of str
            Generated daylight-saving feature column names.

        """
        check_is_fitted(self, ["applicable_features_"])
        return [f"dst_{f}" for f in self.applicable_features_]
