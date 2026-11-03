"""Outlier handling transformers for time series data."""

import numbers
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseActualTransformer
from yohou.utils._compat import HasMethods, Interval, StrOptions, _check_feature_names_in

__all__ = ["OutlierPercentileHandler", "OutlierThresholdHandler"]

#: Internal column holding the segment label of each row while bounds are resolved.
_LABEL_COL = "__yohou_segment_label__"

#: Bound expressions for one column: ``(low, high)``. A None entry means no bound on
#: that side; a null value within an expression means no bound for that row.
_Bounds = tuple[pl.Expr | None, pl.Expr | None]


def _apply_outlier_handling(
    X: pl.DataFrame,
    data_cols: list[str],
    strategy: str,
    get_bounds: Callable[[str], _Bounds],
) -> pl.DataFrame:
    """Apply outlier handling to a DataFrame.

    Parameters
    ----------
    X : pl.DataFrame
        Input time series with a "time" column, plus any helper columns the bound
        expressions read.
    data_cols : list of str
        Columns to handle; the output holds "time" and these columns only.
    strategy : str
        Either "clip" or "nan".
    get_bounds : callable
        Function that takes a column name and returns ``(low, high)`` bound
        expressions, evaluated per row. A row whose bound is null is not handled on
        that side.

    Returns
    -------
    pl.DataFrame
        Time series with outliers handled.

    """
    exprs = [pl.col("time")]

    for col_name in data_cols:
        col = pl.col(col_name)
        low, high = get_bounds(col_name)

        if strategy == "clip":
            if low is not None:
                col = col.clip(lower_bound=low)
            if high is not None:
                col = col.clip(upper_bound=high)
        else:  # nan
            if low is not None:
                col = pl.when(col < low).then(None).otherwise(col)
            if high is not None:
                col = pl.when(col > high).then(None).otherwise(col)

        exprs.append(col.alias(col_name))

    return X.select(exprs)


def _literal_bounds(low: float | None, high: float | None, inside: pl.Expr | None = None) -> _Bounds:
    """Turn scalar thresholds into bound expressions, optionally only for inside rows."""

    def to_expr(value: float | None) -> pl.Expr | None:
        """Return a literal bound, masked to inside rows when ``inside`` is given; None stays None."""
        if value is None:
            return None
        expr = pl.lit(value)
        return expr if inside is None else pl.when(inside).then(expr)

    return to_expr(low), to_expr(high)


def _finite_values(series: pl.Series) -> pl.Series:
    """Drop nulls, and NaN for float series."""
    series = series.drop_nulls()
    return series.drop_nans() if series.dtype.is_float() else series


def _label_key(label: Any) -> Any:
    """Python dict key for a segment label: struct labels become tuples of field values."""
    return tuple(label.values()) if isinstance(label, dict) else label


def _check_segmenter_label_length(labels: pl.Series, X: pl.DataFrame) -> pl.Series:
    """Raise if the segmenter returned a label count that differs from the row count."""
    if len(labels) != len(X):
        msg = f"segmenter returned {len(labels)} labels, expected {len(X)} (one per row)"
        raise ValueError(msg)
    return labels


class OutlierThresholdHandler(BaseActualTransformer):
    """Handle outliers based on fixed threshold values.

    Values outside the specified thresholds are either clipped to the threshold
    values or set to NaN. This is useful for removing known invalid readings
    or physical impossibilities from sensor data.

    Parameters
    ----------
    low : float or None, default=None
        Lower threshold. Values below this are handled according to strategy.
        If None, no lower bound is applied.
    high : float or None, default=None
        Upper threshold. Values above this are handled according to strategy.
        If None, no upper bound is applied.
    strategy : {"clip", "nan"}, default="clip"
        How to handle outliers:
        - "clip": Replace outliers with threshold values
        - "nan": Replace outliers with NaN
    segmenter : BaseSegmenter or None, default=None
        Restricts handling to rows the segmenter labels non-null (inside a
        segment), e.g. ``CalendarSegmenter("hour", values=range(8, 19))`` for
        business hours. Rows outside every segment are returned unchanged. If
        None, every row is handled.

    Attributes
    ----------
    low_ : float or None
        Validated lower threshold.
    high_ : float or None
        Validated upper threshold.
    segmenter_ : BaseSegmenter or None
        Clone of ``segmenter`` fitted at ``fit``, used by ``transform``.

    Raises
    ------
    ValueError
        At fit time if both ``low`` and ``high`` are provided and ``low`` > ``high``.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> from yohou.preprocessing import OutlierThresholdHandler

    >>> X = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "value": [-100.0, 50.0, 100.0, 150.0, 999.0],
    ... })

    >>> # Clip values to [0, 200]
    >>> handler = OutlierThresholdHandler(low=0.0, high=200.0, strategy="clip")
    >>> handler.fit(X)
    OutlierThresholdHandler(high=200.0, low=0.0)
    >>> X_handled = handler.transform(X)
    >>> X_handled["value"].to_list()
    [0.0, 50.0, 100.0, 150.0, 200.0]

    >>> # Set out-of-range values to NaN
    >>> handler = OutlierThresholdHandler(low=0.0, high=200.0, strategy="nan")
    >>> handler.fit(X)  # doctest: +ELLIPSIS
    OutlierThresholdHandler(...)
    >>> X_handled = handler.transform(X)
    >>> X_handled["value"].null_count()
    2

    >>> # Only clip during business hours; the 03:00 reading is left alone
    >>> from yohou.segmenting import CalendarSegmenter
    >>> X_hourly = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1, 3), datetime(2020, 1, 1, 12)],
    ...     "value": [500.0, 500.0],
    ... })
    >>> handler = OutlierThresholdHandler(
    ...     high=100.0, segmenter=CalendarSegmenter("hour", values=range(8, 19))
    ... )
    >>> handler.fit(X_hourly).transform(X_hourly)["value"].to_list()
    [500.0, 100.0]

    See Also
    --------
    - [`OutlierPercentileHandler`][yohou.preprocessing.outlier.OutlierPercentileHandler] : Handle outliers based on percentiles.
    - [`CalendarSegmenter`][yohou.segmenting.segmenters.CalendarSegmenter] : Segments by calendar feature.

    """

    _valid_strategies = {"clip", "nan"}

    _parameter_constraints: dict = {
        "low": [numbers.Real, None],
        "high": [numbers.Real, None],
        "strategy": [StrOptions(_valid_strategies)],
        "segmenter": [HasMethods(["compute_labels"]), None],
    }

    # Batch invariant: elementwise against fixed thresholds and per-row segment
    # labels (a row's label depends only on its own timestamp), so chunking cannot
    # change a value.
    _tags = {"stateful": False, "invertible": False, "batch_invariant": True}

    def __init__(
        self,
        low: float | None = None,
        high: float | None = None,
        strategy: str = "clip",
        segmenter=None,
    ):
        self.low = low
        self.high = high
        self.strategy = strategy
        self.segmenter = segmenter

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit the internal model."""
        # Validate threshold ordering before assigning any fitted attributes so
        # a failed fit does not leave the transformer in a half-fitted state.
        if self.low is not None and self.high is not None and self.low > self.high:
            msg = f"low ({self.low}) must be <= high ({self.high})"
            raise ValueError(msg)

        self.segmenter_ = None if self.segmenter is None else clone(self.segmenter).fit(X["time"])
        self.low_ = self.low
        self.high_ = self.high

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Handle outliers in time series.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            Transformed time series.

        """
        data_cols = [c for c in X.columns if c != "time"]
        if self.segmenter_ is None:
            bounds = _literal_bounds(self.low_, self.high_)
        else:
            labels = _check_segmenter_label_length(self.segmenter_.compute_labels(X["time"]), X)
            X = X.with_columns(labels.alias(_LABEL_COL))
            bounds = _literal_bounds(self.low_, self.high_, inside=pl.col(_LABEL_COL).is_not_null())
        return _apply_outlier_handling(X, data_cols, self.strategy, lambda _col_name: bounds)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : list of str or None, default=None
            Column names of the input features.  If ``None``, uses the
            feature names seen during ``fit``.

        Returns
        -------
        list of str
            Output feature names after transformation.

        """
        check_is_fitted(self, ["feature_names_in_"])
        input_features = _check_feature_names_in(self, input_features)
        return list(input_features)


class OutlierPercentileHandler(BaseActualTransformer):
    """Handle outliers based on percentile thresholds.

    Values outside the specified percentile range are either clipped to the
    percentile values or set to NaN. Percentiles are computed during fit.

    Parameters
    ----------
    low : float or None, default=None
        Lower percentile (0-100). Values below this percentile are handled.
        If None, no lower bound is applied.
    high : float or None, default=None
        Upper percentile (0-100). Values above this percentile are handled.
        If None, no upper bound is applied.
    strategy : {"clip", "nan"}, default="clip"
        How to handle outliers:
        - "clip": Replace outliers with percentile values
        - "nan": Replace outliers with NaN
    segmenter : BaseSegmenter or None, default=None
        Restricts fitting and handling to rows the segmenter labels non-null
        (inside a segment). Percentiles are computed from in-segment values only,
        and rows outside every segment are returned unchanged. If None, every row
        is used and handled.
    by_segment : bool, default=False
        If True, fit one threshold pair per segment label (e.g. per hour of the
        day with ``CalendarSegmenter("hour")``), so the normal range can depend on
        the time of day. Requires ``segmenter``.
    min_samples : int, default=30
        With ``by_segment=True``, the minimum number of finite values a label needs
        in a column for its own thresholds. Labels with fewer use the pooled
        thresholds; see Notes.

    Attributes
    ----------
    thresholds_ : dict
        Dictionary mapping column names to (low_value, high_value) tuples, computed
        from the finite in-segment values pooled across labels (from all values when
        ``segmenter`` is None).
    segment_thresholds_ : dict
        Only with ``by_segment=True``. Maps each column name to a dictionary from
        segment label to the (low_value, high_value) pair ``transform`` uses for
        that label. Labels are ints for
        [`CalendarSegmenter`][yohou.segmenting.segmenters.CalendarSegmenter] and
        tuples of field values for composite segmenters.
    segmenter_ : BaseSegmenter or None
        Clone of ``segmenter`` fitted at ``fit``, used by ``transform``.

    Raises
    ------
    ValueError
        At fit time if both ``low`` and ``high`` are provided and ``low`` > ``high``,
        or if ``by_segment=True`` without a ``segmenter``.

    Warns
    -----
    UserWarning
        At fit time if, with ``by_segment=True``, some (column, label) pairs have
        fewer than ``min_samples`` finite values; the warning counts and names them.
        Also at fit time for each column with no finite in-segment value; that
        column is returned unchanged.

    Notes
    -----
    Fallback to pooled thresholds. With ``by_segment=True``, a label seen at fit
    with fewer than ``min_samples`` finite values in a column uses that column's
    pooled thresholds (``thresholds_``) instead of its own, and so does a label
    that ``transform`` meets but ``fit`` never saw (for example June, with
    ``CalendarSegmenter("month")``, when the training data covered January to March). The first case warns at fit; the second is
    silent. ``segment_thresholds_`` records the thresholds actually used for each
    label seen at fit, including the fallbacks.

    Finite values exclude nulls and NaN, as in the percentile computation.

    With both ``low`` and ``high`` None the handler changes nothing, with or
    without a segmenter, and emits no warning; ``segment_thresholds_`` then holds
    ``(None, None)`` for every label.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> import numpy as np
    >>> from yohou.preprocessing import OutlierPercentileHandler

    >>> np.random.seed(42)
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)],
    ...     "value": np.random.randn(100).tolist(),
    ... })

    >>> # Clip to 5th-95th percentile
    >>> handler = OutlierPercentileHandler(low=5, high=95)
    >>> handler.fit(X)
    OutlierPercentileHandler(high=95, low=5)
    >>> X_handled = handler.transform(X)
    >>> "time" in X_handled.columns
    True

    >>> # Clip to the interquartile range (25th to 75th percentile)
    >>> handler = OutlierPercentileHandler(low=25, high=75)
    >>> handler.fit(X)  # doctest: +ELLIPSIS
    OutlierPercentileHandler(...)
    >>> X_handled = handler.transform(X)
    >>> len(X_handled)
    100

    >>> # One upper limit per hour of the day: 150 is an outlier at night only
    >>> from yohou.segmenting import CalendarSegmenter
    >>> hours = [datetime(2020, 1, 1) + timedelta(hours=i) for i in range(24 * 60)]
    >>> load = [300.0 if 8 <= t.hour <= 18 else 60.0 for t in hours]
    >>> X_hourly = pl.DataFrame({"time": hours, "load": load})
    >>> handler = OutlierPercentileHandler(high=95, segmenter=CalendarSegmenter("hour"), by_segment=True)
    >>> handler.fit(X_hourly)  # doctest: +ELLIPSIS
    OutlierPercentileHandler(...)
    >>> handler.segment_thresholds_["load"][3], handler.segment_thresholds_["load"][12]
    ((None, 60.0), (None, 300.0))

    See Also
    --------
    - [`OutlierThresholdHandler`][yohou.preprocessing.outlier.OutlierThresholdHandler] : Handle outliers based on fixed thresholds.
    - [`CalendarSegmenter`][yohou.segmenting.segmenters.CalendarSegmenter] : Segments by calendar feature.

    """

    _valid_strategies = {"clip", "nan"}

    _parameter_constraints: dict = {
        "low": [Interval(numbers.Real, 0, 100, closed="both"), None],
        "high": [Interval(numbers.Real, 0, 100, closed="both"), None],
        "strategy": [StrOptions(_valid_strategies)],
        "segmenter": [HasMethods(["compute_labels"]), None],
        "by_segment": ["boolean"],
        "min_samples": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    # Batch invariant: elementwise against fit-time percentiles and per-row segment
    # labels (a row's label depends only on its own timestamp), so chunking cannot
    # change a value.
    _tags = {"stateful": False, "invertible": False, "batch_invariant": True}

    def __init__(
        self,
        low: float | None = None,
        high: float | None = None,
        strategy: str = "clip",
        segmenter=None,
        by_segment: bool = False,
        min_samples: int = 30,
    ):
        self.low = low
        self.high = high
        self.strategy = strategy
        self.segmenter = segmenter
        self.by_segment = by_segment
        self.min_samples = min_samples

    def _percentiles(self, values: pl.Series) -> tuple[float | None, float | None]:
        """Lower and upper percentile thresholds of finite values (None when unset or empty)."""
        finite = _finite_values(values).to_numpy()
        if len(finite) == 0:
            return None, None
        low_val = None if self.low is None else float(np.percentile(finite, self.low))
        high_val = None if self.high is None else float(np.percentile(finite, self.high))
        return low_val, high_val

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit the internal model."""
        # Validate percentile ordering
        if self.low is not None and self.high is not None and self.low > self.high:
            msg = f"low ({self.low}) must be <= high ({self.high})"
            raise ValueError(msg)
        if self.by_segment and self.segmenter is None:
            raise ValueError("OutlierPercentileHandler with by_segment=True requires a segmenter")

        data_cols = [c for c in X.columns if c != "time"]
        has_bounds = self.low is not None or self.high is not None

        self.segmenter_ = None if self.segmenter is None else clone(self.segmenter).fit(X["time"])
        if self.segmenter_ is None:
            X_in = X
        else:
            labels = _check_segmenter_label_length(self.segmenter_.compute_labels(X["time"]), X)
            X_in = X.with_columns(labels.alias(_LABEL_COL)).filter(pl.col(_LABEL_COL).is_not_null())

        # Pooled thresholds over the in-segment rows (all rows without a segmenter)
        self.thresholds_: dict[str, tuple[float | None, float | None]] = {}
        for col_name in data_cols:
            self.thresholds_[col_name] = self._percentiles(X_in[col_name])
            if has_bounds and self.segmenter_ is not None and self.thresholds_[col_name] == (None, None):
                warnings.warn(
                    f"OutlierPercentileHandler: the segmenter selects no finite value of column "
                    f"{col_name!r} at fit, so the column is returned unchanged",
                    UserWarning,
                    stacklevel=4,
                )

        if self.by_segment:
            self._fit_segments(X_in, data_cols, has_bounds)

    def _fit_segments(self, X_in: pl.DataFrame, data_cols: list[str], has_bounds: bool) -> None:
        """Fit per-label thresholds, falling back to the pooled ones for small labels."""
        self.segment_thresholds_: dict[str, dict[Any, tuple[float | None, float | None]]] = {
            col_name: {} for col_name in data_cols
        }
        fallbacks: list[str] = []
        for (label,), group in X_in.group_by(_LABEL_COL, maintain_order=True):
            key = _label_key(label)
            for col_name in data_cols:
                if not has_bounds:
                    self.segment_thresholds_[col_name][key] = (None, None)
                elif len(_finite_values(group[col_name])) < self.min_samples:
                    self.segment_thresholds_[col_name][key] = self.thresholds_[col_name]
                    fallbacks.append(f"{col_name}/{key}")
                else:
                    self.segment_thresholds_[col_name][key] = self._percentiles(group[col_name])

        if fallbacks:
            shown = ", ".join(fallbacks[:10]) + (", ..." if len(fallbacks) > 10 else "")
            warnings.warn(
                f"OutlierPercentileHandler: {len(fallbacks)} (column, label) pairs have fewer than "
                f"min_samples={self.min_samples} finite values and use the pooled thresholds: {shown}",
                UserWarning,
                stacklevel=5,
            )

        # Join table for transform: one row per label seen at fit, one bound column
        # per (column, side). Unseen labels miss the join and fall back to pooled.
        label_dtype = X_in.schema[_LABEL_COL]
        labels = X_in[_LABEL_COL].unique(maintain_order=True)
        columns: dict[str, pl.Series] = {_LABEL_COL: labels}
        for col_name in data_cols:
            pairs = [self.segment_thresholds_[col_name][_label_key(label)] for label in labels.to_list()]
            columns[f"__low__{col_name}"] = pl.Series([p[0] for p in pairs], dtype=pl.Float64)
            columns[f"__high__{col_name}"] = pl.Series([p[1] for p in pairs], dtype=pl.Float64)
        self._segment_bounds = pl.DataFrame(columns).with_columns(pl.col(_LABEL_COL).cast(label_dtype))

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Handle outliers based on fitted percentile thresholds.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            Transformed time series.

        """
        data_cols = [c for c in X.columns if c != "time"]
        if self.segmenter_ is None:
            return _apply_outlier_handling(
                X,
                data_cols,
                self.strategy,
                lambda col_name: _literal_bounds(*self.thresholds_.get(col_name, (None, None))),
            )

        labels = _check_segmenter_label_length(self.segmenter_.compute_labels(X["time"]), X)
        X = X.with_columns(labels.alias(_LABEL_COL))
        inside = pl.col(_LABEL_COL).is_not_null()

        if not self.by_segment:
            return _apply_outlier_handling(
                X,
                data_cols,
                self.strategy,
                lambda col_name: _literal_bounds(*self.thresholds_.get(col_name, (None, None)), inside=inside),
            )

        X = X.join(self._segment_bounds, on=_LABEL_COL, how="left", maintain_order="left")

        def get_bounds(col_name: str) -> _Bounds:
            """Return per-row bounds: the label's thresholds, pooled for unseen labels, none outside."""
            pooled_low, pooled_high = self.thresholds_.get(col_name, (None, None))
            if f"__low__{col_name}" not in X.columns:
                return _literal_bounds(pooled_low, pooled_high, inside=inside)

            def side(fitted: str, pooled: float | None) -> pl.Expr:
                """Return the bound from the joined label column, falling back to the pooled value."""
                # Labels unseen at fit miss the join (null) and take the pooled value.
                return pl.when(inside).then(pl.col(fitted).fill_null(pl.lit(pooled, dtype=pl.Float64)))

            return (
                None if self.low is None else side(f"__low__{col_name}", pooled_low),
                None if self.high is None else side(f"__high__{col_name}", pooled_high),
            )

        return _apply_outlier_handling(X, data_cols, self.strategy, get_bounds)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : list of str or None, default=None
            Column names of the input features.  If ``None``, uses the
            feature names seen during ``fit``.

        Returns
        -------
        list of str
            Output feature names after transformation.

        """
        check_is_fitted(self, ["feature_names_in_"])
        input_features = _check_feature_names_in(self, input_features)
        return list(input_features)
