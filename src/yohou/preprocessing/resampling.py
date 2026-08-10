"""Time series resampling transformers for frequency conversion."""

from datetime import datetime
from typing import Literal, cast

import polars as pl
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseActualTransformer
from yohou.utils._compat import StrOptions, _check_feature_names_in
from yohou.utils.validation import (
    check_interval_consistency,
    interval_to_timedelta,
    parse_interval,
    representative_interval,
)

__all__ = ["Downsampler", "Upsampler"]


class Downsampler(BaseActualTransformer):
    """Downsample time series to a lower frequency using aggregation.

    Reduces the frequency of time series data by grouping consecutive time
    points into bins and applying an aggregation function. Uses polars'
    `group_by_dynamic` for efficient windowed aggregation.

    Because `group_by_dynamic` bins by wall-clock windows, the input does not need a
    uniform grid: `Downsampler` declares `accepts_irregular_grid=True`, so a jittered
    or gapped sub-hourly feed is accepted at fit and transform (the strict
    interval-consistency check is skipped and a representative median interval is
    recorded for the `target >= input` guard). Behavior on a uniform grid is unchanged.

    Accepting a gapped input axis means the output can carry gaps too: a window with
    no rows produces no bin, so a gap in the input becomes a gap in the output. A
    downstream transformer that requires a uniform grid may not notice, because the
    strict interval check tolerates a sub-day delta spread and will infer an interval
    from a gapped axis rather than reject it. A lag or rolling transformer placed after
    a `Downsampler` on gapped input therefore computes over rows that are not the
    real-time distance apart that its parameters imply. Fill or validate the gaps
    (see `SimpleTimeImputer`, `Upsampler`) before an order-dependent step.

    Parameters
    ----------
    interval : str, default='1h'
        Target time interval (e.g., "1h", "1d", "5m", "30s").
        Uses polars duration string syntax. Must be greater than or equal to
        the input data's interval.
    aggregation : {"mean", "sum", "min", "max", "first", "last", "median", "std"}, default="mean"
        Aggregation function to apply within each time bin:
        - "mean": Average values in each bin
        - "sum": Sum values in each bin
        - "min": Minimum value in each bin
        - "max": Maximum value in each bin
        - "first": First value in each bin
        - "last": Last value in each bin
        - "median": Median value in each bin
        - "std": Sample standard deviation of the values in each bin, which
          measures how much the series moved *within* the bin rather than where
          it sat. A bin holding fewer than two values yields null, since a
          spread is undefined for a single point.
    closed : {"left", "right"}, default="left"
        Which side of the interval is closed.
    label : {"left", "right"}, default="left"
        Which side of the interval to use as the label for each bin.
    include_boundaries : bool, default=False
        Whether to include the interval boundaries in output.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : list of str
        Names of features seen during fit.
    input_interval_ : timedelta or None
        Detected time interval of input data.
    target_interval_ : timedelta or None
        Target time interval.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> from yohou.preprocessing import Downsampler

    >>> # Create hourly data
    >>> times = [datetime(2020, 1, 1) + timedelta(hours=i) for i in range(24)]
    >>> X = pl.DataFrame({"time": times, "value": list(range(24))})

    >>> # Downsample to daily (24h) using mean aggregation
    >>> downsampler = Downsampler(interval="1d", aggregation="mean")
    >>> downsampler.fit(X)
    Downsampler(interval='1d')
    >>> X_daily = downsampler.transform(X)
    >>> len(X_daily) == 1  # Single day
    True

    See Also
    --------
    - [`Upsampler`][yohou.preprocessing.resampling.Upsampler] : Upsample time series to higher frequency.

    """

    _valid_aggregations = {"mean", "sum", "min", "max", "first", "last", "median", "std"}

    _parameter_constraints: dict = {
        "interval": [str],
        "aggregation": [StrOptions(_valid_aggregations)],
        "closed": [StrOptions({"left", "right"})],
        "label": [StrOptions({"left", "right"})],
        "include_boundaries": ["boolean"],
    }

    # Bins via group_by_dynamic, which is correct on a non-uniform grid, so it opts
    # into the irregular-grid contract: a jittered or gapped sub-hourly feed can be
    # downsampled without first being placed on a strict uniform grid.
    _tags = {"stateful": False, "accepts_irregular_grid": True}

    def __init__(
        self,
        interval: str = "1h",
        aggregation: Literal["mean", "sum", "min", "max", "first", "last", "median", "std"] = "mean",
        closed: Literal["left", "right"] = "left",
        label: Literal["left", "right"] = "left",
        include_boundaries: bool = False,
    ):
        self.interval = interval
        self.aggregation = aggregation
        self.closed = closed
        self.label = label
        self.include_boundaries = include_boundaries

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit the internal model."""
        # Detect the input interval with the representative (frequency-weighted median)
        # measure, which tolerates a jittered or gapped grid and agrees with the strict
        # check on a uniform one. The strict check is not tried first: on a sub-day axis
        # it medians the *unique* deltas, so a few gaps skew it above the true cadence
        # (a 5m feed with two gaps reads as 10m), and it succeeds rather than raising,
        # which would wrongly reject a target at the feed's real cadence in the guard
        # below. group_by_dynamic bins either way, so the transform is unaffected.
        self.input_interval_str_ = representative_interval(X)
        self.input_interval_ = interval_to_timedelta(self.input_interval_str_)
        self.target_interval_ = interval_to_timedelta(self.interval)

        # Normalize interval to polars-native format (e.g. "30min" → "30m")
        _mult, _unit = parse_interval(self.interval)
        self.polars_interval_ = f"{_mult}{_unit}"

        # Validate: target must be >= input for downsampling
        if (
            self.input_interval_ is not None
            and self.target_interval_ is not None
            and self.target_interval_ < self.input_interval_
        ):
            msg = (
                f"Target interval ({self.interval}) is smaller than input interval "
                f"({self.input_interval_str_}). Use Upsampler for increasing frequency."
            )
            raise ValueError(msg)

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Downsample time series to target frequency.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            Downsampled time series.

        """
        # Get data columns
        data_cols = [c for c in X.columns if c != "time"]

        # Build aggregation expressions
        agg_exprs = []
        for col in data_cols:
            if self.aggregation == "mean":
                agg_exprs.append(pl.col(col).mean())
            elif self.aggregation == "sum":
                agg_exprs.append(pl.col(col).sum())
            elif self.aggregation == "min":
                agg_exprs.append(pl.col(col).min())
            elif self.aggregation == "max":
                agg_exprs.append(pl.col(col).max())
            elif self.aggregation == "first":
                agg_exprs.append(pl.col(col).first())
            elif self.aggregation == "last":
                agg_exprs.append(pl.col(col).last())
            elif self.aggregation == "median":
                agg_exprs.append(pl.col(col).median())
            elif self.aggregation == "std":
                agg_exprs.append(pl.col(col).std())

        result = (
            X
            .sort("time")
            .group_by_dynamic(
                "time",
                every=self.polars_interval_,
                closed=self.closed,
                label=self.label,
                include_boundaries=self.include_boundaries,
            )
            .agg(agg_exprs)
        )

        # group_by_dynamic prepends _lower_boundary/_upper_boundary columns when
        # include_boundaries=True; drop them so the output schema is just "time"
        # plus the original feature columns (they are not reported by
        # get_feature_names_out and duplicate the "time" anchor).
        if self.include_boundaries:
            result = result.drop(["_lower_boundary", "_upper_boundary"], strict=False)

        return result

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


class Upsampler(BaseActualTransformer):
    """Upsample time series to a higher frequency using interpolation.

    Increases the frequency of time series data by creating new time points
    and filling values using interpolation. Supports various interpolation
    methods including linear, nearest neighbor, and forward/backward fill.

    Parameters
    ----------
    interval : str, default='1h'
        Target time interval (e.g., "1h", "1d", "5m", "30s").
        Uses polars duration string syntax. Must be smaller than the input
        data's interval.
    interpolation : {"linear", "nearest", "forward", "backward"}, default="linear"
        Interpolation method to fill new time points:
        - "linear": Linear interpolation between known points
        - "nearest": Use the known value closest in time (ties go to the preceding value)
        - "forward": Forward fill (carry last observation forward)
        - "backward": Backward fill (carry next observation backward)

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : list of str
        Names of features seen during fit.
    input_interval_ : timedelta or None
        Detected time interval of input data.
    target_interval_ : timedelta or None
        Target time interval.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> from yohou.preprocessing import Upsampler

    >>> # Create daily data
    >>> times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(7)]
    >>> X = pl.DataFrame({"time": times, "value": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]})

    >>> # Upsample to hourly using linear interpolation
    >>> upsampler = Upsampler(interval="12h", interpolation="linear")
    >>> upsampler.fit(X)
    Upsampler(interval='12h')
    >>> X_hourly = upsampler.transform(X)
    >>> len(X_hourly) > len(X)  # More time points
    True

    See Also
    --------
    - [`Downsampler`][yohou.preprocessing.resampling.Downsampler] : Downsample time series to lower frequency.

    """

    _valid_interpolations = {"linear", "nearest", "forward", "backward"}

    _parameter_constraints: dict = {
        "interval": [str],
        "interpolation": [StrOptions(_valid_interpolations)],
    }

    _tags = {"stateful": False}

    def __init__(
        self,
        interval: str = "1h",
        interpolation: Literal["linear", "nearest", "forward", "backward"] = "linear",
    ):
        self.interval = interval
        self.interpolation = interpolation

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit the internal model."""
        # Detect input interval
        self.input_interval_str_ = check_interval_consistency(X)
        self.input_interval_ = interval_to_timedelta(self.input_interval_str_)
        self.target_interval_ = interval_to_timedelta(self.interval)

        # Normalize interval to polars-native format (e.g. "30min" → "30m")
        _mult, _unit = parse_interval(self.interval)
        self.polars_interval_ = f"{_mult}{_unit}"

        # Validate: target must be <= input for upsampling
        if (
            self.input_interval_ is not None
            and self.target_interval_ is not None
            and self.target_interval_ > self.input_interval_
        ):
            msg = (
                f"Target interval ({self.interval}) is larger than input interval "
                f"({self.input_interval_str_}). Use Downsampler for decreasing frequency."
            )
            raise ValueError(msg)

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Upsample time series to target frequency.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            Upsampled time series.

        """
        # Create new time range
        time_min = X["time"].min()
        time_max = X["time"].max()

        if time_min is None or time_max is None:
            raise ValueError("Upsampler received an empty time series.")

        # Generate new timestamps (cast to datetime for type narrowing)
        new_times = pl.datetime_range(
            cast(datetime, time_min), cast(datetime, time_max), interval=self.polars_interval_, eager=True
        )
        new_df = pl.DataFrame({"time": new_times})

        # Join with original data
        X_sorted = X.sort("time")
        joined = new_df.join(X_sorted, on="time", how="left")

        # Interpolate based on method
        data_cols = list(self.feature_names_in_)

        if self.interpolation == "linear":
            joined = joined.with_columns([pl.col(col).interpolate() for col in data_cols])
        elif self.interpolation == "nearest":
            # Fill each new timestamp with the value closest in time, comparing
            # distance to the previous and next known observations. Ties go to
            # the trailing (forward) anchor.
            for col in data_cols:
                value = pl.col(col)
                known_time = pl.when(value.is_not_null()).then(pl.col("time")).otherwise(None)
                prev_val = value.forward_fill()
                next_val = value.backward_fill()
                prev_time = known_time.forward_fill()
                next_time = known_time.backward_fill()
                joined = joined.with_columns(
                    pl
                    .when(value.is_not_null())
                    .then(value)
                    .when(prev_val.is_null())
                    .then(next_val)
                    .when(next_val.is_null())
                    .then(prev_val)
                    .when((next_time - pl.col("time")) < (pl.col("time") - prev_time))
                    .then(next_val)
                    .otherwise(prev_val)
                    .alias(col)
                )
        elif self.interpolation == "forward":
            joined = joined.with_columns([pl.col(col).forward_fill() for col in data_cols])
        elif self.interpolation == "backward":
            joined = joined.with_columns([pl.col(col).backward_fill() for col in data_cols])

        return joined

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
