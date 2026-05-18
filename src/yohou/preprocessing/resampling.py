"""Time series resampling transformers for frequency conversion."""

from datetime import datetime
from typing import Literal, cast

import polars as pl
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseTransformer
from yohou.utils._compat import StrOptions, _check_feature_names_in
from yohou.utils.validation import check_interval_consistency, interval_to_timedelta, parse_interval

__all__ = ["Downsampler", "Upsampler"]


class Downsampler(BaseTransformer):
    """Downsample time series to a lower frequency using aggregation.

    Reduces the frequency of time series data by grouping consecutive time
    points into bins and applying an aggregation function. Uses polars'
    `group_by_dynamic` for efficient windowed aggregation.

    Parameters
    ----------
    interval : str
        Target time interval (e.g., "1h", "1d", "5m", "30s").
        Uses polars duration string syntax. Must be larger than the input
        data's interval.
    aggregation : {"mean", "sum", "min", "max", "first", "last", "median"}, default="mean"
        Aggregation function to apply within each time bin:
        - "mean": Average values in each bin
        - "sum": Sum values in each bin
        - "min": Minimum value in each bin
        - "max": Maximum value in each bin
        - "first": First value in each bin
        - "last": Last value in each bin
        - "median": Median value in each bin
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

    _valid_aggregations = {"mean", "sum", "min", "max", "first", "last", "median"}

    _parameter_constraints: dict = {
        "interval": [str],
        "aggregation": [StrOptions(_valid_aggregations)],
        "closed": [StrOptions({"left", "right"})],
        "label": [StrOptions({"left", "right"})],
        "include_boundaries": ["boolean"],
    }

    _tags = {"stateful": False}

    def __init__(
        self,
        interval: str = "1h",
        aggregation: Literal["mean", "sum", "min", "max", "first", "last", "median"] = "mean",
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
        # Detect input interval
        self.input_interval_str_ = check_interval_consistency(X)
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


class Upsampler(BaseTransformer):
    """Upsample time series to a higher frequency using interpolation.

    Increases the frequency of time series data by creating new time points
    and filling values using interpolation. Supports various interpolation
    methods including linear, nearest neighbor, and forward/backward fill.

    Parameters
    ----------
    interval : str
        Target time interval (e.g., "1h", "1d", "5m", "30s").
        Uses polars duration string syntax. Must be smaller than the input
        data's interval.
    interpolation : {"linear", "nearest", "forward", "backward"}, default="linear"
        Interpolation method to fill new time points:
        - "linear": Linear interpolation between known points
        - "nearest": Use nearest known value (forward then backward fill)
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

        # Assert non-null (X should have at least one row after fit validation)
        assert time_min is not None and time_max is not None, "Empty time series"

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
            for col in data_cols:
                joined = joined.with_columns(pl.col(col).interpolate())
        elif self.interpolation == "nearest":
            # Forward fill then backward fill for nearest approximation
            for col in data_cols:
                joined = joined.with_columns(pl.col(col).fill_null(strategy="forward").fill_null(strategy="backward"))
        elif self.interpolation == "forward":
            for col in data_cols:
                joined = joined.with_columns(pl.col(col).forward_fill())
        elif self.interpolation == "backward":
            for col in data_cols:
                joined = joined.with_columns(pl.col(col).backward_fill())

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
