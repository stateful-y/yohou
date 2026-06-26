"""Outlier handling transformers for time series data."""

import numbers
from collections.abc import Callable

import numpy as np
import polars as pl
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseTransformer
from yohou.utils._compat import Interval, StrOptions, _check_feature_names_in

__all__ = ["OutlierPercentileHandler", "OutlierThresholdHandler"]


def _apply_outlier_handling(
    X: pl.DataFrame,
    strategy: str,
    get_thresholds: Callable[[str], tuple[float | None, float | None]],
) -> pl.DataFrame:
    """Apply outlier handling to a DataFrame.

    Parameters
    ----------
    X : pl.DataFrame
        Input time series with "time" column.
    strategy : str
        Either "clip" or "nan".
    get_thresholds : callable
        Function that takes a column name and returns (low, high) thresholds.

    Returns
    -------
    pl.DataFrame
        Time series with outliers handled.

    """
    data_cols = [c for c in X.columns if c != "time"]
    exprs = [pl.col("time")]

    for col_name in data_cols:
        col = pl.col(col_name)
        low_val, high_val = get_thresholds(col_name)

        if strategy == "clip":
            if low_val is not None:
                col = col.clip(lower_bound=low_val)
            if high_val is not None:
                col = col.clip(upper_bound=high_val)
        else:  # nan
            if low_val is not None:
                col = pl.when(col < low_val).then(None).otherwise(col)
            if high_val is not None:
                col = pl.when(col > high_val).then(None).otherwise(col)

        exprs.append(col.alias(col_name))

    return X.select(exprs)


class OutlierThresholdHandler(BaseTransformer):
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

    Attributes
    ----------
    low_ : float or None
        Validated lower threshold.
    high_ : float or None
        Validated upper threshold.

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

    See Also
    --------
    - [`OutlierPercentileHandler`][yohou.preprocessing.outlier.OutlierPercentileHandler] : Handle outliers based on percentiles.

    """

    _valid_strategies = {"clip", "nan"}

    _parameter_constraints: dict = {
        "low": [numbers.Real, None],
        "high": [numbers.Real, None],
        "strategy": [StrOptions(_valid_strategies)],
    }

    _tags = {"stateful": False, "invertible": False}

    def __init__(
        self,
        low: float | None = None,
        high: float | None = None,
        strategy: str = "clip",
    ):
        self.low = low
        self.high = high
        self.strategy = strategy

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit the internal model."""
        # Validate threshold ordering before assigning any fitted attributes so
        # a failed fit does not leave the transformer in a half-fitted state.
        if self.low is not None and self.high is not None and self.low > self.high:
            msg = f"low ({self.low}) must be <= high ({self.high})"
            raise ValueError(msg)

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
        return _apply_outlier_handling(
            X,
            self.strategy,
            lambda _col_name: (self.low_, self.high_),
        )

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


class OutlierPercentileHandler(BaseTransformer):
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

    Attributes
    ----------
    thresholds_ : dict
        Dictionary mapping column names to (low_value, high_value) tuples.

    Raises
    ------
    ValueError
        At fit time if both ``low`` and ``high`` are provided and ``low`` > ``high``.

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

    See Also
    --------
    - [`OutlierThresholdHandler`][yohou.preprocessing.outlier.OutlierThresholdHandler] : Handle outliers based on fixed thresholds.

    """

    _valid_strategies = {"clip", "nan"}

    _parameter_constraints: dict = {
        "low": [Interval(numbers.Real, 0, 100, closed="both"), None],
        "high": [Interval(numbers.Real, 0, 100, closed="both"), None],
        "strategy": [StrOptions(_valid_strategies)],
    }

    _tags = {"stateful": False, "invertible": False}

    def __init__(
        self,
        low: float | None = None,
        high: float | None = None,
        strategy: str = "clip",
    ):
        self.low = low
        self.high = high
        self.strategy = strategy

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit the internal model."""
        # Validate percentile ordering
        if self.low is not None and self.high is not None and self.low > self.high:
            msg = f"low ({self.low}) must be <= high ({self.high})"
            raise ValueError(msg)

        # Compute thresholds for each column
        data_cols = [c for c in X.columns if c != "time"]
        self.thresholds_: dict[str, tuple[float | None, float | None]] = {}

        for col_name in data_cols:
            col_data = X[col_name].drop_nulls().drop_nans().to_numpy()

            low_val = None
            high_val = None

            if len(col_data) > 0:
                if self.low is not None:
                    low_val = float(np.percentile(col_data, self.low))
                if self.high is not None:
                    high_val = float(np.percentile(col_data, self.high))

            self.thresholds_[col_name] = (low_val, high_val)

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
        return _apply_outlier_handling(
            X,
            self.strategy,
            lambda col_name: self.thresholds_.get(col_name, (None, None)),
        )

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
