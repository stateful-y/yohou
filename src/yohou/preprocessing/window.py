"""Implementation of window transformations."""

import numbers
from collections.abc import Callable

import numpy as np
import polars as pl
from pydantic import StrictInt
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseTransformer
from yohou.utils import tabularize, validate_transformer_data
from yohou.utils._compat import Interval, _check_feature_names_in

__all__ = [
    "ExponentialMovingAverage",
    "LagTransformer",
    "MeanLagTransformer",
    "RollingStatisticsTransformer",
    "SlidingWindowFunctionTransformer",
]


class LagTransformer(BaseTransformer):
    """Create lagged features from time series data.

    Creates lagged versions of each feature column, where each lag shifts the
    data by a specified number of time steps. This is essential for time series
    forecasting using supervised learning approaches.

    Parameters
    ----------
    lag : int >= 1 or list of ints >= 1, default=1
        Lag(s) to create. Can be a single integer or a list of integers.
        Each lag value must be >= 1.

    Attributes
    ----------
    lags_ : list of int
        Effective list of lags used for transformation.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.preprocessing import LagTransformer

    >>> # Create sample data
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 11)],
    ...     "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    ... })

    >>> # Create lag-1 and lag-2 features
    >>> transformer = LagTransformer(lag=[1, 2])
    >>> transformer.fit(X)  # doctest: +ELLIPSIS
    LagTransformer(...)
    >>> X_lagged = transformer.transform(X)
    >>> X_lagged.columns
    ['time', 'value_lag_1', 'value_lag_2']
    >>> len(X_lagged)  # First 2 rows dropped (max(lag) = 2)
    8

    See Also
    --------
    - [`MeanLagTransformer`][yohou.preprocessing.window.MeanLagTransformer] : Averages multiple lag multiples into a single feature.
    - [`RollingStatisticsTransformer`][yohou.preprocessing.window.RollingStatisticsTransformer] : Compute rolling statistics (mean, std, etc.).
    - [`SlidingWindowFunctionTransformer`][yohou.preprocessing.window.SlidingWindowFunctionTransformer] : Apply custom functions to sliding windows.
    - `tabularize` : Underlying tabularization function.

    Notes
    -----
    Lag features are created using ``yohou.utils.tabularization.tabularize``,
    which shifts each numeric column by the specified number of time steps.
    The first ``max(lags)`` rows are dropped because they contain incomplete
    lag values, setting ``observation_horizon = max(lags)``.

    When used inside a pipeline with ``observe``/``rewind``, the observation
    buffer retains enough history to produce lag features without data loss
    on subsequent ``transform`` calls.

    """

    _parameter_constraints: dict = {
        "lag": [Interval(numbers.Integral, 1, None, closed="left"), list],
    }

    _tags = {"stateful": True}

    def __init__(self, lag: StrictInt | list[StrictInt] = 1):
        self.lag = lag

    @property
    def observation_horizon(self) -> int:  # noqa: D102
        """Return the number of past observations needed."""
        lags = self.lag if isinstance(self.lag, list) else [self.lag]
        return max(lags)

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit the internal model."""
        self.lags_: list[int] = self.lag if isinstance(self.lag, list) else [self.lag]

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input time series.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            Transformed time series with a ``"time"`` column and transformed
            value columns.

        """
        X_t = tabularize(X, self.lags_)

        return X_t

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Column names of the input features.  If ``None``, uses the
            feature names seen during ``fit``.

        Returns
        -------
        list of str
            Output feature names after transformation.

        """
        input_features = _check_feature_names_in(self, input_features)
        feature_names = [f"{col}_lag_{lag}" for col in input_features for lag in self.lags_]

        arr: list[str] = np.asarray(feature_names, dtype=object).tolist()
        return arr


class MeanLagTransformer(BaseTransformer):
    """Create mean-lagged features by averaging across lag multiples.

    For each input column and each base lag ``k``, this transformer computes
    the arithmetic mean of the column shifted by ``k, 2k, ..., n_lags * k``
    time steps.  This captures averaged seasonal patterns as features for
    supervised learning.

    Parameters
    ----------
    lag : int >= 1 or list of ints >= 1, default=1
        Base lag(s) to create. Can be a single integer or a list of integers.
        Each lag value must be >= 1.
    n_lags : int >= 1, default=1
        Number of lag multiples to average. For a base lag ``k`` the shifts
        ``k, 2k, ..., n_lags * k`` are averaged. When ``n_lags=1`` the output
        is a single shift (equivalent to ``LagTransformer``).

    Attributes
    ----------
    lags_ : list of int
        Effective list of base lags used for transformation.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.preprocessing import MeanLagTransformer

    >>> # Create sample data
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 13)],
    ...     "value": list(range(12)),
    ... })

    >>> # Average lag-3 and lag-6 into a single feature
    >>> transformer = MeanLagTransformer(lag=3, n_lags=2)
    >>> transformer.fit(X)  # doctest: +ELLIPSIS
    MeanLagTransformer(...)
    >>> X_t = transformer.transform(X)
    >>> X_t.columns
    ['time', 'value_mean_lag_3']
    >>> len(X_t)  # First 6 rows dropped (max(lag) * n_lags = 6)
    6

    See Also
    --------
    - [`LagTransformer`][yohou.preprocessing.window.LagTransformer] : Creates individual lag features without averaging.
    - [`RollingStatisticsTransformer`][yohou.preprocessing.window.RollingStatisticsTransformer] : Compute rolling statistics over consecutive windows.

    Notes
    -----
    For a base lag ``k`` and ``n_lags=3``, the output at time ``t`` is
    ``mean(x[t-k], x[t-2k], x[t-3k])``.  This differs from
    ``RollingStatisticsTransformer`` which uses consecutive time steps
    rather than seasonal multiples.

    The first ``max(lags) * n_lags`` rows are dropped because they contain
    incomplete lookback windows, setting
    ``observation_horizon = max(lags) * n_lags``.

    When ``n_lags=1`` the output values are identical to ``LagTransformer``
    and the original column dtype is preserved.  When ``n_lags > 1`` the
    averaging produces ``Float64`` columns.

    Output column names follow the pattern ``{input_col}_mean_lag_{k}``.

    """

    _parameter_constraints: dict = {
        **BaseTransformer._parameter_constraints,
        "lag": [Interval(numbers.Integral, 1, None, closed="left"), list],
        "n_lags": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    _tags = {"stateful": True}

    def __init__(self, lag: StrictInt | list[StrictInt] = 1, n_lags: StrictInt = 1):
        self.lag = lag
        self.n_lags = n_lags

    @property
    def observation_horizon(self) -> int:  # noqa: D102
        """Return the number of past observations needed."""
        lags = self.lag if isinstance(self.lag, list) else [self.lag]
        return max(lags) * self.n_lags

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit the internal model."""
        self.lags_: list[int] = self.lag if isinstance(self.lag, list) else [self.lag]

    def transform(self, X: pl.DataFrame, **params) -> pl.DataFrame:
        """Transform the input time series.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Transformed time series with a ``"time"`` column and
            ``{col}_mean_lag_{k}`` columns.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)

        data_cols = [c for c in X.columns if c != "time"]

        exprs: list[pl.Expr] = [pl.col("time")]
        for col in data_cols:
            for lag in self.lags_:
                if self.n_lags == 1:
                    exprs.append(pl.col(col).shift(lag).alias(f"{col}_mean_lag_{lag}"))
                else:
                    shifted = [pl.col(col).shift(lag * j) for j in range(1, self.n_lags + 1)]
                    exprs.append(pl.mean_horizontal(*shifted).alias(f"{col}_mean_lag_{lag}"))

        X_t = X.select(exprs)
        X_t = X_t[self._observation_horizon :]

        return X_t

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Column names of the input features.  If ``None``, uses the
            feature names seen during ``fit``.

        Returns
        -------
        list of str
            Output feature names after transformation.

        """
        input_features = _check_feature_names_in(self, input_features)
        feature_names = [f"{col}_mean_lag_{lag}" for col in input_features for lag in self.lags_]

        arr: list[str] = np.asarray(feature_names, dtype=object).tolist()
        return arr


class SlidingWindowFunctionTransformer(BaseTransformer):
    """Transform time series by applying a function over sliding windows.

    This transformer applies a user-defined function to sliding windows of the
    input time series. It is useful for computing rolling aggregates, custom
    statistics, or any windowed transformation.

    The function receives a polars DataFrame containing `window_size` rows
    (one window) and should return a scalar or a 1D array for that window.

    Parameters
    ----------
    func : callable
        Function to apply to each sliding window. It receives a polars DataFrame
        with shape (window_size, n_features + 1); the window is a full slice of the
        input and includes the "time" column, which the function must handle or
        exclude explicitly (see the example below). It should return:
        - A scalar (applied to all columns)
        - A dict mapping column names to scalars
        - A numpy array of shape (n_features,)
        - Any other value castable to float (applied to all columns)
    window_size : int, default=1
        Size of the sliding window. Must be >= 1.
    kw_args : dict or None, default=None
        Dictionary of additional keyword arguments to pass to func.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : list of str
        Names of features seen during fit.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> import numpy as np
    >>> from yohou.preprocessing import SlidingWindowFunctionTransformer

    >>> times = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 1, 10), interval="1d", eager=True
    ... )
    >>> X = pl.DataFrame({"time": times, "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})

    >>> # Compute rolling mean with window size 3
    >>> def rolling_mean(window):
    ...     return window.select(pl.all().exclude("time").mean()).to_numpy().flatten()
    >>> transformer = SlidingWindowFunctionTransformer(func=rolling_mean, window_size=3)
    >>> transformer.fit(X)  # doctest: +ELLIPSIS
    SlidingWindowFunctionTransformer(...)
    >>> X_t = transformer.transform(X)
    >>> len(X_t)  # Original length minus (window_size - 1)
    8

    See Also
    --------
    - [`LagTransformer`][yohou.preprocessing.window.LagTransformer] : Create lagged features.
    - [`RollingStatisticsTransformer`][yohou.preprocessing.window.RollingStatisticsTransformer] : Pre-built rolling statistics.
    - `FunctionTransformer` : Apply function element-wise.

    """

    _parameter_constraints: dict = {
        "func": [callable],
        "window_size": [Interval(numbers.Integral, 1, None, closed="left")],
        "kw_args": [dict, None],
    }

    _tags = {"stateful": True}

    def __init__(
        self,
        func: Callable,
        window_size: int = 1,
        *,
        kw_args: dict | None = None,
    ):
        self.func = func
        self.window_size = window_size
        self.kw_args = kw_args

    @property
    def observation_horizon(self) -> int:  # noqa: D102
        """Return the number of past observations needed."""
        return self.window_size - 1

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform X by applying func to sliding windows.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            Transformed time series with a ``"time"`` column and transformed
            value columns.

        """
        # Get data columns (excluding time)
        data_cols = [c for c in X.columns if c != "time"]

        # Apply function over sliding windows
        kwargs = self.kw_args if self.kw_args is not None else {}
        results = []

        n_rows = len(X)
        for i in range(n_rows - self.window_size + 1):
            window = X[i : i + self.window_size]
            result = self.func(window, **kwargs)

            # Handle different return types
            if isinstance(result, dict):
                results.append(result)
            elif np.isscalar(result):
                results.append(dict.fromkeys(data_cols, result))
            elif isinstance(result, np.ndarray):
                results.append({col: result[j] for j, col in enumerate(data_cols)})
            else:
                results.append({col: float(result) for col in data_cols})

        # The output time is the last point of each window, i.e. a single slice
        # of the original time column rather than per-iteration scalar lookups.
        result_df = pl.DataFrame(results)
        time_df = pl.DataFrame({"time": X["time"][self.window_size - 1 :]})

        return pl.concat([time_df, result_df], how="horizontal")

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
        input_features = _check_feature_names_in(self, input_features)
        arr: list[str] = np.asarray(input_features, dtype=object).tolist()
        return arr


class RollingStatisticsTransformer(BaseTransformer):
    """Compute rolling window statistics for time series.

    This transformer computes one or more rolling statistics (mean, std, min,
    max, median, quantiles) over sliding windows. It is a convenience wrapper
    around polars rolling functions with a sklearn-compatible interface.

    Parameters
    ----------
    window_size : int, default=7
        Size of the rolling window. Must be >= 1.
    statistics : str or list of str, default="mean"
        Statistic(s) to compute. Options:
        - "mean": Rolling mean
        - "std": Rolling standard deviation
        - "min": Rolling minimum
        - "max": Rolling maximum
        - "median": Rolling median
        - "sum": Rolling sum
        - "var": Rolling variance
        - "q25": 25th percentile
        - "q75": 75th percentile

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : list of str
        Names of features seen during fit.
    statistics_ : list of str
        Effective list of statistics to compute.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.preprocessing import RollingStatisticsTransformer

    >>> times = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 1, 10), interval="1d", eager=True
    ... )
    >>> X = pl.DataFrame({"time": times, "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})

    >>> # Compute rolling mean with window size 3
    >>> transformer = RollingStatisticsTransformer(window_size=3, statistics="mean")
    >>> transformer.fit(X)
    RollingStatisticsTransformer(window_size=3)
    >>> X_t = transformer.transform(X)
    >>> len(X_t)
    8
    >>> "value_mean" in X_t.columns
    True

    >>> # Multiple statistics
    >>> transformer = RollingStatisticsTransformer(window_size=3, statistics=["mean", "std", "min", "max"])
    >>> transformer.fit(X)  # doctest: +ELLIPSIS
    RollingStatisticsTransformer(...)
    >>> X_t = transformer.transform(X)
    >>> len([c for c in X_t.columns if c != "time"])
    4

    See Also
    --------
    - [`SlidingWindowFunctionTransformer`][yohou.preprocessing.window.SlidingWindowFunctionTransformer] : Apply custom function over windows.
    - [`LagTransformer`][yohou.preprocessing.window.LagTransformer] : Create lagged features.

    Notes
    -----
    Rolling statistics are computed via native polars rolling expressions
    (``rolling_mean``, ``rolling_std``, etc.), which are significantly faster
    than Python-level iteration. Quantile statistics (``q25``, ``q75``) use
    ``rolling_quantile`` with linear interpolation.

    The first ``window_size - 1`` rows produce nulls from incomplete windows
    and are dropped from the output, setting
    ``observation_horizon = window_size - 1``.

    Output column names follow the pattern ``{input_col}_{statistic}``,
    e.g., ``"value_mean"``, ``"value_std"``.

    """

    _valid_statistics = {"mean", "std", "min", "max", "median", "sum", "var", "q25", "q75"}

    _parameter_constraints: dict = {
        "window_size": [Interval(numbers.Integral, 1, None, closed="left")],
        "statistics": [str, list],
    }

    _tags = {"stateful": True}

    def __init__(
        self,
        window_size: int = 7,
        statistics: str | list[str] = "mean",
    ):
        self.window_size = window_size
        self.statistics = statistics

    @property
    def observation_horizon(self) -> int:  # noqa: D102
        """Return the number of past observations needed."""
        return self.window_size - 1

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit the internal model."""
        # Normalize statistics to list
        if isinstance(self.statistics, str):
            self.statistics_ = [self.statistics]
        else:
            self.statistics_ = list(self.statistics)

        # Validate statistics
        invalid = set(self.statistics_) - self._valid_statistics
        if invalid:
            msg = f"Invalid statistics: {invalid}. Valid options: {self._valid_statistics}"
            raise ValueError(msg)

    def _apply_rolling_stat(self, col: pl.Expr, stat: str) -> pl.Expr:
        """Apply a rolling statistic to a column expression.

        Parameters
        ----------
        col : pl.Expr
            Column expression.
        stat : str
            Statistic name.

        Returns
        -------
        pl.Expr
            Rolling statistic expression.

        """
        if stat == "mean":
            return col.rolling_mean(self.window_size)
        elif stat == "std":
            return col.rolling_std(self.window_size)
        elif stat == "min":
            return col.rolling_min(self.window_size)
        elif stat == "max":
            return col.rolling_max(self.window_size)
        elif stat == "median":
            return col.rolling_median(self.window_size)
        elif stat == "sum":
            return col.rolling_sum(self.window_size)
        elif stat == "var":
            return col.rolling_var(self.window_size)
        elif stat == "q25":
            return col.rolling_quantile(0.25, window_size=self.window_size)
        elif stat == "q75":
            return col.rolling_quantile(0.75, window_size=self.window_size)
        raise AssertionError("unreachable: statistic validated in _fit")

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform X by computing rolling statistics.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            Transformed time series with a ``"time"`` column and transformed
            value columns.

        """
        # Get data columns
        data_cols = [c for c in X.columns if c != "time"]

        # Build expressions for all statistics
        exprs = [pl.col("time")]
        for col_name in data_cols:
            for stat in self.statistics_:
                col_expr = pl.col(col_name)
                stat_expr = self._apply_rolling_stat(col_expr, stat)
                exprs.append(stat_expr.alias(f"{col_name}_{stat}"))

        X_t = X.select(exprs)

        # Drop first observation_horizon rows (contain nulls from incomplete windows)
        if self._observation_horizon > 0:
            X_t = X_t[self._observation_horizon :]

        return X_t

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
        check_is_fitted(self, ["statistics_"])
        input_features = _check_feature_names_in(self, input_features)
        feature_names = [f"{col}_{stat}" for col in input_features for stat in self.statistics_]
        arr: list[str] = np.asarray(feature_names, dtype=object).tolist()
        return arr


class ExponentialMovingAverage(BaseTransformer):
    """Exponentially Weighted Moving Average (EWMA) transformer.

    Computes the exponentially weighted moving average for time series data.
    The EWMA gives more weight to recent observations with exponentially
    decreasing weights for older observations.

    Parameters
    ----------
    alpha : float
        Smoothing factor (0 < alpha <= 1). Higher values give more weight
        to recent observations.
    adjust : bool, default=True
        If True, uses adjusted weights (divide by decaying adjustment factor).
        If False, uses standard exponential decay.
    ignore_nulls : bool, default=True
        If True, ignore null values when computing EWMA.
        If False, propagate null values.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : list of str
        Names of features seen during fit.

    Notes
    -----
    The EWMA is commonly used for:
    - Smoothing noisy time series
    - Technical indicators (e.g., EMA in finance)
    - Adaptive feature engineering

    Unlike the other window transformers, this transformer is stateless: it has
    no internal buffer, so ``observe``/``rewind`` fall through to the base no-op.
    Each ``transform`` call must therefore receive the full history needed by the
    EWMA computation.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.preprocessing import ExponentialMovingAverage

    >>> times = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 1, 10), interval="1d", eager=True
    ... )
    >>> X = pl.DataFrame({"time": times, "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})

    >>> transformer = ExponentialMovingAverage(alpha=0.5)
    >>> transformer.fit(X)  # doctest: +ELLIPSIS
    ExponentialMovingAverage(...)
    >>> X_t = transformer.transform(X)
    >>> len(X_t) == len(X)
    True

    See Also
    --------
    - [`RollingStatisticsTransformer`][yohou.preprocessing.window.RollingStatisticsTransformer] : Fixed-window rolling statistics.
    - [`SlidingWindowFunctionTransformer`][yohou.preprocessing.window.SlidingWindowFunctionTransformer] : Custom window functions.

    """

    _parameter_constraints: dict = {
        "alpha": [Interval(numbers.Real, 0.0, 1.0, closed="right")],
        "adjust": ["boolean"],
        "ignore_nulls": ["boolean"],
    }

    def __init__(
        self,
        alpha: float,
        adjust: bool = True,
        ignore_nulls: bool = True,
    ):
        self.alpha = alpha
        self.adjust = adjust
        self.ignore_nulls = ignore_nulls

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform X by computing EWMA.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            Transformed time series with a ``"time"`` column and transformed
            value columns.

        """
        # Get data columns
        data_cols = [c for c in X.columns if c != "time"]

        # Build EWMA expressions
        exprs = [pl.col("time")]
        for col_name in data_cols:
            ewma_expr = pl.col(col_name).ewm_mean(
                alpha=self.alpha,
                adjust=self.adjust,
                ignore_nulls=self.ignore_nulls,
            )
            exprs.append(ewma_expr.alias(f"{col_name}_ewma"))

        return X.select(exprs)

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
        input_features = _check_feature_names_in(self, input_features)
        feature_names = [f"{col}_ewma" for col in input_features]
        arr: list[str] = np.asarray(feature_names, dtype=object).tolist()
        return arr
