"""Implementation of window transformations."""

import math
import numbers
from collections.abc import Callable

import numpy as np
import polars as pl
from pydantic import StrictInt
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseActualTransformer
from yohou.utils import tabularize
from yohou.utils._compat import Interval, _check_feature_names_in

__all__ = [
    "ExponentialMovingAverage",
    "HorizonRollingStatisticsTransformer",
    "LagTransformer",
    "RollingStatisticsTransformer",
    "SlidingWindowFunctionTransformer",
]


#: Statistics accepted by the rolling-statistics transformers.
_VALID_STATISTICS = frozenset({"mean", "std", "min", "max", "median", "sum", "var", "q25", "q75"})


def _rolling_statistic(expr: pl.Expr, stat: str, window_size: int) -> pl.Expr:
    """Apply a trailing rolling statistic over ``window_size`` consecutive values.

    Parameters
    ----------
    expr : pl.Expr
        Column expression.
    stat : str
        Statistic name, one of ``_VALID_STATISTICS``.
    window_size : int
        Number of values in the window.

    Returns
    -------
    pl.Expr
        Rolling statistic expression.

    """
    if stat == "mean":
        return expr.rolling_mean(window_size)
    if stat == "std":
        return expr.rolling_std(window_size)
    if stat == "min":
        return expr.rolling_min(window_size)
    if stat == "max":
        return expr.rolling_max(window_size)
    if stat == "median":
        return expr.rolling_median(window_size)
    if stat == "sum":
        return expr.rolling_sum(window_size)
    if stat == "var":
        return expr.rolling_var(window_size)
    if stat == "q25":
        return expr.rolling_quantile(0.25, window_size=window_size)
    if stat == "q75":
        return expr.rolling_quantile(0.75, window_size=window_size)
    raise AssertionError("unreachable: statistic validated in _fit")


def _seasonal_rolling_statistic(expr: pl.Expr, stat: str, window_size: int, seasonality: int) -> pl.Expr:
    """Apply a rolling statistic over ``window_size`` values spaced ``seasonality`` rows apart.

    The value at row ``t`` summarises ``x[t], x[t - k], ..., x[t - (window_size - 1) * k]``
    with ``k = seasonality``.

    Parameters
    ----------
    expr : pl.Expr
        Column expression.
    stat : str
        Statistic name, one of ``_VALID_STATISTICS``.
    window_size : int
        Number of values in the window.
    seasonality : int
        Spacing between consecutive window values, in rows.

    Returns
    -------
    pl.Expr
        Seasonal rolling statistic expression.

    Notes
    -----
    Rows ``k`` apart are consecutive members of the same residue
    class of the row index, so the ordinary rolling kernel applied within each class is
    the seasonal window, and it reuses the exact kernels (``ddof``, quantile
    interpolation) of the consecutive case. Each class holds the same rows wherever a
    frame starts, so the result does not depend on how rows are batched.

    To end the window ``s`` rows before ``t``, shift the *result* by ``s``. A shift
    placed inside the grouping would move values within each residue class, that is by
    ``s * k`` rows.

    """
    rolled = _rolling_statistic(expr, stat, window_size)
    if seasonality == 1:
        return rolled
    return rolled.over(pl.int_range(pl.len()) % seasonality)


def _normalize_statistics(statistics: str | list[str]) -> list[str]:
    """Normalise ``statistics`` to a list and validate every entry.

    Parameters
    ----------
    statistics : str or list of str
        One statistic or a list of statistics.

    Returns
    -------
    list of str
        The statistics, in the order given.

    Raises
    ------
    ValueError
        If any statistic is not one of ``_VALID_STATISTICS``.

    """
    normalized = [statistics] if isinstance(statistics, str) else list(statistics)
    invalid = set(normalized) - _VALID_STATISTICS
    if invalid:
        raise ValueError(f"Invalid statistics: {invalid}. Valid options: {set(_VALID_STATISTICS)}")
    return normalized


class LagTransformer(BaseActualTransformer):
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
    - [`HorizonRollingStatisticsTransformer`][yohou.preprocessing.window.HorizonRollingStatisticsTransformer] : Seasonal rolling statistics per forecast step.
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

    # Causal: `.shift(lag)` reads backwards only, and `observation_horizon` is
    # `max(lags)`, exactly the depth reached.
    _tags = {"stateful": True, "batch_invariant": True}

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


class SlidingWindowFunctionTransformer(BaseActualTransformer):
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

    # Causal: windows are trailing (`X[i : i + window_size]`) and stamped at the
    # window's last point, so row t reads only rows in [t - window_size + 1, t]
    # whatever `func` does inside them.
    _tags = {"stateful": True, "batch_invariant": True}

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


class RollingStatisticsTransformer(BaseActualTransformer):
    """Compute rolling window statistics for time series.

    This transformer computes one or more rolling statistics (mean, std, min,
    max, median, quantiles) over sliding windows. It is a convenience wrapper
    around polars rolling functions with a sklearn-compatible interface.

    With ``seasonality=k`` the window holds ``window_size`` values spaced ``k``
    rows apart instead of consecutive ones, so ``window_size=7, seasonality=24``
    on hourly data summarises the same hour over the last seven days.

    Parameters
    ----------
    window_size : int, default=7
        Number of values in the rolling window. Must be >= 1.
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
    seasonality : int, default=1
        Spacing, in rows, between consecutive values of the window. ``1`` gives an
        ordinary rolling window; ``k`` gives a seasonal window over the same
        position in the last ``window_size`` seasons of length ``k``. Must be >= 1.

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

    >>> # Seasonal window: every 3rd value, over the last 2 seasons
    >>> transformer = RollingStatisticsTransformer(window_size=2, seasonality=3)
    >>> transformer.fit(X)  # doctest: +ELLIPSIS
    RollingStatisticsTransformer(...)
    >>> X_t = transformer.transform(X)
    >>> X_t.columns
    ['time', 'value_s3_mean']
    >>> X_t["value_s3_mean"].to_list()
    [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]

    See Also
    --------
    - [`HorizonRollingStatisticsTransformer`][yohou.preprocessing.window.HorizonRollingStatisticsTransformer] : Seasonal rolling statistics per forecast step.
    - [`SlidingWindowFunctionTransformer`][yohou.preprocessing.window.SlidingWindowFunctionTransformer] : Apply custom function over windows.
    - [`LagTransformer`][yohou.preprocessing.window.LagTransformer] : Create lagged features.

    Notes
    -----
    Rolling statistics are computed via native polars rolling expressions
    (``rolling_mean``, ``rolling_std``, etc.), which are significantly faster
    than Python-level iteration. Quantile statistics (``q25``, ``q75``) use
    ``rolling_quantile`` with polars' default ``"nearest"`` interpolation.

    The first ``(window_size - 1) * seasonality`` rows produce nulls from
    incomplete windows and are dropped from the output, setting
    ``observation_horizon = (window_size - 1) * seasonality``.

    Output column names follow the pattern ``{input_col}_{statistic}``
    (e.g. ``"value_mean"``) and, when ``seasonality > 1``,
    ``{input_col}_s{seasonality}_{statistic}`` (e.g. ``"value_s24_mean"``). The
    window size is deliberately not part of the name, so tuning it leaves the
    feature names unchanged.

    To average seasonal lags that exclude the current row, as in
    ``mean(x[t-k], x[t-2k], ..., x[t-nk])``, lag first:
    ``FeaturePipeline([("lag", LagTransformer(lag=k)), ("mean",
    RollingStatisticsTransformer(window_size=n, seasonality=k))])``.

    """

    _parameter_constraints: dict = {
        "window_size": [Interval(numbers.Integral, 1, None, closed="left")],
        "statistics": [str, list],
        "seasonality": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    # Causal: polars rolling aggregations are trailing, and `observation_horizon`
    # is `(window_size - 1) * seasonality`, the deepest row reached. Batching does reassociate the accumulator, so bulk and
    # per row results differ by about one ULP; the conformance check compares on a
    # relative tolerance for exactly this reason.
    _tags = {"stateful": True, "batch_invariant": True}

    def __init__(
        self,
        window_size: int = 7,
        statistics: str | list[str] = "mean",
        seasonality: int = 1,
    ):
        self.window_size = window_size
        self.statistics = statistics
        self.seasonality = seasonality

    @property
    def observation_horizon(self) -> int:  # noqa: D102
        """Return the number of past observations needed."""
        return (self.window_size - 1) * self.seasonality

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit the internal model."""
        self.statistics_ = _normalize_statistics(self.statistics)

    def _output_name(self, col: str, stat: str) -> str:
        """Name of the output column for one input column and statistic."""
        return f"{col}_{stat}" if self.seasonality == 1 else f"{col}_s{self.seasonality}_{stat}"

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
        return _seasonal_rolling_statistic(col, stat, self.window_size, self.seasonality)

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
                exprs.append(stat_expr.alias(self._output_name(col_name, stat)))

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
        feature_names = [self._output_name(col, stat) for col in input_features for stat in self.statistics_]
        arr: list[str] = np.asarray(feature_names, dtype=object).tolist()
        return arr


class HorizonRollingStatisticsTransformer(BaseActualTransformer):
    """Seasonal rolling statistics laid out per forecast step.

    Seasonal naive forecasts (and their spread) as per-step features. For each
    forecast step ``h`` of the horizon, this transformer computes a statistic over
    the ``n_seasons`` most recent observed values that share the target time's
    position in the season. With ``statistics="mean"`` each step column equals, step
    for step, what
    [`MeanSeasonalNaive`][yohou.point.naive.MeanSeasonalNaive] forecasts.

    A reduction forecaster builds one feature row at the forecast origin and uses it
    for every step, so an origin-anchored seasonal statistic describes the wrong
    hour for most steps. This transformer instead emits one column per step,
    ``{col}_s{seasonality}_{stat}_step_{h}``, and declares so through its
    ``produces_step_columns`` tag. A
    [`PointReductionForecaster`][yohou.point.reduction.PointReductionForecaster]
    with ``reduction_strategy="direct"`` and ``step_feature_alignment="matched"``
    then gives the model for step ``h`` only its own step's columns.

    Parameters
    ----------
    seasonality : int
        Season length ``k``, in rows (e.g. ``24`` for a daily cycle in hourly data).
        Must be >= 2: with ``1`` every step would carry the same value.
    n_seasons : int, default=1
        Number of seasons ``n`` in each window. Must be >= 1.
    statistics : str or list of str, default="mean"
        Statistic(s) to compute: ``"mean"``, ``"std"``, ``"min"``, ``"max"``,
        ``"median"``, ``"sum"``, ``"var"``, ``"q25"``, ``"q75"``.

    Attributes
    ----------
    forecasting_horizon_ : int
        Forecasting horizon ``H`` received as fit metadata; steps ``1..H`` are emitted.
    statistics_ : list of str
        Effective list of statistics to compute.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : list of str
        Names of features seen during fit.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.preprocessing import HorizonRollingStatisticsTransformer

    >>> times = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 1, 8), interval="1d", eager=True
    ... )
    >>> X = pl.DataFrame({"time": times, "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})

    >>> # Season of 4, averaged over the last 2 seasons, for a 4-step horizon
    >>> transformer = HorizonRollingStatisticsTransformer(seasonality=4, n_seasons=2)
    >>> _ = transformer.fit(X, forecasting_horizon=4)
    >>> X_t = transformer.transform(X)
    >>> X_t.columns
    ['time', 'value_s4_mean_step_1', 'value_s4_mean_step_2', 'value_s4_mean_step_3', 'value_s4_mean_step_4']
    >>> X_t.row(0)[1:]  # step 1 = mean(x[t-3], x[t-7]), ..., step 4 = mean(x[t], x[t-4])
    (3.0, 4.0, 5.0, 6.0)

    See Also
    --------
    - [`RollingStatisticsTransformer`][yohou.preprocessing.window.RollingStatisticsTransformer] : Seasonal or consecutive rolling statistics at the forecast origin.
    - [`MeanSeasonalNaive`][yohou.point.naive.MeanSeasonalNaive] : The forecast this transformer's mean reproduces per step.
    - [`PointReductionForecaster`][yohou.point.reduction.PointReductionForecaster] : Aligns per-step columns to per-step models.

    Notes
    -----
    For an output row at origin ``t``, the column for step ``h`` summarises the
    ``n_seasons`` values ``x[t + h - k*j]`` for ``j = ceil(h/k), ..., ceil(h/k) +
    n_seasons - 1``, all at or before ``t``. Steps ``h`` and ``h + k`` share a
    position in the season and therefore carry equal values, as
    ``MeanSeasonalNaive`` repeats its pattern past one season.

    The forecasting horizon is not a constructor parameter. It arrives as
    ``forecasting_horizon`` fit metadata, which this class requests by default, so a
    reduction forecaster supplies its own horizon through metadata routing and the two
    cannot disagree. Called directly, pass it to ``fit(X, forecasting_horizon=H)``.

    The first ``seasonality * n_seasons - 1`` rows are dropped, whatever the horizon,
    and fitting needs at least ``seasonality * n_seasons`` rows.

    Feed this transformer the series as it is known at the forecast origin, and make
    it the last step of a
    [`FeaturePipeline`][yohou.compose.feature_pipeline.FeaturePipeline]. Lagging its
    input by ``d`` rows shifts every step's target position by ``d``, so step ``h``
    would describe the wrong time unless ``d`` is a whole number of seasons; lagging
    its output renames the step columns, which a reduction forecaster rejects at fit.

    Seasons are counted in rows. On a UTC axis, the local hour a daily season lines
    up with moves by one hour for about ``n_seasons`` days after a daylight saving
    switch.

    """

    _parameter_constraints: dict = {
        "seasonality": [Interval(numbers.Integral, 2, None, closed="left")],
        "n_seasons": [Interval(numbers.Integral, 1, None, closed="left")],
        "statistics": [str, list],
    }

    # Causal: every step column is a seasonal rolling statistic shifted backwards by
    # its offset (at most `seasonality - 1` rows), and `observation_horizon` is
    # `seasonality * n_seasons - 1`, the deepest row reached. Rolling kernels over
    # residue classes do not depend on where a batch starts.
    _tags = {"stateful": True, "batch_invariant": True, "produces_step_columns": True}

    # The horizon is fit metadata, requested by default so that a routing forecaster
    # or composite delivers it without per-instance ``set_fit_request`` calls.
    __metadata_request__fit = {"forecasting_horizon": True}

    def __init__(
        self,
        seasonality: StrictInt,
        n_seasons: StrictInt = 1,
        statistics: str | list[str] = "mean",
    ):
        self.seasonality = seasonality
        self.n_seasons = n_seasons
        self.statistics = statistics

    @property
    def observation_horizon(self) -> int:  # noqa: D102
        """Return the number of past observations needed."""
        return self.seasonality * self.n_seasons - 1

    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "HorizonRollingStatisticsTransformer":
        """Fit the transformer, reading the forecasting horizon from fit metadata.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column and one or more numeric columns.
        y : pl.DataFrame or None, default=None
            Ignored.  Present for API compatibility.
        **params : dict
            Fit metadata. Must contain ``forecasting_horizon``, a positive integer;
            other keys are ignored.

        Returns
        -------
        self
            The fitted transformer instance.

        Raises
        ------
        ValueError
            If ``forecasting_horizon`` is missing or not a positive integer, if a
            parameter is invalid, or if ``X`` has fewer than
            ``seasonality * n_seasons`` rows.

        """
        horizon = params.get("forecasting_horizon")
        if horizon is None:
            raise ValueError(
                f"{type(self).__name__} requires `forecasting_horizon` as fit metadata. A reduction "
                "forecaster supplies its own horizon through metadata routing; when fitting directly, "
                "call fit(X, forecasting_horizon=H)."
            )
        if isinstance(horizon, bool) or not isinstance(horizon, numbers.Integral) or horizon < 1:
            raise ValueError(f"`forecasting_horizon` must be a positive integer, got {horizon!r}.")
        super().fit(X, y)
        self.forecasting_horizon_ = int(horizon)
        return self

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Validate the amount of data and the statistics."""
        required = self.seasonality * self.n_seasons
        if len(X) < required:
            raise ValueError(
                f"{type(self).__name__} needs at least seasonality * n_seasons = {required} rows to fill "
                f"one window, but X has {len(X)} rows."
            )
        self.statistics_ = _normalize_statistics(self.statistics)

    def _profile_name(self, col: str, stat: str) -> str:
        """Name of the seasonal profile column for one input column and statistic."""
        return f"{col}_s{self.seasonality}_{stat}"

    def _step_offsets(self) -> list[int]:
        """Rows between the origin and the most recent value each step reads, for steps ``1..H``."""
        k = self.seasonality
        return [k * math.ceil(h / k) - h for h in range(1, self.forecasting_horizon_ + 1)]

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform X into per-step seasonal rolling statistics.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            A ``"time"`` column and one column per input column, statistic and step.

        """
        data_cols = [c for c in X.columns if c != "time"]
        # One seasonal rolling pass per column and statistic; every step is a shift of it.
        profiles = X.select(
            pl.col("time"),
            *[
                _seasonal_rolling_statistic(pl.col(col), stat, self.n_seasons, self.seasonality).alias(
                    self._profile_name(col, stat)
                )
                for col in data_cols
                for stat in self.statistics_
            ],
        )
        offsets = self._step_offsets()
        X_t = profiles.select(
            pl.col("time"),
            *[
                pl.col(self._profile_name(col, stat)).shift(offset).alias(f"{self._profile_name(col, stat)}_step_{h}")
                for col in data_cols
                for stat in self.statistics_
                for h, offset in enumerate(offsets, start=1)
            ],
        )
        return X_t[self._observation_horizon :]

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
            Output feature names, ordered by input column, statistic, then step.

        """
        check_is_fitted(self, ["statistics_", "forecasting_horizon_"])
        input_features = _check_feature_names_in(self, input_features)
        feature_names = [
            f"{self._profile_name(col, stat)}_step_{h}"
            for col in input_features
            for stat in self.statistics_
            for h in range(1, self.forecasting_horizon_ + 1)
        ]
        arr: list[str] = np.asarray(feature_names, dtype=object).tolist()
        return arr


class ExponentialMovingAverage(BaseActualTransformer):
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
