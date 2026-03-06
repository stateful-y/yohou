"""Implementation of invertible transformations for stationarization."""

import numbers

import numpy as np
import polars as pl
import polars.selectors as cs
from pydantic import StrictFloat, StrictInt
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseTransformer
from yohou.utils import Tags, validate_transformer_data
from yohou.utils._compat import Interval, _check_feature_names_in, _fit_context
from yohou.utils.panel import panel_aware_prefix

__all__ = [
    "ASinhTransformer",
    "AbsoluteSeasonalReturn",
    "BoxCoxTransformer",
    "LogTransformer",
    "SeasonalDifferencing",
    "SeasonalLogDifferencing",
    "SeasonalReturn",
]


class BoxCoxTransformer(BaseTransformer):
    """Box-Cox power transformation time series transformer.

    The Box-Cox transformation is a parametric transformation that stabilizes
    variance and makes the data more normally distributed. The transformation is:

    - If lambda != 0: y = (x^lambda - 1) / lambda
    - If lambda == 0: y = log(x)

    Parameters
    ----------
    lmbda : float, default=0.0
        The transformation parameter. If 0, applies log transform.
        Common values: 0 (log), 0.5 (square root), 1 (no transform), 2 (square).

    offset : float >= 0.0, default=0.0
        Offset to apply to the input time series before the Box-Cox transform.
        Useful for ensuring data is strictly positive.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : list of str
        Names of features seen during fit (excluding "time" column).

    Notes
    -----
    Box-Cox requires strictly positive input data.


    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.stationarity import BoxCoxTransformer
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, i) for i in range(1, 6)],
    ...     "value": [1.0, 4.0, 9.0, 16.0, 25.0],
    ... })
    >>> transformer = BoxCoxTransformer(lmbda=0.5)  # Square root transform
    >>> transformer.fit(X)  # doctest: +ELLIPSIS
    BoxCoxTransformer(...)
    >>> X_t = transformer.transform(X)
    >>> "time" in X_t.columns
    True

    See Also
    --------
    `LogTransformer` : Logarithmic transformation (Box-Cox with lambda=0).
    `ASinhTransformer` : Inverse hyperbolic sine transformation for data with zeros.
    sklearn.preprocessing.PowerTransformer : sklearn's power transformations.

    """

    _parameter_constraints: dict = {
        **BaseTransformer._parameter_constraints,
        "lmbda": [Interval(numbers.Real, None, None, closed="neither")],
        "offset": [Interval(numbers.Real, 0, None, closed="left")],
    }

    def __init__(self, lmbda: StrictFloat = 0.0, offset: StrictFloat = 0.0):
        self.lmbda = lmbda
        self.offset = offset

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.input_tags is not None
        # Box-Cox requires positive data (after offset)
        tags.input_tags.min_value = -self.offset if self.offset > 0.0 else 0.0
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "BoxCoxTransformer":
        """Fit the transformer to input data.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.

        y : pl.DataFrame or None, default=None
            Ignored.  Present for API compatibility with yohou pipelines.

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted transformer instance.

        """
        X = validate_transformer_data(self, X=X, reset=True)

        BaseTransformer.fit(self, X, y, **params)

        return self

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
            Transformed time series with a ``"time"`` column and transformed
            value columns.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)

        time = X.select(cs.by_name("time"))
        X_shifted = X.select(~cs.by_name("time")) + self.offset

        if self.lmbda == 0:
            X_t = X_shifted.with_columns(pl.all().log())
        else:
            # Box-Cox: (x^lambda - 1) / lambda
            X_t = X_shifted.with_columns((pl.all().pow(self.lmbda) - 1) / self.lmbda)

        feature_names = self.get_feature_names_out()
        X_t = X_t.rename(dict(zip(X_t.columns, feature_names, strict=False)))
        X_t = pl.concat([time, X_t], how="horizontal")

        return X_t

    def inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame | None) -> pl.DataFrame:
        """Inverse-transform the time series.

        Parameters
        ----------
        X_t : pl.DataFrame
            Transformed time series.

        X_p : pl.DataFrame or None
            Untransformed time series corresponding to at least `observation_horizon` immediately
            previous time stamps. Can be None if `observation_horizon == 0`.

        Returns
        -------
        pl.DataFrame
            Inverse-transformed time series, reversing the transformation
            applied by ``transform``.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])

        X_t, _ = validate_transformer_data(
            self,
            X=X_t,
            reset=False,
            inverse=True,
            X_p=X_p,
            observation_horizon=self.observation_horizon,
        )

        time = X_t.select(cs.by_name("time"))

        if self.lmbda == 0:
            X = X_t.select(~cs.by_name("time")).with_columns(pl.all().exp()) - self.offset
        else:
            # Inverse Box-Cox: (lambda * y + 1)^(1/lambda)
            X = (
                X_t.select(~cs.by_name("time")).with_columns((pl.all() * self.lmbda + 1).pow(1 / self.lmbda))
                - self.offset
            )

        X = X.rename(dict(zip(X.columns, self.feature_names_in_, strict=False)))
        X = pl.concat([time, X], how="horizontal")

        return X

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
        feature_names = [panel_aware_prefix(col, f"boxcox_l_{self.lmbda}_off_{self.offset}") for col in input_features]

        return feature_names


class LogTransformer(BoxCoxTransformer):
    """Logarithmic time series transformer.

    This is a convenience class equivalent to BoxCoxTransformer with lmbda=0.

    Parameters
    ----------
    offset : float >= 0.0, default=0.0
        Offset to apply to the input time series before the log transform.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : list of str
        Names of features seen during fit (excluding "time" column).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.stationarity import LogTransformer
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, i) for i in range(1, 6)],
    ...     "value": [1.0, 4.0, 9.0, 16.0, 25.0],
    ... })
    >>> transformer = LogTransformer(offset=0.0)
    >>> transformer.fit(X)  # doctest: +ELLIPSIS
    LogTransformer(...)
    >>> X_t = transformer.transform(X)
    >>> "time" in X_t.columns
    True

    See Also
    --------
    `BoxCoxTransformer` : Generalized power transform (parent class).
    `ASinhTransformer` : Variance stabilization for data with negatives.
    SeasonalLogDifferencing : Combined log + seasonal differencing.

    """

    _parameter_constraints: dict = {
        **BaseTransformer._parameter_constraints,
        "offset": [Interval(numbers.Real, 0, None, closed="left")],
    }

    def __init__(self, offset: StrictFloat = 0.0):
        super().__init__(lmbda=0.0, offset=offset)

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
        feature_names = [panel_aware_prefix(col, f"log_off_{self.offset}") for col in input_features]

        return feature_names


class SeasonalDifferencing(BaseTransformer):
    """Seasonal differencing time series transformer.

    Computes the difference between each value and its value at a seasonal
    lag: ``y_t - y_{t-s}`` where ``s`` is the seasonality. This removes
    seasonal patterns and is a common stationarization technique.

    Parameters
    ----------
    seasonality : int >= 1, default=1
        Seasonality for the differencing.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : list of str
        Names of features seen during fit.

    Notes
    -----
    This transformer is stateful with ``observation_horizon = seasonality``.
    The first ``seasonality`` rows are dropped in the output since they lack
    sufficient history for differencing.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.stationarity import SeasonalDifferencing

    >>> X = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 8)],
    ...     "value": [10.0, 12.0, 15.0, 13.0, 11.0, 14.0, 16.0],
    ... })

    >>> # First-order differencing (seasonality=1)
    >>> transformer = SeasonalDifferencing(seasonality=1)
    >>> transformer.fit(X)  # doctest: +ELLIPSIS
    SeasonalDifferencing(...)
    >>> X_diff = transformer.transform(X)
    >>> len(X_diff)  # One row dropped
    6

    See Also
    --------
    `SeasonalLogDifferencing` : Log transform followed by seasonal differencing.
    `SeasonalReturn` : Compute seasonal returns ((x_t - x_{t-s}) / x_{t-s}).

    """

    _parameter_constraints: dict = {
        **BaseTransformer._parameter_constraints,
        "seasonality": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(self, seasonality: StrictInt = 1):
        self.seasonality = seasonality

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.transformer_tags is not None
        tags.transformer_tags.stateful = True
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "SeasonalDifferencing":
        """Fit the transformer to input data.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.

        y : pl.DataFrame or None, default=None
            Ignored.  Present for API compatibility with yohou pipelines.

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted transformer instance.

        """
        self._observation_horizon = self.seasonality
        X = validate_transformer_data(self, X=X, reset=True)

        BaseTransformer.fit(self, X, y, **params)

        return self

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
            Transformed time series with a ``"time"`` column and transformed
            value columns.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)

        time = X.select(cs.by_name("time"))[self.seasonality :]
        X_t = X.select(~cs.by_name("time")).select(pl.all().diff(self.seasonality))[self.seasonality :]
        feature_names = self.get_feature_names_out()
        X_t = X_t.rename(dict(zip(X_t.columns, feature_names, strict=False)))
        X_t = pl.concat([time, X_t], how="horizontal")

        return X_t

    def inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame | None) -> pl.DataFrame:
        """Inverse-transform the time series.

        Parameters
        ----------
        X_t : pl.DataFrame
            Transformed time series.

        X_p : pl.DataFrame or None
            Untransformed time series corresponding to at least `observation_horizon` immediately
            previous time stamps. Can be None if `observation_horizon == 0`.

        Returns
        -------
        pl.DataFrame
            Inverse-transformed time series, reversing the transformation
            applied by ``transform``.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        X_t, X_p = validate_transformer_data(
            self,
            X=X_t,
            reset=False,
            inverse=True,
            X_p=X_p,
            observation_horizon=self.observation_horizon,
            stateful=True,
        )

        time = X_t.select(cs.by_name("time"))
        X_t.columns = X_p.columns
        X = pl.concat([X_p, X_t])

        # Get the columns and their dtypes (excluding "time")
        X_no_time = X.select(~cs.by_name("time"))
        cols_and_dtypes = list(zip(X_no_time.columns, X_no_time.dtypes, strict=False))

        def inverse_diff_col(series: pl.Series) -> pl.Series:
            """Reverse seasonal differencing for a single series."""
            # Convert to numpy for in-place mutation
            arr = series.to_numpy().copy()
            for i in range(len(X_p), len(arr)):
                arr[i] += arr[i - self.seasonality]
            return pl.Series(arr)

        X = X_no_time.with_columns([
            pl.col(col).map_batches(inverse_diff_col, return_dtype=dtype) for col, dtype in cols_and_dtypes
        ])[len(X_p) :]
        X.columns = self.feature_names_in_
        X = pl.concat([time, X], how="horizontal")

        return X

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
        feature_names = [panel_aware_prefix(col, f"diff_s_{self.seasonality}") for col in input_features]

        return feature_names


class SeasonalLogDifferencing(SeasonalDifferencing, LogTransformer):
    """Seasonal log differencing time series transformer.

    Applies log transform followed by seasonal differencing. This combines
    variance stabilization with seasonal stationarization.

    Parameters
    ----------
    seasonality : int >= 1, default=1
        Seasonality for the differencing.

    offset : float >= 0.0, default=0.0
        Offset to apply to the input time series before the log transform.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : list of str
        Names of features seen during fit (excluding "time" column).
    log_transform_ : LogTransformer
        Fitted log transform component.
    seasonal_diff_transform_ : SeasonalDifferencing
        Fitted seasonal differencing component.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.stationarity import SeasonalLogDifferencing
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, i) for i in range(1, 6)],
    ...     "value": [1.0, 2.0, 4.0, 8.0, 16.0],
    ... })
    >>> transformer = SeasonalLogDifferencing(seasonality=1)
    >>> transformer.fit(X)  # doctest: +ELLIPSIS
    SeasonalLogDifferencing(...)
    >>> X_t = transformer.transform(X)
    >>> "time" in X_t.columns
    True

    Notes
    -----
    This is equivalent to computing ``log(x_t + offset) - log(x_{t-s} + offset)``
    which equals ``log((x_t + offset) / (x_{t-s} + offset))``.

    See Also
    --------
    `SeasonalDifferencing` : Simple seasonal differencing without log transform.
    `LogTransformer` : Log transform without differencing.
    SeasonalReturn : Percentage returns instead of log differences.

    """

    _parameter_constraints: dict = {
        **BaseTransformer._parameter_constraints,
        "seasonality": [Interval(numbers.Integral, 1, None, closed="left")],
        "offset": [Interval(numbers.Real, 0, None, closed="left")],
    }

    def __init__(self, seasonality: StrictInt = 1, offset: StrictFloat = 0.0):
        SeasonalDifferencing.__init__(self, seasonality=seasonality)
        LogTransformer.__init__(self, offset=offset)

    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "SeasonalLogDifferencing":
        """Fit the transformer to input data.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.

        y : pl.DataFrame or None, default=None
            Ignored.  Present for API compatibility with yohou pipelines.

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted transformer instance.

        """
        self._observation_horizon = self.seasonality
        X = validate_transformer_data(self, X=X, reset=True)

        BaseTransformer.fit(self, X, y, **params)

        self.log_transform_ = LogTransformer(offset=self.offset).fit(X=X, y=y, **params)
        self.seasonal_diff_transform_ = SeasonalDifferencing(seasonality=self.seasonality).fit(X=X, y=y, **params)

        return self

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
            Transformed time series with a ``"time"`` column and transformed
            value columns.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)

        # Apply log transform
        X_t = LogTransformer.transform(self, X=X)

        # Apply seasonal differencing manually (skip validate_data since columns are transformed)
        time = X_t.select(cs.by_name("time"))[self.seasonality :]
        X_diff = X_t.select(~cs.by_name("time")).select(pl.all().diff(self.seasonality))[self.seasonality :]
        feature_names = self.get_feature_names_out()
        X_diff = X_diff.rename(dict(zip(X_diff.columns, feature_names, strict=False)))
        X_t = pl.concat([time, X_diff], how="horizontal")

        return X_t

    def inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame | None) -> pl.DataFrame:
        """Inverse-transform the time series.

        Parameters
        ----------
        X_t : pl.DataFrame
            Transformed time series.

        X_p : pl.DataFrame or None
            Untransformed time series corresponding to at least `observation_horizon` immediately
            previous time stamps. Can be None if `observation_horizon == 0`.

        Returns
        -------
        pl.DataFrame
            Inverse-transformed time series, reversing the transformation
            applied by ``transform``.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        X_t, X_p = validate_transformer_data(
            self,
            X=X_t,
            reset=False,
            inverse=True,
            X_p=X_p,
            observation_horizon=self.observation_horizon,
            stateful=True,
        )

        X_p = self.log_transform_.transform(X=X_p)
        X = self.seasonal_diff_transform_.inverse_transform(X_t=X_t, X_p=X_p)
        X = self.log_transform_.inverse_transform(X_t=X, X_p=None)

        return X

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
        feature_names = [
            panel_aware_prefix(col, f"log_off_{self.offset}_diff_s_{self.seasonality}") for col in input_features
        ]

        return feature_names


class SeasonalReturn(BaseTransformer):
    """Seasonal percentage return time series transformer.

    Computes the percentage return relative to the value from `seasonality`
    time steps ago:

        return_t = (X_t + offset) / (X_{t-seasonality} + offset) - 1

    This is useful for modeling relative changes in time series with
    seasonal patterns, such as year-over-year percentage growth.

    Parameters
    ----------
    seasonality : int > 1, default=1
        Seasonality lag for computing returns.

    offset : float >= 0.0, default=0.0
        Offset to apply to avoid division by zero. Should be positive if
        data contains zeros or near-zero values.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : list of str
        Names of features seen during fit (excluding "time" column).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.stationarity import SeasonalReturn
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, i) for i in range(1, 6)],
    ...     "value": [100.0, 110.0, 105.0, 115.0, 120.0],
    ... })
    >>> transformer = SeasonalReturn(seasonality=2, offset=0.0)
    >>> transformer.fit(X)  # doctest: +ELLIPSIS
    SeasonalReturn(...)
    >>> X_t = transformer.transform(X)
    >>> len(X_t) == len(X) - 2  # First 2 rows dropped
    True

    See Also
    --------
    `AbsoluteSeasonalReturn` : Absolute difference instead of percentage return.
    `SeasonalDifferencing` : Simple differencing without percentage computation.
    SeasonalLogDifferencing : Log-differencing for multiplicative relationships.

    """

    _parameter_constraints: dict = {
        **BaseTransformer._parameter_constraints,
        "seasonality": [Interval(numbers.Integral, 1, None, closed="left")],
        "offset": [Interval(numbers.Real, 0, None, closed="left")],
    }

    def __init__(self, seasonality: StrictInt = 1, offset: StrictFloat = 0.0):
        self.seasonality = seasonality
        self.offset = offset

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.transformer_tags is not None
        tags.transformer_tags.stateful = True
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "SeasonalReturn":
        """Fit the transformer to input data.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.

        y : pl.DataFrame or None, default=None
            Ignored.  Present for API compatibility with yohou pipelines.

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted transformer instance.

        """
        self._observation_horizon = self.seasonality
        X = validate_transformer_data(self, X=X, reset=True)

        BaseTransformer.fit(self, X, y, **params)

        return self

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
            Transformed time series with a ``"time"`` column and transformed
            value columns.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)

        time = X.select(cs.by_name("time"))[self.seasonality :]
        X_numeric = X.select(~cs.by_name("time")) + self.offset

        # Compute return: (X_t / X_{t-seasonality}) - 1
        X_lagged = X_numeric.shift(self.seasonality)
        X_t = (X_numeric / X_lagged - 1)[self.seasonality :]

        feature_names = self.get_feature_names_out()
        X_t = X_t.rename(dict(zip(X_t.columns, feature_names, strict=False)))
        X_t = pl.concat([time, X_t], how="horizontal")

        return X_t

    def inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame | None) -> pl.DataFrame:
        """Inverse-transform the time series.

        Parameters
        ----------
        X_t : pl.DataFrame
            Transformed time series (returns).

        X_p : pl.DataFrame or None
            Untransformed time series corresponding to at least `observation_horizon` immediately
            previous time stamps. Required for inverse transform.

        Returns
        -------
        pl.DataFrame
            Inverse-transformed time series, reversing the transformation
            applied by ``transform``.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        X_t, X_p = validate_transformer_data(
            self,
            X=X_t,
            reset=False,
            inverse=True,
            X_p=X_p,
            observation_horizon=self.observation_horizon,
            stateful=True,
        )

        time = X_t.select(cs.by_name("time"))

        # Shift X_p by offset for computation
        X_p_shifted = X_p.select(~cs.by_name("time")) + self.offset
        X_t_numeric = X_t.select(~cs.by_name("time"))
        X_t_numeric.columns = X_p_shifted.columns

        # Inverse: X_t = (return_t + 1) * X_{t-seasonality} - offset
        # We need to reconstruct step by step
        X_combined = pl.concat([X_p_shifted, X_t_numeric])

        # Get the columns and their dtypes (excluding "time")
        cols_and_dtypes = list(zip(X_combined.columns, X_combined.dtypes, strict=False))

        def inverse_return_col(series: pl.Series) -> pl.Series:
            """Reverse seasonal return for a single series."""
            arr = series.to_numpy().copy()
            for i in range(len(X_p), len(arr)):
                # X_t = (return_t + 1) * X_{t-seasonality}
                arr[i] = (arr[i] + 1) * arr[i - self.seasonality]
            return pl.Series(arr)

        X = (
            X_combined.with_columns([
                pl.col(col).map_batches(inverse_return_col, return_dtype=dtype) for col, dtype in cols_and_dtypes
            ])[len(X_p) :]
            - self.offset
        )

        X.columns = self.feature_names_in_
        X = pl.concat([time, X], how="horizontal")

        return X

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
        feature_names = [
            panel_aware_prefix(col, f"return_s_{self.seasonality}_off_{self.offset}") for col in input_features
        ]

        return feature_names


class AbsoluteSeasonalReturn(BaseTransformer):
    """Absolute seasonal return (difference) time series transformer.

    Computes the absolute difference relative to the value from `seasonality`
    time steps ago:

        diff_t = (X_t + offset) - (X_{t-seasonality} + offset) = X_t - X_{t-seasonality}

    This is semantically similar to `SeasonalDifferencing` but provides a
    consistent API with `SeasonalReturn` and explicitly handles the offset
    parameter for API symmetry.

    Parameters
    ----------
    seasonality : int > 1, default=1
        Seasonality lag for computing differences.

    offset : float >= 0.0, default=0.0
        Offset parameter (provided for API consistency with SeasonalReturn,
        but note that offset cancels out in the computation).

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : list of str
        Names of features seen during fit (excluding "time" column).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.stationarity import AbsoluteSeasonalReturn
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, i) for i in range(1, 6)],
    ...     "value": [100.0, 110.0, 105.0, 115.0, 120.0],
    ... })
    >>> transformer = AbsoluteSeasonalReturn(seasonality=2, offset=0.0)
    >>> transformer.fit(X)  # doctest: +ELLIPSIS
    AbsoluteSeasonalReturn(...)
    >>> X_t = transformer.transform(X)
    >>> len(X_t) == len(X) - 2  # First 2 rows dropped
    True

    See Also
    --------
    `SeasonalReturn` : Percentage returns instead of absolute differences.
    `SeasonalDifferencing` : Equivalent computation with different API.

    """

    _parameter_constraints: dict = {
        **BaseTransformer._parameter_constraints,
        "seasonality": [Interval(numbers.Integral, 1, None, closed="left")],
        "offset": [Interval(numbers.Real, 0, None, closed="left")],
    }

    def __init__(self, seasonality: StrictInt = 1, offset: StrictFloat = 0.0):
        self.seasonality = seasonality
        self.offset = offset

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.transformer_tags is not None
        tags.transformer_tags.stateful = True
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "AbsoluteSeasonalReturn":
        """Fit the transformer to input data.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.

        y : pl.DataFrame or None, default=None
            Ignored.  Present for API compatibility with yohou pipelines.

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted transformer instance.

        """
        self._observation_horizon = self.seasonality
        X = validate_transformer_data(self, X=X, reset=True)

        BaseTransformer.fit(self, X, y, **params)

        return self

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
            Transformed time series with a ``"time"`` column and transformed
            value columns.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)

        time = X.select(cs.by_name("time"))[self.seasonality :]

        # Compute absolute difference: X_t - X_{t-seasonality}
        # Note: offset cancels out in addition/subtraction
        X_t = X.select(~cs.by_name("time")).select(pl.all().diff(self.seasonality))[self.seasonality :]

        feature_names = self.get_feature_names_out()
        X_t = X_t.rename(dict(zip(X_t.columns, feature_names, strict=False)))
        X_t = pl.concat([time, X_t], how="horizontal")

        return X_t

    def inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame | None) -> pl.DataFrame:
        """Inverse-transform the time series.

        Parameters
        ----------
        X_t : pl.DataFrame
            Transformed time series (absolute differences).

        X_p : pl.DataFrame or None
            Untransformed time series corresponding to at least `observation_horizon` immediately
            previous time stamps. Required for inverse transform.

        Returns
        -------
        pl.DataFrame
            Inverse-transformed time series, reversing the transformation
            applied by ``transform``.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        X_t, X_p = validate_transformer_data(
            self,
            X=X_t,
            reset=False,
            inverse=True,
            X_p=X_p,
            observation_horizon=self.observation_horizon,
            stateful=True,
        )

        time = X_t.select(cs.by_name("time"))
        X_t.columns = X_p.columns
        X = pl.concat([X_p, X_t])

        # Get the columns and their dtypes (excluding "time")
        X_no_time = X.select(~cs.by_name("time"))
        cols_and_dtypes = list(zip(X_no_time.columns, X_no_time.dtypes, strict=False))

        def inverse_diff_col(series: pl.Series) -> pl.Series:
            """Reverse seasonal differencing for a single series."""
            arr = series.to_numpy().copy()
            for i in range(len(X_p), len(arr)):
                arr[i] += arr[i - self.seasonality]
            return pl.Series(arr)

        X = X_no_time.with_columns([
            pl.col(col).map_batches(inverse_diff_col, return_dtype=dtype) for col, dtype in cols_and_dtypes
        ])[len(X_p) :]
        X.columns = self.feature_names_in_
        X = pl.concat([time, X], how="horizontal")

        return X

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
        feature_names = [
            panel_aware_prefix(col, f"abs_return_s_{self.seasonality}_off_{self.offset}") for col in input_features
        ]

        return feature_names


class ASinhTransformer(BaseTransformer):
    """Variance stabilization through arcsinh transform.

    Applies the transformation: asinh((X - median) / MAD)

    Where MAD is the Median Absolute Deviation scaled by 1.4826 to be
    consistent with the standard deviation for normally distributed data.

    This transformation is useful for:
    - Stabilizing variance in heteroscedastic time series
    - Handling data with outliers (asinh is less sensitive than log)
    - Data that can be negative (unlike log transform)

    Parameters
    ----------
    scale : float > 0, default=1.4826
        Scale factor for MAD normalization. Default value makes MAD
        consistent with standard deviation for normal distributions.

    Attributes
    ----------
    median_ : dict[str, float]
        Median values for each column (excluding time).

    mad_ : dict[str, float]
        Scaled MAD values for each column (excluding time).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.stationarity import ASinhTransformer
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, i) for i in range(1, 6)],
    ...     "value": [1.0, 10.0, 100.0, 1000.0, 10000.0],
    ... })
    >>> transformer = ASinhTransformer()
    >>> transformer.fit(X)  # doctest: +ELLIPSIS
    ASinhTransformer(...)
    >>> X_t = transformer.transform(X)
    >>> "time" in X_t.columns
    True

    See Also
    --------
    `BoxCoxTransformer` : Power transform for variance stabilization.
    `LogTransformer` : Simpler variance stabilization for positive data.
    sklearn.preprocessing.PowerTransformer : sklearn's power transforms.

    """

    _parameter_constraints: dict = {
        **BaseTransformer._parameter_constraints,
        "scale": [Interval(numbers.Real, 0, None, closed="neither")],
    }

    def __init__(self, scale: StrictFloat = 1.4826):
        self.scale = scale

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "ASinhTransformer":
        """Fit the transformer to input data.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.

        y : pl.DataFrame or None, default=None
            Ignored.  Present for API compatibility with yohou pipelines.

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted transformer instance.

        """
        X = validate_transformer_data(self, X=X, reset=True)

        BaseTransformer.fit(self, X, y, **params)

        # Compute median and MAD for each column (excluding time)
        X_numeric = X.select(~cs.by_name("time"))

        self.median_: dict[str, float] = {}
        self.mad_: dict[str, float] = {}

        for col in X_numeric.columns:
            col_data = X_numeric.get_column(col)
            median_val = col_data.median()
            # Cast to float for numeric operations (polars median returns numeric type)
            self.median_[col] = float(median_val) if median_val is not None else 0.0  # type: ignore[arg-type]

            # Compute MAD: median(|X - median(X)|) * scale
            abs_dev = (col_data - self.median_[col]).abs()
            mad_val = abs_dev.median()
            mad_scaled = float(mad_val) * self.scale if mad_val is not None else 1.0  # type: ignore[arg-type]

            # Avoid division by zero
            self.mad_[col] = mad_scaled if mad_scaled != 0.0 else 1.0

        return self

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
            Transformed time series with a ``"time"`` column and transformed
            value columns.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_", "median_", "mad_"])
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)

        time = X.select(cs.by_name("time"))
        X_numeric = X.select(~cs.by_name("time"))

        # Apply asinh((X - median) / MAD) for each column
        transformed_cols = []
        for col in X_numeric.columns:
            col_data = X_numeric.get_column(col).to_numpy()
            normalized = (col_data - self.median_[col]) / self.mad_[col]
            transformed = pl.Series(panel_aware_prefix(col, "asinh"), np.arcsinh(normalized))
            transformed_cols.append(transformed)

        X_t = pl.DataFrame(transformed_cols)
        X_t = pl.concat([time, X_t], how="horizontal")

        return X_t

    def inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame | None) -> pl.DataFrame:
        """Inverse-transform the time series.

        Parameters
        ----------
        X_t : pl.DataFrame
            Transformed time series.

        X_p : pl.DataFrame or None
            Untransformed time series corresponding to at least `observation_horizon` immediately
            previous time stamps. Can be None if `observation_horizon == 0`.

        Returns
        -------
        pl.DataFrame
            Inverse-transformed time series, reversing the transformation
            applied by ``transform``.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_", "median_", "mad_"])
        X_t, _ = validate_transformer_data(
            self,
            X=X_t,
            reset=False,
            inverse=True,
            X_p=X_p,
            observation_horizon=self.observation_horizon,
        )

        time = X_t.select(cs.by_name("time"))
        X_t_numeric = X_t.select(~cs.by_name("time"))

        # Apply inverse: sinh(X_t) * MAD + median for each column
        inverse_cols = []
        for i, col in enumerate(X_t_numeric.columns):
            # Get original column name from feature_names_in_
            orig_col = self.feature_names_in_[i]
            col_data = X_t_numeric.get_column(col).to_numpy()
            sinh_val = np.sinh(col_data)
            inverse_val = sinh_val * self.mad_[orig_col] + self.median_[orig_col]
            inverse_cols.append(pl.Series(orig_col, inverse_val))

        X = pl.DataFrame(inverse_cols)
        X = pl.concat([time, X], how="horizontal")

        return X

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
        return [panel_aware_prefix(col, "asinh") for col in input_features]
