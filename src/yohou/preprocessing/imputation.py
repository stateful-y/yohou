"""Imputation transformers for handling missing values in time series."""

import numbers
from copy import deepcopy

import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.impute import KNNImputer as sklearn_KNNImputer
from sklearn.impute import SimpleImputer as sklearn_SimpleImputer
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseTransformer
from yohou.preprocessing.sklearn_base import SklearnTransformer
from yohou.utils import Tags, validate_transformer_data
from yohou.utils._compat import Interval, StrOptions, _check_feature_names_in, _fit_context
from yohou.utils.validation import interval_to_timedelta

__all__ = [
    "SeasonalImputer",
    "SimpleImputer",
    "SimpleTimeImputer",
    "TransformedSpaceKNNImputer",
]


class SimpleImputer(SklearnTransformer):
    """Simple imputation using sklearn's SimpleImputer.

    Replaces missing values using a simple strategy (mean, median, most frequent,
    or constant). Wraps sklearn's SimpleImputer while preserving polars DataFrame
    structure and time column.

    Parameters
    ----------
    strategy : {"mean", "median", "most_frequent", "constant"}, default="mean"
        Imputation strategy:
        - "mean": Replace with mean of each column
        - "median": Replace with median of each column
        - "most_frequent": Replace with most frequent value
        - "constant": Replace with fill_value
    fill_value : str or numerical value, default=None
        When strategy="constant", fill_value is used to replace missing values.
        For string or object columns, fill_value must be a string.
    missing_values : int, float, str, or np.nan, default=np.nan
        The placeholder for missing values. All occurrences of missing_values
        will be imputed.
    copy : bool, default=True
        Passed through to sklearn's SimpleImputer to control whether a copy of
        X is created. If False, imputation may be done in-place where possible.

    Attributes
    ----------
    instance_ : SimpleImputer
        The fitted sklearn SimpleImputer instance.
    statistics_ : ndarray of shape (n_features,)
        The imputation fill value for each feature (same as sklearn's statistics_).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> import numpy as np
    >>> from yohou.preprocessing import SimpleImputer

    >>> X = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "value": [1.0, np.nan, 3.0, np.nan, 5.0],
    ... })
    >>> imputer = SimpleImputer(strategy="mean")
    >>> imputer.fit(X)  # doctest: +ELLIPSIS
    SimpleImputer(...)
    >>> X_imputed = imputer.transform(X)
    >>> X_imputed["value"].null_count()
    0

    See Also
    --------
    - [`TransformedSpaceKNNImputer`][yohou.preprocessing.imputation.TransformedSpaceKNNImputer] : K-nearest neighbors imputation.
    - [`SimpleTimeImputer`][yohou.preprocessing.imputation.SimpleTimeImputer] : Time series specific imputation methods.
    `sklearn.impute.SimpleImputer` : Underlying implementation.

    """

    _estimator_default_class = sklearn_SimpleImputer

    def __init__(self, strategy="mean", fill_value=None, missing_values=np.nan, copy=True, **kwargs):
        super().__init__(strategy=strategy, fill_value=fill_value, missing_values=missing_values, copy=copy, **kwargs)

    @property
    def statistics_(self):
        """Get imputation statistics from fitted imputer."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.statistics_


class TransformedSpaceKNNImputer(BaseTransformer):
    """K-nearest neighbors imputation in a transformed feature space.

    Projects the data through an optional transformer before performing KNN
    imputation.  Neighbor search *and* imputation are both performed in the
    transformed representation, making the result fundamentally different from
    composing a transformer and a KNN imputer sequentially in a pipeline.

    When ``transformer=None`` imputation happens directly on the raw features.
    Setting ``transformer=LagTransformer(lag=k)`` subsumes a window-based KNN
    imputer
    because neighbors are now lag-feature vectors, i.e. temporally similar
    windows, rather than individual time points.  Any projection
    (``PolynomialFeatures``, ``SplineTransformer``, PCA, …) can be used as the
    imputation space.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighboring samples to use for imputation.
    weights : {"uniform", "distance"}, default="uniform"
        Weight function used in prediction:

        - ``"uniform"``: All points in the neighborhood weighted equally.
        - ``"distance"``: Closer neighbors have greater influence.
    metric : {"nan_euclidean"}, default="nan_euclidean"
        Distance metric for searching neighbors.  Only ``nan_euclidean`` is
        supported as it handles missing values.
    transformer : BaseTransformer or None, default=None
        An optional yohou transformer used to project the data before KNN
        imputation.  Must implement ``fit`` / ``transform``.  If ``None``,
        imputation is performed directly on the raw features.

    Attributes
    ----------
    imputer_ : sklearn KNNImputer
        The fitted sklearn KNNImputer instance (fitted in transformed space).
    transformer_ : BaseTransformer or None
        A deep-copied and fitted instance of the transformer (or ``None``).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> import numpy as np
    >>> from yohou.preprocessing import TransformedSpaceKNNImputer

    Basic usage (no transformer, raw-feature KNN):

    >>> X = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 11)],
    ...     "value": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0],
    ... })
    >>> imputer = TransformedSpaceKNNImputer(n_neighbors=3)
    >>> imputer.fit(X)
    TransformedSpaceKNNImputer(...)
    >>> X_imputed = imputer.transform(X)
    >>> X_imputed["value"].null_count()
    0

    With a lag transformer the imputation happens in the projected lag space,
    so the output contains the lag columns (``value_lag_1``, ``value_lag_2``,
    ``value_lag_3``), not the original ``value`` column:

    >>> from yohou.preprocessing import LagTransformer
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 21)],
    ...     "value": [float(i) for i in range(1, 21)],
    ... })
    >>> imputer = TransformedSpaceKNNImputer(
    ...     n_neighbors=3,
    ...     transformer=LagTransformer(lag=3),
    ... )
    >>> imputer.fit(X)
    TransformedSpaceKNNImputer(...)
    >>> X_t = imputer.transform(X)
    >>> X_t["value_lag_3"].null_count()  # null count in the lag-space output
    0

    See Also
    --------
    - [`LagTransformer`][yohou.preprocessing.window.LagTransformer] : Creates lagged features from time series.
    - [`SimpleTimeImputer`][yohou.preprocessing.imputation.SimpleTimeImputer] : Interpolation-based imputation.
    - [`SimpleImputer`][yohou.preprocessing.imputation.SimpleImputer] : Simple constant-strategy imputation.

    """

    _parameter_constraints: dict = {
        "n_neighbors": [Interval(numbers.Integral, 1, None, closed="left")],
        "weights": [StrOptions({"uniform", "distance"})],
        "metric": [StrOptions({"nan_euclidean"})],
        "transformer": [None, BaseTransformer],
    }

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "nan_euclidean",
        transformer: BaseTransformer | None = None,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.transformer = transformer

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags.

        """
        tags = super().__sklearn_tags__()
        assert tags.transformer_tags is not None
        # Stateful when the inner transformer declares itself stateful.
        # We query the *parameter* object's tags (not fitted state) so the
        # tag is stable before and after fit (check_tags_static_after_fit).
        if self.transformer is not None:
            inner_tags = self.transformer.__sklearn_tags__()
            if inner_tags.transformer_tags is not None:
                tags.transformer_tags.stateful = inner_tags.transformer_tags.stateful
        else:
            tags.transformer_tags.stateful = False
        tags.transformer_tags.invertible = False
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "TransformedSpaceKNNImputer":
        """Fit the imputer, optionally projecting through a transformer first.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.
        y : pl.DataFrame or None, default=None
            Ignored.  Present for API compatibility.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted imputer instance.

        """
        X = validate_transformer_data(self, X=X, reset=True)

        # Fit and apply the inner transformer (if any)
        if self.transformer is not None:
            self.transformer_ = deepcopy(self.transformer)
            self.transformer_.fit(X)
            X_projected = self.transformer_.transform(X)
        else:
            self.transformer_ = None
            X_projected = X

        BaseTransformer.fit(self, X, y, **params)

        # Inherit observation_horizon from the inner transformer after the base
        # fit, so it is not overwritten by BaseTransformer.fit's own sync step.
        if self.transformer_ is not None:
            self._observation_horizon = self.transformer_.observation_horizon

        # Fit sklearn KNNImputer on the (optionally transformed) data
        X_no_time = X_projected.select(~cs.by_name("time"))
        self.imputer_ = sklearn_KNNImputer(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
        )
        self.imputer_.fit(X_no_time.to_numpy())

        # Store output schema for transform
        self._output_schema = X_projected.schema

        return self

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Impute missing values, optionally in a transformed feature space.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            Imputed time series.

        """
        # Project via inner transformer (if any)
        X_projected = self.transformer_.transform(X) if self.transformer_ is not None else X

        # Apply sklearn KNNImputer in the (optionally transformed) space
        time = X_projected.select(cs.by_name("time"))
        X_no_time = X_projected.select(~cs.by_name("time"))
        data_cols = X_no_time.columns

        X_imputed_np = self.imputer_.transform(X_no_time.to_numpy())
        X_imputed = pl.DataFrame(X_imputed_np, schema=data_cols)

        return pl.concat([time, X_imputed], how="horizontal")

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
        check_is_fitted(self, ["imputer_"])
        if self.transformer_ is not None:
            return self.transformer_.get_feature_names_out(input_features)
        input_features = _check_feature_names_in(self, input_features)
        return list(input_features)


class SimpleTimeImputer(BaseTransformer):
    """Time series imputation using interpolation or filling methods.

    Imputes missing values using time series-aware methods like linear
    interpolation, forward fill, backward fill, or combinations.

    Parameters
    ----------
    method : {"linear", "forward", "backward", "nearest", "fill_both"}, default="linear"
        Imputation method:
        - "linear": Linear interpolation between known values
        - "forward": Forward fill (last observation carried forward)
        - "backward": Backward fill (next observation carried backward)
        - "nearest": Use nearest non-null value
        - "fill_both": Forward fill then backward fill (handles edges)
    limit : int or None, default=None
        Maximum number of consecutive NaN values to fill. If None, no limit.

    Attributes
    ----------
    method_ : str
        Validated imputation method.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> import numpy as np
    >>> from yohou.preprocessing import SimpleTimeImputer

    >>> X = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 8)],
    ...     "value": [1.0, np.nan, np.nan, 4.0, np.nan, 6.0, 7.0],
    ... })

    >>> # Linear interpolation
    >>> imputer = SimpleTimeImputer(method="linear")
    >>> imputer.fit(X)
    SimpleTimeImputer()
    >>> X_imputed = imputer.transform(X)
    >>> X_imputed["value"].null_count()
    0

    >>> # Forward fill with limit
    >>> imputer = SimpleTimeImputer(method="forward", limit=1)
    >>> imputer.fit(X)  # doctest: +ELLIPSIS
    SimpleTimeImputer(...)
    >>> X_imputed = imputer.transform(X)
    >>> "time" in X_imputed.columns
    True

    See Also
    --------
    - [`SimpleImputer`][yohou.preprocessing.imputation.SimpleImputer] : Simple constant-strategy imputation.
    - [`SeasonalImputer`][yohou.preprocessing.imputation.SeasonalImputer] : Seasonal decomposition-based imputation.

    """

    _valid_methods = {"linear", "forward", "backward", "nearest", "fill_both"}

    _parameter_constraints: dict = {
        "method": [StrOptions(_valid_methods)],
        "limit": [Interval(numbers.Integral, 1, None, closed="left"), None],
    }

    _tags = {"stateful": False, "invertible": False}

    def __init__(
        self,
        method: str = "linear",
        limit: int | None = None,
    ):
        self.method = method
        self.limit = limit

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "SimpleTimeImputer":
        """Fit the imputer (validates parameters).

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.
        y : pl.DataFrame or None, default=None
            Ignored.  Present for API compatibility.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted transformer instance.

        """
        X = validate_transformer_data(self, X=X, reset=True)
        BaseTransformer.fit(self, X, y, **params)

        self.method_ = self.method

        return self

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Impute missing values in time series.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            Imputed time series.

        """
        # Get data columns
        data_cols = [c for c in X.columns if c != "time"]

        # Build expressions for imputation
        exprs = [pl.col("time")]

        for col_name in data_cols:
            col = pl.col(col_name)

            if self.method_ == "linear":
                # Polars interpolate does linear by default
                imputed = col.interpolate()
            elif self.method_ == "forward":
                imputed = col.forward_fill(limit=self.limit)
            elif self.method_ == "backward":
                imputed = col.backward_fill(limit=self.limit)
            elif self.method_ == "nearest":
                # Fill each null with the value closest in time, comparing the
                # distance to the previous and next known observations. Ties go
                # to the trailing (forward) anchor.
                known_time = pl.when(col.is_not_null()).then(pl.col("time")).otherwise(None)
                prev_val = col.forward_fill(limit=self.limit)
                next_val = col.backward_fill(limit=self.limit)
                prev_time = known_time.forward_fill(limit=self.limit)
                next_time = known_time.backward_fill(limit=self.limit)
                imputed = (
                    pl
                    .when(col.is_not_null())
                    .then(col)
                    .when(prev_val.is_null())
                    .then(next_val)
                    .when(next_val.is_null())
                    .then(prev_val)
                    .when((next_time - pl.col("time")) < (pl.col("time") - prev_time))
                    .then(next_val)
                    .otherwise(prev_val)
                )
            elif self.method_ == "fill_both":
                # Forward fill then backward fill
                imputed = col.forward_fill(limit=self.limit).backward_fill(limit=self.limit)
            else:
                imputed = col

            exprs.append(imputed.alias(col_name))

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
        check_is_fitted(self, ["feature_names_in_"])
        input_features = _check_feature_names_in(self, input_features)
        return list(input_features)


class SeasonalImputer(BaseTransformer):
    """Seasonal decomposition-based imputation for missing values.

    Imputes missing values by leveraging seasonal patterns in the data.
    Missing values are replaced with the mean (or median) of the non-missing
    observations that fall at the same position within each seasonal cycle.
    This is simple modular averaging by season, not a trend/seasonal
    decomposition.

    Parameters
    ----------
    period : int
        Seasonal period (e.g., 7 for weekly, 12 for monthly with annual seasonality).
        Must be >= 2.
    fill_method : {"seasonal_mean", "seasonal_median"}, default="seasonal_mean"
        Method to compute seasonal values:
        - "seasonal_mean": Use mean of same-season observations
        - "seasonal_median": Use median of same-season observations

    Attributes
    ----------
    seasonal_values_ : dict
        Dictionary mapping column names to seasonal value arrays of shape (period,).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> import numpy as np
    >>> from yohou.preprocessing import SeasonalImputer

    >>> # Weekly data with missing values
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 15)],
    ...     "value": [10.0, 20.0, 30.0, 25.0, 15.0, 5.0, 8.0, np.nan, 21.0, 31.0, np.nan, 16.0, 6.0, 9.0],
    ... })
    >>> imputer = SeasonalImputer(period=7)
    >>> imputer.fit(X)
    SeasonalImputer(period=7)
    >>> X_imputed = imputer.transform(X)
    >>> X_imputed["value"].null_count()
    0

    See Also
    --------
    - [`SimpleTimeImputer`][yohou.preprocessing.imputation.SimpleTimeImputer] : Interpolation-based imputation.
    - [`SimpleImputer`][yohou.preprocessing.imputation.SimpleImputer] : Simple constant-strategy imputation.

    """

    _valid_fill_methods = {"seasonal_mean", "seasonal_median"}

    _parameter_constraints: dict = {
        "period": [Interval(numbers.Integral, 2, None, closed="left")],
        "fill_method": [StrOptions(_valid_fill_methods)],
    }

    _tags = {"stateful": False, "invertible": False}

    def __init__(
        self,
        period: int,
        fill_method: str = "seasonal_mean",
    ):
        self.period = period
        self.fill_method = fill_method

    def _season_index_expr(self) -> pl.Expr:
        """Build a polars expression for the time-anchored season index.

        The index counts whole sampling steps between each timestamp and the
        first fitted timestamp, modulo ``period``. This keeps the seasonal
        bucket of any timestamp stable regardless of where a transform slice
        starts.
        """
        if self._step_seconds_ is None:
            # Irregular interval: fall back to row position within the frame.
            return pl.arange(0, pl.len()) % self.period
        offset = (pl.col("time") - self._first_time_).dt.total_seconds() / self._step_seconds_
        return (offset.round().cast(pl.Int64) % self.period).cast(pl.Int64)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "SeasonalImputer":
        """Fit the imputer by computing seasonal values.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.
        y : pl.DataFrame or None, default=None
            Ignored.  Present for API compatibility.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted transformer instance.

        """
        X = validate_transformer_data(self, X=X, reset=True)
        BaseTransformer.fit(self, X, y, **params)

        # Anchor the season index to the first observed timestamp so that
        # transform on any sub-range of the series uses the same seasonal
        # bucket as fit did for the same timestamps.
        self._first_time_ = X["time"][0]
        step = interval_to_timedelta(self.interval_)
        self._step_seconds_ = step.total_seconds() if step is not None else None

        # Compute seasonal values for each column
        data_cols = [c for c in X.columns if c != "time"]
        self.seasonal_values_: dict[str, np.ndarray] = {}

        # Add season index
        X_with_season = X.with_columns(self._season_index_expr().alias("_season_idx"))

        # Aggregate all columns per season in a single grouped pass. Nulls and
        # NaNs are excluded so empty seasons remain null (mapped to NaN below).
        def _stat(col: str) -> pl.Expr:
            """Per-season mean or median of ``col``, ignoring nulls and NaNs."""
            expr = pl.col(col).fill_nan(None).drop_nulls()
            return (expr.mean() if self.fill_method == "seasonal_mean" else expr.median()).alias(col)

        season_stats = X_with_season.group_by("_season_idx").agg([_stat(c) for c in data_cols])

        # Index aggregated rows by season for direct lookup.
        stats_by_season = {row["_season_idx"]: row for row in season_stats.to_dicts()}

        for col_name in data_cols:
            seasonal_vals = np.full(self.period, np.nan)
            for season_idx in range(self.period):
                row = stats_by_season.get(season_idx)
                if row is not None and row[col_name] is not None:
                    seasonal_vals[season_idx] = row[col_name]
            self.seasonal_values_[col_name] = seasonal_vals

        return self

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Impute missing values using seasonal patterns.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            Imputed time series.

        """
        # Get data columns
        data_cols = [c for c in X.columns if c != "time"]

        # Time-anchored season index for each row of this slice.
        season_idx_arr = X.select(self._season_index_expr().alias("_season_idx"))["_season_idx"].to_numpy()

        # Impute each column
        result_cols = {"time": X["time"]}

        for col_name in data_cols:
            values = X[col_name].to_numpy().copy()
            seasonal_vals = self.seasonal_values_.get(col_name)

            if seasonal_vals is not None:
                # Find null/nan indices
                null_mask = np.isnan(values) | (X[col_name].is_null().to_numpy())

                # Replace with seasonal values
                for i in np.where(null_mask)[0]:
                    values[i] = seasonal_vals[season_idx_arr[i]]

            # A position whose season bucket was empty at fit time stays NaN;
            # convert back to a proper polars null rather than emitting a float
            # NaN, which would violate the data contract.
            result_cols[col_name] = pl.Series(values).fill_nan(None)

        return pl.DataFrame(result_cols)

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
