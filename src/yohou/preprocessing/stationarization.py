"""Implementation of invertible transformations for stationarization."""

import numbers

import numpy as np
import polars as pl
import polars.selectors as cs
from pydantic import StrictFloat, StrictInt
from sklearn.base import _fit_context
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted

from yohou.base import BaseTransformer, Tags
from yohou.utils import validate_transformer_data


class LogTransform(BaseTransformer):
    """Logarithmic time series transformer.

    Parameters
    ----------
    offset : float >= 0.0, default=0.0
        Offset to apply to the input time series before the log transform.

    """

    _parameter_constraints: dict = {
        **BaseTransformer._parameter_constraints,
        "offset": [Interval(numbers.Real, 0, None, closed="left")],
    }

    def __init__(self, offset: StrictFloat = 0.0):
        self.offset = offset

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        tags.input_tags.min_value = -self.offset if self.offset > 0.0 else 0.0
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> "LogTransform":
        """Fits the transformer and returns it.

        Parameters
        ----------
        X : pl.DataFrame
            Feature time series.

        y : pl.DataFrame or None, default=None
            Target time series. Ignored and only present for
            API consistency.

        Returns
        -------
        self

        """
        self._observation_horizon = 0

        BaseTransformer.fit(self, X, y)

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transforms the input time series.

        Parameters
        ----------
        X : pl.DataFrame
            Feature time series.

        Returns
        -------
        pl.DataFrame
            Transformed time series.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])

        time = X.select(cs.by_name("time"))
        X_t = (X.select(~cs.by_name("time")) + self.offset).with_columns(pl.all().log())
        feature_names = self.get_feature_names_out()
        X_t = X_t.rename(dict(zip(X_t.columns, feature_names, strict=False)))
        X_t = pl.concat([time, X_t], how="horizontal")

        return X_t

    def inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame | None) -> pl.DataFrame:
        """Inverts the input transformed time series.

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
            Inverted transformed time series.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])

        X_t = validate_transformer_data(
            self,
            X=X_t,
            reset=False,
            inverse=True,
            X_p=X_p,
            observation_horizon=self.observation_horizon,
        )

        time = X_t.select(cs.by_name("time"))
        X = X_t.select(~cs.by_name("time")).with_columns(pl.all().exp()) - self.offset
        X = X.rename(dict(zip(X.columns, self.feature_names_in_, strict=False)))
        X = pl.concat([time, X], how="horizontal")

        return X

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        input_features = _check_feature_names_in(self, input_features)
        feature_names = [f"log_off_{self.offset}_{col}" for col in input_features]

        arr: list[str] = np.asarray(feature_names, dtype=object).tolist()
        return arr


class SeasonalDifferencing(BaseTransformer):
    """Seasonal differencing time series transformer.

    Parameters
    ----------
    seasonality : int > 1, default=1
        Seasonality for the differencing.

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
        # SeasonalDifferencing is always stateful (uses observation horizon)
        tags.transformer_tags.stateful = True
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> "SeasonalDifferencing":
        """Fits the transformer and returns it.

        Parameters
        ----------
        X : pl.DataFrame
            Feature time series.

        y : pl.DataFrame or None, default=None
            Target time series. Ignored and only present for
            API consistency.

        Returns
        -------
        self

        """
        self._observation_horizon = self.seasonality

        BaseTransformer.fit(self, X, y)

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transforms the input time series.

        Parameters
        ----------
        X : pl.DataFrame
            Feature time series.

        Returns
        -------
        pl.DataFrame
            Transformed time series.

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
        """Inverts the input transformed time series.

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
            Inverted transformed time series.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        X_t = validate_transformer_data(
            self,
            X=X_t,
            reset=False,
            inverse=True,
            X_p=X_p,
            observation_horizon=self.observation_horizon,
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
            Input features.

        Returns
        -------
        feature_names_out : list of str
            Transformed feature names.
        """
        input_features = _check_feature_names_in(self, input_features)
        feature_names = [f"diff_s_{self.seasonality}_{col}" for col in input_features]

        return feature_names


class SeasonalLogDifferencing(SeasonalDifferencing, LogTransform):
    """Seasonal differencing time series transformer.

    Parameters
    ----------
    seasonality : int > 1, default=1
        Seasonality for the differencing.

    offset : float >= 0.0, default=0.0
        Offset to apply to the input time series before the log transform.

    """

    def __init__(self, seasonality: StrictInt = 1, offset: StrictFloat = 0.0):
        SeasonalDifferencing.__init__(self, seasonality=seasonality)
        LogTransform.__init__(self, offset=offset)

    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> "SeasonalLogDifferencing":
        """Fits the transformer and returns it.

        Parameters
        ----------
        X : pl.DataFrame
            Feature time series.

        y : pl.DataFrame or None, default=None
            Target time series. Ignored and only present for
            API consistency.

        Returns
        -------
        self

        """
        self._observation_horizon = self.seasonality

        BaseTransformer.fit(self, X, y)

        self.log_transform_ = LogTransform(offset=self.offset).fit(X=X, y=y)
        self.seasonal_diff_transform_ = SeasonalDifferencing(seasonality=self.seasonality).fit(X=X, y=y)

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transforms the input time series.

        Parameters
        ----------
        X : pl.DataFrame
            Feature time series.

        Returns
        -------
        pl.DataFrame
            Transformed time series.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)

        # Apply log transform
        X_t = LogTransform.transform(self, X=X)

        # Apply seasonal differencing manually (skip validate_data since columns are transformed)
        time = X_t.select(cs.by_name("time"))[self.seasonality :]
        X_diff = X_t.select(~cs.by_name("time")).select(pl.all().diff(self.seasonality))[self.seasonality :]
        feature_names = self.get_feature_names_out()
        X_diff = X_diff.rename(dict(zip(X_diff.columns, feature_names, strict=False)))
        X_t = pl.concat([time, X_diff], how="horizontal")

        return X_t

    def inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame | None) -> pl.DataFrame:
        """Inverts the input transformed time series.

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
            Inverted transformed time series.
        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        X_t = validate_transformer_data(
            self,
            X=X_t,
            reset=False,
            inverse=True,
            X_p=X_p,
            observation_horizon=self.observation_horizon,
        )

        assert X_p is not None  # for ty
        X_p = self.log_transform_.transform(X=X_p)
        X = self.seasonal_diff_transform_.inverse_transform(X_t=X_t, X_p=X_p)
        X = self.log_transform_.inverse_transform(X_t=X, X_p=None)

        return X

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

        Returns
        -------
        feature_names_out : list of str
            Transformed feature names.
        """
        input_features = _check_feature_names_in(self, input_features)
        feature_names = [f"log_off_{self.offset}_diff_s_{self.seasonality}_{col}" for col in input_features]

        return feature_names
