"""Implementation of window transformations."""

import numbers

import numpy as np
import polars as pl
from pydantic import StrictInt
from sklearn.base import _fit_context
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted

from yohou.base import BaseTransformer
from yohou.utils import validate_transformer_data
from yohou.utils.tabularization import tabularize
from yohou.utils.tags import Tags


class LagTransformer(BaseTransformer):
    """Seasonal differencing time series transformer.

    Parameters
    ----------
    lag : int >= 1 or list of ints >= 1, default=1
        Lag of the transformation.

    Attributes
    ----------
    lags_ : list of int
        Effective list of lags.

    """

    _parameter_constraints: dict = {
        **BaseTransformer._parameter_constraints,
        "lag": [Interval(numbers.Integral, 1, None, closed="left"), list],
    }

    def __init__(self, lag: StrictInt | list[StrictInt] = 1):
        self.lag = lag

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        # LagTransformer always sets _observation_horizon in fit(), so it's always stateful
        tags.transformer_tags.stateful = True
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "LagTransformer":
        """Fits the transformer and returns it.

        Parameters
        ----------
        X : pl.DataFrame
            Feature time series.

        y : pl.DataFrame or None, default=None
            Target time series. Ignored and only present for
            API consistency.

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self

        """
        self.lags_: list[int] = self.lag if isinstance(self.lag, list) else [self.lag]

        self._observation_horizon = max(self.lags_)

        BaseTransformer.fit(self, X, y, **params)

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

        X_t = tabularize(X, self.lags_)

        return X_t

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
        # TODO: Check order of for loops
        feature_names = [f"{col}_lag_{lag}" for col in input_features for lag in self.lags_]

        arr: list[str] = np.asarray(feature_names, dtype=object).tolist()
        return arr
