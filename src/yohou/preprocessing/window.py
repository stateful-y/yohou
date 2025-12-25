"""Implementation of window transformations."""

import numpy as np
import polars as pl
from pydantic import StrictInt
from sklearn.utils.validation import _check_feature_names_in

from yohou.base import BaseTransformer
from yohou.utils.tabularization import tabularize


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

    def __init__(self, lag: StrictInt | list[StrictInt] = 1):
        self.lag = lag

    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> "LagTransformer":
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
        self.lags_: list[int] = self.lag if isinstance(self.lag, list) else [self.lag]

        self._observation_horizon = max(self.lags_)

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
