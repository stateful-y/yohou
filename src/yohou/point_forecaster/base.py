"""Base class for point forecasters."""

import abc
from typing import Optional

import polars as pl
from pydantic import StrictInt

from yohou.base import BaseForecaster


class BasePointForecaster(BaseForecaster, metaclass=abc.ABCMeta):
    """Base class for forecasters.

    Parameters
    ----------
    target_transformer : instance of `BaseTransformer` or None, default=None
        Transformer used to transform the target time series into the new target.

    feature_transformer : instance of `BaseTransformer` or None, default=None
        Transformer used to transform the target time series into features.
        .
    """

    _prediction_type = "point"

    def fit(
        self,
        y: pl.DataFrame,
        X_ante: Optional[pl.DataFrame] = None,
        X_post: Optional[pl.DataFrame] = None,
        forecasting_horizon: StrictInt = 1,
    ) -> "BasePointForecaster":
        """Fits the forecaster and returns it.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        X_ante : pl.DataFrame or None, default=None
            Ex-ante feature time series.

        X_post : pl.DataFrame or None, default=None
            Ex-post feature time series.

        forecasting_horizon : int >= 1, default=1
            Horizon to forecast.

        Returns
        -------
        self

        """
        BaseForecaster._pre_fit(
            self,
            y=y,
            X_ante=X_ante,
            X_post=X_post,
            forecasting_horizon=forecasting_horizon,
        )

        return self
