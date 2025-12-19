"""Implementation of reduction-based point forecasters."""

from typing import Optional

import polars as pl
from pydantic import StrictInt
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression

from yohou.base import BaseReductionForecaster, BaseTransformer

from .base import BasePointForecaster


class PointReductionForecaster(BaseReductionForecaster, BasePointForecaster):
    _supports_cross_learning = True

    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),
        target_transformer: Optional[BaseTransformer] = None,
        feature_transformer: Optional[BaseTransformer] = None,
    ):
        BaseReductionForecaster.__init__(
            self,
            estimator=estimator,
        )

        BasePointForecaster.__init__(
            self,
            target_transformer=target_transformer,
            feature_transformer=feature_transformer,
        )

    def fit(
        self,
        y: pl.DataFrame,
        X_ante: Optional[pl.DataFrame] = None,
        X_post: Optional[pl.DataFrame] = None,
        forecasting_horizon: StrictInt = 1,
    ) -> "PointReductionForecaster":
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
        y_t, X_t = BasePointForecaster._pre_fit(
            self,
            y=y,
            X_ante=X_ante,
            X_post=X_post,
            forecasting_horizon=forecasting_horizon,
        )

        self.estimator_ = self._estimator_fit_one(y_t, X_t, forecasting_horizon)

        return self

    def _predict_one(
        self,
    ) -> pl.DataFrame:
        """Predicts the model forecasting horizon from the
        observation horizon.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        y_pred = self._estimator_predict_one(self.estimator_)
        y_pred = self._add_time_columns(y_pred)

        return y_pred
