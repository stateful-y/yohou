"""Implementation of seasonal naive forecaster."""

from typing import Optional

import polars as pl
import polars.selectors as cs
from pydantic import StrictInt

from .base import BasePointForecaster


class SeasonalNaive(BasePointForecaster):
    """Seasonal naive forecaster that repeats values from previous season.

    Parameters
    ----------
    seasonality : int, default=1
        The seasonal period length. For example, 7 for weekly seasonality
        in daily data, or 12 for monthly seasonality in monthly data.

    """

    def __init__(self, seasonality: StrictInt = 1):
        BasePointForecaster.__init__(self)

        self.seasonality = seasonality

    def reset(
        self,
        y: pl.DataFrame,
        X_ante: Optional[pl.DataFrame],
        X_post: Optional[pl.DataFrame],
    ) -> "SeasonalNaive":
        """Resets the forecaster by resetting the observation horizon.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        X_ante : pl.DataFrame or None
            Ex-ante feature time series.

        X_post : pl.DataFrame or None
            Ex-post feature time series.

        Returns
        -------
        self

        """
        self._y_observed = y
        self._X_ante_observed = X_ante
        self._X_post_observed = X_post

        return self

    def _predict_one(self) -> pl.DataFrame:
        """Predicts the model forecasting horizon from the
        observation horizon.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        if self.seasonality > self.fit_forecasting_horizon_:
            y_pred = self._y_observed.select(~cs.by_name("time"))[
                -self.seasonality : -self.seasonality + self.fit_forecasting_horizon_
            ]

        else:
            y_pred = self._y_observed.select(~cs.by_name("time"))[
                -self.fit_forecasting_horizon_ - 1 :
            ]

            y_pred = y_pred.with_columns(
                y_pred.shift(-self.fit_forecasting_horizon_ + self.seasonality).fill_null(
                    strategy="forward"
                )
            )[1 : self.fit_forecasting_horizon_ + 1]

        y_pred = self._add_time_columns(y_pred)

        return y_pred
