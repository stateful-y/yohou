"""Implementation of seasonal naive forecaster."""

import polars as pl
import polars.selectors as cs
from pydantic import StrictInt

from .base import BasePointForecaster


class SeasonalNaive(BasePointForecaster):
    def __init__(self, seasonality: StrictInt = 1):
        BasePointForecaster.__init__(self)

        self.seasonality = seasonality

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
