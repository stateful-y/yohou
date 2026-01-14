"""Implementation of seasonal naive forecaster."""

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

    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "BasePointForecaster":
        """Fits the forecaster and returns it.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None, default=None
            Exogenous feature time series.
        forecasting_horizon : int >= 1, default=1
            Horizon to forecast.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self

        """
        self._observation_horizon = self.seasonality

        BasePointForecaster.fit(
            self,
            y=y,
            X=X,
            forecasting_horizon=forecasting_horizon,
            **params,
        )

        return self

    def _predict_one(
        self,
        panel_group_names: list[str],
        **params,
    ) -> pl.DataFrame:
        """Predicts `_fit_forecasting_horizon` steps from the observation horizon.

        Parameters
        ----------
        panel_group_names : list of str
            Panel group names to predict for.
        **params : dict
            Additional parameters for prediction.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        # Non-panel data
        if self.panel_group_names_ is None:
            y_pred = self._y_observed.select(~cs.by_name("time"))
            if self.fit_forecasting_horizon_ > self.seasonality:
                # Number of full repetitions needed
                n_repeats = (
                    self.fit_forecasting_horizon_ + self.seasonality - 1
                ) // self.seasonality
                y_pred = pl.concat([y_pred] * n_repeats)

            y_pred = y_pred.head(self.fit_forecasting_horizon_)

        # Panel data
        else:
            y_pred = []
            for panel_group_name in panel_group_names:
                y_group = self._y_observed[panel_group_name]
                y_pred_group = y_group.select(~cs.by_name("time"))

                if self.fit_forecasting_horizon_ > self.seasonality:
                    # Number of full repetitions needed
                    n_repeats = (
                        self.fit_forecasting_horizon_ + self.seasonality - 1
                    ) // self.seasonality
                    y_pred_group = pl.concat([y_pred_group] * n_repeats)

                y_pred_group = y_pred_group.head(self.fit_forecasting_horizon_)

                # Rename columns to add panel prefix
                y_pred_group = y_pred_group.rename(
                    {col: f"{panel_group_name}__{col}" for col in y_pred_group.columns}
                )

                y_pred.append(y_pred_group)

            # Concatenate horizontally (side by side)
            y_pred = pl.concat(y_pred, how="horizontal")

        y_pred = self._add_time_columns(y_pred)

        return y_pred
