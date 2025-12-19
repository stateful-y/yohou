import abc
from typing import Optional

import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.base import BaseEstimator

from yohou.base import BaseForecaster
from yohou.utils import select_struct


class BaseSimilarity(BaseEstimator, metaclass=abc.ABCMeta):  # type: ignore[misc]
    @property
    def discarded_time_stamps(self) -> None:
        return None

    @abc.abstractmethod
    def fit(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_ante: Optional[pl.DataFrame] = None,
        X_post: Optional[pl.DataFrame] = None,
    ) -> "BaseSimilarity":
        raise NotImplementedError()

    @abc.abstractmethod
    def update(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_ante: Optional[pl.DataFrame] = None,
        X_post: Optional[pl.DataFrame] = None,
    ) -> "BaseSimilarity":
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(
        self,
        y_pred: pl.DataFrame,
        X_ante: Optional[pl.DataFrame] = None,
        X_post: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        raise NotImplementedError()


class BaseIntervalForecaster(BaseForecaster, metaclass=abc.ABCMeta):
    """Base class for conformal forecasters.

    Parameters
    ----------
    coverage_rates: list of floats, default=[0.05]
        List of miscoverage levels to generate intervals for.

    """

    _prediction_type = "interval"

    def __init__(
        self,
        coverage_rates: list[float],
        update_strategy: str,
    ) -> None:
        self.coverage_rates = coverage_rates
        self.update_strategy = update_strategy

    @abc.abstractmethod
    def fit(
        self,
        y: pl.DataFrame,
        X_ante: Optional[pl.DataFrame] = None,
        X_post: Optional[pl.DataFrame] = None,
        forecasting_horizon: StrictInt = 1,
    ) -> "BaseIntervalForecaster":
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
        raise NotImplementedError()

    # TODO: Separate reduction code?
    def update(
        self,
        y: pl.DataFrame,
        X_ante: Optional[pl.DataFrame],
        X_post: Optional[pl.DataFrame],
    ) -> "BaseIntervalForecaster":
        """Updates the forecaster with more recent data and
        returns it.

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
        y_contains_points = (
            self.local_group_names_ is None and set(self.local_y_names_) <= set(y.columns)
        ) or (
            self.local_group_names_ is not None
            and set(self.local_y_names_) <= set(y.unnest(self.local_group_names_[0]).columns)
        )

        if "point" in self.prediction_type or y_contains_points:
            y = select_struct(y, local_col_names=self.local_y_names_, select_time=True)

        else:
            time = y.select(cs.by_name("time"))

            match self.update_strategy:
                case "average":
                    if self.local_group_names_ is not None:
                        y_groups = pl.DataFrame()
                        for local_group_name in self.local_group_names_:
                            y_local = y[
                                [
                                    col
                                    for col, dtype in y.schema.items()
                                    if dtype != pl.Struct or col == local_group_name
                                ]
                            ].unnest(local_group_name)

                            y_local = y_local.select(
                                [
                                    pl.concat_list(
                                        [
                                            f"{col}_lower_{coverage_rate}"
                                            for coverage_rate in self.coverage_rates
                                        ]
                                        + [
                                            f"{col}_upper_{coverage_rate}"
                                            for coverage_rate in self.coverage_rates
                                        ]
                                    )
                                    .list.mean()
                                    .alias(col)
                                    for col in self.local_y_names_
                                ]
                            )

                            y_groups = pl.concat(
                                [y_groups, pl.DataFrame({local_group_name: y_local})],
                                how="horizontal",
                            )

                        y = y_groups

                    else:
                        y = y.select(
                            [
                                pl.concat_list(
                                    [
                                        f"{col}_lower_{coverage_rate}"
                                        for coverage_rate in self.coverage_rates
                                    ]
                                    + [
                                        f"{col}_upper_{coverage_rate}"
                                        for coverage_rate in self.coverage_rates
                                    ]
                                )
                                .list.mean()
                                .alias(col)
                                for col in self.local_y_names_
                            ]
                        )

                case "constant":
                    y_old = self._y_observed[[-1]].select(~cs.by_name("time"))
                    y = pl.concat([y_old] * len(time))

            y = pl.concat([time, y], how="horizontal")

        return BaseForecaster.update(self, y, X_ante, X_post)
