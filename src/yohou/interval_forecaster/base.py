"""Base classes for interval forecasters and similarity measures."""

import abc
from copy import deepcopy
from typing import Any

import numpy as np
import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseForecaster
from yohou.utils import filter_panel_columns, select_struct


class BaseSimilarity(BaseEstimator, metaclass=abc.ABCMeta):
    """Base class for similarity measures used in interval forecasting."""

    @property
    def discarded_time_stamps(self) -> None:
        """Get discarded timestamps (placeholder property).

        Returns
        -------
        None

        """
        return None

    @abc.abstractmethod
    def fit(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_post: pl.DataFrame | None = None,
        X_ante: pl.DataFrame | None = None,
    ) -> "BaseSimilarity":
        """Fit the similarity measure.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        y_pred : pl.DataFrame
            Point predictions.

        X_post : pl.DataFrame or None, default=None
            Ex-ante features.

        X_ante : pl.DataFrame or None, default=None
            Ex-post features.

        Returns
        -------
        self

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def update(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_post: pl.DataFrame | None = None,
        X_ante: pl.DataFrame | None = None,
    ) -> "BaseSimilarity":
        """Update the similarity measure with new observations.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.

        y_pred : pl.DataFrame
            New predictions.

        X_post : pl.DataFrame or None, default=None
            New ex-ante features.

        X_ante : pl.DataFrame or None, default=None
            New ex-post features.

        Returns
        -------
        self

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(
        self,
        y_pred: pl.DataFrame,
        X_post: pl.DataFrame | None = None,
        X_ante: pl.DataFrame | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating[Any]]]:
        """Compute similarity weights for predictions.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Predictions to compute similarities for.

        X_post : pl.DataFrame or None, default=None
            Ex-ante features.

        X_ante : pl.DataFrame or None, default=None
            Ex-post features.

        Returns
        -------
        np.ndarray
            Similarity weights.

        """
        raise NotImplementedError()


class BaseIntervalForecaster(BaseForecaster, metaclass=abc.ABCMeta):
    """Base class for conformal forecasters.

    Parameters
    ----------
    coverage_rates: list of floats, default=[0.05]
        List of miscoverage levels to generate intervals for.

    """

    @property
    def prediction_types(self) -> set[str]:
        """Get the prediction types this forecaster produces.

        Returns
        -------
        set of str
            {"interval"}

        """
        return {"interval"}

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
        X_post: pl.DataFrame | None = None,
        X_ante: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "BaseIntervalForecaster":
        """Fits the forecaster and returns it.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        X_post : pl.DataFrame or None, default=None
            Ex-ante feature time series.

        X_ante : pl.DataFrame or None, default=None
            Ex-post feature time series.

        forecasting_horizon : int >= 1, default=1
            Horizon to forecast.

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self

        """
        raise NotImplementedError()

    # TODO: Separate reduction code?
    def update(
        self,
        y: pl.DataFrame,
        X_post: pl.DataFrame | None = None,
        X_ante: pl.DataFrame | None = None,
    ) -> "BaseIntervalForecaster":
        """Updates the forecaster with more recent data and
        returns it.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        X_post : pl.DataFrame or None
            Ex-ante feature time series.

        X_ante : pl.DataFrame or None
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

        if "point" in self.prediction_types or y_contains_points:
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

        BaseForecaster.update(self, y, X_post, X_ante)

        return self

    def predict(
        self,
        X_ante: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        cross_learning_group: str | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Predicts the model forecasting horizon from the observation horizon.

        Parameters
        ----------
        X_ante : pl.DataFrame or None, default=None
            Ex-post feature time series.

        forecasting_horizon : int >= 1 or None, default=None
            Horizon to forecast. If None, uses ``fit_forecasting_horizon_``.

        cross_learning_group : str or None, default=None
            For panel data (local_group_names_ is not None):
            - If None: predict for all groups (default behavior)
            - If str: predict only for the specified group (cross-learning)
            For global data: parameter is ignored.

        predict_transformed : bool, default=False
            If ``True``, the predictions are returned in the transformed space.

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Predicted time series with interval bounds.

        """
        check_is_fitted(self, "fit_forecasting_horizon_")

        # Use fit_forecasting_horizon_ as default
        if forecasting_horizon is None:
            forecasting_horizon = self.fit_forecasting_horizon_

        # Validate cross_learning_group only if provided
        if cross_learning_group is not None and (
            self.local_group_names_ is None or cross_learning_group not in self.local_group_names_
        ):
            raise ValueError(
                f"Group {cross_learning_group} not found in local groups: {self.local_group_names_}"
            )

        # Handle panel data: predict all struct columns if cross_learning_group=None
        # For now, just predict all groups together (default behavior)
        # TODO: Implement individual group predictions if needed

        forecaster = deepcopy(self)

        if self.local_group_names_ is not None and cross_learning_group is not None:
            # Filter _y_observed
            if forecaster._y_observed is not None:
                forecaster._y_observed = filter_panel_columns(
                    forecaster._y_observed,
                    cross_learning_group,
                    self.local_group_names_,
                    include_global=False,
                )

            # Filter _X_post_observed
            if forecaster._X_post_observed is not None:
                forecaster._X_post_observed = filter_panel_columns(
                    forecaster._X_post_observed,
                    cross_learning_group,
                    self.local_group_names_,
                    include_global=True,
                )

            # Filter X_ante
            if X_ante is not None:
                X_ante = filter_panel_columns(
                    X_ante,
                    cross_learning_group,
                    self.local_group_names_,
                    include_global=True,
                )

        y_pred = pl.DataFrame()
        for step in range(1, forecasting_horizon + 1, self.fit_forecasting_horizon_):
            y_pred_step, y_pred_step_inv = BaseForecaster._predict(forecaster, cross_learning_group)

            # Choose which version to accumulate based on predict_transformed
            if predict_transformed:
                y_pred = pl.concat([y_pred, y_pred_step])
            else:
                y_pred = pl.concat([y_pred, y_pred_step_inv])

            if step + self.fit_forecasting_horizon_ <= forecasting_horizon:
                time = y_pred_step.select(cs.by_name("time"))
                # Guard against None - X_post_observed should be set after fit
                if forecaster._X_post_observed is not None:
                    X_post_old = forecaster._X_post_observed[[-1]].select(~cs.by_name("time"))
                    X_post = pl.concat([X_post_old] * len(time))
                    X_post = pl.concat([time, X_post], how="horizontal")
                else:
                    X_post = None

                # Compute midpoints from intervals for recursive update
                if self.local_group_names_ is None:
                    # Global data case
                    y_data = {"time": time["time"]}
                    for col in self.local_y_names_:
                        # Find all coverage rates for this column
                        lower_cols = [
                            c for c in y_pred_step_inv.columns if c.startswith(f"{col}_lower_")
                        ]
                        upper_cols = [
                            c for c in y_pred_step_inv.columns if c.startswith(f"{col}_upper_")
                        ]

                        if lower_cols and upper_cols:
                            # Use the first coverage rate to compute midpoint
                            lower_col = sorted(lower_cols)[0]
                            upper_col = sorted(upper_cols)[0]
                            y_data[col] = (
                                y_pred_step_inv[lower_col] + y_pred_step_inv[upper_col]
                            ) / 2
                    y = pl.DataFrame(y_data)
                else:
                    # Panel data case: reconstruct struct columns
                    y_struct_dict = {}
                    for local_group_name in self.local_group_names_:
                        # Unnest the struct column
                        y_local_pred = y_pred_step_inv[[local_group_name]].unnest(local_group_name)

                        y_local_data = {}
                        for col in self.local_y_names_:
                            # Find coverage rate columns for this target
                            lower_cols = [
                                c for c in y_local_pred.columns if c.startswith(f"{col}_lower_")
                            ]
                            upper_cols = [
                                c for c in y_local_pred.columns if c.startswith(f"{col}_upper_")
                            ]

                            if lower_cols and upper_cols:
                                # Use the first coverage rate to compute midpoint
                                lower_col = sorted(lower_cols)[0]
                                upper_col = sorted(upper_cols)[0]
                                y_local_data[col] = (
                                    y_local_pred[lower_col] + y_local_pred[upper_col]
                                ) / 2

                        y_struct_dict[local_group_name] = pl.DataFrame(y_local_data)

                    y = pl.concat([time, pl.DataFrame(y_struct_dict)], how="horizontal")

                forecaster.update(y, X_post, X_ante)

        y_pred = y_pred.with_columns(observed_time=y_pred["observed_time"][0])

        if forecasting_horizon % self.fit_forecasting_horizon_:
            end = (
                self.fit_forecasting_horizon_ - forecasting_horizon % self.fit_forecasting_horizon_
            )
            y_pred = y_pred[:-end]

        return y_pred
