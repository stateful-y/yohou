"""Base class for point forecasters."""

import abc
from copy import deepcopy

import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseForecaster
from yohou.utils import filter_panel_columns


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

    @property
    def prediction_types(self) -> set[str]:
        """Get the prediction types this forecaster produces.

        Returns
        -------
        set of str
            {"point"}

        """
        return {"point"}

    def fit(
        self,
        y: pl.DataFrame,
        X_post: pl.DataFrame | None = None,
        X_ante: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "BasePointForecaster":
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
        BaseForecaster._pre_fit(
            self,
            y=y,
            X_post=X_post,
            X_ante=X_ante,
            forecasting_horizon=forecasting_horizon,
        )

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
            Predicted time series.

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

        if self.local_group_names_ and cross_learning_group is not None:
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

                # Use inverse-transformed predictions for recursive update
                if self.local_group_names_ is None:
                    # Global data: select flat columns
                    y = y_pred_step_inv.select(["time"] + self.local_y_names_)
                else:
                    # Panel data: predictions already have struct columns
                    y = y_pred_step_inv.select(["time"] + self.local_group_names_)

                forecaster.update(y, X_post, X_ante)

        y_pred = y_pred.with_columns(observed_time=y_pred["observed_time"][0])

        if forecasting_horizon % self.fit_forecasting_horizon_:
            end = (
                self.fit_forecasting_horizon_ - forecasting_horizon % self.fit_forecasting_horizon_
            )
            y_pred = y_pred[:-end]

        return y_pred
