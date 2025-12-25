"""Base class for point forecasters."""

import abc
from copy import deepcopy

import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.utils.validation import check_is_fitted

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
        forecasting_horizon: StrictInt = 1,
        cross_learning_group: str | None = None,
        predict_transformed: bool = True,
    ) -> pl.DataFrame:
        """Predicts the model forecasting horizon from the observation horizon.

        Parameters
        ----------
        X_ante : pl.DataFrame or None, default=None
            Ex-post feature time series.

        forecasting_horizon : int >= 1, default=1
            Horizon to forecast.

        cross_learning_group : str or None, default=None
            For panel data (local_group_names_ is not None):
            - If None: predict for all groups (default behavior)
            - If str: predict only for the specified group (cross-learning)
            For global data: parameter is ignored.

        predict_transformed : bool, default=True
            If ``True``, the predictions are returned in the transformed space.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        check_is_fitted(self, "fit_forecasting_horizon_")
        forecaster = deepcopy(self)

        if cross_learning_group is not None:
            if (
                self.local_group_names_ is None
                or cross_learning_group not in self.local_group_names_
            ):
                raise ValueError(
                    f"Group {cross_learning_group} not found in local groups: {self.local_group_names_}"
                )

            forecaster.local_group_names_ = [cross_learning_group]

            # Filter _y_observed
            if forecaster._y_observed is not None:
                cols_to_keep = [
                    c
                    for c in forecaster._y_observed.columns
                    if c == "time" or c == cross_learning_group
                ]
                forecaster._y_observed = forecaster._y_observed.select(cols_to_keep)

            # Filter _X_post_observed
            if forecaster._X_post_observed is not None:
                cols_to_keep = [
                    c
                    for c in forecaster._X_post_observed.columns
                    if c == "time"
                    or c == cross_learning_group
                    or c not in self.local_group_names_
                ]
                forecaster._X_post_observed = forecaster._X_post_observed.select(cols_to_keep)

            # Filter X_ante
            if X_ante is not None:
                cols_to_keep = [
                    c
                    for c in X_ante.columns
                    if c == "time"
                    or c == cross_learning_group
                    or c not in self.local_group_names_
                ]
                X_ante = X_ante.select(cols_to_keep)

        y_pred = pl.DataFrame()
        for step in range(1, forecasting_horizon + 1, self.fit_forecasting_horizon_):
            y_pred_step, y_pred_step_inv = BaseForecaster._predict(
                forecaster, cross_learning_group
            )

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
