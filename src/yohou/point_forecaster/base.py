"""Base class for point forecasters."""

import abc
from copy import deepcopy

import polars as pl
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
    input_features: "X" | "y_t|X" | "y|X", default="y_t|X"
        Defines how the input to the ``feature_transformer`` is built.
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
        BaseForecaster._pre_fit(
            self,
            y=y,
            X=X,
            forecasting_horizon=forecasting_horizon,
        )

        return self

    def predict(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        cross_learning_group: str | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Predicts the model forecasting horizon from the observation horizon.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous feature time series.
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

        # Handle panel data: predict all panel groups if cross_learning_group=None
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

            # Filter X
            if X is not None:
                X = filter_panel_columns(
                    X,
                    cross_learning_group,
                    self.local_group_names_,
                    include_global=True,
                )

        y_pred = pl.DataFrame()
        for step in range(0, forecasting_horizon, self.fit_forecasting_horizon_):
            y_pred_step, y_pred_step_inv = BaseForecaster._predict(forecaster, cross_learning_group)

            # Choose which version to accumulate based on predict_transformed
            if predict_transformed:
                y_pred = pl.concat([y_pred, y_pred_step])
            else:
                y_pred = pl.concat([y_pred, y_pred_step_inv])

            if step + self.fit_forecasting_horizon_ < forecasting_horizon:
                # Use inverse-transformed predictions for recursive update
                # For both global and panel data, select columns from local_y_schema_
                y = y_pred_step_inv.select(["time"] + list(self.local_y_schema_.keys()))

                X_slice = None
                if X is not None:
                    start_idx = step
                    end_idx = start_idx + self.fit_forecasting_horizon_
                    X_slice = X[start_idx:end_idx]

                    if len(X_slice) != len(y):
                        raise ValueError(
                            f"Missing X for future steps. Needed {len(y)} rows, "
                            f"but X slice has {len(X_slice)} rows."
                        )

                forecaster.update(y, X_slice)

        y_pred = y_pred.with_columns(observed_time=y_pred["observed_time"][0])

        if forecasting_horizon % self.fit_forecasting_horizon_:
            end = (
                self.fit_forecasting_horizon_ - forecasting_horizon % self.fit_forecasting_horizon_
            )
            y_pred = y_pred[:-end]

        return y_pred
