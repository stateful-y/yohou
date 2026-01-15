"""Base class for point forecasters."""

import abc
from copy import deepcopy

import polars as pl
from pydantic import StrictInt

from yohou.base import BaseForecaster, Tags
from yohou.utils import select_panel_columns
from yohou.utils.validation import validate_data


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

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        tags.forecaster_tags.forecaster_type = "point"
        return tags

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
        panel_group_names: list[str] | None = None,
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
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data:
            - If None: predict for all groups
            - If list of str: predict only for the specified panel groups
            Parameter is ignored if the forecaster was not fitted on panel data.
        predict_transformed : bool, default=False
            If ``True``, the predictions are returned in the transformed space.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        _, X, panel_group_names = validate_data(
            self,
            y=None,
            X=X,
            reset=False,
            panel_group_names=panel_group_names,
            check_continuity=False,
        )

        # Use fit_forecasting_horizon_ as default
        if forecasting_horizon is None:
            forecasting_horizon = self.fit_forecasting_horizon_

        forecaster = deepcopy(self)

        if panel_group_names is not None:
            # Filter X
            if X is not None:
                X = select_panel_columns(
                    X,
                    panel_group_names,
                    include_global=True,
                )

        y_pred = pl.DataFrame()
        for step in range(0, forecasting_horizon, self.fit_forecasting_horizon_):
            y_pred_step, y_pred_step_inv = BaseForecaster._predict(forecaster, panel_group_names)

            # Choose which version to accumulate based on predict_transformed
            if predict_transformed:
                y_pred = pl.concat([y_pred, y_pred_step])
            else:
                y_pred = pl.concat([y_pred, y_pred_step_inv])

            if step + self.fit_forecasting_horizon_ < forecasting_horizon:
                # Use inverse-transformed predictions for recursive update
                # Select columns based on whether we have panel data or not
                if self.panel_group_names_ is None:
                    # Non-panel data: schemas contain actual column names
                    y = y_pred_step_inv.select(["time"] + list(self.local_y_schema_.keys()))
                else:
                    # Panel data: reconstruct prefixed column names from schema
                    y_columns = ["time"]
                    for group_name in self.panel_group_names_:
                        y_columns.extend(
                            [f"{group_name}__{col}" for col in self.local_y_schema_.keys()]
                        )
                    y = y_pred_step_inv.select(y_columns)

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

    def update_predict(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        panel_group_names: list[str] | None = None,
        stride: StrictInt | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Alternate recursive `predict` and `update`.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series for updates.
        X : pl.DataFrame or None, default=None
            Feature time series for predictions.
        forecasting_horizon : int >= 1 or None, default=None
            Horizon to forecast recursively. If None, uses ``fit_forecasting_horizon_``.
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data:
            - If None: predict for all groups
            - If list of str: update and predict only for the specified panel groups
            Parameter is ignored if the forecaster was not fitted on panel data.
        stride : int >= 1 or None, default=None
            Number of new observations to use for each update. If None, uses
            ``fit_forecasting_horizon_``.
        predict_transformed : bool, default=False
            If ``True``, the predictions are returned in the transformed
            space.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        y, X, panel_group_names = validate_data(
            self,
            y=y,
            X=X,
            reset=False,
            panel_group_names=panel_group_names,
            check_continuity=True,
        )

        # Use fit_forecasting_horizon_ as default for both parameters
        if forecasting_horizon is None:
            forecasting_horizon = self.fit_forecasting_horizon_
        if stride is None:
            stride = self.fit_forecasting_horizon_

        # Initial prediction with predict_transformed parameter
        y_pred_i = self.predict(
            X=X,
            forecasting_horizon=forecasting_horizon,
            panel_group_names=panel_group_names,
            predict_transformed=predict_transformed,
            **params,
        )

        y_pred = y_pred_i
        for i in range(0, len(y), stride):
            y_slice = y[i : i + stride]

            X_slice = None
            if X is not None:
                # Filter X to match y_slice times
                # Use semi-join to f ilter X rows that have matching times in y_slice
                X_slice = X.join(y_slice.select("time"), on="time", how="semi")

            self.update(y=y_slice, X=X_slice, panel_group_names=panel_group_names)

            X_future = None
            if X is not None:
                # Filter X to start after the last observed time
                # This ensures predict() gets features aligned with the forecast horizon
                last_time = y_slice["time"][-1]
                X_future = X.filter(pl.col("time") > last_time)

            y_pred_i = self.predict(
                X=X_future,
                forecasting_horizon=forecasting_horizon,
                panel_group_names=panel_group_names,
                predict_transformed=predict_transformed,
                **params,
            )

            y_pred = pl.concat([y_pred, y_pred_i])

        return y_pred
