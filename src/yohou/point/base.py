"""Base class for point forecasters."""

import abc
from copy import deepcopy

import polars as pl
from pydantic import StrictInt
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseForecaster
from yohou.utils import Tags, select_panel_columns, validate_forecaster_data

__all__ = ["BasePointForecaster"]


class BasePointForecaster(BaseForecaster, metaclass=abc.ABCMeta):
    """Base class for point forecasters.

    Parameters
    ----------
    target_transformer : instance of `BaseTransformer` or None, default=None
        Transformer used to transform the target time series into the new target.
    feature_transformer : instance of `BaseTransformer` or None, default=None
        Transformer used to transform the target time series into features.
    target_as_feature : {"transformed", "raw"} or None, default="transformed"
        Controls whether the target is included as a feature.
        ``"transformed"`` includes the transformed target, ``"raw"``
        includes the raw target, and ``None`` uses only exogenous features.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See `BaseForecaster` for details.

    Notes
    -----
    Subclasses must implement ``_predict_one`` to produce point
    predictions for a single forecast step.  The ``forecaster_type``
    tag is set to ``"point"``.

    See Also
    --------
    `PointReductionForecaster` : ML-based point forecaster.
    `SeasonalNaive` : Simple seasonal naive forecaster.
    BaseIntervalForecaster : Base class for interval forecasters.

    """

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.forecaster_type = "point"
        return tags

    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "BasePointForecaster":
        """Fit the forecaster to historical data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        forecasting_horizon : int, default=1
            Number of time steps to forecast into the future.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted forecaster instance.

        Raises
        ------
        ValueError
            If ``forecasting_horizon`` < 1, or if ``y`` / ``X`` have invalid
            structure (e.g., missing ``"time"`` column).

        """

        forecasting_horizon = self._validate_fit_params(forecasting_horizon)

        BaseForecaster._pre_fit(
            self,
            y=y,
            X=X,
            forecasting_horizon=forecasting_horizon,
        )

        return self

    def _validate_fit_params(self, forecasting_horizon: StrictInt) -> StrictInt:
        """Validate fit parameters.

        Parameters
        ----------
        forecasting_horizon : int
            Forecasting horizon to validate.

        Returns
        -------
        int
            Validated forecasting horizon.

        Raises
        ------
        ValueError
            If forecasting_horizon < 1.

        """
        if forecasting_horizon < 1:
            raise ValueError(f"forecasting_horizon must be >= 1, got {forecasting_horizon}")
        return forecasting_horizon

    def _validate_predict_params(self, forecasting_horizon: StrictInt | None) -> StrictInt:
        """Validate and return predict parameters.

        Parameters
        ----------
        forecasting_horizon : int or None
            Forecasting horizon to validate. If None, uses fit_forecasting_horizon_.

        Returns
        -------
        int
            Validated forecasting horizon.

        Raises
        ------
        ValueError
            If forecasting_horizon < 1.

        """
        if forecasting_horizon is None:
            forecasting_horizon = self.fit_forecasting_horizon_
        return self._validate_fit_params(forecasting_horizon)

    def predict(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        panel_group_names: list[str] | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Generate point forecasts.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        panel_group_names : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.
        predict_transformed : bool, default=False
            If ``True``, return predictions in the transformed space without
            applying inverse target transformation.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Point predictions with ``"observed_time"``, ``"time"``, and one
            column per target variable.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forecaster has not been fitted yet.
        ValueError
            If ``X`` has invalid structure or ``panel_group_names`` contains
            names not seen during fit.

        """
        check_is_fitted(
            self,
            ["local_y_schema_", "local_X_schema_", "shared_X_schema_", "panel_group_names_"],
        )

        _, X, panel_group_names = validate_forecaster_data(
            self,
            y=None,
            X=X,
            reset=False,
            panel_group_names=panel_group_names,
        )

        forecasting_horizon = self._validate_predict_params(forecasting_horizon)

        forecaster = deepcopy(self)

        if panel_group_names is not None and X is not None:
            # Filter X
            X = select_panel_columns(
                X,
                panel_group_names,
                include_global=True,
            )

        y_pred = pl.DataFrame()
        for step in range(0, forecasting_horizon, self.fit_forecasting_horizon_):
            y_pred_step, y_pred_step_inv = forecaster._predict(panel_group_names or [])

            # Choose which version to accumulate based on predict_transformed
            y_pred = pl.concat([y_pred, y_pred_step]) if predict_transformed else pl.concat([y_pred, y_pred_step_inv])

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
                        y_columns.extend([f"{group_name}__{col}" for col in self.local_y_schema_])
                    y = y_pred_step_inv.select(y_columns)

                X_slice = None
                if X is not None:
                    start_idx = step
                    end_idx = start_idx + self.fit_forecasting_horizon_
                    X_slice = X[start_idx:end_idx]

                    if len(X_slice) != len(y):
                        raise ValueError(
                            f"Missing X for future steps. Needed {len(y)} rows, but X slice has {len(X_slice)} rows."
                        )

                forecaster.observe(y, X_slice)

        y_pred = y_pred.with_columns(observed_time=y_pred["observed_time"][0])

        if forecasting_horizon % self.fit_forecasting_horizon_:
            end = self.fit_forecasting_horizon_ - forecasting_horizon % self.fit_forecasting_horizon_
            y_pred = y_pred[:-end]

        return y_pred

    def observe_predict(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        panel_group_names: list[str] | None = None,
        stride: StrictInt | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Alternate recursive predict and observe.

        Equivalent to calling ``observe(y, X)`` then ``predict(X)``.  Returns
        point predictions.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        panel_group_names : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.
        stride : int or None, default=None
            Step size for rolling update-predict.  If ``None``, defaults to
            ``forecasting_horizon``.
        predict_transformed : bool, default=False
            If ``True``, return predictions in the transformed space without
            applying inverse target transformation.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Point predictions with ``"observed_time"``, ``"time"``, and one
            column per target variable.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forecaster has not been fitted yet.
        ValueError
            If ``y`` / ``X`` have invalid structure or ``panel_group_names``
            contains names not seen during fit.

        """
        check_is_fitted(
            self,
            ["local_y_schema_", "local_X_schema_", "shared_X_schema_", "panel_group_names_"],
        )

        y, X, panel_group_names = validate_forecaster_data(
            self,
            y=y,
            X=X,
            reset=False,
            panel_group_names=panel_group_names,
        )

        forecasting_horizon = self._validate_predict_params(forecasting_horizon)
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

            self.observe(y=y_slice, X=X_slice, panel_group_names=panel_group_names)

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
