"""Base class for point forecasters."""

import abc

import polars as pl
from pydantic import StrictInt
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseForecaster
from yohou.utils import POINT, Tags, select_panel_columns, validate_forecaster_data

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
    tag is set to ``POINT``.

    See Also
    --------
    `PointReductionForecaster` : ML-based point forecaster.
    `SeasonalNaive` : Simple seasonal naive forecaster.
    `BaseIntervalForecaster` : Base class for interval forecasters.

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
        tags.forecaster_tags.forecaster_type = POINT
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
        groups: list[str] | None = None,
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
        groups : list of str or None, default=None
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
            If ``X`` has invalid structure or ``groups`` contains
            names not seen during fit.

        """
        check_is_fitted(
            self,
            ["local_y_schema_", "local_X_schema_", "shared_X_schema_", "groups_"],
        )

        _, X, groups = validate_forecaster_data(
            self,
            y=None,
            X=X,
            reset=False,
            groups=groups,
        )

        forecasting_horizon = self._validate_predict_params(forecasting_horizon)

        if groups is not None and X is not None:
            X = select_panel_columns(
                X,
                groups,
                include_global=True,
            )

        def step_fn(forecaster, groups):
            """Produce one point-prediction block."""
            y_pred_step, y_pred_step_inv = forecaster._predict(groups)
            y_accumulate = y_pred_step if predict_transformed else y_pred_step_inv
            return y_accumulate, y_pred_step_inv

        def derive_observation_fn(forecaster, y_pred_step_inv, X, step):
            """Derive observation from inverse-transformed prediction."""
            if self.groups_ is None:
                y = y_pred_step_inv.select(["time"] + list(self.local_y_schema_.keys()))
            else:
                y_columns = ["time"]
                for group_name in self.groups_:
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

            return y, X_slice

        return self._recursive_predict(
            X=X,
            forecasting_horizon=forecasting_horizon,
            groups=groups,
            step_fn=step_fn,
            derive_observation_fn=derive_observation_fn,
        )

    def observe_predict(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
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
        groups : list of str or None, default=None
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
            If ``y`` / ``X`` have invalid structure or ``groups``
            contains names not seen during fit.

        """
        check_is_fitted(
            self,
            ["local_y_schema_", "local_X_schema_", "shared_X_schema_", "groups_"],
        )

        y, X, groups = validate_forecaster_data(
            self,
            y=y,
            X=X,
            reset=False,
            groups=groups,
        )

        forecasting_horizon = self._validate_predict_params(forecasting_horizon)
        if stride is None:
            stride = self.fit_forecasting_horizon_

        return self._observe_predict_loop(
            predict_fn=self.predict,
            y=y,
            X=X,
            groups=groups,
            stride=stride,
            forecasting_horizon=forecasting_horizon,
            predict_transformed=predict_transformed,
            **params,
        )
