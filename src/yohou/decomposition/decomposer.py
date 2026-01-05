"""Decomposer meta-forecaster for residual-based sequential decomposition."""
from alembic.command import current
from yohou.utils import add_interval
from jedi.debug import reset_time
from copy import deepcopy

import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.base import _fit_context, clone
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing,
)
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseForecaster, BaseTransformer
from yohou.point_forecaster.base import BasePointForecaster


class Decomposer(BasePointForecaster, _BaseComposition):
    """Meta-forecaster that decomposes time series into sequential components.

    Decomposer fits multiple forecasters sequentially, where each forecaster
    models the residuals from all previous forecasters. This enables classic
    decomposition patterns like trend + seasonality + residual, or more
    complex multi-component models.

    The final prediction is the sum of predictions from all component forecasters.

    Parameters
    ----------
    forecasters : list of (str, BaseForecaster) tuples
        List of (name, forecaster) tuples specifying the forecaster objects
        to be applied sequentially. All forecasters must have "point" in
        their prediction_types.

        Typical ordering: trend → seasonality → residual

        name : str
            Unique name for the forecaster component.
        forecaster : BaseForecaster
            Forecaster object with "point" in prediction_types.

    store_residuals : bool, default=False
        If True, stores residuals after each component in `self.residuals_`
        dict for inspection. Keys are forecaster names, values are pl.DataFrame
        with residuals.

    target_transformer : BaseTransformer or None, default=None
        Transformer applied to target time series before decomposition.
        Use `target_transformer=LogTransform()` for multiplicative decomposition
        (additive in log-space).

    feature_transformer : BaseTransformer or None, default=None
        Transformer applied to exogenous features before passing to component
        forecasters. Applied once at Decomposer level; all components receive
        the same transformed features.

    Attributes
    ----------
    forecasters_ : list of (str, BaseForecaster) tuples
        Fitted forecasters.

    residuals_ : dict of str to pl.DataFrame
        Residuals after each component (only if store_residuals=True).
        Keys are forecaster names, values are DataFrames with residuals.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.decomposition import Decomposer
    >>> from yohou.point_forecaster import PolynomialTrendForecaster, SeasonalNaive
    >>> from yohou.preprocessing import LogTransform
    >>>
    >>> # Create example time series
    >>> time = pl.datetime_range(
    ...     start=datetime(2020, 1, 1),
    ...     end=datetime(2020, 12, 31),
    ...     interval="1d",
    ...     eager=True
    ... )
    >>> y = pl.DataFrame({
    ...     "time": time,
    ...     "value": range(len(time))
    ... })
    >>>
    >>> # Additive decomposition: trend + seasonality
    >>> forecaster = Decomposer([
    ...     ("trend", PolynomialTrendForecaster(degree=1)),
    ...     ("seasonality", SeasonalNaive(seasonality=7))
    ... ])
    >>> forecaster.fit(y, forecasting_horizon=7)
    Decomposer(...)
    >>> y_pred = forecaster.predict(forecasting_horizon=7)
    >>>
    >>> # Multiplicative decomposition using LogTransform
    >>> forecaster_mult = Decomposer(
    ...     [
    ...         ("trend", PolynomialTrendForecaster(degree=2)),
    ...         ("seasonality", SeasonalNaive(seasonality=7))
    ...     ],
    ...     target_transformer=LogTransform()
    ... )
    >>> forecaster_mult.fit(y, forecasting_horizon=7)
    Decomposer(...)
    >>>
    >>> # Inspect residuals
    >>> forecaster_inspect = Decomposer(
    ...     [
    ...         ("trend", PolynomialTrendForecaster(degree=1)),
    ...         ("seasonality", SeasonalNaive(seasonality=7))
    ...     ],
    ...     store_residuals=True
    ... )
    >>> forecaster_inspect.fit(y, forecasting_horizon=7)
    Decomposer(...)
    >>> trend_residuals = forecaster_inspect.residuals_["trend"]

    Notes
    -----
    - Components are fitted sequentially (not in parallel) to maintain residual consistency
    - Each component models residuals from all previous components
    - All forecasters must be point forecasters (no interval forecasters)
    - Use target_transformer=LogTransform() for multiplicative decomposition
    - Predictions are summed across all components

    """

    _parameter_constraints: dict = {
        **BasePointForecaster._parameter_constraints,
        "forecasters": [list],
        "store_residuals": ["boolean"],
    }

    def __init__(
        self,
        forecasters: list[tuple[str, BaseForecaster]],
        store_residuals: bool = False,
        target_transformer: BaseTransformer | None = None,
    ):
        BasePointForecaster.__init__(
            self,
            target_transformer=target_transformer,
            input_features="X",
        )
        self.forecasters = forecasters
        self.store_residuals = store_residuals

    # @property
    # def observation_horizon(self) -> int:
    #     """Get the maximum observation horizon across all component forecasters.

    #     Returns
    #     -------
    #     int
    #         Maximum observation horizon needed by any component.

    #     Raises
    #     ------
    #     NotFittedError
    #         If the decomposer has not been fitted yet.

    #     """
    #     check_is_fitted(self, "forecasters_")
    #     return sum(forecaster.observation_horizon for _, forecaster in self.forecasters_)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "Decomposer":
        """Fit all component forecasters sequentially on residuals.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with "time" column (datetime type).
        X : pl.DataFrame, optional
            Exogenous features with "time" column.
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.
        **params : dict
            Additional metadata (routed via sklearn's metadata routing).

        Returns
        -------
        self
            Fitted decomposer.

        """
        # Validate forecaster names are unique
        self._validate_names([name for name, _ in self.forecasters])

        # Apply transformers and get transformed data
        y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)

        # Validate all forecasters are point forecasters
        for name, forecaster in self.forecasters:
            if "point" not in forecaster.prediction_types:
                raise ValueError(
                    f"All forecasters must have 'point' in prediction_types. "
                    f"Forecaster '{name}' has prediction_types: {forecaster.prediction_types}"
                )

        # Process metadata routing
        routed_params = process_routing(self, "fit", **params)

        # Fit forecasters sequentially on residuals
        self.forecasters_ = []
        if self.store_residuals:
            self.residuals_ = {}

        residuals = y_t

        for name, forecaster in self.forecasters:
            # Clone and fit forecaster on current residuals
            forecaster_clone = clone(forecaster)

            # Get routed params for this forecaster
            forecaster_params = routed_params.get(name, {}).get("fit", {})

            forecaster_clone.fit(
                y=residuals,
                X=X_t,
                forecasting_horizon=forecasting_horizon,
                **forecaster_params,
            )
            self.forecasters_.append((name, forecaster_clone))

            # Store predictions on training data (needed for residuals)
            # Use predict_transformed=True to avoid inverse transform
            forecaster_clone_pred = deepcopy(forecaster_clone)
            forecaster_observation_horizon = forecaster_clone_pred.observation_horizon
            if forecaster_clone_pred.feature_transformer is not None:
                # If there is a feature transformer, we need enough data to reset it and update_transform for the last point
                feature_observation_horizon = (
                    forecaster_clone_pred.feature_transformer_.observation_horizon
                ) + 1
                forecaster_observation_horizon = max(
                    forecaster_observation_horizon, feature_observation_horizon
                )

            if not forecaster_observation_horizon:
                reset_time = add_interval(
                    residuals["time"][0], interval=forecaster_clone_pred.interval_, n=-1
                )
                y_reset = pl.DataFrame({"time": [reset_time]})
                X_reset, X_pred = None, None
                if X_t is not None: 
                    X_reset = pl.DataFrame({"time": [reset_time]})
                    X_pred = X_t
            else:
                y_reset = residuals[:forecaster_observation_horizon]
                X_reset, X_pred = None, None
                if X_t is not None:
                    X_reset = X_t[:forecaster_observation_horizon]
                    X_pred = X_t[forecaster_observation_horizon:]

            forecaster_clone_pred.reset(y=y_reset, X=X_reset)

            y_pred_train = forecaster_clone_pred.predict(
                X=X_pred,
                forecasting_horizon=len(residuals) - forecaster_observation_horizon,
            )

            # Align predictions with current residuals on time
            aligned = residuals.join(
                y_pred_train.select(~cs.by_name("observed_time")),
                on="time",
                how="inner",
                suffix="_pred",
            )

            # Calculate residuals (actual - predicted)
            target_cols = [c for c in residuals.columns if c != "time"]
            residuals = aligned.select(
                [pl.col("time")]
                + [
                    (pl.col(col) - pl.col(f"{col}_pred")).alias(col)
                    for col in target_cols
                ]
            )

            # Store residuals if requested
            if self.store_residuals:
                self.residuals_[name] = residuals

        return self

    def predict(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        panel_group: str | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Generate forecasts by summing predictions from all components.

        Parameters
        ----------
        X : pl.DataFrame, optional
            Future exogenous features (must have forecasting_horizon rows).
        forecasting_horizon : int, optional
            Number of steps to forecast. If None, uses horizon from fit().
        panel_group : str or None, default=None
            For panel data: predict only for specified group.
        predict_transformed : bool, default=False
            If True, return predictions in transformed space without inverse transform.
        **params : dict
            Additional metadata.

        Returns
        -------
        pl.DataFrame
            Predictions with columns: "observed_time", "time", <target_columns>

        """
        check_is_fitted(self, "forecasters_")

        # Use fit horizon if not specified
        if forecasting_horizon is None:
            forecasting_horizon = self.fit_forecasting_horizon_

        # Process metadata routing
        routed_params = process_routing(self, "predict", **params)

        X_t = X

        # Get predictions from all forecasters and sum them
        y_pred_sum = None
        time_cols = None

        for name, forecaster in self.forecasters_:
            # Get routed params for this forecaster
            forecaster_params = routed_params.get(name, {}).get("predict", {})

            y_pred = forecaster.predict(
                X=X_t,
                forecasting_horizon=forecasting_horizon,
                predict_transformed=True,
                **forecaster_params,
            )

            # Store time columns from first prediction
            if time_cols is None:
                time_cols = y_pred.select("observed_time", "time")

            # Extract values (without time columns) and sum
            y_pred_values = y_pred.select(~cs.by_name("observed_time", "time"))

            if y_pred_sum is None:
                y_pred_sum = y_pred_values
            else:
                y_pred_sum = y_pred_sum + y_pred_values

        # Combine time columns with summed values
        y_pred = pl.concat([time_cols, y_pred_sum], how="horizontal")

        # Apply inverse target transform if needed
        if not predict_transformed and self.target_transformer_ is not None:
            # Remove observed_time before inverse transform
            observed_time = y_pred.select("observed_time")
            y_pred_no_obs = y_pred.select(~cs.by_name("observed_time"))

            y_pred_inv = self.target_transformer_.inverse_transform(
                X_t=y_pred_no_obs, X_p=self._y_observed
            )

            # Add observed_time back
            y_pred = pl.concat([observed_time, y_pred_inv], how="horizontal")

        return y_pred

    def update(self, y: pl.DataFrame, X: pl.DataFrame | None = None) -> "Decomposer":
        """Update all component forecasters with new observations.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations with "time" column.
        X : pl.DataFrame, optional
            New exogenous features with "time" column.

        Returns
        -------
        self
            Updated decomposer.

        """
        check_is_fitted(self, "forecasters_")

        # TODO: Use base class func valid or panel data
        # Update transformers first
        if self.target_transformer_ is not None:
            self.target_transformer_.update(y)
            y_t = self.target_transformer_.transform(y)
        else:
            y_t = y

        if X is not None and self.feature_transformer_ is not None:
            self.feature_transformer_.update(X)
            X_t = self.feature_transformer_.transform(X)
        else:
            X_t = X

        # Update all forecasters
        residuals = y_t
        for name, forecaster in self.forecasters_:
            # Get predictions on new data to compute residuals
            y_pred = forecaster.update_predict(
                y=residuals,
                X=X_t,
                forecasting_horizon=len(residuals),
            )
            # Align predictions with current residuals on time
            aligned = residuals.join(
                y_pred.select(~cs.by_name("observed_time")),
                on="time",
                how="inner",
                suffix="_pred",
            )

            # Calculate residuals (actual - predicted)
            target_cols = [c for c in residuals.columns if c != "time"]
            residuals = aligned.select(
                [pl.col("time")]
                + [
                    (pl.col(col) - pl.col(f"{col}_pred")).alias(col)
                    for col in target_cols
                ]
            )

            # Store residuals if requested
            if self.store_residuals:
                self.residuals_[name] = pl.concat(
                    [self.residuals_[name], residuals],
                )

        # Update base class observation buffers
        self._y_observed = y_t
        if X_t is not None:
            self._X_observed = X_t

        return self

    def reset(self, y: pl.DataFrame, X: pl.DataFrame | None = None) -> "Decomposer":
        """Reset all component forecasters to new observation window.

        Parameters
        ----------
        y : pl.DataFrame
            Target observations with "time" column.
        X : pl.DataFrame, optional
            Exogenous features with "time" column.

        Returns
        -------
        self
            Reset decomposer.

        """
        check_is_fitted(self, "forecasters_")

        # Reset transformers first
        if self.target_transformer_ is not None:
            self.target_transformer_.reset(y)
            y_t = self.target_transformer_.transform(y)
        else:
            y_t = y

        if X is not None and self.feature_transformer_ is not None:
            self.feature_transformer_.reset(X)
            X_t = self.feature_transformer_.transform(X)
        else:
            X_t = X

        # Reset all forecasters
        for _, forecaster in self.forecasters_:
            forecaster.reset(y_t, X=X_t)

        # Reset base class observation buffers
        self._y_observed = y_t
        if X_t is not None:
            self._X_observed = X_t

        return self

    def get_metadata_routing(self):
        """Get metadata routing for this estimator.

        Returns
        -------
        MetadataRouter
            Metadata routing configuration.

        """
        router = MetadataRouter(owner=self.__class__.__name__)

        # Add routing for each forecaster
        for name, forecaster in self.forecasters:
            router.add(
                **{name: forecaster},
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="predict", callee="predict"),
            )

        # Add routing for transformers
        if self.target_transformer is not None:
            router.add(
                target_transformer=self.target_transformer,
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="fit", callee="transform")
                .add(caller="predict", callee="transform"),
            )

        if self.feature_transformer is not None:
            router.add(
                feature_transformer=self.feature_transformer,
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="fit", callee="transform")
                .add(caller="predict", callee="transform"),
            )

        return router
