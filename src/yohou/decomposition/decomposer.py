"""Decomposer meta-forecaster for residual-based sequential decomposition."""

from copy import deepcopy

import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.base import _fit_context, clone
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    process_routing,
)
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseTransformer, Tags
from yohou.point_forecaster.base import BasePointForecaster
from yohou.utils import add_interval, cast, dict_to_panel, get_group_df, validate_forecaster_data


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
        to be applied sequentially. All forecasters must be point forecasters.

        Typical ordering: trend → seasonality → residual

        name : str
            Unique name for the forecaster component.
        forecaster : BaseForecaster
            Point forecaster object.

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
    >>> from yohou.decomposition import Decomposer, PolynomialTrendForecaster
    >>> from yohou.point_forecaster import SeasonalNaive
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
    ...     "value": range(1, len(time) + 1)
    ... })
    >>>
    >>> # Additive decomposition: trend + seasonality
    >>> forecaster = Decomposer([
    ...     ("trend", PolynomialTrendForecaster(degree=1)),
    ...     ("seasonality", SeasonalNaive(seasonality=7))
    ... ])
    >>> forecaster.fit(y, forecasting_horizon=7)  # doctest: +ELLIPSIS
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
    >>> forecaster_mult.fit(y, forecasting_horizon=7)  # doctest: +ELLIPSIS
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
    >>> forecaster_inspect.fit(y, forecasting_horizon=7)  # doctest: +ELLIPSIS
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
        forecasters: list[tuple[str, BasePointForecaster]],
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

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()

        # Check stateful from forecasters
        stateful = False
        for _, f in self.forecasters:
            f_tags = f.__sklearn_tags__()
            if f_tags.forecaster_tags and f_tags.forecaster_tags.stateful:
                stateful = True
                break

        # Also check target transformer
        if not stateful:
            if self.target_transformer is not None:
                stateful = self.target_transformer.__sklearn_tags__().transformer_tags.stateful

        tags.forecaster_tags.stateful = stateful

        # Determine forecaster_type from nested forecasters' tags
        if self.forecasters:
            _, last_forecaster = self.forecasters[-1]
            last_tags = last_forecaster.__sklearn_tags__()
            tags.forecaster_tags.forecaster_type = "point"
            if last_tags.forecaster_tags and last_tags.forecaster_tags.forecaster_type == "both":
                tags.forecaster_tags.forecaster_type = "both"

        # Aggregate other tags
        tags.forecaster_tags.uses_reduction = any(
            getattr(f.__sklearn_tags__().forecaster_tags, "uses_reduction", False)
            for _, f in self.forecasters
        )
        tags.forecaster_tags.supports_panel_data = all(
            getattr(f.__sklearn_tags__().forecaster_tags, "supports_panel_data", True)
            for _, f in self.forecasters
        )

        return tags

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
        forecasting_horizon = self._validate_fit_params(forecasting_horizon)

        # Validate params before routing
        _raise_for_params(params, self, "fit")

        # Validate forecaster names are unique
        self._validate_names([name for name, _ in self.forecasters])

        # Validate all forecasters are point forecasters
        for name, forecaster in self.forecasters:
            if not isinstance(forecaster, BasePointForecaster):
                raise ValueError(
                    f"All forecasters must be instances of BasePointForecaster. "
                    f"Forecaster '{name}' is {type(forecaster).__name__}"
                )

        # Apply transformers and get transformed data
        y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)

        y_t = dict_to_panel(y_t)
        if X_t is not None:
            X_t = dict_to_panel(X_t)

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

            # Get routed params for this forecaster (direct Bunch access)
            step_params = routed_params[name]

            forecaster_clone.fit(
                y=residuals,
                X=X_t,
                forecasting_horizon=forecasting_horizon,
                **step_params.fit,
            )
            self.forecasters_.append((name, forecaster_clone))

            # Store predictions on training data (needed for residuals)
            # Use predict_transformed=True to avoid inverse transform
            forecaster_clone_pred = deepcopy(forecaster_clone)
            forecaster_observation_horizon = forecaster_clone_pred.observation_horizon
            if forecaster_clone_pred.feature_transformer is not None:
                # If there is a feature transformer, we need enough data to
                # reset it and update_transform for the last point
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
                y_reset = pl.DataFrame(
                    {col: [reset_time] if col == "time" else [None] for col in y_t.columns},
                    schema=y_t.schema,
                )
                X_reset, X_pred = None, None
                if X_t is not None:
                    X_reset = pl.DataFrame(
                        {col: [reset_time] if col == "time" else [None] for col in X_t.columns},
                        schema=X_t.schema,
                    )
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
                + [(pl.col(col) - pl.col(f"{col}_pred")).alias(col) for col in target_cols]
            )

            # Store residuals if requested
            if self.store_residuals:
                self.residuals_[name] = residuals

        return self

    def predict(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        panel_group_names: list[str] | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Generate forecasts by summing predictions from all components.

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
            Predictions with columns: "observed_time", "time", <target_columns>

        """
        check_is_fitted(self, ["forecasters_", "panel_group_names_"])
        _, X, panel_group_names = validate_forecaster_data(
            self,
            y=None,
            X=X,
            reset=False,
            panel_group_names=panel_group_names,
        )

        # Use fit horizon if not specified
        if forecasting_horizon is None:
            forecasting_horizon = self.fit_forecasting_horizon_

        # Validate params before routing
        _raise_for_params(params, self, "predict")

        # Process metadata routing
        routed_params = process_routing(self, "predict", **params)

        # Get predictions from all forecasters and sum them
        y_pred_sum = None
        time_cols = None

        for name, forecaster in self.forecasters_:
            # Get routed params for this forecaster (direct Bunch access)
            step_params = routed_params[name]

            y_pred = forecaster.predict(
                X=X,
                forecasting_horizon=forecasting_horizon,
                predict_transformed=True,
                **step_params.predict,
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

        if not predict_transformed and self.target_transformer is not None:
            # Apply inverse target transform

            # Remove observed_time before inverse transform
            observed_time = y_pred.select("observed_time")
            y_pred_no_obs = y_pred.select(~cs.by_name("observed_time"))

            # Handle panel data (target_transformer_ and _y_observed are dicts)
            if self.panel_group_names_ is None:
                # Non-panel data
                y_pred_inv = self.target_transformer_.inverse_transform(
                    X_t=y_pred_no_obs, X_p=self._y_observed
                )

            else:
                # Panel data
                y_pred_inv_dict = {}
                for panel_group_name in panel_group_names or self.panel_group_names_:
                    transformer = self.target_transformer_[panel_group_name]

                    # Skip if no transformer for this group
                    if transformer is None:
                        # No transformation, just rename with prefix
                        y_pred_group = get_group_df(
                            df=y_pred_no_obs,
                            group_name=panel_group_name,
                            schema=self.local_y_schema_,
                        )
                        # Rename to add prefix
                        rename_map = {
                            col: f"{panel_group_name}__{col}"
                            for col in y_pred_group.columns
                            if col != "time"
                        }
                        y_pred_group = y_pred_group.rename(rename_map)
                        y_pred_inv_dict[panel_group_name] = y_pred_group.select(~cs.by_name("time"))
                        continue

                    y_observed_local = self._y_observed[panel_group_name]

                    # Extract group data (unprefixed)
                    y_pred_group = get_group_df(
                        df=y_pred_no_obs,
                        group_name=panel_group_name,
                        schema=self.local_y_schema_,
                    )

                    # Inverse transform (works with unprefixed columns)
                    y_pred_group_inv = transformer.inverse_transform(
                        X_t=y_pred_group, X_p=y_observed_local
                    )

                    # Cast to restore original dtypes
                    y_pred_group_inv_cast = cast(
                        y_pred_group_inv.select(~cs.by_name("time")), self.local_y_schema_
                    )

                    # Rename to add prefix
                    rename_map = {
                        col: f"{panel_group_name}__{col}" for col in y_pred_group_inv_cast.columns
                    }
                    y_pred_group_inv_cast = y_pred_group_inv_cast.rename(rename_map)

                    # Reconstruct with time column
                    y_pred_group_inv = pl.concat(
                        [y_pred_group_inv.select(cs.by_name("time")), y_pred_group_inv_cast],
                        how="horizontal",
                    )

                    # Store in dict (without time)
                    y_pred_inv_dict[panel_group_name] = y_pred_group_inv.select(~cs.by_name("time"))

                # Reconstruct full dataframe
                times = y_pred_no_obs.select(cs.by_name("time"))
                y_pred_inv_cols = pl.concat(list(y_pred_inv_dict.values()), how="horizontal")
                y_pred_inv = pl.concat([times, y_pred_inv_cols], how="horizontal")

            # Add observed_time back
            y_pred = pl.concat([observed_time, y_pred_inv], how="horizontal")

        return y_pred

    def update(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
    ) -> "Decomposer":
        """Update all component forecasters with new observations.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations with "time" column.
        X : pl.DataFrame, optional
            New exogenous features with "time" column.
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data (ignored - Decomposer updates all groups).
            Parameter is ignored if the forecaster was not fitted on panel data.

        Returns
        -------
        self
            Updated decomposer.

        """
        check_is_fitted(self, ["forecasters_", "panel_group_names_"])
        y, X, panel_group_names = validate_forecaster_data(
            self,
            y=y,
            X=X,
            reset=False,
            panel_group_names=panel_group_names,
        )

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
                + [(pl.col(col) - pl.col(f"{col}_pred")).alias(col) for col in target_cols]
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

    def reset(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
    ) -> "Decomposer":
        """Reset all component forecasters to a new observation horizon.

        Parameters
        ----------
        y : pl.DataFrame
            Target observations with "time" column.
        X : pl.DataFrame, optional
            Exogenous features with "time" column.
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data (ignored - Decomposer resets all groups).
            Parameter is ignored if the forecaster was not fitted on panel data.

        Returns
        -------
        self
            Reset decomposer.

        """
        check_is_fitted(self, ["forecasters_", "panel_group_names_"])
        y, X, panel_group_names = validate_forecaster_data(
            self,
            y=y,
            X=X,
            reset=False,
            panel_group_names=panel_group_names,
        )

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
        router = MetadataRouter(owner=self)

        # Add routing for each forecaster
        for name, forecaster in self.forecasters:
            router.add(
                **{name: forecaster},
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="predict", callee="predict")
                .add(caller="update_predict", callee="update_predict"),
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
