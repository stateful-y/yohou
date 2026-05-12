"""DecompositionPipeline meta-forecaster for residual-based sequential decomposition."""

from copy import deepcopy
from typing import Literal

import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.base import clone
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing,
)
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseTransformer
from yohou.point import BasePointForecaster
from yohou.utils import POINT, Tags, add_interval, cast, dict_to_panel, get_group_df, validate_forecaster_data
from yohou.utils._compat import _BaseComposition, _fit_context, _raise_for_params

__all__ = ["DecompositionPipeline"]


class DecompositionPipeline(BasePointForecaster, _BaseComposition):
    """Meta-forecaster that decomposes time series into sequential components.

    DecompositionPipeline fits multiple forecasters sequentially, where each forecaster
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
        Use `target_transformer=LogTransformer()` for multiplicative decomposition
        (additive in log-space).

    feature_transformer : BaseTransformer or None, default=None
        Transformer applied to exogenous features before passing to component
        forecasters. Applied once at DecompositionPipeline level; all components receive
        the same transformed features.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See `BaseForecaster` for details.

    Attributes
    ----------
    forecasters_ : list of (str, BaseForecaster) tuples
        Fitted forecasters.

    residuals_ : dict of str to pl.DataFrame
        Residuals after each component (only if store_residuals=True).
        Keys are forecaster names, values are DataFrames with residuals.

    See Also
    --------
    `ColumnForecaster` : Separate forecasters for target/feature columns.
    `ForecastedFeatureForecaster` : Chains target and feature forecasters.
    `PolynomialTrendForecaster` : Polynomial trend component for decomposition.
    `FourierSeasonalityForecaster` : Fourier seasonality component for decomposition.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.compose import DecompositionPipeline
    >>> from yohou.point import SeasonalNaive
    >>> from yohou.stationarity import LogTransformer, PolynomialTrendForecaster
    >>>
    >>> # Create example time series
    >>> time = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 12, 31), interval="1d", eager=True
    ... )
    >>> y = pl.DataFrame({"time": time, "value": range(1, len(time) + 1)})
    >>>
    >>> # Additive decomposition: trend + seasonality
    >>> forecaster = DecompositionPipeline([
    ...     ("trend", PolynomialTrendForecaster(degree=1)),
    ...     ("seasonality", SeasonalNaive(seasonality=7)),
    ... ])
    >>> forecaster.fit(y, forecasting_horizon=7)  # doctest: +ELLIPSIS
    DecompositionPipeline(...)
    >>> y_pred = forecaster.predict(forecasting_horizon=7)
    >>>
    >>> # Multiplicative decomposition using LogTransformer
    >>> forecaster_mult = DecompositionPipeline(
    ...     [("trend", PolynomialTrendForecaster(degree=2)), ("seasonality", SeasonalNaive(seasonality=7))],
    ...     target_transformer=LogTransformer(),
    ... )
    >>> forecaster_mult.fit(y, forecasting_horizon=7)  # doctest: +ELLIPSIS
    DecompositionPipeline(...)
    >>>
    >>> # Inspect residuals
    >>> forecaster_inspect = DecompositionPipeline(
    ...     [("trend", PolynomialTrendForecaster(degree=1)), ("seasonality", SeasonalNaive(seasonality=7))],
    ...     store_residuals=True,
    ... )
    >>> forecaster_inspect.fit(y, forecasting_horizon=7)  # doctest: +ELLIPSIS
    DecompositionPipeline(...)
    >>> trend_residuals = forecaster_inspect.residuals_["trend"]

    Notes
    -----
    **Additive decomposition** (default)::

        y = f_1(t) + f_2(t) + ... + f_k(t)

    Each component ``f_i`` is fitted on the residuals from all
    previous components.  Predictions are the sum of all component
    forecasts.

    **Multiplicative decomposition** can be achieved by wrapping the
    pipeline with ``target_transformer=LogTransformer()``, which
    converts the problem to additive in log-space::

        log(y) = f_1(t) + f_2(t) + ... + f_k(t)

    Additional details:

    - Components are fitted sequentially (not in parallel) to maintain
      residual consistency.
    - All forecasters must be point forecasters (no interval forecasters).
    - Training residuals are computed by rewinding each inner forecaster
      clone to the start of the training data, then calling
      ``observe_predict`` with the real residuals and ``X_actual``.
      This produces rolling predictions conditioned on observed data
      (matching inference-time behavior) and avoids ``_recursive_predict``
      which would crash exogenous-aware inner forecasters.
    - ``observe()`` decomposes new observations across components:
      for each inner forecaster, it collects predictions via
      ``observe_predict``, then subtracts them to compute residuals
      for the next component.
    - ``observe_predict()`` is overridden to pass
      ``observe_fn=self.observe`` to ``_observe_predict_loop``,
      ensuring the rolling loop calls this pipeline's custom
      ``observe()`` (which performs sequential residual
      decomposition) rather than the base class's flat observe.

    Raises
    ------
    ValueError
        If any forecaster in ``forecasters`` is not an instance of
        `BasePointForecaster`, if forecaster names are not unique, or
        if ``forecasting_horizon`` < 1.

    """

    _parameter_constraints: dict = {
        "forecasters": [list],
        "store_residuals": ["boolean"],
    }

    def __init__(
        self,
        forecasters: list[tuple[str, BasePointForecaster]],
        store_residuals: bool = False,
        target_transformer: BaseTransformer | None = None,
        feature_transformer: BaseTransformer | None = None,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        BasePointForecaster.__init__(
            self,
            target_transformer=target_transformer,
            feature_transformer=feature_transformer,
            target_as_feature=None,
            panel_strategy=panel_strategy,
        )
        self.forecasters = forecasters
        self.store_residuals = store_residuals

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        return self._get_params("forecasters", deep=deep)

    def set_params(self, **params) -> "DecompositionPipeline":
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        Parameters
        ----------
        **params : dict
            Estimator parameters.  Nested parameters for forecasters use
            double-underscore notation, e.g.:
            ``forecasters__trend__seasonality=7``.

        Returns
        -------
        self
            Estimator instance.

        """
        self._set_params("forecasters", **params)
        return self

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None

        # Check stateful from forecasters
        stateful = False
        for _, f in self.forecasters:
            f_tags = f.__sklearn_tags__()
            if f_tags.forecaster_tags and f_tags.forecaster_tags.stateful:
                stateful = True
                break

        # Also check target transformer
        if not stateful and self.target_transformer is not None:
            target_tags = self.target_transformer.__sklearn_tags__().transformer_tags
            if target_tags is not None:
                stateful = target_tags.stateful

        tags.forecaster_tags.stateful = stateful

        # Determine forecaster_type from nested forecasters' tags
        if self.forecasters:
            _, last_forecaster = self.forecasters[-1]
            last_tags = last_forecaster.__sklearn_tags__()
            tags.forecaster_tags.forecaster_type = POINT
            if (
                last_tags.forecaster_tags
                and last_tags.forecaster_tags.forecaster_type
                and "point" in last_tags.forecaster_tags.forecaster_type
                and "interval" in last_tags.forecaster_tags.forecaster_type
            ):
                tags.forecaster_tags.forecaster_type = last_tags.forecaster_tags.forecaster_type

        # Aggregate other tags
        tags.forecaster_tags.uses_reduction = any(
            getattr(f.__sklearn_tags__().forecaster_tags, "uses_reduction", False) for _, f in self.forecasters
        )
        tags.forecaster_tags.supports_panel_data = all(
            getattr(f.__sklearn_tags__().forecaster_tags, "supports_panel_data", True) for _, f in self.forecasters
        )
        # DecompositionPipeline delegates observation tracking to child forecasters with
        # custom residual-based logic, so standard observe/rewind behavior doesn't apply
        tags.forecaster_tags.tracks_observations = False
        # The pipeline does not require X_actual to function (inner forecasters
        # that need features derive them internally, e.g. via LagTransformer).
        # When X_actual is provided, it is forwarded to all inner forecasters.
        tags.forecaster_tags.requires_exogenous = False

        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> "DecompositionPipeline":
        """Fit all component forecasters sequentially on residuals.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Processed by the feature transformer to produce
            lags, rolling statistics, and other derived features. If
            ``None``, only target-derived features are used.
        forecasting_horizon : int, default=1
            Number of time steps to forecast into the future.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column. Deterministic
            values available for past and future dates. Bypasses the
            feature transformer.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns. Bypasses the feature transformer.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted DecompositionPipeline instance.

        Raises
        ------
        ValueError
            If any forecaster is not a `BasePointForecaster`, if
            forecaster names are not unique, or if
            ``forecasting_horizon`` < 1.

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
        y_t, X_t = self._pre_fit(
            y=y, X_actual=X_actual, forecasting_horizon=forecasting_horizon, X_future=X_future, X_forecast=X_forecast
        )

        y_t = dict_to_panel(y_t)
        if X_t is not None:
            X_t = dict_to_panel(X_t)

        # Type narrowing: y_t should not be None after transformation
        assert y_t is not None

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
                X_actual=X_t,
                forecasting_horizon=forecasting_horizon,
                X_future=X_future,
                X_forecast=X_forecast,
                **step_params.fit,
            )
            self.forecasters_.append((name, forecaster_clone))

            # Compute training residuals via rolling observe_predict on a
            # clone rewound to the start.  This avoids _recursive_predict
            # (which calls observe with X_actual=None, crashing exogenous
            # inner forecasters) and produces predictions conditioned on
            # real observations rather than synthetic feedback.
            forecaster_clone_pred = deepcopy(forecaster_clone)
            forecaster_observation_horizon = forecaster_clone_pred.observation_horizon
            if forecaster_clone_pred.feature_transformer is not None:
                ft_ = forecaster_clone_pred.feature_transformer_
                if isinstance(ft_, dict):
                    feature_observation_horizon = max(ft.observation_horizon for ft in ft_.values()) + 1
                else:
                    feature_observation_horizon = ft_.observation_horizon + 1
                forecaster_observation_horizon = max(forecaster_observation_horizon, feature_observation_horizon)

            if not forecaster_observation_horizon:
                rewind_time = add_interval(residuals["time"][0], interval=forecaster_clone_pred.interval_, n=-1)
                y_rewind = pl.DataFrame(
                    {col: [rewind_time] if col == "time" else [None] for col in y_t.columns},
                    schema=y_t.schema,
                )
                X_rewind = None
                if X_t is not None:
                    X_rewind = pl.DataFrame(
                        {col: [rewind_time] if col == "time" else [None] for col in X_t.columns},
                        schema=X_t.schema,
                    )
            else:
                y_rewind = residuals[:forecaster_observation_horizon]
                X_rewind = None
                if X_t is not None:
                    X_rewind = X_t[:forecaster_observation_horizon]

            forecaster_clone_pred.rewind(y=y_rewind, X_actual=X_rewind, X_future=X_future, X_forecast=X_forecast)

            # Rolling observe_predict: observe real residuals in stride-
            # sized blocks, predict after each.  The inner join below
            # filters out predictions beyond the training range.
            residuals_remaining = residuals[forecaster_observation_horizon:]
            X_remaining = X_t[forecaster_observation_horizon:] if X_t is not None else None
            y_pred_train = forecaster_clone_pred.observe_predict(
                y=residuals_remaining,
                X_actual=X_remaining,
                forecasting_horizon=forecasting_horizon,
                X_future=X_future,
                X_forecast=X_forecast,
            )

            # Align predictions with current residuals on time
            aligned = residuals.join(
                y_pred_train.select(~cs.by_name("vintage_time")),
                on="time",
                how="inner",
                suffix="_pred",
            )

            # Calculate residuals (actual - predicted)
            target_cols = [c for c in residuals.columns if c != "time"]
            residuals = aligned.select(
                [pl.col("time")] + [(pl.col(col) - pl.col(f"{col}_pred")).alias(col) for col in target_cols]
            )

            # Store residuals if requested
            if self.store_residuals:
                self.residuals_[name] = residuals

        return self

    def predict(  # ty: ignore[invalid-method-override]
        self,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
        predict_transformed: bool = False,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate forecasts by summing predictions from all components.

        Parameters
        ----------
        forecasting_horizon : int >= 1 or None, default=None
            Horizon to forecast. If None, uses ``fit_forecasting_horizon_``.
        groups : list of str or None, default=None
            Group prefixes for panel data:
            - If None: predict for all groups
            - If list of str: predict only for the specified panel groups
            Parameter is ignored if the forecaster was not fitted on panel data.
        predict_transformed : bool, default=False
            If ``True``, the predictions are returned in the transformed space.
        X_future : pl.DataFrame or None, default=None
            Known future features override. Re-derives step columns
            without mutating forecaster state.
        X_forecast : pl.DataFrame or None, default=None
            External forecast override with ``"vintage_time"`` and
            ``"time"`` columns. Re-derives step columns without mutating
            forecaster state.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Predictions with columns: "vintage_time", "time", <target_columns>

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the pipeline has not been fitted yet.
        ValueError
            If no fitted forecasters are available.

        """
        check_is_fitted(self, ["forecasters_", "groups_"])
        _, _, groups = validate_forecaster_data(
            self,
            y=None,
            X_actual=None,
            reset=False,
            groups=groups,
        )

        # Use fit horizon if not specified
        if forecasting_horizon is None:
            forecasting_horizon = self.fit_forecasting_horizon_

        # Validate params before routing
        _raise_for_params(params, self, "predict")

        # Validate that we have at least one forecaster
        if not self.forecasters_:
            raise ValueError("DecompositionPipeline has no fitted forecasters. Call fit() first.")

        # Process metadata routing
        routed_params = process_routing(self, "predict", **params)

        # Get prediction from first forecaster to initialize
        first_name, first_forecaster = self.forecasters_[0]
        first_params = routed_params[first_name]

        y_pred_first = first_forecaster.predict(
            forecasting_horizon=forecasting_horizon,
            predict_transformed=True,
            X_future=X_future,
            X_forecast=X_forecast,
            **first_params.predict,
        )

        # Initialize with first prediction
        time_cols = y_pred_first.select("vintage_time", "time")
        y_pred_sum = y_pred_first.select(~cs.by_name("vintage_time", "time"))

        # Process remaining forecasters and accumulate predictions
        for name, forecaster in self.forecasters_[1:]:
            # Get routed params for this forecaster (direct Bunch access)
            step_params = routed_params[name]

            y_pred = forecaster.predict(
                forecasting_horizon=forecasting_horizon,
                predict_transformed=True,
                X_future=X_future,
                X_forecast=X_forecast,
                **step_params.predict,
            )

            # Extract values (without time columns) and sum
            y_pred_values = y_pred.select(~cs.by_name("vintage_time", "time"))
            y_pred_sum = y_pred_sum + y_pred_values

        # Combine time columns with summed values
        y_pred = pl.concat([time_cols, y_pred_sum], how="horizontal")

        if not predict_transformed and self.target_transformer is not None:
            # Apply inverse target transform

            # Remove vintage_time before inverse transform
            vintage_time = y_pred.select("vintage_time")
            y_pred_no_obs = y_pred.select(~cs.by_name("vintage_time"))

            # Handle panel data (target_transformer_ and _y_observed are dicts)
            if self.groups_ is None:
                # Non-panel data
                assert isinstance(self.target_transformer_, BaseTransformer)
                assert not isinstance(self._y_observed, dict)
                y_pred_inv = self.target_transformer_.inverse_transform(X_t=y_pred_no_obs, X_p=self._y_observed)

            else:
                # Panel data
                assert isinstance(self.target_transformer_, dict)
                assert isinstance(self._y_observed, dict)
                y_pred_inv_dict = {}
                for panel_group_name in groups or self.groups_:
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
                            col: f"{panel_group_name}__{col}" for col in y_pred_group.columns if col != "time"
                        }
                        y_pred_group = y_pred_group.rename(rename_map)
                        y_pred_inv_dict[panel_group_name] = y_pred_group.select(~cs.by_name("time"))
                        continue

                    y_observed_local = self._y_observed[panel_group_name]

                    # Extract the group's columns (in transformed space, with prefix)
                    prefix = f"{panel_group_name}__"
                    group_cols = [c for c in y_pred_no_obs.columns if c.startswith(prefix)]
                    y_pred_group = y_pred_no_obs.select(cs.by_name("time") | cs.by_name(group_cols))

                    # Strip group prefix so transformer sees local column names
                    rename_strip = {c: c[len(prefix) :] for c in group_cols}
                    y_pred_group = y_pred_group.rename(rename_strip)

                    # Inverse transform (works with unprefixed/local columns)
                    y_pred_group_inv = transformer.inverse_transform(X_t=y_pred_group, X_p=y_observed_local)

                    # Cast to restore original dtypes
                    y_pred_group_inv_cast = cast(y_pred_group_inv.select(~cs.by_name("time")), self.local_y_schema_)

                    # Rename to add prefix
                    rename_map = {col: f"{panel_group_name}__{col}" for col in y_pred_group_inv_cast.columns}
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

            # Add vintage_time back
            y_pred = pl.concat([vintage_time, y_pred_inv], how="horizontal")

        return y_pred

    def observe_predict(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
        stride: StrictInt | None = None,
        predict_transformed: bool = False,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Alternate recursive predict and observe with residual decomposition.

        Overrides the base ``observe_predict`` to ensure the rolling loop
        calls this pipeline's custom ``observe()`` at each stride step.
        Without this override, the base implementation bypasses
        ``DecompositionPipeline.observe()`` and treats the pipeline as a
        flat forecaster, leaving inner forecasters' states stale.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Sliced and observed incrementally at each step of
            the rolling loop.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future. If ``None``,
            uses the horizon specified at fit time.
        groups : list of str or None, default=None
            Panel group prefixes to operate on. If ``None``, all groups
            are used.
        stride : int or None, default=None
            Step size for rolling update then predict. If ``None``,
            defaults to ``forecasting_horizon``.
        predict_transformed : bool, default=False
            If ``True``, return predictions in the transformed space without
            applying inverse target transformation.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Point predictions with ``"vintage_time"``, ``"time"``, and one
            column per target variable.

        """
        check_is_fitted(self, ["forecasters_", "groups_"])

        y, X_actual, groups = validate_forecaster_data(
            self,
            y=y,
            X_actual=X_actual,
            reset=False,
            groups=groups,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        fh = self._validate_predict_params(forecasting_horizon)
        if stride is None:
            stride = self.fit_forecasting_horizon_

        return self._observe_predict_loop(
            predict_fn=self.predict,
            y=y,
            X_actual=X_actual,
            X_future=X_future,
            X_forecast=X_forecast,
            groups=groups,
            stride=stride,
            observe_fn=self.observe,
            forecasting_horizon=fh,
            predict_transformed=predict_transformed,
            **params,
        )

    def observe(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> "DecompositionPipeline":
        """Observe new data for all component forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations with a ``"time"`` column.
        X_actual : pl.DataFrame or None, default=None
            New actual feature observations with a ``"time"`` column
            aligned with ``y``. Forwarded to each component forecaster.
        groups : list of str or None, default=None
            Group prefixes for panel data.  Ignored for
            DecompositionPipeline (all groups are always observed).
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.

        Returns
        -------
        self
            DecompositionPipeline with updated observation state.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the pipeline has not been fitted yet.

        """
        check_is_fitted(self, ["forecasters_", "groups_"])
        y, X_actual, groups = validate_forecaster_data(
            self,
            y=y,
            X_actual=X_actual,
            reset=False,
            groups=groups,
        )

        # Observe transformers first
        if self.target_transformer_ is not None:
            assert isinstance(self.target_transformer_, BaseTransformer)
            self.target_transformer_.observe(y)
            y_t = self.target_transformer_.transform(y)
        else:
            y_t = y

        if X_actual is not None and self.feature_transformer_ is not None:
            assert isinstance(self.feature_transformer_, BaseTransformer)
            self.feature_transformer_.observe(X_actual)
            X_t = self.feature_transformer_.transform(X_actual)
        else:
            X_t = X_actual

        # Observe all forecasters
        residuals = y_t
        for name, forecaster in self.forecasters_:
            # Rolling observe_predict: observes the inner forecaster while
            # collecting predictions for residual computation.  This avoids
            # _recursive_predict (which calls observe with X_actual=None)
            # when len(residuals) > fit_forecasting_horizon.
            y_pred = forecaster.observe_predict(
                y=residuals,
                X_actual=X_t,
                forecasting_horizon=forecaster.fit_forecasting_horizon_,
                X_future=X_future,
                X_forecast=X_forecast,
            )
            # Align predictions with current residuals on time
            aligned = residuals.join(
                y_pred.select(~cs.by_name("vintage_time")),
                on="time",
                how="inner",
                suffix="_pred",
            )

            # Calculate residuals (actual - predicted)
            target_cols = [c for c in residuals.columns if c != "time"]
            residuals = aligned.select(
                [pl.col("time")] + [(pl.col(col) - pl.col(f"{col}_pred")).alias(col) for col in target_cols]
            )

            # Store residuals if requested
            if self.store_residuals:
                self.residuals_[name] = pl.concat(
                    [self.residuals_[name], residuals],
                )

        # Observe base class observation buffers
        self._y_observed = y_t
        if X_t is not None:
            self._X_observed = X_t

        return self

    def rewind(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> "DecompositionPipeline":
        """Rewind all component forecasters to a new observation horizon.

        Parameters
        ----------
        y : pl.DataFrame
            Target observations with a ``"time"`` column.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations to restore the observation
            state to. Must align with ``y``.
        groups : list of str or None, default=None
            Group prefixes for panel data.  Ignored for
            DecompositionPipeline (all groups are always rewound).
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.

        Returns
        -------
        self
            DecompositionPipeline with rewound observation state.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the pipeline has not been fitted yet.

        """
        check_is_fitted(self, ["forecasters_", "groups_"])
        y, X_actual, groups = validate_forecaster_data(
            self,
            y=y,
            X_actual=X_actual,
            reset=False,
            groups=groups,
        )

        # Rewind transformers first
        if self.target_transformer_ is not None:
            assert isinstance(self.target_transformer_, BaseTransformer)
            y_t = self.target_transformer_.rewind_transform(y)
        else:
            y_t = y

        if X_actual is not None and self.feature_transformer_ is not None:
            assert isinstance(self.feature_transformer_, BaseTransformer)
            X_t = self.feature_transformer_.rewind_transform(X_actual)
        else:
            X_t = X_actual

        # Rewind all forecasters
        for _, forecaster in self.forecasters_:
            forecaster.rewind(y_t, X_actual=X_t, X_future=X_future, X_forecast=X_forecast)

        # Rewind base class observation buffers
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
                .add(caller="observe_predict", callee="observe_predict"),
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
