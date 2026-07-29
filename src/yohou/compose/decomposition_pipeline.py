"""DecompositionPipeline meta-forecaster for residual-based sequential decomposition."""

from copy import deepcopy
from typing import Literal

import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.base import clone
from sklearn.utils import Bunch
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing,
)
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseActualTransformer, BaseForecastTransformer
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
    forecasters : list of (str, BasePointForecaster) tuples
        List of (name, forecaster) tuples specifying the forecaster objects
        to be applied sequentially. All forecasters must be point forecasters.

        Typical ordering: trend → seasonality → residual

        name : str
            Unique name for the forecaster component.
        forecaster : BasePointForecaster
            Point forecaster object.

    store_residuals : bool, default=False
        If True, stores residuals after each component in `self.residuals_`
        dict for inspection. Keys are forecaster names, values are pl.DataFrame
        with residuals.

    target_transformer : BaseActualTransformer or None, default=None
        Transformer applied to target time series before decomposition.
        Use `target_transformer=LogTransformer()` for multiplicative decomposition
        (additive in log-space).

    actual_transformer : BaseActualTransformer or None, default=None
        Transformer applied to exogenous features before passing to component
        forecasters. Applied once at DecompositionPipeline level; all components receive
        the same transformed features.
    forecast_transformer : BaseForecastTransformer or None, default=None
        Transformer applied to ``X_forecast`` before step columns are derived,
        so the step columns reaching the estimator are built from transformed
        values. Must be forecast-kind (vintage-indexed); an actual-kind
        transformer is rejected. ``None`` leaves ``X_forecast`` untouched.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See `BaseForecaster` for details.

    Attributes
    ----------
    forecasters_ : list of (str, BasePointForecaster) tuples
        Fitted forecasters.

    named_forecasters_ : Bunch
        The same fitted forecasters keyed by component name.

    residuals_ : dict of str to pl.DataFrame
        Residuals after each component (only if store_residuals=True).
        Keys are forecaster names, values are DataFrames with residuals.

    See Also
    --------
    - [`ColumnForecaster`][yohou.compose.column_forecaster.ColumnForecaster] : Separate forecasters for target/feature columns.
    - [`ForecastedFeatureForecaster`][yohou.compose.forecasted_feature_forecaster.ForecastedFeatureForecaster] : Chains target and feature forecasters.
    - [`PolynomialTrendForecaster`][yohou.stationarity.trend.PolynomialTrendForecaster] : Polynomial trend component for decomposition.
    - [`FourierSeasonalityForecaster`][yohou.stationarity.seasonality.FourierSeasonalityForecaster] : Fourier seasonality component for decomposition.

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
        *,
        store_residuals: bool = False,
        target_transformer: BaseActualTransformer | None = None,
        actual_transformer: BaseActualTransformer | None = None,
        forecast_transformer: BaseForecastTransformer | None = None,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        BasePointForecaster.__init__(
            self,
            target_transformer=target_transformer,
            actual_transformer=actual_transformer,
            forecast_transformer=forecast_transformer,
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

    @property
    def named_forecasters_(self) -> Bunch:
        """Access fitted component forecasters by name.

        Returns
        -------
        Bunch
            Dictionary-like object with component names as keys and fitted forecaster
            objects as values.

        """
        check_is_fitted(self, ["forecasters_"])
        return Bunch(**dict(self.forecasters_))

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
            with ``y``. Processed by the actual transformer to produce
            lags, rolling statistics, and other derived features. If
            ``None``, only target-derived features are used.
        forecasting_horizon : int, default=1
            Number of time steps to forecast into the future.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column. Deterministic
            values available for past and future dates. Bypasses the
            actual transformer.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns. Bypasses the actual transformer.
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

        # Each component re-derives its own step columns from X_future/X_forecast.
        # _pre_fit already merged the pipeline-level step columns into X_t, so strip
        # them before forwarding to avoid a name collision when a component derives
        # the same step columns again from the (also-forwarded) X_forecast/X_future.
        # _step_column_names_ tracks step columns by their derived name, which is
        # unprefixed for global (non-group) features. Under panel data those columns
        # are distributed per group and become ``group__<feat>_step_h`` in the
        # panel-wide X_t, so match on the group-stripped suffix too; otherwise the
        # global step columns survive and collide with each component's re-derived ones.
        X_t_components = X_t
        if X_t is not None and self._step_column_names_:
            drop_cols = [
                c
                for c in X_t.columns
                if c in self._step_column_names_ or ("__" in c and c.split("__", 1)[1] in self._step_column_names_)
            ]
            if drop_cols:
                X_t_components = X_t.drop(drop_cols)

        # Forward the TRANSFORMED X_forecast to components, so each derives its
        # step columns from transformed values. _pre_fit fit the forecast_transformer
        # slot and cached the result as _X_forecast_t_ (byte-for-byte the raw frame
        # when the slot is unset). Forwarding the raw X_forecast here, while the block
        # above strips the transformed pipeline-level step columns, would make the
        # slot inert: components would re-derive from untransformed values.
        X_forecast_components = self._X_forecast_t_

        residuals = y_t
        for name, forecaster in self.forecasters:
            # Clone and fit forecaster on current residuals
            forecaster_clone = clone(forecaster)

            # Get routed params for this forecaster (direct Bunch access)
            step_params = routed_params[name]

            forecaster_clone.fit(
                y=residuals,
                X_actual=X_t_components,
                forecasting_horizon=forecasting_horizon,
                X_future=X_future,
                X_forecast=X_forecast_components,
                **step_params.fit,
            )
            self.forecasters_.append((name, forecaster_clone))

            # Compute training residuals via rolling observe_predict on a
            # clone rewound to the start.  This avoids _recursive_predict
            # (which calls observe with X_actual=None, crashing exogenous
            # inner forecasters) and produces predictions conditioned on
            # real observations rather than synthetic feedback.
            forecaster_clone_pred = deepcopy(forecaster_clone)
            forecaster_observation_horizon = self._effective_observation_horizon(forecaster_clone_pred)

            if not forecaster_observation_horizon:
                rewind_time = add_interval(residuals["time"][0], interval=forecaster_clone_pred.interval_, n=-1)
                y_rewind = pl.DataFrame(
                    {col: [rewind_time] if col == "time" else [None] for col in y_t.columns},
                    schema=y_t.schema,
                )
                X_rewind = None
                if X_t_components is not None:
                    X_rewind = pl.DataFrame(
                        {col: [rewind_time] if col == "time" else [None] for col in X_t_components.columns},
                        schema=X_t_components.schema,
                    )
            else:
                y_rewind = residuals[:forecaster_observation_horizon]
                X_rewind = None
                if X_t_components is not None:
                    X_rewind = X_t_components[:forecaster_observation_horizon]

            forecaster_clone_pred.rewind(
                y=y_rewind, X_actual=X_rewind, X_future=X_future, X_forecast=X_forecast_components
            )

            # Rolling observe_predict: observe real residuals in stride-
            # sized blocks, predict after each.  The inner join below
            # filters out predictions beyond the training range.
            residuals_remaining = residuals[forecaster_observation_horizon:]
            X_remaining = X_t_components[forecaster_observation_horizon:] if X_t_components is not None else None
            y_pred_train = forecaster_clone_pred.observe_predict(
                y=residuals_remaining,
                X_actual=X_remaining,
                forecasting_horizon=forecasting_horizon,
                X_future=X_future,
                X_forecast=X_forecast_components,
            )

            # Align predictions with current residuals on time. The warmup rows
            # (first forecaster_observation_horizon residuals) have no prediction
            # and are dropped intentionally; every remaining residual must match a
            # prediction, so guard against any extra silent loss.
            aligned = residuals.join(
                y_pred_train.select(~cs.by_name("vintage_time")),
                on="time",
                how="inner",
                suffix="_pred",
            )
            self._check_residual_alignment(name, aligned.height, residuals_remaining.height)

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

        # Validate the horizon (falls back to fit_forecasting_horizon_ when None,
        # and rejects horizons < 1 instead of forwarding them to inner forecasters)
        forecasting_horizon = self._validate_predict_params(forecasting_horizon)

        # Validate params before routing
        _raise_for_params(params, self, "predict")

        # Validate that we have at least one forecaster
        if not self.forecasters_:
            raise ValueError("DecompositionPipeline has no fitted forecasters. Call fit() first.")

        # Process metadata routing
        routed_params = process_routing(self, "predict", **params)

        # Forward the TRANSFORMED X_forecast to components. Resolve the
        # supplied-vs-cache branch before transforming: a supplied frame is raw
        # and is transformed here; an omitted one falls back to the fit-time cache,
        # which is already transformed and must not be transformed again.
        X_forecast_components = (
            self._transform_X_forecast(X_forecast) if X_forecast is not None else self._X_forecast_t_
        )

        # Get prediction from first forecaster to initialize
        first_name, first_forecaster = self.forecasters_[0]
        first_params = routed_params[first_name]

        # Each component predicts in its own original (post-inverse) scale, i.e.
        # the residual-stream scale it was fitted on. fit() computes residuals
        # from component observe_predict() output with predict_transformed=False
        # (the default), so component-level target_transformers are inverted
        # there; predict() must invert them too (predict_transformed=False),
        # otherwise a component target_transformer leaves its forecast in scaled
        # space and the recomposed sum is off by orders of magnitude. The
        # pipeline-level target_transformer is inverted once below, after summing.
        y_pred_first = first_forecaster.predict(
            forecasting_horizon=forecasting_horizon,
            predict_transformed=False,
            X_future=X_future,
            X_forecast=X_forecast_components,
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
                predict_transformed=False,
                X_future=X_future,
                X_forecast=X_forecast_components,
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
                assert isinstance(self.target_transformer_, BaseActualTransformer)
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

    def _transform_panel(
        self,
        transformer_dict: dict,
        df: pl.DataFrame,
        schema: dict,
        method: str,
    ) -> dict[str, pl.DataFrame]:
        """Apply a per-group stateful transform dict to a panel DataFrame.

        Slices each group's columns, calls ``method`` (e.g.
        ``"observe_transform"`` or ``"rewind_transform"``) on the group's
        transformer with the group prefix stripped, and returns the per-group
        transformed frames as a dict keyed by group. Mirrors the per-group
        inverse-transform branch in ``predict`` so that panel-mode
        ``observe``/``rewind`` do not assume a scalar transformer. Callers reuse
        the dict both as the per-group observation buffer (``_y_observed``) and,
        via ``dict_to_panel``, as the panel frame fed to the component
        forecasters.
        """
        assert self.groups_ is not None
        transformed = {}
        for group_name in self.groups_:
            transformer = transformer_dict[group_name]
            group_local = get_group_df(df=df, group_name=group_name, schema=schema)
            transformed[group_name] = group_local if transformer is None else getattr(transformer, method)(group_local)
        return transformed

    def _panel_X_actual_schema(self) -> dict:
        """Build the per-group X_actual schema (local plus shared columns)."""
        assert self.local_X_actual_schema_ is not None
        X_schema = dict(self.local_X_actual_schema_)
        if self.shared_X_actual_schema_:
            X_schema.update(self.shared_X_actual_schema_)
        return X_schema

    @staticmethod
    def _effective_observation_horizon(forecaster) -> int:
        """Observation horizon for the residual rolling observe-predict warmup.

        Returns the component forecaster's own ``observation_horizon``, widened
        to cover a actual transformer's horizon when present. The ``+1`` over
        the actual transformer's horizon reserves one extra row so the first
        prediction has a fully observed feature window, unlike
        `BaseForecaster.observation_horizon`, which is sized for buffer retention
        rather than for warming up this internal rolling loop.

        Parameters
        ----------
        forecaster : BaseForecaster
            Fitted component forecaster (a deepcopy used for residual prediction).

        Returns
        -------
        int
            Number of leading rows to reserve before the rolling observe-predict.

        """
        observation_horizon = forecaster.observation_horizon
        if forecaster.actual_transformer is not None:
            ft_ = forecaster.actual_transformer_
            if isinstance(ft_, dict):
                feature_observation_horizon = max(ft.observation_horizon for ft in ft_.values()) + 1
            else:
                feature_observation_horizon = ft_.observation_horizon + 1
            observation_horizon = max(observation_horizon, feature_observation_horizon)
        return observation_horizon

    def _bounded_observed(
        self,
        y_t: pl.DataFrame,
        y_t_dict: dict[str, pl.DataFrame] | None,
    ) -> pl.DataFrame | dict[str, pl.DataFrame]:
        """Store ``_y_observed`` trimmed to ``observation_horizon`` rows.

        predict() reads ``_y_observed`` only as the prior context (``X_p``) for
        the target transformer's inverse transform, which needs at most
        ``observation_horizon`` most-recent (trailing) rows. Trimming keeps the
        buffer bounded on long-running streams instead of growing with every
        observe/rewind.

        Parameters
        ----------
        y_t : pl.DataFrame
            The transformed target for this call (panel frame in panel mode).
        y_t_dict : dict of str to pl.DataFrame or None
            Per-group transformed frames in panel mode, else ``None``.

        Returns
        -------
        pl.DataFrame or dict of str to pl.DataFrame
            The bounded observation buffer, as a dict in panel mode.

        """
        horizon = self.observation_horizon
        if self.groups_ is not None and y_t_dict is not None:
            if horizon:
                return {group: df.tail(horizon) for group, df in y_t_dict.items()}
            return y_t_dict
        return y_t.tail(horizon) if horizon else y_t

    @staticmethod
    def _check_residual_alignment(name: str, aligned_height: int, expected_height: int) -> None:
        """Raise if the residual/prediction inner join dropped rows unexpectedly.

        Parameters
        ----------
        name : str
            Name of the component forecaster being aligned.
        aligned_height : int
            Number of rows surviving the residual/prediction inner join.
        expected_height : int
            Number of residual rows that should have a matching prediction.

        Raises
        ------
        ValueError
            If the number of rows surviving the join differs from expected.
            Fewer rows signal that residual timestamps were silently dropped;
            more rows signal join fan-out (e.g. duplicate prediction
            timestamps). Both typically indicate a stride or horizon
            misalignment between residuals and predictions.

        """
        if aligned_height != expected_height:
            raise ValueError(
                f"Residual/prediction alignment for component '{name}' lost "
                f"{expected_height - aligned_height} row(s): {aligned_height} aligned "
                f"but {expected_height} expected. This usually indicates a stride or "
                "horizon misalignment between residuals and the component's predictions."
            )

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
            defaults to ``fit_forecasting_horizon_`` (the horizon used at
            fit time).
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

        Notes
        -----
        If ``store_residuals=True``, the residuals computed for each component
        during this call are appended to ``self.residuals_[name]``. When
        ``store_residuals=False`` no residuals are accumulated.

        """
        check_is_fitted(self, ["forecasters_", "groups_"])
        y, X_actual, groups = validate_forecaster_data(
            self,
            y=y,
            X_actual=X_actual,
            reset=False,
            groups=groups,
        )

        # Forward the TRANSFORMED X_forecast to components (see predict): a supplied
        # frame is transformed here, an omitted one falls back to the fit-time cache.
        X_forecast_components = (
            self._transform_X_forecast(X_forecast) if X_forecast is not None else self._X_forecast_t_
        )

        # Observe and transform in one atomic step: observe_transform uses the
        # pre-observe state to transform, then updates the buffer.  A separate
        # observe() then transform() would transform against post-observe state
        # and yield empty output for stateful transformers (e.g. differencing).
        y_t_dict: dict[str, pl.DataFrame] | None = None
        X_t_dict: dict[str, pl.DataFrame] | None = None
        if self.target_transformer_ is not None:
            if self.groups_ is None:
                assert isinstance(self.target_transformer_, BaseActualTransformer)
                y_t = self.target_transformer_.observe_transform(y)
            else:
                assert isinstance(self.target_transformer_, dict)
                y_t_dict = self._transform_panel(self.target_transformer_, y, self.local_y_schema_, "observe_transform")
                y_t = dict_to_panel(y_t_dict)
        else:
            y_t = y

        if X_actual is not None and self.actual_transformer_ is not None:
            if self.groups_ is None:
                assert isinstance(self.actual_transformer_, BaseActualTransformer)
                X_t = self.actual_transformer_.observe_transform(X_actual)
            else:
                assert isinstance(self.actual_transformer_, dict)
                X_t_dict = self._transform_panel(
                    self.actual_transformer_, X_actual, self._panel_X_actual_schema(), "observe_transform"
                )
                X_t = dict_to_panel(X_t_dict)
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
                X_forecast=X_forecast_components,
            )
            # Align predictions with current residuals on time. observe_predict
            # ran on the full residual stream, so every residual must match a
            # prediction; guard against silent row loss.
            aligned = residuals.join(
                y_pred.select(~cs.by_name("vintage_time")),
                on="time",
                how="inner",
                suffix="_pred",
            )
            self._check_residual_alignment(name, aligned.height, residuals.height)

            # Calculate residuals (actual - predicted)
            target_cols = [c for c in residuals.columns if c != "time"]
            residuals = aligned.select(
                [pl.col("time")] + [(pl.col(col) - pl.col(f"{col}_pred")).alias(col) for col in target_cols]
            )

            # Store residuals if requested. Initialize defensively: observe may
            # run before any residuals were stored for this component (e.g. a
            # forecaster added after fit, or store_residuals toggled on).
            if self.store_residuals:
                if not hasattr(self, "residuals_"):
                    self.residuals_ = {}
                prior = self.residuals_.get(name)
                self.residuals_[name] = pl.concat([prior, residuals]) if prior is not None else residuals

        # Store the observation buffer predict() reads as inverse-transform
        # context, bounded to observation_horizon. In panel mode predict() reads
        # it as a per-group dict, so store the dict form there. _X_observed is
        # not read by this forecaster, so it is intentionally not stored.
        self._y_observed = self._bounded_observed(y_t, y_t_dict)

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

        Notes
        -----
        Like ``observe``, ``rewind`` threads the decomposition residuals
        through the component forecasters: each component is rewound on the
        residual stream at its stage (the transformed target minus the sum of
        all preceding components' predictions), mirroring what each component
        was fitted on. The difference from ``observe`` is only that ``rewind``
        resets each component's observation buffer to a reference window rather
        than appending to it.

        Unlike ``observe``, ``rewind`` does not modify ``residuals_``; any
        entries accumulated by prior ``observe`` calls are preserved. To clear
        them, reset ``self.residuals_`` manually or call ``fit`` again.

        """
        check_is_fitted(self, ["forecasters_", "groups_"])
        y, X_actual, groups = validate_forecaster_data(
            self,
            y=y,
            X_actual=X_actual,
            reset=False,
            groups=groups,
        )

        # Forward the TRANSFORMED X_forecast to components (see predict): a supplied
        # frame is transformed here, an omitted one falls back to the fit-time cache.
        X_forecast_components = (
            self._transform_X_forecast(X_forecast) if X_forecast is not None else self._X_forecast_t_
        )

        # Rewind transformers first
        y_t_dict: dict[str, pl.DataFrame] | None = None
        X_t_dict: dict[str, pl.DataFrame] | None = None
        if self.target_transformer_ is not None:
            if self.groups_ is None:
                assert isinstance(self.target_transformer_, BaseActualTransformer)
                y_t = self.target_transformer_.rewind_transform(y)
            else:
                assert isinstance(self.target_transformer_, dict)
                y_t_dict = self._transform_panel(self.target_transformer_, y, self.local_y_schema_, "rewind_transform")
                y_t = dict_to_panel(y_t_dict)
        else:
            y_t = y

        if X_actual is not None and self.actual_transformer_ is not None:
            if self.groups_ is None:
                assert isinstance(self.actual_transformer_, BaseActualTransformer)
                X_t = self.actual_transformer_.rewind_transform(X_actual)
            else:
                assert isinstance(self.actual_transformer_, dict)
                X_t_dict = self._transform_panel(
                    self.actual_transformer_, X_actual, self._panel_X_actual_schema(), "rewind_transform"
                )
                X_t = dict_to_panel(X_t_dict)
        else:
            X_t = X_actual

        # Rewind all forecasters, threading residuals through each stage so
        # every component is rewound on the signal it was fitted on (the
        # transformed target minus preceding components' predictions), not the
        # full target. Predictions for the residuals are computed on a deepcopy
        # (mirroring fit) so the real forecaster's buffer is left in the rewound
        # state, not the observe-predict state.
        residuals = y_t
        for name, forecaster in self.forecasters_:
            forecaster.rewind(residuals, X_actual=X_t, X_future=X_future, X_forecast=X_forecast_components)

            forecaster_pred = deepcopy(forecaster)
            forecaster_observation_horizon = self._effective_observation_horizon(forecaster_pred)

            if not forecaster_observation_horizon:
                rewind_time = add_interval(residuals["time"][0], interval=forecaster_pred.interval_, n=-1)
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
                X_rewind = X_t[:forecaster_observation_horizon] if X_t is not None else None

            forecaster_pred.rewind(y=y_rewind, X_actual=X_rewind, X_future=X_future, X_forecast=X_forecast_components)

            residuals_remaining = residuals[forecaster_observation_horizon:]
            X_remaining = X_t[forecaster_observation_horizon:] if X_t is not None else None
            y_pred = forecaster_pred.observe_predict(
                y=residuals_remaining,
                X_actual=X_remaining,
                forecasting_horizon=forecaster.fit_forecasting_horizon_,
                X_future=X_future,
                X_forecast=X_forecast_components,
            )

            aligned = residuals.join(
                y_pred.select(~cs.by_name("vintage_time")),
                on="time",
                how="inner",
                suffix="_pred",
            )
            self._check_residual_alignment(name, aligned.height, residuals_remaining.height)
            target_cols = [c for c in residuals.columns if c != "time"]
            residuals = aligned.select(
                [pl.col("time")] + [(pl.col(col) - pl.col(f"{col}_pred")).alias(col) for col in target_cols]
            )

        # Store the observation buffer predict() reads as inverse-transform
        # context, bounded to observation_horizon. In panel mode predict() reads
        # it as a per-group dict, so store the dict form there. _X_observed is
        # not read by this forecaster, so it is intentionally not stored.
        self._y_observed = self._bounded_observed(y_t, y_t_dict)

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

        if self.actual_transformer is not None:
            router.add(
                actual_transformer=self.actual_transformer,
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="fit", callee="transform")
                .add(caller="predict", callee="transform"),
            )

        # The same three mappings the two actual slots already register here, which
        # this class needs because it applies its transformers at predict too. No
        # observe or rewind mapping: BaseForecastTransformer has neither method.
        if self.forecast_transformer is not None:
            router.add(
                forecast_transformer=self.forecast_transformer,
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="fit", callee="transform")
                .add(caller="predict", callee="transform"),
            )

        return router
