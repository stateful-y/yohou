"""Base class for forecasters."""

import abc
import warnings
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal
from typing import cast as typing_cast

import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.base import BaseEstimator
from sklearn.utils.metadata_routing import MetadataRouter, MethodMapping
from sklearn.utils.validation import check_is_fitted

from yohou.base.panel import BasePanelForecaster
from yohou.base.standard import BaseStandardForecaster
from yohou.base.transformer import BaseTransformer
from yohou.base.utils import _derive_step_columns
from yohou.utils import (
    Tags,
    cast,
    inspect_panel,
    validate_forecaster_data,
)
from yohou.utils._compat import StrOptions, _fit_context

PredictionType = Literal["point", "interval", "class_proba"]

__all__ = [
    "BaseForecaster",
    "PredictionType",
]


class BaseForecaster(BaseStandardForecaster, BasePanelForecaster, BaseEstimator, metaclass=abc.ABCMeta):
    """Base class for forecasters.

    Provides the full forecaster lifecycle: ``fit``, ``predict``,
    ``observe``, ``rewind``, ``observe_predict``, and their interval
    variants.  Supports panel data via ``__``-prefixed column names.

    Parameters
    ----------
    target_transformer : instance of `BaseTransformer` or None, default=None
        Transformer used to transform the target time series into the new target.
    feature_transformer : instance of `BaseTransformer` or None, default=None
        Transformer used to transform the feature time series into features.
    target_as_feature : {"transformed", "raw"} or None, default="transformed"
        Controls whether the target is included as a feature.
        ``"transformed"`` includes the transformed target, ``"raw"``
        includes the raw target, and ``None`` uses only exogenous features.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data (columns with ``__`` separators):

        - ``"global"`` (default): Detect panel groups, fit per-group
          transformers, pool data for the estimator.  Each group gets
          independent state (observation buffers, transformers) but
          shares a single model.
        - ``"multivariate"``: Skip panel detection entirely.  Treat
          ``__``-prefixed columns as ordinary multivariate columns.
          One transformer and one model see the full wide DataFrame,
          enabling cross-group feature interactions.

        For per-group *independent* models, use
        [LocalPanelForecaster][yohou.compose.LocalPanelForecaster] instead.

    Attributes
    ----------
    interval_ : str
        Detected time interval of the training data.

    Notes
    -----
    ``observe()`` appends new observations to internal buffers **without
    refitting** the model.  ``rewind()`` truncates buffers to the last
    ``observation_horizon`` rows.  Together they enable streaming /
    rolling-window evaluation.

    The ``forecasting_horizon`` is set at ``fit`` time but can be
    overridden at ``predict`` time.

    See Also
    --------
    - [`BasePointForecaster`][yohou.point.base.BasePointForecaster] : Base class for point forecasters.
    - [`BaseIntervalForecaster`][yohou.interval.base.BaseIntervalForecaster] : Base class for interval forecasters.
    - [`BaseReductionForecaster`][yohou.base.reduction.BaseReductionForecaster] : Forecasting via sklearn regressors.

    """

    _parameter_constraints: dict = {
        "target_transformer": [BaseTransformer, None],
        "feature_transformer": [BaseTransformer, None],
        "target_as_feature": [StrOptions({"transformed", "raw"}), None],
        "panel_strategy": [StrOptions({"global", "multivariate"})],
    }

    # Fitted attributes (set during fit())
    interval_: str

    def __init__(
        self,
        feature_transformer: BaseTransformer | None = None,
        target_transformer: BaseTransformer | None = None,
        target_as_feature: Literal["transformed", "raw"] | None = "transformed",
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        self.feature_transformer = feature_transformer
        self.target_transformer = target_transformer
        self.target_as_feature = target_as_feature
        self.panel_strategy = panel_strategy

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Merge parameter constraints from all classes in the MRO."""
        super().__init_subclass__(**kwargs)
        # Auto-merge _parameter_constraints from all classes in the MRO.
        # Walk in reverse so the most-derived class wins on key conflicts.
        merged: dict = {}
        for klass in reversed(cls.__mro__):
            own = klass.__dict__.get("_parameter_constraints")
            if own and isinstance(own, dict):
                merged.update(own)
        cls._parameter_constraints = merged

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        # Create Tags with forecaster-specific defaults
        tags = Tags(estimator_type="forecaster", requires_fit=True)
        assert tags.forecaster_tags is not None

        # Set transformer usage flags (static - based on __init__ params)
        tags.forecaster_tags.uses_target_transformer = self.target_transformer is not None
        tags.forecaster_tags.uses_feature_transformer = self.feature_transformer is not None

        # A forecaster is stateful if it uses a stateful transformer.
        # Subclasses that are intrinsically stateful override __sklearn_tags__
        # and set forecaster_tags.stateful = True directly.
        stateful = False

        if not stateful and self.target_transformer is not None:
            target_tags = self.target_transformer.__sklearn_tags__().transformer_tags
            if target_tags is not None:
                stateful = target_tags.stateful

        if not stateful and self.feature_transformer is not None:
            feature_tags = self.feature_transformer.__sklearn_tags__().transformer_tags
            if feature_tags is not None:
                stateful = feature_tags.stateful

        tags.forecaster_tags.stateful = stateful

        # forecaster_type is set by subclasses in their __sklearn_tags__() method
        # as a frozenset (e.g., POINT, INTERVAL, POINT_INTERVAL, CLASS_PROBA)

        # Merge class-level _tags dict (flat keys) into tag dataclasses.
        # Walk MRO in reverse so most-derived class wins.
        merged_tags: dict[str, Any] = {}
        for klass in reversed(type(self).__mro__):
            class_tags = klass.__dict__.get("_tags")
            if class_tags and isinstance(class_tags, dict):
                merged_tags.update(class_tags)

        if merged_tags:
            for key, value in merged_tags.items():
                # Map flat key to the correct tag dataclass field
                if tags.forecaster_tags is not None and hasattr(tags.forecaster_tags, key):
                    setattr(tags.forecaster_tags, key, value)
                elif tags.transformer_tags is not None and hasattr(tags.transformer_tags, key):  # pragma: no cover
                    setattr(tags.transformer_tags, key, value)
                elif tags.input_tags is not None and hasattr(tags.input_tags, key):
                    setattr(tags.input_tags, key, value)
                elif tags.target_tags is not None and hasattr(tags.target_tags, key):
                    setattr(tags.target_tags, key, value)
                elif hasattr(tags, key):
                    setattr(tags, key, value)

        return tags

    @property
    def _observation_horizon(self) -> int:
        """Internal observation horizon set by the forecaster.

        Subclasses can override this as a ``@property`` to compute from
        constructor params (e.g., ``return self.seasonality``), or set it
        directly via ``self._observation_horizon = value``.

        Returns
        -------
        int
            Forecaster-specific observation horizon (default 0).

        """
        return getattr(self, "_oh_value", 0)

    @_observation_horizon.setter
    def _observation_horizon(self, value: int) -> None:
        """Set the internal observation horizon value."""
        self._oh_value = value

    @property
    def observation_horizon(self) -> int:
        """Get the number of time steps needed for stateful operations.

        The observation horizon defines how many recent observations the forecaster
        needs to maintain in its memory.  Subclasses can override this as a
        ``@property`` to compute from constructor params (e.g., ``return
        self.seasonality``).

        Returns
        -------
        int
            Number of time steps to retain.

        """
        # Compute transformer observation horizons (only available after fit)
        target_observation_horizon = 0
        if self.target_transformer is not None and hasattr(self, "target_transformer_"):
            if isinstance(self.target_transformer_, dict):
                # In panel data, all local transformers share the same horizon
                first_transformer = next(iter(self.target_transformer_.values()))
                if first_transformer is not None:
                    target_observation_horizon = typing_cast(BaseTransformer, first_transformer).observation_horizon
            elif isinstance(self.target_transformer_, BaseTransformer):
                target_observation_horizon = self.target_transformer_.observation_horizon

        # Compute feature transformer observation horizon
        feature_observation_horizon = 0
        if self.feature_transformer is not None and hasattr(self, "feature_transformer_"):
            if isinstance(self.feature_transformer_, dict):
                first_transformer = next(iter(self.feature_transformer_.values()))
                if first_transformer is not None:
                    feature_observation_horizon = typing_cast(BaseTransformer, first_transformer).observation_horizon
            elif isinstance(self.feature_transformer_, BaseTransformer):
                feature_observation_horizon = self.feature_transformer_.observation_horizon

        self_observation_horizon = self._observation_horizon
        return max(self_observation_horizon, target_observation_horizon, feature_observation_horizon)

    def _validate_pre_fit(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> tuple[
        pl.DataFrame,
        pl.DataFrame | None,
        dict[str, list[str]],
        dict[str, list[str]] | None,
    ]:
        """Validate inputs and detect panel structure before fitting.

        This method performs shared validation for both global and panel data,
        setting `fit_forecasting_horizon_` and returning panel groups info.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X_actual : pl.DataFrame or None, default=None
            Features time series.
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"`` columns.

        Returns
        -------
        y : pl.DataFrame
            Validated target time series.
        X_actual : pl.DataFrame or None
            Validated feature time series.
        y_panel_groups : dict[str, list[str]]
            Panel groups from y (empty dict if global data).
        X_panel_groups : dict[str, list[str]] or None
            Panel groups from X_actual (None if X_actual is None).

        """
        y, X_actual, _ = validate_forecaster_data(
            self,
            y,
            X_actual,
            reset=True,
            X_future=X_future,
            X_forecast=X_forecast,
        )
        self.fit_forecasting_horizon_ = forecasting_horizon

        _, y_panel_groups = inspect_panel(y)
        X_panel_groups = None
        if X_actual is not None:
            _, X_panel_groups = inspect_panel(X_actual)

            if len(X_panel_groups) and list(X_panel_groups.keys()) != list(y_panel_groups.keys()):
                raise ValueError("`X_actual` and `y` do not have the same local group names.")

        # Validate that X_actual is provided when target_as_feature=None
        # and a feature transformer is configured.  Failing early here avoids
        # a confusing error at predict time inside _build_feature_input().
        if (
            getattr(self, "target_as_feature", None) is None
            and getattr(self, "feature_transformer", None) is not None
            and X_actual is None
        ):
            raise ValueError(
                "target_as_feature=None with a feature_transformer requires X_actual to be provided, but X_actual is None."
            )

        # Validate that X_actual is provided when target_as_feature=None and the
        # forecaster requires exogenous features.  Forecasters with
        # requires_exogenous=False (e.g. SeasonalNaive, stationarity, decomposition)
        # work without any feature matrix.
        sklearn_tags = self.__sklearn_tags__()
        if (
            getattr(self, "target_as_feature", None) is None
            and X_actual is None
            and sklearn_tags.forecaster_tags is not None
            and sklearn_tags.forecaster_tags.requires_exogenous
        ):
            raise ValueError(
                "target_as_feature=None requires X_actual to be provided when the "
                "forecaster uses exogenous features (requires_exogenous=True), "
                "but X_actual is None."
            )

        # Warn when a forecaster that does not use exogenous receives X_future/X_forecast
        if (
            sklearn_tags.forecaster_tags is not None
            and not sklearn_tags.forecaster_tags.requires_exogenous
            and (X_future is not None or X_forecast is not None)
        ):
            warnings.warn(
                f"{self.__class__.__name__} has requires_exogenous=False. X_future and X_forecast will be ignored.",
                UserWarning,
                stacklevel=4,
            )

        return y, X_actual, y_panel_groups, X_panel_groups

    def _pre_fit(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> tuple[pl.DataFrame | dict[str, pl.DataFrame], pl.DataFrame | dict[str, pl.DataFrame] | None]:
        """Preprocess and transform inputs before fitting.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X_actual : pl.DataFrame or None, default=None
            Features time series.
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.

        Returns
        -------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target.
        X_t : pl.DataFrame or dict[str, pl.DataFrame] or None
            Transformed features.

        Notes
        -----
        For type-narrowed returns, child classes can call mixin methods directly:
        - `BaseStandardForecaster._pre_fit_standard(self, ...)` -> `tuple[DataFrame, DataFrame | None]`
        - `BasePanelForecaster._pre_fit_panel(self, ...)` -> `tuple[dict, dict | None]`

        """
        y, X_actual, y_panel_groups, X_panel_groups = self._validate_pre_fit(
            y,
            X_actual,
            forecasting_horizon,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        # Dispatch to mixin methods based on panel strategy
        if self.panel_strategy == "multivariate" or not y_panel_groups:
            # Standard data or multivariate strategy (skip panel detection)
            return BaseStandardForecaster._pre_fit_standard(
                self, y, X_actual, forecasting_horizon, X_future=X_future, X_forecast=X_forecast
            )
        else:
            # Panel data with global strategy
            return BasePanelForecaster._pre_fit_panel(
                self,
                y,
                X_actual,
                forecasting_horizon,
                y_panel_groups,
                X_panel_groups,
                X_future=X_future,
                X_forecast=X_forecast,
            )

    @abc.abstractmethod
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> "BaseForecaster":
        """Fit the forecaster to historical data.

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
            values that are windowed forward from each observation time.
            Bypasses the feature transformer.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns. Pivoted by ordinal rank within each vintage group.
            Bypasses the feature transformer.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted forecaster instance.

        Raises
        ------
        ValueError
            If ``y`` is missing the ``"time"`` column, if ``y`` and ``X_actual``
            have mismatched panel group names, or if
            ``target_as_feature=None`` without exogenous features when the
            forecaster requires them.

        """

    def _validate_fit_params(self, forecasting_horizon: StrictInt) -> StrictInt:
        """Validate fit parameters.

        Subclasses can override to add type-specific validation.

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

    def _fit(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        X_t: pl.DataFrame | dict[str, pl.DataFrame] | None,
        forecasting_horizon: StrictInt,
    ) -> None:
        """Model-specific fitting logic (Tier 1 hook).

        Called by ``fit()`` after validation and ``_pre_fit()`` have run.
        Override this in simple subclasses instead of overriding ``fit()``
        directly.

        The default implementation does nothing, so forecasters with no
        custom fitting logic (e.g. ``SeasonalNaive``) do not need to
        override it.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series. A single DataFrame for
            standard data, or a dict keyed by group name for panel data
            with ``panel_strategy="global"``.
        X_t : pl.DataFrame or dict[str, pl.DataFrame] or None
            Transformed features. Same structure as ``y_t``. ``None``
            when no exogenous features are provided.
        forecasting_horizon : int
            Number of time steps to forecast.

        Notes
        -----
        The following ``self`` attributes are available after ``_pre_fit()``:

        - ``fit_forecasting_horizon_`` : int
        - ``interval_`` : str (detected time interval)
        - ``groups_`` : dict or None (panel groups)
        - ``local_y_schema_`` : dict (target column schema)
        - ``local_y_t_schema_`` : dict (transformed target schema)
        - ``local_X_actual_schema_`` : dict or None (feature schema)
        - ``local_X_t_schema_`` : dict or None (transformed feature schema)
        - ``shared_X_actual_schema_`` : dict or None
        - ``n_features_in_`` : int
        - ``feature_names_in_`` : list[str]
        - ``observed_time_`` : dict or pl.Series (observation timestamps)
        - ``target_transformer_`` : fitted transformer or None
        - ``feature_transformer_`` : fitted transformer or None

        """

    def rewind(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> "BaseForecaster":
        """Rewind observation buffers to the last ``observation_horizon`` rows.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations to restore the observation state
            to. Must align with ``y``.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"`` columns.

        Returns
        -------
        self
            The forecaster with observation buffers rewound to the last
            ``observation_horizon`` rows.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forecaster has not been fitted yet.
        ValueError
            If ``y`` / ``X_actual`` have invalid structure, non-monotonic time
            index, or ``groups`` contains names not seen during fit.

        """
        check_is_fitted(
            self,
            ["local_y_schema_", "local_X_actual_schema_", "shared_X_actual_schema_", "groups_"],
        )

        # Validate schema, enforce column order, and validate groups (no continuity check - rewind sets new window)
        y, X_actual, groups = validate_forecaster_data(
            self,
            y,
            X_actual,
            reset=False,
            groups=groups,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        # Special handling for forecasters with no observation horizon
        if self.observation_horizon == 0:  # pragma: no cover
            # If there is no observation horizon, only check for time column presence
            if "time" not in y.columns:
                raise ValueError("y must contain 'time' column.")
            if X_actual is not None and "time" not in X_actual.columns:
                raise ValueError("X_actual must contain 'time' column.")

        # Dispatch to mixin methods
        if self.groups_ is None:
            BaseStandardForecaster._rewind_standard(self, y, X_actual, X_future=X_future, X_forecast=X_forecast)
        else:
            BasePanelForecaster._rewind_panel(self, y, X_actual, groups, X_future=X_future, X_forecast=X_forecast)

        return self

    def observe(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> "BaseForecaster":
        """Observe new data and update observation buffers without refitting.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            New actual feature observations with a ``"time"`` column
            aligned with ``y``. Passed through the feature transformer to
            update the internal observation state.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"`` columns.

        Returns
        -------
        self
            The forecaster with updated observation buffers from new data,
            without refitting.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forecaster has not been fitted yet.
        ValueError
            If ``y`` / ``X_actual`` have invalid structure, non-monotonic time
            index, or ``groups`` contains names not seen during fit.

        """
        check_is_fitted(
            self,
            ["local_y_schema_", "local_X_actual_schema_", "shared_X_actual_schema_", "groups_"],
        )

        # Validate schema, enforce column order, and validate groups (includes continuity check)
        y, X_actual, groups = validate_forecaster_data(
            self,
            y,
            X_actual,
            reset=False,
            groups=groups,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        # Dispatch to mixin methods
        if self.groups_ is None:
            BaseStandardForecaster._observe_standard(self, y, X_actual, X_future=X_future, X_forecast=X_forecast)
        else:
            BasePanelForecaster._observe_panel(self, y, X_actual, groups, X_future=X_future, X_forecast=X_forecast)

        return self

    def _predict_with_step_override(
        self,
        *,
        X_future: pl.DataFrame | None,
        X_forecast: pl.DataFrame | None,
        predict_fn: Callable[[], pl.DataFrame],
    ) -> pl.DataFrame:
        """Run predict_fn with temporarily overridden step columns.

        When X_future or X_forecast is provided, re-derives ALL step columns
        from effective raws and swaps them into ``_X_t_observed``. After
        ``predict_fn`` returns, the original step columns and raws are
        restored. This enables multi-vintage predictions without mutating
        forecaster state.

        Also temporarily sets ``_X_future_raw_`` / ``_X_forecast_raw_`` so
        that ``_recursive_predict``'s ``deepcopy(self)`` inherits the
        override (each recursive block's ``observe()`` auto re-derives step
        columns from stored raws per Decision 21).

        Parameters
        ----------
        X_future : pl.DataFrame or None
            Known future features override. If None, uses stored raw.
        X_forecast : pl.DataFrame or None
            External forecast override. If None, uses stored raw.
        predict_fn : callable
            ``predict_fn() -> pl.DataFrame``. Called with overridden state.

        Returns
        -------
        pl.DataFrame
            Result of ``predict_fn()``.

        """
        if not self._step_column_names_:
            # No step columns at all: nothing to swap
            return predict_fn()

        if X_future is None and X_forecast is None:
            # No override requested: skip swap
            return predict_fn()

        # Resolve effective raws
        X_future_eff = X_future if X_future is not None else self._X_future_raw_
        X_forecast_eff = X_forecast if X_forecast is not None else self._X_forecast_raw_

        # Re-derive ALL step columns for current observed_time_
        # Panel data stores observed_time_ as a dict; use first group's time
        obs_time = (
            self.observed_time_[next(iter(self.observed_time_))]
            if isinstance(self.observed_time_, dict)
            else self.observed_time_
        )
        X_step_new = _derive_step_columns(
            X_future_eff,
            X_forecast_eff,
            pl.Series([obs_time]),
            self.fit_forecasting_horizon_,
            self.interval_,
        )

        step_col_list = sorted(self._step_column_names_)

        # For panel data, per-group DataFrames use unprefixed step column names
        if isinstance(self._X_t_observed, dict):
            local_step_cols = sorted(self._step_schema_per_group_) if self._step_schema_per_group_ else []
        else:
            local_step_cols = step_col_list

        # Save current state
        saved_future_raw = self._X_future_raw_
        saved_forecast_raw = self._X_forecast_raw_

        if isinstance(self._X_t_observed, dict):
            # Panel: save per-group step columns (unprefixed)
            saved_step_data = {}
            for group_name, group_df in self._X_t_observed.items():
                cols_present = [c for c in local_step_cols if c in group_df.columns]  # ty: ignore[unresolved-attribute]
                if cols_present:
                    saved_step_data[group_name] = group_df.select(cols_present)  # ty: ignore[unresolved-attribute]
        else:
            # Standard: save step columns from last row
            cols_present = [c for c in local_step_cols if c in self._X_t_observed.columns]  # ty: ignore[unresolved-attribute]
            saved_step_data = self._X_t_observed.select(cols_present) if cols_present else None  # ty: ignore[unresolved-attribute]

        try:
            # Swap raws (for deepcopy in _recursive_predict)
            if X_future is not None:
                self._X_future_raw_ = X_future
            if X_forecast is not None:
                self._X_forecast_raw_ = X_forecast

            # Swap step columns in _X_t_observed
            if X_step_new is not None:
                if isinstance(self._X_t_observed, dict):
                    from yohou.utils.panel import get_group_df  # noqa: PLC0415

                    for group_name, group_df in self._X_t_observed.items():
                        cols_to_drop = [c for c in local_step_cols if c in group_df.columns]  # ty: ignore[unresolved-attribute]
                        new_group_step = get_group_df(X_step_new, group_name, self._step_schema_per_group_).select(  # ty: ignore[invalid-argument-type]
                            ~cs.by_name("time")
                        )
                        if cols_to_drop:
                            updated = group_df.drop(cols_to_drop)  # ty: ignore[unresolved-attribute]
                            self._X_t_observed[group_name] = pl.concat([updated, new_group_step], how="horizontal")  # ty: ignore[invalid-assignment]
                        else:
                            self._X_t_observed[group_name] = pl.concat([group_df, new_group_step], how="horizontal")  # ty: ignore[invalid-assignment]
                else:
                    new_step_only = X_step_new.select(~cs.by_name("time"))
                    cols_to_drop = [c for c in local_step_cols if c in self._X_t_observed.columns]  # ty: ignore[unresolved-attribute]
                    if cols_to_drop:
                        updated = self._X_t_observed.drop(cols_to_drop)  # ty: ignore[unresolved-attribute]
                        self._X_t_observed = pl.concat([updated, new_step_only], how="horizontal")

            return predict_fn()

        finally:
            # Restore raws
            self._X_future_raw_ = saved_future_raw
            self._X_forecast_raw_ = saved_forecast_raw

            # Restore step columns
            if isinstance(self._X_t_observed, dict):
                for group_name, saved_df in saved_step_data.items():  # ty: ignore[unresolved-attribute]
                    group_df = self._X_t_observed[group_name]
                    cols_to_drop = [c for c in local_step_cols if c in group_df.columns]  # ty: ignore[unresolved-attribute]
                    if cols_to_drop:
                        restored = group_df.drop(cols_to_drop)  # ty: ignore[unresolved-attribute]
                        self._X_t_observed[group_name] = pl.concat([restored, saved_df], how="horizontal")
            elif saved_step_data is not None:
                cols_to_drop = [c for c in local_step_cols if c in self._X_t_observed.columns]  # ty: ignore[unresolved-attribute]
                if cols_to_drop:
                    restored = self._X_t_observed.drop(cols_to_drop)  # ty: ignore[unresolved-attribute]
                    self._X_t_observed = pl.concat([restored, saved_step_data], how="horizontal")

    def _recursive_predict(
        self,
        *,
        forecasting_horizon: int,
        groups: list[str] | None,
        step_fn: Callable[["BaseForecaster", list[str]], tuple[pl.DataFrame, pl.DataFrame]],
        derive_observation_fn: Callable[
            ["BaseForecaster", pl.DataFrame],
            pl.DataFrame,
        ],
    ) -> pl.DataFrame:
        """Shared recursive multi-step prediction loop.

        Produces predictions by repeatedly calling ``step_fn`` to get one
        forecast block, then ``derive_observation_fn`` to convert that
        prediction into a y observation that is fed back via ``observe()``
        for the next recursive step.

        X_future step columns are auto re-derived during each block's
        ``observe()`` from ``_X_future_raw_`` (inherited via deepcopy from
        ``_predict_with_step_override``). Do NOT pass explicit X_future or
        X_forecast to ``observe()`` inside this loop.

        Parameters
        ----------
        forecasting_horizon : int
            Total number of time steps to forecast.
        groups : list of str or None
            Panel group prefixes to operate on.
        step_fn : callable
            ``step_fn(forecaster_copy, groups) -> (y_accumulate, y_for_obs)``
            where ``y_accumulate`` is appended to output and ``y_for_obs``
            is passed to ``derive_observation_fn``.
        derive_observation_fn : callable
            ``derive_observation_fn(forecaster_copy, y_for_obs) -> y_obs``
            where ``y_obs`` is passed to ``observe(y=y_obs)``.

        Returns
        -------
        pl.DataFrame
            Concatenated predictions with ``"vintage_time"`` set to the
            first step's value and tail-trimmed to ``forecasting_horizon``.

        Raises
        ------
        ValueError
            If ``forecasting_horizon > fit_forecasting_horizon_`` and the
            forecaster was fitted with ``X_forecast``. Recursive prediction
            cannot re-derive vintage-dependent forecast columns across
            blocks. Use ``ForecastedFeatureForecaster`` instead.

        """
        if forecasting_horizon > self.fit_forecasting_horizon_ and self._X_forecast_raw_ is not None:
            msg = (
                f"Recursive prediction (forecasting_horizon={forecasting_horizon} > "
                f"fit_forecasting_horizon={self.fit_forecasting_horizon_}) is not "
                f"supported when X_forecast was provided at fit time. X_forecast "
                f"step columns are vintage-dependent and cannot be re-derived "
                f"across recursive blocks. Use ForecastedFeatureForecaster to "
                f"compose a forecaster that generates its own step forecasts."
            )
            raise ValueError(msg)

        forecaster = deepcopy(self)

        y_pred = pl.DataFrame()
        for step in range(0, forecasting_horizon, self.fit_forecasting_horizon_):
            y_accumulate, y_for_obs = step_fn(forecaster, groups or [])
            y_pred = pl.concat([y_pred, y_accumulate])

            if step + self.fit_forecasting_horizon_ < forecasting_horizon:
                y_obs = derive_observation_fn(forecaster, y_for_obs)
                # observe with X_actual=None: step columns auto re-derived
                # from _X_future_raw_ / _X_forecast_raw_ (Decision 21)
                forecaster.observe(y_obs)

        y_pred = y_pred.with_columns(vintage_time=y_pred["vintage_time"][0])

        if forecasting_horizon % self.fit_forecasting_horizon_:
            end = self.fit_forecasting_horizon_ - forecasting_horizon % self.fit_forecasting_horizon_
            y_pred = y_pred[:-end]

        return y_pred

    def _observe_predict_loop(
        self,
        *,
        predict_fn: Callable[..., pl.DataFrame],
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        groups: list[str] | None,
        stride: int,
        observe_fn: Callable[..., Any] | None = None,
        **predict_kwargs: Any,
    ) -> pl.DataFrame:
        """Shared observe-then-predict rolling loop.

        Produces an initial prediction, then repeatedly observes a
        ``stride``-sized slice of ``y`` and re-predicts. Used by
        ``observe_predict``, ``observe_predict_interval``, and
        ``observe_predict_class_proba``.

        When ``observe_fn`` is ``None`` (default), step columns are
        pre-computed once at entry via ``_derive_step_columns`` and
        injected through ``_observe_with_precomputed_steps``. When
        ``observe_fn`` is provided (meta-forecasters), the callback
        handles observation and each child derives its own step columns.

        Parameters
        ----------
        predict_fn : callable
            The predict method to call (e.g. ``self.predict``,
            ``self.predict_interval``, ``self.predict_class_proba``).
        y : pl.DataFrame
            Historical target observations to incrementally observe.
        X_actual : pl.DataFrame or None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Sliced and observed incrementally at each step of
            the rolling loop.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.
        groups : list of str or None
            Panel group prefixes to operate on.
        stride : int
            Number of rows to observe between successive predictions.
        observe_fn : callable or None, default=None
            Optional callback for meta-forecasters. When provided, called
            as ``observe_fn(y_slice, X_actual=X_obs_slice, X_future=...,
            X_forecast=...)`` instead of using pre-computed step columns.
        **predict_kwargs : dict
            Extra keyword arguments forwarded to ``predict_fn``
            (e.g. ``forecasting_horizon``, ``coverage_rates``).

        Returns
        -------
        pl.DataFrame
            Concatenated predictions from the initial call plus one
            prediction after each observe step.

        Notes
        -----
        When ``len(y) % stride != 0``, the last observe call consumes
        fewer than ``stride`` rows. The prediction still outputs the
        full forecasting horizon, so no data is lost. However, this
        creates one extra vintage whose observed window is shorter
        than the others. Partial vintages are automatically truncated
        at score time by the scorer.

        """
        # Pre-compute step columns once for all observation times
        step_columns_full = None
        if observe_fn is None:
            step_columns_full = _derive_step_columns(
                X_future,
                X_forecast,
                y["time"],
                self.fit_forecasting_horizon_,
                self.interval_,
            )

        # Initial predict (reads _X_t_observed set during fit/last observe)
        y_pred_i = predict_fn(groups=groups, **predict_kwargs)
        y_pred = y_pred_i

        for i in range(0, len(y), stride):
            y_slice = y[i : i + stride]

            X_obs_slice = None
            if X_actual is not None:
                X_obs_slice = X_actual.join(y_slice.select("time"), on="time", how="semi")

            if observe_fn is not None:
                # Meta-forecaster path: delegate observe to callback
                observe_fn(y_slice, X_actual=X_obs_slice, X_future=X_future, X_forecast=X_forecast)
            elif step_columns_full is not None:
                # Standard/panel path with pre-computed step columns
                X_step_slice = step_columns_full.join(y_slice.select("time"), on="time", how="semi")

                if self.groups_ is None:
                    BaseStandardForecaster._observe_with_precomputed_steps_standard(
                        self, y_slice, X_obs_slice, X_step_slice
                    )
                else:
                    BasePanelForecaster._observe_with_precomputed_steps_panel(
                        self, y_slice, X_obs_slice, X_step_slice, groups or []
                    )
            else:
                # No step columns and no observe_fn: fall back to regular observe
                self.observe(y=y_slice, X_actual=X_obs_slice, groups=groups)

            y_pred_i = predict_fn(groups=groups, **predict_kwargs)
            y_pred = pl.concat([y_pred, y_pred_i])

        return y_pred

    def _add_time_columns(self, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Add time metadata columns to predictions.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Predictions without time columns.

        Returns
        -------
        pl.DataFrame
            Predictions with vintage_time and time columns.

        """
        # Dispatch to mixin methods
        if self.groups_ is not None:
            return BasePanelForecaster._add_time_columns_panel(self, y_pred)
        else:
            return BaseStandardForecaster._add_time_columns_standard(self, y_pred)

    def _predict_one(
        self,
        groups: list[str],
        **params,
    ) -> pl.DataFrame:
        """Predicts `_fit_forecasting_horizon` steps from the observation horizon.

        Parameters
        ----------
        groups : list of str
            Panel group names to predict for.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        raise NotImplementedError(f"The forecaster of type {type(self)} does not implement_predict_one.")

    def _predict(
        self,
        groups: list[str],
        **predict_one_params,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Generate one-step or multi-step prediction.

        Parameters
        ----------
        groups : list of str or None, default=None
            Group prefixes for panel data:
            - If None: predict for all groups
            - If list of str: predict only for the specified panel groups
            Parameter is ignored if the forecaster was not fitted on panel data.
        **predict_one_params : dict
            Params to the _predict_one method.

        Returns
        -------
        y_pred_step : pl.DataFrame
            Predicted time series in transformed space.
        y_pred_step_inv : pl.DataFrame
            Inverse transformed predicted time series (original scale).

        """
        y_pred_step = self._predict_one(groups=groups, **predict_one_params)

        if self.target_transformer is None:
            if not groups:
                # Non-panel data

                y_pred_step = cast(y_pred_step, self.local_y_schema_)

            else:
                # Panel data
                y_pred_step = cast(
                    y_pred_step,
                    {
                        f"{panel_group_name}__{col}": dtype
                        for panel_group_name in groups
                        for col, dtype in self.local_y_schema_.items()
                    },
                )

            y_pred_step_inv = y_pred_step

        elif not groups:
            # Non-panel data
            assert self.target_transformer_ is not None
            assert not isinstance(self.target_transformer_, dict)

            # Remove "vintage_time" before inverse_transform (transformers don't handle it)
            vintage_time = y_pred_step.select(cs.by_name("vintage_time"))
            y_pred_step_no_obs = y_pred_step.select(~cs.by_name("vintage_time"))

            transformer = typing_cast(Any, self.target_transformer_)
            y_pred_step_inv = transformer.inverse_transform(
                X_t=y_pred_step_no_obs,
                X_p=self._y_observed,
            )

            # Cast to restore original dtypes
            y_pred_step_inv_cast = cast(y_pred_step_inv.select(~cs.by_name("time")), self.local_y_schema_)

            # Reconstruct with time column
            y_pred_step_inv = pl.concat(
                [y_pred_step_inv.select(cs.by_name("time")), y_pred_step_inv_cast],
                how="horizontal",
            )

            # Add "vintage_time" back
            y_pred_step_inv = pl.concat([vintage_time, y_pred_step_inv], how="horizontal")

        else:
            # Panel data
            y_pred_step_inv_dict = {}

            # Type narrowing: target_transformer_ is not None and is dict in panel data branch
            assert self.target_transformer_ is not None
            assert isinstance(self.target_transformer_, dict)
            assert self._y_observed is not None
            assert isinstance(self._y_observed, dict)
            target_transformers = typing_cast(dict[str, BaseTransformer | None], self.target_transformer_)
            y_observed_dict = typing_cast(dict[str, pl.DataFrame | None], self._y_observed)

            for panel_group_name in groups:
                transformer = target_transformers[panel_group_name]
                assert transformer is not None

                # Remove "vintage_time" before extracting group data
                vintage_time = y_pred_step.select(cs.by_name("vintage_time")).head(1)

                # Extract the group's columns (in transformed space, with prefix)
                group_cols = [c for c in y_pred_step.columns if c.startswith(f"{panel_group_name}__")]
                y_pred_step_group = y_pred_step.select(cs.by_name("time") | cs.by_name(group_cols))

                # Strip group prefix so transformer sees local column names
                prefix = f"{panel_group_name}__"
                rename_strip = {c: c[len(prefix) :] for c in group_cols}
                y_pred_step_group = y_pred_step_group.rename(rename_strip)

                # Inverse transform (works with unprefixed/local columns)
                y_observed_local = y_observed_dict[panel_group_name]
                y_pred_step_group_inv = transformer.inverse_transform(
                    X_t=y_pred_step_group,
                    X_p=y_observed_local,
                )

                # Cast to restore original dtypes
                y_pred_step_group_inv_cast = cast(
                    y_pred_step_group_inv.select(~cs.by_name("time")), self.local_y_schema_
                )

                # Rename to add prefix back
                rename_map = {col: f"{panel_group_name}__{col}" for col in y_pred_step_group_inv_cast.columns}
                y_pred_step_group_inv_cast = y_pred_step_group_inv_cast.rename(rename_map)

                # Reconstruct with time column
                y_pred_step_group_inv = pl.concat(
                    [
                        y_pred_step_group_inv.select(cs.by_name("time")),
                        y_pred_step_group_inv_cast,
                    ],
                    how="horizontal",
                )

                # Add "vintage_time" back
                y_pred_step_group_inv = pl.concat([vintage_time, y_pred_step_group_inv], how="horizontal")

                # Store in dict (without time columns)
                y_pred_step_inv_dict[panel_group_name] = y_pred_step_group_inv.select(
                    ~cs.by_name("vintage_time") & ~cs.by_name("time")
                )

            times = y_pred_step.select(cs.by_name("vintage_time") | cs.by_name("time"))
            y_pred_inv_cols = pl.concat(list(y_pred_step_inv_dict.values()), how="horizontal")

            y_pred_step_inv = pl.concat([times, y_pred_inv_cols], how="horizontal")

        return y_pred_step, y_pred_step_inv

    def get_metadata_routing(self) -> MetadataRouter:
        """Get metadata routing for this forecaster.

        BaseForecaster is both a consumer AND a router:
        - Consumer: Can accept metadata like forecasting_horizon
        - Router: Forwards metadata to target_transformer and feature_transformer

        Subclasses with additional nested estimators should call super() and
        add their own child routing.

        Returns
        -------
        router : MetadataRouter
            Router that forwards metadata to transformers.
        """
        router = MetadataRouter(owner=self)
        router.add_self_request(self)

        # Route to target_transformer if present
        # This allows target_transformer to receive metadata if it requests it
        if hasattr(self, "target_transformer") and self.target_transformer is not None:
            router.add(
                target_transformer=self.target_transformer,
                method_mapping=MethodMapping().add(caller="fit", callee="fit").add(caller="fit", callee="transform"),
            )

        # Route to feature_transformer if present
        if hasattr(self, "feature_transformer") and self.feature_transformer is not None:
            router.add(
                feature_transformer=self.feature_transformer,
                method_mapping=MethodMapping().add(caller="fit", callee="fit").add(caller="fit", callee="transform"),
            )

        return router
