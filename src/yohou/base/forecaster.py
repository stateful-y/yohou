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
from sklearn.base import BaseEstimator, clone
from sklearn.utils.metadata_routing import MetadataRouter, MethodMapping
from sklearn.utils.validation import check_is_fitted

from yohou.base.forecast_transformer import FORECAST_INDEX_COLS, BaseForecastTransformer
from yohou.base.panel import BasePanelForecaster
from yohou.base.standard import BaseStandardForecaster
from yohou.base.transformer import BaseActualTransformer
from yohou.base.utils import _densify_forecast_vintages, _derive_step_columns, _require_forecast_transformer
from yohou.utils import (
    Tags,
    cast,
    check_panel_groups_match,
    get_group_df,
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
    target_transformer : instance of `BaseActualTransformer` or None, default=None
        Transformer used to transform the target time series into the new target.
    actual_transformer : instance of `BaseActualTransformer` or None, default=None
        Transformer used to transform the feature time series into features.
    forecast_transformer : instance of `BaseForecastTransformer` or None, default=None
        Transformer applied to ``X_forecast`` before step columns are derived,
        so the step columns reaching the estimator are built from transformed
        values. Must be forecast-kind (vintage-indexed); an actual-kind
        transformer is rejected. ``None`` leaves ``X_forecast`` untouched.
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
    refitting** the model.  ``rewind()`` rebuilds internal buffers from the
    provided historical window, allowing state to be reset to any prior point
    without refitting.  Together they enable streaming / rolling-window
    evaluation.

    The ``forecasting_horizon`` is set at ``fit`` time but can be
    overridden at ``predict`` time.

    See Also
    --------
    - [`BasePointForecaster`][yohou.point.base.BasePointForecaster] : Base class for point forecasters.
    - [`BaseIntervalForecaster`][yohou.interval.base.BaseIntervalForecaster] : Base class for interval forecasters.
    - [`BaseReductionForecaster`][yohou.base.reduction.BaseReductionForecaster] : Forecasting via sklearn regressors.

    """

    _parameter_constraints: dict = {
        "target_transformer": [BaseActualTransformer, None],
        "actual_transformer": [BaseActualTransformer, None],
        # Loose structurally, strict by tag, mirroring "actual_transformer" above.
        # A forecast-kind composition (a FeatureUnion of forecast transformers) is a
        # BaseActualTransformer subclass reporting kind="forecast", so listing only
        # BaseForecastTransformer here would reject it. _require_forecast_transformer
        # rejects what reports kind="actual".
        "forecast_transformer": [BaseForecastTransformer, BaseActualTransformer, None],
        "target_as_feature": [StrOptions({"transformed", "raw"}), None],
        "panel_strategy": [StrOptions({"global", "multivariate"})],
    }

    # Fitted attributes (set during fit())
    interval_: str

    def __init__(
        self,
        *,
        target_transformer: BaseActualTransformer | None = None,
        actual_transformer: BaseActualTransformer | None = None,
        forecast_transformer: BaseForecastTransformer | None = None,
        target_as_feature: Literal["transformed", "raw"] | None = "transformed",
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        self.target_transformer = target_transformer
        self.actual_transformer = actual_transformer
        self.forecast_transformer = forecast_transformer
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
        tags.forecaster_tags.uses_actual_transformer = self.actual_transformer is not None
        tags.forecaster_tags.uses_forecast_transformer = self.forecast_transformer is not None

        # A forecaster is stateful if it uses a stateful transformer.
        # Subclasses that are intrinsically stateful override __sklearn_tags__
        # and set forecaster_tags.stateful = True directly.
        stateful = False

        if self.target_transformer is not None:
            target_tags = self.target_transformer.__sklearn_tags__().transformer_tags
            if target_tags is not None:
                stateful = target_tags.stateful

        if not stateful and self.actual_transformer is not None:
            feature_tags = self.actual_transformer.__sklearn_tags__().transformer_tags
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
                    target_observation_horizon = typing_cast(
                        BaseActualTransformer, first_transformer
                    ).observation_horizon
            elif isinstance(self.target_transformer_, BaseActualTransformer):
                target_observation_horizon = self.target_transformer_.observation_horizon

        # Compute actual transformer observation horizon
        feature_observation_horizon = 0
        if self.actual_transformer is not None and hasattr(self, "actual_transformer_"):
            if isinstance(self.actual_transformer_, dict):
                first_transformer = next(iter(self.actual_transformer_.values()))
                if first_transformer is not None:
                    feature_observation_horizon = typing_cast(
                        BaseActualTransformer, first_transformer
                    ).observation_horizon
            elif isinstance(  # pragma: no branch - actual_transformer_ is a dict or a BaseActualTransformer, never neither
                self.actual_transformer_, BaseActualTransformer
            ):
                feature_observation_horizon = self.actual_transformer_.observation_horizon

        self_observation_horizon = self._observation_horizon
        return max(self_observation_horizon, target_observation_horizon, feature_observation_horizon)

    def _fit_forecast_transformer(
        self,
        X_forecast: pl.DataFrame | None,
        forecasting_horizon: int,
        groups: list[str] | None = None,
    ) -> pl.DataFrame | None:
        """Fit the ``forecast_transformer`` slot, assert the horizon, and transform.

        Ordering here is load-bearing: the slot must be fitted before
        ``min_vintage_rows`` is readable, the horizon assert must precede
        ``_derive_step_columns``, and the returned frame is what derivation
        consumes.

        Parameters
        ----------
        X_forecast : pl.DataFrame or None
            The raw wide ``X_forecast`` frame, or ``None``.
        forecasting_horizon : int
            The fit horizon, checked against the slot's ``min_vintage_rows``.
        groups : list of str or None, default=None
            Panel group names for the per-group fit, or ``None`` for a single
            instance over the wide frame (standard and multivariate).

        Returns
        -------
        pl.DataFrame or None
            The transformed frame, or the input unchanged when the slot is unset.

        """
        self.forecast_transformer_ = None
        # Densify the vintage axis unconditionally, before the slot check and
        # before fitting the transformer, so every forecaster that consumes
        # X_forecast resolves each channel per column, with or without a
        # forecast_transformer, and a transformer that is present fits and
        # transforms the dense frame. Placing this after the None early return
        # would skip it for the common case (no transformer), collapsing
        # mixed cadence for a plain or panel forecaster.
        if X_forecast is not None:
            X_forecast = _densify_forecast_vintages(X_forecast)

        if self.forecast_transformer is None:
            return X_forecast

        # Reject a wrong-kind slot at fit whether or not X_forecast is present. A
        # wrong-kind transformer is a static misconfiguration, so it must not be
        # silently accepted on a fit that omits X_forecast and then surface only if
        # X_forecast later appears at predict. The guard therefore runs before the
        # X_forecast-None early return.
        _require_forecast_transformer(self.forecast_transformer, "forecast_transformer")

        if X_forecast is None:
            return X_forecast

        if groups is None:
            self.forecast_transformer_ = clone(self.forecast_transformer).fit(X_forecast)
        else:
            schema = self._build_X_forecast_schema()
            self.forecast_transformer_ = {
                group_name: clone(self.forecast_transformer).fit(
                    get_group_df(
                        df=X_forecast,
                        group_name=group_name,
                        schema=schema,
                        key_cols=FORECAST_INDEX_COLS,
                    )
                )
                for group_name in groups
            }

        self._assert_forecasting_horizon_meets_minimum(forecasting_horizon)
        X_forecast_t = self._transform_X_forecast(X_forecast)
        self._warn_dead_forecast_steps(X_forecast, X_forecast_t, forecasting_horizon)
        return X_forecast_t

    def _warn_dead_forecast_steps(
        self,
        X_forecast: pl.DataFrame | None,
        X_forecast_t: pl.DataFrame | None,
        forecasting_horizon: int,
    ) -> None:
        """Warn when the transform leaves the nearest-term step columns entirely null.

        A stateful inner consumes rows from the **start** of every vintage, and those
        are the nearest-term forecast steps. Step-column derivation pads them back as
        nulls, which the horizon guard permits: the transform still yields rows, so
        it is partial loss rather than total loss.

        What the guard cannot see is that the loss is the same in every vintage, so
        those step columns are null in every row rather than in some. An entirely
        null column is not merely lossy. It defeats ``nan_handling="drop"``, since
        every row then carries a null, and it breaks estimators that otherwise
        tolerate NaN, including ``HistGradientBoostingRegressor``, whose binner
        cannot bin an all-NaN feature. No horizon value avoids this, because the
        loss scales with the transformer rather than the horizon.

        So this reports the measurement rather than raising: a lifted lag is a
        legitimate feature (the ramp between steps of one vintage), and the caller
        may know their estimator tolerates the dead columns. It exists so that a
        search over ``forecast_transformer__transformer__lag`` cannot trade away the
        nearest-term steps silently.

        Parameters
        ----------
        X_forecast : pl.DataFrame or None
            The raw frame, before the transform.
        X_forecast_t : pl.DataFrame or None
            The transformed frame.
        forecasting_horizon : int
            The fit horizon, which bounds how many step columns exist.

        """
        if self.forecast_transformer_ is None or X_forecast is None or X_forecast_t is None:
            return
        if X_forecast.is_empty() or "vintage_time" not in X_forecast_t.columns:
            return

        before = X_forecast.group_by("vintage_time").len()
        after = X_forecast_t.group_by("vintage_time").len().rename({"len": "len_t"})
        per_vintage = before.join(after, on="vintage_time", how="left")
        # min, not max: the warning claims the loss holds in every vintage, so the
        # least-affected vintage is what licenses it. A vintage dropped whole (too
        # short to fit, which is the documented tail case) joins to null and reads
        # as total loss, so a max would let one dropped tail vintage assert that a
        # transformer consuming nothing empties every step column.
        min_lost = (per_vintage["len"] - per_vintage["len_t"].fill_null(0)).min()

        if (
            min_lost is None
        ):  # pragma: no cover - defensive: per_vintage is non-empty once the empty guard above returns
            return
        lost = typing_cast(int, min_lost)
        if lost < 1:
            return

        dead = min(lost, forecasting_horizon)
        warnings.warn(
            f"forecast_transformer consumes {lost} row(s) from the start of every vintage, "
            f"so step columns 1 to {dead} are derived from no rows and will be null for every "
            f"training instance, not merely for some. Those are the nearest-term forecast steps. "
            f"An entirely null column is dropped by nan_handling='drop' (every row carries a null) "
            f"and rejected by estimators that otherwise tolerate NaN. Raising forecasting_horizon "
            f"does not help, because the loss scales with the transformer rather than the horizon; "
            f"reduce the transformer's row consumption to keep those steps.",
            UserWarning,
            stacklevel=2,
        )

    def _assert_forecasting_horizon_meets_minimum(self, forecasting_horizon: int) -> None:
        """Raise unless the horizon reaches the slot's minimum vintage length.

        A served vintage spans the forecasting horizon, so
        ``forecasting_horizon >= min_vintage_rows`` is the condition under which a
        fresh single-vintage frame survives the transform. Failing here turns two
        silent-degradation modes into a fit-time error.

        Forecast-kind compositions report a minimum too, aggregated from their
        children, so a composition is guarded exactly as the leaf it wraps is.
        Anything that reports no minimum at all is skipped rather than assumed to
        be ``1``: assuming a minimum it never stated would report a guarantee the
        guard has not checked.

        Parameters
        ----------
        forecasting_horizon : int
            The fit horizon.

        Raises
        ------
        ValueError
            If the horizon is below the slot's reported minimum.

        """
        transformer = self.forecast_transformer_
        if transformer is None:
            return
        representative = next(iter(transformer.values()), None) if isinstance(transformer, dict) else transformer

        minimum = getattr(representative, "min_vintage_rows", None)
        if minimum is None:
            return

        if forecasting_horizon < minimum:
            raise ValueError(
                f"forecasting_horizon ({forecasting_horizon}) is below the minimum vintage "
                f"length the forecast_transformer requires ({minimum}). A served vintage spans "
                f"the forecasting horizon, so every vintage would transform to an empty frame. "
                f"Raise forecasting_horizon to at least {minimum}, or reduce the transformer's "
                f"row consumption."
            )

    def _is_step_column(self, name: str) -> bool:
        """Report whether ``name`` names a derived step column, in either spelling.

        Step columns carry two names. Panel-wide frames spell one
        ``{group}__{col}_step_{h}`` and record it in ``_step_column_names_``; the
        per-group frames, and the matrix stacked from them under
        ``panel_strategy="global"``, spell the same column ``{col}_step_{h}`` and
        record it in ``_step_column_local_names_``. Which spelling a caller holds
        depends on which frame it is inspecting, and callers that inspect both
        (or that inspect a stacked matrix without knowing how it was built) cannot
        pick one set and be right.

        The mismatch runs in both directions, which is why the query is asked here
        rather than open-coded per caller. A stacked panel matrix holds local names
        against panel-wide records, so a caller testing only ``_step_column_names_``
        answers "no" for every column, which reads as "there are no step columns"
        rather than as the name mismatch it is. A panel-wide frame holds prefixed
        names against records that are unprefixed whenever the source column was
        global, so a caller testing only the verbatim name misses those. Hence the
        three forms below: the name as given against either record, and the
        group-stripped suffix against either record.

        Over-matching is not a practical risk: every recorded name ends in
        ``_step_{h}`` and is generated by step-column derivation, so an ordinary
        engineered feature does not collide with one. On standard data the two
        sets coincide and this reduces to a plain lookup.

        Parameters
        ----------
        name : str
            A column name from any frame the forecaster handles.

        Returns
        -------
        bool
            True when the column is a derived step column under either spelling.

        """
        step_names = getattr(self, "_step_column_names_", set())
        local_names = getattr(self, "_step_column_local_names_", set())
        if name in step_names or name in local_names:
            return True
        if "__" in name:
            suffix = name.split("__", 1)[1]
            return suffix in step_names or suffix in local_names
        return False

    def _transform_X_forecast(self, X_forecast: pl.DataFrame | None) -> pl.DataFrame | None:
        """Apply the fitted ``forecast_transformer`` to an ``X_forecast`` frame.

        The single entry point for every ``_derive_step_columns`` call site, which
        is what makes the sites safe rather than merely tidy. Under
        ``panel_strategy="global"`` the fitted slot is a dict keyed by group, and
        two of the sites (predict and the observe-predict loop) live in this class,
        shared by standard and panel forecasters rather than overridden per mode.
        A caller-side ``forecast_transformer_.transform(...)`` would raise
        ``AttributeError`` on the dict at exactly those two sites, which carry the
        walk-forward path. Resolving the three shapes here means no caller has to.

        Callers pass the frame the transform should apply to, so an
        ``X_forecast_eff`` fallback must resolve *before* calling this. Applying it
        afterwards would re-transform the already-transformed cached frame.

        Parameters
        ----------
        X_forecast : pl.DataFrame or None
            A raw wide ``X_forecast`` frame, or ``None``.

        Returns
        -------
        pl.DataFrame or None
            The transformed frame, or the input unchanged when the slot is unset
            or the input is ``None``.

        """
        # Densify before transforming, so a provided frame reaches derivation
        # dense on the serve path exactly as it does at fit. Idempotent, so a
        # frame already densified at fit is unchanged; a no-op on uniform cadence.
        if X_forecast is not None:
            X_forecast = _densify_forecast_vintages(X_forecast)

        transformer = getattr(self, "forecast_transformer_", None)
        if transformer is None or X_forecast is None:
            return X_forecast

        if not isinstance(transformer, dict):
            return transformer.transform(X_forecast)

        schema = self._build_X_forecast_schema()
        index_cols = list(FORECAST_INDEX_COLS)
        parts: list[pl.DataFrame] = []
        for group_name, group_transformer in transformer.items():
            if group_transformer is None:
                continue
            local = get_group_df(
                df=X_forecast,
                group_name=group_name,
                schema=schema,
                key_cols=FORECAST_INDEX_COLS,
            )
            transformed = group_transformer.transform(local)
            parts.append(
                transformed.rename({c: f"{group_name}__{c}" for c in transformed.columns if c not in index_cols})
            )

        if not parts:
            return X_forecast

        # Join on both index columns. Joining on "time" alone would fan rows out
        # across every vintage sharing a timestamp, which is silent corruption
        # rather than an error.
        wide = parts[0]
        for part in parts[1:]:
            wide = wide.join(part, on=index_cols, how="full", coalesce=True)
        return wide.sort(index_cols)

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
            External forecasts. See ``fit()`` for full parameter
            description.

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

            # Use the canonical mismatch check so the error message lists both
            # group sets and matches the shape produced elsewhere (e.g.
            # validate_forecaster_data). Global-only X_actual is accepted.
            check_panel_groups_match(y, X_actual)

        # Validate that X_actual is provided when target_as_feature=None
        # and a actual transformer is configured.  Failing early here avoids
        # a confusing error at predict time inside _build_feature_input().
        if (
            getattr(self, "target_as_feature", None) is None
            and getattr(self, "actual_transformer", None) is not None
            and X_actual is None
        ):
            raise ValueError(
                "target_as_feature=None with an actual_transformer requires X_actual to be provided, but X_actual is None."
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
            External forecasts. See ``fit()`` for full parameter
            description.

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
            with ``y``. Processed by the actual transformer to produce
            lags, rolling statistics, and other derived features. If
            ``None``, only target-derived features are used.
        forecasting_horizon : int, default=1
            Number of time steps to forecast into the future.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column. Deterministic
            values that are windowed forward from each observation time.
            Bypasses the actual transformer.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns. Vintage times do not need to align exactly with
            observation times; the latest vintage at or before each
            observation time is selected automatically (as-of matching).
            Bypasses the actual transformer.
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
            have mismatched panel group names, if
            ``target_as_feature=None`` without exogenous features when the
            forecaster requires them, or if ``target_as_feature=None`` and a
            ``actual_transformer`` is configured but ``X_actual`` is ``None``.

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
        - ``observed_time_`` : datetime or dict[str, datetime] (scalar in
          standard mode, per-group dict in panel mode)
        - ``target_transformer_`` : fitted transformer or None
        - ``actual_transformer_`` : fitted transformer or None

        """

    def rewind(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> "BaseForecaster":
        """Rewind state to the end of the provided historical window.

        Re-runs the transformers on the supplied window and retains the last
        ``observation_horizon`` rows in the observation buffer; this is a
        transformer rewind, not a pure buffer tail-slice.

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
            Vintage times do not need to align exactly with observation
            times; the latest vintage at or before ``observed_time_`` is
            selected automatically (as-of matching).

        Returns
        -------
        self
            The forecaster with state rewound to the end of the provided
            window: transformers re-run on that window and the last
            ``observation_horizon`` rows retained in the observation buffer.

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

        Stateful transformers (``target_transformer_``, ``actual_transformer_``)
        are updated via their ``observe()`` method; the model weights themselves
        are not changed.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            New actual feature observations with a ``"time"`` column
            aligned with ``y``. Passed through the actual transformer to
            update the internal observation state.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"`` columns.
            Vintage times do not need to align exactly with observation
            times; the latest vintage at or before ``observed_time_`` is
            selected automatically (as-of matching).

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

        # Reject an empty observation batch up front. An empty ``y`` would
        # otherwise update the transformers' observation state with zero rows,
        # leaving ``_X_t_observed`` empty (0 rows) while ``_y_observed`` keeps
        # its prepended history. The corruption is silent here but surfaces deep
        # in the regressor at the next ``predict`` ("Found array with 0
        # sample(s)"), so fail fast with a clear message instead.
        if len(y) == 0:
            raise ValueError(
                "observe() received an empty `y` (0 rows). There is nothing to "
                "observe; pass at least one new observation row."
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
            External forecast override. See ``predict()`` for full
            parameter description. If None, uses stored raw.
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
        # A supplied frame is raw and is transformed below, after the remap. An
        # omitted one falls back to the transformed cache, which must not be
        # transformed again.
        X_forecast_eff = X_forecast if X_forecast is not None else self._X_forecast_t_

        # Re-derive ALL step columns for current observed_time_
        # Panel data stores observed_time_ as a dict; use first group's time
        obs_time = (
            self.observed_time_[next(iter(self.observed_time_))]
            if isinstance(self.observed_time_, dict)
            else self.observed_time_
        )

        # When the caller overrides X_forecast with a single vintage whose
        # vintage_time differs from observed_time_, remap vintage_time so
        # the join against observation_times in _derive_step_columns works
        # correctly.  Multi-vintage overrides are left untouched (one of
        # their vintages should already match obs_time).
        if X_forecast is not None and X_forecast_eff is not None:
            vintages = X_forecast_eff["vintage_time"].unique()
            if len(vintages) == 1 and vintages[0] != obs_time:
                X_forecast_eff = X_forecast_eff.with_columns(vintage_time=pl.lit(obs_time))
            # Transform after the remap. The remap rewrites vintage_time, so a
            # transformer that keyed on the value would partition differently
            # either side of it. PerVintageActualTransformer partitions by
            # vintage_time without reading the value, so it is insensitive to this
            # order today; the contract does not promise that for every subclass,
            # hence the choice is fixed here rather than left implicit.
            X_forecast_eff = self._transform_X_forecast(X_forecast_eff)

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
            observed_dict = typing_cast(dict[str, pl.DataFrame], self._X_t_observed)
            saved_step_data: dict[str, pl.DataFrame] = {}
            for group_name, group_df in observed_dict.items():
                cols_present = [c for c in local_step_cols if c in group_df.columns]
                if cols_present:
                    saved_step_data[group_name] = group_df.select(cols_present)
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
            if isinstance(self._X_t_observed, dict) and isinstance(saved_step_data, dict):
                restore_dict = typing_cast(dict[str, pl.DataFrame], self._X_t_observed)
                saved_dict = typing_cast(dict[str, pl.DataFrame], saved_step_data)
                for group_name, saved_df in saved_dict.items():
                    group_df = restore_dict[group_name]
                    cols_to_drop = [c for c in local_step_cols if c in group_df.columns]
                    if cols_to_drop:
                        restored = group_df.drop(cols_to_drop)
                        restore_dict[group_name] = pl.concat([restored, saved_df], how="horizontal")
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

        This method operates on a deep copy of ``self``; the original
        forecaster state is unchanged after the call.

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

    def _chains_are_batch_invariant(self) -> bool:
        """Whether every fitted transformer may be observed in bulk rather than per row.

        True only when both the target and actual chains report
        ``batch_invariant``. A transformer that does not declare the tag is treated as
        not invariant, so an unaudited stack keeps the rolling path: a missing
        declaration costs a speedup and never a result.

        Returns
        -------
        bool
            ``True`` when a bulk observe over a block reproduces, within floating point
            reassociation, what observing the block one row at a time would produce.

        See Also
        --------
        `yohou.testing.check_batch_invariance` : Verifies a transformer's declaration.

        """

        def declared(transformer: Any) -> bool:
            if transformer is None:
                return True
            tags = getattr(transformer, "__sklearn_tags__", None)
            if tags is None:
                return False
            transformer_tags = tags().transformer_tags
            return transformer_tags is not None and transformer_tags.batch_invariant

        def every(slot: Any) -> bool:
            if slot is None:
                return True
            if isinstance(slot, dict):
                return all(declared(t) for t in slot.values())
            return declared(slot)

        return every(getattr(self, "target_transformer_", None)) and every(getattr(self, "actual_transformer_", None))

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
        reduce_fn: Callable[[list[Any]], Any] | None = None,
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
        reduce_fn : callable or None, default=None
            How to combine the per-origin results of ``predict_fn``. ``None``
            concatenates them, which is the ordinary rolling forecast. A caller
            that wants to defer work out of the loop passes a ``predict_fn``
            that records state rather than predicting, and a ``reduce_fn`` that
            does the deferred work once over every origin.
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
            # The only site with no raw fallback: it derives from the argument
            # alone, so the transform applies unconditionally rather than through
            # a ternary. Shared by standard and panel, so it reaches the helper's
            # per-group dict branch under panel_strategy="global".
            step_columns_full = _derive_step_columns(
                X_future,
                self._transform_X_forecast(X_forecast),
                y["time"],
                self.fit_forecasting_horizon_,
                self.interval_,
            )

        # Initial predict (reads _X_t_observed set during fit/last observe).
        # Accumulated into a list rather than concatenated per iteration: a running
        # `pl.concat` reallocates the whole frame every origin, which is quadratic in
        # the origin count for a result that is only needed once.
        outputs: list[Any] = [predict_fn(groups=groups, **predict_kwargs)]

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

            outputs.append(predict_fn(groups=groups, **predict_kwargs))

        return pl.concat(outputs) if reduce_fn is None else reduce_fn(outputs)

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
        """Predicts ``fit_forecasting_horizon_`` steps from the observation horizon.

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
        raise NotImplementedError(f"The forecaster of type {type(self)} does not implement _predict_one.")

    def _predict(
        self,
        groups: list[str],
        y_pred_step: pl.DataFrame | None = None,
        **predict_one_params,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Generate one-step or multi-step prediction.

        Parameters
        ----------
        groups : list of str
            Group prefixes for panel data:
            - Pass an empty list to predict for non-panel (global) data.
            - If a list of str: predict only for the specified panel groups.
            Parameter is ignored if the forecaster was not fitted on panel data.
        y_pred_step : pl.DataFrame or None, default=None
            Predictions in transformed space, already carrying their time columns.
            When given, ``_predict_one`` is skipped and these are inverse-transformed
            and assembled instead. This lets a caller holding many origins lift the
            estimator work out of its loop and run it once, then reuse this method for
            the per-origin inverse rather than duplicating it.
        **predict_one_params : dict
            Params to the _predict_one method. Ignored when ``y_pred_step`` is given.

        Returns
        -------
        y_pred_step : pl.DataFrame
            Predicted time series in transformed space.
        y_pred_step_inv : pl.DataFrame
            Inverse transformed predicted time series (original scale).

        """
        if y_pred_step is None:
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
            target_transformers = typing_cast(dict[str, BaseActualTransformer | None], self.target_transformer_)
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
        - Router: Forwards metadata to target_transformer and actual_transformer

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

        # Route to actual_transformer if present
        if hasattr(self, "actual_transformer") and self.actual_transformer is not None:
            router.add(
                actual_transformer=self.actual_transformer,
                method_mapping=MethodMapping().add(caller="fit", callee="fit").add(caller="fit", callee="transform"),
            )

        # Route to forecast_transformer if present. Wider than the actual slots by
        # one mapping: predict -> transform. That is not decoration. The slot's
        # transform is genuinely called on the predict path, and
        # BaseForecastTransformer.transform accepts **params, so omitting the
        # mapping would make process_routing reject a request for a call that
        # demonstrably happens. The actual slots need no such mapping because they
        # reach their transformer through the observe/rewind memory API at predict
        # rather than through transform.
        #
        # No observe or rewind mapping: BaseForecastTransformer has neither method,
        # and that absence is what keeps the slot clear of the memory-API guard.
        if hasattr(self, "forecast_transformer") and self.forecast_transformer is not None:
            router.add(
                forecast_transformer=self.forecast_transformer,
                method_mapping=(
                    MethodMapping()
                    .add(caller="fit", callee="fit")
                    .add(caller="fit", callee="transform")
                    .add(caller="predict", callee="transform")
                ),
            )

        return router
