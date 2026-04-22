"""Base class for forecasters."""

import abc
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
    `BasePointForecaster` : Base class for point forecasters.
    `BaseIntervalForecaster` : Base class for interval forecasters.
    `BaseReductionForecaster` : Forecasting via sklearn regressors.

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

        return tags

    @property
    def observation_horizon(self) -> int:
        """Get the number of time steps needed for stateful operations.

        The observation horizon defines how many recent observations the forecaster
        needs to maintain in its memory.

        Returns
        -------
        int
            Number of time steps to retain.

        Raises
        ------
        NotFittedError
            If the forecaster has not been fitted yet.

        """
        check_is_fitted(self, ["target_transformer_"])

        target_observation_horizon = 0
        if self.target_transformer is not None:
            if isinstance(self.target_transformer_, dict):
                # In panel data, all local transformers share the same horizon
                first_transformer = next(iter(self.target_transformer_.values()))
                if first_transformer is not None:
                    target_observation_horizon = typing_cast(BaseTransformer, first_transformer).observation_horizon
            elif isinstance(self.target_transformer_, BaseTransformer):
                target_observation_horizon = self.target_transformer_.observation_horizon

        # Compute feature transformer observation horizon
        feature_observation_horizon = 0
        if self.feature_transformer is not None:
            if isinstance(self.feature_transformer_, dict):
                first_transformer = next(iter(self.feature_transformer_.values()))
                if first_transformer is not None:
                    feature_observation_horizon = typing_cast(BaseTransformer, first_transformer).observation_horizon
            elif isinstance(self.feature_transformer_, BaseTransformer):
                feature_observation_horizon = self.feature_transformer_.observation_horizon

        self_observation_horizon = getattr(self, "_observation_horizon", 0)
        return max(self_observation_horizon, target_observation_horizon, feature_observation_horizon)

    def _set_input_attributes(self, y: pl.DataFrame, X: pl.DataFrame | None) -> None:
        """Detect and validate panel data structure across target and features.

        Inspects whether the data contains global (single time series) or local
        (panel columns with multiple time series) and ensures consistency
        across y and X. Sets instance attributes for downstream use.

            Sets the following attributes:
            - `groups_` : list of str or None
                Group prefixes for panel data (e.g., ["sales", "inventory"])
            - `local_y_schema_` : dict of str to pl.DataType
                Schema (column names -> dtypes) for target columns
            - `local_X_schema_` : dict of str to pl.DataType
                Schema (column names -> dtypes) for feature columns
            - `shared_X_schema_` : dict of str to pl.DataType or None
                Schema (column names -> dtypes) for shared feature columns found in X
                alongside local groups.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None
            Feature time series.

        Raises
        ------
        ValueError
            If y contains both global and local columns (ambiguous structure),
            or if group column suffixes don't match across groups,
            or if X local groups don't match y's structure.

        Notes
        -----
        Panel data example:
            y has columns ["sales__store_1", "sales__store_2"] (both Int64)
            -> groups_ = ["sales"]
            -> local_y_schema_ = {"store_1": pl.Int64, "store_2": pl.Int64}
            (Note: schema has unprefixed column names)

        Global data example:
            y has regular column "sales" (Int64)
            -> groups_ = None
            -> local_y_schema_ = {"sales": pl.Int64}

        See Also
        --------
        `inspect_panel` : Detects group columns.

        """
        y_global_names, y_panel_groups = inspect_panel(y)
        X_panel_groups = None
        if X is not None:
            _, X_panel_groups = inspect_panel(X)

            if len(X_panel_groups) and list(X_panel_groups.keys()) != list(y_panel_groups.keys()):
                raise ValueError("`X` and `y` do not have the same local group names.")

        # Non-panel data or multivariate strategy: dispatch to standard mixin
        if self.panel_strategy == "multivariate" or not y_panel_groups:
            BaseStandardForecaster._set_input_attributes_standard(self, y, X)
        # Panel data with global strategy: dispatch to panel mixin
        else:
            # Check for ambiguous structure (both standard and local columns)
            if len(y_global_names):
                raise ValueError("`y` contains both local and standard columns.")
            BasePanelForecaster._set_input_attributes_panel(self, y, X, y_panel_groups, X_panel_groups)

    def _validate_pre_fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
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
        X : pl.DataFrame or None, default=None
            Features time series.
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.

        Returns
        -------
        y : pl.DataFrame
            Validated target time series.
        X : pl.DataFrame or None
            Validated feature time series.
        y_panel_groups : dict[str, list[str]]
            Panel groups from y (empty dict if global data).
        X_panel_groups : dict[str, list[str]] or None
            Panel groups from X (None if X is None).

        """
        y, X, _ = validate_forecaster_data(self, y, X, reset=True)
        self.fit_forecasting_horizon_ = forecasting_horizon

        _, y_panel_groups = inspect_panel(y)
        X_panel_groups = None
        if X is not None:
            _, X_panel_groups = inspect_panel(X)

            if len(X_panel_groups) and list(X_panel_groups.keys()) != list(y_panel_groups.keys()):
                raise ValueError("`X` and `y` do not have the same local group names.")

        # Validate that X is provided when target_as_feature=None
        # and a feature transformer is configured.  Failing early here avoids
        # a confusing error at predict time inside _build_feature_input().
        if (
            getattr(self, "target_as_feature", None) is None
            and getattr(self, "feature_transformer", None) is not None
            and X is None
        ):
            raise ValueError(
                "target_as_feature=None with a feature_transformer requires X to be provided, but X is None."
            )

        # Validate that X is provided when target_as_feature=None and the
        # forecaster actually needs exogenous features.  Forecasters with
        # ignores_exogenous=True (e.g. SeasonalNaive, stationarity, decomposition)
        # work without any feature matrix.
        sklearn_tags = self.__sklearn_tags__()
        if (
            getattr(self, "target_as_feature", None) is None
            and X is None
            and sklearn_tags.forecaster_tags is not None
            and not sklearn_tags.forecaster_tags.ignores_exogenous
        ):
            raise ValueError(
                "target_as_feature=None requires X to be provided when the "
                "forecaster uses exogenous features (ignores_exogenous=False), "
                "but X is None."
            )

        return y, X, y_panel_groups, X_panel_groups

    def _pre_fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
    ) -> tuple[pl.DataFrame | dict[str, pl.DataFrame], pl.DataFrame | dict[str, pl.DataFrame] | None]:
        """Preprocess and transform inputs before fitting.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None, default=None
            Features time series.
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.

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
        y, X, y_panel_groups, X_panel_groups = self._validate_pre_fit(y, X, forecasting_horizon)

        # Dispatch to mixin methods based on panel strategy
        if self.panel_strategy == "multivariate" or not y_panel_groups:
            # Standard data or multivariate strategy (skip panel detection)
            return BaseStandardForecaster._pre_fit_standard(self, y, X, forecasting_horizon)
        else:
            # Panel data with global strategy
            return BasePanelForecaster._pre_fit_panel(self, y, X, forecasting_horizon, y_panel_groups, X_panel_groups)

    @abc.abstractmethod
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "BaseForecaster":
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
            If ``y`` is missing the ``"time"`` column, if ``y`` and ``X``
            have mismatched panel group names, or if
            ``target_as_feature=None`` without exogenous features when the
            forecaster requires them.

        """

    def rewind(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        groups: list[str] | None = None,
    ) -> "BaseForecaster":
        """Rewind observation buffers to the last ``observation_horizon`` rows.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.

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
            If ``y`` / ``X`` have invalid structure or ``groups``
            contains names not seen during fit.

        """
        check_is_fitted(
            self,
            ["local_y_schema_", "local_X_schema_", "shared_X_schema_", "groups_"],
        )

        # Validate schema, enforce column order, and validate groups (no continuity check - rewind sets new window)
        y, X, groups = validate_forecaster_data(self, y, X, reset=False, groups=groups)

        # Special handling for forecasters with no observation horizon
        if self.observation_horizon == 0:
            # If there is no observation horizon, only check for time column presence
            if "time" not in y.columns:
                raise ValueError("y must contain 'time' column.")
            if X is not None and "time" not in X.columns:
                raise ValueError("X must contain 'time' column.")

        # Dispatch to mixin methods
        if self.groups_ is None:
            BaseStandardForecaster._rewind_standard(self, y, X)
        else:
            BasePanelForecaster._rewind_panel(self, y, X, groups)

        return self

    def observe(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        groups: list[str] | None = None,
    ) -> "BaseForecaster":
        """Observe new data and update observation buffers without refitting.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.

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
            If ``y`` / ``X`` have invalid structure, non-contiguous time
            index, or ``groups`` contains names not seen during fit.

        """
        check_is_fitted(
            self,
            ["local_y_schema_", "local_X_schema_", "shared_X_schema_", "groups_"],
        )

        # Validate schema, enforce column order, and validate groups (includes continuity check)
        y, X, groups = validate_forecaster_data(self, y, X, reset=False, groups=groups)

        # Dispatch to mixin methods
        if self.groups_ is None:
            BaseStandardForecaster._observe_standard(self, y, X)
        else:
            BasePanelForecaster._observe_panel(self, y, X, groups)

        return self

    def _recursive_predict(
        self,
        *,
        X: pl.DataFrame | None,
        forecasting_horizon: int,
        groups: list[str] | None,
        step_fn: Callable[["BaseForecaster", list[str]], tuple[pl.DataFrame, pl.DataFrame]],
        derive_observation_fn: Callable[
            ["BaseForecaster", pl.DataFrame, pl.DataFrame | None, int],
            tuple[pl.DataFrame, pl.DataFrame | None],
        ],
    ) -> pl.DataFrame:
        """Shared recursive multi-step prediction loop.

        Produces predictions by repeatedly calling ``step_fn`` to get one
        forecast block, then ``derive_observation_fn`` to convert that
        prediction into an observation that is fed back via ``observe()``
        for the next recursive step.

        Parameters
        ----------
        X : pl.DataFrame or None
            Exogenous features, already filtered for panel groups by the
            caller if needed.
        forecasting_horizon : int
            Total number of time steps to forecast.
        groups : list of str or None
            Panel group prefixes to operate on.
        step_fn : callable
            ``step_fn(forecaster_copy, groups) -> (y_accumulate, y_for_obs)``
            where ``y_accumulate`` is appended to output and ``y_for_obs``
            is passed to ``derive_observation_fn``.
        derive_observation_fn : callable
            ``derive_observation_fn(forecaster_copy, y_for_obs, X, step_idx)
            -> (y_obs, X_slice)`` where ``y_obs`` and ``X_slice`` are passed
            to ``observe()``.

        Returns
        -------
        pl.DataFrame
            Concatenated predictions with ``"vintage_time"`` set to the
            first step's value and tail-trimmed to ``forecasting_horizon``.

        """
        forecaster = deepcopy(self)

        y_pred = pl.DataFrame()
        for step in range(0, forecasting_horizon, self.fit_forecasting_horizon_):
            y_accumulate, y_for_obs = step_fn(forecaster, groups or [])
            y_pred = pl.concat([y_pred, y_accumulate])

            if step + self.fit_forecasting_horizon_ < forecasting_horizon:
                y_obs, X_slice = derive_observation_fn(forecaster, y_for_obs, X, step)
                forecaster.observe(y_obs, X_slice)

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
        X: pl.DataFrame | None,
        groups: list[str] | None,
        stride: int,
        **predict_kwargs: Any,
    ) -> pl.DataFrame:
        """Shared observe-then-predict rolling loop.

        Produces an initial prediction, then repeatedly observes a
        ``stride``-sized slice of ``y`` and re-predicts. Used by
        ``observe_predict``, ``observe_predict_interval``, and
        ``observe_predict_class_proba``.

        Parameters
        ----------
        predict_fn : callable
            The predict method to call (e.g. ``self.predict``,
            ``self.predict_interval``, ``self.predict_class_proba``).
        y : pl.DataFrame
            Historical target observations to incrementally observe.
        X : pl.DataFrame or None
            Exogenous features aligned with ``y``.
        groups : list of str or None
            Panel group prefixes to operate on.
        stride : int
            Number of rows to observe between successive predictions.
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
        y_pred_i = predict_fn(X=X, groups=groups, **predict_kwargs)
        y_pred = y_pred_i

        for i in range(0, len(y), stride):
            y_slice = y[i : i + stride]

            X_slice = None
            if X is not None:
                X_slice = X.join(y_slice.select("time"), on="time", how="semi")

            self.observe(y=y_slice, X=X_slice, groups=groups)

            X_future = None
            if X is not None:
                last_time = y_slice["time"][-1]
                X_future = X.filter(pl.col("time") > last_time)

            y_pred_i = predict_fn(X=X_future, groups=groups, **predict_kwargs)
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
