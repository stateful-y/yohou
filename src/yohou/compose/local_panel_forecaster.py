"""LocalPanelForecaster meta-forecaster for per-group independent forecasting."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Literal

import polars as pl
from pydantic import StrictInt
from sklearn.base import clone
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseForecaster
from yohou.utils import Tags
from yohou.utils._compat import _fit_context, _raise_for_params
from yohou.utils.panel import get_group_df, inspect_panel
from yohou.utils.validation import check_interval_consistency

__all__ = ["LocalPanelForecaster"]


def _forecaster_has(method_name: str):
    """Check if the wrapped forecaster has a given method.

    Parameters
    ----------
    method_name : str
        Name of the method to check for.

    Returns
    -------
    callable
        A check function for ``available_if`` decorator.

    """

    def check(self):
        """Check if wrapped forecaster has the required method."""
        forecaster = getattr(self, "forecasters_", {})
        if forecaster:
            return all(hasattr(f, method_name) for f in forecaster.values())
        return hasattr(self.forecaster, method_name)

    return check


def _fit_one_group(
    forecaster: BaseForecaster,
    group_name: str,
    y_group: pl.DataFrame,
    X_group: pl.DataFrame | None,
    forecasting_horizon: int,
    params: Any,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> tuple[str, BaseForecaster]:
    """Fit a cloned forecaster on a single panel group.

    Parameters
    ----------
    forecaster : BaseForecaster
        Forecaster to clone and fit.
    group_name : str
        Panel group name.
    y_group : pl.DataFrame
        Target time series for this group (unprefixed columns).
    X_group : pl.DataFrame or None
        Exogenous features for this group (unprefixed columns).
    forecasting_horizon : int
        Forecasting horizon.
    params : Bunch
        Routed parameters.
    X_future : pl.DataFrame or None, default=None
        Known future features with a ``"time"`` column.
    X_forecast : pl.DataFrame or None, default=None
        External forecasts with ``"vintage_time"`` and ``"time"``
        columns.

    Returns
    -------
    tuple[str, BaseForecaster]
        Group name and fitted forecaster.

    """
    forecaster_clone = clone(forecaster)
    forecaster_clone.fit(
        y_group,
        X_group,
        forecasting_horizon=forecasting_horizon,
        X_future=X_future,
        X_forecast=X_forecast,
        **params.fit,
    )
    return group_name, forecaster_clone


class LocalPanelForecaster(BaseForecaster):
    """Fits independent forecaster clones per panel group.

    Wraps any forecaster and fits separate ``clone()``-d instances for each
    panel group.  Each clone sees standard (non-panel) data with group prefixes
    stripped.  Predictions are reassembled back into prefixed-column format.

    Use ``LocalPanelForecaster`` when panel groups are heterogeneous and a
    single pooled model cannot capture group-specific dynamics.

    Parameters
    ----------
    forecaster : BaseForecaster
        Forecaster to clone per group.  Must support fitting on non-panel data.
    n_jobs : int or None, default=None
        Number of parallel jobs for fitting per-group clones.
        ``None`` means 1 unless in a ``joblib.parallel_backend`` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    forecasters_ : dict of str to BaseForecaster
        Mapping from group name to fitted forecaster clone.
    groups_ : list of str
        Names of panel groups discovered at fit time.
    local_y_schema_ : dict of str to DataType
        Schema of unprefixed target columns (shared across all groups).
    local_X_actual_schema_ : dict[str, polars.DataType] or None
        Schema of unprefixed exogenous columns, or ``None`` if X_actual was not
        provided.
    interval_ : timedelta
        Time interval between observations.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.compose import LocalPanelForecaster
    >>> from yohou.point import SeasonalNaive
    >>>
    >>> time = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 4, 9), interval="1d", eager=True
    ... )
    >>> y = pl.DataFrame({
    ...     "time": time,
    ...     "store_a__sales": range(100),
    ...     "store_b__sales": range(100, 200),
    ... })
    >>>
    >>> forecaster = LocalPanelForecaster(
    ...     forecaster=SeasonalNaive(seasonality=7),
    ... )
    >>> forecaster.fit(y, forecasting_horizon=5)  # doctest: +ELLIPSIS
    LocalPanelForecaster(...)
    >>> y_pred = forecaster.predict(forecasting_horizon=5)
    >>> sorted(c for c in y_pred.columns if c not in ("time", "vintage_time"))
    ['store_a__sales', 'store_b__sales']

    See Also
    --------
    - [`ColumnForecaster`][yohou.compose.column_forecaster.ColumnForecaster] : Apply different forecasters to different column subsets.

    Notes
    -----
    - Raises ``ValueError`` if the input data is not panel data (no ``__``
      separator detected).
    - Each group clone is completely independent (no parameter sharing).
    - ``groups`` argument on ``predict``, ``observe``, and
      ``rewind`` allows operating on a subset of groups.

    """

    _parameter_constraints: dict = {
        "forecaster": [BaseForecaster],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        forecaster: BaseForecaster,
        *,
        n_jobs: int | None = None,
    ):
        # LocalPanelForecaster does NOT call super().__init__() with panel_strategy
        # because it manages panel data entirely itself. BaseForecaster's
        # _pre_fit panel dispatch is not used.
        super().__init__()
        self.forecaster = forecaster
        self.n_jobs = n_jobs

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None

        # Inherit forecaster_type from wrapped forecaster
        child_tags = self.forecaster.__sklearn_tags__()
        if child_tags.forecaster_tags:
            tags.forecaster_tags.forecaster_type = child_tags.forecaster_tags.forecaster_type
            tags.forecaster_tags.stateful = child_tags.forecaster_tags.stateful
            tags.forecaster_tags.uses_reduction = child_tags.forecaster_tags.uses_reduction

        tags.forecaster_tags.supports_panel_data = True
        tags.forecaster_tags.tracks_observations = False

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
    ) -> LocalPanelForecaster:
        """Fit independent forecaster clones per panel group.

        Parameters
        ----------
        y : pl.DataFrame
            Panel target time series with ``"time"`` column and columns
            following the ``<group>__<series>`` naming convention.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with ``"time"`` column and columns
            following the ``<group>__<series>`` naming convention.
            Forwarded to each local forecaster.
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column. Deterministic
            values available for past and future dates. Bypasses the
            feature transformer.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns. Bypasses the feature transformer.
        **params : dict
            Metadata routing parameters forwarded to the wrapped forecaster.

        Returns
        -------
        self
            Fitted ``LocalPanelForecaster``.

        Raises
        ------
        ValueError
            If ``y`` is not panel data (no ``__`` separator found).

        """
        _raise_for_params(params, self, "fit")
        routed_params = process_routing(self, "fit", **params)

        # Discover panel structure
        global_cols, y_panel_groups = inspect_panel(y)
        if not y_panel_groups:
            raise ValueError(
                "LocalPanelForecaster requires panel data (columns with __ separator). "
                "Got only global columns. Use the wrapped forecaster directly for non-panel data."
            )

        groups_: list[str] = sorted(y_panel_groups.keys())
        self.groups_ = groups_  # ty: ignore[invalid-assignment]

        # Derive local schemas (unprefixed column names + dtypes) from the first
        # group, then verify every other group matches so a heterogeneous panel
        # surfaces an error instead of silently adopting the first group's schema.
        first_group = groups_[0]
        self.local_y_schema_ = {col.split("__", 1)[1]: y[col].dtype for col in y_panel_groups[first_group]}
        for group in groups_[1:]:
            group_schema = {col.split("__", 1)[1]: y[col].dtype for col in y_panel_groups[group]}
            if group_schema != self.local_y_schema_:
                raise ValueError(
                    f"LocalPanelForecaster requires all panel groups to share the same local "
                    f"target schema. Group '{group}' has {group_schema}, but group "
                    f"'{first_group}' has {self.local_y_schema_}."
                )

        # Handle X_actual panel structure
        if X_actual is not None:
            _, X_panel_groups = inspect_panel(X_actual)
            if X_panel_groups:
                first_X_group = sorted(X_panel_groups.keys())[0]
                self.local_X_actual_schema_ = {
                    col.split("__", 1)[1]: X_actual[col].dtype for col in X_panel_groups[first_X_group]
                }
            else:
                # Global X_actual shared across all groups
                self.local_X_actual_schema_ = {col: X_actual[col].dtype for col in X_actual.columns if col != "time"}
        else:
            self.local_X_actual_schema_ = None

        # Compute interval
        self.interval_ = check_interval_consistency(y)
        self.fit_forecasting_horizon_ = forecasting_horizon

        # Derive X_future / X_forecast local schemas (unprefixed names + dtypes)
        prefix = f"{first_group}__"
        if X_future is not None:
            local_future = {
                col.removeprefix(prefix): X_future[col].dtype for col in X_future.columns if col.startswith(prefix)
            }
            global_future = {col: X_future[col].dtype for col in X_future.columns if col != "time" and "__" not in col}
            self._local_X_future_schema_ = {**local_future, **global_future}
            self._X_future_schema_ = {col: X_future[col].dtype for col in X_future.columns if col != "time"}
        else:
            self._local_X_future_schema_ = None
            self._X_future_schema_ = None

        if X_forecast is not None:
            local_forecast = {
                col.removeprefix(prefix): X_forecast[col].dtype for col in X_forecast.columns if col.startswith(prefix)
            }
            global_forecast = {
                col: X_forecast[col].dtype
                for col in X_forecast.columns
                if col not in ("time", "vintage_time") and "__" not in col
            }
            self._local_X_forecast_schema_ = {**local_forecast, **global_forecast}
            self._X_forecast_schema_ = {
                col: X_forecast[col].dtype for col in X_forecast.columns if col not in ("time", "vintage_time")
            }
        else:
            self._local_X_forecast_schema_ = None
            self._X_forecast_schema_ = None

        # Extract per-group DataFrames and fit in parallel
        group_data = []
        for group_name in groups_:
            y_group = get_group_df(y, group_name, schema=self.local_y_schema_)
            X_group = (
                get_group_df(X_actual, group_name, schema=self.local_X_actual_schema_)
                if X_actual is not None and self.local_X_actual_schema_ is not None
                else None
            )
            X_future_group, X_forecast_group = self._split_exogenous_for_group(group_name, X_future, X_forecast)
            group_data.append((group_name, y_group, X_group, X_future_group, X_forecast_group))

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_group)(
                self.forecaster,
                group_name,
                y_group,
                X_group,
                forecasting_horizon,
                routed_params.forecaster,
                X_future=X_future_group,
                X_forecast=X_forecast_group,
            )
            for group_name, y_group, X_group, X_future_group, X_forecast_group in group_data
        )

        self.forecasters_ = dict(results)
        return self

    @available_if(_forecaster_has("predict"))
    def predict(
        self,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
        predict_transformed: bool = False,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Predict from each per-group forecaster and reassemble.

        Parameters
        ----------
        forecasting_horizon : int or None, default=None
            Number of steps ahead.  If ``None``, uses the value from ``fit``.
        groups : list of str or None, default=None
            Subset of groups to predict.  ``None`` predicts all groups.
        predict_transformed : bool, default=False
            If ``True``, return predictions in the transformed space.
        X_future : pl.DataFrame or None, default=None
            Known future features override. Re-derives step columns
            without mutating forecaster state.
        X_forecast : pl.DataFrame or None, default=None
            External forecast override with ``"vintage_time"`` and
            ``"time"`` columns. Re-derives step columns without mutating
            forecaster state.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Predictions with ``"vintage_time"`` (when present), ``"time"``,
            and prefixed panel columns.

        """
        check_is_fitted(self, ["forecasters_"])
        _raise_for_params(params, self, "predict")
        routed_params = process_routing(self, "predict", **params)

        groups: list[str] = groups if groups is not None else (self.groups_ or [])
        horizon = forecasting_horizon if forecasting_horizon is not None else self.fit_forecasting_horizon_

        return self._predict_groups(
            groups,
            horizon,
            routed_params,
            method="predict",
            X_future=X_future,
            X_forecast=X_forecast,
            predict_transformed=predict_transformed,
        )

    @available_if(_forecaster_has("predict_interval"))
    def predict_interval(
        self,
        forecasting_horizon: StrictInt | None = None,
        coverage_rates: list[float] | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Predict intervals from each per-group forecaster and reassemble.

        Parameters
        ----------
        forecasting_horizon : int or None, default=None
            Number of steps ahead.  If ``None``, uses the value from ``fit``.
        coverage_rates : list of float or None, default=None
            Coverage rates for prediction intervals.
        groups : list of str or None, default=None
            Subset of groups to predict.  ``None`` predicts all groups.
        X_future : pl.DataFrame or None, default=None
            Known future features override. Re-derives step columns
            without mutating forecaster state.
        X_forecast : pl.DataFrame or None, default=None
            External forecast override with ``"vintage_time"`` and
            ``"time"`` columns. Re-derives step columns without mutating
            forecaster state.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Interval predictions with ``"vintage_time"`` (when present),
            ``"time"``, and prefixed panel columns.

        """
        check_is_fitted(self, ["forecasters_"])
        _raise_for_params(params, self, "predict_interval")
        routed_params = process_routing(self, "predict_interval", **params)

        groups: list[str] = groups if groups is not None else (self.groups_ or [])
        horizon = forecasting_horizon if forecasting_horizon is not None else self.fit_forecasting_horizon_

        return self._predict_groups(
            groups,
            horizon,
            routed_params,
            method="predict_interval",
            coverage_rates=coverage_rates,
            X_future=X_future,
            X_forecast=X_forecast,
        )

    def observe(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> LocalPanelForecaster:
        """Observe new data per group without refitting.

        Parameters
        ----------
        y : pl.DataFrame
            New panel target observations.
        X_actual : pl.DataFrame or None, default=None
            New actual feature observations with panel columns.
            Forwarded to each local forecaster.
        groups : list of str or None, default=None
            Subset of groups to observe.  ``None`` observes all groups.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.

        Returns
        -------
        self

        """
        check_is_fitted(self, ["forecasters_"])

        groups: list[str] = groups if groups is not None else (self.groups_ or [])

        for group_name in groups:
            y_group = get_group_df(y, group_name, schema=self.local_y_schema_)
            X_group = (
                get_group_df(X_actual, group_name, schema=self.local_X_actual_schema_)
                if X_actual is not None and self.local_X_actual_schema_ is not None
                else None
            )
            X_future_group, X_forecast_group = self._split_exogenous_for_group(group_name, X_future, X_forecast)
            self.forecasters_[group_name].observe(
                y=y_group, X_actual=X_group, X_future=X_future_group, X_forecast=X_forecast_group
            )

        return self

    def rewind(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> LocalPanelForecaster:
        """Rewind each per-group forecaster's observation window.

        Parameters
        ----------
        y : pl.DataFrame
            Panel target data to rewind to.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations to restore the observation
            state to. Must align with ``y``.
        groups : list of str or None, default=None
            Subset of groups to rewind.  ``None`` rewinds all groups.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.

        Returns
        -------
        self

        """
        check_is_fitted(self, ["forecasters_"])

        groups: list[str] = groups if groups is not None else (self.groups_ or [])

        for group_name in groups:
            y_group = get_group_df(y, group_name, schema=self.local_y_schema_)
            X_group = (
                get_group_df(X_actual, group_name, schema=self.local_X_actual_schema_)
                if X_actual is not None and self.local_X_actual_schema_ is not None
                else None
            )
            X_future_group, X_forecast_group = self._split_exogenous_for_group(group_name, X_future, X_forecast)
            self.forecasters_[group_name].rewind(
                y=y_group, X_actual=X_group, X_future=X_future_group, X_forecast=X_forecast_group
            )

        return self

    @available_if(_forecaster_has("predict"))
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
        """Observe new data then predict for each group.

        Delegates to each clone's ``observe_predict`` so the rolling
        loop with ``stride`` is preserved per group.

        Parameters
        ----------
        y : pl.DataFrame
            New panel target observations.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Sliced and observed incrementally at each step
            of the rolling loop.
        forecasting_horizon : int or None, default=None
            Number of steps ahead. If ``None``, uses the value from
            ``fit``.
        groups : list of str or None, default=None
            Subset of groups.  ``None`` means all groups.
        stride : int or None, default=None
            Step size for rolling update and predict. If ``None``,
            defaults to ``fit_forecasting_horizon_``.
        predict_transformed : bool, default=False
            If ``True``, return predictions in the transformed space
            without applying inverse target transformation.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Predictions with prefixed panel columns.

        """
        check_is_fitted(self, ["forecasters_"])
        _raise_for_params(params, self, "observe_predict")
        routed_params = process_routing(self, "observe_predict", **params)

        groups_: list[str] = groups if groups is not None else (self.groups_ or [])

        group_predictions: dict[str, pl.DataFrame] = {}
        for group_name in groups_:
            y_group = get_group_df(y, group_name, schema=self.local_y_schema_)
            X_group = (
                get_group_df(X_actual, group_name, schema=self.local_X_actual_schema_)
                if X_actual is not None and self.local_X_actual_schema_ is not None
                else None
            )
            X_future_group, X_forecast_group = self._split_exogenous_for_group(group_name, X_future, X_forecast)
            group_predictions[group_name] = self.forecasters_[group_name].observe_predict(
                y=y_group,
                X_actual=X_group,
                forecasting_horizon=forecasting_horizon,
                stride=stride,
                predict_transformed=predict_transformed,
                X_future=X_future_group,
                X_forecast=X_forecast_group,
                **routed_params.forecaster.observe_predict,
            )

        return self._reassemble_panel_predictions(group_predictions)

    @available_if(_forecaster_has("predict_interval"))
    def observe_predict_interval(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        coverage_rates: list[float] | None = None,
        strategy: Literal["mean", "median", "point"] | None = None,
        groups: list[str] | None = None,
        stride: StrictInt | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Observe new data then predict intervals for each group.

        Delegates to each clone's ``observe_predict_interval`` so the
        rolling loop with ``stride`` is preserved per group.

        Parameters
        ----------
        y : pl.DataFrame
            New panel target observations.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Sliced and observed incrementally at each step
            of the rolling loop.
        forecasting_horizon : int or None, default=None
            Number of steps ahead. If ``None``, uses the value from
            ``fit``.
        coverage_rates : list of float or None, default=None
            Coverage rates for prediction intervals.
        strategy : {"mean", "median", "point"} or None, default=None
            Strategy for deriving point predictions from prediction
            intervals during recursive multi-step forecasting.
        groups : list of str or None, default=None
            Subset of groups.  ``None`` means all groups.
        stride : int or None, default=None
            Step size for rolling update and predict. If ``None``,
            defaults to ``fit_forecasting_horizon_``.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Interval predictions with prefixed panel columns.

        """
        check_is_fitted(self, ["forecasters_"])
        _raise_for_params(params, self, "observe_predict_interval")
        routed_params = process_routing(self, "observe_predict_interval", **params)

        groups_: list[str] = groups if groups is not None else (self.groups_ or [])

        group_predictions: dict[str, pl.DataFrame] = {}
        for group_name in groups_:
            y_group = get_group_df(y, group_name, schema=self.local_y_schema_)
            X_group = (
                get_group_df(X_actual, group_name, schema=self.local_X_actual_schema_)
                if X_actual is not None and self.local_X_actual_schema_ is not None
                else None
            )
            X_future_group, X_forecast_group = self._split_exogenous_for_group(group_name, X_future, X_forecast)
            group_predictions[group_name] = self.forecasters_[group_name].observe_predict_interval(
                y=y_group,
                X_actual=X_group,
                forecasting_horizon=forecasting_horizon,
                coverage_rates=coverage_rates,
                strategy=strategy,
                stride=stride,
                X_future=X_future_group,
                X_forecast=X_forecast_group,
                **routed_params.forecaster.observe_predict_interval,
            )

        return self._reassemble_panel_predictions(group_predictions)

    def _split_exogenous_for_group(
        self,
        group_name: str,
        X_future: pl.DataFrame | None,
        X_forecast: pl.DataFrame | None,
    ) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
        """Split X_future and X_forecast for a single panel group.

        Uses stored schemas from ``fit()`` to extract the group's local
        (prefixed) columns plus global (unprefixed) columns. When a
        schema is ``None`` but the DataFrame is provided (predict-time
        override after ``fit(X_future=None)``), derives the schema on
        the fly.

        Parameters
        ----------
        group_name : str
            Panel group name.
        X_future : pl.DataFrame or None
            Known future features (panel-level).
        X_forecast : pl.DataFrame or None
            External forecasts (panel-level).

        Returns
        -------
        tuple of (pl.DataFrame or None, pl.DataFrame or None)
            ``(X_future_group, X_forecast_group)`` with unprefixed
            columns for this group.

        """
        groups = self.groups_
        assert groups is not None, "fit() must be called before _split_exogenous_for_group()"

        X_future_group = None
        if X_future is not None:
            schema = self._local_X_future_schema_
            if schema is None:
                # On-the-fly schema derivation for predict-time overrides
                prefix = f"{groups[0]}__"
                local = {
                    col.removeprefix(prefix): X_future[col].dtype for col in X_future.columns if col.startswith(prefix)
                }
                global_ = {col: X_future[col].dtype for col in X_future.columns if col != "time" and "__" not in col}
                schema = {**local, **global_}
            if schema:
                X_future_group = get_group_df(X_future, group_name, schema=schema)

        X_forecast_group = None
        if X_forecast is not None:
            schema = self._local_X_forecast_schema_
            if schema is None:
                prefix = f"{groups[0]}__"
                local = {
                    col.removeprefix(prefix): X_forecast[col].dtype
                    for col in X_forecast.columns
                    if col.startswith(prefix)
                }
                global_ = {
                    col: X_forecast[col].dtype
                    for col in X_forecast.columns
                    if col not in ("time", "vintage_time") and "__" not in col
                }
                schema = {**local, **global_}
            if schema:
                X_forecast_group = get_group_df(
                    X_forecast, group_name, schema=schema, key_cols=("vintage_time", "time")
                )

        return X_future_group, X_forecast_group

    def _reassemble_panel_predictions(
        self,
        group_predictions: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """Reassemble per-group predictions into a panel DataFrame.

        Prefixes each group's value columns with ``<group_name>__`` and
        joins the per-group frames on their time keys (``"time"`` and,
        when present, ``"vintage_time"``). Joining on the time keys (rather
        than concatenating horizontally by position) guarantees that values
        are aligned by timestamp; a group with a misaligned time grid
        surfaces as nulls instead of silently corrupted associations.

        Parameters
        ----------
        group_predictions : dict of str to pl.DataFrame
            Mapping from group name to prediction DataFrame. Each
            DataFrame must share the same time keys.

        Returns
        -------
        pl.DataFrame
            Panel predictions with prefixed columns.

        """
        result: pl.DataFrame | None = None
        first_group_name: str | None = None
        expected_height = 0

        for group_name, df in group_predictions.items():
            time_cols = [c for c in ["time", "vintage_time"] if c in df.columns]
            non_time = [c for c in df.columns if c not in ("time", "vintage_time")]
            prefixed = df.rename({c: f"{group_name}__{c}" for c in non_time})

            if result is None:
                result = prefixed
                first_group_name = group_name
                expected_height = prefixed.height
            elif time_cols:
                if prefixed.height != expected_height:
                    raise ValueError(
                        f"Group '{group_name}' produced {prefixed.height} predictions, "
                        f"but the first group '{first_group_name}' produced {expected_height}. "
                        "Per-group prediction time grids must align to form a panel."
                    )
                result = result.join(prefixed, on=time_cols, how="full", coalesce=True)
                if result.height != expected_height:
                    raise ValueError(
                        f"Group '{group_name}' has a time grid that does not align with "
                        f"the first group '{first_group_name}'. Per-group prediction time grids "
                        "must match to form a panel."
                    )
            else:
                result = pl.concat([result, prefixed], how="horizontal")

        if result is None:
            return pl.DataFrame()

        sort_cols = [c for c in ["vintage_time", "time"] if c in result.columns]
        if sort_cols:
            result = result.sort(sort_cols)

        return result

    def get_metadata_routing(self) -> MetadataRouter:
        """Get metadata routing for this meta-estimator.

        Returns
        -------
        MetadataRouter
            Metadata routing.

        """
        router = MetadataRouter(owner=self.__class__.__name__)
        router.add(
            forecaster=self.forecaster,
            method_mapping=MethodMapping()
            .add(callee="fit", caller="fit")
            .add(callee="predict", caller="predict")
            .add(callee="predict_interval", caller="predict_interval")
            .add(callee="observe_predict", caller="observe_predict")
            .add(callee="observe_predict_interval", caller="observe_predict_interval"),
        )
        return router

    def _predict_groups(
        self,
        groups: list[str],
        horizon: int,
        routed_params: Any,
        method: str,
        coverage_rates: list[float] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        predict_transformed: bool = False,
    ) -> pl.DataFrame:
        """Predict (point or interval) per group and concatenate.

        Parameters
        ----------
        groups : list of str
            Panel group names to predict.
        horizon : int
            Forecasting horizon.
        routed_params : Bunch
            Routed metadata parameters.
        method : str
            ``"predict"`` or ``"predict_interval"``.
        coverage_rates : list of float or None
            Coverage rates (only for ``predict_interval``).
        X_future : pl.DataFrame or None, default=None
            Known future features override. Re-derives step columns
            without mutating forecaster state.
        X_forecast : pl.DataFrame or None, default=None
            External forecast override with ``"vintage_time"`` and
            ``"time"`` columns. Re-derives step columns without mutating
            forecaster state.
        predict_transformed : bool, default=False
            If ``True``, return predictions in the transformed space.

        Returns
        -------
        pl.DataFrame
            Reassembled panel predictions.

        """
        group_predictions: dict[str, pl.DataFrame] = {}

        for group_name in groups:
            forecaster = self.forecasters_[group_name]

            # Split exogenous overrides per group
            X_future_group, X_forecast_group = self._split_exogenous_for_group(group_name, X_future, X_forecast)

            # Call the appropriate predict method
            predict_kwargs: dict[str, Any] = {"forecasting_horizon": horizon}
            # predict_transformed is a parameter of point predict only; the
            # interval predict_interval signature does not accept it.
            if method == "predict":
                predict_kwargs["predict_transformed"] = predict_transformed
            predict_kwargs.update(routed_params.forecaster.get(method, {}))
            if method == "predict_interval" and coverage_rates is not None:
                predict_kwargs["coverage_rates"] = coverage_rates
            if X_future_group is not None:
                predict_kwargs["X_future"] = X_future_group
            if X_forecast_group is not None:
                predict_kwargs["X_forecast"] = X_forecast_group

            group_predictions[group_name] = getattr(forecaster, method)(**predict_kwargs)

        return self._reassemble_panel_predictions(group_predictions)
