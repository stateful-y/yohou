"""VotingForecaster for combining predictions from multiple forecasters."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import StrictInt
from sklearn.utils import Bunch
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseForecaster
from yohou.utils import Tags
from yohou.utils._compat import StrOptions, _BaseComposition, _fit_context, _raise_for_params

from ._base import _BaseEnsembleForecaster, _ensemble_has

__all__ = ["VotingForecaster"]


class VotingForecaster(BaseForecaster, _BaseEnsembleForecaster, _BaseComposition):
    """Combines predictions from multiple forecasters via averaging.

    Aggregates point predictions using mean or median, and optionally
    aggregates prediction intervals using a configurable strategy.
    All base forecasters must be of the same type (all point or all
    interval).

    If a base forecaster fails during ``fit``, it is silently skipped
    with a warning. The ensemble raises only when all base forecasters
    fail.

    Parameters
    ----------
    forecasters : list of (name, forecaster) tuples
        Named base forecasters to combine. Each entry is a
        ``(name, forecaster)`` tuple where *name* is a unique string
        identifier and *forecaster* is a `BaseForecaster` instance.
    method : {"mean", "median"}, default="mean"
        Aggregation method for point predictions. ``"mean"`` computes
        the (optionally weighted) arithmetic mean; ``"median"`` computes
        the unweighted median (``weights`` are ignored).
    weights : list of float or None, default=None
        Per-forecaster weights used when ``method="mean"``. Raw values
        are passed to ``numpy.average`` which normalizes internally.
        Silently ignored when ``method="median"``. Length must match the
        number of forecasters.
    interval_strategy : {"mean", "median", "envelope"}, default="envelope"
        How to aggregate prediction intervals from base interval
        forecasters:

        - ``"mean"``: average lower and upper bounds separately.
        - ``"median"``: take the median of lower and upper bounds.
        - ``"envelope"``: take the minimum of lower bounds and the
          maximum of upper bounds (widest coverage, most conservative).
    n_jobs : int or None, default=None
        Number of parallel jobs for fitting base forecasters.
        ``None`` means 1 unless in a ``joblib.parallel_backend`` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    forecasters_ : list of (str, BaseForecaster)
        Successfully fitted base forecasters as ``(name, forecaster)``
        pairs. Forecasters that failed during ``fit`` are excluded.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.ensemble import VotingForecaster
    >>> from yohou.point import SeasonalNaive
    >>>
    >>> time = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 4, 9), interval="1d", eager=True
    ... )
    >>> y = pl.DataFrame({"time": time, "value": range(len(time))})
    >>>
    >>> forecaster = VotingForecaster(
    ...     forecasters=[
    ...         ("naive_1", SeasonalNaive(seasonality=1)),
    ...         ("naive_7", SeasonalNaive(seasonality=7)),
    ...     ],
    ...     method="mean",
    ... )
    >>> forecaster.fit(y, forecasting_horizon=3)  # doctest: +ELLIPSIS
    VotingForecaster(...)
    >>> y_pred = forecaster.predict(forecasting_horizon=3)
    >>> len(y_pred)
    3

    See Also
    --------
    `VotingClassProbaForecaster` : Ensemble for class-probability forecasters.
    `ColumnForecaster` : Apply different forecasters to different column subsets.
    `LocalPanelForecaster` : Fit independent clones per panel group.

    Notes
    -----
    - All base forecasters must predict the same target columns. A
      ``ValueError`` is raised after fitting if schemas differ.
    - Weights are only used with ``method="mean"``; they are silently
      ignored with ``method="median"``.
    - The ensemble inherits ``forecaster_type`` from its children and
      exposes ``predict_interval`` only if all surviving base
      forecasters support it.

    """

    _parameter_constraints: dict = {
        "forecasters": [list],
        "method": [StrOptions({"mean", "median"})],
        "weights": [list, None],
        "interval_strategy": [StrOptions({"mean", "median", "envelope"})],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        forecasters: list[tuple[str, BaseForecaster]],
        *,
        method: Literal["mean", "median"] = "mean",
        weights: list[float] | None = None,
        interval_strategy: Literal["mean", "median", "envelope"] = "envelope",
        n_jobs: int | None = None,
    ):
        super().__init__()
        self.forecasters = forecasters
        self.method = method
        self.weights = weights
        self.interval_strategy = interval_strategy
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

        tags.forecaster_tags.tracks_observations = False
        tags.forecaster_tags.supports_panel_data = True

        forecasters_to_check = (
            [f for _, f in self.forecasters_] if hasattr(self, "forecasters_") else [f for _, f in self.forecasters]
        )

        if forecasters_to_check:
            all_types: frozenset[str] = frozenset()
            for f in forecasters_to_check:
                f_tags = f.__sklearn_tags__()
                if f_tags.forecaster_tags and f_tags.forecaster_tags.forecaster_type:
                    all_types = all_types | f_tags.forecaster_tags.forecaster_type

            if all_types:
                tags.forecaster_tags.forecaster_type = all_types

            tags.forecaster_tags.stateful = any(
                getattr(f.__sklearn_tags__().forecaster_tags, "stateful", False) for f in forecasters_to_check
            )

        return tags

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, returns parameters for contained sub-estimators.

        Returns
        -------
        dict
            Parameter names mapped to their values.

        """
        return self._get_params("_forecasters", deep=deep)

    def set_params(self, **params: Any) -> VotingForecaster:
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self

        """
        self._set_params("_forecasters", **params)
        return self

    def _validate_homogeneous_types(self) -> None:
        """Verify all base forecasters have the same forecaster_type.

        Raises
        ------
        ValueError
            If forecasters have mixed types.

        """
        types: set[frozenset[str]] = set()
        for _name, forecaster in self.forecasters:
            f_tags = forecaster.__sklearn_tags__()
            if f_tags.forecaster_tags and f_tags.forecaster_tags.forecaster_type:
                types.add(f_tags.forecaster_tags.forecaster_type)

        if len(types) > 1:
            raise ValueError(f"All base forecasters must have the same forecaster_type, got types: {types}")

    def _validate_schemas_match(self) -> None:
        """Verify all surviving fitted forecasters predict the same columns.

        Raises
        ------
        ValueError
            If target column schemas differ across forecasters.

        """
        schemas = {}
        for name, forecaster in self.forecasters_:
            schema = dict(forecaster.local_y_schema_)
            schemas[name] = schema

        reference_name, reference_schema = next(iter(schemas.items()))
        for name, schema in schemas.items():
            if schema != reference_schema:
                raise ValueError(
                    f"Forecaster '{name}' predicts columns {set(schema.keys())} "
                    f"but '{reference_name}' predicts {set(reference_schema.keys())}. "
                    f"All base forecasters must predict the same target columns."
                )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> VotingForecaster:
        """Fit all base forecasters on the same data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with ``"time"`` column.
        X : pl.DataFrame or None, default=None
            Exogenous features with ``"time"`` column.
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.
        **params : dict
            Metadata routing parameters forwarded to base forecasters.

        Returns
        -------
        self
            Fitted ensemble.

        Raises
        ------
        ValueError
            If ``weights`` length does not match the number of forecasters,
            or if forecaster types are not homogeneous, or if fitted
            forecasters have mismatched target column schemas.
        RuntimeError
            If all base forecasters fail during fitting.

        """
        _raise_for_params(params, self, "fit")
        routed_params = process_routing(self, "fit", **params)

        if forecasting_horizon < 1:
            raise ValueError(f"forecasting_horizon must be >= 1, got {forecasting_horizon}")

        self._validate_forecasters_list()
        self._validate_homogeneous_types()

        if self.weights is not None and len(self.weights) != len(self.forecasters):
            raise ValueError(
                f"Number of weights ({len(self.weights)}) must match number of forecasters ({len(self.forecasters)})"
            )

        self.forecasters_ = self._fit_forecasters_parallel(
            y=y,
            X=X,
            forecasting_horizon=forecasting_horizon,
            routed_params=routed_params,
            n_jobs=self.n_jobs,
        )

        self._validate_schemas_match()

        # Derive fitted attributes from the first surviving forecaster
        _first_name, first_forecaster = self.forecasters_[0]
        self.fit_forecasting_horizon_ = forecasting_horizon
        self.interval_ = first_forecaster.interval_
        self.panel_group_names_ = first_forecaster.panel_group_names_
        self.local_y_schema_ = dict(first_forecaster.local_y_schema_)
        self.local_X_schema_ = getattr(first_forecaster, "local_X_schema_", None)
        self.shared_X_schema_ = getattr(first_forecaster, "shared_X_schema_", None)
        self.local_y_t_schema_ = self.local_y_schema_
        self.local_X_t_schema_ = self.local_X_schema_
        self._y_observed = y
        self._X_observed = X
        self._X_t_observed = X

        # Compute effective weights for surviving forecasters
        if self.weights is not None:
            fitted_names = {name for name, _ in self.forecasters_}
            self.weights_ = [
                w for (name, _), w in zip(self.forecasters, self.weights, strict=True) if name in fitted_names
            ]
        else:
            self.weights_ = None

        return self

    def _aggregate_predictions(
        self,
        predictions: list[pl.DataFrame],
        target_cols: list[str],
    ) -> pl.DataFrame:
        """Aggregate target columns from multiple predictions.

        Parameters
        ----------
        predictions : list of pl.DataFrame
            Predictions from each base forecaster.
        target_cols : list of str
            Column names to aggregate.

        Returns
        -------
        pl.DataFrame
            Aggregated predictions with time columns.

        """
        time_cols = [c for c in ("observed_time", "time") if c in predictions[0].columns]
        time_df = predictions[0].select(time_cols)

        agg_exprs = []
        for col in target_cols:
            values = np.column_stack([pred[col].to_numpy() for pred in predictions])

            if self.method == "mean":
                if self.weights_ is not None:
                    aggregated = np.average(values, axis=1, weights=self.weights_)
                else:
                    aggregated = np.mean(values, axis=1)
            else:
                aggregated = np.median(values, axis=1)

            agg_exprs.append(pl.Series(name=col, values=aggregated))

        return time_df.with_columns(agg_exprs)

    @available_if(_ensemble_has("predict"))
    def predict(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        panel_group_names: list[str] | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Generate aggregated point predictions.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous features for the forecast period.
        forecasting_horizon : int or None, default=None
            Number of steps ahead. If ``None``, uses value from ``fit``.
        panel_group_names : list of str or None, default=None
            Panel group prefixes to predict.
        predict_transformed : bool, default=False
            If ``True``, return predictions in transformed space.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Aggregated predictions with ``"observed_time"``, ``"time"``,
            and target columns.

        """
        check_is_fitted(self, ["forecasters_"])
        _raise_for_params(params, self, "predict")
        routed_params = process_routing(self, "predict", **params)

        predictions = []
        for name, forecaster in self.forecasters_:
            forecaster_params = getattr(routed_params.get(name, Bunch(predict={})), "predict", {})
            y_pred = forecaster.predict(  # ty: ignore[unresolved-attribute]
                X=X,
                forecasting_horizon=forecasting_horizon,
                panel_group_names=panel_group_names,
                predict_transformed=predict_transformed,
                **forecaster_params,
            )
            predictions.append(y_pred)

        target_cols = [c for c in predictions[0].columns if c not in ("observed_time", "time")]
        return self._aggregate_predictions(predictions, target_cols)

    @available_if(_ensemble_has("predict_interval"))
    def predict_interval(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        coverage_rates: list[float] | None = None,
        panel_group_names: list[str] | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Generate aggregated interval predictions.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous features for the forecast period.
        forecasting_horizon : int or None, default=None
            Number of steps ahead. If ``None``, uses value from ``fit``.
        coverage_rates : list of float or None, default=None
            Coverage rates for prediction intervals.
        panel_group_names : list of str or None, default=None
            Panel group prefixes to predict.
        predict_transformed : bool, default=False
            If ``True``, return predictions in transformed space.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Aggregated interval predictions with ``"observed_time"``,
            ``"time"``, and lower/upper bound columns.

        """
        check_is_fitted(self, ["forecasters_"])
        _raise_for_params(params, self, "predict_interval")
        routed_params = process_routing(self, "predict_interval", **params)

        predictions = []
        for name, forecaster in self.forecasters_:
            forecaster_params = getattr(
                routed_params.get(name, Bunch(predict_interval={})),
                "predict_interval",
                {},
            )
            y_pred = forecaster.predict_interval(  # ty: ignore[unresolved-attribute]
                X=X,
                forecasting_horizon=forecasting_horizon,
                coverage_rates=coverage_rates,
                panel_group_names=panel_group_names,
                predict_transformed=predict_transformed,
                **forecaster_params,
            )
            predictions.append(y_pred)

        interval_cols = [c for c in predictions[0].columns if c not in ("observed_time", "time")]
        return self._aggregate_intervals(predictions, interval_cols)

    def _aggregate_intervals(
        self,
        predictions: list[pl.DataFrame],
        interval_cols: list[str],
    ) -> pl.DataFrame:
        """Aggregate interval columns using the configured strategy.

        Parameters
        ----------
        predictions : list of pl.DataFrame
            Interval predictions from each base forecaster.
        interval_cols : list of str
            Interval column names (lower/upper bounds).

        Returns
        -------
        pl.DataFrame
            Aggregated interval predictions.

        """
        time_cols = [c for c in ("observed_time", "time") if c in predictions[0].columns]
        time_df = predictions[0].select(time_cols)

        agg_exprs = []
        for col in interval_cols:
            values = np.column_stack([pred[col].to_numpy() for pred in predictions])

            if self.interval_strategy == "envelope":
                if "_lower_" in col:
                    aggregated = np.min(values, axis=1)
                elif "_upper_" in col:
                    aggregated = np.max(values, axis=1)
                else:
                    aggregated = np.mean(values, axis=1)
            elif self.interval_strategy == "median":
                aggregated = np.median(values, axis=1)
            elif self.weights_ is not None:
                aggregated = np.average(values, axis=1, weights=self.weights_)
            else:
                aggregated = np.mean(values, axis=1)

            agg_exprs.append(pl.Series(name=col, values=aggregated))

        return time_df.with_columns(agg_exprs)

    def observe(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> VotingForecaster:
        """Observe new data on all surviving base forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.
        X : pl.DataFrame or None, default=None
            New exogenous observations.
        panel_group_names : list of str or None, default=None
            Panel group prefixes to observe.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        self

        """
        check_is_fitted(self, ["forecasters_"])
        for _name, forecaster in self.forecasters_:
            forecaster.observe(y=y, X=X, panel_group_names=panel_group_names, **params)
        return self

    def rewind(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> VotingForecaster:
        """Rewind all surviving base forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            Target data to rewind to.
        X : pl.DataFrame or None, default=None
            Exogenous data to rewind to.
        panel_group_names : list of str or None, default=None
            Panel group prefixes to rewind.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        self

        """
        check_is_fitted(self, ["forecasters_"])
        for _name, forecaster in self.forecasters_:
            forecaster.rewind(y=y, X=X, panel_group_names=panel_group_names, **params)
        return self

    @available_if(_ensemble_has("predict"))
    def observe_predict(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Observe new data then predict.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.
        X : pl.DataFrame or None, default=None
            Exogenous features.
        forecasting_horizon : int or None, default=None
            Number of steps ahead.
        panel_group_names : list of str or None, default=None
            Panel group prefixes.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Aggregated predictions after observing new data.

        """
        self.observe(y=y, X=X, panel_group_names=panel_group_names, **params)
        return self.predict(
            X=X,
            forecasting_horizon=forecasting_horizon,
            panel_group_names=panel_group_names,
            **params,
        )

    @available_if(_ensemble_has("predict_interval"))
    def observe_predict_interval(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        coverage_rates: list[float] | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Observe new data then predict intervals.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.
        X : pl.DataFrame or None, default=None
            Exogenous features.
        forecasting_horizon : int or None, default=None
            Number of steps ahead.
        coverage_rates : list of float or None, default=None
            Coverage rates for prediction intervals.
        panel_group_names : list of str or None, default=None
            Panel group prefixes.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Aggregated interval predictions after observing new data.

        """
        self.observe(y=y, X=X, panel_group_names=panel_group_names, **params)
        return self.predict_interval(
            X=X,
            forecasting_horizon=forecasting_horizon,
            coverage_rates=coverage_rates,
            panel_group_names=panel_group_names,
            **params,
        )

    def get_metadata_routing(self) -> MetadataRouter:
        """Get metadata routing configuration.

        Returns
        -------
        MetadataRouter
            Router with mappings for all base forecasters.

        """
        router = MetadataRouter(owner=self.__class__.__name__)

        for name, forecaster in self.forecasters:
            router.add(
                **{name: forecaster},
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="predict", callee="predict")
                .add(caller="predict_interval", callee="predict_interval"),
            )

        return router
