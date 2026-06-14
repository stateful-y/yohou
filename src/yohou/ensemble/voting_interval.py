"""VotingIntervalForecaster for combining interval predictions from multiple forecasters."""

from __future__ import annotations

from numbers import Integral
from typing import Literal

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
from yohou.interval import BaseIntervalForecaster
from yohou.utils import INTERVAL, POINT_INTERVAL, Tags
from yohou.utils._compat import StrOptions, _BaseComposition, _fit_context, _raise_for_params

from ._base import _BaseEnsembleForecaster, _ensemble_has

__all__ = ["VotingIntervalForecaster"]


class VotingIntervalForecaster(_BaseEnsembleForecaster, BaseIntervalForecaster, _BaseComposition):
    """Combines interval predictions from multiple forecasters.

    Aggregates prediction intervals using mean, median, or envelope
    strategies. Optionally aggregates point predictions when all base
    forecasters support ``predict()``.

    If a base forecaster fails during ``fit``, it is silently skipped
    with a warning. The ensemble raises only when all base forecasters
    fail.

    Parameters
    ----------
    forecasters : list of (name, forecaster) tuples
        Named base forecasters to combine. Each entry is a
        ``(name, forecaster)`` tuple where *name* is a unique string
        identifier and *forecaster* is a `BaseForecaster` instance
        supporting ``predict_interval()``.
    method : {"mean", "median", "envelope"}, default="envelope"
        Aggregation strategy for interval predictions:

        - ``"mean"``: average lower and upper bounds separately.
        - ``"median"``: take the median of lower and upper bounds.
        - ``"envelope"``: take the minimum of lower bounds and the
          maximum of upper bounds (widest coverage, most conservative).
    point_method : {"mean", "median"}, default="mean"
        Aggregation method for point predictions used by ``predict()``
        when all base forecasters support it:

        - ``"mean"``: (optionally weighted) arithmetic mean.
        - ``"median"``: unweighted median.
    weights : list of float or None, default=None
        Per-forecaster weights used when ``method="mean"`` or
        ``point_method="mean"``. Raw values are passed to
        ``numpy.average`` which normalizes internally. Silently ignored
        with ``"median"`` or ``"envelope"``. Length must match the number
        of forecasters.
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
    >>> from yohou.ensemble import VotingIntervalForecaster
    >>> from yohou.interval import SplitConformalForecaster
    >>> from yohou.point import SeasonalNaive
    >>>
    >>> time = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 4, 9), interval="1d", eager=True
    ... )
    >>> y = pl.DataFrame({"time": time, "value": range(len(time))})
    >>>
    >>> forecaster = VotingIntervalForecaster(
    ...     forecasters=[
    ...         (
    ...             "conf_1",
    ...             SplitConformalForecaster(
    ...                 point_forecaster=SeasonalNaive(seasonality=1),
    ...                 calibration_size=10,
    ...             ),
    ...         ),
    ...         (
    ...             "conf_7",
    ...             SplitConformalForecaster(
    ...                 point_forecaster=SeasonalNaive(seasonality=7),
    ...                 calibration_size=10,
    ...             ),
    ...         ),
    ...     ],
    ...     method="envelope",
    ... )
    >>> forecaster.fit(y, forecasting_horizon=3)  # doctest: +ELLIPSIS
    VotingIntervalForecaster(...)
    >>> y_pred = forecaster.predict_interval(forecasting_horizon=3)
    >>> len(y_pred)
    3

    See Also
    --------
    - [`VotingPointForecaster`][yohou.ensemble.voting_point.VotingPointForecaster] : Ensemble for point forecasters.
    - [`VotingClassProbaForecaster`][yohou.ensemble.voting_class_proba.VotingClassProbaForecaster] : Ensemble for class-probability forecasters.
    - [`SplitConformalForecaster`][yohou.interval.split_conformal.SplitConformalForecaster] : Conformal prediction intervals.

    Notes
    -----
    - All base forecasters must predict the same target columns. A
      ``ValueError`` is raised after fitting if schemas differ.
    - Weights are only used with ``method="mean"`` or
      ``point_method="mean"``; they are silently ignored with
      ``"median"`` or ``"envelope"``.
    - Point predictions via ``predict()`` are only available when all
      base forecasters also support ``predict()``.

    """

    _parameter_constraints: dict = {
        "forecasters": [list],
        "method": [StrOptions({"mean", "median", "envelope"})],
        "point_method": [StrOptions({"mean", "median"})],
        "weights": [list, None],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        forecasters: list[tuple[str, BaseForecaster]],
        *,
        method: Literal["mean", "median", "envelope"] = "envelope",
        point_method: Literal["mean", "median"] = "mean",
        weights: list[float] | None = None,
        n_jobs: int | None = None,
    ):
        super().__init__()
        self.forecasters = forecasters
        self.method = method
        self.point_method = point_method
        self.weights = weights
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

        forecasters_to_check = (
            [f for _, f in self.forecasters_] if hasattr(self, "forecasters_") else [f for _, f in self.forecasters]
        )

        all_have_point = forecasters_to_check and all(
            f_tags.forecaster_tags is not None
            and f_tags.forecaster_tags.forecaster_type is not None
            and "point" in f_tags.forecaster_tags.forecaster_type
            for f in forecasters_to_check
            if (f_tags := f.__sklearn_tags__())
        )

        tags.forecaster_tags.forecaster_type = POINT_INTERVAL if all_have_point else INTERVAL
        tags.forecaster_tags.tracks_observations = False
        tags.forecaster_tags.supports_panel_data = True

        if forecasters_to_check:
            tags.forecaster_tags.stateful = any(
                getattr(f.__sklearn_tags__().forecaster_tags, "stateful", False) for f in forecasters_to_check
            )

        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        coverage_rates: list[float] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> VotingIntervalForecaster:
        """Fit all base forecasters on the same data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with ``"time"`` column.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Forwarded to each child forecaster.
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.
        coverage_rates : list of float or None, default=None
            Coverage rates for prediction intervals.
        X_future : pl.DataFrame or None, default=None
            Known future features with ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"`` columns.
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
            or if fitted forecasters have mismatched target column schemas.
        RuntimeError
            If all base forecasters fail during fitting.

        """
        _raise_for_params(params, self, "fit")
        routed_params = process_routing(self, "fit", **params)

        if forecasting_horizon < 1:
            raise ValueError(f"forecasting_horizon must be >= 1, got {forecasting_horizon}")

        self._validate_forecasters_list()

        if self.weights is not None and len(self.weights) != len(self.forecasters):
            raise ValueError(
                f"Number of weights ({len(self.weights)}) must match number of forecasters ({len(self.forecasters)})"
            )

        if coverage_rates is not None:
            for rate in coverage_rates:
                if rate < 0 or rate > 1:
                    raise ValueError(f"All coverage_rates must be in [0, 1], got {rate}")

        extra_fit_kwargs = {"coverage_rates": coverage_rates} if coverage_rates is not None else None

        self.forecasters_ = self._fit_forecasters_parallel(
            y=y,
            X_actual=X_actual,
            forecasting_horizon=forecasting_horizon,
            routed_params=routed_params,
            n_jobs=self.n_jobs,
            extra_fit_kwargs=extra_fit_kwargs,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        self._validate_schemas_match()
        self._derive_fitted_attributes(self.forecasters_[0][1], forecasting_horizon, y, X_actual)
        self._compute_effective_weights()

        # Copy coverage rates from first surviving child
        first_forecaster = self.forecasters_[0][1]
        self.fit_coverage_rates_ = getattr(first_forecaster, "fit_coverage_rates_", coverage_rates or [0.9])

        return self

    def _predict_one(
        self,
        groups: list[str],
        coverage_rates: list[float] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Raise; voting aggregates children's intervals via predict_interval.

        The base ``predict`` pipeline routes single-step prediction through
        ``_predict_one``. ``VotingIntervalForecaster`` bypasses that path by
        overriding ``predict_interval`` (and ``predict``) to aggregate the
        children's predictions directly, so this method is intentionally never
        reached and only exists to satisfy the abstract base contract.

        Parameters
        ----------
        groups : list of str
            Panel group prefixes.
        coverage_rates : list of float or None
            Coverage rates.
        **params : dict
            Additional parameters.

        Raises
        ------
        NotImplementedError
            Always raised.

        """
        raise NotImplementedError(
            "VotingIntervalForecaster aggregates children's predictions directly via predict_interval()"
        )

    def predict_interval(  # ty: ignore[invalid-method-override]
        self,
        forecasting_horizon: StrictInt | None = None,
        coverage_rates: list[float] | None = None,
        strategy: Literal["mean", "median", "point"] | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate aggregated interval predictions.

        Parameters
        ----------
        forecasting_horizon : int or None, default=None
            Number of steps ahead. If ``None``, uses value from ``fit``.
        coverage_rates : list of float or None, default=None
            Coverage rates for prediction intervals.
        strategy : {"mean", "median", "point"} or None, default=None
            Ignored for ensemble forecasters.
        groups : list of str or None, default=None
            Panel group prefixes to predict.
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
            Aggregated interval predictions with ``"vintage_time"``,
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
                forecasting_horizon=forecasting_horizon,
                coverage_rates=coverage_rates,
                groups=groups,
                X_future=X_future,
                X_forecast=X_forecast,
                **forecaster_params,
            )
            predictions.append(y_pred)

        interval_cols = [c for c in predictions[0].columns if c not in ("vintage_time", "time")]
        return self._aggregate_interval_values(predictions, interval_cols, self.method, self.weights_)

    @available_if(_ensemble_has("predict"))
    def predict(
        self,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
        predict_transformed: bool = False,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate aggregated point predictions.

        Only available when all base forecasters support ``predict()``.

        Parameters
        ----------
        forecasting_horizon : int or None, default=None
            Number of steps ahead. If ``None``, uses value from ``fit``.
        groups : list of str or None, default=None
            Panel group prefixes to predict.
        predict_transformed : bool, default=False
            If ``True``, return predictions in transformed space.
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
            Aggregated predictions with ``"vintage_time"``, ``"time"``,
            and target columns.

        """
        check_is_fitted(self, ["forecasters_"])
        _raise_for_params(params, self, "predict")
        routed_params = process_routing(self, "predict", **params)

        predictions = []
        for name, forecaster in self.forecasters_:
            forecaster_params = getattr(routed_params.get(name, Bunch(predict={})), "predict", {})
            y_pred = forecaster.predict(  # ty: ignore[unresolved-attribute]
                forecasting_horizon=forecasting_horizon,
                groups=groups,
                predict_transformed=predict_transformed,
                X_future=X_future,
                X_forecast=X_forecast,
                **forecaster_params,
            )
            predictions.append(y_pred)

        target_cols = [c for c in predictions[0].columns if c not in ("vintage_time", "time")]
        return self._aggregate_values(predictions, target_cols, self.point_method, self.weights_)

    @available_if(_ensemble_has("predict"))
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
        """Alternate recursive observe and predict on each child, then aggregate.

        Only available when all base forecasters support ``predict()``.
        Delegates the rolling observe-predict loop to each base forecaster
        and aggregates the resulting predictions.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Sliced and observed incrementally at each step
            of the rolling loop.
        forecasting_horizon : int or None, default=None
            Number of steps ahead.
        groups : list of str or None, default=None
            Panel group prefixes.
        stride : int or None, default=None
            Step size for rolling update-predict.
        predict_transformed : bool, default=False
            If ``True``, return predictions in transformed space.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.
        **params : dict
            Metadata routing parameters. Routed to each child through the
            ``predict`` routing namespace (there is no separate
            ``observe_predict`` mapping).

        Returns
        -------
        pl.DataFrame
            Aggregated point predictions after rolling observe-predict.

        """
        check_is_fitted(self, ["forecasters_"])
        _raise_for_params(params, self, "predict")
        routed_params = process_routing(self, "predict", **params)

        predictions = []
        for name, forecaster in self.forecasters_:
            forecaster_params = getattr(routed_params.get(name, Bunch(predict={})), "predict", {})
            y_pred = forecaster.observe_predict(  # ty: ignore[unresolved-attribute]
                y=y,
                X_actual=X_actual,
                forecasting_horizon=forecasting_horizon,
                groups=groups,
                stride=stride,
                predict_transformed=predict_transformed,
                X_future=X_future,
                X_forecast=X_forecast,
                **forecaster_params,
            )
            predictions.append(y_pred)

        target_cols = [c for c in predictions[0].columns if c not in ("vintage_time", "time")]
        return self._aggregate_values(predictions, target_cols, self.point_method, self.weights_)

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
        """Alternate recursive observe and predict_interval on each child, then aggregate.

        Delegates the rolling observe-predict loop to each base forecaster
        and aggregates the resulting interval predictions.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Sliced and observed incrementally at each step
            of the rolling loop.
        forecasting_horizon : int or None, default=None
            Number of steps ahead.
        coverage_rates : list of float or None, default=None
            Coverage rates for prediction intervals.
        strategy : {"mean", "median", "point"} or None, default=None
            Strategy for deriving point predictions during recursive
            multi-step forecasting.
        groups : list of str or None, default=None
            Panel group prefixes.
        stride : int or None, default=None
            Step size for rolling update-predict.
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
            Aggregated interval predictions after rolling observe-predict.

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
            y_pred = forecaster.observe_predict_interval(  # ty: ignore[unresolved-attribute]
                y=y,
                X_actual=X_actual,
                forecasting_horizon=forecasting_horizon,
                coverage_rates=coverage_rates,
                strategy=strategy,
                groups=groups,
                stride=stride,
                X_future=X_future,
                X_forecast=X_forecast,
                **forecaster_params,
            )
            predictions.append(y_pred)

        interval_cols = [c for c in predictions[0].columns if c not in ("vintage_time", "time")]
        return self._aggregate_interval_values(predictions, interval_cols, self.method, self.weights_)

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
