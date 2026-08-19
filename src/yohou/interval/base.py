"""Base classes for interval forecasters and similarity measures."""

import abc
from typing import Any, Literal

import numpy as np
import polars as pl
import polars.selectors as cs
from pydantic import StrictFloat, StrictInt
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseActualTransformer, BaseForecaster, BaseForecastTransformer, BaseStepTransformer
from yohou.utils import INTERVAL, Tags, cast, validate_forecaster_data
from yohou.utils._compat import StrOptions, _fit_context

__all__ = ["BaseConformalAdapter", "BaseIntervalForecaster", "BaseSimilarity"]


class BaseSimilarity(BaseEstimator, metaclass=abc.ABCMeta):
    """Base class for similarity measures used in interval forecasting.

    Similarity measures assign weights to calibration residuals based
    on how similar past prediction contexts are to the current one.

    Notes
    -----
    Used by ``SplitConformalForecaster`` to produce adaptive (locally
    weighted) prediction intervals.  When ``similarity=None``, uniform
    weights are used.

    See Also
    --------
    - [`DistanceSimilarity`][yohou.interval.similarity.DistanceSimilarity] : Distance-based similarity measure.
    - [`SplitConformalForecaster`][yohou.interval.split_conformal.SplitConformalForecaster] : Conformal forecaster that uses similarities.

    """

    _parameter_constraints: dict = {}

    @staticmethod
    def _fit_feature_scale(features: np.ndarray) -> np.ndarray:
        """Fit a per-column spread for standardizing a feature matrix.

        Without this, a distance over columns of mixed magnitude is decided
        almost entirely by the largest column, so every other column's
        neighbourhood is chosen for it.

        Parameters
        ----------
        features : numpy.ndarray
            Fit-time feature matrix of shape ``(n_rows, n_features)``.

        Returns
        -------
        numpy.ndarray
            One positive scale per column. A column with zero or non-finite
            spread (a constant feature) scales by ``1.0`` rather than
            dividing by zero.

        """
        scale = np.std(np.asarray(features, dtype=np.float64), axis=0)
        scale = np.atleast_1d(scale)
        scale[~np.isfinite(scale) | (scale <= 0.0)] = 1.0
        return scale

    @staticmethod
    def _fit_distance_scale(features: np.ndarray, metric: str, metric_params: dict[str, object] | None) -> float:
        """Fit the distance scale as the median pairwise fit-time distance.

        The softmax below has no width of its own, so without dividing by a
        fitted scale the weight concentration is a function of the units of
        the data: the same series at 100x the magnitude collapses onto its
        single nearest calibration row.

        Parameters
        ----------
        features : numpy.ndarray
            Fit-time feature matrix, already standardized where the caller
            standardizes.
        metric : str
            Distance metric, matching the one used at predict time.
        metric_params : dict or None
            Extra keyword arguments forwarded to the metric.

        Returns
        -------
        float
            The median pairwise distance, or ``1.0`` when it is zero, not
            finite, or undefined for fewer than two rows.

        """
        features = np.asarray(features, dtype=np.float64)
        if features.shape[0] < 2:
            return 1.0

        pairwise = pdist(features, metric=metric, **(metric_params or {}))  # ty: ignore[no-matching-overload]
        if pairwise.size == 0:
            return 1.0

        median = float(np.median(pairwise))
        return median if np.isfinite(median) and median > 0.0 else 1.0

    @staticmethod
    def _to_weights(
        distances: np.ndarray,
        distance_scale: float = 1.0,
        bandwidth: float = 1.0,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating[Any]]]:
        r"""Convert a distance matrix to calibration weights.

        Divides distances by the fitted scale and the bandwidth, then applies
        a numerically-stable softmax of negative scaled distances, reserving
        uniform mass for the (hypothetical) test point over the calibration
        axis. Writing $\tilde{d} = d / (\text{bandwidth} \cdot
        \text{distance\_scale})$:

        $$w_{ji} = \frac{\exp(-(\tilde{d}_{ji} - \min_k \tilde{d}_{jk}))}
        {1 + \sum_k \exp(-(\tilde{d}_{jk} - \min_k \tilde{d}_{jk}))}$$

        Dividing by the fitted scale is what makes the weights invariant to a
        rescaling of the data. Each output row is non-negative and sums to a
        value strictly less than 1; the remainder ``1 / (1 + \sum_k raw)`` is
        the mass reserved for the new test point, following the
        non-exchangeable conformal construction (Barber et al., 2023).

        Parameters
        ----------
        distances : numpy.ndarray
            Distance matrix of shape ``(n_pred, n_calibration)``.
        distance_scale : float, default=1.0
            Fitted distance scale from :meth:`_fit_distance_scale`.
        bandwidth : float, default=1.0
            Multiplier on the scale. Below 1 concentrates weight on nearer
            rows, above 1 flattens toward uniform.

        Returns
        -------
        numpy.ndarray
            Weight matrix of shape ``(n_pred, n_calibration)``.

        """
        scaled = distances / (bandwidth * distance_scale)
        # Stable softmax of the exponent -d: subtract its row max (= -min(d)),
        # so every exponent is <= 0 and exp() cannot overflow.
        neg_d = -scaled
        neg_d = neg_d - neg_d.max(axis=1, keepdims=True)
        return BaseSimilarity._reserve_mass(np.exp(neg_d))

    @staticmethod
    def _reserve_mass(
        raw_weights: np.ndarray,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating[Any]]]:
        r"""Normalize non-negative weights, reserving uniform mass per row.

        Returns ``raw / (\sum_k raw + 1)`` so each row is non-negative and
        sums to a value strictly less than 1; the remainder is reserved for
        the test point. Shared by the distance softmax (:meth:`_to_weights`)
        and the
        [`CompositeSimilarity`][yohou.interval.similarity.CompositeSimilarity]
        ``"multiply"`` combination.

        Parameters
        ----------
        raw_weights : numpy.ndarray
            Non-negative weight matrix of shape ``(n_pred, n_calibration)``.

        Returns
        -------
        numpy.ndarray
            Row-normalized weight matrix of the same shape.

        """
        return raw_weights / (raw_weights.sum(axis=1, keepdims=True) + 1.0)

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with similarity-specific attributes.

        """
        tags = Tags(estimator_type="similarity", requires_fit=True)

        # Most similarity measures are symmetric and require predictions
        assert tags.similarity_tags is not None
        tags.similarity_tags.symmetric = True
        tags.similarity_tags.requires_predictions = True
        tags.similarity_tags.produces_weights = True

        return tags

    @abc.abstractmethod
    def fit(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> "BaseSimilarity":
        """Fit the similarity measure.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        y_pred : pl.DataFrame
            Point predictions.

        X_actual : pl.DataFrame or None, default=None
            Exogenous features.

        Returns
        -------
        self

        """

    @abc.abstractmethod
    def observe(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> "BaseSimilarity":
        """Observe new data and update the similarity measure.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.

        y_pred : pl.DataFrame
            New predictions.

        X_actual : pl.DataFrame or None, default=None
            New exogenous features.

        Returns
        -------
        self

        """

    @abc.abstractmethod
    def predict(
        self,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating[Any]]]:
        """Compute similarity weights for predictions.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Predictions to compute similarities for.

        X_actual : pl.DataFrame or None, default=None
            Exogenous features.

        Returns
        -------
        np.ndarray
            Similarity weights.

        """

    def rewind(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
    ) -> "BaseSimilarity":
        """Rewind observed data from the similarity measure.

        Default implementation is a no-op. Concrete subclasses that
        track observed data should override this to remove the most
        recently observed rows.

        Parameters
        ----------
        y : pl.DataFrame
            Target observations to rewind.

        y_pred : pl.DataFrame
            Predictions to rewind.

        X_actual : pl.DataFrame or None, default=None
            Exogenous features to rewind.

        Returns
        -------
        self

        """
        return self


class BaseConformalAdapter(BaseEstimator, metaclass=abc.ABCMeta):
    """Base class for adaptive conformal inference adapters.

    A conformal adapter maintains a time-varying effective miscoverage
    level and updates it online from the coverage realized on newly
    observed data. It is an optional, pluggable add-on for
    ``SplitConformalForecaster``: when ``adapter=None`` the forecaster uses
    the static calibrated level, and when an adapter is supplied the
    forecaster feeds it per-row miscoverage indicators and reads back the
    effective level to use when constructing intervals.

    The adapter owns a single pooling slot. The forecaster clones one
    adapter per slot into an ``adapters_`` dict (mirroring ``similarities_``)
    and drives its lifecycle: one clone per ``(horizon step, value column)``
    under ``alpha_pooling="per_step"``, and one per value column shared
    across the step keys under ``"shared"``. Inside its slot the adapter
    tracks one effective level per coverage rate for symmetric conformity
    scorers, and two (lower and upper) for asymmetric ones. The forecaster,
    which holds the calibration scores and any similarity weights, computes
    the miscoverage indicators; the adapter is the level-recursion state
    machine only.

    Parameters
    ----------
    alpha_pooling : {"per_step", "shared"}, default="per_step"
        Which axis the effective level lives on. ``"per_step"`` tracks an
        independent level per horizon step, respecting horizon-dependent
        coverage. ``"shared"`` pools miscoverage across steps into one
        trajectory per value column, which stays closer to the single-sequence
        setting the underlying theory covers. Pooling never crosses value
        columns under either value.

        Declared here so every adapter in the family carries it and the
        enclosing forecaster can read it as a contract rather than probing
        for it. A subclass MUST still accept it in its own constructor and
        forward it here: estimator parameter discovery reads the most derived
        constructor only, so omitting it would drop the setting from
        ``get_params``, make ``adapter__alpha_pooling`` unaddressable in a
        search, and let ``clone`` silently reset a configured ``"shared"``.

    Notes
    -----
    The lifecycle mirrors the rest of the library: ``fit`` seeds the level
    from the target coverage, ``observe`` advances it per newly observed
    row, ``predict`` returns the current level, and ``rewind`` rolls it
    back so backtests and production replay share one code path.

    See Also
    --------
    - [`AdaptiveConformalInference`][yohou.interval.adapter.AdaptiveConformalInference] :
        Concrete Gibbs-Candes online level adjustment.
    - [`SplitConformalForecaster`][yohou.interval.split_conformal.SplitConformalForecaster] :
        Conformal forecaster that consumes an adapter.

    """

    _parameter_constraints: dict = {
        "alpha_pooling": [StrOptions({"per_step", "shared"})],
    }

    def __init__(self, alpha_pooling: Literal["per_step", "shared"] = "per_step") -> None:
        self.alpha_pooling = alpha_pooling

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with conformal-adapter-specific attributes.

        """
        tags = Tags(estimator_type="conformal_adapter", requires_fit=True)

        assert tags.conformal_adapter_tags is not None
        tags.conformal_adapter_tags.online = True
        tags.conformal_adapter_tags.requires_coverage_rates = True
        tags.conformal_adapter_tags.tail_aware = True

        return tags

    @abc.abstractmethod
    def fit(self, coverage_rates: list[float], *, symmetric: bool) -> "BaseConformalAdapter":
        """Seed the effective level(s) from the target coverage rates.

        Parameters
        ----------
        coverage_rates : list of float
            The nominal coverage rates to track, one effective level seeded
            per rate at ``1 - coverage_rate``.
        symmetric : bool
            Whether the conformity scorer is symmetric. Symmetric scorers
            use one level per rate; asymmetric scorers use two (lower and
            upper), each targeting half the miscoverage.

        Returns
        -------
        self

        """

    @abc.abstractmethod
    def observe(self, errors: list[dict[float, Any]]) -> "BaseConformalAdapter":
        """Advance the effective level(s) from per-row miscoverage.

        Parameters
        ----------
        errors : list of dict
            One entry per newly observed row. Each entry maps a tracked
            coverage rate to its miscoverage signal: a float in ``[0, 1]``
            for symmetric scorers, or a ``(lower, upper)`` tuple of such
            floats for asymmetric scorers.

        Returns
        -------
        self

        """

    @abc.abstractmethod
    def predict(self) -> dict[float, Any]:
        """Return the current effective level(s).

        Returns
        -------
        dict
            Maps each tracked coverage rate to its current effective level:
            a float for symmetric scorers, or a ``(lower, upper)`` tuple for
            asymmetric scorers.

        """

    @abc.abstractmethod
    def rewind(self, n_rows: int) -> "BaseConformalAdapter":
        """Roll the effective level(s) back by ``n_rows`` observations.

        Parameters
        ----------
        n_rows : int
            Number of most-recently observed rows to undo, never dropping
            below the fit-time seed.

        Returns
        -------
        self

        """


class BaseIntervalForecaster(BaseForecaster, metaclass=abc.ABCMeta):
    """Base class for interval forecasters.

    Parameters
    ----------
    actual_transformer : instance of `BaseActualTransformer` or None, default=None
        Transformer used to transform the feature time series into features.
    forecast_transformer : instance of `BaseForecastTransformer` or None, default=None
        Transformer applied to ``X_forecast`` before step columns are derived,
        so the step columns reaching the estimator are built from transformed
        values. Must be forecast-kind (vintage-indexed); an actual-kind
        transformer is rejected. ``None`` leaves ``X_forecast`` untouched.
    step_transformer : BaseStepTransformer or None, default=None
        Transformer applied to the derived ``{base}_step_1..H`` frame after
        step columns are built from ``X_future``/``X_forecast`` and before they
        join the design matrix. Reduces or rescales along the horizon axis.
        ``None`` leaves the step columns as derived.
    target_as_feature : {"transformed", "raw"} or None, default="transformed"
        Controls whether the target is included as a feature.
        ``"transformed"`` includes the transformed target, ``"raw"``
        includes the raw target, and ``None`` uses only exogenous features.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See `BaseForecaster` for details.

    Attributes
    ----------
    fit_coverage_rates_ : list of float
        Coverage rates used during fit.

    Notes
    -----
    Interval forecasters produce prediction intervals at specified
    coverage rates.  The ``forecaster_type`` tag is ``INTERVAL``
    (or ``POINT_INTERVAL`` if point predictions are also available).

    Unlike point forecasters, interval forecasters do not expose a
    ``target_transformer`` parameter; ``__init__`` fixes it to ``None``.
    Interval bounds must remain in the original target scale to stay
    interpretable, so a target transformation is not applied.

    See Also
    --------
    - [`SplitConformalForecaster`][yohou.interval.split_conformal.SplitConformalForecaster] : Conformal interval forecaster.
    - [`IntervalReductionForecaster`][yohou.interval.reduction.IntervalReductionForecaster] : ML-based interval forecaster.
    - [`BasePointForecaster`][yohou.point.base.BasePointForecaster] : Base class for point forecasters.

    """

    _tags: dict = {"forecaster_type": INTERVAL}

    _parameter_constraints: dict = {}

    def __init__(
        self,
        *,
        actual_transformer: BaseActualTransformer | None = None,
        forecast_transformer: BaseForecastTransformer | None = None,
        step_transformer: BaseStepTransformer | None = None,
        target_as_feature: Literal["transformed", "raw"] | None = "transformed",
        panel_strategy: Literal["global", "multivariate"] = "global",
    ) -> None:
        super().__init__(
            actual_transformer=actual_transformer,
            forecast_transformer=forecast_transformer,
            step_transformer=step_transformer,
            target_transformer=None,
            target_as_feature=target_as_feature,
            panel_strategy=panel_strategy,
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        *,
        coverage_rates: list[float] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> "BaseIntervalForecaster":
        """Fit the forecaster to historical data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``.  If ``None``, no exogenous features are used.
        forecasting_horizon : int, default=1
            Number of time steps to forecast into the future.
        coverage_rates : list of float or None, default=None
            Coverage levels for prediction intervals (e.g., ``[0.9, 0.95]``
            for 90 % and 95 % intervals).  If ``None``, defaults to
            ``[0.95]``.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column. Deterministic
            values available for past and future dates.
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
            If ``forecasting_horizon`` < 1, ``coverage_rates`` not in [0, 1],
            or if ``y`` / ``X_actual`` have invalid structure.

        """
        forecasting_horizon, coverage_rates = self._validate_interval_fit_params(forecasting_horizon, coverage_rates)
        self.fit_coverage_rates_ = coverage_rates

        y_t, X_t = self._pre_fit(
            y=y,
            X_actual=X_actual,
            forecasting_horizon=forecasting_horizon,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        self._fit(y_t, X_t, forecasting_horizon)

        return self

    def _validate_interval_fit_params(
        self,
        forecasting_horizon: StrictInt,
        coverage_rates: list[StrictFloat] | None = None,
    ) -> tuple[StrictInt, list[StrictFloat]]:
        """Validate fit parameters.

        Parameters
        ----------
        forecasting_horizon : int
            Forecasting horizon to validate.
        coverage_rates : list of float or None
            Coverage rates to validate. If None, uses [0.95].

        Returns
        -------
        tuple of (int, list of float)
            Validated forecasting horizon and coverage rates.

        Raises
        ------
        ValueError
            If forecasting_horizon < 1 or coverage_rates not in [0, 1].

        """
        if forecasting_horizon < 1:
            raise ValueError(f"forecasting_horizon must be >= 1, got {forecasting_horizon}")

        if coverage_rates is None:
            coverage_rates = [0.95]

        # Validate coverage rates
        for rate in coverage_rates:
            if not (0 <= rate <= 1):
                raise ValueError(f"All coverage_rates must be in [0, 1], got {rate}")

        return forecasting_horizon, coverage_rates

    def _validate_predict_params(
        self,
        forecasting_horizon: StrictInt | None,
        coverage_rates: list[StrictFloat] | None = None,
    ) -> tuple[StrictInt, list[StrictFloat]]:
        """Validate and return predict parameters.

        Parameters
        ----------
        forecasting_horizon : int or None
            Forecasting horizon to validate. If None, uses fit_forecasting_horizon_.
        coverage_rates : list of float or None
            Coverage rates to validate. If None, uses fit_coverage_rates_.

        Returns
        -------
        tuple of (int, list of float)
            Validated forecasting horizon and coverage rates.

        Raises
        ------
        ValueError
            If forecasting_horizon < 1 or coverage_rates not in [0, 1].

        """
        if forecasting_horizon is None:
            forecasting_horizon = self.fit_forecasting_horizon_
        if coverage_rates is None:
            # fit_coverage_rates_ is set by concrete subclasses during fit().
            coverage_rates = self.fit_coverage_rates_
        return self._validate_interval_fit_params(forecasting_horizon, coverage_rates)

    def predict_interval(
        self,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        coverage_rates: list[float] | None = None,
        strategy: Literal["mean", "median", "point"] | None = None,
        groups: list[str] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate interval forecasts.

        Parameters
        ----------
        X_future : pl.DataFrame or None, default=None
            Known future features override. Re-derives step columns
            without mutating forecaster state.
        X_forecast : pl.DataFrame or None, default=None
            External forecast override with ``"vintage_time"`` and
            ``"time"`` columns. Re-derives step columns using as-of
            matching without mutating forecaster state.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        coverage_rates : list of float or None, default=None
            Coverage levels for prediction intervals (e.g., ``[0.9, 0.95]``
            for 90 % and 95 % intervals).  If ``None``, defaults to the rates
            used at fit time.
        strategy : {"mean", "median", "point"} or None, default=None
            Strategy for deriving point predictions from prediction intervals
            during recursive multi-step forecasting:

            - ``"mean"``: use the mean of the interval bounds
            - ``"median"``: use the median of the interval bounds
            - ``"point"``: use the point forecast directly (if available)

            If ``None``, defaults to ``"mean"``.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Interval predictions with ``"vintage_time"``, ``"time"``, and
            lower/upper bound columns for each target at each coverage rate.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forecaster has not been fitted yet.
        ValueError
            If ``coverage_rates`` not in [0, 1],
            or ``groups`` contains names not seen during fit.

        """
        check_is_fitted(self, ["groups_", "local_y_schema_", "fit_forecasting_horizon_"])
        _, _, groups = validate_forecaster_data(
            self,
            y=None,
            X_actual=None,
            reset=False,
            groups=groups,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        forecasting_horizon, coverage_rates = self._validate_predict_params(forecasting_horizon, coverage_rates)

        y_columns = list(self.local_y_schema_.keys())
        if groups is not None:
            y_columns = [f"{panel_group}__{col}" for panel_group in groups for col in self.local_y_schema_]

        def step_fn(forecaster, groups):
            """Produce one interval-prediction block."""
            y_pred_step, y_pred_step_inv = forecaster._predict(groups, coverage_rates=coverage_rates)
            return y_pred_step_inv, y_pred_step_inv

        def derive_observation_fn(forecaster, y_pred_step_inv):
            """Derive observation from interval bounds."""
            time = y_pred_step_inv.select(cs.by_name("time"))

            y_data: dict[str, Any] = {"time": time["time"]}
            for col in y_columns:
                lower_cols = [c for c in y_pred_step_inv.columns if c.startswith(f"{col}_lower_")]
                upper_cols = [c for c in y_pred_step_inv.columns if c.startswith(f"{col}_upper_")]

                all_bound_cols = lower_cols + upper_cols

                if strategy == "point":
                    if col not in y_pred_step_inv.columns:
                        raise ValueError(
                            f"strategy='point' requires a bare point column '{col}' in the "
                            f"interval predictions, but {type(forecaster).__name__} did not "
                            f"emit one. Use strategy='mean' or strategy='median' instead."
                        )
                    y_data[col] = y_pred_step_inv[col]
                elif strategy == "median":
                    y_data[col] = y_pred_step_inv.select(
                        pl.median_horizontal(all_bound_cols)  # ty: ignore[unresolved-attribute]
                    ).to_series()
                else:
                    y_data[col] = y_pred_step_inv.select(pl.mean_horizontal(all_bound_cols)).to_series()

            y = pl.DataFrame(y_data)

            if groups is not None:
                cast_schema = {}
                for group_name in groups:
                    for col_name, dtype in forecaster.local_y_schema_.items():
                        cast_schema[f"{group_name}__{col_name}"] = dtype
            else:
                cast_schema = forecaster.local_y_schema_

            y = cast(y.select(~cs.by_name("time")), cast_schema)
            y = pl.concat([y_data["time"].to_frame(), y], how="horizontal")
            return y

        def predict_fn():
            """Run recursive predict with step columns."""
            return self._recursive_predict(
                forecasting_horizon=forecasting_horizon,
                groups=groups,
                step_fn=step_fn,
                derive_observation_fn=derive_observation_fn,
            )

        return self._predict_with_step_override(
            X_future=X_future,
            X_forecast=X_forecast,
            predict_fn=predict_fn,
        )

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
        """Alternate recursive predict_interval and observe.

        Runs a rolling observe-predict loop over ``y``, emitting an
        initial prediction and then one interval prediction after each
        ``stride``-row observation block. Returns the concatenated
        predictions across all vintages.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Sliced and observed incrementally at each step
            of the rolling loop.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        coverage_rates : list of float or None, default=None
            Coverage levels for prediction intervals (e.g., ``[0.9, 0.95]``
            for 90 % and 95 % intervals).  If ``None``, defaults to the rates
            used at fit time.
        strategy : {"mean", "median", "point"} or None, default=None
            Strategy for deriving point predictions from prediction intervals
            during recursive multi-step forecasting:

            - ``"mean"``: use the mean of the interval bounds
            - ``"median"``: use the median of the interval bounds
            - ``"point"``: use the point forecast directly (if available)

            If ``None``, defaults to ``"mean"``.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.
        stride : int or None, default=None
            Step size for rolling update-predict.  If ``None``, defaults to
            the forecasting horizon used at fit time
            (``fit_forecasting_horizon_``).
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
            Interval predictions with ``"vintage_time"``, ``"time"``, and
            lower/upper bound columns for each target at each coverage rate.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forecaster has not been fitted yet.
        ValueError
            If ``y`` / ``X_actual`` have invalid structure, ``coverage_rates`` not in
            [0, 1], or ``groups`` contains names not seen during fit.

        """
        check_is_fitted(self, ["groups_", "local_y_schema_", "fit_forecasting_horizon_"])
        y, X_actual, groups = validate_forecaster_data(
            self,
            y=y,
            X_actual=X_actual,
            reset=False,
            groups=groups,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        forecasting_horizon, _ = self._validate_predict_params(forecasting_horizon, coverage_rates)

        if stride is None:
            stride = self.fit_forecasting_horizon_

        return self._observe_predict_loop(
            predict_fn=self.predict_interval,
            y=y,
            X_actual=X_actual,
            X_future=X_future,
            X_forecast=X_forecast,
            groups=groups,
            stride=stride,
            forecasting_horizon=forecasting_horizon,
            coverage_rates=coverage_rates,
            strategy=strategy,
            **params,
        )

    def _predict_one(
        self,
        groups: list[str],
        coverage_rates: list[StrictFloat] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Predicts `fit_forecasting_horizon_` steps from the observation horizon.

        Parameters
        ----------
        groups : list of str
            Panel group names to predict for.
        coverage_rates : list of float or None, default=None
            Coverage rates for the prediction intervals. If ``None``,
            falls back to ``fit_coverage_rates_``.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        raise NotImplementedError()
