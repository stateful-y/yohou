"""ForecastedFeatureForecaster meta-forecaster for chaining feature and target forecasting."""

from __future__ import annotations

import numbers
from typing import Literal

import polars as pl
from pydantic import StrictInt
from sklearn.base import clone
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseForecaster
from yohou.utils import Tags
from yohou.utils._compat import Interval, StrOptions, _fit_context, _raise_for_params

__all__ = ["ForecastedFeatureForecaster"]


def _target_forecaster_has(method_name: str):
    """Check if target_forecaster has a given method.

    This function is used as the check for sklearn.utils.metaestimators.available_if
    decorator to conditionally expose predict/predict_interval methods based on
    the target forecaster's capabilities.

    Parameters
    ----------
    method_name : str
        Name of the method to check for (e.g., "predict", "predict_interval").

    Returns
    -------
    callable
        A check function for available_if decorator.

    """

    def check(self):
        """Check if target_forecaster has the required method."""
        # Use fitted forecaster_ if available, else unfitted forecaster
        forecaster = getattr(self, "target_forecaster_", self.target_forecaster)
        return hasattr(forecaster, method_name)

    return check


class ForecastedFeatureForecaster(BaseForecaster):
    """Meta-forecaster that chains feature forecasting into target forecasting.

    Fits a `feature_forecaster` to forecast exogenous features X_actual, then
    feeds those forecasts into a `target_forecaster` through its ``X_forecast``
    channel to predict y. The target consumes the forecast as contemporaneous
    step features (``feat_step_1 .. feat_step_H``), so the feature forecast
    actually influences the prediction at every horizon step.

    This is useful when exogenous features are not known in advance at
    prediction time and must be forecasted first.

    Parameters
    ----------
    target_forecaster : BaseForecaster
        Forecaster for the target variable y. Receives predicted X_actual at predict time.
    feature_forecaster : BaseForecaster
        Forecaster for exogenous features X_actual. Trained to forecast X_actual as if it were y.
    strategy : {"actual", "predicted", "rewind"}, default="rewind"
        Quality of the in-sample feature forecast the target trains on. In every
        strategy the forecast is supplied to the target through the ``X_forecast``
        channel (as ``vintage_time + time + features``), and the target decides how
        to consume the resulting step features via its own ``step_feature_alignment``.

        - "actual": Train the target on perfect-foresight features (actual values
          windowed forward and labelled with a vintage). Uses all data, but creates
          a train/serve mismatch since predict() uses forecasted features.
        - "predicted": Split data, fit the feature forecaster on the first portion,
          and train the target on a rolling forecast over the second portion.
          Requires more data but matches train and serve forecast quality.
        - "rewind": Fit the feature forecaster on full data, rewind to
          ``observation_horizon + 1`` rows (the extra row covers the transform
          step), then roll forward producing one vintage per origin and train
          the target on that rolling forecast. Requires
          ``len(X_actual) > observation_horizon + 1``. Uses all data for feature
          learning while matching train and serve forecast quality. This is the
          default: it has no train/serve mismatch.
    split_ratio : float, default=0.5
        Fraction of data used to fit feature_forecaster when strategy="predicted".
        Remaining data used for target_forecaster training with predicted X_actual.
        Ignored when strategy="actual" or "rewind". Must be in (0, 1).
    feature_stride : int, default=1
        Cadence (in time steps) at which the feature forecaster regenerates its
        forecast, applied identically at fit and serve so the target trains on
        features of the same vintage age it sees in production. With
        ``feature_stride=F`` the feature forecaster is fit and rolled at horizon
        ``H + F - 1`` (``H`` is the fit ``forecasting_horizon``) so a vintage up
        to ``F - 1`` steps old still covers the target's ``H`` steps. The default
        ``1`` regenerates every step (feature horizon ``H``). ``feature_stride > 1``
        is honored only when serving through ``observe_predict``/
        ``observe_predict_interval``/``observe_predict_class_proba`` (a bare
        ``predict`` always produces a single fresh forecast). Must be ``>= 1``.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See `BaseForecaster` for details.

    Attributes
    ----------
    target_forecaster_ : BaseForecaster
        Fitted target forecaster.
    feature_forecaster_ : BaseForecaster
        Fitted feature forecaster.
    fit_forecasting_horizon_ : int
        Forecasting horizon used during fit.
    interval_ : timedelta
        Time interval between observations.
    groups_ : list of str or None
        Panel group names if fitted on panel data.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from sklearn.linear_model import Ridge
    >>> from yohou.compose import ForecastedFeatureForecaster
    >>> from yohou.point import PointReductionForecaster
    >>>
    >>> # Create example time series with exogenous features
    >>> time = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 3, 31), interval="1d", eager=True
    ... )
    >>> y = pl.DataFrame({"time": time, "sales": range(1, len(time) + 1)})
    >>> X_actual = pl.DataFrame({"time": time, "price": [10 + i % 5 for i in range(len(time))]})
    >>>
    >>> # Create forecaster that predicts price first, then uses it for sales
    >>> forecaster = ForecastedFeatureForecaster(
    ...     target_forecaster=PointReductionForecaster(estimator=Ridge()),
    ...     feature_forecaster=PointReductionForecaster(estimator=Ridge()),
    ... )
    >>> forecaster.fit(y, X_actual, forecasting_horizon=7)  # doctest: +ELLIPSIS
    ForecastedFeatureForecaster(...)
    >>> y_pred = forecaster.predict(forecasting_horizon=7)
    >>> len(y_pred)
    7
    >>>
    >>> # Refresh the feature forecast every 2 steps instead of every step
    >>> forecaster_strided = ForecastedFeatureForecaster(
    ...     target_forecaster=PointReductionForecaster(estimator=Ridge()),
    ...     feature_forecaster=PointReductionForecaster(estimator=Ridge()),
    ...     feature_stride=2,
    ... )
    >>> forecaster_strided.fit(y, X_actual, forecasting_horizon=7)  # doctest: +ELLIPSIS
    ForecastedFeatureForecaster(...)
    >>> # feature_stride takes effect through the rolling observe_predict
    >>> rolled = forecaster_strided.observe_predict(y, X_actual, stride=7)
    >>> rolled["vintage_time"].n_unique() > 1
    True

    Notes
    -----
    - The feature_forecaster is trained with X_actual as y (forecasting the features).
    - At both fit and predict the feature forecast is passed to the target as
      ``X_forecast`` (keeping ``vintage_time``); the target builds step columns and
      consumes them per its ``step_feature_alignment``. A target with
      ``requires_exogenous=False`` (e.g. a naive forecaster) ignores the forecast.
    - Use strategy="predicted" or "rewind" when feature forecasts are noisy and you
      want the target to learn from similar-quality inputs as it sees at predict time.
    - A caller-supplied ``X_forecast`` (external forecasts for other features) is
      merged with the meta's feature forecast on ``(vintage_time, time)``; a value
      column-name collision raises ``ValueError``. ``X_future`` is forwarded untouched.
    - ``observe`` and ``rewind`` require ``X_actual`` (the feature forecaster needs new
      feature observations to advance in step with the target); passing ``None`` raises.
    - ``observe_predict`` (and its interval/class-proba variants) roll over the data one
      ``stride``-sized slice at a time, predicting at each origin, and regenerate the
      feature forecast every ``feature_stride`` steps. See the ``feature_stride`` parameter.
    - Cost: ``strategy="rewind"`` (default) and ``"predicted"`` build the in-sample
      forecast with a rolling ``observe_predict`` over the training span, so ``fit`` time
      grows with the training length (a larger ``feature_stride`` produces fewer vintages
      and is cheaper). ``strategy="actual"`` is the cheapest.

    See Also
    --------
    - [`ColumnForecaster`][yohou.compose.column_forecaster.ColumnForecaster] : Apply different forecasters to different column subsets.
    - [`DecompositionPipeline`][yohou.compose.decomposition_pipeline.DecompositionPipeline] : Sequential decomposition into trend + seasonality + residual.

    """

    _parameter_constraints: dict = {
        **BaseForecaster._parameter_constraints,
        "target_forecaster": [BaseForecaster],
        "feature_forecaster": [BaseForecaster],
        "strategy": [StrOptions({"actual", "predicted", "rewind"})],
        "split_ratio": [Interval(numbers.Real, 0.0, 1.0, closed="neither")],
        "feature_stride": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        target_forecaster: BaseForecaster,
        feature_forecaster: BaseForecaster,
        *,
        strategy: Literal["actual", "predicted", "rewind"] = "rewind",
        split_ratio: float = 0.5,
        feature_stride: int = 1,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        super().__init__(panel_strategy=panel_strategy)
        self.target_forecaster = target_forecaster
        self.feature_forecaster = feature_forecaster
        self.strategy = strategy
        self.split_ratio = split_ratio
        self.feature_stride = feature_stride

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None

        # Aggregate forecaster_type from target (determines output type)
        target_tags = self.target_forecaster.__sklearn_tags__()
        if target_tags.forecaster_tags:
            tags.forecaster_tags.forecaster_type = target_tags.forecaster_tags.forecaster_type

        # Aggregate stateful from both
        feature_tags = self.feature_forecaster.__sklearn_tags__()
        target_stateful = target_tags.forecaster_tags.stateful if target_tags.forecaster_tags else False
        feature_stateful = feature_tags.forecaster_tags.stateful if feature_tags.forecaster_tags else False
        tags.forecaster_tags.stateful = target_stateful or feature_stateful

        # Aggregate other tags
        # Note: uses_reduction is False since this meta-forecaster doesn't have an `estimator`
        # attribute directly - child forecasters may use reduction, but that's their internal detail
        tags.forecaster_tags.uses_reduction = False
        tags.forecaster_tags.supports_panel_data = getattr(
            target_tags.forecaster_tags, "supports_panel_data", True
        ) and getattr(feature_tags.forecaster_tags, "supports_panel_data", True)

        # Delegates observation tracking to child forecasters
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
    ) -> ForecastedFeatureForecaster:
        """Fit feature and target forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with "time" column.
        X_actual : pl.DataFrame
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Required. The feature forecaster uses these as
            its target variable.
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
            Metadata routing parameters.

        Returns
        -------
        self
            Fitted forecaster.

        Raises
        ------
        ValueError
            If ``X_actual`` is None (exogenous features are required); if
            ``forecasting_horizon < 1``; if ``strategy="rewind"`` and
            ``len(X_actual) <= observation_horizon + 1``; or if
            ``strategy="predicted"`` and the split leaves fewer than 2 rows
            for either the feature-forecaster fit or the rolling forecast.

        Notes
        -----
        When the target forecaster consumes exogenous features, its effective
        training window starts at the first vintage the feature forecast
        covers: ``y`` is filtered to ``time >= first_covered``. For
        ``strategy="predicted"`` this excludes roughly the first
        ``split_ratio`` fraction of ``y`` from target training.

        """
        if X_actual is None:
            raise ValueError(
                "ForecastedFeatureForecaster requires X_actual (exogenous features). "
                "Pass exogenous features that need to be forecasted."
            )

        # Validate forecasting horizon
        if forecasting_horizon < 1:
            raise ValueError(f"forecasting_horizon must be >= 1, got {forecasting_horizon}")

        # Validate params before routing
        _raise_for_params(params, self, "fit")

        # Process metadata routing
        routed_params = process_routing(self, "fit", **params)

        # Clone forecasters
        self.feature_forecaster_ = clone(self.feature_forecaster)
        self.target_forecaster_ = clone(self.target_forecaster)

        # Build an in-sample feature forecast as an X_forecast frame
        # (vintage_time + time + features) and leave feature_forecaster_ observed
        # up to the end of the training data so its vintages line up with the
        # target at predict time. The forecast is routed to the target through the
        # X_forecast channel (not X_actual), so the target builds step columns and
        # actually consumes the forecast at every horizon step.
        #
        # feature_stride sets the feature forecaster's vintage cadence: vintages are
        # spaced feature_stride apart and each spans feature_horizon = H + F - 1
        # steps, so a vintage up to F-1 steps old still covers the target's H steps.
        # The same cadence is replayed at serve (see the observe_predict variants).
        #
        # A target with requires_exogenous=False (e.g. a naive forecaster) ignores
        # X_forecast. Each strategy still runs its validation and leaves
        # feature_forecaster_ observed at the end of training (observed-time parity),
        # but for such a target the rolling in-sample forecast is skipped (it would
        # only be built to be discarded); feature_forecast stays None.
        self._target_requires_exogenous_ = bool(
            self.target_forecaster_.__sklearn_tags__().forecaster_tags.requires_exogenous
        )
        feature_horizon = forecasting_horizon + self.feature_stride - 1
        if self.feature_stride > 1 and feature_horizon >= len(X_actual):
            raise ValueError(
                f"feature_stride={self.feature_stride} requires the feature forecaster to "
                f"forecast H + feature_stride - 1 = {feature_horizon} steps, but "
                f"len(X_actual)={len(X_actual)} is too short. Reduce feature_stride or "
                f"provide more data."
            )
        feature_forecast = None
        if self.strategy == "actual":
            # Perfect foresight: train the feature forecaster on all data (leaving
            # it observed at the end) and feed the target the actual feature values
            # windowed forward as a vintage-labelled forecast. Window only the
            # columns the feature forecaster actually produces (it may forecast a
            # subset of X_actual, e.g. a ColumnForecaster that drops known-ahead
            # columns), so the fit-time and predict-time X_forecast schemas match.
            self.feature_forecaster_.fit(
                y=X_actual,
                X_actual=None,
                forecasting_horizon=feature_horizon,
                **routed_params.feature_forecaster.fit,
            )
            if self._target_requires_exogenous_:
                # The forecast output columns are the feature forecaster's fitted target
                # schema (read directly, without a predict call that would otherwise mutate
                # nothing but pollute call-tracking and waste work). local_y_schema_ holds
                # the local (unprefixed) names, so compare panel columns by their
                # group-stripped base name (`group__column` -> `column`).
                local_schema = set(self.feature_forecaster_.local_y_schema_)
                forecast_cols = [
                    c
                    for c in X_actual.columns
                    if c != "time" and (c.split("__", 1)[1] if "__" in c else c) in local_schema
                ]
                feature_forecast = self._actuals_as_forecast(
                    X_actual.select(["time", *forecast_cols]), feature_horizon, self.feature_stride
                )

        elif self.strategy == "rewind":
            # Fit on full data, rewind to the observation horizon, then roll forward
            # producing one vintage per feature_stride origin. observe_predict leaves
            # the feature forecaster observed at the end of the data (state synced).
            self.feature_forecaster_.fit(
                y=X_actual,
                X_actual=None,
                forecasting_horizon=feature_horizon,
                **routed_params.feature_forecaster.fit,
            )
            obs_horizon = self.feature_forecaster_.observation_horizon
            # Need obs_horizon + 1 rows: obs_horizon for transformer memory + 1 for transform
            rewind_size = obs_horizon + 1
            if len(X_actual) <= rewind_size:
                raise ValueError(
                    f"Cannot use strategy='rewind' (the default): the feature forecaster's "
                    f"observation_horizon={obs_horizon} requires len(X_actual) > observation_horizon + 1, "
                    f"but len(X_actual)={len(X_actual)}. Provide more data, or use "
                    f"strategy='actual' or strategy='predicted' for very short series."
                )
            self.feature_forecaster_.rewind(y=X_actual[:rewind_size], X_actual=None)
            if self._target_requires_exogenous_:
                feature_forecast = self.feature_forecaster_.observe_predict(
                    y=X_actual[rewind_size:],
                    X_actual=None,
                    forecasting_horizon=feature_horizon,
                    stride=self.feature_stride,
                )
            else:
                # Sync state to the end of training without the rolling forecast.
                self.feature_forecaster_.observe(y=X_actual[rewind_size:], X_actual=None)

        else:  # strategy == "predicted"
            # Split the feature series (X_actual) since that is what gets sliced
            # for the feature forecaster below. Using len(X_actual) keeps the
            # split point consistent across both portions even when X_actual and
            # y differ in length.
            n_split = int(len(X_actual) * self.split_ratio)

            if n_split < 2:
                raise ValueError(
                    f"split_ratio={self.split_ratio} results in n_split={n_split}. "
                    f"Increase split_ratio or provide more data (len(X_actual)={len(X_actual)})."
                )
            if len(X_actual) - n_split < 2:
                raise ValueError(
                    f"split_ratio={self.split_ratio} results in n_split={n_split}, "
                    f"leaving only {len(X_actual) - n_split} rows for the feature forecast roll. "
                    f"Decrease split_ratio to leave at least 2 rows."
                )

            # Fit the feature forecaster on the first portion, then roll forward over
            # the second portion (one vintage per feature_stride origin). observe_predict
            # leaves the feature forecaster observed at the end of the data (state synced).
            self.feature_forecaster_.fit(
                y=X_actual[:n_split],
                X_actual=None,
                forecasting_horizon=feature_horizon,
                **routed_params.feature_forecaster.fit,
            )
            if self._target_requires_exogenous_:
                feature_forecast = self.feature_forecaster_.observe_predict(
                    y=X_actual[n_split:],
                    X_actual=None,
                    forecasting_horizon=feature_horizon,
                    stride=self.feature_stride,
                )
            else:
                # Sync state to the end of training without the rolling forecast.
                self.feature_forecaster_.observe(y=X_actual[n_split:], X_actual=None)

        # Feed the forecast to the target only when it consumes exogenous features
        # (_target_requires_exogenous_ was resolved before the strategy block above).
        if self._target_requires_exogenous_:
            # An exogenous target always built a forecast in the strategy block above.
            assert feature_forecast is not None
            # Merge any caller-supplied external forecast, then align the target's
            # training window to the span the feature forecast actually covers (early
            # rows have no vintage and would yield null step features).
            target_X_forecast = self._merge_forecasts(feature_forecast, X_forecast)
            first_covered = feature_forecast["vintage_time"].min()
            y_target = y.filter(pl.col("time") >= first_covered)
        else:
            target_X_forecast = None
            y_target = y

        # The target consumes the feature forecast via X_forecast; X_actual is left
        # to genuine lagged-history features (none here), so it is None.
        self.target_forecaster_.fit(
            y=y_target,
            X_actual=None,
            forecasting_horizon=forecasting_horizon,
            X_future=X_future,
            X_forecast=target_X_forecast,
            **routed_params.target_forecaster.fit,
        )

        # Store standard fitted attributes
        self.fit_forecasting_horizon_ = forecasting_horizon
        self.interval_ = self.target_forecaster_.interval_
        self.groups_ = self.target_forecaster_.groups_
        if hasattr(self.target_forecaster_, "local_y_schema_"):
            self.local_y_schema_ = self.target_forecaster_.local_y_schema_
        # The meta-forecaster's public X_actual is the exogenous feature block the
        # user passes in; describe it from the input. (The target itself is fit with
        # X_actual=None because the features flow through the X_forecast channel.)
        self.local_X_actual_schema_ = {col: X_actual.schema[col] for col in X_actual.columns if col != "time"}
        self.shared_X_actual_schema_ = None  # No shared X_actual features for this forecaster

        # Set transformed schema attributes (no transformation for meta-forecaster)
        self.local_y_t_schema_ = self.local_y_schema_
        self.local_X_t_schema_ = self.local_X_actual_schema_

        # Set observation buffers for observe/rewind
        self._y_observed = y
        self._X_t_observed = X_actual
        # Expose observed_time_ per the fitted-forecaster contract; it tracks
        # the end of the data this meta-forecaster has observed (full y).
        self.observed_time_ = y["time"][-1]

        return self

    def _actuals_as_forecast(self, X_actual: pl.DataFrame, forecasting_horizon: int, stride: int = 1) -> pl.DataFrame:
        """Window actual feature values into a perfect-foresight X_forecast frame.

        For each origin time ``T`` on the ``stride`` grid emits the actual feature
        values at ``T+1 .. T+forecasting_horizon``, tagged with ``vintage_time=T``.
        Used by ``strategy="actual"`` so the target trains on perfect-foresight step
        features through the same ``X_forecast`` channel it receives forecasts from
        at predict time. ``stride`` is the feature forecast cadence: with ``stride>1``
        only every ``stride``-th origin is a vintage, matching the serve-time refresh.

        Parameters
        ----------
        X_actual : pl.DataFrame
            Actual feature observations with a ``"time"`` column.
        forecasting_horizon : int
            Number of forward steps per origin (``H + feature_stride - 1``).
        stride : int, default=1
            Vintage cadence; keep only origins whose row index is a multiple of it.

        Returns
        -------
        pl.DataFrame
            Long frame with ``["vintage_time", "time", <features>]``.

        """
        feature_cols = [c for c in X_actual.columns if c != "time"]
        # Window forward on the full (unfiltered) time axis so shift(-h) means h true
        # steps ahead: vintage at every origin T covers T+1 .. T+forecasting_horizon.
        parts = [
            X_actual.select(
                pl.col("time").alias("vintage_time"),
                pl.col("time").shift(-h).alias("time"),
                *[pl.col(c).shift(-h).alias(c) for c in feature_cols],
            ).drop_nulls(subset=["time"])
            for h in range(1, forecasting_horizon + 1)
        ]
        feature_forecast = pl.concat(parts).sort("vintage_time", "time")
        if stride > 1:
            # Keep only every stride-th origin as a vintage (the feature_stride grid),
            # so the target trains on features aged 0..stride-1, matching serve.
            grid_times = X_actual["time"].gather_every(stride).to_list()
            feature_forecast = feature_forecast.filter(pl.col("vintage_time").is_in(grid_times))
        return feature_forecast

    def _merge_forecasts(
        self,
        feature_forecast: pl.DataFrame,
        user_X_forecast: pl.DataFrame | None,
    ) -> pl.DataFrame:
        """Merge the meta's feature forecast with a caller-supplied X_forecast.

        Joins on ``(vintage_time, time)``. Raises ``ValueError`` on a value-column
        name collision. Returns ``feature_forecast`` unchanged when
        ``user_X_forecast`` is None.

        Parameters
        ----------
        feature_forecast : pl.DataFrame
            The meta's forecast of the exogenous features.
        user_X_forecast : pl.DataFrame or None
            Caller-supplied external forecast for other features.

        Returns
        -------
        pl.DataFrame
            The combined external-forecast frame.

        """
        if user_X_forecast is None:
            return feature_forecast
        keys = {"vintage_time", "time"}
        collision = (set(feature_forecast.columns) - keys) & (set(user_X_forecast.columns) - keys)
        if collision:
            raise ValueError(
                "X_forecast supplied to ForecastedFeatureForecaster collides with "
                f"forecasted feature column(s) {sorted(collision)}. Rename the external "
                "forecast columns or drop them from the feature block."
            )
        # Left join anchored on the meta forecast's (vintage_time, time) grid: the
        # caller forecast contributes columns where its rows align and never adds
        # rows of its own (which would be null in every feature column).
        return feature_forecast.join(user_X_forecast, on=["vintage_time", "time"], how="left")

    def _predict_feature_forecast(
        self,
        forecasting_horizon: int,
        groups: list[str] | None,
        user_X_forecast: pl.DataFrame | None,
        routed_predict_params: dict,
    ) -> pl.DataFrame | None:
        """Produce the X_forecast frame to pass to the target at predict time.

        Forecasts the exogenous features and merges them with any caller-supplied
        ``X_forecast``. Returns the caller's ``X_forecast`` unchanged (no feature
        forecast generated) when the target ignores exogenous features.

        Parameters
        ----------
        forecasting_horizon : int
            Number of steps ahead to forecast.
        groups : list of str or None
            Panel group prefixes.
        user_X_forecast : pl.DataFrame or None
            Caller-supplied external forecast.
        routed_predict_params : dict
            Metadata-routing params for the feature forecaster's predict call.

        Returns
        -------
        pl.DataFrame or None
            The combined external-forecast frame, or None when the target does
            not consume exogenous features.

        """
        if not getattr(self, "_target_requires_exogenous_", True):
            return None
        # During a feature_stride>1 rolling serve, the loop holds a single reused
        # vintage (refreshed every feature_stride steps) so the target consumes an
        # aging vintage via as-of selection instead of a fresh forecast.
        buffer = getattr(self, "_feature_forecast_buffer", None)
        if buffer is not None:
            return self._merge_forecasts(buffer, user_X_forecast)
        feature_forecast = self.feature_forecaster_.predict(
            forecasting_horizon=forecasting_horizon,
            groups=groups,
            **routed_predict_params,
        )
        return self._merge_forecasts(feature_forecast, user_X_forecast)

    @available_if(_target_forecaster_has("predict"))
    def predict(
        self,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
        predict_transformed: bool = False,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate point forecasts.

        Forecasts X_actual using feature_forecaster, then uses those predictions
        as exogenous features for target_forecaster to predict y.

        Parameters
        ----------
        forecasting_horizon : int, optional
            Number of steps ahead to forecast. If None, uses value from fit().
        groups : list of str or None, default=None
            Group prefixes for panel data prediction.
        predict_transformed : bool, default=False
            If True, return predictions in the transformed space without
            applying inverse target transformation.
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
            Predictions with "vintage_time", "time", and target columns.

        """
        check_is_fitted(self, ["target_forecaster_", "feature_forecaster_"])

        if forecasting_horizon is None:
            forecasting_horizon = self.fit_forecasting_horizon_

        # Validate params before routing
        _raise_for_params(params, self, "predict")

        # Process metadata routing
        routed_params = process_routing(self, "predict", **params)

        # Forecast the exogenous features, then feed that forecast to the target
        # through the X_forecast channel (merged with any caller-supplied forecast).
        target_X_forecast = self._predict_feature_forecast(
            forecasting_horizon, groups, X_forecast, routed_params.feature_forecaster.predict
        )

        return self.target_forecaster_.predict(
            forecasting_horizon=forecasting_horizon,
            groups=groups,
            predict_transformed=predict_transformed,
            X_future=X_future,
            X_forecast=target_X_forecast,
            **routed_params.target_forecaster.predict,
        )

    @available_if(_target_forecaster_has("predict_interval"))
    def predict_interval(
        self,
        forecasting_horizon: StrictInt | None = None,
        coverage_rates: list[float] | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate interval forecasts.

        Only available if target_forecaster supports interval predictions.
        Feature forecaster always produces point predictions for X_actual.

        Parameters
        ----------
        forecasting_horizon : int, optional
            Number of steps ahead to forecast. If None, uses value from fit().
        coverage_rates : list of float, optional
            Coverage levels for prediction intervals (e.g., [0.9, 0.95]).
        groups : list of str or None, default=None
            Group prefixes for panel data prediction.
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
            Interval predictions with lower/upper bounds.

        """
        check_is_fitted(self, ["target_forecaster_", "feature_forecaster_"])

        if forecasting_horizon is None:
            forecasting_horizon = self.fit_forecasting_horizon_

        # Validate params before routing
        _raise_for_params(params, self, "predict_interval")

        # Process metadata routing
        routed_params = process_routing(self, "predict_interval", **params)

        # The feature forecaster always produces point forecasts of the features;
        # feed them to the target through X_forecast.
        target_X_forecast = self._predict_feature_forecast(
            forecasting_horizon, groups, X_forecast, routed_params.feature_forecaster.predict
        )

        return self.target_forecaster_.predict_interval(
            forecasting_horizon=forecasting_horizon,
            coverage_rates=coverage_rates,
            groups=groups,
            X_future=X_future,
            X_forecast=target_X_forecast,
            **routed_params.target_forecaster.predict_interval,
        )

    def observe(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> ForecastedFeatureForecaster:
        """Observe new data for both forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations with "time" column.
        X_actual : pl.DataFrame
            New actual feature observations with a ``"time"`` column
            aligned with ``y``. Required: the feature forecaster needs new
            feature observations to advance in step with the target.
        groups : list of str or None, default=None
            Group prefixes for panel data.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.

        Returns
        -------
        self
            Forecaster with new observations incorporated.

        Raises
        ------
        ValueError
            If ``X_actual`` is None.

        """
        check_is_fitted(self, ["target_forecaster_", "feature_forecaster_"])

        if X_actual is None:
            raise ValueError(
                "ForecastedFeatureForecaster.observe requires X_actual. The feature "
                "forecaster forecasts X_actual, so observing new y without the aligned "
                "new X_actual would desynchronize the feature and target forecasters' "
                "observation state."
            )

        # Observe new X_actual for feature forecaster (X_actual is y for feature forecaster)
        self.feature_forecaster_.observe(
            y=X_actual,
            X_actual=None,
            groups=groups,
        )

        # Observe new y for the target forecaster. X_actual is None because the
        # target consumes the features through X_forecast, not its lagged-history
        # channel; the feature forecast is regenerated at predict time.
        self.target_forecaster_.observe(
            y=y,
            X_actual=None,
            groups=groups,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        # Advance the meta-forecaster's observed_time_ to the latest observation.
        self.observed_time_ = y["time"][-1]

        return self

    def rewind(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> ForecastedFeatureForecaster:
        """Rewind both forecasters to last observation_horizon rows.

        Parameters
        ----------
        y : pl.DataFrame
            Target data to rewind to (last observation_horizon rows kept).
        X_actual : pl.DataFrame
            Actual feature observations to restore the observation
            state to. Required and must align with ``y``.
        groups : list of str or None, default=None
            Group prefixes for panel data.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.

        Returns
        -------
        self
            Rewound forecaster.

        Raises
        ------
        ValueError
            If ``X_actual`` is None.

        """
        check_is_fitted(self, ["target_forecaster_", "feature_forecaster_"])

        if X_actual is None:
            raise ValueError(
                "ForecastedFeatureForecaster.rewind requires X_actual. The feature "
                "forecaster forecasts X_actual, so rewinding y without the aligned "
                "X_actual would desynchronize the feature and target forecasters' "
                "observation state."
            )

        # Rewind feature forecaster (X_actual is y for feature forecaster)
        self.feature_forecaster_.rewind(
            y=X_actual,
            X_actual=None,
            groups=groups,
        )

        # Rewind target forecaster. X_actual is None because the target consumes
        # the features through X_forecast, not its lagged-history channel.
        self.target_forecaster_.rewind(
            y=y,
            X_actual=None,
            groups=groups,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        # Reset the meta-forecaster's observed_time_ to the rewind window end.
        self.observed_time_ = y["time"][-1]

        return self

    def _run_observe_predict(
        self,
        *,
        predict_fn,
        routing_method: str,
        routing_params: dict,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None,
        groups: list[str] | None,
        stride: StrictInt | None,
        X_future: pl.DataFrame | None,
        X_forecast: pl.DataFrame | None,
        forecasting_horizon: StrictInt | None,
        **predict_kwargs,
    ) -> pl.DataFrame:
        """Shared rolling-origin dispatch for the observe_predict variants.

        Uses the base rolling loop (one fresh feature forecast per origin) for
        ``feature_stride == 1`` or non-exogenous targets, and a feature-stride
        loop (refresh every ``feature_stride`` steps, reusing an aging vintage)
        otherwise.
        """
        check_is_fitted(self, ["target_forecaster_", "feature_forecaster_"])
        if X_actual is None:
            raise ValueError(
                "ForecastedFeatureForecaster.observe_predict requires X_actual. The "
                "feature forecaster needs new feature observations to advance in step "
                "with the target."
            )
        fh = forecasting_horizon if forecasting_horizon is not None else self.fit_forecasting_horizon_
        if stride is None:
            stride = self.fit_forecasting_horizon_
        # Resolve the feature forecaster's routed predict params once. The per-origin
        # predict already routes them, but the strided buffer refresh bypasses that
        # path, so it must apply the same params (parity with feature_stride == 1).
        routed = process_routing(self, routing_method, **routing_params)
        feature_predict_params = routed.feature_forecaster.predict
        predict_kwargs = {"forecasting_horizon": fh, **routing_params, **predict_kwargs}

        if self.feature_stride > 1 and getattr(self, "_target_requires_exogenous_", True):
            return self._feature_stride_serve_loop(
                predict_fn=predict_fn,
                feature_predict_params=feature_predict_params,
                y=y,
                X_actual=X_actual,
                X_future=X_future,
                X_forecast=X_forecast,
                groups=groups,
                stride=stride,
                **predict_kwargs,
            )
        return self._observe_predict_loop(
            predict_fn=predict_fn,
            y=y,
            X_actual=X_actual,
            X_future=X_future,
            X_forecast=X_forecast,
            groups=groups,
            stride=stride,
            observe_fn=self.observe,
            **predict_kwargs,
        )

    def _feature_stride_serve_loop(
        self,
        *,
        predict_fn,
        feature_predict_params: dict,
        y: pl.DataFrame,
        X_actual: pl.DataFrame,
        X_future: pl.DataFrame | None,
        X_forecast: pl.DataFrame | None,
        groups: list[str] | None,
        stride: int,
        forecasting_horizon: int,
        **predict_kwargs,
    ) -> pl.DataFrame:
        """Roll forward refreshing the feature forecast every feature_stride.

        Holds a single (possibly aging) feature-forecaster vintage: it is replaced
        with a fresh one only when the observation clock crosses a ``feature_stride``
        boundary, otherwise the target reuses the current vintage via as-of
        selection. Because the observation clock only advances, an older vintage can
        never win the as-of join again, so one slot suffices (no unbounded concat).
        The target predicts at the loop ``stride``. The buffer is exposed to
        ``predict`` through ``_feature_forecast_buffer`` so all of predict's
        routing/merge logic is reused.
        """
        feature_stride = self.feature_stride
        feature_horizon = forecasting_horizon + feature_stride - 1
        self._feature_forecast_buffer = self.feature_forecaster_.predict(
            forecasting_horizon=feature_horizon, groups=groups, **feature_predict_params
        )
        try:
            y_pred = predict_fn(groups=groups, forecasting_horizon=forecasting_horizon, **predict_kwargs)
            steps_since_refresh = 0
            for i in range(0, len(y), stride):
                y_slice = y[i : i + stride]
                x_obs_slice = X_actual.join(y_slice.select("time"), on="time", how="semi")
                self.observe(y=y_slice, X_actual=x_obs_slice, groups=groups, X_future=X_future, X_forecast=X_forecast)
                steps_since_refresh += len(y_slice)
                if steps_since_refresh >= feature_stride:
                    # The clock only moves forward, so the previous vintage can never
                    # win the as-of join again: replace it rather than accumulate.
                    self._feature_forecast_buffer = self.feature_forecaster_.predict(
                        forecasting_horizon=feature_horizon, groups=groups, **feature_predict_params
                    )
                    steps_since_refresh = steps_since_refresh % feature_stride
                y_pred_i = predict_fn(groups=groups, forecasting_horizon=forecasting_horizon, **predict_kwargs)
                y_pred = pl.concat([y_pred, y_pred_i])
            return y_pred
        finally:
            self._feature_forecast_buffer = None

    @available_if(_target_forecaster_has("predict"))
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
        """Observe new data and roll forward generating point forecasts.

        Runs a rolling-origin loop: predict, observe a ``stride``-sized slice,
        re-predict, and concatenate the multi-vintage result (consistent with the
        base forecaster contract). The feature forecast is regenerated every
        ``feature_stride`` observed steps (default 1 = every step).

        Parameters
        ----------
        y : pl.DataFrame
            New target observations with "time" column.
        X_actual : pl.DataFrame
            Actual feature observations with a ``"time"`` column aligned with ``y``,
            sliced and observed incrementally. Required (raises if None).
        forecasting_horizon : int, optional
            Steps per prediction block. If None, uses the fit horizon.
        groups : list of str or None, default=None
            Group prefixes for panel data.
        stride : int, optional
            Rows observed between successive predictions. If None, uses the fit horizon.
        predict_transformed : bool, default=False
            If True, return predictions in the transformed space.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"`` columns.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Rolling point predictions with "vintage_time", "time", and target columns.

        """
        return self._run_observe_predict(
            predict_fn=self.predict,
            routing_method="predict",
            routing_params=params,
            y=y,
            X_actual=X_actual,
            groups=groups,
            stride=stride,
            X_future=X_future,
            X_forecast=X_forecast,
            forecasting_horizon=forecasting_horizon,
            predict_transformed=predict_transformed,
        )

    @available_if(_target_forecaster_has("predict_interval"))
    def observe_predict_interval(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        coverage_rates: list[float] | None = None,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
        stride: StrictInt | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Observe new data and roll forward generating interval forecasts.

        Rolling-origin loop (see ``observe_predict``); the feature forecast is
        regenerated every ``feature_stride`` observed steps.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations with "time" column.
        X_actual : pl.DataFrame
            Actual feature observations aligned with ``y``. Required (raises if None).
        coverage_rates : list of float, optional
            Coverage levels for prediction intervals.
        forecasting_horizon : int, optional
            Steps per prediction block. If None, uses the fit horizon.
        groups : list of str or None, default=None
            Group prefixes for panel data.
        stride : int, optional
            Rows observed between successive predictions. If None, uses the fit horizon.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"`` columns.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Rolling interval predictions with lower/upper bounds.

        """
        return self._run_observe_predict(
            predict_fn=self.predict_interval,
            routing_method="predict_interval",
            routing_params=params,
            y=y,
            X_actual=X_actual,
            groups=groups,
            stride=stride,
            X_future=X_future,
            X_forecast=X_forecast,
            forecasting_horizon=forecasting_horizon,
            coverage_rates=coverage_rates,
        )

    @available_if(_target_forecaster_has("predict_class_proba"))
    def predict_class_proba(
        self,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate class-probability forecasts.

        Only available if target_forecaster supports class-probability predictions.
        Feature forecaster always produces point predictions for X_actual.

        Parameters
        ----------
        forecasting_horizon : int, optional
            Number of steps ahead to forecast. If None, uses value from fit().
        groups : list of str or None, default=None
            Group prefixes for panel data prediction.
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
            Class-probability predictions with "vintage_time", "time", and
            probability columns.

        """
        check_is_fitted(self, ["target_forecaster_", "feature_forecaster_"])

        if forecasting_horizon is None:
            forecasting_horizon = self.fit_forecasting_horizon_

        _raise_for_params(params, self, "predict_class_proba")
        routed_params = process_routing(self, "predict_class_proba", **params)

        target_X_forecast = self._predict_feature_forecast(
            forecasting_horizon, groups, X_forecast, routed_params.feature_forecaster.predict
        )

        return self.target_forecaster_.predict_class_proba(
            forecasting_horizon=forecasting_horizon,
            groups=groups,
            X_future=X_future,
            X_forecast=target_X_forecast,
            **routed_params.target_forecaster.predict_class_proba,
        )

    @available_if(_target_forecaster_has("predict_class_proba"))
    def observe_predict_class_proba(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
        stride: StrictInt | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Observe new data and roll forward generating class-probability forecasts.

        Rolling-origin loop (see ``observe_predict``); the feature forecast is
        regenerated every ``feature_stride`` observed steps.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations with "time" column.
        X_actual : pl.DataFrame
            Actual feature observations aligned with ``y``. Required (raises if None).
        forecasting_horizon : int, optional
            Steps per prediction block. If None, uses the fit horizon.
        groups : list of str or None, default=None
            Group prefixes for panel data.
        stride : int, optional
            Rows observed between successive predictions. If None, uses the fit horizon.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"`` columns.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Rolling class-probability predictions.

        """
        return self._run_observe_predict(
            predict_fn=self.predict_class_proba,
            routing_method="predict_class_proba",
            routing_params=params,
            y=y,
            X_actual=X_actual,
            groups=groups,
            stride=stride,
            X_future=X_future,
            X_forecast=X_forecast,
            forecasting_horizon=forecasting_horizon,
        )

    def get_metadata_routing(self):
        """Get metadata routing for both forecasters.

        Returns
        -------
        MetadataRouter
            Router with mappings for target_forecaster and feature_forecaster.

        """
        router = MetadataRouter(owner=self.__class__.__name__)

        router.add(
            target_forecaster=self.target_forecaster,
            method_mapping=MethodMapping()
            .add(caller="fit", callee="fit")
            .add(caller="predict", callee="predict")
            .add(caller="predict_interval", callee="predict_interval")
            .add(caller="predict_class_proba", callee="predict_class_proba")
            .add(caller="observe_predict", callee="observe_predict")
            .add(caller="observe_predict_interval", callee="observe_predict_interval")
            .add(caller="observe_predict_class_proba", callee="observe_predict_class_proba"),
        )

        router.add(
            feature_forecaster=self.feature_forecaster,
            method_mapping=MethodMapping()
            .add(caller="fit", callee="fit")
            .add(caller="predict", callee="predict")
            .add(caller="predict_interval", callee="predict")
            .add(caller="predict_class_proba", callee="predict")
            .add(caller="observe_predict", callee="observe_predict")
            .add(caller="observe_predict_interval", callee="observe_predict")
            .add(caller="observe_predict_class_proba", callee="observe_predict"),
        )

        return router
