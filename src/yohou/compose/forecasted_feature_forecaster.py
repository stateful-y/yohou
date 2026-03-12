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

    Fits a `feature_forecaster` to forecast exogenous features X, then
    feeds those predicted features into a `target_forecaster` to predict y.

    This is useful when exogenous features are not known in advance at
    prediction time and must be forecasted first.

    Parameters
    ----------
    target_forecaster : BaseForecaster
        Forecaster for the target variable y. Receives predicted X at predict time.
    feature_forecaster : BaseForecaster
        Forecaster for exogenous features X. Trained to forecast X as if it were y.
    strategy : {"actual", "predicted", "rewind"}, default="actual"
        Training data strategy for target forecaster:

        - "actual": Train target forecaster on actual X values (perfect foresight).
          Simpler and uses all data, but may cause train-test mismatch since
          predict() uses forecasted X.
        - "predicted": Split data and train target forecaster on predicted X values.
          Requires more data but avoids distribution shift between train and predict.
        - "rewind": Fit feature_forecaster on full data, rewind to observation horizon,
          then predict X and train target_forecaster on predicted X. Uses all data
          for feature learning while avoiding distribution shift.
    split_ratio : float, default=0.5
        Fraction of data used to fit feature_forecaster when strategy="predicted".
        Remaining data used for target_forecaster training with predicted X.
        Ignored when strategy="actual" or "rewind". Must be in (0, 1).
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
    panel_group_names_ : list of str or None
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
    >>> X = pl.DataFrame({"time": time, "price": [10 + i % 5 for i in range(len(time))]})
    >>>
    >>> # Create forecaster that predicts price first, then uses it for sales
    >>> forecaster = ForecastedFeatureForecaster(
    ...     target_forecaster=PointReductionForecaster(estimator=Ridge()),
    ...     feature_forecaster=PointReductionForecaster(estimator=Ridge()),
    ... )
    >>> forecaster.fit(y, X, forecasting_horizon=7)  # doctest: +ELLIPSIS
    ForecastedFeatureForecaster(...)
    >>> y_pred = forecaster.predict(forecasting_horizon=7)
    >>> len(y_pred)
    7

    Notes
    -----
    - The feature_forecaster is trained with X as y (forecasting the features)
    - The target_forecaster receives forecasted X at predict time
    - Use strategy="predicted" when feature forecasts are noisy and you want
      the target forecaster to learn from similar quality inputs as it will see
      at prediction time
    - At predict time, X can contain known-ahead features (e.g., holidays,
      promotions) that don't need forecasting. These are merged with the
      forecasted features before being passed to the target_forecaster.

    See Also
    --------
    `ColumnForecaster` : Apply different forecasters to different column subsets.
    `DecompositionPipeline` : Sequential decomposition into trend + seasonality + residual.

    """

    _parameter_constraints: dict = {
        **BaseForecaster._parameter_constraints,
        "target_forecaster": [BaseForecaster],
        "feature_forecaster": [BaseForecaster],
        "strategy": [StrOptions({"actual", "predicted", "rewind"})],
        "split_ratio": [Interval(numbers.Real, 0.0, 1.0, closed="neither")],
    }

    def __init__(
        self,
        target_forecaster: BaseForecaster,
        feature_forecaster: BaseForecaster,
        *,
        strategy: Literal["actual", "predicted", "rewind"] = "actual",
        split_ratio: float = 0.5,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        super().__init__(panel_strategy=panel_strategy)
        self.target_forecaster = target_forecaster
        self.feature_forecaster = feature_forecaster
        self.strategy = strategy
        self.split_ratio = split_ratio

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
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> ForecastedFeatureForecaster:
        """Fit feature and target forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with "time" column.
        X : pl.DataFrame
            Exogenous features with "time" column. Required.
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        self
            Fitted forecaster.

        Raises
        ------
        ValueError
            If X is None (exogenous features are required).

        """
        if X is None:
            raise ValueError(
                "ForecastedFeatureForecaster requires X (exogenous features). "
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

        if self.strategy == "actual":
            # Fit feature forecaster: X is treated as y (what to forecast)
            self.feature_forecaster_.fit(
                y=X,
                X=None,
                forecasting_horizon=forecasting_horizon,
                **routed_params.feature_forecaster.fit,
            )

            # Fit target forecaster with actual X
            self.target_forecaster_.fit(
                y=y,
                X=X,
                forecasting_horizon=forecasting_horizon,
                **routed_params.target_forecaster.fit,
            )

        elif self.strategy == "rewind":
            # Fit feature forecaster on full data
            self.feature_forecaster_.fit(
                y=X,
                X=None,
                forecasting_horizon=forecasting_horizon,
                **routed_params.feature_forecaster.fit,
            )

            # Rewind feature forecaster to minimal observation horizon
            obs_horizon = self.feature_forecaster_.observation_horizon
            # Need obs_horizon + 1 rows: obs_horizon for transformer memory + 1 for transform
            rewind_size = obs_horizon + 1
            if obs_horizon <= 0 or len(X) <= rewind_size:
                raise ValueError(
                    f"Cannot use strategy='rewind': observation_horizon={obs_horizon} "
                    f"but len(X)={len(X)}. Need len(X) > observation_horizon + 1 to rewind."
                )
            self.feature_forecaster_.rewind(y=X[:rewind_size], X=None)

            # Predict X from rewind point to end
            X_pred = self.feature_forecaster_.predict(
                forecasting_horizon=len(X) - rewind_size,
            )

            # Prepare X for target forecaster (drop observed_time, keep time)
            X_for_target = X_pred.drop(["observed_time"], strict=False)

            # Fit target forecaster with predicted X
            self.target_forecaster_.fit(
                y=y[rewind_size:],
                X=X_for_target,
                forecasting_horizon=forecasting_horizon,
                **routed_params.target_forecaster.fit,
            )

            # Sync feature_forecaster to same observation time as target_forecaster
            self.feature_forecaster_.observe(y=X[rewind_size:], X=None)

        else:  # strategy == "predicted"
            n_split = int(len(y) * self.split_ratio)

            if n_split < 2:
                raise ValueError(
                    f"split_ratio={self.split_ratio} results in n_split={n_split}. "
                    f"Increase split_ratio or provide more data (len(y)={len(y)})."
                )
            if len(y) - n_split < 2:
                raise ValueError(
                    f"split_ratio={self.split_ratio} results in n_split={n_split}, "
                    f"leaving only {len(y) - n_split} rows for target forecaster. "
                    f"Decrease split_ratio to leave at least 2 rows."
                )

            # Fit feature forecaster on first portion
            self.feature_forecaster_.fit(
                y=X[:n_split],
                X=None,
                forecasting_horizon=len(y) - n_split,
                **routed_params.feature_forecaster.fit,
            )

            # Predict X for second portion
            X_pred = self.feature_forecaster_.predict(
                forecasting_horizon=len(y) - n_split,
            )

            # Prepare X for target forecaster (drop time columns)
            X_for_target = X_pred.drop(["observed_time"], strict=False)

            # Fit target forecaster on second portion with predicted X
            self.target_forecaster_.fit(
                y=y[n_split:],
                X=X_for_target,
                forecasting_horizon=forecasting_horizon,
                **routed_params.target_forecaster.fit,
            )

            # Sync feature_forecaster to the end of training data so both forecasters
            # share the same observed_time_. We feed actual X (not predicted) because:
            # 1. observe() updates the observation buffer/clock, not the model weights.
            # 2. At predict time and during rolling observe_predict, the feature
            #    forecaster should always base its state on actuals.
            # 3. The "predicted" strategy only governs what X the target_forecaster was
            #    *trained* on: the feature_forecaster's runtime state should track reality.
            # TODO: This might be more complicated than just predict+observe. It depends on
            # how predictions are going to be made (e.g., rolling vs single-shot, whether
            # actual X becomes available after prediction, etc.)
            self.feature_forecaster_.observe(y=X[n_split:], X=None)

        # Store standard fitted attributes
        self.fit_forecasting_horizon_ = forecasting_horizon
        self.interval_ = self.target_forecaster_.interval_
        self.panel_group_names_ = self.target_forecaster_.panel_group_names_
        if hasattr(self.target_forecaster_, "local_y_schema_"):
            self.local_y_schema_ = self.target_forecaster_.local_y_schema_
        if hasattr(self.target_forecaster_, "local_X_schema_"):
            self.local_X_schema_ = self.target_forecaster_.local_X_schema_
        else:
            # Build local_X_schema_ from X columns (excluding time)
            self.local_X_schema_ = {col: X.schema[col] for col in X.columns if col != "time"}
        self.shared_X_schema_ = None  # No shared X features for this forecaster

        # Set transformed schema attributes (no transformation for meta-forecaster)
        self.local_y_t_schema_ = self.local_y_schema_
        self.local_X_t_schema_ = self.local_X_schema_

        # Set observation buffers for observe/rewind
        self._y_observed = y
        self._X_t_observed = X

        return self

    @available_if(_target_forecaster_has("predict"))
    def predict(
        self,
        forecasting_horizon: StrictInt | None = None,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Generate point forecasts.

        Forecasts X using feature_forecaster, then uses those predictions
        as exogenous features for target_forecaster to predict y.

        Parameters
        ----------
        forecasting_horizon : int, optional
            Number of steps ahead to forecast. If None, uses value from fit().
        X : pl.DataFrame, optional
            Known-ahead exogenous features (e.g., holidays, promotions) that don't
            need forecasting. These will be merged with the forecasted features.
            Must have "time" column matching the forecast period.
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data prediction.
        predict_transformed : bool, default=False
            If True, return predictions in the transformed space without
            applying inverse target transformation.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Predictions with "observed_time", "time", and target columns.

        """
        check_is_fitted(self, ["target_forecaster_", "feature_forecaster_"])

        if forecasting_horizon is None:
            forecasting_horizon = self.fit_forecasting_horizon_

        # Validate params before routing
        _raise_for_params(params, self, "predict")

        # Process metadata routing
        routed_params = process_routing(self, "predict", **params)

        # Forecast X using feature forecaster
        X_pred = self.feature_forecaster_.predict(
            forecasting_horizon=forecasting_horizon,
            panel_group_names=panel_group_names,
            **routed_params.feature_forecaster.predict,
        )

        # Prepare X for target forecaster (drop observed_time, keep time)
        X_for_target = X_pred.drop(["observed_time"], strict=False)

        # Merge with known-ahead features from X if provided
        if X is not None:
            # Add columns from X that aren't already in X_for_target (excluding time)
            known_ahead_cols = [c for c in X.columns if c != "time" and c not in X_for_target.columns]
            if known_ahead_cols:
                X_for_target = X_for_target.join(
                    X.select(["time", *known_ahead_cols]),
                    on="time",
                    how="left",
                )

        # Predict y using forecasted X
        return self.target_forecaster_.predict(
            forecasting_horizon=forecasting_horizon,
            X=X_for_target,
            panel_group_names=panel_group_names,
            predict_transformed=predict_transformed,
            **routed_params.target_forecaster.predict,
        )

    @available_if(_target_forecaster_has("predict_interval"))
    def predict_interval(
        self,
        forecasting_horizon: StrictInt | None = None,
        X: pl.DataFrame | None = None,
        coverage_rates: list[float] | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate interval forecasts.

        Only available if target_forecaster supports interval predictions.
        Feature forecaster always produces point predictions for X.

        Parameters
        ----------
        forecasting_horizon : int, optional
            Number of steps ahead to forecast. If None, uses value from fit().
        X : pl.DataFrame, optional
            Known-ahead exogenous features (e.g., holidays, promotions) that don't
            need forecasting. These will be merged with the forecasted features.
            Must have "time" column matching the forecast period.
        coverage_rates : list of float, optional
            Coverage levels for prediction intervals (e.g., [0.9, 0.95]).
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data prediction.
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

        # Forecast X using feature forecaster (always point predictions)
        X_pred = self.feature_forecaster_.predict(
            forecasting_horizon=forecasting_horizon,
            panel_group_names=panel_group_names,
            **routed_params.feature_forecaster.predict,
        )

        # Prepare X for target forecaster
        X_for_target = X_pred.drop(["observed_time"], strict=False)

        # Merge with known-ahead features from X if provided
        if X is not None:
            # Add columns from X that aren't already in X_for_target (excluding time)
            known_ahead_cols = [c for c in X.columns if c != "time" and c not in X_for_target.columns]
            if known_ahead_cols:
                X_for_target = X_for_target.join(
                    X.select(["time", *known_ahead_cols]),
                    on="time",
                    how="left",
                )

        # Predict interval for y using forecasted X
        return self.target_forecaster_.predict_interval(
            forecasting_horizon=forecasting_horizon,
            X=X_for_target,
            coverage_rates=coverage_rates,
            panel_group_names=panel_group_names,
            **routed_params.target_forecaster.predict_interval,
        )

    def observe(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
    ) -> ForecastedFeatureForecaster:
        """Observe new data for both forecasters.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations with "time" column.
        X : pl.DataFrame, optional
            New exogenous feature observations with "time" column.
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data.

        Returns
        -------
        self
            Forecaster with new observations incorporated.

        """
        check_is_fitted(self, ["target_forecaster_", "feature_forecaster_"])

        # Observe new X for feature forecaster (X is y for feature forecaster)
        if X is not None:
            self.feature_forecaster_.observe(
                y=X,
                X=None,
                panel_group_names=panel_group_names,
            )

        # Observe new y and X for target forecaster
        self.target_forecaster_.observe(
            y=y,
            X=X,
            panel_group_names=panel_group_names,
        )

        return self

    def rewind(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
    ) -> ForecastedFeatureForecaster:
        """Rewind both forecasters to last observation_horizon rows.

        Parameters
        ----------
        y : pl.DataFrame
            Target data to rewind to (last observation_horizon rows kept).
        X : pl.DataFrame, optional
            Exogenous features to rewind to.
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data.

        Returns
        -------
        self
            Rewound forecaster.

        """
        check_is_fitted(self, ["target_forecaster_", "feature_forecaster_"])

        # Rewind feature forecaster (X is y for feature forecaster)
        if X is not None:
            self.feature_forecaster_.rewind(
                y=X,
                X=None,
                panel_group_names=panel_group_names,
            )

        # Rewind target forecaster
        self.target_forecaster_.rewind(
            y=y,
            X=X,
            panel_group_names=panel_group_names,
        )

        return self

    @available_if(_target_forecaster_has("predict"))
    def observe_predict(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Observe new data and generate point forecasts.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations with "time" column.
        X : pl.DataFrame, optional
            New exogenous feature observations with "time" column.
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Point predictions with "observed_time", "time", and target columns.

        """
        self.observe(y=y, X=X, panel_group_names=panel_group_names)
        return self.predict(panel_group_names=panel_group_names, **params)

    @available_if(_target_forecaster_has("predict_interval"))
    def observe_predict_interval(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        coverage_rates: list[float] | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Observe new data and generate interval forecasts.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations with "time" column.
        X : pl.DataFrame, optional
            New exogenous feature observations with "time" column.
        coverage_rates : list of float, optional
            Coverage levels for prediction intervals.
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Interval predictions with lower/upper bounds.

        """
        self.observe(y=y, X=X, panel_group_names=panel_group_names)
        return self.predict_interval(
            coverage_rates=coverage_rates,
            panel_group_names=panel_group_names,
            **params,
        )

    @available_if(_target_forecaster_has("predict_class_proba"))
    def predict_class_proba(
        self,
        forecasting_horizon: StrictInt | None = None,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate class-probability forecasts.

        Only available if target_forecaster supports class-probability predictions.
        Feature forecaster always produces point predictions for X.

        Parameters
        ----------
        forecasting_horizon : int, optional
            Number of steps ahead to forecast. If None, uses value from fit().
        X : pl.DataFrame, optional
            Known-ahead exogenous features that don't need forecasting.
            Must have "time" column matching the forecast period.
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data prediction.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Class-probability predictions with "observed_time", "time", and
            probability columns.

        """
        check_is_fitted(self, ["target_forecaster_", "feature_forecaster_"])

        if forecasting_horizon is None:
            forecasting_horizon = self.fit_forecasting_horizon_

        _raise_for_params(params, self, "predict_class_proba")
        routed_params = process_routing(self, "predict_class_proba", **params)

        X_pred = self.feature_forecaster_.predict(
            forecasting_horizon=forecasting_horizon,
            panel_group_names=panel_group_names,
            **routed_params.feature_forecaster.predict,
        )

        X_for_target = X_pred.drop(["observed_time"], strict=False)

        if X is not None:
            known_ahead_cols = [c for c in X.columns if c != "time" and c not in X_for_target.columns]
            if known_ahead_cols:
                X_for_target = X_for_target.join(
                    X.select(["time", *known_ahead_cols]),
                    on="time",
                    how="left",
                )

        return self.target_forecaster_.predict_class_proba(
            forecasting_horizon=forecasting_horizon,
            X=X_for_target,
            panel_group_names=panel_group_names,
            **routed_params.target_forecaster.predict_class_proba,
        )

    @available_if(_target_forecaster_has("predict_class_proba"))
    def observe_predict_class_proba(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Observe new data and generate class-probability forecasts.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations with "time" column.
        X : pl.DataFrame, optional
            New exogenous feature observations with "time" column.
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Class-probability predictions.

        """
        self.observe(y=y, X=X, panel_group_names=panel_group_names)
        return self.predict_class_proba(
            panel_group_names=panel_group_names,
            **params,
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
