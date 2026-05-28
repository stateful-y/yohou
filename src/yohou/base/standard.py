"""Standard (non-panel) forecaster mixin for type-safe operations."""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs

from yohou.base.utils import (
    _derive_step_columns,
    _fit_transform_transformers_one,
    _observe_transformers_one,
    _rewind_transformers_one,
)
from yohou.utils import add_interval

if TYPE_CHECKING:
    from yohou.base.transformer import BaseTransformer

__all__ = ["BaseStandardForecaster"]


class BaseStandardForecaster:
    """Mixin providing standard (single DataFrame) forecaster operations.

    This mixin provides methods with narrow return types for standard data
    (pl.DataFrame). Child classes that need type narrowing can explicitly
    call these methods via `BaseStandardForecaster._pre_fit_standard(self, ...)`.

    See Also
    --------
    - [`BaseForecaster`][yohou.base.forecaster.BaseForecaster] : Main forecaster base combining standard and panel operations.
    - [`BasePanelForecaster`][yohou.base.panel.BasePanelForecaster] : Panel (multi-series) forecaster mixin.
    - [`BaseReductionForecaster`][yohou.base.reduction.BaseReductionForecaster] : Reduction-based forecaster using sklearn regressors.

    """

    # Type hints for attributes set by BaseForecaster
    target_transformer: "BaseTransformer | None"
    feature_transformer: "BaseTransformer | None"
    target_as_feature: str | None
    groups_: None
    local_y_schema_: dict[str, pl.DataType]
    local_X_actual_schema_: dict[str, pl.DataType] | None
    shared_X_actual_schema_: None
    observation_horizon: int
    observed_time_: datetime
    interval_: timedelta | str

    def _set_input_attributes_standard(self, y: pl.DataFrame, X_actual: pl.DataFrame | None) -> None:
        """Set input attributes for standard (non-panel) data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series (standard data).
        X_actual : pl.DataFrame or None
            Feature time series (standard data).

        """
        self.groups_ = None
        self.local_y_schema_ = dict(y.select(~cs.by_name("time")).schema)
        self.shared_X_actual_schema_ = None

        self.local_X_actual_schema_ = None
        if X_actual is not None:
            self.local_X_actual_schema_ = dict(X_actual.select(~cs.by_name("time")).schema)

    def _fit_transform_inputs_standard(
        self, y: pl.DataFrame, X_actual: pl.DataFrame | None
    ) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """Fit transformers and transform inputs for standard data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series (standard data).
        X_actual : pl.DataFrame or None
            Feature time series (standard data).

        Returns
        -------
        y_t : pl.DataFrame
            Transformed target.
        X_t : pl.DataFrame or None
            Transformed features.

        """
        # Standard data: schemas contain actual column names
        y = y.select(["time"] + list(self.local_y_schema_.keys()))

        if X_actual is not None and self.local_X_actual_schema_ is not None:
            X_actual = X_actual.select(["time"] + list(self.local_X_actual_schema_.keys()))

        y_t, X_t, target_transformer, feature_transformer = _fit_transform_transformers_one(
            y=y,
            X_actual=X_actual,
            target_transformer=self.target_transformer,
            feature_transformer=self.feature_transformer,
            target_as_feature=self.target_as_feature,
        )

        self.target_transformer_ = target_transformer
        self.feature_transformer_ = feature_transformer

        return y_t, X_t

    def _set_transformed_attributes_standard(
        self,
        y_t: pl.DataFrame,
        X_t: pl.DataFrame | None,
    ) -> None:
        """Set transformed data attributes for standard data.

        Parameters
        ----------
        y_t : pl.DataFrame
            Transformed target (standard data).
        X_t : pl.DataFrame or None
            Transformed features (standard data).

        """
        self.local_y_t_schema_ = dict(y_t.select(~cs.by_name("time")).schema)

        if X_t is not None:
            self.local_X_t_schema_ = dict(X_t.select(~cs.by_name("time")).schema)
        else:
            self.local_X_t_schema_ = None

        # Store n_features_in_ and feature_names_in_ for sklearn compatibility
        if self.local_X_t_schema_:
            self.n_features_in_ = len(self.local_X_t_schema_)
            self.feature_names_in_ = list(self.local_X_t_schema_.keys())
        else:
            self.n_features_in_ = 0
            self.feature_names_in_ = []

    def _update_y_X_t_observed_standard(
        self,
        y: pl.DataFrame,
        X_t: pl.DataFrame | None,
        observation_horizon: int,
    ) -> None:
        """Update stored observed data for standard data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series (untransformed, standard data).
        X_t : pl.DataFrame or None
            Transformed feature time series (standard data).
        observation_horizon : int
            Number of time steps to retain.

        """
        self.observed_time_ = y["time"][-1]

        self._X_t_observed = None
        if X_t is not None:
            self._X_t_observed = X_t[[-1]]

        # Store untransformed data for inverse_transform
        y_observed = None
        if observation_horizon > 0:
            if observation_horizon > len(y):
                raise ValueError("Not enough data to set observed y.")
            y_observed = y[-observation_horizon:]

        self._y_observed = y_observed

    def _pre_fit_standard(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None,
        forecasting_horizon: int,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """Preprocessing and transform for standard data (narrow types).

        Parameters
        ----------
        y : pl.DataFrame
            Target time series (standard data, already validated).
        X_actual : pl.DataFrame or None
            Feature time series (standard data, already validated).
        forecasting_horizon : int
            Number of steps ahead to forecast.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"`` columns.

        Returns
        -------
        y_t : pl.DataFrame
            Transformed target.
        X_t : pl.DataFrame or None
            Transformed features.

        """
        self._set_input_attributes_standard(y, X_actual)
        y_t, X_t = self._fit_transform_inputs_standard(y, X_actual)

        # Inject step columns from X_future / X_forecast
        X_step = _derive_step_columns(
            X_future=X_future,
            X_forecast=X_forecast,
            observation_times=y_t["time"],
            forecasting_horizon=forecasting_horizon,
            interval=self.interval_,
            existing_columns=set(X_t.columns) - {"time"} if X_t is not None else None,
        )
        if X_step is not None:
            self._step_column_names_ = set(X_step.columns) - {"time"}
            self._X_future_raw_ = X_future
            self._X_forecast_raw_ = X_forecast
            self._X_future_schema_ = dict(X_future.select(~cs.by_name("time")).schema) if X_future is not None else None
            self._X_forecast_schema_ = (
                dict(X_forecast.select(~cs.by_name("time", "vintage_time")).schema) if X_forecast is not None else None
            )
            if X_t is not None:
                X_t = X_t.join(X_step, on="time", how="left")
            else:
                X_t = X_step.join(y_t.select("time"), on="time", how="semi")
        else:
            self._step_column_names_ = set()
            self._X_future_raw_ = None
            self._X_forecast_raw_ = None
            self._X_future_schema_ = None
            self._X_forecast_schema_ = None

        self._set_transformed_attributes_standard(y_t, X_t)
        self._update_y_X_t_observed_standard(y, X_t, self.observation_horizon)

        return y_t, X_t

    def _rewind_standard(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> "BaseStandardForecaster":
        """Reset state for standard (non-panel) data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series (standard data).
        X_actual : pl.DataFrame or None
            Actual feature observations to restore the observation
            state to (standard data).
        X_future : pl.DataFrame or None, default=None
            Known future features. If None, re-derived from stored raws.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts. If None, re-derived from stored raws.

        Returns
        -------
        self

        """
        X_t = _rewind_transformers_one(
            y,
            X_actual,
            self.target_transformer_,
            self.feature_transformer_,
            self.observation_horizon,
            self.target_as_feature,
        )

        self._update_y_X_t_observed_standard(y, X_t, self.observation_horizon)

        # Re-derive step columns and append to single-row _X_t_observed
        self._inject_step_columns_after_update(X_future, X_forecast)

        return self

    def _observe_standard(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> "BaseStandardForecaster":
        """Update state with new observations for standard (non-panel) data.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations (standard data).
        X_actual : pl.DataFrame or None
            New actual feature observations (standard data).
        X_future : pl.DataFrame or None, default=None
            Known future features. If None, re-derived from stored raws.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts. If None, re-derived from stored raws.
            The latest vintage at or before ``observed_time_`` is
            selected (as-of matching), so vintage times do not need
            to align exactly with observation times.

        Returns
        -------
        self

        """
        # Update transformers with only new data (X_actual only, no step columns)
        X_t_updated = _observe_transformers_one(
            y, X_actual, self.target_transformer_, self.feature_transformer_, self.target_as_feature
        )

        # Prepare full y for state update (needs history to maintain observation_horizon)
        y_updated = y
        if self._y_observed is not None:
            y_updated = pl.concat([self._y_observed, y], how="vertical")

        # Update observed state using full history (tail)
        self._update_y_X_t_observed_standard(y_updated, X_t_updated, self.observation_horizon)

        # Re-derive step columns and append to single-row _X_t_observed
        self._inject_step_columns_after_update(X_future, X_forecast)

        return self

    def _inject_step_columns_after_update(
        self,
        X_future: pl.DataFrame | None,
        X_forecast: pl.DataFrame | None,
    ) -> None:
        """Re-derive step columns and append to _X_t_observed after state update.

        Uses stored raws as fallback when X_future or X_forecast is omitted.
        Updates stored raws when new data is provided.

        """
        if not self._step_column_names_:
            return

        X_future_eff = X_future if X_future is not None else self._X_future_raw_
        X_forecast_eff = X_forecast if X_forecast is not None else self._X_forecast_raw_

        X_step = _derive_step_columns(
            X_future_eff,
            X_forecast_eff,
            pl.Series([self.observed_time_]),
            self.fit_forecasting_horizon_,  # ty: ignore[unresolved-attribute]
            self.interval_,
        )
        if X_step is not None and self._X_t_observed is not None:
            self._X_t_observed = pl.concat(
                [self._X_t_observed, X_step.select(~cs.by_name("time"))],
                how="horizontal",
            )
        elif X_step is not None:
            # _X_t_observed was None (no transformer output), use step cols only
            self._X_t_observed = X_step.filter(pl.col("time") == self.observed_time_).select(~cs.by_name("time"))

        # Update stored raws
        if X_future is not None:
            self._X_future_raw_ = X_future
        if X_forecast is not None:
            # Select the latest vintage at or before observed_time_ (as-of selection)
            latest_vintage = X_forecast.filter(pl.col("vintage_time") <= self.observed_time_)["vintage_time"].max()
            if latest_vintage is not None:
                self._X_forecast_raw_ = X_forecast.filter(pl.col("vintage_time") == latest_vintage)
            else:
                self._X_forecast_raw_ = X_forecast.clear()

    def _observe_with_precomputed_steps_standard(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None,
        X_step_precomputed: pl.DataFrame | None,
    ) -> None:
        """Observe with pre-computed step columns (avoids re-pivoting in loops).

        Used by ``_observe_predict_loop`` to inject step columns that were
        derived once at loop entry instead of re-deriving per stride.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations (standard data).
        X_actual : pl.DataFrame or None
            New actual feature observations (standard data).
        X_step_precomputed : pl.DataFrame or None
            Pre-computed step columns for this slice (already semi-joined
            to the slice's time range). ``None`` when no step columns exist.

        """
        X_t_updated = _observe_transformers_one(
            y, X_actual, self.target_transformer_, self.feature_transformer_, self.target_as_feature
        )

        y_updated = y
        if self._y_observed is not None:
            y_updated = pl.concat([self._y_observed, y], how="vertical")

        # Hstack step columns BEFORE storage: both X_t_updated and
        # X_step_precomputed have len(y_slice) rows (same time range).
        if X_t_updated is not None and X_step_precomputed is not None:
            X_t_updated = pl.concat(
                [X_t_updated, X_step_precomputed.select(~cs.by_name("time"))],
                how="horizontal",
            )
        elif X_t_updated is None and X_step_precomputed is not None:
            X_t_updated = X_step_precomputed.select(~cs.by_name("time"))

        self._update_y_X_t_observed_standard(y_updated, X_t_updated, self.observation_horizon)

    def _add_time_columns_standard(self, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Add time metadata columns to predictions for standard data.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Predictions without time columns.

        Returns
        -------
        pl.DataFrame
            Predictions with vintage_time and time columns.

        """
        predicted_times = [add_interval(self.observed_time_, self.interval_, n=n) for n in range(1, len(y_pred) + 1)]

        time = pl.DataFrame({"vintage_time": [self.observed_time_] * len(y_pred), "time": predicted_times})

        return pl.concat([time, y_pred], how="horizontal")
