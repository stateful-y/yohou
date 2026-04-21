"""Standard (non-panel) forecaster mixin for type-safe operations."""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs

from yohou.base.utils import (
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
    `BaseForecaster` : Main forecaster base combining standard and panel operations.
    `BasePanelForecaster` : Panel (multi-series) forecaster mixin.
    `BaseReductionForecaster` : Reduction-based forecaster using sklearn regressors.

    """

    # Type hints for attributes set by BaseForecaster
    target_transformer: "BaseTransformer | None"
    feature_transformer: "BaseTransformer | None"
    target_as_feature: str | None
    groups_: None
    local_y_schema_: dict[str, pl.DataType]
    local_X_schema_: dict[str, pl.DataType] | None
    shared_X_schema_: None
    observation_horizon: int
    observed_time_: datetime
    interval_: timedelta | str

    def _set_input_attributes_standard(self, y: pl.DataFrame, X: pl.DataFrame | None) -> None:
        """Set input attributes for standard (non-panel) data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series (standard data).
        X : pl.DataFrame or None
            Feature time series (standard data).

        """
        self.groups_ = None
        self.local_y_schema_ = dict(y.select(~cs.by_name("time")).schema)
        self.shared_X_schema_ = None

        self.local_X_schema_ = None
        if X is not None:
            self.local_X_schema_ = dict(X.select(~cs.by_name("time")).schema)

    def _fit_transform_inputs_standard(
        self, y: pl.DataFrame, X: pl.DataFrame | None
    ) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """Fit transformers and transform inputs for standard data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series (standard data).
        X : pl.DataFrame or None
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

        if X is not None and self.local_X_schema_ is not None:
            X = X.select(["time"] + list(self.local_X_schema_.keys()))

        y_t, X_t, target_transformer, feature_transformer = _fit_transform_transformers_one(
            y=y,
            X=X,
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
        X: pl.DataFrame | None,
        forecasting_horizon: int,
    ) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """Preprocessing and transform for standard data (narrow types).

        Parameters
        ----------
        y : pl.DataFrame
            Target time series (standard data, already validated).
        X : pl.DataFrame or None
            Feature time series (standard data, already validated).
        forecasting_horizon : int
            Number of steps ahead to forecast.

        Returns
        -------
        y_t : pl.DataFrame
            Transformed target.
        X_t : pl.DataFrame or None
            Transformed features.

        """
        self._set_input_attributes_standard(y, X)
        y_t, X_t = self._fit_transform_inputs_standard(y, X)
        self._set_transformed_attributes_standard(y_t, X_t)
        self._update_y_X_t_observed_standard(y, X_t, self.observation_horizon)

        return y_t, X_t

    def _rewind_standard(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None,
    ) -> "BaseStandardForecaster":
        """Reset state for standard (non-panel) data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series (standard data).
        X : pl.DataFrame or None
            Feature time series (standard data).

        Returns
        -------
        self

        """
        X_t = _rewind_transformers_one(
            y,
            X,
            self.target_transformer_,
            self.feature_transformer_,
            self.observation_horizon,
            self.target_as_feature,
        )

        self._update_y_X_t_observed_standard(y, X_t, self.observation_horizon)
        return self

    def _observe_standard(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None,
    ) -> "BaseStandardForecaster":
        """Update state with new observations for standard (non-panel) data.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations (standard data).
        X : pl.DataFrame or None
            New feature observations (standard data).

        Returns
        -------
        self

        """
        # Update transformers with only new data
        X_t_updated = _observe_transformers_one(
            y, X, self.target_transformer_, self.feature_transformer_, self.target_as_feature
        )

        # Prepare full y for state update (needs history to maintain observation_horizon)
        y_updated = y
        if self._y_observed is not None:
            y_updated = pl.concat([self._y_observed, y], how="vertical")

        # Update observed state using full history (tail)
        self._update_y_X_t_observed_standard(y_updated, X_t_updated, self.observation_horizon)

        return self

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
