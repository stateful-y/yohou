"""Abstract base class for trend and seasonality forecasters."""

import numbers

import numpy as np
import polars as pl
from pydantic import StrictInt
from sklearn.base import RegressorMixin, clone

from yohou.base import BaseTransformer
from yohou.point_forecaster.base import BasePointForecaster


class _BaseTrendForecaster(BasePointForecaster):
    """Abstract base class for trend forecasters.

    Provides common infrastructure for trend-based forecasting methods,
    including data validation and one-step-ahead prediction interface.

    Parameters
    ----------
    target_transformer : BaseTransformer, optional
        Transformer applied to target before forecasting.
    model_panel : bool, default=False
        Strategy for panel data modeling.

        - If False (default): Pooled strategy - treats all panel groups as
        independent samples for a single global model. Shares parameters
        across series. Lower memory usage. Best when series have similar
        patterns or limited per-series data.

        - If True: Per-group strategy - fits separate model for each panel
        group. Captures group-specific dynamics. Higher memory usage.
        Best when series have distinct patterns and sufficient per-group data.

        Ignored for global (non-panel) data.

    """

    _parameter_constraints: dict = {
        **BasePointForecaster._parameter_constraints,
        "model_panel": [bool],
    }

    def __init__(self, target_transformer=None, model_panel=False):
        """Initialize _BaseTrendForecaster.

        Parameters
        ----------
        target_transformer : BaseTransformer, optional
            Transformer for target variable.
        model_panel : bool, default=False
            Strategy for panel data modeling.
            - If False (default): Pooled strategy - treats all panel groups as
            independent samples for a single global model. Shares parameters
            across series. Lower memory usage. Best when series have similar
            patterns or limited per-series data.
            - If True: Per-group strategy - fits separate model for each panel
            group. Captures group-specific dynamics. Higher memory usage.
            Best when series have distinct patterns and sufficient per-group data.
            Ignored for global (non-panel) data.

        """
        super().__init__(target_transformer=target_transformer, input_features="X")

        self.model_panel = model_panel

    def _pre_fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
    ) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
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
        y_t : pl.DataFrame or None
            Transformed target.
        X_t : pl.DataFrame or None
            Transformed features.

        """

        y_t, X_t = super()._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)

        # Panel data
        if self.panel_group_names_ is not None:
            self._first_observed_time = {
                group: y_t[group]["time"][0] for group in self.panel_group_names_
            }

        # Non-panel data
        else:
            self._first_observed_time = y_t["time"][0]

        return y_t, X_t

    def reset(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        panel_group_names: list[str] | None = None,
    ) -> "_BaseTrendForecaster":
        """Resets the forecaster by resetting the observation horizon.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None
            Feature time series.
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data:
            - If None: predict for all groups
            - If list of str: predict only for the specified panel groups
            Parameter is ignored if the forecaster was not fitted on panel data.

        Returns
        -------
        self

        """
        super().reset(y=y, X=X, panel_group_names=panel_group_names)

        if panel_group_names is None:
            panel_group_names = self.panel_group_names_

        target_observation_horizon = 0
        if self.target_transformer_ is not None:
            if isinstance(self.target_transformer_, dict):
                first_target_transformer = next(iter(self.target_transformer_.values()))
                if first_target_transformer is not None:
                    target_observation_horizon = first_target_transformer.observation_horizon
            else:
                target_observation_horizon = self.target_transformer_.observation_horizon

        # Panel data
        if panel_group_names is not None:
            first_observed_time = {
                group: y["time"][target_observation_horizon] for group in panel_group_names
            }
            self._first_observed_time |= first_observed_time

        # Non-panel data
        else:
            self._first_observed_time = y["time"][target_observation_horizon]

        return self

    def _get_time_indices(
        self, forecasting_horizon: int | None = None, panel_group_name: str | None = None
    ) -> pl.Series:
        """Generate indices for future predictions.

        Continues from current position (_y_observed length) and wraps around
        seasonal cycle.

        Parameters
        ----------
        forecasting_horizon : int
            Number of steps to predict.
        panel_group_name : str or None
            Panel group name for which to get time indices.

        Returns
        -------
        pl.Series
            Phase indices for next forecasting_horizon steps.

        """
        if panel_group_name is not None:
            observed_time = self.observed_time_[panel_group_name]
            first_observed_time = self._first_observed_time[panel_group_name]

        else:
            observed_time = self.observed_time_
            first_observed_time = self._first_observed_time

        current_time_index = pl.datetime_range(
            start=first_observed_time,
            end=observed_time,
            interval=self.interval_,
            eager=True,
        ).len()

        if forecasting_horizon is not None:
            indices = pl.arange(
                current_time_index,
                current_time_index + forecasting_horizon,
                eager=True,
            )

        else:
            indices = pl.arange(
                0,
                current_time_index,
                eager=True,
            )

        return indices

    def _prepare_data(self, y: pl.DataFrame, panel_group_name: str | None = None) -> tuple:
        """Prepare time indices and target array for a single group.

        Parameters
        ----------
        y : pl.DataFrame
            Target DataFrame for a single group.
        panel_group_name : str or None
            Panel group name for which to get time indices.

        Returns
        -------
        tuple
            (X_time_indices, y_array) ready for model fitting.
        """
        X_time_indices = (
            self._get_time_indices(panel_group_name=panel_group_name).to_numpy().reshape(-1, 1)
        )
        if panel_group_name is not None:
            y_array = (
                y[panel_group_name]
                .select([col for col in y[panel_group_name].columns if col != "time"])
                .to_numpy()
            )
        else:
            y_array = y.select([col for col in y.columns if col != "time"]).to_numpy()

        return X_time_indices, y_array

    def _fit_estimator(
        self, estimator: RegressorMixin, y_t: pl.DataFrame | dict[str, pl.DataFrame]
    ) -> None:
        # Non-panel data
        if self.panel_group_names_ is None:
            X_time_indices, y_array = self._prepare_data(y_t)
            self.estimator_ = clone(estimator)
            self.estimator_.fit(X_time_indices, y_array)

        # Panel data with per-group estimators
        elif self.model_panel:
            self.estimator_ = {}
            for panel_group_name in self.panel_group_names_:
                estimator_group = clone(estimator)
                X_time_indices, y_array = self._prepare_data(y_t, panel_group_name=panel_group_name)
                estimator_group.fit(X_time_indices, y_array)
                self.estimator_[panel_group_name] = estimator_group
        # Panel data with pooled estimator
        else:
            X_time_indices, y_array = [], []
            for panel_group_name in self.panel_group_names_:
                X_time_indices_group, y_array_group = self._prepare_data(
                    y_t, panel_group_name=panel_group_name
                )
                X_time_indices.append(X_time_indices_group)
                y_array.append(y_array_group)

            # Stack all groups
            X_time_indices = np.vstack(X_time_indices)
            y_array = np.vstack(y_array)

            self.estimator_ = clone(estimator)
            self.estimator_.fit(X_time_indices, y_array)

    def _predict_one(
        self,
        panel_group_names: list[str],
    ) -> pl.DataFrame:
        """Predicts `_fit_forecasting_horizon` steps from the observation horizon.

        Parameters
        ----------
        panel_group_names : list of str
            Panel group names to predict for.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        y_t_columns = list(self.local_y_t_schema_.keys())

        # Non-panel data
        if self.panel_group_names_ is None:
            X_time_indices_pred = (
                self._get_time_indices(self.fit_forecasting_horizon_).to_numpy().reshape(-1, 1)
            )
            y_pred_array = self.estimator_.predict(X_time_indices_pred)
            y_pred_array = y_pred_array.reshape(-1, len(y_t_columns))

            # Convert to polars DataFrame with correct column names
            y_pred = pl.DataFrame({col: y_pred_array[:, i] for i, col in enumerate(y_t_columns)})

        # Panel data
        else:
            y_pred = []
            for panel_group_name in panel_group_names:
                X_time_indices_pred = (
                    self._get_time_indices(
                        self.fit_forecasting_horizon_, panel_group_name=panel_group_name
                    )
                    .to_numpy()
                    .reshape(-1, 1)
                )

                # Use per-group estimator
                if self.model_panel:
                    estimator_group = self.estimator_[panel_group_name]

                # Use pooled estimator
                else:
                    estimator_group = self.estimator_

                # Predict using model
                y_pred_array = estimator_group.predict(X_time_indices_pred)
                y_pred_array = y_pred_array.reshape(-1, len(y_t_columns))

                # Convert to polars DataFrame with unprefixed column names
                y_pred_group = pl.DataFrame(
                    {
                        f"{panel_group_name}__{col}": y_pred_array[:, i]
                        for i, col in enumerate(y_t_columns)
                    }
                )
                y_pred.append(y_pred_group)

            y_pred = pl.concat(y_pred, how="horizontal")

        y_pred = self._add_time_columns(y_pred)

        return y_pred


class _BaseSeasonalityForecaster(_BaseTrendForecaster):
    """Abstract base class for seasonality forecasters.

    Provides common infrastructure for pattern-based and Fourier-based
    seasonality forecasting, including time-to-phase conversion, phase
    tracking, and data validation.

    Parameters
    ----------
    seasonality : int
        Length of seasonal cycle (number of time steps).
    target_transformer : BaseTransformer, optional
        Transformer applied to target before forecasting.
    model_panel : bool, default=False
        Strategy for panel data modeling.
        - If False (default): Pooled strategy - treats all panel groups as
        independent samples for a single global model. Shares parameters
        across series. Lower memory usage. Best when series have similar
        patterns or limited per-series data.
        - If True: Per-group strategy - fits separate model for each panel
        group. Captures group-specific dynamics. Higher memory usage.
        Best when series have distinct patterns and sufficient per-group data.

        Ignored for global (non-panel) data.

    """

    _parameter_constraints: dict = {
        "seasonality": [numbers.Real],
        "target_transformer": [BaseTransformer, None],
        "model_panel": [bool],
    }

    def __init__(self, seasonality: float, target_transformer=None, model_panel=False):
        """Initialize _BaseSeasonalityForecaster.

        Parameters
        ----------
        seasonality : int
            Length of seasonal cycle.
        target_transformer : BaseTransformer, optional
            Transformer for target variable.
        model_panel : bool, default=False
            Strategy for panel data modeling.
            - If False (default): Pooled strategy - treats all panel groups as
            independent samples for a single global model. Shares parameters
            across series. Lower memory usage. Best when series have similar
            patterns or limited per-series data.
            - If True: Per-group strategy - fits separate model for each panel
            group. Captures group-specific dynamics. Higher memory usage.
            Best when series have distinct patterns and sufficient per-group data.

            Ignored for global (non-panel) data.

        """
        super().__init__(target_transformer=target_transformer, model_panel=model_panel)
        self.seasonality = seasonality

    def _validate_sufficient_data(self, y: pl.DataFrame | dict[str, pl.DataFrame]) -> None:
        """Validate that y has at least one complete seasonal cycle.

        Parameters
        ----------
        y : pl.DataFrame or dict[str, pl.DataFrame]
            Target time series (DataFrame for global data, dict for panel data).

        Raises
        ------
        ValueError
            If y has fewer than seasonality rows.

        """
        # Handle panel data (dict of DataFrames)
        if isinstance(y, dict):
            for group_name, group_df in y.items():
                if len(group_df) < self.seasonality:
                    raise ValueError(
                        f"Insufficient data for group '{group_name}': need at least "
                        f"{self.seasonality} observations (one seasonal cycle), got {len(group_df)}"
                    )
        # Handle global data (single DataFrame)
        else:
            if len(y) < self.seasonality:
                raise ValueError(
                    f"Insufficient data: need at least {self.seasonality} observations "
                    f"(one seasonal cycle), got {len(y)}"
                )
