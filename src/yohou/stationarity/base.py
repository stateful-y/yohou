"""Abstract base class for trend and seasonality forecasters."""

import numbers
from datetime import datetime
from typing import Literal
from typing import cast as typing_cast

import numpy as np
import polars as pl
from pydantic import StrictInt
from sklearn.base import RegressorMixin, clone
from sklearn.pipeline import Pipeline

from yohou.base import BaseTransformer
from yohou.point import BasePointForecaster
from yohou.utils._compat import StrOptions
from yohou.utils.tags import Tags


class _BaseTrendForecaster(BasePointForecaster):
    """Abstract base class for trend forecasters.

    Provides common infrastructure for trend-based forecasting methods,
    including data validation and one-step-ahead prediction interface.

    Parameters
    ----------
    target_transformer : BaseTransformer, optional
        Transformer applied to target before forecasting.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data.  See `BaseForecaster` for details.

    """

    _parameter_constraints: dict = {
        **BasePointForecaster._parameter_constraints,
    }

    def __init__(
        self,
        target_transformer: BaseTransformer | None = None,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        """Initialize _BaseTrendForecaster.

        Parameters
        ----------
        target_transformer : BaseTransformer, optional
            Transformer for target variable.
        panel_strategy : {"global", "multivariate"}, default="global"
            How to handle panel data.  See `BaseForecaster` for details.

        """
        super().__init__(
            target_transformer=target_transformer,
            target_as_feature=None,
            panel_strategy=panel_strategy,
        )

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.ignores_exogenous = True
        return tags

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

        """
        y_t, X_t = super()._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)

        # Panel data
        if self.groups_ is not None:
            assert isinstance(y_t, dict)
            self._first_observed_time = {group: y_t[group]["time"][0] for group in self.groups_}

        # Non-panel data
        else:
            assert isinstance(y_t, pl.DataFrame)
            self._first_observed_time = y_t["time"][0]

        return y_t, X_t

    def rewind(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        groups: list[str] | None = None,
    ) -> "_BaseTrendForecaster":
        """Rewinds the forecaster by rewinding the observation horizon.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None
            Feature time series.
        groups : list of str or None, default=None
            Group prefixes for panel data:
            - If None: predict for all groups
            - If list of str: predict only for the specified panel groups
            Parameter is ignored if the forecaster was not fitted on panel data.

        Returns
        -------
        self

        """
        super().rewind(y=y, X=X, groups=groups)

        if groups is None:
            groups = self.groups_

        target_observation_horizon = 0
        if self.target_transformer_ is not None:
            if isinstance(self.target_transformer_, dict):
                first_target_transformer = next(iter(self.target_transformer_.values()))
                if first_target_transformer is not None:
                    target_observation_horizon = typing_cast(
                        BaseTransformer,
                        first_target_transformer,
                    ).observation_horizon
                else:
                    target_observation_horizon = 0
            else:
                target_observation_horizon = self.target_transformer_.observation_horizon

        # Panel data
        if groups is not None:
            first_observed_time = dict.fromkeys(groups, y["time"][target_observation_horizon])
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
            assert isinstance(self.observed_time_, dict)
            assert isinstance(self._first_observed_time, dict)
            observed_time = self.observed_time_[panel_group_name]
            first_observed_time = self._first_observed_time[panel_group_name]

        else:
            assert isinstance(self.observed_time_, datetime)
            assert isinstance(self._first_observed_time, datetime)
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

    def _prepare_data(self, y: pl.DataFrame | dict[str, pl.DataFrame], panel_group_name: str | None = None) -> tuple:
        """Prepare time indices and target array for a single group.

        Parameters
        ----------
        y : pl.DataFrame or dict[str, pl.DataFrame]
            Target data for a single group (DataFrame) or all groups (dict).
        panel_group_name : str or None
            Panel group name for which to get time indices.

        Returns
        -------
        tuple
            (X_time_indices, y_array) ready for model fitting.
        """
        X_time_indices = self._get_time_indices(panel_group_name=panel_group_name).to_numpy().reshape(-1, 1)
        if panel_group_name is not None:
            assert isinstance(y, dict)
            y_group = y[panel_group_name]
            assert isinstance(y_group, pl.DataFrame)
            y_array = y_group.select([col for col in y_group.columns if col != "time"]).to_numpy()
        else:
            assert isinstance(y, pl.DataFrame)
            y_array = y.select([col for col in y.columns if col != "time"]).to_numpy()

        return X_time_indices, y_array

    def _fit_estimator(self, estimator: RegressorMixin | Pipeline, y_t: pl.DataFrame | dict[str, pl.DataFrame]) -> None:
        """Fit the underlying estimator on prepared time series data.

        Parameters
        ----------
        estimator : RegressorMixin or Pipeline
            The sklearn estimator or pipeline to fit.
        y_t : pl.DataFrame or dict of str to pl.DataFrame
            Transformed target time series, either a single DataFrame or a dict of panel-group DataFrames.

        """
        # Non-panel data
        if self.groups_ is None:
            X_time_indices, y_array = self._prepare_data(y_t)
            self.estimator_ = clone(estimator)
            self.estimator_.fit(X_time_indices, y_array)

        # Panel data: pooled estimator (global strategy)
        else:
            X_time_indices, y_array = [], []
            for panel_group_name in self.groups_:
                X_time_indices_group, y_array_group = self._prepare_data(y_t, panel_group_name=panel_group_name)
                X_time_indices.append(X_time_indices_group)
                y_array.append(y_array_group)

            # Stack all groups
            X_time_indices = np.vstack(X_time_indices)
            y_array = np.vstack(y_array)

            self.estimator_ = clone(estimator)
            self.estimator_.fit(X_time_indices, y_array)

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
        y_t_columns = list(self.local_y_t_schema_.keys())

        # Non-panel data
        if self.groups_ is None:
            X_time_indices_pred = self._get_time_indices(self.fit_forecasting_horizon_).to_numpy().reshape(-1, 1)
            y_pred_array = self.estimator_.predict(X_time_indices_pred)
            y_pred_array = y_pred_array.reshape(-1, len(y_t_columns))

            # Convert to polars DataFrame with correct column names
            y_pred = pl.DataFrame({col: y_pred_array[:, i] for i, col in enumerate(y_t_columns)})

        # Panel data
        else:
            y_pred = []
            for panel_group_name in groups:
                X_time_indices_pred = (
                    self
                    ._get_time_indices(self.fit_forecasting_horizon_, panel_group_name=panel_group_name)
                    .to_numpy()
                    .reshape(-1, 1)
                )

                # Use pooled estimator for all groups
                estimator_group = self.estimator_

                # Predict using model
                y_pred_array = estimator_group.predict(X_time_indices_pred)
                y_pred_array = y_pred_array.reshape(-1, len(y_t_columns))

                # Convert to polars DataFrame with unprefixed column names
                y_pred_group = pl.DataFrame({
                    f"{panel_group_name}__{col}": y_pred_array[:, i] for i, col in enumerate(y_t_columns)
                })
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
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data.  See `BaseForecaster` for details.

    """

    _parameter_constraints: dict = {
        "seasonality": [numbers.Real],
        "target_transformer": [BaseTransformer, None],
        "panel_strategy": [StrOptions({"global", "multivariate"})],
    }

    def __init__(
        self,
        seasonality: float,
        target_transformer: BaseTransformer | None = None,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        """Initialize _BaseSeasonalityForecaster.

        Parameters
        ----------
        seasonality : int
            Length of seasonal cycle.
        target_transformer : BaseTransformer, optional
            Transformer for target variable.
        panel_strategy : {"global", "multivariate"}, default="global"
            How to handle panel data.  See `BaseForecaster` for details.

        """
        super().__init__(target_transformer=target_transformer, panel_strategy=panel_strategy)
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
                assert isinstance(group_df, pl.DataFrame)
                if len(group_df) < self.seasonality:
                    raise ValueError(
                        f"Insufficient data for group '{group_name}': need at least "
                        f"{self.seasonality} observations (one seasonal cycle), got {len(group_df)}"
                    )
        # Handle global data (single DataFrame)
        elif len(y) < self.seasonality:
            raise ValueError(
                f"Insufficient data: need at least {self.seasonality} observations (one seasonal cycle), got {len(y)}"
            )
