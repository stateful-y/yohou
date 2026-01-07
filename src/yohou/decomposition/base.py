"""Abstract base class for trend and seasonality forecasters."""

import numbers
from abc import abstractmethod

import polars as pl
from pydantic import StrictInt

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

    """

    _parameter_constraints: dict = {
        "target_transformer": [BaseTransformer, None],
    }

    def __init__(self, target_transformer=None):
        """Initialize _BaseTrendForecaster.

        Parameters
        ----------
        target_transformer : BaseTransformer, optional
            Transformer for target variable.

        """
        super().__init__(target_transformer=target_transformer, input_features="X")

    def _get_time_indices(self, forecasting_horizon: int | None = None) -> pl.Series:
        """Generate indices for future predictions.

        Continues from current position (_y_observed length) and wraps around
        seasonal cycle.

        Parameters
        ----------
        forecasting_horizon : int
            Number of steps to predict.

        Returns
        -------
        pl.Series
            Phase indices for next forecasting_horizon steps.

        """
        # Get future time indices
        current_time_index = pl.datetime_range(
            start=self._first_observed_time,
            end=self.observed_time_,
            interval=self.interval_,
            eager=True,
        ).len()

        print("Current time point:", current_time_index)

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

        print(indices)

        return indices

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

        self._first_observed_time = y_t["time"][0]

        return y_t, X_t

    def reset(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> "_BaseTrendForecaster":
        """Resets the forecaster by resetting the observation horizon.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None
            Feature time series.

        Returns
        -------
        self

        """
        super().reset(y=y, X=X)

        self._first_observed_time = y["time"][0]
        if self.target_transformer_ is not None:
            target_observation_horizon = self.target_transformer_.observation_horizon
            if target_observation_horizon > 0:
                self._first_observed_time = y["time"][target_observation_horizon]

        return self


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

    """

    _parameter_constraints: dict = {
        "seasonality": [numbers.Real],
        "target_transformer": [BaseTransformer, None],
    }

    def __init__(self, seasonality: float, target_transformer=None):
        """Initialize _BaseSeasonalityForecaster.

        Parameters
        ----------
        seasonality : int
            Length of seasonal cycle.
        target_transformer : BaseTransformer, optional
            Transformer for target variable.

        """
        super().__init__(target_transformer=target_transformer)
        self.seasonality = seasonality

    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "BasePointForecaster":
        """Fits the forecaster and returns it.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None, default=None
            Exogenous feature time series.
        forecasting_horizon : int >= 1, default=1
            Horizon to forecast.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self

        """
        self._observation_horizon = self.seasonality

        BasePointForecaster.fit(
            self,
            y=y,
            X=X,
            forecasting_horizon=forecasting_horizon,
            **params,
        )

        return self

    def _validate_sufficient_data(self, y: pl.DataFrame) -> None:
        """Validate that y has at least one complete seasonal cycle.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        Raises
        ------
        ValueError
            If y has fewer than seasonality rows.

        """
        if len(y) < self.seasonality:
            raise ValueError(
                f"Insufficient data: need at least {self.seasonality} observations "
                f"(one seasonal cycle), got {len(y)}"
            )

    def _time_to_phase(self, time_col: pl.Series) -> pl.Series:
        """Convert time column to seasonal phase indices.

        Handles irregular intervals by computing phase based on row position
        relative to first observation.

        Parameters
        ----------
        time_col : pl.Series
            Time column (datetime type).

        Returns
        -------
        pl.Series
            Integer phase indices in range [0, seasonality).

        """
        # Compute row indices relative to first observation
        row_indices = pl.arange(0, len(time_col), eager=True)
        # Map to seasonal phases with wrap-around
        phases = row_indices % self.seasonality
        return phases

    @abstractmethod
    def _extract_pattern(self, y: pl.DataFrame):
        """Extract seasonal pattern from training data.

        Must be implemented by subclasses.

        Parameters
        ----------
        y : pl.DataFrame
            Transformed target time series.

        Returns
        -------
        Any
            Seasonal pattern representation (subclass-specific).

        """
        pass

    @abstractmethod
    def _predict_from_pattern(self, forecasting_horizon: int) -> pl.DataFrame:
        """Generate predictions from stored seasonal pattern.

        Must be implemented by subclasses.

        Parameters
        ----------
        forecasting_horizon : int
            Number of steps to predict.

        Returns
        -------
        pl.DataFrame
            Predictions for next forecasting_horizon steps.

        """
        pass

    def _predict_one(self) -> pl.DataFrame:
        """Generate one prediction.

        Returns
        -------
        pl.DataFrame
            Predictions with time columns added.

        """
        y_pred = self._predict_from_pattern(forecasting_horizon=self.fit_forecasting_horizon_)
        return y_pred
