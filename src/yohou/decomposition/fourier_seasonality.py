"""Fourier series seasonality forecasting implementation."""

import numbers

import numpy as np
import polars as pl
from pydantic import StrictInt
from sklearn.base import _fit_context
from sklearn.linear_model import ElasticNet
from sklearn.utils._param_validation import Interval

from .base import _BaseSeasonalityForecaster


class FourierSeasonalityForecaster(_BaseSeasonalityForecaster):
    """Forecast using Fourier series representation of seasonality.

    Represents seasonality using Fourier series with specified harmonics,
    fitted via ElasticNet regression. More flexible than pattern-based
    methods and can handle non-integer seasonality.

    Parameters
    ----------
    seasonality : float
        Seasonal period length (can be non-integer, e.g., 365.25 for yearly).
    harmonics : list of int, default=[1, 2, 3]
        List of Fourier harmonics to use (e.g., [1, 2, 3] uses first 3 harmonics).
    alpha : float, default=1.0
        Constant that multiplies the penalty terms (L1 + L2 regularization strength).
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter (0 <= l1_ratio <= 1).
        l1_ratio=0 is equivalent to Ridge, l1_ratio=1 is equivalent to Lasso.
    target_transformer : BaseTransformer, optional
        Transformer for target variable.

    Attributes
    ----------
    model_ : ElasticNet
        Fitted ElasticNet model for all columns.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> from datetime import datetime
    >>> from yohou.point_forecaster import FourierSeasonalityForecaster
    >>>
    >>> # Create time series with sinusoidal seasonality
    >>> time_range = pl.datetime_range(
    ...     start=datetime(2020, 1, 1),
    ...     end=datetime(2020, 2, 29),
    ...     interval="1d",
    ...     eager=True
    ... )
    >>> y = pl.DataFrame({
    ...     "time": time_range,
    ...     "value": [np.sin(2 * np.pi * i / 12) for i in range(len(time_range))]
    ... })
    >>>
    >>> # Fit Fourier seasonality forecaster
    >>> forecaster = FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2, 3])
    >>> forecaster.fit(y, forecasting_horizon=30)
    FourierSeasonalityForecaster(seasonality=12)
    >>>
    >>> # Forecast next 30 days
    >>> y_pred = forecaster.predict(forecasting_horizon=30)

    Notes
    -----
    - Handles non-integer seasonality (e.g., 365.25 days/year)
    - Produces smooth seasonal curves
    - Can represent multiple seasonalities by using more harmonics
    - Unlike pattern-based methods, representation is continuous and differentiable

    """

    _parameter_constraints: dict = {
        **_BaseSeasonalityForecaster._parameter_constraints,
        "harmonics": [list],
        "alpha": [Interval(numbers.Real, 0, None, closed="left")],
        "l1_ratio": [Interval(numbers.Real, 0, 1, closed="both")],
    }

    def __init__(
        self,
        seasonality: float,
        harmonics: list[int] | None = None,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        target_transformer=None,
    ):
        # Call parent with int seasonality for base class validation
        super().__init__(seasonality=int(seasonality), target_transformer=target_transformer)
        # Store actual (potentially non-integer) seasonality
        self.seasonality = seasonality
        self.harmonics = harmonics if harmonics is not None else [1, 2, 3]
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "FourierSeasonalityForecaster":
        """Fit Fourier series model to historical data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with "time" column.
        X : pl.DataFrame, optional
            Exogenous features (currently not used, reserved for future).
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.
        **params : dict
            Additional metadata (routed via sklearn's metadata routing).

        Returns
        -------
        self
            Fitted forecaster.

        """
        # Domain-specific validation: harmonics must be positive and not exceed
        # seasonality/2 (Nyquist limit)
        if not self.harmonics:
            raise ValueError("harmonics list cannot be empty")
        if any(h < 1 for h in self.harmonics):
            raise ValueError("All harmonics must be positive integers")
        max_harmonic = max(self.harmonics)
        if max_harmonic > self.seasonality / 2:
            raise ValueError(
                f"Maximum harmonic ({max_harmonic}) cannot exceed seasonality/2 "
                f"({self.seasonality / 2:.1f}) due to Nyquist sampling theorem."
            )

        # Pre-fit: validate inputs, apply target transformer, set attributes
        y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)

        # Validate sufficient data (at least one cycle)
        self._validate_sufficient_data(y_t)

        # Fit single ElasticNet model for all columns
        self.model_ = self._extract_pattern(y_t)
        self._column_names = [col for col in y_t.columns if col != "time"]

        return self

    def _extract_pattern(self, y: pl.DataFrame) -> ElasticNet:
        """Fit single ElasticNet model for all columns.

        Parameters
        ----------
        y : pl.DataFrame
            Transformed target time series.

        Returns
        -------
        ElasticNet
            Fitted ElasticNet model.

        """
        time_col = pl.arange(0, len(y), eager=True)

        # Build Fourier feature matrix
        X_fourier = self._build_fourier_features(time_col)

        # Stack all target columns
        y_values = y.select([col for col in y.columns if col != "time"]).to_numpy()

        # Fit single ElasticNet model for all columns
        model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=10000)
        model.fit(X_fourier, y_values)

        return model

    def _build_fourier_features(self, time_indices: pl.Series) -> np.ndarray:
        """Construct Fourier feature matrix.

        Parameters
        ----------
        time_indices : pl.Series
            Time step indices.

        Returns
        -------
        np.ndarray
            Shape (n_samples, 2 * len(harmonics)) with sin/cos features.

        """
        t = time_indices.to_numpy()
        features = []

        for k in self.harmonics:
            features.append(np.sin(2 * np.pi * k * t / self.seasonality))
            features.append(np.cos(2 * np.pi * k * t / self.seasonality))

        return np.column_stack(features)

    def _predict_from_pattern(self, forecasting_horizon: int, **params) -> pl.DataFrame:
        """Generate predictions using Fourier coefficients.

        Parameters
        ----------
        forecasting_horizon : int
            Number of steps to predict.
        **params : dict
            Additional parameters.

        Returns
        -------
        pl.DataFrame
            Predictions for next forecasting_horizon steps.

        """
        # Get future time indices
        current_time_index = pl.datetime_range(
            start=self._first_observed_time,
            end=self.observed_time_,
            interval=self.interval_,
            eager=True,
        ).len()
        future_indices = pl.arange(
            current_time_index,
            current_time_index + forecasting_horizon,
            eager=True,
        )

        # Build Fourier features for future times
        X_future = self._build_fourier_features(future_indices)

        # Predict all columns at once
        predictions = self.model_.predict(X_future)

        # Handle both single and multiple columns (ElasticNet returns 1D for single target)
        if len(self._column_names) == 1:
            predictions = predictions.reshape(-1, 1)

        # Create DataFrame with column names
        y_pred = pl.DataFrame(
            {col_name: predictions[:, i] for i, col_name in enumerate(self._column_names)}
        )
        return self._add_time_columns(y_pred)
