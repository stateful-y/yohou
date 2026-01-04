"""Polynomial trend forecasting implementation."""

import numbers

import numpy as np
import polars as pl
from pydantic import StrictInt
from sklearn.base import _fit_context
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils._param_validation import Interval

from .base import _BaseTrendForecaster


class PolynomialTrendForecaster(_BaseTrendForecaster):
    """Forecast using polynomial trend extrapolation with ElasticNet regularization.

    Fits a polynomial of specified degree to the historical data using ElasticNet
    regression and extrapolates into the future. Linear trend is the special case
    with degree=1.

    Parameters
    ----------
    degree : int, default=1
        Polynomial degree. degree=1 gives linear trend, degree=2 quadratic, etc.
        Higher degrees can overfit - typically use degree <= 3.
    alpha : float, default=1.0
        Regularization strength. Must be positive. alpha=0 is equivalent to
        ordinary least squares (no regularization).
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter in [0, 1]. l1_ratio=0 is Ridge (L2),
        l1_ratio=1 is Lasso (L1), 0 < l1_ratio < 1 is ElasticNet.
    target_transformer : BaseTransformer, optional
        Transformer for target variable (e.g., LogTransform).

    Attributes
    ----------
    model_ : ElasticNet
        Fitted ElasticNet model for all target columns.
    poly_features_ : PolynomialFeatures
        Fitted polynomial feature transformer.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.point_forecaster import PolynomialTrendForecaster
    >>>
    >>> # Create time series with linear trend
    >>> y = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         start=datetime(2020, 1, 1),
    ...         end=datetime(2020, 12, 31),
    ...         interval="1d",
    ...         eager=True
    ...     ),
    ...     "value": range(366)
    ... })
    >>>
    >>> # Fit linear trend forecaster
    >>> forecaster = PolynomialTrendForecaster(degree=1)
    >>> forecaster.fit(y, forecasting_horizon=7)
    PolynomialTrendForecaster()
    >>>
    >>> # Forecast next 7 days
    >>> y_pred = forecaster.predict(forecasting_horizon=7)

    Notes
    -----
    - For exponential trends, consider using target_transformer=LogTransform()
      with degree=1
    - Polynomial trends can overfit - use with caution (typically degree <= 3)
    - Time is converted to numeric values (days since first observation)

    """

    _parameter_constraints: dict = {
        **_BaseTrendForecaster._parameter_constraints,
        "degree": [Interval(numbers.Integral, 0, None, closed="left")],
        "alpha": [Interval(numbers.Real, 0.0, None, closed="left")],
        "l1_ratio": [Interval(numbers.Real, 0.0, 1.0, closed="both")],
    }

    def __init__(
        self,
        degree: StrictInt = 1,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        target_transformer=None,
    ):
        super().__init__(target_transformer=target_transformer)
        self.degree = degree
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _predict_one(self) -> pl.DataFrame:
        """Generate one prediction.

        Returns
        -------
        pl.DataFrame
            Predictions for next forecasting_horizon steps (without time columns).

        """
        time_indices = self._get_time_indices(self.fit_forecasting_horizon_).to_numpy().reshape(-1, 1)

        # Transform time points to polynomial features
        X_pred = self.poly_features_.transform(time_indices)

        # Predict using ElasticNet
        y_pred_array = self.model_.predict(X_pred)

        print("X_pred:", X_pred)
        print("y_pred_array:", y_pred_array)
        # Handle 1D vs 2D output (ElasticNet returns 1D for single target column)
        if len(self._column_names) == 1:
            y_pred_array = y_pred_array.reshape(-1, 1)

        # Convert to polars DataFrame with correct column names
        y_pred = pl.DataFrame(
            {col: y_pred_array[:, i] for i, col in enumerate(self._column_names)}
        )

        print("y_pred:", y_pred)

        return self._add_time_columns(y_pred)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "PolynomialTrendForecaster":
        """Fit polynomial trend model to historical data.

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
        # Pre-fit: validate inputs, apply target transformer, set attributes
        y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)

        # Fit polynomial trend model
        self._fit_polynomial(y_t)

        return self

    def _fit_polynomial(self, y: pl.DataFrame) -> None:
        """Fit polynomial trend model using ElasticNet.

        Parameters
        ----------
        y : pl.DataFrame
            Transformed target time series (without "time" column after _pre_fit).

        """
        # Store column names (excluding "time")
        self._column_names = [col for col in y.columns if col != "time"]

        # Create time index (0, 1, 2, ..., n-1)
        time_indices = self._get_time_indices().to_numpy().reshape(-1, 1)

        # Create polynomial features
        self.poly_features_ = PolynomialFeatures(degree=self.degree, include_bias=True)
        X_poly = self.poly_features_.fit_transform(time_indices)

        # Prepare target array (all columns, excluding "time")
        y_array = y.select(self._column_names).to_numpy()

        # Use LinearRegression when alpha=0 for better numerical precision
        if self.alpha == 0.0:
            self.model_ = LinearRegression(fit_intercept=False)
        else:
            self.model_ = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=False,  # Polynomial features already include bias term
                random_state=None,
            )
        print("X_poly:", X_poly)
        print("y_array:", y_array)
        self.model_.fit(X_poly, y_array)
