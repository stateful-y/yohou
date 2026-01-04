"""Exponential trend forecasting implementation."""

import numbers

import numpy as np
import polars as pl
from pydantic import StrictInt
from sklearn.base import _fit_context
from sklearn.linear_model import ElasticNet
from sklearn.utils._param_validation import Interval

from .base import _BaseTrendForecaster


class ExponentialTrendForecaster(_BaseTrendForecaster):
    """Forecast using exponential trend extrapolation with ElasticNet regularization.

    Fits an exponential trend of the form y = a * exp(b*t) to historical data
    by transforming to linear via logarithm: log(y) = log(a) + b*t, then fits
    using ElasticNet regression.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength. Must be non-negative. alpha=0 is equivalent to
        ordinary least squares (no regularization).
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter in [0, 1]. l1_ratio=0 is Ridge (L2),
        l1_ratio=1 is Lasso (L1), 0 < l1_ratio < 1 is ElasticNet.
    target_transformer : BaseTransformer, optional
        Transformer for target variable (applied before exponential fitting).

    Attributes
    ----------
    model_ : ElasticNet or LinearRegression
        Fitted regression model for all target columns (in log space).

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> from datetime import datetime
    >>> from yohou.point_forecaster import ExponentialTrendForecaster
    >>>
    >>> # Create time series with exponential growth
    >>> time_range = pl.datetime_range(
    ...     start=datetime(2020, 1, 1),
    ...     end=datetime(2020, 12, 31),
    ...     interval="1d",
    ...     eager=True
    ... )
    >>> y = pl.DataFrame({
    ...     "time": time_range,
    ...     "value": [10 * np.exp(0.01 * i) for i in range(366)]
    ... })
    >>>
    >>> # Fit exponential trend forecaster
    >>> forecaster = ExponentialTrendForecaster()
    >>> forecaster.fit(y, forecasting_horizon=7)
    ExponentialTrendForecaster()
    >>>
    >>> # Forecast next 7 days
    >>> y_pred = forecaster.predict(forecasting_horizon=7)

    Notes
    -----
    - Requires all y values to be strictly positive (raises ValueError otherwise)
    - Good for modeling growth/decay processes (e.g., compound interest, viral spread)
    - For negative growth rates, values decay exponentially toward zero

    """

    _parameter_constraints: dict = {
        **_BaseTrendForecaster._parameter_constraints,
        "alpha": [Interval(numbers.Real, 0.0, None, closed="left")],
        "l1_ratio": [Interval(numbers.Real, 0.0, 1.0, closed="both")],
    }

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        target_transformer=None,
    ):
        super().__init__(target_transformer=target_transformer)
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

        # Predict in log space
        log_y_pred = self.model_.predict(time_indices)

        # Handle 1D vs 2D output (LinearRegression/ElasticNet returns 1D for single target)
        if len(self._column_names) == 1:
            log_y_pred = log_y_pred.reshape(-1, 1)

        # Transform back from log space: y = exp(log_y)
        y_pred_array = np.exp(log_y_pred)

        # Convert to polars DataFrame with correct column names
        y_pred = pl.DataFrame(
            {col: y_pred_array[:, i] for i, col in enumerate(self._column_names)}
        )

        return self._add_time_columns(y_pred)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "ExponentialTrendForecaster":
        """Fit exponential trend model to historical data.

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

        Raises
        ------
        ValueError
            If y contains non-positive values.

        """
        # Pre-fit: validate inputs, apply target transformer, set attributes
        y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)

        # Validate that all values are positive
        self._validate_positive(y_t)

        # Fit exponential trend model
        self._fit_exponential(y_t)

        return self

    def _validate_positive(self, y: pl.DataFrame) -> None:
        """Validate that all y values are strictly positive.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        Raises
        ------
        ValueError
            If any values are non-positive.

        """
        for col_name in y.columns:
            if col_name == "time":
                continue

            min_value = y[col_name].min()
            if min_value is None or min_value <= 0:
                raise ValueError(
                    f"ExponentialTrendForecaster requires all positive values. "
                    f"Column '{col_name}' has minimum value {min_value}. "
                    f"Consider using PolynomialTrendForecaster with a log transform instead."
                )

    def _fit_exponential(self, y: pl.DataFrame) -> None:
        """Fit exponential trend model using ElasticNet in log space.

        Parameters
        ----------
        y : pl.DataFrame
            Transformed target time series (without "time" column after _pre_fit).

        """
        # Store column names (excluding "time")
        self._column_names = [col for col in y.columns if col != "time"]

        # Create time index (0, 1, 2, ..., n-1)
        time_indices = self._get_time_indices().to_numpy().reshape(-1, 1)

        # Transform all target columns to log space
        y_array = y.select(self._column_names).to_numpy()
        log_y_array = np.log(y_array)

        self.model_ = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=True,
            random_state=None,
        )
        self.model_.fit(time_indices, log_y_array)
