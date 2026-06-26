"""Polynomial trend forecasting implementation."""

import numbers

import polars as pl
from pydantic import StrictInt
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from yohou.utils._compat import Interval

from .base import _BaseTrendForecaster

__all__ = ["PolynomialTrendForecaster"]


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
    estimator : RegressorMixin, default=ElasticNet()
        Regression model used to fit polynomial coefficients.
    target_transformer : BaseTransformer, optional
        Transformer for target variable (e.g., LogTransformer).
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data.  See `BaseForecaster` for details.

    Attributes
    ----------
    estimator_ : Pipeline
        Fitted sklearn Pipeline with a polynomial feature transformer and a
        clone of the provided `estimator` model.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.stationarity import PolynomialTrendForecaster
    >>>
    >>> # Create time series with linear trend
    >>> y = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         start=datetime(2020, 1, 1), end=datetime(2020, 12, 31), interval="1d", eager=True
    ...     ),
    ...     "value": range(366),
    ... })
    >>>
    >>> # Fit linear trend forecaster
    >>> forecaster = PolynomialTrendForecaster(degree=1)
    >>> forecaster.fit(y, forecasting_horizon=7)
    PolynomialTrendForecaster()
    >>>
    >>> # Forecast next 7 days
    >>> y_pred = forecaster.predict(forecasting_horizon=7)

    See Also
    --------
    - [`PatternSeasonalityForecaster`][yohou.stationarity.seasonality.PatternSeasonalityForecaster] : Seasonal pattern extraction for periodic components.
    - [`FourierSeasonalityForecaster`][yohou.stationarity.seasonality.FourierSeasonalityForecaster] : Fourier-based seasonality estimation.
    - [`DecompositionPipeline`][yohou.compose.decomposition_pipeline.DecompositionPipeline] : Combines trend + seasonality + residual forecasters.

    Notes
    -----
    - For exponential trends, consider using target_transformer=LogTransformer()
      with degree=1
    - Polynomial trends can overfit - use with caution (typically degree <= 3)
    - Time is converted to numeric values (number of intervals since first observation)

    References
    ----------
    [1] Hyndman, R.J., & Athanasopoulos, G. (2021). "Forecasting:
        principles and practice," 3rd edition, OTexts: Melbourne, Australia.
        OTexts.com/fpp3. Chapters 3.2 and 7.4.

    """

    _parameter_constraints: dict = {
        "degree": [Interval(numbers.Integral, 0, None, closed="left")],
        "estimator": [RegressorMixin],
    }

    def __init__(
        self,
        degree: StrictInt = 1,
        estimator: RegressorMixin = ElasticNet(),
        target_transformer=None,
        panel_strategy="global",
    ):
        super().__init__(target_transformer=target_transformer, panel_strategy=panel_strategy)
        self.degree = degree
        self.estimator = estimator

    def _fit(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        X_t: pl.DataFrame | dict[str, pl.DataFrame] | None,
        forecasting_horizon: StrictInt,
    ) -> None:
        """Fit polynomial trend model to transformed data.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series.
        X_t : pl.DataFrame or dict[str, pl.DataFrame] or None
            Transformed features (unused).
        forecasting_horizon : int
            Number of steps ahead to forecast (not used; stored by the base
            class as ``fit_forecasting_horizon_``).

        """
        estimator = Pipeline([
            ("poly_features", PolynomialFeatures(degree=self.degree, include_bias=True)),
            ("regressor", clone(self.estimator)),
        ])

        self._fit_estimator(estimator, y_t)
