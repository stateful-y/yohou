"""Point forecasting metrics for evaluating prediction accuracy."""

from __future__ import annotations

import numbers
import warnings
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from datetime import datetime

from yohou.utils import validate_scorer_data
from yohou.utils._compat import Interval, _fit_context

from .base import BasePointScorer

__all__ = [
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanAbsoluteScaledError",
    "MeanSquaredError",
    "MedianAbsoluteError",
    "RootMeanSquaredError",
    "RootMeanSquaredScaledError",
    "SymmetricMeanAbsolutePercentageError",
]


class MeanAbsoluteError(BasePointScorer):
    r"""Mean Absolute Error metric for point forecasts.

    Computes the average of absolute differences between predictions and actual values.
    This metric is robust to outliers and provides intuitive interpretation in the
    original units of the target variable.

    The MAE is defined as:

    $$\text{MAE} = \frac{1}{n}\\sum_{i=1}^{n}|y_i - \\hat{y}_i|$$

    where $y_i$ is the actual value, $\\hat{y}_i$ is the predicted value, and
    $n$ is the number of observations.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. Options:
        - "stepwise": Aggregate across forecasting steps.
        - "vintagewise": Aggregate across vintages (observed times).
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["stepwise", "vintagewise", "componentwise", "groupwise"].
        Example outputs:
        - ["stepwise", "vintagewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["stepwise", "vintagewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict).
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict).

    Attributes
    ----------
    lower_is_better : bool
        Always True for MAE.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MeanAbsoluteError
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [10.0, 20.0, 30.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [12.0, 19.0, 28.0],
    ... })
    >>> mae = MeanAbsoluteError()
    >>> _ = mae.fit(y_true)
    >>> mae.score(y_true, y_pred)  # doctest: +ELLIPSIS
    1.666...

    Notes
    -----
    - MAE treats all errors equally regardless of direction (over or under prediction)
    - Less sensitive to outliers compared to MSE/RMSE
    - Interpretable in the same units as the target variable
    - Suitable for most forecasting tasks where outliers should not dominate

    See Also
    --------
    `MeanSquaredError` : Mean Squared Error, more sensitive to large errors
    `RootMeanSquaredError` : Root Mean Squared Error, MeanSquaredError in original units
    `RootMeanSquaredScaledError` : Root Mean Squared Scaled Error, scale-independent version
    `MAPE` : Mean Absolute Percentage Error, scale-independent

    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
    }

    _metric_name = "mae"

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
        )

    def _compute_raw_errors(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Compute per-row absolute errors."""
        return (y_truth - y_pred).select(pl.all().abs())


class MeanSquaredError(BasePointScorer):
    r"""Mean Squared Error metric for point forecasts.

    Computes the average of squared differences between predictions and actual values.
    This metric heavily penalizes large errors, making it sensitive to outliers.

    The MSE is defined as:

    $$\\text{MSE} = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2$$

    where $y_i$ is the actual value, $\\hat{y}_i$ is the predicted value, and
    $n$ is the number of observations.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. Options:
        - "stepwise": Aggregate across forecasting steps.
        - "vintagewise": Aggregate across vintages (observed times).
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["stepwise", "vintagewise", "componentwise", "groupwise"].
        Example outputs:
        - ["stepwise", "vintagewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["stepwise", "vintagewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict).
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict).

    Attributes
    ----------
    lower_is_better : bool
        Always True for MeanSquaredError.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MeanSquaredError
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [10.0, 20.0, 30.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [12.0, 19.0, 28.0],
    ... })
    >>> mse = MeanSquaredError()
    >>> _ = mse.fit(y_true)
    >>> mse.score(y_true, y_pred)  # doctest: +ELLIPSIS
    3.0

    Notes
    -----
    - Squaring errors penalizes large deviations more than small ones
    - More sensitive to outliers compared to MAE
    - Units are squared, making direct interpretation less intuitive (use RMSE for original units)
    - Commonly used in regression and when large errors are particularly undesirable

    See Also
    --------
    `MeanAbsoluteError` : Mean Absolute Error, less sensitive to outliers
    `RootMeanSquaredError` : Root Mean Squared Error, MeanSquaredError in original units
    `RootMeanSquaredScaledError` : Root Mean Squared Scaled Error, scale-independent version

    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
    }

    _metric_name = "mse"

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
        )

    def _compute_raw_errors(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Compute per-row squared errors."""
        return (y_truth - y_pred).select(pl.all().pow(2))


class RootMeanSquaredError(BasePointScorer):
    r"""Root Mean Squared Error metric for point forecasts.

    Computes the square root of the average of squared differences between predictions
    and actual values. This metric penalizes large errors while maintaining the same
    units as the target variable, making it more interpretable than MeanSquaredError.

    The RMSE is defined as:

    $$\\text{RMSE} = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2}$$

    where $y_i$ is the actual value, $\\hat{y}_i$ is the predicted value, and
    $n$ is the number of observations.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. Options:
        - "stepwise": Aggregate across forecasting steps.
        - "vintagewise": Aggregate across vintages (observed times).
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["stepwise", "vintagewise", "componentwise", "groupwise"].
        Example outputs:
        - ["stepwise", "vintagewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["stepwise", "vintagewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict).
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict).

    Attributes
    ----------
    lower_is_better : bool
        Always True for MSE.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import RootMeanSquaredError
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [10.0, 20.0, 30.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [12.0, 19.0, 28.0],
    ... })
    >>> rmse = RootMeanSquaredError()
    >>> _ = rmse.fit(y_true)
    >>> rmse.score(y_true, y_pred)  # doctest: +ELLIPSIS
    1.732...

    Notes
    -----
    - RMSE is the square root of MSE, providing errors in original units
    - More sensitive to outliers compared to MeanAbsoluteError but less than MSE
    - Commonly used when large errors are particularly undesirable
    - Interpretable in the same units as the target variable

    See Also
    --------
    `MeanAbsoluteError` : Mean Absolute Error, less sensitive to outliers
    `MeanSquaredError` : Mean Squared Error, RMSE squared
    `RootMeanSquaredScaledError` : Root Mean Squared Scaled Error, scale-independent version

    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
    }

    _metric_name = "rmse"

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
        )

    def _compute_raw_errors(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Compute per-row squared errors for RMSE."""
        return (y_truth - y_pred).select(pl.all().pow(2))

    def _post_aggregate(self, result: float | pl.DataFrame) -> float | pl.DataFrame:
        """Apply square root to aggregated squared errors."""
        if isinstance(result, pl.DataFrame):
            numeric_cols = [c for c in result.columns if c != "time"]
            return result.with_columns([pl.col(c).sqrt() for c in numeric_cols])
        return float(np.sqrt(result))


class RootMeanSquaredScaledError(BasePointScorer):
    r"""Root Mean Squared Scaled Error metric for point forecasts.

    Computes RMSE scaled by the in-sample naive seasonal forecast error. This provides
    a scale-independent metric that enables comparison across time series with different
    magnitudes. Requires training data to compute scaling factors.

    The RootMeanSquaredScaledError is defined as:

    $$\\text{RMSSE} = \\sqrt{\\frac{1}{h}\\sum_{t=1}^{h}\\left(\\frac{y_t - \\hat{y}_t}{\\text{scale}}\\right)^2}$$

    where the scale is computed from training data as:

    $$\\text{scale}_j = \\frac{1}{T-m}\\sum_{t=m+1}^{T}(y_{t,j} - y_{t-m,j})^2$$

    with $m$ = seasonality, $T$ = training length, $h$ = forecast horizon, and $j$ = column index.
    Per-column RootMeanSquaredScaledError values are averaged to produce the final score.

    Parameters
    ----------
    seasonality : int, default=1
        Seasonal period for computing scaling factors. Must be at least 1.
        Common values: 1 (non-seasonal), 7 (weekly), 12 (monthly), 24 (hourly daily pattern).

    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. Options:
        - "stepwise": Aggregate across forecasting steps.
        - "vintagewise": Aggregate across vintages (observed times).
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["stepwise", "vintagewise", "componentwise", "groupwise"].
        Example outputs:
        - ["stepwise", "vintagewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["stepwise", "vintagewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict).
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict).

    Attributes
    ----------
    lower_is_better : bool
        Always True for RootMeanSquaredScaledError.
    scales_ : dict[str, float]
        Fitted per-column scaling factors. Computed during fit() from training data
        naive seasonal forecast errors.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> from yohou.metrics import RootMeanSquaredScaledError
    >>> # Training data
    >>> y_train = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
    ...     "value": [10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0],
    ... })
    >>> # Test predictions
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 11), datetime(2020, 1, 12)],
    ...     "value": [15.0, 17.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2020, 1, 10)] * 2,
    ...     "time": [datetime(2020, 1, 11), datetime(2020, 1, 12)],
    ...     "value": [15.5, 16.5],
    ... })
    >>> rmsse = RootMeanSquaredScaledError(seasonality=2)
    >>> rmsse.fit(y_train)
    RootMeanSquaredScaledError(seasonality=2)
    >>> rmsse.score(y_true, y_pred)
    0.5

    Notes
    -----
    - RootMeanSquaredScaledError values are scale-independent, enabling comparison across different time series
    - Values < 1 indicate better performance than naive seasonal forecast on training data
    - Values > 1 indicate worse performance than naive seasonal baseline
    - Requires training data with length > seasonality
    - Per-column scaling factors are stored and applied independently

    See Also
    --------
    `RootMeanSquaredError` : Root Mean Squared Error, non-scaled version
    `MeanAbsoluteError` : Mean Absolute Error, non-scaled alternative
    `MeanSquaredError` : Mean Squared Error, squared version

    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
        "seasonality": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    _metric_name = "rmsse"

    def __init__(
        self,
        seasonality: int = 1,
        aggregation_method: list[str] | str = "all",
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
        )
        self.seasonality = seasonality

    def __sklearn_tags__(self):
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with requires_calibration=True.

        """
        tags = super().__sklearn_tags__()
        if tags.scorer_tags is not None:
            tags.scorer_tags.requires_calibration = True
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y_train: pl.DataFrame, *, forecaster=None, **params) -> RootMeanSquaredScaledError:
        """Fit the scorer by computing per-column scaling factors.

        Parameters
        ----------
        y_train : pl.DataFrame
            Training set target values with "time" column.
        forecaster : BaseForecaster or None, default=None
            If provided, metadata is extracted directly from the fitted
            forecaster instead of being re-inferred from ``y_train``.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If y_train is None or seasonality > len(y_train) - 1.

        """
        # Call parent fit() to validate parameters (aggregation_method, groups, etc.)
        super().fit(y_train, forecaster=forecaster, **params)

        # Validate training data and remove time column
        y_train_values, _, _ = validate_scorer_data(scorer=self, y_true=y_train, y_pred=None, reset=True)

        # Compute per-column scaling factors using seasonal naive forecast errors
        self.scales_ = {}
        for col in y_train_values.columns:
            # Compute seasonal differences: y_t - y_{t-seasonality}
            col_data = y_train_values[col].to_numpy()
            seasonal_errors = col_data[self.seasonality :] - col_data[: -self.seasonality]

            # Scale is mean squared error of seasonal naive forecast
            scale = float(np.mean(seasonal_errors**2))

            if scale == 0:
                warnings.warn(
                    f"Training data for column '{col}' has zero scale "
                    f"(constant values with seasonality={self.seasonality}). "
                    "RMSSE scores for this column will use a scale floor of 1e-10.",
                    UserWarning,
                    stacklevel=2,
                )

            # Store non-zero scale (avoid division by zero in score())
            self.scales_[col] = max(scale, 1e-10)

        return self

    def _compute_raw_errors(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Compute per-row scaled squared errors for RMSSE."""
        scaled_squared_errors_data = {}
        for col in y_truth.columns:
            errors = (y_truth[col] - y_pred[col]).to_numpy()
            scale = self.scales_[col]
            scaled_squared_errors_data[col] = (errors / np.sqrt(scale)) ** 2
        return pl.DataFrame(scaled_squared_errors_data)

    def _post_aggregate(self, result: float | pl.DataFrame) -> float | pl.DataFrame:
        """Apply square root to aggregated scaled squared errors."""
        if isinstance(result, pl.DataFrame):
            numeric_cols = [c for c in result.columns if c != "time"]
            return result.with_columns([pl.col(c).sqrt() for c in numeric_cols])
        return float(np.sqrt(result))


class MeanAbsolutePercentageError(BasePointScorer):
    r"""Mean Absolute Percentage Error metric for point forecasts.

    Computes the average percentage error between predictions and actual values.
    This provides a scale-independent metric that enables comparison across time series
    with different magnitudes.

    The MAPE is defined as:

    $$\\text{MAPE} = \\frac{100}{n}\\sum_{i=1}^{n}\\left|\\frac{y_i - \\hat{y}_i}{y_i}\\right|$$

    where $y_i$ is the actual value, $\\hat{y}_i$ is the predicted value, and
    $n$ is the number of observations.

    Parameters
    ----------
    epsilon : float, default=1e-8
        Small constant added to denominator to prevent division by zero when actual values
        are zero or near-zero.
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. Options:
        - "stepwise": Aggregate across forecasting steps.
        - "vintagewise": Aggregate across vintages (observed times).
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["stepwise", "vintagewise", "componentwise", "groupwise"].
        Example outputs:
        - ["stepwise", "vintagewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["stepwise", "vintagewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict).
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict).

    Attributes
    ----------
    lower_is_better : bool
        Always True for MAPE.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MeanAbsolutePercentageError
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [10.0, 20.0, 30.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [12.0, 19.0, 28.0],
    ... })
    >>> mape = MeanAbsolutePercentageError()
    >>> _ = mape.fit(y_true)
    >>> mape.score(y_true, y_pred)  # doctest: +ELLIPSIS
    10.55...

    Notes
    -----
    - MAPE is scale-independent and useful for comparing forecasts across different series
    - Asymmetric: penalizes over-predictions more heavily than under-predictions
    - Undefined when actual values are zero; epsilon parameter prevents division by zero
    - Values are expressed as percentages (0-100 scale)
    - May be sensitive to very small actual values even with epsilon protection

    See Also
    --------
    `SymmetricMeanAbsolutePercentageError` : Symmetric version of MAPE
    `MeanAbsoluteError` : Absolute error in original units
    `MeanAbsoluteScaledError` : Scaled by naive forecast error

    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
        "epsilon": [Interval(numbers.Real, 0, None, closed="neither")],
    }

    _metric_name = "mape"

    def __init__(
        self,
        epsilon: float = 1e-8,
        aggregation_method: list[str] | str = "all",
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
        )
        self.epsilon = epsilon

    def _compute_raw_errors(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Compute per-row absolute percentage errors."""
        pct_errors_data = {}
        for col in y_truth.columns:
            abs_errors = (y_truth[col] - y_pred[col]).abs()
            pct_errors_data[col] = (abs_errors / (y_truth[col].abs() + self.epsilon)) * 100.0
        return pl.DataFrame(pct_errors_data)


class SymmetricMeanAbsolutePercentageError(BasePointScorer):
    r"""Symmetric Mean Absolute Percentage Error metric for point forecasts.

    Computes the symmetric average percentage error between predictions and actual values.
    This provides a scale-independent metric that treats over and under-predictions equally,
    unlike MAPE which is asymmetric.

    The sMAPE is defined as:

    $$\\text{sMAPE} = \\frac{100}{n}\\sum_{i=1}^{n}\\frac{|y_i - \\hat{y}_i|}{(|y_i| + |\\hat{y}_i|)/2}$$

    where $y_i$ is the actual value, $\\hat{y}_i$ is the predicted value, and
    $n$ is the number of observations. Values are bounded between 0 and 200.

    Parameters
    ----------
    epsilon : float, default=1e-8
        Small constant added to denominator to prevent division by zero when both
        actual and predicted values are zero or near-zero.

    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. Options:
        - "stepwise": Aggregate across forecasting steps.
        - "vintagewise": Aggregate across vintages (observed times).
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["stepwise", "vintagewise", "componentwise", "groupwise"].
        Example outputs:
        - ["stepwise", "vintagewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["stepwise", "vintagewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict).
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict).

    Attributes
    ----------
    lower_is_better : bool
        Always True for sMAPE.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import SymmetricMeanAbsolutePercentageError
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [10.0, 20.0, 30.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [12.0, 19.0, 28.0],
    ... })
    >>> smape = SymmetricMeanAbsolutePercentageError()
    >>> _ = smape.fit(y_true)
    >>> smape.score(y_true, y_pred)  # doctest: +ELLIPSIS
    10.06...

    Notes
    -----
    - sMAPE is symmetric: treats over-predictions and under-predictions equally
    - Scale-independent and useful for comparing forecasts across different series
    - Bounded between 0 and 200 (unlike MAPE which is unbounded)
    - Less sensitive to very small actual values compared to MAPE
    - Undefined when both actual and predicted values are zero; epsilon prevents division by zero

    See Also
    --------
    `MeanAbsolutePercentageError` : Asymmetric version of percentage error
    `MeanAbsoluteError` : Absolute error in original units
    `MeanAbsoluteScaledError` : Scaled by naive forecast error

    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
        "epsilon": [Interval(numbers.Real, 0, None, closed="neither")],
    }

    _metric_name = "smape"

    def __init__(
        self,
        epsilon: float = 1e-8,
        aggregation_method: list[str] | str = "all",
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
        )
        self.epsilon = epsilon

    def _compute_raw_errors(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Compute per-row symmetric absolute percentage errors."""
        smape_errors_data = {}
        for col in y_truth.columns:
            abs_errors = (y_truth[col] - y_pred[col]).abs()
            denominator = (y_truth[col].abs() + y_pred[col].abs()) / 2.0 + self.epsilon
            smape_errors_data[col] = (abs_errors / denominator) * 100.0
        return pl.DataFrame(smape_errors_data)


class MeanAbsoluteScaledError(BasePointScorer):
    r"""Mean Absolute Scaled Error metric for point forecasts.

    Computes MAE scaled by the in-sample naive seasonal forecast error. This provides
    a scale-independent metric that enables comparison across time series with different
    magnitudes. Requires training data to compute scaling factors.

    The MASE is defined as:

    $$\\text{MASE} = \\frac{1}{h}\\sum_{t=1}^{h}\\left|\\frac{y_t - \\hat{y}_t}{\\text{scale}}\\right|$$

    where the scale is computed from training data as:

    $$\\text{scale}_j = \\frac{1}{T-m}\\sum_{t=m+1}^{T}|y_{t,j} - y_{t-m,j}|$$

    with $m$ = seasonality, $T$ = training length, $h$ = forecast horizon, and $j$ = column index.
    Per-column MASE values are averaged to produce the final score.

    Parameters
    ----------
    seasonality : int, default=1
        Seasonal period for computing scaling factors. Must be at least 1.
        Common values: 1 (non-seasonal), 7 (weekly), 12 (monthly), 24 (hourly daily pattern).

    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. Options:
        - "stepwise": Aggregate across forecasting steps.
        - "vintagewise": Aggregate across vintages (observed times).
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["stepwise", "vintagewise", "componentwise", "groupwise"].
        Example outputs:
        - ["stepwise", "vintagewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["stepwise", "vintagewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict).
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict).

    Attributes
    ----------
    lower_is_better : bool
        Always True for MASE.
    naive_errors_ : dict[str, float]
        Fitted per-column scaling factors based on naive seasonal forecast MAE.
        Computed during fit() from training data.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> from yohou.metrics import MeanAbsoluteScaledError
    >>> # Training data
    >>> y_train = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
    ...     "value": [10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0],
    ... })
    >>> # Test predictions
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 11), datetime(2020, 1, 12)],
    ...     "value": [15.0, 17.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2020, 1, 10)] * 2,
    ...     "time": [datetime(2020, 1, 11), datetime(2020, 1, 12)],
    ...     "value": [15.5, 16.5],
    ... })
    >>> mase = MeanAbsoluteScaledError(seasonality=2)
    >>> mase.fit(y_train)
    MeanAbsoluteScaledError(seasonality=2)
    >>> mase.score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.5...

    Notes
    -----
    - MASE values are scale-independent, enabling comparison across different time series
    - Values < 1 indicate better performance than naive seasonal forecast on training data
    - Values > 1 indicate worse performance than naive seasonal baseline
    - Requires training data with length > seasonality
    - Per-column scaling factors are stored and applied independently
    - More interpretable than RMSSE as it uses absolute errors rather than squared errors

    See Also
    --------
    `RootMeanSquaredScaledError` : Squared error version of scaled metric
    `MeanAbsoluteError` : Non-scaled MAE
    `MeanAbsolutePercentageError` : Percentage-based scale-independent metric

    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
        "seasonality": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    _metric_name = "mase"

    def __init__(
        self,
        seasonality: int = 1,
        aggregation_method: list[str] | str = "all",
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
        )
        self.seasonality = seasonality

    def __sklearn_tags__(self):
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with requires_calibration=True.

        """
        tags = super().__sklearn_tags__()
        if tags.scorer_tags is not None:
            tags.scorer_tags.requires_calibration = True
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y_train: pl.DataFrame, *, forecaster=None, **params) -> MeanAbsoluteScaledError:
        """Fit the scorer by computing per-column scaling factors.

        Parameters
        ----------
        y_train : pl.DataFrame
            Training set target values with "time" column.
        forecaster : BaseForecaster or None, default=None
            If provided, metadata is extracted directly from the fitted
            forecaster instead of being re-inferred from ``y_train``.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If y_train is None or seasonality > len(y_train) - 1.

        """
        # Call parent fit() to validate parameters (aggregation_method, groups, etc.)
        super().fit(y_train, forecaster=forecaster, **params)

        # Validate training data and remove time column
        y_train_values, _, _ = validate_scorer_data(scorer=self, y_true=y_train, y_pred=None, reset=True)

        # Compute per-column scaling factors using seasonal naive forecast MAE
        self.naive_errors_ = {}
        for col in y_train_values.columns:
            # Compute seasonal differences: |y_t - y_{t-seasonality}|
            col_data = y_train_values[col].to_numpy()
            naive_errors = np.abs(col_data[self.seasonality :] - col_data[: -self.seasonality])

            # Scale is mean absolute error of seasonal naive forecast
            scale = float(np.mean(naive_errors))

            if scale == 0:
                warnings.warn(
                    f"Training data for column '{col}' has zero scale "
                    f"(constant values with seasonality={self.seasonality}). "
                    "MASE scores for this column will use a scale floor of 1e-10.",
                    UserWarning,
                    stacklevel=2,
                )

            # Store non-zero scale (avoid division by zero in score())
            self.naive_errors_[col] = max(scale, 1e-10)

        return self

    def _compute_raw_errors(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Compute per-row scaled absolute errors."""
        scaled_errors_data = {}
        for col in y_truth.columns:
            errors = (y_truth[col] - y_pred[col]).abs().to_numpy()
            scale = self.naive_errors_[col]
            scaled_errors_data[col] = errors / scale
        return pl.DataFrame(scaled_errors_data)


class MedianAbsoluteError(BasePointScorer):
    r"""Median Absolute Error metric for point forecasts.

    Computes the median of absolute differences between predictions and actual values.
    This metric is highly robust to outliers and provides a more stable measure of
    typical error magnitude compared to mean-based metrics.

    The MedianAE is defined as:

    $$\\text{MedianAE} = \\text{median}(|y_i - \\hat{y}_i|)$$

    where $y_i$ is the actual value and $\\hat{y}_i$ is the predicted value.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. Options:
        - "stepwise": Aggregate across forecasting steps.
        - "vintagewise": Aggregate across vintages (observed times).
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["stepwise", "vintagewise", "componentwise", "groupwise"].
        Example outputs:
        - ["stepwise", "vintagewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["stepwise", "vintagewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict).
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict).

    Attributes
    ----------
    lower_is_better : bool
        Always True for MedianAE.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MedianAbsoluteError
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [10.0, 20.0, 30.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [12.0, 19.0, 28.0],
    ... })
    >>> medae = MedianAbsoluteError()
    >>> _ = medae.fit(y_true)
    >>> medae.score(y_true, y_pred)
    2.0

    Notes
    -----
    - MedianAE is highly robust to outliers and extreme errors
    - Provides a better measure of typical error when error distribution is skewed
    - Less sensitive to a few very large prediction errors compared to MAE
    - Interpretable in the same units as the target variable
    - Suitable when outliers should not dominate the evaluation

    See Also
    --------
    `MeanAbsoluteError` : Mean-based absolute error, more sensitive to outliers
    `MaxError` : Maximum absolute error, worst-case measure

    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
    }

    _metric_name = "median_ae"

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
        )

    def _compute_raw_errors(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Compute per-row absolute errors for median aggregation."""
        return (y_truth - y_pred).select(pl.all().abs())

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **params) -> float | pl.DataFrame:  # type: ignore
        """Compute median absolute error.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True values with "time" column.
        y_pred : pl.DataFrame
            Predicted values with "time" column.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        float or pl.DataFrame
            If aggregation_method includes stepwise+vintagewise, returns DataFrame with per-component MedianAE values.
            If aggregation_method includes "componentwise", returns DataFrame with "time" and "median_ae" columns.
            If aggregation_method="all", returns scalar float.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, context = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        abs_errors = self._compute_raw_errors(y_truth, y_pred)

        # Apply median aggregation
        agg_method = self.aggregation_method
        if isinstance(agg_method, str):
            agg_method = [agg_method]

        collapse_steps = "stepwise" in agg_method
        collapse_vintages = "vintagewise" in agg_method
        collapse_all_rows = collapse_steps and collapse_vintages

        if "all" in agg_method or (
            collapse_all_rows and "componentwise" in agg_method and ("groupwise" in agg_method or self.groups is None)
        ):
            # Fully aggregated: global median
            result = float(abs_errors.select(pl.all().median()).to_numpy().flatten().mean())
        elif collapse_all_rows:
            # Per-component: median across time
            result = abs_errors.select(pl.all().median())
        elif "componentwise" in agg_method:
            # Per-timestep: median across components
            result = abs_errors.select(pl.concat_list(pl.all()).alias("errors")).select(
                pl.col("errors").list.eval(pl.element().median()).list.first().alias("score")
            )
            time_values = context.time_values if context is not None else None
            if time_values is not None:
                result = result.with_columns(pl.Series("time", time_values).cast(pl.Datetime))
                result = result.select(["time"] + [c for c in result.columns if c != "time"])
        else:
            result = abs_errors.select(pl.all().median())

        if isinstance(result, pl.DataFrame):
            result = self._rename_metric_columns(result)

        return result
