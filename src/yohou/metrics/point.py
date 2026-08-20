"""Point forecasting metrics for evaluating prediction accuracy."""

from __future__ import annotations

import numbers
import warnings
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from yohou.utils._context import ScoringContext

from yohou.utils import validate_scorer_data
from yohou.utils._compat import Interval, _fit_context
from yohou.weighting import BaseWeighter

from .base import BasePointScorer

__all__ = [
    "MaxAbsoluteError",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanAbsoluteScaledError",
    "MeanDirectionalAccuracy",
    "MeanSquaredError",
    "MedianAbsoluteError",
    "R2Score",
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
    ...     "vintage_time": [datetime(2019, 12, 31)] * 3,
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
    - [`MeanSquaredError`][yohou.metrics.point.MeanSquaredError] : Mean Squared Error, more sensitive to large errors
    - [`RootMeanSquaredError`][yohou.metrics.point.RootMeanSquaredError] : Root Mean Squared Error, MeanSquaredError in original units
    - [`RootMeanSquaredScaledError`][yohou.metrics.point.RootMeanSquaredScaledError] : Root Mean Squared Scaled Error, scale-independent version
    - [`MeanAbsolutePercentageError`][yohou.metrics.point.MeanAbsolutePercentageError] : Mean Absolute Percentage Error, scale-independent

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
        time_weighter: BaseWeighter | None = None,
        step_weighter: BaseWeighter | None = None,
        vintage_weighter: BaseWeighter | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
            time_weighter=time_weighter,
            step_weighter=step_weighter,
            vintage_weighter=vintage_weighter,
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
    ...     "vintage_time": [datetime(2019, 12, 31)] * 3,
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
    - [`MeanAbsoluteError`][yohou.metrics.point.MeanAbsoluteError] : Mean Absolute Error, less sensitive to outliers
    - [`RootMeanSquaredError`][yohou.metrics.point.RootMeanSquaredError] : Root Mean Squared Error, MeanSquaredError in original units
    - [`RootMeanSquaredScaledError`][yohou.metrics.point.RootMeanSquaredScaledError] : Root Mean Squared Scaled Error, scale-independent version

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
        time_weighter: BaseWeighter | None = None,
        step_weighter: BaseWeighter | None = None,
        vintage_weighter: BaseWeighter | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
            time_weighter=time_weighter,
            step_weighter=step_weighter,
            vintage_weighter=vintage_weighter,
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
        Always True for RMSE.

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
    ...     "vintage_time": [datetime(2019, 12, 31)] * 3,
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
    - [`MeanAbsoluteError`][yohou.metrics.point.MeanAbsoluteError] : Mean Absolute Error, less sensitive to outliers
    - [`MeanSquaredError`][yohou.metrics.point.MeanSquaredError] : Mean Squared Error, RMSE squared
    - [`RootMeanSquaredScaledError`][yohou.metrics.point.RootMeanSquaredScaledError] : Root Mean Squared Scaled Error, scale-independent version

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
        time_weighter: BaseWeighter | None = None,
        step_weighter: BaseWeighter | None = None,
        vintage_weighter: BaseWeighter | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
            time_weighter=time_weighter,
            step_weighter=step_weighter,
            vintage_weighter=vintage_weighter,
        )

    def _compute_raw_errors(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Compute per-row squared errors for RMSE."""
        return (y_truth - y_pred).select(pl.all().pow(2))

    def _transform_scores(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply square root to aggregated squared errors."""
        return df.select(pl.all().sqrt())


class RootMeanSquaredScaledError(BasePointScorer):
    r"""Root Mean Squared Scaled Error metric for point forecasts.

    Computes RMSE scaled by the in-sample naive seasonal forecast error. This provides
    a scale-independent metric that enables comparison across time series with different
    magnitudes. Requires training data to compute scaling factors.

    The RootMeanSquaredScaledError is defined as:

    $$\\text{RMSSE} = \\sqrt{\\frac{1}{h}\\sum_{t=1}^{h}\\frac{(y_t - \\hat{y}_t)^2}{\\text{scale}_j}}$$

    where the scale is the mean squared seasonal-naive error computed from
    training data as:

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
    ...     "vintage_time": [datetime(2020, 1, 10)] * 2,
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
    - [`RootMeanSquaredError`][yohou.metrics.point.RootMeanSquaredError] : Root Mean Squared Error, non-scaled version
    - [`MeanAbsoluteError`][yohou.metrics.point.MeanAbsoluteError] : Mean Absolute Error, non-scaled alternative
    - [`MeanSquaredError`][yohou.metrics.point.MeanSquaredError] : Mean Squared Error, squared version

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
        time_weighter: BaseWeighter | None = None,
        step_weighter: BaseWeighter | None = None,
        vintage_weighter: BaseWeighter | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
            time_weighter=time_weighter,
            step_weighter=step_weighter,
            vintage_weighter=vintage_weighter,
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
            Accepted for signature compatibility and ignored: scorers route no
            metadata to nested estimators.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If y_train is None or seasonality > len(y_train) - 1.

        Warns
        -----
        UserWarning
            If any training column has zero variance for the given
            seasonality; a scale floor of 1e-10 is used instead.

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
        missing = [col for col in y_truth.columns if col not in self.scales_]
        if missing:
            raise ValueError(
                f"Columns {missing} were not seen during fit(); RMSSE has no scale "
                f"for them. Fitted columns: {list(self.scales_)}."
            )
        diff = y_truth - y_pred
        return diff.select((pl.col(col) / float(np.sqrt(self.scales_[col]))).pow(2) for col in y_truth.columns)

    def _transform_scores(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply square root to aggregated scaled squared errors."""
        return df.select(pl.all().sqrt())


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
    ...     "vintage_time": [datetime(2019, 12, 31)] * 3,
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
    - Values are expressed as percentages; they are unbounded above (unlike sMAPE which is bounded to 200)
    - May be sensitive to very small actual values even with epsilon protection

    See Also
    --------
    - [`SymmetricMeanAbsolutePercentageError`][yohou.metrics.point.SymmetricMeanAbsolutePercentageError] : Symmetric version of MAPE
    - [`MeanAbsoluteError`][yohou.metrics.point.MeanAbsoluteError] : Absolute error in original units
    - [`MeanAbsoluteScaledError`][yohou.metrics.point.MeanAbsoluteScaledError] : Scaled by naive forecast error

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
        time_weighter: BaseWeighter | None = None,
        step_weighter: BaseWeighter | None = None,
        vintage_weighter: BaseWeighter | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
            time_weighter=time_weighter,
            step_weighter=step_weighter,
            vintage_weighter=vintage_weighter,
        )
        self.epsilon = epsilon

    def _compute_raw_errors(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Compute per-row absolute percentage errors."""
        abs_errors = (y_truth - y_pred).select(pl.all().abs())
        denom = y_truth.select(pl.all().abs() + self.epsilon)
        return abs_errors / denom * 100.0


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
    ...     "vintage_time": [datetime(2019, 12, 31)] * 3,
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
    - [`MeanAbsolutePercentageError`][yohou.metrics.point.MeanAbsolutePercentageError] : Asymmetric version of percentage error
    - [`MeanAbsoluteError`][yohou.metrics.point.MeanAbsoluteError] : Absolute error in original units
    - [`MeanAbsoluteScaledError`][yohou.metrics.point.MeanAbsoluteScaledError] : Scaled by naive forecast error

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
        time_weighter: BaseWeighter | None = None,
        step_weighter: BaseWeighter | None = None,
        vintage_weighter: BaseWeighter | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
            time_weighter=time_weighter,
            step_weighter=step_weighter,
            vintage_weighter=vintage_weighter,
        )
        self.epsilon = epsilon

    def _compute_raw_errors(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Compute per-row symmetric absolute percentage errors."""
        abs_errors = (y_truth - y_pred).select(pl.all().abs())
        denom = (y_truth.select(pl.all().abs()) + y_pred.select(pl.all().abs())) / 2.0 + self.epsilon
        return abs_errors / denom * 100.0


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
    ...     "vintage_time": [datetime(2020, 1, 10)] * 2,
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
    - [`RootMeanSquaredScaledError`][yohou.metrics.point.RootMeanSquaredScaledError] : Squared error version of scaled metric
    - [`MeanAbsoluteError`][yohou.metrics.point.MeanAbsoluteError] : Non-scaled MAE
    - [`MeanAbsolutePercentageError`][yohou.metrics.point.MeanAbsolutePercentageError] : Percentage-based scale-independent metric

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
        time_weighter: BaseWeighter | None = None,
        step_weighter: BaseWeighter | None = None,
        vintage_weighter: BaseWeighter | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
            time_weighter=time_weighter,
            step_weighter=step_weighter,
            vintage_weighter=vintage_weighter,
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
            Accepted for signature compatibility and ignored: scorers route no
            metadata to nested estimators.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If y_train is None or seasonality > len(y_train) - 1.

        Warns
        -----
        UserWarning
            If any training column has zero variance for the given
            seasonality; a scale floor of 1e-10 is used instead.

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
        missing = [col for col in y_truth.columns if col not in self.naive_errors_]
        if missing:
            raise ValueError(
                f"Columns {missing} were not seen during fit(); MASE has no scale "
                f"for them. Fitted columns: {list(self.naive_errors_)}."
            )
        diff = y_truth - y_pred
        return diff.select((pl.col(col).abs() / float(self.naive_errors_[col])) for col in y_truth.columns)


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
    time_weighter : BaseWeighter or None, default=None
        Weighter applied along the time axis (observed timestamps). Because the
        median is non-linear, only zero-weight timestamps are dropped; non-zero
        weights do not otherwise reweight the median. If None, all timestamps
        contribute equally.
    step_weighter : BaseWeighter or None, default=None
        Weighter applied along the forecasting-step axis. Only zero-weight steps
        are dropped (see ``time_weighter``). If None, all forecasting steps
        contribute equally.
    vintage_weighter : BaseWeighter or None, default=None
        Weighter applied along the vintage-time axis. If None, all vintages
        contribute equally.

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
    ...     "vintage_time": [datetime(2019, 12, 31)] * 3,
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
    - [`MeanAbsoluteError`][yohou.metrics.point.MeanAbsoluteError] : Mean-based absolute error, more sensitive to outliers
    - [`MaxAbsoluteError`][yohou.metrics.point.MaxAbsoluteError] : Maximum absolute error, worst-case measure

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
        time_weighter: BaseWeighter | None = None,
        step_weighter: BaseWeighter | None = None,
        vintage_weighter: BaseWeighter | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
            time_weighter=time_weighter,
            step_weighter=step_weighter,
            vintage_weighter=vintage_weighter,
        )

    def _compute_raw_errors(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Compute per-row absolute errors for median aggregation."""
        return (y_truth - y_pred).select(pl.all().abs())

    def score(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
        /,
        **params,
    ) -> float | pl.DataFrame:
        """Compute median absolute error.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True values with "time" column.
        y_pred : pl.DataFrame
            Predicted values with "time" column.
        **params : dict
            Accepted for signature compatibility and ignored: scorers route no
            metadata to nested estimators.

        Returns
        -------
        float or pl.DataFrame
            Aggregated median absolute error.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, context = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        # Resolve all weighters: drop zero-weight time/step rows and store
        # vintage weights in context (mirrors the base point pipeline). Median
        # is non-linear, so continuous time/step weights are not multiplied
        # into the errors; only zero-weight rows are removed.
        y_truth, y_pred, context, _, _, _ = self._pre_filter_zero_weights(
            y_truth,
            y_pred,
            context,
            self.time_weighter,
            self.step_weighter,
            self.vintage_weighter,
        )

        dims = self._normalize_agg_methods(self.aggregation_method)
        collapse_steps = "stepwise" in dims
        collapse_vintages = "vintagewise" in dims

        if not collapse_steps and not collapse_vintages:
            # Componentwise/groupwise only: keep per-row scores, let
            # _aggregate_per_vintage_scores handle component/group collapse.
            abs_errors = self._compute_raw_errors(y_truth, y_pred)
            # Per-row median across columns (components)
            result = abs_errors.select(pl.concat_list(pl.all()).alias("_err")).select(
                pl.col("_err").list.eval(pl.element().median()).list.first().alias("score")
            )
            time_values = context.time_values if context is not None else None
            if time_values is not None:
                result = result.with_columns(pl.Series("time", time_values).cast(pl.Datetime))
                result = result.select(["time"] + [c for c in result.columns if c != "time"])
            result = self._rename_metric_columns(result)
            return result

        def _compute_median(yt_slice: pl.DataFrame, yp_slice: pl.DataFrame) -> pl.DataFrame:
            """Compute per-column median absolute error."""
            errors = (yt_slice - yp_slice).select(pl.all().abs())
            return errors.select(pl.all().median())

        result = self._map_per_vintage(y_truth, y_pred, context, _compute_median)
        return self._aggregate_per_vintage_scores(result, context)


class MaxAbsoluteError(BasePointScorer):
    r"""Maximum Absolute Error metric for point forecasts.

    Computes the maximum of absolute differences between predictions and actual
    values. This metric captures worst case prediction error, providing a bound
    on how far off the forecast can be.

    The MaxAE is defined as:

    $$\text{MaxAE} = \max_{i=1}^{n}|y_i - \hat{y}_i|$$

    where $y_i$ is the actual value, $\hat{y}_i$ is the predicted value, and
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
    time_weighter : BaseWeighter or None, default=None
        Weighter applied along the time axis (observed timestamps). If None,
        all timestamps contribute equally.
    step_weighter : BaseWeighter or None, default=None
        Weighter applied along the forecasting-step axis. If None, all
        forecasting steps contribute equally.
    vintage_weighter : BaseWeighter or None, default=None
        Weighter applied along the vintage-time axis. If None, all vintages
        contribute equally.

    Attributes
    ----------
    lower_is_better : bool
        Always True for MaxAE.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MaxAbsoluteError
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [10.0, 20.0, 30.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "vintage_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [12.0, 19.0, 25.0],
    ... })
    >>> max_ae = MaxAbsoluteError()
    >>> _ = max_ae.fit(y_true)
    >>> max_ae.score(y_true, y_pred)
    5.0

    Notes
    -----
    - MaxAE captures the worst case prediction error in a forecast
    - Highly sensitive to outliers by design
    - Interpretable in the same units as the target variable
    - Row collapse uses ``max`` (not ``mean``), while component and group collapse
      use weighted ``mean`` (consistent with the pipeline convention)

    See Also
    --------
    - [`MeanAbsoluteError`][yohou.metrics.point.MeanAbsoluteError] : Mean Absolute Error, average case measure
    - [`MedianAbsoluteError`][yohou.metrics.point.MedianAbsoluteError] : Median Absolute Error, robust central tendency measure

    """

    _metric_name = "max_ae"

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
        time_weighter: BaseWeighter | None = None,
        step_weighter: BaseWeighter | None = None,
        vintage_weighter: BaseWeighter | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
            time_weighter=time_weighter,
            step_weighter=step_weighter,
            vintage_weighter=vintage_weighter,
        )

    def _compute_raw_errors(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Compute per-row absolute errors."""
        return (y_truth - y_pred).select(pl.all().abs())

    def _collapse_rows(
        self,
        df: pl.DataFrame,
        context: ScoringContext | None,
        dims: set[str],
    ) -> pl.DataFrame:
        """Collapse row dimensions using max instead of mean."""
        return self._collapse_rows_with(df, context, dims, agg_fn="max")


class R2Score(BasePointScorer):
    r"""R-squared (Coefficient of Determination) metric for point forecasts.

    Computes the proportion of variance in the true values that is explained
    by the predictions. A score of 1.0 indicates perfect prediction, 0.0
    indicates performance equivalent to predicting the mean, and negative
    values indicate worse performance than predicting the mean.

    The R² is defined as:

    $$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

    where $y_i$ is the actual value, $\hat{y}_i$ is the predicted value,
    $\bar{y}$ is the mean of actual values, and $n$ is the number of observations.

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
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict).
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict).
    time_weighter : BaseWeighter or None, default=None
        Weighter applied along the time axis (observed timestamps). Because R²
        is non-linear, only zero-weight timestamps are dropped; non-zero weights
        do not otherwise reweight the score. If None, all timestamps contribute
        equally.
    step_weighter : BaseWeighter or None, default=None
        Weighter applied along the forecasting-step axis. Only zero-weight steps
        are dropped (see ``time_weighter``). If None, all forecasting steps
        contribute equally.
    vintage_weighter : BaseWeighter or None, default=None
        Weighter applied along the vintage-time axis. If None, all vintages
        contribute equally.

    Attributes
    ----------
    lower_is_better : bool
        Always False for R². Higher values indicate better fit.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import R2Score
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [10.0, 20.0, 30.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "vintage_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [12.0, 18.0, 31.0],
    ... })
    >>> r2 = R2Score()
    >>> _ = r2.fit(y_true)
    >>> r2.score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.955

    Notes
    -----
    - R² = 1.0 means perfect prediction
    - R² = 0.0 means predictions are as good as predicting the mean
    - R² < 0 means predictions are worse than predicting the mean
    - When SS_tot = 0 (constant true values), returns 0.0 by convention
    - Overrides ``score()`` because computing the denominator (SS_tot) requires
      access to the full ``y_truth`` column, not just per-row errors

    See Also
    --------
    - [`MeanSquaredError`][yohou.metrics.point.MeanSquaredError] : Mean Squared Error, the numerator component of R²
    - [`MeanAbsoluteError`][yohou.metrics.point.MeanAbsoluteError] : Mean Absolute Error, alternative regression metric

    """

    _metric_name = "r2"

    _lower_is_better = False

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
        time_weighter: BaseWeighter | None = None,
        step_weighter: BaseWeighter | None = None,
        vintage_weighter: BaseWeighter | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
            time_weighter=time_weighter,
            step_weighter=step_weighter,
            vintage_weighter=vintage_weighter,
        )

    def _compute_raw_errors(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Not reachable. R² fully overrides ``score()`` and never calls this hook."""
        raise NotImplementedError("R2Score overrides score(); _compute_raw_errors is never called.")

    def score(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
        /,
        **params,
    ) -> float | pl.DataFrame:
        """Compute R-squared score.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True values with "time" column.
        y_pred : pl.DataFrame
            Predicted values with "time" column.
        **params : dict
            Accepted for signature compatibility and ignored: scorers route no
            metadata to nested estimators.

        Returns
        -------
        float or pl.DataFrame
            R² score. 1.0 for perfect predictions, 0.0 for mean-level predictions.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, context = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        # Resolve all weighters: drop zero-weight time/step rows and store
        # vintage weights in context (mirrors the base point pipeline). R² is
        # non-linear, so continuous time/step weights are not multiplied into
        # the errors; only zero-weight rows are removed.
        y_truth, y_pred, context, _, _, _ = self._pre_filter_zero_weights(
            y_truth,
            y_pred,
            context,
            self.time_weighter,
            self.step_weighter,
            self.vintage_weighter,
        )

        def _compute_r2(yt_slice: pl.DataFrame, yp_slice: pl.DataFrame) -> pl.DataFrame:
            """Compute per-column R² score."""
            r2_values = {}
            for col in yt_slice.columns:
                truth = yt_slice[col].to_numpy().astype(np.float64)
                pred = yp_slice[col].to_numpy().astype(np.float64)
                ss_res = np.sum((truth - pred) ** 2)
                ss_tot = np.sum((truth - np.mean(truth)) ** 2)
                r2_values[col] = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
            return pl.DataFrame(r2_values).select(yt_slice.columns)

        result = self._map_per_vintage(y_truth, y_pred, context, _compute_r2)
        return self._aggregate_per_vintage_scores(result, context)


class MeanDirectionalAccuracy(BasePointScorer):
    r"""Mean Directional Accuracy metric for point forecasts.

    Computes the proportion of time steps where the predicted direction of
    change matches the actual direction of change. This metric evaluates
    whether the forecast correctly predicts upward or downward movements.

    The MDA is defined as:

    $$\text{MDA} = \frac{1}{n-1}\sum_{i=2}^{n}\mathbf{1}[\text{sign}(\Delta y_i) = \text{sign}(\Delta \hat{y}_i)]$$

    where $\Delta y_i = y_i - y_{i-1}$ and $\Delta \hat{y}_i = \hat{y}_i - \hat{y}_{i-1}$.

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
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict).
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict).
    time_weighter : BaseWeighter or None, default=None
        Weighter applied along the time axis (observed timestamps). Because MDA
        is non-linear, only zero-weight timestamps are dropped; non-zero weights
        do not otherwise reweight the score. If None, all timestamps contribute
        equally.
    step_weighter : BaseWeighter or None, default=None
        Weighter applied along the forecasting-step axis. Only zero-weight steps
        are dropped (see ``time_weighter``). If None, all forecasting steps
        contribute equally.
    vintage_weighter : BaseWeighter or None, default=None
        Weighter applied along the vintage-time axis. If None, all vintages
        contribute equally.

    Attributes
    ----------
    lower_is_better : bool
        Always False for MDA. Higher values indicate better directional prediction.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MeanDirectionalAccuracy
    >>> y_true = pl.DataFrame({
    ...     "time": [
    ...         datetime(2020, 1, 1),
    ...         datetime(2020, 1, 2),
    ...         datetime(2020, 1, 3),
    ...         datetime(2020, 1, 4),
    ...         datetime(2020, 1, 5),
    ...     ],
    ...     "value": [10.0, 15.0, 12.0, 18.0, 20.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "vintage_time": [datetime(2019, 12, 31)] * 5,
    ...     "time": [
    ...         datetime(2020, 1, 1),
    ...         datetime(2020, 1, 2),
    ...         datetime(2020, 1, 3),
    ...         datetime(2020, 1, 4),
    ...         datetime(2020, 1, 5),
    ...     ],
    ...     "value": [10.0, 14.0, 15.0, 17.0, 19.0],
    ... })
    >>> mda = MeanDirectionalAccuracy()
    >>> _ = mda.fit(y_true)
    >>> mda.score(y_true, y_pred)
    0.75

    Notes
    -----
    - MDA = 1.0 means all directional changes were predicted correctly
    - MDA = 0.5 is equivalent to random guessing for direction
    - MDA = 0.0 means all directional predictions were wrong
    - Requires at least 2 time steps (N-1 comparisons from ``.diff()``)
    - Returns 0.0 when fewer than 2 rows are available and
      ``aggregation_method='all'``; raises ``ValueError`` for all other
      aggregation settings
    - Overrides ``score()`` because computing direction requires ``.diff()``
      on the full columns, not per-row errors

    See Also
    --------
    - [`MeanAbsoluteError`][yohou.metrics.point.MeanAbsoluteError] : Error magnitude metric (not directional)
    - [`R2Score`][yohou.metrics.point.R2Score] : Variance explained metric

    """

    _metric_name = "mda"

    _lower_is_better = False

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
        time_weighter: BaseWeighter | None = None,
        step_weighter: BaseWeighter | None = None,
        vintage_weighter: BaseWeighter | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
            time_weighter=time_weighter,
            step_weighter=step_weighter,
            vintage_weighter=vintage_weighter,
        )

    def _compute_raw_errors(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Not reachable. MDA fully overrides ``score()`` and never calls this hook."""
        raise NotImplementedError("MeanDirectionalAccuracy overrides score(); _compute_raw_errors is never called.")

    def score(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
        /,
        **params,
    ) -> float | pl.DataFrame:
        """Compute Mean Directional Accuracy.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True values with "time" column.
        y_pred : pl.DataFrame
            Predicted values with "time" column.
        **params : dict
            Accepted for signature compatibility and ignored: scorers route no
            metadata to nested estimators.

        Returns
        -------
        float or pl.DataFrame
            MDA score between 0 and 1. 1.0 for perfect directional prediction.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, context = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        if len(y_truth) < 2:
            # Direction requires at least 2 rows. Only the fully-collapsed
            # (scalar) aggregation has a well-defined default; for any
            # DataFrame-shaped aggregation the return-type contract cannot be
            # honored with too few rows, so fail loudly instead.
            dims = self._normalize_agg_methods(self.aggregation_method)
            scalar_dims = {"stepwise", "vintagewise", "componentwise", "groupwise"}
            if scalar_dims.issubset(dims):
                return 0.0
            raise ValueError(
                "MeanDirectionalAccuracy requires at least 2 rows to compute a "
                f"directional change, but received {len(y_truth)}. Provide more "
                "data or use aggregation_method='all'."
            )

        # Resolve all weighters: drop zero-weight time/step rows and store
        # vintage weights in context (mirrors the base point pipeline). MDA is
        # non-linear, so continuous time/step weights are not multiplied into
        # the errors; only zero-weight rows are removed.
        y_truth, y_pred, context, _, _, _ = self._pre_filter_zero_weights(
            y_truth,
            y_pred,
            context,
            self.time_weighter,
            self.step_weighter,
            self.vintage_weighter,
        )

        def _compute_mda(yt_slice: pl.DataFrame, yp_slice: pl.DataFrame) -> pl.DataFrame | None:
            """Compute per-column mean directional accuracy."""
            if len(yt_slice) < 2:
                return None
            mda_values = {}
            for col in yt_slice.columns:
                truth_diff = np.diff(yt_slice[col].to_numpy().astype(np.float64))
                pred_diff = np.diff(yp_slice[col].to_numpy().astype(np.float64))
                matches = (np.sign(truth_diff) == np.sign(pred_diff)).astype(np.float64)
                mda_values[col] = float(np.mean(matches))
            return pl.DataFrame(mda_values).select(yt_slice.columns)

        result = self._map_per_vintage(y_truth, y_pred, context, _compute_mda)
        return self._aggregate_per_vintage_scores(result, context)
