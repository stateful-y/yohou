"""Point forecasting metrics for evaluating prediction accuracy."""

import numbers
from collections.abc import Callable

import numpy as np
import polars as pl
from sklearn.base import _fit_context
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from yohou.utils import inspect_locality, validate_scorer_data

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
    """Mean Absolute Error metric for point forecasts.

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
        - "timewise": Aggregate across time, return per-component DataFrame
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["timewise", "componentwise", "groupwise"].
        Example outputs:
        - "timewise" or ["timewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["timewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    panel_group_names : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
    panel_group_weight : dict or None, default=None
        Dictionary mapping panel group names to weights for weighted aggregation.
        If None, all panel groups weighted equally. Only applicable for panel data.

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
    MeanSquaredError : Mean Squared Error, more sensitive to large errors
    RootMeanSquaredError : Root Mean Squared Error, MeanSquaredError in original units
    RootMeanSquaredScaledError : Root Mean Squared Scaled Error, scale-independent version
    MAPE : Mean Absolute Percentage Error, scale-independent

    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            panel_group_names=panel_group_names,
            component_names=component_names,
            panel_group_weight=panel_group_weight,
        )

    def score(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
        /,
        time_weight: Callable | pl.DataFrame | None = None,
        **params,
    ) -> float | pl.DataFrame:
        """Compute mean absolute error.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True values with "time" column.
        y_pred : pl.DataFrame
            Predicted values with "time" column.
        time_weight : callable, pl.DataFrame, or None, default=None
            Time-based evaluation weights. See Notes for format details.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        float or pl.DataFrame
            If aggregation_method=["timewise"], returns DataFrame with per-component MAE values.
            If aggregation_method=["componentwise"], returns DataFrame with "time" and "mae" columns.
            If aggregation_method="all", returns scalar float.

        Notes
        -----
        **Time Weight Formats**:

        1. **DataFrame**: Must have "time" column plus weight columns:
           - Global data: "weight" column
           - Panel data: "{group}_weight" columns (one per group) or "weight" (applies to all groups)

        2. **Callable (1 parameter)**: `f(time: pl.Series) -> pl.Series`
           - Returns weights for each time step
           - Same weights applied to all panel groups

        3. **Callable (2 parameters)**: `f(time: pl.Series, group_name: str | None) -> pl.Series`
           - Panel-aware: different weights per group
           - group_name is None for global data

        **Weight Properties**:
        - Must be non-negative, finite, and sum to positive value
        - Normalized internally to preserve scale
        - Applied to per-timestep errors before aggregation

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, time_values = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        # Compute raw per-timestep per-component absolute errors
        scores = (y_truth - y_pred).select(pl.all().abs())

        # Apply time weights if provided
        if time_weight is not None:
            _, panel_groups = inspect_locality(scores)

            if len(panel_groups) > 0:
                # Panel data: apply weights per group
                weighted_parts = []
                for group_name, group_cols in panel_groups.items():
                    group_scores = scores.select(group_cols)
                    weighted_group = self._process_time_weights(group_scores, time_weight, time_values, group_name)
                    weighted_parts.append(weighted_group)
                scores = pl.concat(weighted_parts, how="horizontal")
            else:
                # Global data: apply weights directly
                scores = self._process_time_weights(scores, time_weight, time_values, group_name=None)

        # Apply aggregation strategy from base class
        result = self._aggregate_scores(scores, time_values=time_values)

        # Customize column name for componentwise aggregation
        if "componentwise" in (
            self.aggregation_method if isinstance(self.aggregation_method, list) else []
        ) and isinstance(result, pl.DataFrame):
            # Rename global and panel columns
            rename_map = {}
            if "score" in result.columns:
                rename_map["score"] = "mae"
            for col in result.columns:
                if col.endswith("__score"):
                    rename_map[col] = col.replace("__score", "__mae")

            if rename_map:
                result = result.rename(rename_map)

        return result


class MeanSquaredError(BasePointScorer):
    """Mean Squared Error metric for point forecasts.

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
        - "timewise": Aggregate across time, return per-component DataFrame
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["timewise", "componentwise", "groupwise"].
        Example outputs:
        - "timewise" or ["timewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["timewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    panel_group_names : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data. Validated at fit time.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
        Validated at fit time.
    panel_group_weight : dict or None, default=None
        Weights for panel groups. See BaseScorer for details.

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
    MeanAbsoluteError : Mean Absolute Error, less sensitive to outliers
    RootMeanSquaredError : Root Mean Squared Error, MeanSquaredError in original units
    RootMeanSquaredScaledError : Root Mean Squared Scaled Error, scale-independent version

    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            panel_group_names=panel_group_names,
            component_names=component_names,
            panel_group_weight=panel_group_weight,
        )

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **params) -> float | pl.DataFrame:
        """Compute mean squared error.

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
            If aggregation_method=["timewise"], returns DataFrame with per-component MSE values.
            If aggregation_method=["componentwise"], returns DataFrame with "time" and "mse" columns.
            If aggregation_method="all", returns scalar float.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, time_values = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        # Compute raw per-timestep per-component squared errors
        scores = (y_truth - y_pred).select(pl.all().pow(2))

        # Apply aggregation strategy from base class
        result = self._aggregate_scores(scores, time_values=time_values)

        # Customize column name for componentwise aggregation
        if "componentwise" in (
            self.aggregation_method if isinstance(self.aggregation_method, list) else []
        ) and isinstance(result, pl.DataFrame):
            # Rename global and panel columns
            rename_map = {}
            if "score" in result.columns:
                rename_map["score"] = "mse"
            for col in result.columns:
                if col.endswith("__score"):
                    rename_map[col] = col.replace("__score", "__mse")

            if rename_map:
                result = result.rename(rename_map)

        return result


class RootMeanSquaredError(BasePointScorer):
    """Root Mean Squared Error metric for point forecasts.

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
        - "timewise": Aggregate across time, return per-component DataFrame
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["timewise", "componentwise", "groupwise"].
        Example outputs:
        - "timewise" or ["timewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["timewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    panel_group_names : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
    panel_group_weight : dict or None, default=None
        Dictionary mapping panel group names to weights for weighted aggregation.
        If None, all panel groups weighted equally. Only applicable for panel data.

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
    MeanAbsoluteError : Mean Absolute Error, less sensitive to outliers
    MeanSquaredError : Mean Squared Error, RMSE squared
    RootMeanSquaredScaledError : Root Mean Squared Scaled Error, scale-independent version

    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            panel_group_names=panel_group_names,
            component_names=component_names,
            panel_group_weight=panel_group_weight,
        )

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **params) -> float | pl.DataFrame:
        """Compute root mean squared error.

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
            If aggregate="timewise", returns DataFrame with per-component RMSE values.
            If aggregate="componentwise", returns DataFrame with "time" and "rmse" columns.
            If aggregate="both", returns scalar float.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, time_values = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        # Compute raw per-timestep per-component squared errors
        squared_errors = (y_truth - y_pred).select(pl.all().pow(2))

        # Apply aggregation strategy from base class
        result = self._aggregate_scores(squared_errors, time_values=time_values)

        # Take square root of the aggregated values
        if isinstance(result, pl.DataFrame):
            if "timewise" in (self.aggregation_method if isinstance(self.aggregation_method, list) else []):
                # Per-component: sqrt each component's mean
                result = result.select(pl.all().sqrt())
            elif "componentwise" in (self.aggregation_method if isinstance(self.aggregation_method, list) else []):
                # Per-timestep: sqrt each timestep's score, rename column
                target_cols = [col for col in result.columns if col == "score" or col.endswith("__score")]
                if target_cols:
                    result = result.with_columns([pl.col(c).sqrt() for c in target_cols])

                    rename_map = {}
                    if "score" in result.columns:
                        rename_map["score"] = "rmse"
                    for col in result.columns:
                        if col.endswith("__score"):
                            rename_map[col] = col.replace("__score", "__rmse")

                    if rename_map:
                        result = result.rename(rename_map)
        else:
            # Scalar: sqrt the overall mean
            result = float(np.sqrt(result))

        return result


class RootMeanSquaredScaledError(BasePointScorer):
    """Root Mean Squared Scaled Error metric for point forecasts.

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
        - "timewise": Aggregate across time, return per-component DataFrame
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["timewise", "componentwise", "groupwise"].
        Example outputs:
        - "timewise" or ["timewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["timewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    panel_group_names : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
    panel_group_weight : dict or None, default=None
        Dictionary mapping panel group names to weights for weighted aggregation.
        If None, all panel groups weighted equally. Only applicable for panel data.

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
    RootMeanSquaredError : Root Mean Squared Error, non-scaled version
    MeanAbsoluteError : Mean Absolute Error, non-scaled alternative
    MeanSquaredError : Mean Squared Error, squared version

    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
        "seasonality": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        seasonality: int = 1,
        aggregation_method: list[str] | str = "all",
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            panel_group_names=panel_group_names,
            component_names=component_names,
            panel_group_weight=panel_group_weight,
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
    def fit(self, y_train: pl.DataFrame, **params) -> "RootMeanSquaredScaledError":
        """Fit the scorer by computing per-column scaling factors.

        Parameters
        ----------
        y_train : pl.DataFrame
            Training set target values with "time" column.
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
        # Call parent fit() to validate parameters (aggregation_method, panel_group_names, etc.)
        super().fit(y_train, **params)

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

            # Store non-zero scale (avoid division by zero in score())
            self.scales_[col] = max(scale, 1e-10)

        return self

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **params) -> float | pl.DataFrame:
        """Compute root mean squared scaled error.

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
            If aggregate="timewise", returns DataFrame with per-component RootMeanSquaredScaledError values.
            If aggregate="componentwise", returns DataFrame with "time" and "rmsse" columns.
            If aggregate="both", returns scalar float.

        Raises
        ------
        NotFittedError
            If fit() has not been called.

        """
        check_is_fitted(self, ["scales_"])

        y_truth, y_pred, time_values = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        # Compute raw per-timestep per-component squared scaled errors
        scaled_squared_errors_data = {}
        for col in y_truth.columns:
            errors = (y_truth[col] - y_pred[col]).to_numpy()
            scale = self.scales_[col]
            # Compute squared scaled errors per timestep
            scaled_squared_errors_data[col] = (errors / np.sqrt(scale)) ** 2
        scaled_squared_errors = pl.DataFrame(scaled_squared_errors_data)

        # Apply aggregation strategy from base class
        result = self._aggregate_scores(scaled_squared_errors, time_values=time_values)

        # Take square root of the aggregated values
        if isinstance(result, pl.DataFrame):
            if "timewise" in (self.aggregation_method if isinstance(self.aggregation_method, list) else []):
                # Per-component: sqrt each component's mean
                result = result.select(pl.all().sqrt())
            elif "componentwise" in (self.aggregation_method if isinstance(self.aggregation_method, list) else []):
                # Per-timestep: sqrt each timestep's score, rename column
                target_cols = [col for col in result.columns if col == "score" or col.endswith("__score")]
                if target_cols:
                    result = result.with_columns([pl.col(c).sqrt() for c in target_cols])

                    rename_map = {}
                    if "score" in result.columns:
                        rename_map["score"] = "rmsse"
                    for col in result.columns:
                        if col.endswith("__score"):
                            rename_map[col] = col.replace("__score", "__rmsse")

                    if rename_map:
                        result = result.rename(rename_map)
        else:
            # Scalar: sqrt the overall mean
            result = float(np.sqrt(result))

        return result


class MeanAbsolutePercentageError(BasePointScorer):
    """Mean Absolute Percentage Error metric for point forecasts.

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
        - "timewise": Aggregate across time, return per-component DataFrame
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["timewise", "componentwise", "groupwise"].
        Example outputs:
        - "timewise" or ["timewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["timewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    panel_group_names : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
    panel_group_weight : dict or None, default=None
        Dictionary mapping panel group names to weights for weighted aggregation.
        If None, all panel groups weighted equally. Only applicable for panel data.

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
    SymmetricMeanAbsolutePercentageError : Symmetric version of MAPE
    MeanAbsoluteError : Absolute error in original units
    MeanAbsoluteScaledError : Scaled by naive forecast error

    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
        "epsilon": [Interval(numbers.Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        epsilon: float = 1e-8,
        aggregation_method: list[str] | str = "all",
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            panel_group_names=panel_group_names,
            component_names=component_names,
            panel_group_weight=panel_group_weight,
        )
        self.epsilon = epsilon

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **params) -> float | pl.DataFrame:
        """Compute mean absolute percentage error.

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
            If aggregation_method includes "timewise", returns DataFrame with per-component MAPE values.
            If aggregation_method includes "componentwise", returns DataFrame with "time" and "mape" columns.
            If aggregation_method="all", returns scalar float.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, time_values = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        # Compute per-timestep per-component percentage errors
        pct_errors_data = {}
        for col in y_truth.columns:
            abs_errors = (y_truth[col] - y_pred[col]).abs()
            pct_errors_data[col] = (abs_errors / (y_truth[col].abs() + self.epsilon)) * 100.0
        pct_errors = pl.DataFrame(pct_errors_data)

        # Apply aggregation strategy from base class
        result = self._aggregate_scores(pct_errors, time_values=time_values)

        # Rename score columns to mape
        if isinstance(result, pl.DataFrame):
            rename_map = {}
            if "score" in result.columns:
                rename_map["score"] = "mape"
            for col in result.columns:
                if col.endswith("__score"):
                    rename_map[col] = col.replace("__score", "__mape")

            if rename_map:
                result = result.rename(rename_map)

        return result


class SymmetricMeanAbsolutePercentageError(BasePointScorer):
    """Symmetric Mean Absolute Percentage Error metric for point forecasts.

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
        - "timewise": Aggregate across time, return per-component DataFrame
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["timewise", "componentwise", "groupwise"].
        Example outputs:
        - "timewise" or ["timewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["timewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    panel_group_names : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
    panel_group_weight : dict or None, default=None
        Dictionary mapping panel group names to weights for weighted aggregation.
        If None, all panel groups weighted equally. Only applicable for panel data.

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
    MeanAbsolutePercentageError : Asymmetric version of percentage error
    MeanAbsoluteError : Absolute error in original units
    MeanAbsoluteScaledError : Scaled by naive forecast error

    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
        "epsilon": [Interval(numbers.Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        epsilon: float = 1e-8,
        aggregation_method: list[str] | str = "all",
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            panel_group_names=panel_group_names,
            component_names=component_names,
            panel_group_weight=panel_group_weight,
        )
        self.epsilon = epsilon

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **params) -> float | pl.DataFrame:
        """Compute symmetric mean absolute percentage error.

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
            If aggregation_method includes "timewise", returns DataFrame with per-component sMAPE values.
            If aggregation_method includes "componentwise", returns DataFrame with "time" and "smape" columns.
            If aggregation_method="all", returns scalar float.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, time_values = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        # Compute per-timestep per-component symmetric percentage errors
        smape_errors_data = {}
        for col in y_truth.columns:
            abs_errors = (y_truth[col] - y_pred[col]).abs()
            denominator = (y_truth[col].abs() + y_pred[col].abs()) / 2.0 + self.epsilon
            smape_errors_data[col] = (abs_errors / denominator) * 100.0
        smape_errors = pl.DataFrame(smape_errors_data)

        # Apply aggregation strategy from base class
        result = self._aggregate_scores(smape_errors, time_values=time_values)

        # Rename score columns to smape
        if isinstance(result, pl.DataFrame):
            rename_map = {}
            if "score" in result.columns:
                rename_map["score"] = "smape"
            for col in result.columns:
                if col.endswith("__score"):
                    rename_map[col] = col.replace("__score", "__smape")

            if rename_map:
                result = result.rename(rename_map)

        return result


class MeanAbsoluteScaledError(BasePointScorer):
    """Mean Absolute Scaled Error metric for point forecasts.

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
        - "timewise": Aggregate across time, return per-component DataFrame
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["timewise", "componentwise", "groupwise"].
        Example outputs:
        - "timewise" or ["timewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["timewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    panel_group_names : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
    panel_group_weight : dict or None, default=None
        Dictionary mapping panel group names to weights for weighted aggregation.
        If None, all panel groups weighted equally. Only applicable for panel data.

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
    RootMeanSquaredScaledError : Squared error version of scaled metric
    MeanAbsoluteError : Non-scaled MAE
    MeanAbsolutePercentageError : Percentage-based scale-independent metric

    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
        "seasonality": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        seasonality: int = 1,
        aggregation_method: list[str] | str = "all",
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            panel_group_names=panel_group_names,
            component_names=component_names,
            panel_group_weight=panel_group_weight,
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
    def fit(self, y_train: pl.DataFrame, **params) -> "MeanAbsoluteScaledError":
        """Fit the scorer by computing per-column scaling factors.

        Parameters
        ----------
        y_train : pl.DataFrame
            Training set target values with "time" column.
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
        # Call parent fit() to validate parameters (aggregation_method, panel_group_names, etc.)
        super().fit(y_train, **params)

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

            # Store non-zero scale (avoid division by zero in score())
            self.naive_errors_[col] = max(scale, 1e-10)

        return self

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **params) -> float | pl.DataFrame:
        """Compute mean absolute scaled error.

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
            If aggregation_method includes "timewise", returns DataFrame with per-component MASE values.
            If aggregation_method includes "componentwise", returns DataFrame with "time" and "mase" columns.
            If aggregation_method="all", returns scalar float.

        Raises
        ------
        NotFittedError
            If fit() has not been called.

        """
        check_is_fitted(self, ["naive_errors_"])

        y_truth, y_pred, time_values = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        # Compute raw per-timestep per-component scaled absolute errors
        scaled_errors_data = {}
        for col in y_truth.columns:
            errors = (y_truth[col] - y_pred[col]).abs().to_numpy()
            scale = self.naive_errors_[col]
            # Compute scaled absolute errors per timestep
            scaled_errors_data[col] = errors / scale
        scaled_errors = pl.DataFrame(scaled_errors_data)

        # Apply aggregation strategy from base class
        result = self._aggregate_scores(scaled_errors, time_values=time_values)

        # Rename score columns to mase
        if isinstance(result, pl.DataFrame):
            rename_map = {}
            if "score" in result.columns:
                rename_map["score"] = "mase"
            for col in result.columns:
                if col.endswith("__score"):
                    rename_map[col] = col.replace("__score", "__mase")

            if rename_map:
                result = result.rename(rename_map)

        return result


class MedianAbsoluteError(BasePointScorer):
    """Median Absolute Error metric for point forecasts.

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
        - "timewise": Aggregate across time, return per-component DataFrame
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["timewise", "componentwise", "groupwise"].
        Example outputs:
        - "timewise" or ["timewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["timewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    panel_group_names : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
    panel_group_weight : dict or None, default=None
        Dictionary mapping panel group names to weights for weighted aggregation.
        If None, all panel groups weighted equally. Only applicable for panel data.

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
    MeanAbsoluteError : Mean-based absolute error, more sensitive to outliers
    MaxError : Maximum absolute error, worst-case measure

    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            panel_group_names=panel_group_names,
            component_names=component_names,
            panel_group_weight=panel_group_weight,
        )

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **params) -> float | pl.DataFrame:
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
            If aggregation_method includes "timewise", returns DataFrame with per-component MedianAE values.
            If aggregation_method includes "componentwise", returns DataFrame with "time" and "median_ae" columns.
            If aggregation_method="all", returns scalar float.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, time_values = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        # Compute per-timestep per-component absolute errors
        abs_errors = (y_truth - y_pred).select(pl.all().abs())

        # Apply median aggregation
        agg_method = self.aggregation_method
        if isinstance(agg_method, str):
            agg_method = [agg_method]

        if "all" in agg_method or (
            "timewise" in agg_method
            and "componentwise" in agg_method
            and ("groupwise" in agg_method or self.panel_group_names is None)
        ):
            # Fully aggregated: global median
            result = float(abs_errors.select(pl.all().median()).to_numpy().flatten().mean())
        elif "timewise" in agg_method:
            # Per-component: median across time
            result = abs_errors.select(pl.all().median())
        elif "componentwise" in agg_method:
            # Per-timestep: median across components
            # Compute median per row
            result = abs_errors.select(pl.concat_list(pl.all()).alias("errors")).select(
                pl.col("errors").list.eval(pl.element().median()).list.first().alias("score")
            )
            # Add time column back
            if time_values is not None:
                result = result.with_columns(pl.Series("time", time_values).cast(pl.Datetime))
                # Ensure time is first
                result = result.select(["time"] + [c for c in result.columns if c != "time"])
        else:
            # Default: use base class aggregation with median
            # This handles panel data and other edge cases
            result = abs_errors.select(pl.all().median())

        # Rename score columns to median_ae
        if isinstance(result, pl.DataFrame):
            rename_map = {}
            if "score" in result.columns:
                rename_map["score"] = "median_ae"
            for col in result.columns:
                if col.endswith("__score"):
                    rename_map[col] = col.replace("__score", "__median_ae")

            if rename_map:
                result = result.rename(rename_map)

        return result
