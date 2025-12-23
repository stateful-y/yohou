"""Point forecasting metrics for evaluating prediction accuracy."""

import numpy as np
import polars as pl

from .base import BasePointScorer


class MAE(BasePointScorer):
    """Mean Absolute Error metric for point forecasts.

    Computes the average of absolute differences between predictions and actual values.
    This metric is robust to outliers and provides intuitive interpretation in the
    original units of the target variable.

    The MAE is defined as:

    $$\text{MAE} = \frac{1}{n}\\sum_{i=1}^{n}|y_i - \\hat{y}_i|$$

    where $y_i$ is the actual value, $\\hat{y}_i$ is the predicted value, and
    $n$ is the number of observations.

    Attributes
    ----------
    lower_is_better : bool
        Always True for MAE.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MAE
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [10.0, 20.0, 30.0]
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [12.0, 19.0, 28.0]
    ... })
    >>> mae = MAE()
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
    MSE : Mean Squared Error, more sensitive to large errors
    RMSE : Root Mean Squared Error, MSE in original units
    MAPE : Mean Absolute Percentage Error, scale-independent

    """

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> float:
        """Compute mean absolute error.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True values.

        y_pred : pl.DataFrame
            Predicted values.

        Returns
        -------
        float
            Mean absolute error.

        """
        y_truth, y_pred = self._validate_inputs(y_truth, y_pred)

        score = (y_truth - y_pred).select(pl.all().abs().mean())

        score_value: float = float(np.mean(score.rows()))

        return score_value


class MSE(BasePointScorer):
    """Mean Squared Error metric for point forecasts.

    Computes the average of squared differences between predictions and actual values.
    This metric heavily penalizes large errors, making it sensitive to outliers.

    The MSE is defined as:

    $$\text{MSE} = \frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2$$

    where $y_i$ is the actual value, $\\hat{y}_i$ is the predicted value, and
    $n$ is the number of observations.

    Attributes
    ----------
    lower_is_better : bool
        Always True for MSE.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MSE
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [10.0, 20.0, 30.0]
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [12.0, 19.0, 28.0]
    ... })
    >>> mse = MSE()
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
    MAE : Mean Absolute Error, less sensitive to outliers
    RMSE : Root Mean Squared Error, MSE in original units

    """

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> float:
        """Compute mean squared error.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True values.

        y_pred : pl.DataFrame
            Predicted values.

        Returns
        -------
        float
            Mean squared error.

        """
        y_truth, y_pred = self._validate_inputs(y_truth, y_pred)

        score = (y_truth - y_pred).select(pl.all().pow(2).mean())

        score_value: float = float(np.mean(score.rows()))

        return score_value
