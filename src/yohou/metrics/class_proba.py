"""Class-probability forecasting metrics for evaluating predicted distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from .base import BaseClassProbaScorer

if TYPE_CHECKING:
    from datetime import datetime

__all__ = [
    "BrierScore",
    "Accuracy",
    "LogLoss",
]


class LogLoss(BaseClassProbaScorer):
    r"""Logarithmic loss (cross-entropy) for class-probability forecasts.

    Measures the quality of predicted probability distributions by computing
    the negative log-likelihood of the true class under the predicted
    distribution.

    The log loss for a single observation is:

    $$\\text{LogLoss} = -\\frac{1}{n}\\sum_{i=1}^{n}\\log(\\hat{p}_{i,y_i})$$

    where $\\hat{p}_{i,y_i}$ is the predicted probability assigned to the
    true class $y_i$ for observation $i$.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. See `BaseClassProbaScorer`.
    groups : list of str or None, default=None
        Panel groups to include. See `BaseClassProbaScorer`.
    component_names : list of str or None, default=None
        Components to include. See `BaseClassProbaScorer`.
    group_weight : dict or None, default=None
        Panel group weights. See `BaseClassProbaScorer`.

    Attributes
    ----------
    lower_is_better : bool
        Always True for LogLoss.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import LogLoss
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "weather": ["sunny", "rainy", "cloudy"],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "weather_proba_sunny": [0.7, 0.1, 0.2],
    ...     "weather_proba_rainy": [0.2, 0.8, 0.1],
    ...     "weather_proba_cloudy": [0.1, 0.1, 0.7],
    ... })
    >>> scorer = LogLoss()
    >>> _ = scorer.fit(y_true)
    >>> scorer.score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.312...

    Notes
    -----
    - Lower values indicate better calibrated probability estimates.
    - Heavily penalizes confident wrong predictions (assigning near-zero
      probability to the true class).
    - Probabilities are clipped to ``[1e-15, 1 - 1e-15]`` to avoid
      numerical issues with ``log(0)``.

    See Also
    --------
    `BrierScore` : Multi-class Brier score.
    `Accuracy` : Classification accuracy from argmax.

    """

    _parameter_constraints: dict = {
        **BaseClassProbaScorer._parameter_constraints,
    }

    _metric_name = "log_loss"

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

    def _compute_raw_errors(self, y_truth, y_pred):
        """Compute per-row log loss values."""
        target_cols = self._extract_target_columns(y_truth)
        scores_dict: dict[str, list[float]] = {}

        for target_col in target_cols:
            proba_cols, class_labels = self._extract_class_proba_columns(y_pred, target_col)
            true_labels = y_truth[target_col].cast(pl.String)

            per_row_scores = []
            for row_idx in range(len(y_truth)):
                true_label = true_labels[row_idx]
                label_idx = class_labels.index(true_label) if true_label in class_labels else None
                if label_idx is not None:
                    prob = float(y_pred[proba_cols[label_idx]][row_idx])
                    prob = np.clip(prob, 1e-15, 1 - 1e-15)
                    per_row_scores.append(-np.log(prob))
                else:
                    per_row_scores.append(-np.log(1e-15))

            scores_dict[target_col] = per_row_scores

        return pl.DataFrame(scores_dict)


class BrierScore(BaseClassProbaScorer):
    r"""Multi-class Brier score for class-probability forecasts.

    Measures the mean squared difference between predicted probabilities and
    one-hot encoded true class labels. Equivalent to the Brier score
    generalized to multiple classes.

    The multi-class Brier score is:

    $$\\text{BS} = \\frac{1}{n}\\sum_{i=1}^{n}\\sum_{k=1}^{K}(\\hat{p}_{ik} - o_{ik})^2$$

    where $\\hat{p}_{ik}$ is the predicted probability for class $k$,
    $o_{ik}$ is 1 if class $k$ is the true class and 0 otherwise,
    and $K$ is the number of classes.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. See `BaseClassProbaScorer`.
    groups : list of str or None, default=None
        Panel groups to include. See `BaseClassProbaScorer`.
    component_names : list of str or None, default=None
        Components to include. See `BaseClassProbaScorer`.
    group_weight : dict or None, default=None
        Panel group weights. See `BaseClassProbaScorer`.

    Attributes
    ----------
    lower_is_better : bool
        Always True for BrierScore.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import BrierScore
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "weather": ["sunny", "rainy", "cloudy"],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "weather_proba_sunny": [0.7, 0.1, 0.2],
    ...     "weather_proba_rainy": [0.2, 0.8, 0.1],
    ...     "weather_proba_cloudy": [0.1, 0.1, 0.7],
    ... })
    >>> scorer = BrierScore()
    >>> _ = scorer.fit(y_true)
    >>> scorer.score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.113...

    Notes
    -----
    - Ranges from 0 (perfect) to 2 (worst possible for binary).
    - More sensitive to calibration than accuracy.
    - Proper scoring rule: optimized by the true probability distribution.

    See Also
    --------
    `LogLoss` : Logarithmic loss (cross-entropy).
    `Accuracy` : Classification accuracy from argmax.

    """

    _parameter_constraints: dict = {
        **BaseClassProbaScorer._parameter_constraints,
    }

    _metric_name = "brier_score"

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

    def _compute_raw_errors(self, y_truth, y_pred):
        """Compute per-row Brier score values."""
        target_cols = self._extract_target_columns(y_truth)
        scores_dict: dict[str, list[float]] = {}

        for target_col in target_cols:
            proba_cols, class_labels = self._extract_class_proba_columns(y_pred, target_col)
            true_labels = y_truth[target_col].cast(pl.String)

            per_row_scores = []
            for row_idx in range(len(y_truth)):
                true_label = true_labels[row_idx]
                row_score = 0.0
                for k, label in enumerate(class_labels):
                    prob = float(y_pred[proba_cols[k]][row_idx])
                    indicator = 1.0 if label == true_label else 0.0
                    row_score += (prob - indicator) ** 2
                per_row_scores.append(row_score)

            scores_dict[target_col] = per_row_scores

        return pl.DataFrame(scores_dict)


class Accuracy(BaseClassProbaScorer):
    r"""Categorical accuracy from class-probability forecasts.

    Computes the fraction of time steps where the predicted class (argmax
    of probabilities) matches the true class.

    $$\\text{Accuracy} = \\frac{1}{n}\\sum_{i=1}^{n}\\mathbb{1}[\\hat{y}_i = y_i]$$

    where $\\hat{y}_i = \\arg\\max_k \\hat{p}_{ik}$ is the predicted class.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. See `BaseClassProbaScorer`.
    groups : list of str or None, default=None
        Panel groups to include. See `BaseClassProbaScorer`.
    component_names : list of str or None, default=None
        Components to include. See `BaseClassProbaScorer`.
    group_weight : dict or None, default=None
        Panel group weights. See `BaseClassProbaScorer`.

    Attributes
    ----------
    lower_is_better : bool
        Always False for accuracy (higher is better).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import Accuracy
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "weather": ["sunny", "rainy", "cloudy"],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "weather_proba_sunny": [0.7, 0.1, 0.2],
    ...     "weather_proba_rainy": [0.2, 0.8, 0.1],
    ...     "weather_proba_cloudy": [0.1, 0.1, 0.7],
    ... })
    >>> scorer = Accuracy()
    >>> _ = scorer.fit(y_true)
    >>> scorer.score(y_true, y_pred)
    1.0

    Notes
    -----
    - Returns 1.0 for perfect predictions, 0.0 for all wrong.
    - Does not penalize prediction confidence, only correctness.
    - For proper scoring that rewards calibration, use `LogLoss` or
      `BrierScore` instead.

    See Also
    --------
    `LogLoss` : Logarithmic loss (cross-entropy).
    `BrierScore` : Multi-class Brier score.

    """

    _parameter_constraints: dict = {
        **BaseClassProbaScorer._parameter_constraints,
    }

    _metric_name = "accuracy"
    _lower_is_better = False

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

    def _compute_raw_errors(self, y_truth, y_pred):
        """Compute per-row accuracy values."""
        target_cols = self._extract_target_columns(y_truth)
        scores_dict: dict[str, list[float]] = {}

        for target_col in target_cols:
            proba_cols, class_labels = self._extract_class_proba_columns(y_pred, target_col)
            true_labels = y_truth[target_col].cast(pl.String)

            per_row_scores = []
            for row_idx in range(len(y_truth)):
                true_label = true_labels[row_idx]
                probs = [float(y_pred[pc][row_idx]) for pc in proba_cols]
                pred_label = class_labels[int(np.argmax(probs))]
                per_row_scores.append(1.0 if pred_label == true_label else 0.0)

            scores_dict[target_col] = per_row_scores

        return pl.DataFrame(scores_dict)
