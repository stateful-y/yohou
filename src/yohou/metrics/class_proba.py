"""Class-probability forecasting metrics for evaluating predicted distributions."""

from __future__ import annotations

import numpy as np
import polars as pl

from yohou.weighting import BaseWeighter

from .base import BaseClassProbaScorer

__all__ = [
    "BrierScore",
    "LogLoss",
    "RankedProbabilityScore",
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
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict). See `BaseClassProbaScorer`.
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict). See `BaseClassProbaScorer`.

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
    ...     "vintage_time": [datetime(2019, 12, 31)] * 3,
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
    - [`BrierScore`][yohou.metrics.class_proba.BrierScore] : Multi-class Brier score.
    - [`Accuracy`][yohou.metrics.classification.Accuracy] : Classification accuracy from argmax.

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

    def _compute_raw_errors(self, y_truth, y_pred):
        """Compute per-row log loss values."""
        target_cols = self._extract_target_columns(y_truth)
        scores_dict: dict[str, np.ndarray] = {}

        for target_col in target_cols:
            proba_cols, class_labels = self._extract_class_proba_columns(y_pred, target_col)
            true_arr = y_truth[target_col].cast(pl.String).to_numpy().astype(str)
            labels_arr = np.array(class_labels)
            proba_arr = y_pred.select(proba_cols).to_numpy()  # (n, K)

            # One-hot selector for the true class; unknown labels select nothing.
            one_hot = true_arr[:, None] == labels_arr[None, :]  # (n, K)
            true_prob = (proba_arr * one_hot).sum(axis=1)  # 0.0 when label unknown
            true_prob = np.clip(true_prob, 1e-15, 1 - 1e-15)
            scores_dict[target_col] = -np.log(true_prob)

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
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict). See `BaseClassProbaScorer`.
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict). See `BaseClassProbaScorer`.

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
    ...     "vintage_time": [datetime(2019, 12, 31)] * 3,
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
    - [`LogLoss`][yohou.metrics.class_proba.LogLoss] : Logarithmic loss (cross-entropy).
    - [`Accuracy`][yohou.metrics.classification.Accuracy] : Classification accuracy from argmax.

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

    def _compute_raw_errors(self, y_truth, y_pred):
        """Compute per-row Brier score values."""
        target_cols = self._extract_target_columns(y_truth)
        scores_dict: dict[str, np.ndarray] = {}

        for target_col in target_cols:
            proba_cols, class_labels = self._extract_class_proba_columns(y_pred, target_col)
            true_arr = y_truth[target_col].cast(pl.String).to_numpy().astype(str)
            labels_arr = np.array(class_labels)
            proba_arr = y_pred.select(proba_cols).to_numpy()  # (n, K)

            one_hot = (true_arr[:, None] == labels_arr[None, :]).astype(np.float64)  # (n, K)
            scores_dict[target_col] = np.sum((proba_arr - one_hot) ** 2, axis=1)

        return pl.DataFrame(scores_dict)


class RankedProbabilityScore(BaseClassProbaScorer):
    r"""Ranked Probability Score for class-probability forecasts.

    Measures the quality of predicted probability distributions for ordered
    (ordinal) classes by comparing cumulative probability distributions.
    Generalizes the Brier score to ordinal multi-class settings by penalizing
    predictions that place probability mass far from the true class.

    The RPS for a single observation is:

    $$\text{RPS} = \frac{1}{K-1}\sum_{k=1}^{K-1}\left(\sum_{j=1}^{k}\hat{p}_{j} - \sum_{j=1}^{k}o_{j}\right)^2$$

    where $\hat{p}_j$ is the predicted probability for class $j$, $o_j$ is
    1 if the true class is $j$ and 0 otherwise, and $K$ is the number of
    classes. The normalization by $K-1$ follows the standard forecasting
    convention.

    Parameters
    ----------
    class_order : list of str or None, default=None
        Explicit ordering of class labels for the cumulative sum. When
        None, classes are ordered by their column order in ``y_pred``
        (i.e. the ``{target}_proba_{class}`` column order).
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. See `BaseClassProbaScorer`.
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict).
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict).

    Attributes
    ----------
    lower_is_better : bool
        Always True for RPS.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import RankedProbabilityScore
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "weather": ["sunny", "rainy", "cloudy"],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "vintage_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "weather_proba_sunny": [0.7, 0.1, 0.2],
    ...     "weather_proba_rainy": [0.2, 0.8, 0.1],
    ...     "weather_proba_cloudy": [0.1, 0.1, 0.7],
    ... })
    >>> scorer = RankedProbabilityScore()
    >>> _ = scorer.fit(y_true)
    >>> scorer.score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.041...

    Notes
    -----
    - RPS is a proper scoring rule for ordinal outcomes.
    - For K=2, RPS equals the Brier score (up to normalization).
    - Sensitive to the distance between predicted and true class in the
      ordinal ranking, unlike Brier score which treats all misclassifications
      equally.
    - The ``class_order`` parameter lets you specify a meaningful ordering
      for ordinal variables (e.g. ``["low", "medium", "high"]``).

    See Also
    --------
    - [`BrierScore`][yohou.metrics.class_proba.BrierScore] : Brier score (unordered multi-class).
    - [`LogLoss`][yohou.metrics.class_proba.LogLoss] : Logarithmic loss (cross-entropy).

    """

    _parameter_constraints: dict = {
        **BaseClassProbaScorer._parameter_constraints,
        "class_order": [list, None],
    }

    _metric_name = "rps"

    def __init__(
        self,
        class_order: list[str] | None = None,
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
        self.class_order = class_order

    def _compute_raw_errors(self, y_truth, y_pred):
        """Compute per-row RPS values."""
        target_cols = self._extract_target_columns(y_truth)
        scores_dict: dict[str, list[float]] = {}

        for target_col in target_cols:
            proba_cols, class_labels = self._extract_class_proba_columns(y_pred, target_col)
            true_labels = y_truth[target_col].cast(pl.String)

            # Determine class order
            if self.class_order is not None:
                order = self.class_order
                # Reorder proba_cols to match class_order
                label_to_col = dict(zip(class_labels, proba_cols, strict=True))
                unknown = [label for label in order if label not in label_to_col]
                if unknown:
                    raise ValueError(
                        f"class_order contains labels not found in y_pred for target "
                        f"'{target_col}': {unknown}. Available class labels: {class_labels}."
                    )
                ordered_cols = [label_to_col[label] for label in order]
                ordered_labels = order
            else:
                ordered_cols = proba_cols
                ordered_labels = class_labels

            k = len(ordered_labels)
            norm = max(k - 1, 1)  # Avoid division by zero for K=1

            # Vectorized computation
            proba_arr = y_pred.select(ordered_cols).to_numpy()  # (n, K)
            true_arr = true_labels.to_numpy().astype(str)
            labels_arr = np.array(ordered_labels)
            one_hot = (true_arr[:, None] == labels_arr[None, :]).astype(np.float64)  # (n, K)

            cum_pred = np.cumsum(proba_arr, axis=1)[:, :-1]  # (n, K-1)
            cum_true = np.cumsum(one_hot, axis=1)[:, :-1]

            scores_dict[target_col] = (np.sum((cum_pred - cum_true) ** 2, axis=1) / norm).tolist()

        return pl.DataFrame(scores_dict)
