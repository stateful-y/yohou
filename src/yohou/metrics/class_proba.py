"""Class-probability forecasting metrics for evaluating predicted distributions."""

from collections.abc import Callable

import numpy as np
import polars as pl
from sklearn.utils.validation import check_is_fitted

from yohou.utils import inspect_panel, validate_scorer_data

from .base import BaseClassProbaScorer

__all__ = [
    "BrierScore",
    "Accuracy",
    "LogLoss",
]


class LogLoss(BaseClassProbaScorer):
    """Logarithmic loss (cross-entropy) for class-probability forecasts.

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
    panel_group_names : list of str or None, default=None
        Panel groups to include. See `BaseClassProbaScorer`.
    component_names : list of str or None, default=None
        Components to include. See `BaseClassProbaScorer`.
    panel_group_weight : dict or None, default=None
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
    0.277...

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
        """Compute log loss.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True class labels with "time" column.
        y_pred : pl.DataFrame
            Predicted probabilities with "time" and ``{target}_proba_{class}``
            columns.
        time_weight : callable, pl.DataFrame, or None, default=None
            Time-based evaluation weights.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        float or pl.DataFrame
            Log loss score.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, time_values = validate_scorer_data(self, y_truth, y_pred)

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

        scores = pl.DataFrame(scores_dict)

        if time_weight is not None:
            _, panel_groups = inspect_panel(scores)
            if len(panel_groups) > 0:
                weighted_parts = []
                for group_name, group_cols in panel_groups.items():
                    group_scores = scores.select(group_cols)
                    weighted_group = self._process_time_weights(group_scores, time_weight, time_values, group_name)
                    weighted_parts.append(weighted_group)
                scores = pl.concat(weighted_parts, how="horizontal")
            else:
                scores = self._process_time_weights(scores, time_weight, time_values, group_name=None)

        result = self._aggregate_scores(scores, time_values=time_values)

        if "componentwise" in (
            self.aggregation_method if isinstance(self.aggregation_method, list) else []
        ) and isinstance(result, pl.DataFrame):
            rename_map = {}
            if "score" in result.columns:
                rename_map["score"] = "log_loss"
            for col in result.columns:
                if col.endswith("__score"):
                    rename_map[col] = col.replace("__score", "__log_loss")
            if rename_map:
                result = result.rename(rename_map)

        return result


class BrierScore(BaseClassProbaScorer):
    """Multi-class Brier score for class-probability forecasts.

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
    panel_group_names : list of str or None, default=None
        Panel groups to include. See `BaseClassProbaScorer`.
    component_names : list of str or None, default=None
        Components to include. See `BaseClassProbaScorer`.
    panel_group_weight : dict or None, default=None
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
    0.18...

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
        """Compute multi-class Brier score.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True class labels with "time" column.
        y_pred : pl.DataFrame
            Predicted probabilities with "time" and ``{target}_proba_{class}``
            columns.
        time_weight : callable, pl.DataFrame, or None, default=None
            Time-based evaluation weights.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        float or pl.DataFrame
            Brier score.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, time_values = validate_scorer_data(self, y_truth, y_pred)

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

        scores = pl.DataFrame(scores_dict)

        if time_weight is not None:
            _, panel_groups = inspect_panel(scores)
            if len(panel_groups) > 0:
                weighted_parts = []
                for group_name, group_cols in panel_groups.items():
                    group_scores = scores.select(group_cols)
                    weighted_group = self._process_time_weights(group_scores, time_weight, time_values, group_name)
                    weighted_parts.append(weighted_group)
                scores = pl.concat(weighted_parts, how="horizontal")
            else:
                scores = self._process_time_weights(scores, time_weight, time_values, group_name=None)

        result = self._aggregate_scores(scores, time_values=time_values)

        if "componentwise" in (
            self.aggregation_method if isinstance(self.aggregation_method, list) else []
        ) and isinstance(result, pl.DataFrame):
            rename_map = {}
            if "score" in result.columns:
                rename_map["score"] = "brier_score"
            for col in result.columns:
                if col.endswith("__score"):
                    rename_map[col] = col.replace("__score", "__brier_score")
            if rename_map:
                result = result.rename(rename_map)

        return result


class Accuracy(BaseClassProbaScorer):
    """Categorical accuracy from class-probability forecasts.

    Computes the fraction of time steps where the predicted class (argmax
    of probabilities) matches the true class.

    $$\\text{Accuracy} = \\frac{1}{n}\\sum_{i=1}^{n}\\mathbb{1}[\\hat{y}_i = y_i]$$

    where $\\hat{y}_i = \\arg\\max_k \\hat{p}_{ik}$ is the predicted class.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. See `BaseClassProbaScorer`.
    panel_group_names : list of str or None, default=None
        Panel groups to include. See `BaseClassProbaScorer`.
    component_names : list of str or None, default=None
        Components to include. See `BaseClassProbaScorer`.
    panel_group_weight : dict or None, default=None
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

    @property
    def lower_is_better(self) -> bool:
        """Whether lower scores are better.

        Returns
        -------
        bool
            Always False for accuracy.

        """
        return False

    def __sklearn_tags__(self):
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with ``lower_is_better=False``.

        """
        tags = super().__sklearn_tags__()
        assert tags.scorer_tags is not None
        tags.scorer_tags.lower_is_better = False
        return tags

    def score(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
        /,
        time_weight: Callable | pl.DataFrame | None = None,
        **params,
    ) -> float | pl.DataFrame:
        """Compute categorical accuracy.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True class labels with "time" column.
        y_pred : pl.DataFrame
            Predicted probabilities with "time" and ``{target}_proba_{class}``
            columns.
        time_weight : callable, pl.DataFrame, or None, default=None
            Time-based evaluation weights.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        float or pl.DataFrame
            Accuracy score.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, time_values = validate_scorer_data(self, y_truth, y_pred)

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

        scores = pl.DataFrame(scores_dict)

        if time_weight is not None:
            _, panel_groups = inspect_panel(scores)
            if len(panel_groups) > 0:
                weighted_parts = []
                for group_name, group_cols in panel_groups.items():
                    group_scores = scores.select(group_cols)
                    weighted_group = self._process_time_weights(group_scores, time_weight, time_values, group_name)
                    weighted_parts.append(weighted_group)
                scores = pl.concat(weighted_parts, how="horizontal")
            else:
                scores = self._process_time_weights(scores, time_weight, time_values, group_name=None)

        result = self._aggregate_scores(scores, time_values=time_values)

        if "componentwise" in (
            self.aggregation_method if isinstance(self.aggregation_method, list) else []
        ) and isinstance(result, pl.DataFrame):
            rename_map = {}
            if "score" in result.columns:
                rename_map["score"] = "accuracy"
            for col in result.columns:
                if col.endswith("__score"):
                    rename_map[col] = col.replace("__score", "__accuracy")
            if rename_map:
                result = result.rename(rename_map)

        return result
