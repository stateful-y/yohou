"""Hard-label and ranking classification metrics."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from yohou.metrics.base import BaseClassProbaScorer, BaseHardLabelScorer, BaseRankingScorer

__all__ = [
    "Accuracy",
    "FBetaScore",
    "Precision",
    "PRAuC",
    "Recall",
    "ROCAuC",
]


class Precision(BaseHardLabelScorer):
    r"""Precision from class-probability forecasts.

    Computes precision (positive predictive value) by argmaxing predicted
    probabilities into hard labels and computing the ratio of true
    positives to all predicted positives.

    $$\text{Precision}_k = \frac{TP_k}{TP_k + FP_k}$$

    Parameters
    ----------
    average : str, default="macro"
        Class averaging strategy: ``"macro"`` (unweighted mean across
        classes), ``"micro"`` (aggregate counts first), or ``"weighted"``
        (support-weighted mean).
    zero_division : float, default=0.0
        Value returned when the denominator is zero.
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over.
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter or filter with weights.
    components : list of str, dict of str to float, or None, default=None
        Component filter or filter with weights.

    Attributes
    ----------
    lower_is_better : bool
        Always False (higher precision is better).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics.classification import Precision
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "weather": ["sunny", "rainy", "cloudy", "sunny", "rainy"],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "vintage_time": [datetime(2019, 12, 31)] * 5,
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "weather_proba_sunny": [0.7, 0.1, 0.2, 0.2, 0.1],
    ...     "weather_proba_rainy": [0.2, 0.8, 0.1, 0.1, 0.8],
    ...     "weather_proba_cloudy": [0.1, 0.1, 0.7, 0.7, 0.1],
    ... })
    >>> scorer = Precision()
    >>> _ = scorer.fit(y_true)
    >>> scorer.score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.833...

    See Also
    --------
    - [`Recall`][yohou.metrics.classification.Recall] : Recall (sensitivity).
    - [`FBetaScore`][yohou.metrics.classification.FBetaScore] : Weighted harmonic mean of precision and recall.

    """

    _metric_name = "precision"
    _lower_is_better = False

    def _compute_metric_from_counts(self, counts: pl.DataFrame) -> pl.DataFrame:
        """Compute precision from confusion counts."""
        return counts.select(
            pl
            .when(pl.col("tp") + pl.col("fp") == 0)
            .then(self.zero_division)
            .otherwise(pl.col("tp") / (pl.col("tp") + pl.col("fp")))
            .alias("value")
        )


class Recall(BaseHardLabelScorer):
    r"""Recall (sensitivity) from class-probability forecasts.

    Computes recall by argmaxing predicted probabilities into hard labels
    and computing the ratio of true positives to all actual positives.

    $$\text{Recall}_k = \frac{TP_k}{TP_k + FN_k}$$

    Parameters
    ----------
    average : str, default="macro"
        Class averaging strategy: ``"macro"`` (unweighted mean across
        classes), ``"micro"`` (aggregate counts first), or ``"weighted"``
        (support-weighted mean).
    zero_division : float, default=0.0
        Value returned when the denominator is zero.
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over.
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter or filter with weights.
    components : list of str, dict of str to float, or None, default=None
        Component filter or filter with weights.

    Attributes
    ----------
    lower_is_better : bool
        Always False (higher recall is better).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics.classification import Recall
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "weather": ["sunny", "rainy", "cloudy", "sunny", "rainy"],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "vintage_time": [datetime(2019, 12, 31)] * 5,
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "weather_proba_sunny": [0.7, 0.1, 0.2, 0.2, 0.1],
    ...     "weather_proba_rainy": [0.2, 0.8, 0.1, 0.1, 0.8],
    ...     "weather_proba_cloudy": [0.1, 0.1, 0.7, 0.7, 0.1],
    ... })
    >>> scorer = Recall()
    >>> _ = scorer.fit(y_true)
    >>> scorer.score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.833...

    See Also
    --------
    - [`Precision`][yohou.metrics.classification.Precision] : Precision (positive predictive value).
    - [`FBetaScore`][yohou.metrics.classification.FBetaScore] : Weighted harmonic mean of precision and recall.

    """

    _metric_name = "recall"
    _lower_is_better = False

    def _compute_metric_from_counts(self, counts: pl.DataFrame) -> pl.DataFrame:
        """Compute recall from confusion counts."""
        return counts.select(
            pl
            .when(pl.col("tp") + pl.col("fn") == 0)
            .then(self.zero_division)
            .otherwise(pl.col("tp") / (pl.col("tp") + pl.col("fn")))
            .alias("value")
        )


class FBetaScore(BaseHardLabelScorer):
    r"""F-beta score from class-probability forecasts.

    Computes the weighted harmonic mean of precision and recall. With
    ``beta=1.0`` this is the standard F1 score.

    $$F_\beta = \frac{(1+\beta^2) \cdot TP}{(1+\beta^2) \cdot TP + \beta^2 \cdot FN + FP}$$

    Parameters
    ----------
    beta : float, default=1.0
        Weight of recall relative to precision. ``beta=1.0`` gives
        equal weight (F1), ``beta=2.0`` weights recall twice as much (F2).
    average : str, default="macro"
        Class averaging strategy: ``"macro"`` (unweighted mean across
        classes), ``"micro"`` (aggregate counts first), or ``"weighted"``
        (support-weighted mean).
    zero_division : float, default=0.0
        Value returned when the denominator is zero.
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over.
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter or filter with weights.
    components : list of str, dict of str to float, or None, default=None
        Component filter or filter with weights.

    Attributes
    ----------
    lower_is_better : bool
        Always False (higher F-beta is better).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics.classification import FBetaScore
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "weather": ["sunny", "rainy", "cloudy", "sunny", "rainy"],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "vintage_time": [datetime(2019, 12, 31)] * 5,
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "weather_proba_sunny": [0.7, 0.1, 0.2, 0.2, 0.1],
    ...     "weather_proba_rainy": [0.2, 0.8, 0.1, 0.1, 0.8],
    ...     "weather_proba_cloudy": [0.1, 0.1, 0.7, 0.7, 0.1],
    ... })
    >>> scorer = FBetaScore(beta=1.0)
    >>> _ = scorer.fit(y_true)
    >>> scorer.score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.777...

    See Also
    --------
    - [`Precision`][yohou.metrics.classification.Precision] : Precision (positive predictive value).
    - [`Recall`][yohou.metrics.classification.Recall] : Recall (sensitivity).

    """

    _parameter_constraints: dict = {
        **BaseHardLabelScorer._parameter_constraints,
        "beta": "no_validation",
    }

    _lower_is_better = False

    @property
    def _metric_name(self) -> str:
        """Return metric name based on beta value."""
        if self.beta == 1.0:
            return "f1"
        if self.beta == 2.0:
            return "f2"
        return "fbeta"

    def __init__(
        self,
        beta: float = 1.0,
        average: str = "macro",
        zero_division: float = 0.0,
        aggregation_method: list[str] | str = "all",
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ):
        super().__init__(
            average=average,
            zero_division=zero_division,
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
        )
        self.beta = beta

    def _compute_metric_from_counts(self, counts: pl.DataFrame) -> pl.DataFrame:
        """Compute F-beta score from confusion counts."""
        b2 = self.beta**2
        return counts.select(
            pl
            .when((1 + b2) * pl.col("tp") + b2 * pl.col("fn") + pl.col("fp") == 0)
            .then(self.zero_division)
            .otherwise((1 + b2) * pl.col("tp") / ((1 + b2) * pl.col("tp") + b2 * pl.col("fn") + pl.col("fp")))
            .alias("value")
        )


class Accuracy(BaseHardLabelScorer):
    r"""Categorical accuracy from class-probability forecasts.

    Computes the fraction of time steps where the predicted class (argmax
    of probabilities) matches the true class.

    $$\text{Accuracy} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}[\hat{y}_i = y_i]$$

    where $\hat{y}_i = \arg\max_k \hat{p}_{ik}$ is the predicted class.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. See `BaseClassProbaScorer`.
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter or filter with weights.
    components : list of str, dict of str to float, or None, default=None
        Component filter or filter with weights.

    Attributes
    ----------
    lower_is_better : bool
        Always False for accuracy (higher is better).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics.classification import Accuracy
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
    >>> scorer = Accuracy()
    >>> _ = scorer.fit(y_true)
    >>> scorer.score(y_true, y_pred)
    1.0

    Notes
    -----
    Accuracy uses micro averaging internally: TP / (TP + FP). In
    multiclass settings each prediction is either a TP for one class or
    an FP for another, so TP + FP equals the total sample count and
    TP / (TP + FP) = correct / N.

    See Also
    --------
    - [`LogLoss`][yohou.metrics.class_proba.LogLoss] : Logarithmic loss (cross-entropy).
    - [`BrierScore`][yohou.metrics.class_proba.BrierScore] : Multi-class Brier score.
    - [`Precision`][yohou.metrics.classification.Precision] : Precision (positive predictive value).

    """

    # Spread BaseClassProbaScorer constraints directly, skipping
    # BaseHardLabelScorer's average and zero_division additions.
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
            average="micro",
            zero_division=0.0,
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
        )

    def _compute_metric_from_counts(self, counts: pl.DataFrame) -> pl.DataFrame:
        """Compute accuracy from confusion counts."""
        return counts.select(
            pl
            .when(pl.col("tp") + pl.col("fp") == 0)
            .then(self.zero_division)
            .otherwise(pl.col("tp") / (pl.col("tp") + pl.col("fp")))
            .alias("value")
        )


class ROCAuC(BaseRankingScorer):
    r"""ROC AUC from class-probability forecasts.

    Computes the Area Under the Receiver Operating Characteristic curve
    using a one-vs-rest (OvR) strategy for multiclass problems.

    Parameters
    ----------
    average : str, default="macro"
        Class averaging strategy: ``"macro"`` (unweighted mean across
        classes) or ``"weighted"`` (support-weighted mean).
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over.
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter or filter with weights.
    components : list of str, dict of str to float, or None, default=None
        Component filter or filter with weights.

    Attributes
    ----------
    lower_is_better : bool
        Always False (higher AUC is better).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics.classification import ROCAuC
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "weather": ["sunny", "rainy", "cloudy", "sunny", "rainy"],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "vintage_time": [datetime(2019, 12, 31)] * 5,
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "weather_proba_sunny": [0.7, 0.1, 0.2, 0.6, 0.1],
    ...     "weather_proba_rainy": [0.2, 0.8, 0.1, 0.3, 0.8],
    ...     "weather_proba_cloudy": [0.1, 0.1, 0.7, 0.1, 0.1],
    ... })
    >>> scorer = ROCAuC()
    >>> _ = scorer.fit(y_true)
    >>> scorer.score(y_true, y_pred)
    1.0

    See Also
    --------
    - [`PRAuC`][yohou.metrics.classification.PRAuC] : Precision-Recall AUC.

    """

    _metric_name = "roc_auc"
    _lower_is_better = False

    def _compute_ranking_metric(
        self,
        y_true_binary: np.ndarray,
        y_proba: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """Compute ROC AUC via sklearn."""
        return float(roc_auc_score(y_true_binary, y_proba, sample_weight=sample_weight))


class PRAuC(BaseRankingScorer):
    r"""Precision-Recall AUC from class-probability forecasts.

    Computes the Area Under the Precision-Recall curve using a one-vs-rest
    (OvR) strategy for multiclass problems. Uses ``average_precision_score``
    from scikit-learn.

    Parameters
    ----------
    average : str, default="macro"
        Class averaging strategy: ``"macro"`` (unweighted mean across
        classes) or ``"weighted"`` (support-weighted mean).
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over.
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter or filter with weights.
    components : list of str, dict of str to float, or None, default=None
        Component filter or filter with weights.

    Attributes
    ----------
    lower_is_better : bool
        Always False (higher AUC is better).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics.classification import PRAuC
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "weather": ["sunny", "rainy", "cloudy", "sunny", "rainy"],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "vintage_time": [datetime(2019, 12, 31)] * 5,
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "weather_proba_sunny": [0.7, 0.1, 0.2, 0.6, 0.1],
    ...     "weather_proba_rainy": [0.2, 0.8, 0.1, 0.3, 0.8],
    ...     "weather_proba_cloudy": [0.1, 0.1, 0.7, 0.1, 0.1],
    ... })
    >>> scorer = PRAuC()
    >>> _ = scorer.fit(y_true)
    >>> scorer.score(y_true, y_pred)
    1.0

    See Also
    --------
    - [`ROCAuC`][yohou.metrics.classification.ROCAuC] : ROC AUC.

    """

    _metric_name = "pr_auc"
    _lower_is_better = False

    def _compute_ranking_metric(
        self,
        y_true_binary: np.ndarray,
        y_proba: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """Compute PR AUC via sklearn."""
        return float(average_precision_score(y_true_binary, y_proba, sample_weight=sample_weight))
