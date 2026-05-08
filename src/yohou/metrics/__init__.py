"""Scoring functions for point, interval, class-probability, and conformity predictions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    BaseClassProbaScorer,
    BaseHardLabelScorer,
    BaseIntervalScorer,
    BasePointScorer,
    BaseRankingScorer,
    BaseScorer,
)
from .class_proba import BrierScore, LogLoss, RankedProbabilityScore
from .classification import Accuracy, FBetaScore, PRAuC, Precision, Recall, ROCAuC
from .conformity import (
    AbsoluteGammaResidual,
    AbsoluteQuantileResidual,
    AbsoluteResidual,
    GammaResidual,
    QuantileResidual,
    Residual,
)
from .conformity_base import BaseConformityScorer
from .interval import (
    CalibrationError,
    ContinuousRankedProbabilityScore,
    EmpiricalCoverage,
    IntervalScore,
    MeanIntervalWidth,
    PinballLoss,
)
from .point import (
    MaxAbsoluteError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
    MeanDirectionalAccuracy,
    MeanSquaredError,
    MedianAbsoluteError,
    R2Score,
    RootMeanSquaredError,
    RootMeanSquaredScaledError,
    SymmetricMeanAbsolutePercentageError,
)

if TYPE_CHECKING:
    from typing import Any

# Registry: maps metric name -> scorer class (16 scoring scorers, no conformity)
_SCORER_REGISTRY: dict[str, type[BaseScorer]] = {
    # Point scorers
    "mae": MeanAbsoluteError,
    "mse": MeanSquaredError,
    "rmse": RootMeanSquaredError,
    "rmsse": RootMeanSquaredScaledError,
    "mape": MeanAbsolutePercentageError,
    "smape": SymmetricMeanAbsolutePercentageError,
    "mase": MeanAbsoluteScaledError,
    "median_ae": MedianAbsoluteError,
    "max_ae": MaxAbsoluteError,
    "r2": R2Score,
    "mda": MeanDirectionalAccuracy,
    # Interval scorers
    "coverage": EmpiricalCoverage,
    "width": MeanIntervalWidth,
    "interval_score": IntervalScore,
    "pinball_loss": PinballLoss,
    "calibration_error": CalibrationError,
    "crps": ContinuousRankedProbabilityScore,
    # Class-probability scorers
    "log_loss": LogLoss,
    "brier_score": BrierScore,
    "rps": RankedProbabilityScore,
    # Hard-label classification scorers
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "fbeta": FBetaScore,
    "f1": FBetaScore,
    # Ranking classification scorers
    "roc_auc": ROCAuC,
    "pr_auc": PRAuC,
}


def get_scorer(name: str) -> BaseScorer:
    """Get a scorer instance by name.

    Parameters
    ----------
    name : str
        Name of the scorer (e.g., ``"mae"``, ``"coverage"``).
        See ``_SCORER_REGISTRY`` for available names.

    Returns
    -------
    BaseScorer
        A default-configured scorer instance.

    Raises
    ------
    ValueError
        If the name is not in the registry.

    Examples
    --------
    >>> scorer = get_scorer("mae")
    >>> type(scorer).__name__
    'MeanAbsoluteError'

    """
    if name not in _SCORER_REGISTRY:
        available = sorted(_SCORER_REGISTRY.keys())
        raise ValueError(f"Unknown scorer {name!r}. Available: {available}")
    return _SCORER_REGISTRY[name]()


def make_scorer(name: str, **params: Any) -> BaseScorer:
    """Create a scorer instance with custom parameters.

    Parameters
    ----------
    name : str
        Name of the scorer (e.g., ``"mae"``, ``"coverage"``).
    **params : dict
        Parameters passed to the scorer constructor.

    Returns
    -------
    BaseScorer
        A configured scorer instance.

    Raises
    ------
    ValueError
        If the name is not in the registry.

    Examples
    --------
    >>> scorer = make_scorer("mae", aggregation_method=["stepwise", "vintagewise"])
    >>> scorer.aggregation_method
    ['stepwise', 'vintagewise']

    """
    if name not in _SCORER_REGISTRY:
        available = sorted(_SCORER_REGISTRY.keys())
        raise ValueError(f"Unknown scorer {name!r}. Available: {available}")
    return _SCORER_REGISTRY[name](**params)


__all__ = [
    # Base classes
    "BaseClassProbaScorer",
    "BaseConformityScorer",
    "BaseHardLabelScorer",
    "BaseIntervalScorer",
    "BasePointScorer",
    "BaseRankingScorer",
    "BaseScorer",
    # Point scorers
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
    # Conformity scorers
    "AbsoluteGammaResidual",
    "AbsoluteQuantileResidual",
    "AbsoluteResidual",
    "GammaResidual",
    "QuantileResidual",
    "Residual",
    # Interval scorers
    "CalibrationError",
    "ContinuousRankedProbabilityScore",
    "EmpiricalCoverage",
    "IntervalScore",
    "MeanIntervalWidth",
    "PinballLoss",
    # Class-probability scorers
    "BrierScore",
    "Accuracy",
    "LogLoss",
    "RankedProbabilityScore",
    # Classification scorers
    "FBetaScore",
    "Precision",
    "PRAuC",
    "Recall",
    "ROCAuC",
    # Registry and factories
    "get_scorer",
    "make_scorer",
]
