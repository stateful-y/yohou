"""Scoring functions for point, interval, class-probability, and conformity predictions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseClassProbaScorer, BaseIntervalScorer, BasePointScorer, BaseScorer
from .class_proba import Accuracy, BrierScore, LogLoss
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
    EmpiricalCoverage,
    IntervalScore,
    MeanIntervalWidth,
    PinballLoss,
)
from .point import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
    MeanSquaredError,
    MedianAbsoluteError,
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
    # Interval scorers
    "coverage": EmpiricalCoverage,
    "width": MeanIntervalWidth,
    "interval_score": IntervalScore,
    "pinball_loss": PinballLoss,
    "calibration_error": CalibrationError,
    # Class-probability scorers
    "accuracy": Accuracy,
    "log_loss": LogLoss,
    "brier_score": BrierScore,
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
    "BaseIntervalScorer",
    "BasePointScorer",
    "BaseScorer",
    # Point scorers
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanAbsoluteScaledError",
    "MeanSquaredError",
    "MedianAbsoluteError",
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
    "EmpiricalCoverage",
    "IntervalScore",
    "MeanIntervalWidth",
    "PinballLoss",
    # Class-probability scorers
    "BrierScore",
    "Accuracy",
    "LogLoss",
    # Registry and factories
    "get_scorer",
    "make_scorer",
]
