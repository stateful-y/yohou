"""Scoring functions for point, interval, class-probability, and conformity predictions."""

from .base import BaseClassProbaScorer, BaseIntervalScorer, BasePointScorer
from .class_proba import BrierScore, Accuracy, LogLoss
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

__all__ = [
    # Base classes
    "BaseClassProbaScorer",
    "BaseConformityScorer",
    "BaseIntervalScorer",
    "BasePointScorer",
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
]
