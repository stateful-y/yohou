"""Scoring functions for point, interval, and conformity predictions."""

from .base import BaseConformityScorer, BaseIntervalScorer, BasePointScorer
from .conformity import (
    AbsoluteGammaResidual,
    AbsoluteQuantileResidual,
    AbsoluteResidual,
    GammaResidual,
    QuantileResidual,
    Residual,
)
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
]
