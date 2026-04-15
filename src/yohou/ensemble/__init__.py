"""Ensemble forecasters for combining predictions from multiple forecasters."""

from .voting_class_proba import VotingClassProbaForecaster
from .voting_interval import VotingIntervalForecaster
from .voting_point import VotingPointForecaster

__all__ = [
    "VotingClassProbaForecaster",
    "VotingIntervalForecaster",
    "VotingPointForecaster",
]
