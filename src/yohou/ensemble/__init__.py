"""Ensemble forecasters for combining predictions from multiple forecasters."""

from .voting import VotingForecaster
from .voting_class_proba import VotingClassProbaForecaster

__all__ = [
    "VotingClassProbaForecaster",
    "VotingForecaster",
]
