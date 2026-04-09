"""Class-probability forecasters for categorical time series."""

from .base import BaseClassProbaForecaster
from .reduction import ClassProbaReductionForecaster

__all__ = [
    "BaseClassProbaForecaster",
    "ClassProbaReductionForecaster",
]
