"""The module :mod:`interval_forecaster`"""

from .reduction import IntervalReductionForecaster
from .split_conformal import SplitConformalForecaster

__all__ = [
    "IntervalReductionForecaster",
    "SplitConformalForecaster",
]
