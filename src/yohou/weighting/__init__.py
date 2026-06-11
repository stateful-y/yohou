"""Weighter estimators for time-axis weighting of training and evaluation.

Weighters map a key series (timestamps, forecasting steps, or vintage times) to
a series of non-negative weights. They are configured on a forecaster's or
scorer's ``__init__`` (``time_weighter``, ``vintage_weighter``, ``step_weighter``)
rather than passed as routed metadata, so their parameters are introspectable,
clonable, and searchable.
"""

from .weighters import (
    BaseWeighter,
    CompositeWeighter,
    ExponentialDecayWeighter,
    LinearDecayWeighter,
    LookupWeighter,
    SeasonalEmphasisWeighter,
    TableWeighter,
)

__all__ = [
    "BaseWeighter",
    "CompositeWeighter",
    "ExponentialDecayWeighter",
    "LinearDecayWeighter",
    "LookupWeighter",
    "SeasonalEmphasisWeighter",
    "TableWeighter",
]
