"""Segmenter estimators that label timestamps by calendar segment.

Segmenters map a series of timestamps to one label per element, with a null label
for elements outside every segment. They are configured on a weighter or an outlier
handler (``segmenter=``) to restrict its behaviour to part of the calendar, such as
specific hours of each day. Their parameters are introspectable, clonable, and
searchable.
"""

from .segmenters import (
    BaseSegmenter,
    CalendarSegmenter,
    ComplementSegmenter,
    IntersectionSegmenter,
    UnionSegmenter,
)

__all__ = [
    "BaseSegmenter",
    "CalendarSegmenter",
    "ComplementSegmenter",
    "IntersectionSegmenter",
    "UnionSegmenter",
]
