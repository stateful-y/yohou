---
template: api-submodule.html
---

# yohou.segmenting

Segmenter estimators that label timestamps by calendar segment.

### Classes

| Name | Description |
|------|-------------|
| [`BaseSegmenter`](generated/yohou.segmenting.BaseSegmenter.md) | Base class for time-axis segmenter estimators. |
| [`CalendarSegmenter`](generated/yohou.segmenting.CalendarSegmenter.md) | Segments timestamps by the value of a calendar feature. |
| [`ComplementSegmenter`](generated/yohou.segmenting.ComplementSegmenter.md) | Rows outside a segmenter. |
| [`IntersectionSegmenter`](generated/yohou.segmenting.IntersectionSegmenter.md) | Rows inside every component segmenter. |
| [`UnionSegmenter`](generated/yohou.segmenting.UnionSegmenter.md) | Rows inside any component segmenter. |
