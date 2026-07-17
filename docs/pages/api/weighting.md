---
template: api-submodule.html
---

# yohou.weighting

Weighter estimators for time-axis weighting of training and evaluation.

### Classes

| Name | Description |
|------|-------------|
| [`BaseWeighter`](generated/yohou.weighting.BaseWeighter.md) | Base class for time-axis weighter estimators. |
| [`CompositeWeighter`](generated/yohou.weighting.CompositeWeighter.md) | Combine multiple named weighters into a single weight series. |
| [`ExponentialDecayWeighter`](generated/yohou.weighting.ExponentialDecayWeighter.md) | Exponential decay weights giving more weight to recent keys. |
| [`LinearDecayWeighter`](generated/yohou.weighting.LinearDecayWeighter.md) | Linear decay weights giving more weight to recent keys. |
| [`LookupWeighter`](generated/yohou.weighting.LookupWeighter.md) | Explicit per-key weights from a mapping. |
| [`SeasonalEmphasisWeighter`](generated/yohou.weighting.SeasonalEmphasisWeighter.md) | Weights emphasizing keys in phase with the most recent seasonal position. |
| [`TableWeighter`](generated/yohou.weighting.TableWeighter.md) | DataFrame-driven weights resolved by joining on a key column. |
