---
template: api-submodule.html
---

# yohou.weighting

Weighter estimators for time-axis weighting of training and evaluation. Configure
them on a forecaster's or scorer's `__init__` (`time_weighter`,
`vintage_weighter`, `step_weighter`); see
[Weighting](/pages/explanation/weighting/) and
[How to Use Time Weighting](/pages/how-to/time-weighting/).

### Weighters

| Name | Description |
| --- | --- |
| [`BaseWeighter`](generated/yohou.weighting.weighters.BaseWeighter.md) | Base class for time-axis weighter estimators. |
| [`ExponentialDecayWeighter`](generated/yohou.weighting.weighters.ExponentialDecayWeighter.md) | Exponential decay weights giving more weight to recent keys. |
| [`LinearDecayWeighter`](generated/yohou.weighting.weighters.LinearDecayWeighter.md) | Linear decay weights giving more weight to recent keys. |
| [`SeasonalEmphasisWeighter`](generated/yohou.weighting.weighters.SeasonalEmphasisWeighter.md) | Weights emphasizing keys in phase with the most recent seasonal position. |
| [`LookupWeighter`](generated/yohou.weighting.weighters.LookupWeighter.md) | Explicit per-key weights from a mapping. |
| [`TableWeighter`](generated/yohou.weighting.weighters.TableWeighter.md) | DataFrame-driven weights resolved by joining on a key column. |
| [`CompositeWeighter`](generated/yohou.weighting.weighters.CompositeWeighter.md) | Combine weighters by product or mean. |
