---
template: api-submodule.html
---

# yohou.base

Abstract base classes used internally by all Yohou estimators. See the concrete implementations in the submodule pages.

### Transformers

| Name | Description |
| --- | --- |
| [`BaseTransformer`](generated/yohou.base.transformer.BaseTransformer.md) | Base class for time series transformers. |

### Forecasters

| Name | Description |
| --- | --- |
| [`BaseForecaster`](generated/yohou.base.forecaster.BaseForecaster.md) | Base class for forecasters. |
| [`BaseStandardForecaster`](generated/yohou.base.standard.BaseStandardForecaster.md) | Mixin providing standard (single DataFrame) forecaster operations. |
| [`BasePanelForecaster`](generated/yohou.base.panel.BasePanelForecaster.md) | Mixin providing panel (dict of DataFrames) forecaster operations. |
| [`BaseReductionForecaster`](generated/yohou.base.reduction.BaseReductionForecaster.md) | Base class for forecasters using reduction to supervised learning. |
