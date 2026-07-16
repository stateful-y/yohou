---
template: api-submodule.html
---

# yohou.base

Abstract base classes used internally by all Yohou estimators. See the concrete implementations in the submodule pages.

**User guide**: See [Extending Yohou](../explanation/extending-yohou.md) for design rationale and extension patterns.

### Transformers

| Name | Description |
| --- | --- |
| [`BaseActualTransformer`](generated/yohou.base.transformer.BaseActualTransformer.md) | Base class for single-axis (`"actual"`-kind) time series transformers. |
| [`BaseForecastTransformer`](generated/yohou.base.forecast_transformer.BaseForecastTransformer.md) | Base class for `"forecast"`-kind transformers over `X_forecast` frames (two time axes: `vintage_time`, `time`). |

### Forecasters

| Name | Description |
| --- | --- |
| [`BaseForecaster`](generated/yohou.base.forecaster.BaseForecaster.md) | Base class for forecasters. |
| [`BaseStandardForecaster`](generated/yohou.base.standard.BaseStandardForecaster.md) | Mixin providing standard (single DataFrame) forecaster operations. |
| [`BasePanelForecaster`](generated/yohou.base.panel.BasePanelForecaster.md) | Mixin providing panel (`group__column` DataFrame) forecaster operations. |
| [`BaseReductionForecaster`](generated/yohou.base.reduction.BaseReductionForecaster.md) | Base class for forecasters using reduction to supervised learning. |

### Types

| Name | Description |
| --- | --- |
| [`PredictionType`](generated/yohou.base.forecaster.PredictionType.md) | Literal type alias for the kind of prediction a forecaster produces. |
