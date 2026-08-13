---
template: api-submodule.html
---

# yohou.base

Base classes for transformers and forecasters.

### Classes

| Name | Description |
|------|-------------|
| [`BaseForecastTransformer`](generated/yohou.base.BaseForecastTransformer.md) | Base class for ``"forecast"``-kind transformers over ``X_forecast`` frames. |
| [`BaseForecaster`](generated/yohou.base.BaseForecaster.md) | Base class for forecasters. |
| [`BasePanelForecaster`](generated/yohou.base.BasePanelForecaster.md) | Mixin providing panel (dict of DataFrames) forecaster operations. |
| [`BaseReductionForecaster`](generated/yohou.base.BaseReductionForecaster.md) | Base class for forecasters using reduction to supervised learning. |
| [`BaseStandardForecaster`](generated/yohou.base.BaseStandardForecaster.md) | Mixin providing standard (single DataFrame) forecaster operations. |
| [`BaseStepTransformer`](generated/yohou.base.BaseStepTransformer.md) | Base class for ``"step"``-kind transformers over the derived step frame. |
| [`BaseActualTransformer`](generated/yohou.base.BaseActualTransformer.md) | Base class for single-axis (``"actual"``-kind) time series transformers. |
| [`ForecastCoverageWarning`](generated/yohou.base.ForecastCoverageWarning.md) | Raised when ``X_forecast`` covers fewer steps than the forecasting horizon. |
