---
template: api-submodule.html
---

# yohou.compose

Composition utilities for transformers and forecasters.

### Classes

| Name | Description |
|------|-------------|
| [`AdditiveForecaster`](generated/yohou.compose.AdditiveForecaster.md) | Point meta-forecaster that sums independently fitted per-term forecasts. |
| [`ColumnForecaster`](generated/yohou.compose.ColumnForecaster.md) | Applies different forecasters to different column subsets. |
| [`ColumnTransformer`](generated/yohou.compose.ColumnTransformer.md) | Applies transformers to columns of a polars DataFrame. |
| [`DecompositionPipeline`](generated/yohou.compose.DecompositionPipeline.md) | Meta-forecaster that decomposes time series into sequential components. |
| [`FeaturePipeline`](generated/yohou.compose.FeaturePipeline.md) | A sequence of time series transformers. |
| [`FeatureUnion`](generated/yohou.compose.FeatureUnion.md) | Concatenates results of multiple transformer objects. |
| [`ForecastedFeatureForecaster`](generated/yohou.compose.ForecastedFeatureForecaster.md) | Meta-forecaster that chains feature forecasting into target forecasting. |
| [`LocalPanelForecaster`](generated/yohou.compose.LocalPanelForecaster.md) | Fits independent forecaster clones per panel group. |
| [`PerVintageActualTransformer`](generated/yohou.compose.PerVintageActualTransformer.md) | Apply a single-axis transformer to each vintage of an ``X_forecast`` frame. |
