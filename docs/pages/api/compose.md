---
template: api-submodule.html
---

# yohou.compose

Composition classes for combining forecasters and transformers into complex workflows.

### Transformers

| Name | Description |
| --- | --- |
| [`ColumnTransformer`](generated/yohou.compose.column_transformer.ColumnTransformer.md) | Applies transformers to columns of a polars DataFrame. |
| [`FeaturePipeline`](generated/yohou.compose.feature_pipeline.FeaturePipeline.md) | A sequence of time series transformers. |
| [`FeatureUnion`](generated/yohou.compose.feature_union.FeatureUnion.md) | Concatenates results of multiple transformer objects. |

### Forecasters

| Name | Description |
| --- | --- |
| [`DecompositionPipeline`](generated/yohou.compose.decomposition_pipeline.DecompositionPipeline.md) | Meta-forecaster that decomposes time series into sequential components. |
| [`ColumnForecaster`](generated/yohou.compose.column_forecaster.ColumnForecaster.md) | Applies different forecasters to different column subsets. |
| [`ForecastedFeatureForecaster`](generated/yohou.compose.forecasted_feature_forecaster.ForecastedFeatureForecaster.md) | Meta-forecaster that chains feature forecasting into target forecasting. |
| [`LocalPanelForecaster`](generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster.md) | Fits independent forecaster clones per panel group. |
