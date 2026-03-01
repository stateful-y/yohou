---
template: api-submodule.html
---

# yohou.stationarity

Trend and seasonality estimation, stationarity transforms, and decomposition components.

### Transformers

| Name | Description |
| --- | --- |
| [`AbsoluteSeasonalReturn`](generated/yohou.stationarity.transformers.AbsoluteSeasonalReturn.md) | Absolute seasonal return (difference) time series transformer. |
| [`ASinhTransformer`](generated/yohou.stationarity.transformers.ASinhTransformer.md) | Variance stabilization through arcsinh transform. |
| [`BoxCoxTransformer`](generated/yohou.stationarity.transformers.BoxCoxTransformer.md) | Box-Cox power transformation time series transformer. |
| [`LogTransformer`](generated/yohou.stationarity.transformers.LogTransformer.md) | Logarithmic time series transformer. |
| [`SeasonalDifferencing`](generated/yohou.stationarity.transformers.SeasonalDifferencing.md) | Seasonal differencing time series transformer. |
| [`SeasonalLogDifferencing`](generated/yohou.stationarity.transformers.SeasonalLogDifferencing.md) | Seasonal log differencing time series transformer. |
| [`SeasonalReturn`](generated/yohou.stationarity.transformers.SeasonalReturn.md) | Seasonal percentage return time series transformer. |

### Point Forecasters

| Name | Description |
| --- | --- |
| [`FourierSeasonalityForecaster`](generated/yohou.stationarity.seasonality.FourierSeasonalityForecaster.md) | Forecast using Fourier series representation of seasonality. |
| [`PatternSeasonalityForecaster`](generated/yohou.stationarity.seasonality.PatternSeasonalityForecaster.md) | Forecast using seasonal pattern extraction and repetition. |
| [`PolynomialTrendForecaster`](generated/yohou.stationarity.trend.PolynomialTrendForecaster.md) | Forecast using polynomial trend extrapolation with ElasticNet regularization. |
