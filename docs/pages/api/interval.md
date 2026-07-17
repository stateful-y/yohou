---
template: api-submodule.html
---

# yohou.interval

Interval forecasters for prediction uncertainty quantification.

### Classes

| Name | Description |
|------|-------------|
| [`BaseIntervalForecaster`](generated/yohou.interval.BaseIntervalForecaster.md) | Base class for interval forecasters. |
| [`BaseSimilarity`](generated/yohou.interval.BaseSimilarity.md) | Base class for similarity measures used in interval forecasting. |
| [`IntervalReductionForecaster`](generated/yohou.interval.IntervalReductionForecaster.md) | Interval forecaster using sklearn estimators on tabularized time series. |
| [`CompositeSimilarity`](generated/yohou.interval.CompositeSimilarity.md) | Combine multiple named similarity measures into a single weight vector. |
| [`DistanceSimilarity`](generated/yohou.interval.DistanceSimilarity.md) | Distance-based similarity using scipy metrics for weighting observations. |
| [`SeasonalSimilarity`](generated/yohou.interval.SeasonalSimilarity.md) | Temporal similarity using Fourier features for weighting observations. |
| [`SplitConformalForecaster`](generated/yohou.interval.SplitConformalForecaster.md) | Split conformal forecaster implementation. |

### Functions

| Name | Description |
|------|-------------|
| [`weighted_quantile`](generated/yohou.interval.weighted_quantile.md) | Compute weighted quantile using cumulative sum approach. |
