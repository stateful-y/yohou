---
template: api-submodule.html
---

# yohou.interval

Interval forecasters for generating prediction intervals with uncertainty quantification.

### Interval Forecasters

| Name | Description |
| --- | --- |
| [`BaseIntervalForecaster`](generated/yohou.interval.base.BaseIntervalForecaster.md) | Base class for interval forecasters. |
| [`IntervalReductionForecaster`](generated/yohou.interval.reduction.IntervalReductionForecaster.md) | Interval forecaster using sklearn estimators on tabularized time series. |
| [`SplitConformalForecaster`](generated/yohou.interval.split_conformal.SplitConformalForecaster.md) | Split conformal forecaster implementation. |

### Similarity estimators

| Name | Description |
| --- | --- |
| [`BaseSimilarity`](generated/yohou.interval.base.BaseSimilarity.md) | Base class for similarity measures used in interval forecasting. |
| [`DistanceSimilarity`](generated/yohou.interval.similarity.DistanceSimilarity.md) | Distance-based similarity using scipy metrics for weighting observations. |

### Utilities

| Name | Description |
| --- | --- |
| [`weighted_quantile`](generated/yohou.interval.utils.weighted_quantile.md) | Compute weighted quantile using cumulative sum approach. |
