---
template: api-submodule.html
---

# yohou.interval

Interval forecasters produce prediction intervals at specified coverage rates. `SplitConformalForecaster` uses conformal prediction: it computes a conformity score on a calibration set and optionally weights it by a similarity measure to the current prediction context.

**User guide**: See [Interval Forecasting](../explanation/interval-forecasting.md) for design rationale and usage patterns.

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
| [`SeasonalSimilarity`](generated/yohou.interval.similarity.SeasonalSimilarity.md) | Seasonal-phase similarity using Fourier features for weighting observations. |
| [`CompositeSimilarity`](generated/yohou.interval.similarity.CompositeSimilarity.md) | Combines multiple named similarity measures into a single weight matrix. |

### Utilities

| Name | Description |
| --- | --- |
| [`weighted_quantile`](generated/yohou.interval.utils.weighted_quantile.md) | Compute weighted quantile using cumulative sum approach. |
