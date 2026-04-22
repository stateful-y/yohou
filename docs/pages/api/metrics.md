---
template: api-submodule.html
---

# yohou.metrics

Scorers for evaluating point forecasts, prediction intervals, and conformal prediction calibration.

### Base Classes

| Name | Description |
| --- | --- |
| [`BaseScorer`](generated/yohou.metrics.base.BaseScorer.md) | Base class for all forecasting metrics. |
| [`BasePointScorer`](generated/yohou.metrics.base.BasePointScorer.md) | Base class for point forecast metrics. |
| [`BaseIntervalScorer`](generated/yohou.metrics.base.BaseIntervalScorer.md) | Base class for interval forecast metrics. |
| [`BaseConformityScorer`](generated/yohou.metrics.conformity_base.BaseConformityScorer.md) | Base class for conformal prediction conformity scorers. |

### Point Scorers

| Name | Description |
| --- | --- |
| [`MeanAbsoluteError`](generated/yohou.metrics.point.MeanAbsoluteError.md) | Mean Absolute Error metric for point forecasts. |
| [`MeanAbsolutePercentageError`](generated/yohou.metrics.point.MeanAbsolutePercentageError.md) | Mean Absolute Percentage Error metric for point forecasts. |
| [`MeanAbsoluteScaledError`](generated/yohou.metrics.point.MeanAbsoluteScaledError.md) | Mean Absolute Scaled Error metric for point forecasts. |
| [`MeanSquaredError`](generated/yohou.metrics.point.MeanSquaredError.md) | Mean Squared Error metric for point forecasts. |
| [`MedianAbsoluteError`](generated/yohou.metrics.point.MedianAbsoluteError.md) | Median Absolute Error metric for point forecasts. |
| [`RootMeanSquaredError`](generated/yohou.metrics.point.RootMeanSquaredError.md) | Root Mean Squared Error metric for point forecasts. |
| [`RootMeanSquaredScaledError`](generated/yohou.metrics.point.RootMeanSquaredScaledError.md) | Root Mean Squared Scaled Error metric for point forecasts. |
| [`SymmetricMeanAbsolutePercentageError`](generated/yohou.metrics.point.SymmetricMeanAbsolutePercentageError.md) | Symmetric Mean Absolute Percentage Error metric for point forecasts. |

### Interval Scorers

| Name | Description |
| --- | --- |
| [`CalibrationError`](generated/yohou.metrics.interval.CalibrationError.md) | Calibration Error for prediction intervals. |
| [`EmpiricalCoverage`](generated/yohou.metrics.interval.EmpiricalCoverage.md) | Empirical coverage rate for prediction intervals. |
| [`IntervalScore`](generated/yohou.metrics.interval.IntervalScore.md) | Interval Score (Winkler Score) for prediction intervals. |
| [`MeanIntervalWidth`](generated/yohou.metrics.interval.MeanIntervalWidth.md) | Mean width of prediction intervals. |
| [`PinballLoss`](generated/yohou.metrics.interval.PinballLoss.md) | Pinball Loss (Quantile Score) for prediction intervals. |

### Conformity Scorers

| Name | Description |
| --- | --- |
| [`AbsoluteGammaResidual`](generated/yohou.metrics.conformity.AbsoluteGammaResidual.md) | Absolute gamma residual scorer using absolute relative errors. |
| [`AbsoluteResidual`](generated/yohou.metrics.conformity.AbsoluteResidual.md) | Absolute residual conformity scorer using unsigned prediction errors. |
| [`GammaResidual`](generated/yohou.metrics.conformity.GammaResidual.md) | Gamma residual scorer using relative prediction errors. |
| [`Residual`](generated/yohou.metrics.conformity.Residual.md) | Residual-based conformity scorer using signed prediction errors. |

### Classification Scorers

| Name | Description |
| --- | --- |
| [`Accuracy`](generated/yohou.metrics.class_proba.Accuracy.md) | Categorical accuracy from class-probability forecasts. |
| [`BrierScore`](generated/yohou.metrics.class_proba.BrierScore.md) | Multi-class Brier score for class-probability forecasts. |
| [`LogLoss`](generated/yohou.metrics.class_proba.LogLoss.md) | Logarithmic loss (cross-entropy) for class-probability forecasts. |
