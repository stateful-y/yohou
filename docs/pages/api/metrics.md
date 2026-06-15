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
| [`BaseClassProbaScorer`](generated/yohou.metrics.base.BaseClassProbaScorer.md) | Base class for class-probability metrics. |
| [`BaseHardLabelScorer`](generated/yohou.metrics.base.BaseHardLabelScorer.md) | Base class for hard-label classification metrics. |
| [`BaseRankingScorer`](generated/yohou.metrics.base.BaseRankingScorer.md) | Base class for ranking-based classification metrics. |
| [`BaseConformityScorer`](generated/yohou.metrics.conformity_base.BaseConformityScorer.md) | Base class for conformal prediction conformity scorers. |

### Point Scorers

| Name | Description |
| --- | --- |
| [`MeanAbsoluteError`](generated/yohou.metrics.point.MeanAbsoluteError.md) | Mean Absolute Error metric for point forecasts. |
| [`MeanAbsolutePercentageError`](generated/yohou.metrics.point.MeanAbsolutePercentageError.md) | Mean Absolute Percentage Error metric for point forecasts. |
| [`MeanAbsoluteScaledError`](generated/yohou.metrics.point.MeanAbsoluteScaledError.md) | Mean Absolute Scaled Error metric for point forecasts. |
| [`MaxAbsoluteError`](generated/yohou.metrics.point.MaxAbsoluteError.md) | Maximum Absolute Error metric for point forecasts. |
| [`MeanDirectionalAccuracy`](generated/yohou.metrics.point.MeanDirectionalAccuracy.md) | Mean Directional Accuracy metric for point forecasts. |
| [`MeanSquaredError`](generated/yohou.metrics.point.MeanSquaredError.md) | Mean Squared Error metric for point forecasts. |
| [`MedianAbsoluteError`](generated/yohou.metrics.point.MedianAbsoluteError.md) | Median Absolute Error metric for point forecasts. |
| [`R2Score`](generated/yohou.metrics.point.R2Score.md) | Coefficient of determination (R-squared) metric for point forecasts. |
| [`RootMeanSquaredError`](generated/yohou.metrics.point.RootMeanSquaredError.md) | Root Mean Squared Error metric for point forecasts. |
| [`RootMeanSquaredScaledError`](generated/yohou.metrics.point.RootMeanSquaredScaledError.md) | Root Mean Squared Scaled Error metric for point forecasts. |
| [`SymmetricMeanAbsolutePercentageError`](generated/yohou.metrics.point.SymmetricMeanAbsolutePercentageError.md) | Symmetric Mean Absolute Percentage Error metric for point forecasts. |

### Interval Scorers

| Name | Description |
| --- | --- |
| [`CalibrationError`](generated/yohou.metrics.interval.CalibrationError.md) | Calibration Error for prediction intervals. |
| [`ContinuousRankedProbabilityScore`](generated/yohou.metrics.interval.ContinuousRankedProbabilityScore.md) | Continuous Ranked Probability Score for prediction intervals. |
| [`EmpiricalCoverage`](generated/yohou.metrics.interval.EmpiricalCoverage.md) | Empirical coverage rate for prediction intervals. |
| [`IntervalScore`](generated/yohou.metrics.interval.IntervalScore.md) | Interval Score (Winkler Score) for prediction intervals. |
| [`MeanIntervalWidth`](generated/yohou.metrics.interval.MeanIntervalWidth.md) | Mean width of prediction intervals. |
| [`PinballLoss`](generated/yohou.metrics.interval.PinballLoss.md) | Pinball Loss (Quantile Score) for prediction intervals. |

### Conformity Scorers

| Name | Description |
| --- | --- |
| [`AbsoluteGammaResidual`](generated/yohou.metrics.conformity.AbsoluteGammaResidual.md) | Absolute gamma residual scorer using absolute relative errors. |
| [`AbsoluteQuantileResidual`](generated/yohou.metrics.conformity.AbsoluteQuantileResidual.md) | Absolute quantile residual conformity scorer using unsigned pinball errors. |
| [`AbsoluteResidual`](generated/yohou.metrics.conformity.AbsoluteResidual.md) | Absolute residual conformity scorer using unsigned prediction errors. |
| [`GammaResidual`](generated/yohou.metrics.conformity.GammaResidual.md) | Gamma residual scorer using relative prediction errors. |
| [`QuantileResidual`](generated/yohou.metrics.conformity.QuantileResidual.md) | Quantile residual conformity scorer using signed pinball errors. |
| [`Residual`](generated/yohou.metrics.conformity.Residual.md) | Residual-based conformity scorer using signed prediction errors. |

### Classification Scorers

| Name | Description |
| --- | --- |
| [`Accuracy`](generated/yohou.metrics.classification.Accuracy.md) | Categorical accuracy from class-probability forecasts. |
| [`BrierScore`](generated/yohou.metrics.class_proba.BrierScore.md) | Multi-class Brier score for class-probability forecasts. |
| [`FBetaScore`](generated/yohou.metrics.classification.FBetaScore.md) | F-beta score (hard-label) from class-probability forecasts. |
| [`LogLoss`](generated/yohou.metrics.class_proba.LogLoss.md) | Logarithmic loss (cross-entropy) for class-probability forecasts. |
| [`PRAuC`](generated/yohou.metrics.classification.PRAuC.md) | Area under the precision-recall curve from class-probability forecasts. |
| [`Precision`](generated/yohou.metrics.classification.Precision.md) | Precision (hard-label) from class-probability forecasts. |
| [`RankedProbabilityScore`](generated/yohou.metrics.class_proba.RankedProbabilityScore.md) | Ranked Probability Score for ordered class-probability forecasts. |
| [`Recall`](generated/yohou.metrics.classification.Recall.md) | Recall (hard-label) from class-probability forecasts. |
| [`ROCAuC`](generated/yohou.metrics.classification.ROCAuC.md) | Area under the ROC curve from class-probability forecasts. |

### Utilities

| Name | Description |
| --- | --- |
| [`get_scorer`](generated/yohou.metrics.get_scorer.md) | Get a default-configured scorer instance by name. |
| [`make_scorer`](generated/yohou.metrics.make_scorer.md) | Create a scorer instance with custom parameters. |
