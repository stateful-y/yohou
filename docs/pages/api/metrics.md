---
template: api-submodule.html
---

# yohou.metrics

Scoring functions for point, interval, class-probability, and conformity predictions.

### Classes

| Name | Description |
|------|-------------|
| [`BaseClassProbaScorer`](generated/yohou.metrics.BaseClassProbaScorer.md) | Base class for class-probability forecast metrics. |
| [`BaseHardLabelScorer`](generated/yohou.metrics.BaseHardLabelScorer.md) | Base class for confusion-matrix classification metrics. |
| [`BaseIntervalScorer`](generated/yohou.metrics.BaseIntervalScorer.md) | Base class for interval forecast metrics. |
| [`BasePointScorer`](generated/yohou.metrics.BasePointScorer.md) | Base class for point forecast metrics. |
| [`BaseRankingScorer`](generated/yohou.metrics.BaseRankingScorer.md) | Base class for ranking classification metrics. |
| [`BaseScorer`](generated/yohou.metrics.BaseScorer.md) | Base class for all forecasting metrics. |
| [`BrierScore`](generated/yohou.metrics.BrierScore.md) | Multi-class Brier score for class-probability forecasts. |
| [`LogLoss`](generated/yohou.metrics.LogLoss.md) | Logarithmic loss (cross-entropy) for class-probability forecasts. |
| [`RankedProbabilityScore`](generated/yohou.metrics.RankedProbabilityScore.md) | Ranked Probability Score for class-probability forecasts. |
| [`Accuracy`](generated/yohou.metrics.Accuracy.md) | Categorical accuracy from class-probability forecasts. |
| [`FBetaScore`](generated/yohou.metrics.FBetaScore.md) | F-beta score from class-probability forecasts. |
| [`PRAuC`](generated/yohou.metrics.PRAuC.md) | Precision-Recall AUC from class-probability forecasts. |
| [`Precision`](generated/yohou.metrics.Precision.md) | Precision from class-probability forecasts. |
| [`Recall`](generated/yohou.metrics.Recall.md) | Recall (sensitivity) from class-probability forecasts. |
| [`ROCAuC`](generated/yohou.metrics.ROCAuC.md) | ROC AUC from class-probability forecasts. |
| [`AbsoluteGammaResidual`](generated/yohou.metrics.AbsoluteGammaResidual.md) | Absolute gamma residual scorer using absolute relative errors. |
| [`AbsoluteNormalizedResidual`](generated/yohou.metrics.AbsoluteNormalizedResidual.md) | Symmetric variant of `NormalizedResidual` using absolute scores. |
| [`AbsoluteQuantileResidual`](generated/yohou.metrics.AbsoluteQuantileResidual.md) | Absolute quantile residual scorer for interval forecasts. |
| [`AbsoluteResidual`](generated/yohou.metrics.AbsoluteResidual.md) | Absolute residual conformity scorer using unsigned prediction errors. |
| [`GammaResidual`](generated/yohou.metrics.GammaResidual.md) | Gamma residual scorer using relative prediction errors. |
| [`NormalizedResidual`](generated/yohou.metrics.NormalizedResidual.md) | Residual scorer normalised by each column's own dispersion. |
| [`QuantileResidual`](generated/yohou.metrics.QuantileResidual.md) | Quantile residual scorer for interval forecasts. |
| [`Residual`](generated/yohou.metrics.Residual.md) | Residual-based conformity scorer using signed prediction errors. |
| [`BaseConformityScorer`](generated/yohou.metrics.BaseConformityScorer.md) | Base class for conformal prediction conformity scorers. |
| [`CalibrationError`](generated/yohou.metrics.CalibrationError.md) | Calibration Error for prediction intervals. |
| [`ContinuousRankedProbabilityScore`](generated/yohou.metrics.ContinuousRankedProbabilityScore.md) | Continuous Ranked Probability Score (CRPS) for prediction intervals. |
| [`EmpiricalCoverage`](generated/yohou.metrics.EmpiricalCoverage.md) | Empirical coverage rate for prediction intervals. |
| [`IntervalScore`](generated/yohou.metrics.IntervalScore.md) | Interval Score (Winkler Score) for prediction intervals. |
| [`MeanIntervalWidth`](generated/yohou.metrics.MeanIntervalWidth.md) | Mean width of prediction intervals. |
| [`PinballLoss`](generated/yohou.metrics.PinballLoss.md) | Pinball Loss (Quantile Score) for prediction intervals. |
| [`MaxAbsoluteError`](generated/yohou.metrics.MaxAbsoluteError.md) | Maximum Absolute Error metric for point forecasts. |
| [`MeanAbsoluteError`](generated/yohou.metrics.MeanAbsoluteError.md) | Mean Absolute Error metric for point forecasts. |
| [`MeanAbsolutePercentageError`](generated/yohou.metrics.MeanAbsolutePercentageError.md) | Mean Absolute Percentage Error metric for point forecasts. |
| [`MeanAbsoluteScaledError`](generated/yohou.metrics.MeanAbsoluteScaledError.md) | Mean Absolute Scaled Error metric for point forecasts. |
| [`MeanDirectionalAccuracy`](generated/yohou.metrics.MeanDirectionalAccuracy.md) | Mean Directional Accuracy metric for point forecasts. |
| [`MeanSquaredError`](generated/yohou.metrics.MeanSquaredError.md) | Mean Squared Error metric for point forecasts. |
| [`MedianAbsoluteError`](generated/yohou.metrics.MedianAbsoluteError.md) | Median Absolute Error metric for point forecasts. |
| [`R2Score`](generated/yohou.metrics.R2Score.md) | R-squared (Coefficient of Determination) metric for point forecasts. |
| [`RootMeanSquaredError`](generated/yohou.metrics.RootMeanSquaredError.md) | Root Mean Squared Error metric for point forecasts. |
| [`RootMeanSquaredScaledError`](generated/yohou.metrics.RootMeanSquaredScaledError.md) | Root Mean Squared Scaled Error metric for point forecasts. |
| [`SymmetricMeanAbsolutePercentageError`](generated/yohou.metrics.SymmetricMeanAbsolutePercentageError.md) | Symmetric Mean Absolute Percentage Error metric for point forecasts. |

### Functions

| Name | Description |
|------|-------------|
| [`get_scorer`](generated/yohou.metrics.get_scorer.md) | Get a scorer instance by name. |
| [`make_scorer`](generated/yohou.metrics.make_scorer.md) | Create a scorer instance with custom parameters. |
