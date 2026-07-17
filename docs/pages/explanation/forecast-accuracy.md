# Forecast Accuracy

Measuring forecast quality requires choosing among metric families that reward fundamentally different behaviors. A model ranked best under mean absolute error may rank poorly under a proper scoring rule, because each metric encodes different assumptions about the cost of errors and the structure of uncertainty. This page maps those assumptions, the metric families that embody them, and the pitfalls they introduce. For individual metric parameters and usage, see the [API Reference: yohou.metrics](/pages/api/metrics/).

## Forecast Errors vs. Residuals

Two closely related quantities are easy to confuse. Residuals compare fitted (in-sample) values to actuals on the training set. Forecast errors compare genuine out-of-sample predictions to actuals on held-out data. Residuals tell you how well the model explains history. Forecast errors tell you how well it predicts the future. For model selection, forecast errors are what matter. Residuals are useful for [diagnostics](residual-diagnostics.md) but not for assessing predictive skill.

## Point Metrics

Point metrics evaluate single-value predictions against observed actuals. They
differ in how they weight errors, whether they depend on the target's scale, and
which pathological behaviors they are susceptible to.

### Scale-Dependent Metrics

Metrics expressed in the same units as the target are the simplest to interpret.
The fundamental limitation is that they cannot be compared across series with
different scales. A MAE of 10 on a series ranging from 0 to 100 is excellent; the
same MAE on a series from 0 to 10 is terrible. This also affects multivariate
forecasting: when aggregating across components with different scales (e.g.,
temperature in °C and energy in MWh), scale-dependent metrics are dominated by the
component with the largest values. Use scaled metrics like MASE, or exclude
`"componentwise"` from the aggregation to keep per-component scores (e.g.
`aggregation_method=["stepwise", "vintagewise"]`).

### Percentage Metrics

Percentage metrics normalize errors by true values, offering an intuitive
"percent off" interpretation. MAPE is the most commonly requested format for
business stakeholders. However, it has well-known problems: it is undefined when
true values are zero, and it is asymmetric (systematically favoring models that
under-predict). sMAPE addresses the asymmetry but becomes unstable when both truth
and prediction are small.

The asymmetry of MAPE means models optimized by it systematically bias toward under-prediction, making it unreliable as a selection criterion even when its intuitive scale makes it appropriate for stakeholder reporting.

### Scaled Metrics

Scaled metrics normalize errors against a naive seasonal baseline on the training
data, enabling cross-series comparison without the problems of percentage metrics.
A MASE of 0.8 means the model is 20% better than the seasonal naive baseline. A
MASE above 1.0 means the model is worse than simply repeating the last seasonal
cycle. This interpretability makes scaled metrics the recommended choice for model
selection, especially when comparing forecasters across multiple series with
different scales.

Both MASE and RMSSE accept a `seasonality` parameter (default 1) that defines the
seasonal period for the naive baseline. Set it to match the data's natural cycle
(e.g., 7 for daily data with weekly patterns, 12 for monthly data with yearly
patterns). A value of 1 uses the simple random walk as the baseline.

### Complete Point Metric Reference

| Metric | What it measures | Direction |
|---|---|---|
| [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.MeanAbsoluteError/) | Average absolute difference. Treats all errors equally, robust to outliers. | Lower |
| [`MeanSquaredError`](/pages/api/generated/yohou.metrics.MeanSquaredError/) | Average squared difference. Penalizes large errors disproportionately. | Lower |
| [`RootMeanSquaredError`](/pages/api/generated/yohou.metrics.RootMeanSquaredError/) | Square root of MSE. Penalizes large errors while maintaining original units. The gap between RMSE and MAE indicates how much error is dominated by occasional large misses. | Lower |
| [`MedianAbsoluteError`](/pages/api/generated/yohou.metrics.MedianAbsoluteError/) | Median of absolute differences. A single catastrophic forecast does not affect the metric at all. | Lower |
| [`MaxAbsoluteError`](/pages/api/generated/yohou.metrics.MaxAbsoluteError/) | Largest absolute error. Captures worst-case forecast deviation. | Lower |
| [`MeanAbsolutePercentageError`](/pages/api/generated/yohou.metrics.MeanAbsolutePercentageError/) | Average percentage error. Scale-independent but undefined at zero and asymmetric. | Lower |
| [`SymmetricMeanAbsolutePercentageError`](/pages/api/generated/yohou.metrics.SymmetricMeanAbsolutePercentageError/) | Symmetric average percentage error. Addresses MAPE's asymmetry but unstable when both truth and prediction are small. | Lower |
| [`MeanAbsoluteScaledError`](/pages/api/generated/yohou.metrics.MeanAbsoluteScaledError/) | MAE scaled by naive seasonal baseline error. Values below 1.0 indicate improvement over naive forecasting. Configurable `seasonality`. | Lower |
| [`RootMeanSquaredScaledError`](/pages/api/generated/yohou.metrics.RootMeanSquaredScaledError/) | RMSE scaled by naive seasonal baseline error. Combines large-error sensitivity with scale independence. Configurable `seasonality`. | Lower |
| [`R2Score`](/pages/api/generated/yohou.metrics.R2Score/) | Proportion of variance explained. 1.0 is perfect, 0.0 equals predicting the mean, negative means worse than the mean. | Higher |
| [`MeanDirectionalAccuracy`](/pages/api/generated/yohou.metrics.MeanDirectionalAccuracy/) | Proportion of steps where predicted direction of change matches actual direction. Evaluates trend capture independently of magnitude. | Higher |

## Interval Metrics

Evaluating prediction intervals requires balancing two properties that trade off
against each other:

- **Calibration**: does the interval contain the right proportion of observations?
  A 90% prediction interval should contain about 90% of true values.
- **Sharpness**: how narrow is the interval? Narrower is better, but only
  meaningful when compared at equal coverage.

Neither property alone is sufficient. An interval from negative infinity to
positive infinity has perfect coverage but is useless. The narrowest possible
interval has great sharpness but terrible coverage. The metrics below capture
different aspects of this tradeoff:

| Metric | What it measures | Direction |
|---|---|---|
| [`EmpiricalCoverage`](/pages/api/generated/yohou.metrics.EmpiricalCoverage/) | Proportion of true values inside the interval. Target equals the nominal coverage rate. | Match nominal |
| [`MeanIntervalWidth`](/pages/api/generated/yohou.metrics.MeanIntervalWidth/) | Average width of the prediction interval. Only meaningful when compared at equal coverage. | Lower |
| [`IntervalScore`](/pages/api/generated/yohou.metrics.IntervalScore/) | Interval width plus a penalty for observations outside the bounds (Winkler score). Combines sharpness and calibration in one number. | Lower |
| [`PinballLoss`](/pages/api/generated/yohou.metrics.PinballLoss/) | Asymmetric quantile loss for interval bounds. Penalizes under-prediction and over-prediction at different rates depending on the quantile. | Lower |
| [`CalibrationError`](/pages/api/generated/yohou.metrics.CalibrationError/) | Aggregate discrepancy between nominal and empirical coverage across all requested rates. Scale-independent (always 0 to 1). Requires at least two coverage rates. | Lower |
| [`ContinuousRankedProbabilityScore`](/pages/api/generated/yohou.metrics.ContinuousRankedProbabilityScore/) | CRPS approximated by averaging pinball losses across coverage rates. Integrates quantile loss as a proxy for the full predictive distribution. Requires at least two rates. | Lower |

[`IntervalScore`](/pages/api/generated/yohou.metrics.IntervalScore/) is the most widely used single metric for interval forecast
evaluation (used in the M4 and M5 competitions). For richer diagnostics, combine
[`EmpiricalCoverage`](/pages/api/generated/yohou.metrics.EmpiricalCoverage/) (is the interval well-calibrated?) with [`MeanIntervalWidth`](/pages/api/generated/yohou.metrics.MeanIntervalWidth/)
(is it sharp?). [`ContinuousRankedProbabilityScore`](/pages/api/generated/yohou.metrics.ContinuousRankedProbabilityScore/) is the strongest choice when
you evaluate across many coverage rates, because it captures the quality of the
entire predictive distribution rather than a single interval.

## Class-Probability Metrics

Class-probability metrics evaluate predicted probability distributions over
categorical classes. They divide into three groups: proper scoring rules that
evaluate calibration of the full distribution, hard-label metrics that convert
probabilities to class assignments via argmax, and ranking metrics that assess
discrimination ability across thresholds.

### Proper Scoring Rules

Proper scoring rules are uniquely minimized when predicted probabilities match true
class frequencies, making them more reliable for model selection than accuracy. They
penalize confident wrong predictions: a model that says "95% probability of class A"
when the answer is class B gets punished far more than one that says "55%
probability." Use proper scoring rules for model selection over accuracy.

| Metric | What it measures | Direction |
|---|---|---|
| [`LogLoss`](/pages/api/generated/yohou.metrics.LogLoss/) | Negative log-likelihood of the true class under the predicted distribution. Heavily penalizes confident wrong predictions. | Lower |
| [`BrierScore`](/pages/api/generated/yohou.metrics.BrierScore/) | Mean squared difference between predicted probabilities and one-hot encoded true labels. Multi-class generalization of the original Brier score. | Lower |
| [`RankedProbabilityScore`](/pages/api/generated/yohou.metrics.RankedProbabilityScore/) | Compares cumulative probability distributions for ordinal classes. Penalizes predictions far from the true class more than nearby misses. | Lower |

### Hard-Label Metrics

Hard-label metrics convert predicted probabilities to class assignments (via
argmax) before evaluation. They are familiar from standard classification but
discard calibration information. For multiclass problems, `Precision`, `Recall`,
`FBetaScore`, and the ranking metrics accept an `average` parameter (default
`"macro"`) that controls how per-class scores are combined.

| Metric | What it measures | Direction |
|---|---|---|
| [`Accuracy`](/pages/api/generated/yohou.metrics.Accuracy/) | Fraction of steps where predicted class matches true class. Misleading when classes are imbalanced. | Higher |
| [`Precision`](/pages/api/generated/yohou.metrics.Precision/) | Ratio of true positives to predicted positives. Measures how trustworthy positive predictions are. | Higher |
| [`Recall`](/pages/api/generated/yohou.metrics.Recall/) | Ratio of true positives to actual positives. Measures how many positive cases are captured. | Higher |
| [`FBetaScore`](/pages/api/generated/yohou.metrics.FBetaScore/) | Weighted harmonic mean of precision and recall. `beta=1.0` (F1) gives equal weight; `beta>1.0` emphasizes recall. | Higher |

### Ranking Metrics

Ranking metrics evaluate how well predicted probabilities separate classes across
all possible decision thresholds.

| Metric | What it measures | Direction |
|---|---|---|
| [`ROCAuC`](/pages/api/generated/yohou.metrics.ROCAuC/) | Area under the ROC curve. Uses one-vs-rest strategy for multiclass problems. | Higher |
| [`PRAuC`](/pages/api/generated/yohou.metrics.PRAuC/) | Area under the precision-recall curve. More informative than ROC AuC when classes are imbalanced. Uses one-vs-rest strategy. | Higher |

See [Class-Probability Forecasting](class-probability-forecasting.md)
for the full treatment of categorical prediction.

## Aggregation

A forecast error distribution has structure across multiple dimensions: forecast
steps within a single prediction (the horizon dimension), forecast origins (the
vintage dimension), target columns (the component dimension), and, for panel data,
entities (the group dimension). The `aggregation_method` parameter controls which
of these dimensions the scorer collapses (averages over).

The parameter accepts a single string or a list of strings. Each string names one
dimension to collapse. When used alone, it collapses only that dimension while
preserving all others:

- `"stepwise"`: collapses forecast steps. Returns per-vintage, per-component
  scores. Reveals whether accuracy changes across forecast origins, which is
  the key diagnostic for detecting concept drift in deployed forecasters.
- `"vintagewise"`: collapses vintage origins. Returns per-step, per-component
  scores. Reveals whether errors grow with forecast horizon, a signature of
  accumulated uncertainty in recursive forecasting or an insufficient feature
  set at longer lead times.
- `"componentwise"`: collapses target columns into a single score. Returns
  per-step, per-vintage scores. Useful for reducing multivariate forecasts
  to one aggregate error trajectory.
- `"groupwise"`: collapses panel entities. Returns per-step, per-vintage,
  per-component scores. Averages across groups while preserving target
  columns and time granularity.
- `"coveragewise"` (interval scorers only): collapses coverage rates. Averages
  scores across all requested coverage levels (e.g., 50%, 80%, 95%).

The default `"all"` collapses everything into a single scalar. That scalar is
what model selection procedures consume, but it hides variation that diagnostic
questions require. Passing a list like `["stepwise", "componentwise"]` collapses
both steps and components while preserving vintages.

No re-fitting is needed to switch between aggregation views.

## Vintage-based Evaluation

A *vintage* is a single forecast origin: the point in time at which the forecaster
last observed data before predicting. During rolling evaluation, each call to
`observe_predict` produces one vintage. The resulting predictions carry a
`vintage_time` column that records the last observed timestamp, so every predicted
row can be traced back to the information that was available when the prediction
was made.

Evaluating across vintages answers a different question than evaluating across
horizon steps. `"stepwise"` aggregation collapses the forecast-step dimension,
producing per-vintage scores that reveal whether the model is degrading (or
improving) as more data arrives. `"vintagewise"` aggregation collapses the
vintage dimension, producing per-step scores that reveal whether the model is
worse at longer lead times. Both views are available from the same scorer by
changing the `aggregation_method` parameter.

## Scorer Workflow

Every scorer follows a two-step pattern: `fit`, then `score`.

```python
scorer = MeanAbsoluteError()
scorer.fit(y_train)              # stores training-set statistics
result = scorer.score(y_test, y_pred)
```

The `fit` call is not optional. Scaled metrics (MASE, RMSSE) use training-set
error as a denominator, and all scorers use `fit` to register the data schema.
Calling `score` without a prior `fit` raises an error.

Internally, `score` is a template method that calls `_compute_raw_errors` (the
metric-specific logic), applies time weights if provided, aggregates according to
`aggregation_method`, and runs any post-aggregation transform (e.g. square root
for RMSE). Custom scorers override only `_compute_raw_errors`; the rest of the
pipeline is inherited. See
[How to Create Custom Scorers](/pages/how-to/create-a-scorer/) for a walkthrough.

## Weighting

Most scorers (for example
[`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.MeanAbsoluteError/))
accept optional weighter parameters that apply non-uniform emphasis before
aggregation. Each takes a weighter estimator rather than a raw weight series:

- `time_weighter` weights per-timestep errors. This matters when recent errors
  carry more business value than older ones (rolling deployment), or when
  certain periods are critical (holiday weeks in retail).
- `step_weighter` weights per-forecasting-step errors (1-step-ahead,
  2-step-ahead, etc.). Useful when near-term accuracy matters more than
  distant forecasts.
- `vintage_weighter` weights per-vintage (forecast origin) scores. Controls how
  much each forecast origin contributes to the aggregated result.

See [Weighting](/pages/explanation/weighting/) for weight types, formats, and
normalization rules, and
[How to Use Time Weighting](/pages/how-to/time-weighting/) for practical patterns.

## References

- Hyndman, R.J. & Athanasopoulos, G. (2021). [Forecasting: principles and practice](https://otexts.com/fpp3/), 3rd edition, OTexts. Chapters 5.8 and 5.9.
- Hyndman, R.J. & Koehler, A.B. (2006). "Another look at measures of forecast accuracy." International Journal of Forecasting, 22(4), 679-688. [doi:10.1016/j.ijforecast.2006.03.001](https://doi.org/10.1016/j.ijforecast.2006.03.001)
- Gneiting, T. & Raftery, A.E. (2007). "Strictly proper scoring rules, prediction, and estimation." Journal of the American Statistical Association, 102(477), 359-378. [doi:10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)

## Connections

Metrics tie directly into [Model Selection](model-selection.md), where scorers
define the objective function for cross-validation and hyperparameter search. For
understanding what model residuals reveal about predictive gaps, see
[Residual Diagnostics](residual-diagnostics.md). The
[Interval Forecasting](interval-forecasting.md) page explains the conformal
prediction framework that produces the intervals these metrics evaluate.
[API Reference: yohou.metrics](/pages/api/metrics/) has the full listing with
parameters and examples for each scorer class.

For practical recipes, see [How to Evaluate Forecast Accuracy](../how-to/evaluate-forecast-accuracy.md) and [How to Create a Custom Scorer](../how-to/create-a-scorer.md).
