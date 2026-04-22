# Forecast Accuracy

A forecast is only useful if you can measure how good it is. Choosing the right metric is not a formality: different metrics reward different behaviors, and the "best" model under one metric may be mediocre under another. This page explains the reasoning behind metric families and common pitfalls, helping you make informed choices. For individual metric parameters and usage, see the [API Reference: yohou.metrics](/pages/api/metrics/).

## Forecast Errors vs. Residuals

Two closely related quantities are easy to confuse. Residuals compare fitted (in-sample) values to actuals on the training set. Forecast errors compare genuine out-of-sample predictions to actuals on held-out data. Residuals tell you how well the model explains history. Forecast errors tell you how well it predicts the future. For model selection, forecast errors are what matter. Residuals are useful for [diagnostics](residual-diagnostics.md) but not for assessing predictive skill.

## Metric Families

### Scale-Dependent Metrics (MAE, MSE, RMSE, MdAE)

Metrics expressed in the same units as the target are the simplest to interpret.
MAE treats all errors equally. MSE penalizes large errors disproportionately,
which is useful when large forecast misses carry outsized costs. RMSE restores the
original units while keeping that sensitivity. The gap between RMSE and MAE is
informative: wide gaps indicate that error is dominated by occasional large misses.

MdAE reports the median rather than the mean, making it the most robust option: a
single catastrophic forecast does not affect the metric at all.

The fundamental limitation of scale-dependent metrics is that they cannot be
compared across series with different scales. A MAE of 10 on a series ranging
from 0 to 100 is excellent; the same MAE on a series from 0 to 10 is terrible.

### Percentage Metrics (MAPE, sMAPE)

Percentage metrics normalize errors by true values, offering an intuitive
"percent off" interpretation. MAPE is the most commonly requested format for
business stakeholders. However, it has well-known problems: it is undefined when
true values are zero, and it is asymmetric (systematically favoring models that
under-predict). sMAPE addresses the asymmetry but becomes unstable when both truth
and prediction are small.

Use percentage metrics for reporting, not for model selection.

### Scaled Metrics (MASE, RMSSE)

Scaled metrics normalize errors against a naive seasonal baseline on the training
data, enabling cross-series comparison without the problems of percentage metrics.
A MASE of 0.8 means the model is 20% better than the seasonal naive baseline. A
MASE above 1.0 means the model is worse than simply repeating the last seasonal
cycle.

This interpretability makes scaled metrics the recommended choice for model
selection, especially when comparing forecasters across multiple series with
different scales.

## Interval Metrics

Evaluating prediction intervals requires balancing two properties that trade off
against each other:

- **Calibration**: does the interval contain the right proportion of observations?
  A 90% prediction interval should contain about 90% of true values.
- **Sharpness**: how narrow is the interval? Narrower is better, but only
  meaningful when compared at equal coverage.

Neither property alone is sufficient. An interval from negative infinity to
positive infinity has perfect coverage but is useless. The narrowest possible
interval has great sharpness but terrible coverage. The Interval Score (Winkler
score) combines both into a single metric: it equals the interval width plus a
penalty for observations that fall outside the bounds. Lower is better.

Calibration Error aggregates coverage deviations across all requested coverage
rates, providing a single number for overall interval quality.

## Classification Scoring Rules

For categorical forecasts, accuracy alone is misleading when classes are imbalanced.
Proper scoring rules (Log Loss, Brier Score) are uniquely minimized when predicted
probabilities match true class frequencies, making them more reliable for model
selection than accuracy. They penalize confident wrong predictions: a model that
says "95% probability of class A" when the answer is class B gets punished far
more than one that says "55% probability." Use proper scoring rules for model
selection over accuracy. See [Class-Probability Forecasting](class-probability-forecasting.md)
for the full treatment.

## Aggregation

The same scorer can produce different views of the same error distribution via the
`aggregation_method` parameter:

- **"all"** (default): a single scalar summarizing the full error. This is what
  most model selection procedures need.
- **"stepwise"**: per-component scores collapsed across forecast steps, useful
  for identifying which target columns or panel groups the model struggles with.
- **"vintagewise"**: per-component scores collapsed across vintages, useful for
  tracking how accuracy changes as new data arrives.
- **"componentwise"**: per-timestep scores, useful for seeing whether errors grow
  with forecast horizon.
- **"groupwise"**: per-component per-timestep for panel data, analyzing temporal
  patterns without group noise.

## Vintage-based Evaluation

A *vintage* is a single forecast origin: the point in time at which the forecaster
last observed data before predicting. During rolling evaluation, each call to
`observe_predict` produces one vintage. The resulting predictions carry a
`vintage_time` column that records the last observed timestamp, so every predicted
row can be traced back to the information that was available when the prediction
was made.

Evaluating across vintages answers a different question than evaluating across
horizon steps. Stepwise aggregation reveals whether the model is worse at longer
lead times. Vintagewise aggregation reveals whether the model is degrading (or
improving) as more data arrives, which is critical for monitoring deployed
forecasters. Both views are available from the same scorer output by changing the
`aggregation_method` parameter.

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
[How to Create Custom Scorers](/pages/how-to/custom-scorers/) for a walkthrough.

## Weighting

All scorers accept optional weight parameters that apply non-uniform emphasis
before aggregation:

- `time_weight` weights per-timestep errors. This matters when recent errors
  carry more business value than older ones (rolling deployment), or when
  certain periods are critical (holiday weeks in retail).
- `step_weight` weights per-forecasting-step errors (1-step-ahead,
  2-step-ahead, etc.). Useful when near-term accuracy matters more than
  distant forecasts.
- `vintage_weight` weights per-vintage (forecast origin) scores. Controls how
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
