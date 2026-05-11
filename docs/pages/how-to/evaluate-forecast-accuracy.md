# How to Evaluate Forecast Accuracy

This guide shows how to measure and compare forecast performance using yohou's metrics and cross-validation tools.

## Score a Single Forecast

Fit a scorer on the training data, then call `score` with the test set and predictions:

```python
from yohou.metrics import MeanAbsoluteError

scorer = MeanAbsoluteError()
scorer.fit(y_train)
mae = scorer.score(y_test, y_pred)
```

Every scorer follows this pattern: `fit` sets internal state (e.g., the training mean for scaled metrics), and `score` returns a single numeric value.

## Choose the Right Metric

Different metrics answer different questions:

| Metric | Best for | Limitation |
|---|---|---|
| [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.point.MeanAbsoluteError/) | General-purpose, interpretable in original units | Depends on series scale |
| [`RootMeanSquaredError`](/pages/api/generated/yohou.metrics.point.RootMeanSquaredError/) | Penalizing large errors | Sensitive to outliers |
| [`MeanAbsoluteScaledError`](/pages/api/generated/yohou.metrics.point.MeanAbsoluteScaledError/) | Comparing across different series | Undefined when training data is constant |
| [`MeanAbsolutePercentageError`](/pages/api/generated/yohou.metrics.point.MeanAbsolutePercentageError/) | Relative accuracy, scale-free | Undefined when actual values are zero |
| [`SymmetricMeanAbsolutePercentageError`](/pages/api/generated/yohou.metrics.point.SymmetricMeanAbsolutePercentageError/) | Bounded alternative to MAPE | Treats over- and under-prediction asymmetrically |

For cross-series comparison, prefer scaled metrics like [`MeanAbsoluteScaledError`](/pages/api/generated/yohou.metrics.point.MeanAbsoluteScaledError/). For single-series evaluation, scale-dependent metrics like [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.point.MeanAbsoluteError/) are easier to interpret.

## Evaluate with Cross-Validation

Use a temporal splitter with [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) or [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/) to get robust estimates across multiple train-test folds:

```python
from yohou.model_selection import GridSearchCV, ExpandingWindowSplitter
from yohou.metrics import MeanAbsoluteError

search = GridSearchCV(
    forecaster=my_forecaster,
    param_grid={},  # empty grid evaluates the forecaster as-is
    scoring=MeanAbsoluteError(),
    cv=ExpandingWindowSplitter(n_splits=5, test_size=14),
)
search.fit(y, X_actual=X, forecasting_horizon=14)

# Per-fold and mean scores
print(search.cv_results_)
```

## Compare Against a Naive Baseline

A model is only useful if it outperforms simple benchmarks. Evaluate a [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/) forecaster on the same splits:

```python
from yohou.point import SeasonalNaive

baseline_search = GridSearchCV(
    forecaster=SeasonalNaive(seasonality=7),
    param_grid={},
    scoring=MeanAbsoluteError(),
    cv=ExpandingWindowSplitter(n_splits=5, test_size=14),
)
baseline_search.fit(y, forecasting_horizon=14)

print(f"Model MAE: {search.best_score_:.2f}")
print(f"Baseline MAE: {baseline_search.best_score_:.2f}")
```

## Use Multiple Metrics Simultaneously

Pass a dictionary of scorers to evaluate on several metrics at once:

```python
from yohou.metrics import MeanAbsoluteError, RootMeanSquaredError, MeanAbsoluteScaledError

search = GridSearchCV(
    forecaster=my_forecaster,
    param_grid=param_grid,
    scoring={
        "mae": MeanAbsoluteError(),
        "rmse": RootMeanSquaredError(),
        "mase": MeanAbsoluteScaledError(),
    },
    refit="mae",
    cv=ExpandingWindowSplitter(n_splits=5, test_size=14),
)
search.fit(y, X_actual=X, forecasting_horizon=14)
```

The `refit` parameter specifies which metric determines the best configuration. All metrics appear in `cv_results_`.

## Evaluate Interval Forecasts

Interval forecasters require interval-specific metrics. [`EmpiricalCoverage`](/pages/api/generated/yohou.metrics.interval.EmpiricalCoverage/) checks whether the stated coverage rate is achieved; [`IntervalScore`](/pages/api/generated/yohou.metrics.interval.IntervalScore/) penalizes both width and miscoverage:

```python
from yohou.metrics import EmpiricalCoverage, IntervalScore

coverage_scorer = EmpiricalCoverage()
coverage_scorer.fit(y_train)
coverage = coverage_scorer.score(y_test, y_pred_interval)

interval_scorer = IntervalScore()
interval_scorer.fit(y_train)
score = interval_scorer.score(y_test, y_pred_interval)
```

A well-calibrated 90% interval should achieve empirical coverage close to 0.9. If coverage is substantially lower, the intervals are too narrow.

## Apply Time Weighting

Weight recent errors more heavily using [`exponential_decay_weight`](/pages/api/generated/yohou.utils.weighting.exponential_decay_weight/):

```python
from yohou.utils.weighting import exponential_decay_weight

weight_fn = exponential_decay_weight(half_life=365)
weighted_mae = scorer.score(y_test, y_pred, time_weight=weight_fn)
```

See [Time Weighting](time-weighting.md) for the full guide on weight functions.

## Visualize Evaluation Results

Plot per-timestep residuals and cross-validation results:

```python
from yohou.plotting import plot_residuals, plot_cv_results_scatter

# Residual analysis
plot_residuals(y_pred, y_test)

# Compare hyperparameter configurations from search
plot_cv_results_scatter(search.cv_results_, param_name="estimator__alpha")
```

For a model-level comparison, [`plot_score_summary`](/pages/api/generated/yohou.plotting.evaluation.plot_score_summary/) visualizes aggregate scores as a grouped bar chart:

```python
from yohou.plotting import plot_score_summary

plot_score_summary(
    {"MAE": mae, "RMSE": rmse},
    y_test,
    {"Model A": y_pred_a, "Model B": y_pred_b},
)
```

To see how accuracy varies across forecast horizon steps or over successive vintages,
use [`plot_score_per_step`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_step/)
and [`plot_score_per_vintage`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_vintage/).
See [Visualization](../explanation/visualization.md) for the full evaluation plotting
workflow.

## Classification Metrics

For class-probability forecasts, use classification-specific metrics:

```python
from yohou.metrics import LogLoss, BrierScore, Accuracy

# Evaluate probability predictions
log_loss = LogLoss().fit(y_test).score(y_test, y_proba)
brier = BrierScore().fit(y_test).score(y_test, y_proba)
accuracy = Accuracy().fit(y_test).score(y_test, y_proba)
```

Prefer [`LogLoss`](/pages/api/generated/yohou.metrics.class_proba.LogLoss/) or [`BrierScore`](/pages/api/generated/yohou.metrics.class_proba.BrierScore/) over [`Accuracy`](/pages/api/generated/yohou.metrics.class_proba.Accuracy/) for model selection. They
are proper scoring rules that reward calibrated probabilities.

Assess calibration visually with [`plot_calibration()`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/):

```python
from yohou.plotting import plot_calibration

plot_calibration(y_test, y_proba)
```
