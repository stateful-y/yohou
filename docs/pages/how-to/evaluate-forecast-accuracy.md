# How to Evaluate Forecast Accuracy

This guide shows you how to measure and compare forecast performance using
yohou's scorers, cross-validation, and baseline comparisons.

## Prerequisites

- Yohou installed ([Getting Started](../tutorials/getting-started.md))
- Familiarity with the fit-predict workflow ([Getting Started](../tutorials/getting-started.md))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## 1. Score a Single Forecast

Every scorer follows a two-step pattern: `fit` on training data (to set
internal state such as the training mean for scaled metrics), then `score`
with the test set and predictions:

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster
from yohou.metrics import MeanAbsoluteError
from yohou.datasets import fetch_electricity_demand
from yohou.model_selection import train_test_split

data = fetch_electricity_demand()
y = data.frame

y_train, y_test = train_test_split(y, test_size=48)

forecaster = PointReductionForecaster(estimator=Ridge())
forecaster.fit(y_train, forecasting_horizon=24)
y_pred = forecaster.predict()

scorer = MeanAbsoluteError()
scorer.fit(y_train)
mae = scorer.score(y_test, y_pred)
```

If you need to compare across series with different scales, use
[`MeanAbsoluteScaledError`](/pages/api/generated/yohou.metrics.point.MeanAbsoluteScaledError/)
instead. See [Forecast Accuracy](../explanation/forecast-accuracy.md) for
guidance on choosing the right metric, and the
[metrics API reference](../api/metrics.md) for the complete list.

## 2. Evaluate with Cross-Validation

Use a temporal splitter with
[`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/)
or [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/)
to get robust estimates across multiple train-test folds:

```python
from yohou.model_selection import GridSearchCV, ExpandingWindowSplitter
from yohou.metrics import MeanAbsoluteError

search = GridSearchCV(
    forecaster=PointReductionForecaster(estimator=Ridge()),
    param_grid={},  # empty grid evaluates the forecaster as-is
    scoring=MeanAbsoluteError(),
    cv=ExpandingWindowSplitter(n_splits=5, test_size=14),
)
search.fit(y, forecasting_horizon=14)

# Per-fold and mean scores
print(search.cv_results_)
```

## 3. Compare Against a Naive Baseline

Evaluate a [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/)
forecaster on the same splits to confirm your model outperforms simple
benchmarks:

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

## 4. Use Multiple Metrics Simultaneously

Pass a dictionary of scorers to evaluate on several metrics at once:

```python
from yohou.metrics import MeanAbsoluteError, RootMeanSquaredError, MeanAbsoluteScaledError

search = GridSearchCV(
    forecaster=PointReductionForecaster(estimator=Ridge()),
    param_grid={"estimator__alpha": [0.1, 1.0, 10.0]},
    scoring={
        "mae": MeanAbsoluteError(),
        "rmse": RootMeanSquaredError(),
        "mase": MeanAbsoluteScaledError(),
    },
    refit="mae",
    cv=ExpandingWindowSplitter(n_splits=5, test_size=14),
)
search.fit(y, forecasting_horizon=14)
```

The `refit` parameter specifies which metric determines the best
configuration. All metrics appear in `cv_results_`.

## 5. Evaluate Interval Forecasts

Use [`EmpiricalCoverage`](/pages/api/generated/yohou.metrics.interval.EmpiricalCoverage/)
to check whether intervals contain the true values at the claimed rate,
and [`IntervalScore`](/pages/api/generated/yohou.metrics.interval.IntervalScore/)
to penalize both under-coverage and unnecessarily wide intervals:

```python
from yohou.metrics import EmpiricalCoverage, IntervalScore
from yohou.interval import SplitConformalForecaster

interval_forecaster = SplitConformalForecaster(
    point_forecaster=PointReductionForecaster(estimator=Ridge()),
)
interval_forecaster.fit(y_train, forecasting_horizon=24, coverage_rates=[0.90])
y_pred_interval = interval_forecaster.predict_interval()

coverage_scorer = EmpiricalCoverage()
coverage_scorer.fit(y_train)
print(coverage_scorer.score(y_test, y_pred_interval))

interval_scorer = IntervalScore()
interval_scorer.fit(y_train)
print(interval_scorer.score(y_test, y_pred_interval))
```

A well-calibrated 90% interval should achieve empirical coverage close
to 0.9. If coverage is substantially lower, the intervals are too narrow.
See [Produce Prediction Intervals](interval-forecasting.md) for the full
interval forecasting workflow.

## 6. Apply Time Weighting

Weight recent errors more heavily using
[`exponential_decay_weight`](/pages/api/generated/yohou.utils.weighting.exponential_decay_weight/):

```python
from yohou.utils.weighting import exponential_decay_weight

weight_fn = exponential_decay_weight(half_life=365)
weighted_mae = scorer.score(y_test, y_pred, time_weight=weight_fn)
```

See [Time Weighting](time-weighting.md) for the full guide on weight
functions.

## 7. Evaluate Classification Forecasts

For class-probability forecasts, use proper scoring rules such as
[`LogLoss`](/pages/api/generated/yohou.metrics.class_proba.LogLoss/) and
[`BrierScore`](/pages/api/generated/yohou.metrics.class_proba.BrierScore/).
See [Forecast with Class Probabilities](class-probability-forecasting.md) for
the full classification workflow and scoring examples.

## 8. Score Panel Forecasts

Scorers handle panel data automatically. Use
`aggregation_method="groupwise"` to get one score per group so you can
spot underperforming entities:

```python
from yohou.metrics import MeanAbsoluteError

scorer = MeanAbsoluteError(aggregation_method="groupwise")
scorer.fit(y_train)
scores = scorer.score(y_test, y_pred)  # one row per group
```

See [Work with Panel Data](panel-data.md) for the full panel forecasting
workflow and [Forecast Accuracy](../explanation/forecast-accuracy.md) for
aggregation mode details.

## See Also

- [Visualize and Compare Model Scores](visualize-scores.md) for per-step accuracy, per-vintage trends, and model comparison plots
- [Forecast Accuracy](../explanation/forecast-accuracy.md) for conceptual background on metrics and proper scoring rules
- [Work with Panel Data](panel-data.md) for panel-level scoring and aggregation strategies
- [API Reference: yohou.metrics](../api/metrics.md) for the full list of available metrics
