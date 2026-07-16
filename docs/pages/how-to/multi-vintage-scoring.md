# How to Evaluate Forecasts with Multi-vintage Scoring

This guide shows you how to generate forecasts from multiple observation points
and score them across vintages. Use this when you need to assess whether a
model's accuracy is stable over time or when you want to break down errors by
forecast horizon step.

## Prerequisites

- A fitted forecaster ([Getting Started](../tutorials/getting-started.md))
- Familiarity with basic scorer usage ([Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md))
- `yohou[plotting]` installed for visualization steps (`pip install "yohou[plotting]"`)

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## 1. Generate Multi-vintage Predictions

Call `observe_predict` with a `stride` to produce predictions from successive
observation points. Each prediction is a *vintage* identified by a
`vintage_time` column in the output:

```python
from copy import deepcopy

forecaster.fit(y_train, forecasting_horizon=7)

y_pred = deepcopy(forecaster).observe_predict(
    y_test, forecasting_horizon=7, stride=1
)
```

`stride=1` creates one vintage per test row. Larger strides produce fewer
vintages. When omitted, `stride` defaults to the forecasting horizon
(non-overlapping windows).

!!! tip
    Always `deepcopy` the forecaster before calling `observe_predict`. The
    method mutates internal state, so a copy preserves the original for
    further use.

If your forecaster uses exogenous features, pass them via `X_actual` and
optionally `X_future` or `X_forecast`:

```python
y_pred = deepcopy(forecaster).observe_predict(
    y_test, X_actual=X_test, forecasting_horizon=7, stride=1
)
```

## 2. Score Across Vintages

Fit a scorer on the training data and call `score` with the multi-vintage
predictions:

```python
from yohou.metrics import MeanAbsoluteError

mae = MeanAbsoluteError()
mae.fit(y_train)
score = mae.score(y_test, y_pred)  # single aggregate score
```

To get scores along a specific axis, set `aggregation_method` at construction:

```python
# One score per forecast origin
mae_vw = MeanAbsoluteError(aggregation_method="vintagewise")
mae_vw.fit(y_train)
scores_per_vintage = mae_vw.score(y_test, y_pred)

# One score per horizon position
mae_sw = MeanAbsoluteError(aggregation_method="stepwise")
mae_sw.fit(y_train)
scores_per_step = mae_sw.score(y_test, y_pred)
```

The available methods are `"vintagewise"`, `"stepwise"`, `"componentwise"` (per
target column), `"groupwise"` (per panel group), and `"all"` (the default single
scalar). See [Aggregation](../explanation/forecast-accuracy.md#aggregation) for
guidance on when to use each.

## 3. Visualize Accuracy by Horizon Step

[`plot_score_per_step`](/pages/api/generated/yohou.plotting.plot_score_per_step/)
reveals whether accuracy degrades at longer horizon positions. Pass a dict of
predictions to compare multiple models:

```python
from yohou.plotting import plot_score_per_step

plot_score_per_step(
    mae,
    y_test,
    {"Model A": y_pred_a, "Model B": y_pred_b},
)
```

To add a linear trend overlay, pass `show_trend=True`. To switch from lines to
bars, pass `kind="bar"`.

## 4. Track Accuracy Over Forecast Origins

[`plot_score_per_vintage`](/pages/api/generated/yohou.plotting.plot_score_per_vintage/)
shows whether accuracy is stable across successive vintages or drifting over
time:

```python
from yohou.plotting import plot_score_per_vintage

plot_score_per_vintage(mae, y_test, y_pred, show_trend=True)
```

## 5. Inspect the Full Step x Vintage Grid

[`plot_score_heatmap`](/pages/api/generated/yohou.plotting.plot_score_heatmap/)
plots a 2D grid where each cell is the error for one step at one vintage,
useful for spotting localized pockets of poor accuracy:

```python
from yohou.plotting import plot_score_heatmap

plot_score_heatmap(mae, y_test, y_pred)
```

Pass `x_dim="vintage", y_dim="step"` to swap axes. This function accepts a
single scorer and a single prediction DataFrame (not dicts).

## 6. Score Interval Forecast Vintages

For interval forecasters, use `observe_predict_interval` with an
interval scorer:

```python
from yohou.metrics import IntervalScore

y_pred_interval = deepcopy(interval_forecaster).observe_predict_interval(
    y_test, forecasting_horizon=7, stride=1
)

interval_scorer = IntervalScore()
interval_scorer.fit(y_train)
score = interval_scorer.score(y_test, y_pred_interval)
```

To restrict evaluation to specific coverage rates, pass `coverage_rates` to the
scorer constructor. All aggregation methods and plotting functions described
above work with interval predictions.

## 7. Verify Splitter Alignment

When combining multi-vintage scoring with cross-validation, run
`check_cv_alignment` to confirm the interaction between `test_size`, `stride`,
and `forecasting_horizon` produces the evaluation geometry you expect:

```python
from yohou.model_selection import SlidingWindowSplitter, check_cv_alignment

cv = SlidingWindowSplitter(n_splits=3, test_size=10, stride=4)
info = check_cv_alignment(cv, forecasting_horizon=7)
print(info["is_balanced"])  # True if every step has equal vintage coverage
print(info["n_vintages"])   # vintages per fold
```

See [Checking Splitter Alignment](../explanation/model-selection.md#checking-splitter-alignment)
for details on interpreting the output.

## See Also

- [Vintage-based Evaluation](../explanation/forecast-accuracy.md#vintage-based-evaluation) for the conceptual background
- [Visualize and Compare Model Scores](visualize-scores.md) for the full plotting workflow
- [`yohou.plotting` API reference](/pages/api/plotting/) for all evaluation plot options
