# How to Evaluate Forecasts with Multi-vintage Scoring

This guide shows how to generate forecasts from multiple observation points and
score them across vintages. Use this when you need to assess whether a model's
accuracy is stable over time or when you want to break down errors by forecast
horizon step.

## Prerequisites

- A fitted forecaster ([First Forecast](../tutorials/first-forecast.md))
- Familiarity with basic scorer usage ([Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md))

## Generate Multi-vintage Predictions

Call `observe_predict` with a `stride` parameter to produce predictions from
successive observation points. Each prediction is a *vintage* anchored to a
`vintage_time` column:

```python
from copy import deepcopy

forecaster.fit(y_train, forecasting_horizon=7)

y_pred = deepcopy(forecaster).observe_predict(
    y_test, forecasting_horizon=7, stride=1
)
```

`stride=1` advances the observation window by one time step between predictions,
creating one vintage per test row. Larger strides produce fewer vintages. When
`stride` is omitted, it defaults to the forecasting horizon (non-overlapping
windows).

The output is a single DataFrame where each `vintage_time` value identifies one
forecast origin. Use `y_pred["vintage_time"].unique()` to see all vintages.

!!! tip
    Always `deepcopy` the forecaster before calling `observe_predict`. The method
    mutates internal state, so a copy preserves the original for further use.

If your forecaster uses exogenous features, pass them via `X_actual` (for historical observations) and optionally `X_future` or `X_forecast` (for known-ahead or external forecast data):

```python
y_pred = deepcopy(forecaster).observe_predict(
    y_test, X_actual=X_test, forecasting_horizon=7, stride=1
)
```

## Score Across Vintages

Fit a scorer on the training data and call `score` with the multi-vintage
predictions:

```python
from yohou.metrics import MeanAbsoluteError

mae = MeanAbsoluteError()
mae.fit(y_train)
score = mae.score(y_test, y_pred)  # single aggregate score
```

To break scores down by dimension, set `aggregation_method` at construction:

```python
# Per-vintage scores (one value per forecast origin)
mae_vw = MeanAbsoluteError(aggregation_method="vintagewise")
mae_vw.fit(y_train)
scores_per_vintage = mae_vw.score(y_test, y_pred)

# Per-step scores (one value per horizon position)
mae_sw = MeanAbsoluteError(aggregation_method="stepwise")
mae_sw.fit(y_train)
scores_per_step = mae_sw.score(y_test, y_pred)
```

Other aggregation methods include `"componentwise"` (per target column) and
`"groupwise"` (per panel group). Pass `"all"` (the default) for a single
aggregate scalar.

## Visualize Per-step Accuracy

[`plot_score_per_step`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_step/)
shows how the scorer value varies across forecast horizon steps. Pass a dict of
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

## Visualize Per-vintage Accuracy

[`plot_score_per_vintage`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_vintage/)
tracks accuracy over successive forecast origins:

```python
from yohou.plotting import plot_score_per_vintage

plot_score_per_vintage(mae, y_test, y_pred, show_trend=True)
```

## Visualize the Step x Vintage Heatmap

[`plot_score_heatmap`](/pages/api/generated/yohou.plotting.evaluation.plot_score_heatmap/)
displays a 2D grid where each cell is the error for a specific step at a specific
vintage:

```python
from yohou.plotting import plot_score_heatmap

plot_score_heatmap(mae, y_test, y_pred)
```

This function accepts a single scorer and a single prediction DataFrame (not
dicts). To swap axes, pass `x_dim="vintage", y_dim="step"`.

## Score Interval Forecast Vintages

For interval forecasters, use `observe_predict_interval` and interval-specific
scorers:

```python
y_pred_interval = deepcopy(interval_forecaster).observe_predict_interval(
    y_test, forecasting_horizon=7, stride=1
)

from yohou.metrics import IntervalScore

interval_scorer = IntervalScore()
interval_scorer.fit(y_train)
score = interval_scorer.score(y_test, y_pred_interval)
```

To restrict evaluation to specific coverage rates, pass `coverage_rates` to the
scorer constructor. All the same aggregation methods and plotting functions work
with interval predictions.

## Check Splitter Alignment

When using cross-validation, the interaction between `test_size`, `stride`, and
`forecasting_horizon` determines how many vintages each fold produces and whether
all forecast steps are represented equally:

```python
from yohou.model_selection import SlidingWindowSplitter, check_cv_alignment

cv = SlidingWindowSplitter(n_splits=3, test_size=10, stride=4)
info = check_cv_alignment(cv, forecasting_horizon=7)
print(info["is_balanced"])  # True if every step has equal vintage coverage
print(info["n_vintages"])   # vintages per fold
```

Run this before starting a search to confirm the evaluation geometry matches
your expectations.

## See Also

- [Forecast Accuracy: Vintage-based Evaluation](../explanation/forecast-accuracy.md#vintage-based-evaluation) for the conceptual background
- [How to Visualize and Compare Model Scores](visualize-scores.md) for the full plotting workflow
- [Model Selection](../explanation/model-selection.md#checking-splitter-alignment) for splitter alignment details
