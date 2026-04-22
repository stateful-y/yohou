# How to Visualize and Compare Model Scores

This guide shows how to use yohou's evaluation plotting functions to compare
models, diagnose weaknesses, and communicate results. Use this after generating
predictions and fitting scorers.

## Prerequisites

- Scored predictions from one or more models ([Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md))

## Compare Models with a Summary Bar Chart

[`plot_score_summary`](/pages/api/generated/yohou.plotting.evaluation.plot_score_summary/)
produces a grouped bar chart comparing aggregate scores across models and
metrics:

```python
from yohou.metrics import MeanAbsoluteError, RootMeanSquaredError
from yohou.plotting import plot_score_summary

scorer = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
for s in scorer.values():
    s.fit(y_train)

y_pred = {"Naive": y_pred_naive, "Ridge": y_pred_ridge}
plot_score_summary(scorer, y_truth, y_pred)
```

To sort bars by score value, pass `sort_ascending=True` (or `False` for
descending).

## Check Horizon Degradation

[`plot_score_per_step`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_step/)
shows how accuracy changes at each forecast horizon step:

```python
from yohou.plotting import plot_score_per_step

# Line chart (default)
plot_score_per_step(scorer, y_truth, y_pred)

# Bar chart with a linear trend overlay
plot_score_per_step(scorer, y_truth, y_pred, kind="bar", show_trend=True)
```

If you have multiple models, pass a dict to `y_pred` and set `compare_by="model"`
to overlay them on the same axes.

## Track Accuracy Over Time

[`plot_score_time_series`](/pages/api/generated/yohou.plotting.evaluation.plot_score_time_series/)
plots scorer values at each timestep:

```python
from yohou.plotting import plot_score_time_series

plot_score_time_series(scorer, y_truth, y_pred)
```

For panel data, set `facet_by="group"` to get one subplot per group. To apply
time weights, pass a callable or DataFrame via `time_weight`.

## Score by Forecast Vintage

[`plot_score_per_vintage`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_vintage/)
shows how score changes by forecast origin time (requires multi-vintage
predictions):

```python
from yohou.plotting import plot_score_per_vintage

plot_score_per_vintage(scorer, y_truth, y_pred, show_trend=True)
```

## Examine the Error Distribution

[`plot_score_distribution`](/pages/api/generated/yohou.plotting.evaluation.plot_score_distribution/)
shows the distribution of per-timestep scorer values:

```python
from yohou.plotting import plot_score_distribution

# Histogram (default), KDE, or both
plot_score_distribution(scorer, y_truth, y_pred, kind="histogram")
plot_score_distribution(scorer, y_truth, y_pred, kind="kde")
plot_score_distribution(scorer, y_truth, y_pred, kind="both")
```

To compare multiple models, pass a dict to `y_pred`. The `show_mean` flag (on
by default) adds a vertical line at the mean score.

## Score Heatmap Across Two Dimensions

[`plot_score_heatmap`](/pages/api/generated/yohou.plotting.evaluation.plot_score_heatmap/)
creates a 2D heatmap of scores across forecast step and vintage:

```python
from yohou.plotting import plot_score_heatmap

# Default: step on x-axis, vintage on y-axis
plot_score_heatmap(scorer, y_truth, y_pred)

# Swap axes
plot_score_heatmap(scorer, y_truth, y_pred, x_dim="vintage", y_dim="step")
```

This takes a single scorer and a single prediction DataFrame (not dicts).

## Break Down Scores by Panel Group

[`plot_group_scores`](/pages/api/generated/yohou.plotting.evaluation.plot_group_scores/)
shows per-group performance for panel data. Three `kind` options are available:

```python
from yohou.plotting import plot_group_scores

# Bar chart (default): one bar per group
plot_group_scores(scorer, y_truth, y_pred, kind="bar")

# Box plot: distribution of scores within each group
plot_group_scores(scorer, y_truth, y_pred, kind="box")

# Heatmap: group x model grid
plot_group_scores(
    scorer,
    y_truth,
    {"Model A": y_pred_a, "Model B": y_pred_b},
    kind="heatmap",
)
```

For box plots, `distribute_by` controls the variability dimension (`"time"`,
`"vintage"`, or `"step"`).

## Diagnose Residual Patterns

[`plot_residuals`](/pages/api/generated/yohou.plotting.evaluation.plot_residuals/)
provides residual diagnostics:

```python
from yohou.plotting import plot_residuals

plot_residuals(y_pred, y_truth)
```

For a single column, this produces a 4-panel diagnostic (residuals over time,
histogram, ACF, Q-Q plot). For multiple columns or panel data, it creates
faceted residuals-over-time subplots.

## Check Interval Calibration

If you are working with interval or class-probability forecasts, use
[`plot_calibration`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/)
to assess reliability:

```python
from yohou.plotting import plot_calibration

plot_calibration(y_pred, y_truth)
```

Points close to the diagonal indicate well-calibrated intervals.

## Combine Views for a Complete Assessment

A thorough evaluation typically uses several of these plots together:

1. `plot_score_summary` for the headline comparison
2. `plot_score_per_step` to check horizon degradation
3. `plot_score_time_series` to spot temporal patterns
4. `plot_group_scores` to verify no panel group is left behind
5. `plot_residuals` to diagnose systematic error patterns

If the summary chart picks a winner but the per-step or per-group views reveal
weaknesses, the model may need different tuning for those specific cases.

## See Also

- [Visualization](../explanation/visualization.md#evaluating-model-quality) for the conceptual overview
- [How to Evaluate Forecasts with Multi-vintage Scoring](multi-vintage-scoring.md) for generating and scoring multi-vintage predictions
- [API Reference: yohou.plotting](/pages/api/plotting/) for the full parameter listing
