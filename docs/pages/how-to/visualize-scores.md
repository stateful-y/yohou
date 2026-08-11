# How to Visualize and Compare Model Scores

This guide shows you how to plot and compare evaluation metrics across models,
forecast steps, and time using yohou's evaluation plotting functions.

## Prerequisites

- `yohou[plotting]` installed (`pip install "yohou[plotting]"`)
- Predictions from one or more models ([Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md))

<!-- COMPANION_NOTEBOOKS -->

## Setup

The examples below use two models and two metrics. Replace these with your
own forecasters and scorers:

```python
from sklearn.linear_model import Ridge

from yohou.datasets import fetch_electricity_demand
from yohou.metrics import MeanAbsoluteError, RootMeanSquaredError
from yohou.model_selection import train_test_split
from yohou.point import PointReductionForecaster, SeasonalNaive

data = fetch_electricity_demand()
y = data.frame

y_train, y_test = train_test_split(y, test_size=48)

naive = SeasonalNaive(seasonality=1)
naive.fit(y_train, forecasting_horizon=24)
y_pred_naive = naive.predict()

ridge = PointReductionForecaster(estimator=Ridge())
ridge.fit(y_train, forecasting_horizon=24)
y_pred_ridge = ridge.predict()

scorer = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
for s in scorer.values():
    s.fit(y_train)

y_pred = {"Naive": y_pred_naive, "Ridge": y_pred_ridge}
```

## 1. Compare Models with a Summary Bar Chart

[`plot_score_summary`](/pages/api/generated/yohou.plotting.plot_score_summary/)
produces a grouped bar chart comparing aggregate scores across models and
metrics:

```python
from yohou.plotting import plot_score_summary

plot_score_summary(scorer, y_test, y_pred)
```

To sort bars by score value, pass `sort_ascending=True` (or `False` for
descending).

## 2. Check Horizon Degradation

[`plot_score_per_step`](/pages/api/generated/yohou.plotting.plot_score_per_step/)
shows how accuracy changes at each forecast horizon step:

```python
from yohou.plotting import plot_score_per_step

# Line chart (default)
plot_score_per_step(scorer, y_test, y_pred)

# Bar chart with a linear trend overlay
plot_score_per_step(scorer, y_test, y_pred, kind="bar", show_trend=True)
```

If you have multiple models, pass a dict to `y_pred` and set `compare_by="model"`
to overlay them on the same axes.

## 3. Track Accuracy Over Time

[`plot_score_time_series`](/pages/api/generated/yohou.plotting.plot_score_time_series/)
plots scorer values at each timestep:

```python
from yohou.plotting import plot_score_time_series

plot_score_time_series(scorer, y_test, y_pred)
```

For panel data, set `facet_by="group"` to get one subplot per group. To
emphasize recent errors, construct the scorer with an
[`ExponentialDecayWeighter`](/pages/api/generated/yohou.weighting.ExponentialDecayWeighter/)
before passing it to this function. See [Time Weighting](time-weighting.md).

## 4. Score by Forecast Vintage

[`plot_score_per_vintage`](/pages/api/generated/yohou.plotting.plot_score_per_vintage/)
shows how score changes by forecast origin time. This requires multi-vintage
predictions (see [Multi-vintage Scoring](multi-vintage-scoring.md)):

```python
from yohou.plotting import plot_score_per_vintage

plot_score_per_vintage(scorer, y_test, y_pred, show_trend=True)
```

## 5. Examine the Error Distribution

[`plot_score_distribution`](/pages/api/generated/yohou.plotting.plot_score_distribution/)
shows the distribution of per-timestep scorer values:

```python
from yohou.plotting import plot_score_distribution

plot_score_distribution(scorer, y_test, y_pred, kind="histogram")
```

The `kind` parameter accepts `"histogram"` (default), `"kde"`, or `"both"`.
To compare multiple models, pass a dict to `y_pred`. The `show_mean` flag (on
by default) adds a vertical line at the mean score.

## 6. Score Heatmap Across Two Dimensions

[`plot_score_heatmap`](/pages/api/generated/yohou.plotting.plot_score_heatmap/)
creates a 2D heatmap of scores across forecast step and vintage. Unlike the
other functions, this takes a single scorer and a single prediction DataFrame
(not dicts):

```python
from yohou.plotting import plot_score_heatmap

# Default: step on x-axis, vintage on y-axis
plot_score_heatmap(scorer["MAE"], y_test, y_pred_ridge)

# Swap axes
plot_score_heatmap(scorer["MAE"], y_test, y_pred_ridge, x_dim="vintage", y_dim="step")
```

## 7. Break Down Scores by Panel Group

[`plot_group_scores`](/pages/api/generated/yohou.plotting.plot_group_scores/)
shows per-group performance for panel data:

```python
from yohou.plotting import plot_group_scores

# Bar chart (default): one bar per group
plot_group_scores(scorer, y_test, y_pred, kind="bar")

# Box plot: distribution of scores within each group
plot_group_scores(scorer, y_test, y_pred, kind="box")

# Heatmap: group x model grid
plot_group_scores(scorer, y_test, y_pred, kind="heatmap")
```

The vintage views in sections 4 and 6 average the entity columns unless you ask
them not to, so a model that is accurate at most entities and wrong at one reads
as mildly mediocre. Pass `facet_by="group"` to see the entities separately:

```python
# One per-vintage subplot per entity, models overlaid inside each
plot_score_per_vintage(scorer["MAE"], y_test, {"ridge": y_pred_ridge}, facet_by="group")

# One step-by-vintage heatmap per entity, sharing a single colour scale
plot_score_heatmap(scorer["MAE"], y_test, y_pred_ridge, facet_by="group")

# Restrict a wide panel to the entities worth looking at
plot_score_heatmap(scorer["MAE"], y_test, y_pred_ridge, groups=["store_1"], facet_by="group")
```

Both views spend their subplot grid on the entities, so `plot_score_per_vintage`
raises when it is given several scorers and `facet_by` together. Score one metric
at a time when faceting.

For box plots, `distribute_by` controls the variability dimension (`"time"`,
`"vintage"`, or `"step"`).

## See Also

- [How to Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md) for cross-validation setup and metric selection
- [How to Visualize Forecasts](visualize-forecasts.md) for plotting predictions vs actuals, residual diagnostics, and calibration checks
- [Visualization](../explanation/visualization.md#evaluating-model-quality) for the conceptual overview
- [How to Evaluate Forecasts with Multi-vintage Scoring](multi-vintage-scoring.md) for generating and scoring multi-vintage predictions
- [API Reference: yohou.plotting](/pages/api/plotting/) for the full parameter listing
