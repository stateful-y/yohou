# How to Use Time Weighting

This guide shows you how to weight observations by recency or seasonal
relevance so that forecasters and scorers focus on the most informative parts
of your history.

## Prerequisites

- A fitted forecaster or scorer ([Getting Started](../tutorials/getting-started.md))
- For panel-aware weights: familiarity with panel data ([Work with Panel Data](panel-data.md))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## 1. Choose a Weighting Strategy

The `time_weight` parameter accepts a callable, a `pl.DataFrame`, or a `dict`.
Built-in weight functions (callables) cover the most common patterns.

If you want to down-weight old observations smoothly, use
[`exponential_decay_weight`](/pages/api/generated/yohou.utils.weighting.exponential_decay_weight/).
It halves the weight every `half_life` days, keeping the most recent
observation at 1.0:

```python
from yohou.utils.weighting import exponential_decay_weight

weight_fn = exponential_decay_weight(half_life=365)
```

`half_life` accepts an `int`, `float` (both interpreted as days), or a
`timedelta` for explicit units.

If you prefer a simple ramp from 0 (oldest) to 1 (newest), use
[`linear_decay_weight`](/pages/api/generated/yohou.utils.weighting.linear_decay_weight/):

```python
from yohou.utils.weighting import linear_decay_weight

weight_fn = linear_decay_weight()
```

To zero out observations older than a fixed window, pass `max_steps`:

```python
weight_fn = linear_decay_weight(max_steps=100)
```

If seasonality matters more than recency, use
[`seasonal_emphasis_weight`](/pages/api/generated/yohou.utils.weighting.seasonal_emphasis_weight/)
to boost observations at the same seasonal position as the most recent one:

```python
from yohou.utils.weighting import seasonal_emphasis_weight

# Emphasize same-month observations (monthly data with yearly cycle)
weight_fn = seasonal_emphasis_weight(seasonality=12, emphasis=2.0)
```

In-phase observations get the `emphasis` weight (default 2.0), all others get
1.0. For multiple seasonalities, pass a list:

```python
weight_fn = seasonal_emphasis_weight(seasonality=[7, 365], emphasis=1.5)
```

If you already have pre-computed weights, pass a `pl.DataFrame` with `"time"`
and `"weight"` columns:

```python
import polars as pl

time_weight = pl.DataFrame({
    "time": y_train["time"],
    "weight": [1.0, 1.0, 0.5, 0.5, 0.0],
})
```

For panel data, use group-specific columns (e.g. `"store_a_weight"`,
`"store_b_weight"`) or a single `"weight"` column applied to all groups.

You can also pass a `dict` mapping timestamps to weights. Timestamps not in
the dict get a default weight of 1.0 (or use `"*"` to set a different default):

```python
from datetime import datetime

time_weight = {
    datetime(2024, 6, 1): 2.0,
    datetime(2024, 7, 1): 2.0,
    "*": 0.5,  # all other timestamps
}
```

## 2. Compose Multiple Weights

To combine recency and seasonal effects, use
[`compose_weights`](/pages/api/generated/yohou.utils.weighting.compose_weights/).
It multiplies the weight functions element-wise:

```python
from yohou.utils.weighting import compose_weights

weight_fn = compose_weights(
    exponential_decay_weight(half_life=365),
    seasonal_emphasis_weight(seasonality=12, emphasis=2.0),
)
```

## 3. Apply Weights During Training

Pass the weight function as `time_weight` when fitting a reduction forecaster:

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster

forecaster = PointReductionForecaster(estimator=Ridge())
forecaster.fit(
    y_train,
    forecasting_horizon=12,
    time_weight=weight_fn,
)
```

The forecaster converts time weights to sklearn `sample_weight` internally.
Because each training sample spans multiple forecast steps, the per-timestamp
weights must be collapsed into one weight per sample. The
`sample_weight_alignment` parameter controls how:

```python
forecaster.fit(
    y_train,
    forecasting_horizon=12,
    time_weight=weight_fn,
    sample_weight_alignment="mean_step",
)
```

The default is `"first_step"`. See [Weighting](../explanation/weighting.md) for
a full comparison of alignment strategies.

## 4. Apply Weights During Scoring

Scorers accept `time_weight` in `score()` to weight per-timestep errors:

```python
from yohou.metrics import MeanAbsoluteError

scorer = MeanAbsoluteError()
scorer.fit(y_train)
weighted_score = scorer.score(y_test, y_pred, time_weight=weight_fn)
```

Scorers also support `step_weight` and `vintage_weight` for multi-vintage
predictions. See [Multi-vintage Scoring](multi-vintage-scoring.md) for details.

## 5. Customize Weights for Panel Data

Weight functions can accept a second parameter with the group name, letting you
assign different weight profiles per panel:

```python
def panel_weight(time: pl.Series, group_name: str | None) -> pl.Series:
    if group_name == "store_a":
        return exponential_decay_weight(half_life=180)(time)
    return exponential_decay_weight(half_life=365)(time)

forecaster.fit(y_train, forecasting_horizon=12, time_weight=panel_weight)
```

The framework detects whether your callable takes one or two parameters
automatically. For global (non-panel) data, `group_name` is `None`.

## 6. Visualize the Weight Profile

[`plot_time_weight`](/pages/api/generated/yohou.plotting.forecasting.plot_time_weight/)
shows weights over time. Build the expected DataFrame and pass it in:

```python
from yohou.plotting import plot_time_weight

weights = weight_fn(y_train["time"])
weights_df = y_train.select("time").with_columns(time_weight=weights)

plot_time_weight(weights_df)
```

If your weight column has a different name, pass `weight_column="my_col"`. To
disable the filled area under the curve, pass `fill=False`.

## See Also

- [Weighting](../explanation/weighting.md) for the conceptual overview of weight types, alignment strategies, and normalization
- [Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md) for using scorers with and without weights
- [Handle Long Series](handle-long-series.md) for limiting history length as an alternative to down-weighting old data
- [Multi-vintage Scoring](multi-vintage-scoring.md) for `step_weight` and `vintage_weight` in context
- [API Reference: yohou.utils.weighting](/pages/api/utils/) for the full parameter listing
