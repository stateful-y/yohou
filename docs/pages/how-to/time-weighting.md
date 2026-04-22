# How to Use Time Weighting

Time weighting lets you emphasize recent or seasonally relevant observations
during training and scoring. Weight functions are callables that take a polars
`Series` of datetimes and return a `Series` of weights.

## Prerequisites

- A fitted forecaster or scorer ([First Forecast](../tutorials/first-forecast.md))
- For panel-aware weights: familiarity with panel data ([Work with Panel Data](panel-data.md))

## Exponential Decay

[`exponential_decay_weight`](/pages/api/generated/yohou.utils.weighting.exponential_decay_weight/)
halves the weight every `half_life` days. The most recent observation always
gets weight 1.0:

```python
from yohou.utils.weighting import exponential_decay_weight

weight_fn = exponential_decay_weight(half_life=365)
```

`half_life` accepts an `int`, `float` (both interpreted as days), or a
`timedelta` for explicit units.

## Linear Decay

[`linear_decay_weight`](/pages/api/generated/yohou.utils.weighting.linear_decay_weight/)
scales weights linearly from 0 (oldest) to 1 (newest):

```python
from yohou.utils.weighting import linear_decay_weight

weight_fn = linear_decay_weight()
```

To zero out observations older than a fixed window, pass `max_steps`:

```python
weight_fn = linear_decay_weight(max_steps=100)
```

## Seasonal Emphasis

[`seasonal_emphasis_weight`](/pages/api/generated/yohou.utils.weighting.seasonal_emphasis_weight/)
boosts observations at the same seasonal position as the most recent
observation:

```python
from yohou.utils.weighting import seasonal_emphasis_weight

# Emphasize same-month observations (monthly data with yearly cycle)
weight_fn = seasonal_emphasis_weight(seasonality=12, emphasis=2.0)
```

In-phase observations get the `emphasis` weight (default 2.0), all others get
1.0. For multiple seasonalities, pass a list (combined with OR logic):

```python
weight_fn = seasonal_emphasis_weight(seasonality=[7, 365], emphasis=1.5)
```

## Compose Multiple Weights

[`compose_weights`](/pages/api/generated/yohou.utils.weighting.compose_weights/)
multiplies multiple weight functions element-wise:

```python
from yohou.utils.weighting import compose_weights

weight_fn = compose_weights(
    exponential_decay_weight(half_life=365),
    seasonal_emphasis_weight(seasonality=12, emphasis=2.0),
)
```

## Pass Weights to a Forecaster

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

The forecaster converts time weights to sklearn `sample_weight` internally. The
`sample_weight_alignment` parameter on `fit()` controls how multi-step weights
are reduced to a single weight per sample:

```python
forecaster.fit(
    y_train,
    forecasting_horizon=12,
    time_weight=weight_fn,
    sample_weight_alignment="mean_step",
)
```

Valid alignment strategies:

| Value | Behavior |
|---|---|
| `"first_step"` (default) | Weight at the first forecast step |
| `"mean_step"` | Average weight across forecast horizon |
| `"weighted_mean_step"` | Exponentially weighted mean (near-term emphasized) |
| `"max_weight_step"` | Maximum weight across horizon |
| `"min_weight_step"` | Minimum weight across horizon |

You can also pass a pre-computed `pl.DataFrame` instead of a callable. The
DataFrame must have a `"time"` column and a `"time_weight"` column.

## Pass Weights to a Scorer

Scorers accept `time_weight` in the `score()` method to weight per-timestep
errors:

```python
from yohou.metrics import MeanAbsoluteError

scorer = MeanAbsoluteError()
scorer.fit(y_train)
weighted_score = scorer.score(y_test, y_pred, time_weight=weight_fn)
```

For multi-vintage predictions, scorers also accept `step_weight` and
`vintage_weight` to weight by horizon position or forecast origin:

```python
# Weight horizon steps (e.g., prioritize near-term accuracy)
weighted_score = scorer.score(
    y_test, y_pred,
    step_weight={1: 1.0, 2: 0.8, 3: 0.5},
)

# Weight specific vintages
from datetime import datetime
weighted_score = scorer.score(
    y_test, y_pred,
    vintage_weight={datetime(2024, 1, 1): 2.0, datetime(2024, 6, 1): 1.0},
)
```

## Panel-Aware Weights

Weight functions can accept a second parameter with the group name for
panel-specific weighting:

```python
def panel_weight(time: pl.Series, group_name: str) -> pl.Series:
    if group_name == "store_a":
        return exponential_decay_weight(half_life=180)(time)
    return exponential_decay_weight(half_life=365)(time)

forecaster.fit(y_train, forecasting_horizon=12, time_weight=panel_weight)
```

The framework detects whether your callable takes one or two parameters
automatically. For global (non-panel) data, `group_name` is `None`.

## Visualize Weights

[`plot_time_weight`](/pages/api/generated/yohou.plotting.forecasting.plot_time_weight/)
shows the weight profile over a time series. Pass a DataFrame with
pre-computed weights:

```python
from yohou.plotting import plot_time_weight

weights = weight_fn(y_train["time"])
weights_df = y_train.select("time").with_columns(time_weight=weights)

plot_time_weight(weights_df)
```

If your weight column has a different name, pass `weight_column="my_col"`. To
disable the filled area under the curve, pass `fill=False`.

## See Also

- [Weighting](/pages/explanation/weighting/) for the conceptual overview of weight types, formats, and normalization
- [Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md) for using scorers with and without weights
- [How to Evaluate Forecasts with Multi-vintage Scoring](multi-vintage-scoring.md) for `step_weight` and `vintage_weight` in context
- [API Reference: yohou.utils.weighting](/pages/api/utils/) for the full parameter listing
