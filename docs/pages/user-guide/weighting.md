# Weighting

Yohou supports per-observation weighting in both training and scoring.
Weighting lets you express domain knowledge about which data points
matter most, for example giving recent observations more influence
during training or focusing evaluation on specific forecasting steps.

**API Reference**: [`yohou.utils.weighting`](../api/utils.md#weighting)

## Weight Types

| Weight | Training (`fit`) | Scoring (`score`) | Key column |
|--------|:-:|:-:|------------|
| `time_weight` | Yes | Yes | `"time"` |
| `step_weight` | No | Yes | `"forecasting_step"` |
| `vintage_weight` | Yes | Yes | `"vintage_time"` |

`time_weight` controls how much each time point influences the fitted
model (training) and the aggregated score (scoring).

`step_weight` controls how much each forecasting step (1-step-ahead,
2-step-ahead, etc.) contributes to the aggregated score. Only available
in scoring.

`vintage_weight` controls how much each vintage (forecast origin date)
contributes. In training it controls per-observation emphasis; in
scoring it controls per-vintage score aggregation.

## Weight Formats

All weight parameters accept the same three input formats.

### Dict

A `{key: weight}` mapping. Missing keys default to `1.0`. Use the
wildcard `"*"` key to change the default, for example to zero out
unmentioned keys:

```python
# Score only step 1 (whitelist mode)
scorer.score(y_true, y_pred, step_weight={1: 1.0, "*": 0.0})

# Double the weight of step 1, keep others at 1.0
scorer.score(y_true, y_pred, step_weight={1: 2.0})
```

### Callable

A function `f(series) -> pl.Series` that receives the key column as a
Polars Series and returns a weight Series of the same length:

```python
from yohou.utils.weighting import exponential_decay_weight

# Built-in decay function
forecaster.fit(y, time_weight=exponential_decay_weight(half_life=30))

# Custom callable
forecaster.fit(y, time_weight=lambda t: pl.Series([1.0] * len(t)))
```

For panel data, a 2-parameter callable `f(series, group_name)` is also
supported. It receives the group name as a second argument for
per-group weighting:

```python
def panel_weight(time, group_name):
    base = exponential_decay_weight(half_life=30)(time)
    if group_name == "store_A":
        return base * 2.0
    return base

forecaster.fit(y, time_weight=panel_weight)
```

### DataFrame

A DataFrame joined on the key column. Looks for a `"weight"` column
(or `"{group}_weight"` for panel data):

```python
tw = pl.DataFrame({
    "time": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
    "weight": [1.0, 2.0],
})
forecaster.fit(y, time_weight=tw)
```

## Combination and Normalization

When multiple weight types are provided, they are combined
multiplicatively. All combined weights are normalized so their sum
equals the number of samples, preserving score and loss scale.

In **scorers**, `time_weight` is normalized and applied first, then
`step_weight` and `vintage_weight` are combined, normalized together,
and applied. This two-stage normalization ensures consistent behavior
regardless of which weights are provided.

In **forecasters**, `time_weight` (after alignment) and
`vintage_weight` (direct lookup) are combined multiplicatively into a
single `sample_weight` array before calling `estimator.fit()`.
`sample_weight_alignment` governs only `time_weight`; `vintage_weight`
maps 1:1 to samples with no alignment needed.

## Built-in Weight Functions

Yohou provides ready-made weight functions in `yohou.utils.weighting`:

| Function | Description |
|----------|-------------|
| `exponential_decay_weight(half_life)` | Exponential decay giving more weight to recent times |
| `linear_decay_weight(rate)` | Linear decay giving more weight to recent times |
| `seasonal_emphasis_weight(period, ...)` | Emphasizes specific seasonal positions |
| `compose_weights(*fns)` | Multiplies multiple weight functions together |

```python
from yohou.utils.weighting import (
    compose_weights,
    exponential_decay_weight,
    seasonal_emphasis_weight,
)

# Combine recency decay with seasonal emphasis
weight_fn = compose_weights(
    exponential_decay_weight(half_life=30),
    seasonal_emphasis_weight(period=7, positions=[0, 6]),  # weekends
)
forecaster.fit(y, time_weight=weight_fn)
```

## Zero-weight Pre-filtering

Rows with resolved weight `0.0` are pre-filtered before computation
rather than multiplied by zero. This prevents `0 * NaN` artifacts from
metrics like MAPE that can produce NaN for valid data (e.g., zero
denominator).

## Validation

All weight formats are validated after resolution:

- No NaN values
- No negative values
- No infinite values
- No all-zero arrays

Invalid weights raise `ValueError` with descriptive messages including
the affected indices.
