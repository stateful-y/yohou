# Weighting

Yohou supports per-observation weighting in both training and scoring.
Weighting lets you express domain knowledge about which data points
matter most, for example giving recent observations more influence
during training or focusing evaluation on specific forecasting steps.

For practical recipes on creating and applying weights, see
[How to Use Time Weighting](/pages/how-to/time-weighting/).

**API Reference**: [`yohou.utils.weighting`](/pages/api/utils/#weighting)

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

All weight parameters accept the same three input formats:

**Dict**: a `{key: weight}` mapping. Missing keys default to `1.0`. The
wildcard `"*"` key changes the default, for example to zero out
unmentioned keys:

```python
step_weight = {1: 1.0, "*": 0.0}   # score only step 1
step_weight = {1: 2.0}              # double step 1, others at 1.0
```

**Callable**: a function `f(series) -> pl.Series` that receives the key
column as a Polars Series and returns a weight Series of the same
length. For panel data, a 2-parameter form `f(series, group_name)` is
also supported for per-group weighting.

**DataFrame**: a DataFrame joined on the key column, containing a
`"weight"` column (or `"{group}_weight"` for panel data).

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
