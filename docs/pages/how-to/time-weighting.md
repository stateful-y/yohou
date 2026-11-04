# How to Use Time Weighting

This guide shows you how to weight observations by recency or seasonal
relevance so that forecasters and scorers focus on the most informative parts
of your history.

## Prerequisites

- A fitted forecaster or scorer ([Getting Started](../tutorials/getting-started.md))
- For panel-aware weights: familiarity with panel data ([Work with Panel Data](panel-data.md))

<!-- COMPANION_NOTEBOOKS -->

## 1. Choose a Weighting Strategy

Time weighting is expressed with **weighter estimators**: small scikit-learn
estimators that map a key series (observation times, vintage times, or integer
forecasting steps) to a series of weights. Because they are estimators, their
parameters are introspectable, clonable, and tunable by search (covered in
[§7](#7-tune-your-weighting)).

If you want to down-weight old observations smoothly, use
[`ExponentialDecayWeighter`](/pages/api/generated/yohou.weighting.ExponentialDecayWeighter/).
It halves the weight every `half_life`, keeping the most recent observation at
1.0:

```python
from yohou.weighting import ExponentialDecayWeighter

weighter = ExponentialDecayWeighter(half_life=365)
```

`ExponentialDecayWeighter` has a `scale` parameter that sets the decay *basis*
(real elapsed time versus rank position), which matters when sampling is
irregular. When `scale=None` (the default) it is inferred from the key dtype:

| Key dtype             | Inferred `scale` | `half_life` units            | Decay basis                          |
| --------------------- | ---------------- | ---------------------------- | ------------------------------------ |
| datetime              | `"elapsed"`      | days (`int`/`float`) or `timedelta` | real elapsed time to the latest key |
| numeric / integer step| `"position"`     | steps (`int`/`float`)        | rank-index distance to the latest key |

Set `scale` explicitly to override the inference, e.g. `scale="position"` to
decay regularly-spaced datetimes by row position, or to weight integer
forecasting steps. A `timedelta` `half_life` with `scale="position"` raises
`ValueError`.

If you prefer a simple ramp from 0 (oldest) to 1 (newest), use
[`LinearDecayWeighter`](/pages/api/generated/yohou.weighting.LinearDecayWeighter/):

```python
from yohou.weighting import LinearDecayWeighter

weighter = LinearDecayWeighter()
```

To zero out observations older than a fixed window, pass `window_size`:

```python
weighter = LinearDecayWeighter(window_size=100)
```

If seasonality matters more than recency, use
[`SeasonalEmphasisWeighter`](/pages/api/generated/yohou.weighting.SeasonalEmphasisWeighter/)
to boost observations at the same seasonal position as the most recent one:

```python
from yohou.weighting import SeasonalEmphasisWeighter

# Emphasize same-month observations (monthly data with yearly cycle)
weighter = SeasonalEmphasisWeighter(seasonality=12, emphasis=2.0)
```

In-phase observations get the `emphasis` weight (default 2.0), all others get
1.0. For multiple seasonalities, pass a list:

```python
weighter = SeasonalEmphasisWeighter(seasonality=[7, 365], emphasis=1.5)
```

## 2. Use Explicit or Table-Driven Weights

When you want weights assigned by key rather than by a decay rule, use the
lookup and table weighters.

[`LookupWeighter`](/pages/api/generated/yohou.weighting.LookupWeighter/)
maps keys to weights via a `dict`. Keys absent from the mapping receive the
tunable `default` weight (this replaces the old `"*"` wildcard):

```python
from datetime import datetime
from yohou.weighting import LookupWeighter

weighter = LookupWeighter(
    mapping={datetime(2024, 6, 1): 2.0, datetime(2024, 7, 1): 2.0},
    default=0.5,  # weight for all other keys
)
```

[`TableWeighter`](/pages/api/generated/yohou.weighting.TableWeighter/)
resolves weights by joining the key series to a `pl.DataFrame` on a key column:

```python
import polars as pl
from yohou.weighting import TableWeighter

frame = pl.DataFrame({
    "time": y_train["time"],
    "weight": [1.0, 1.0, 0.5, 0.5, 0.0],
})
weighter = TableWeighter(frame=frame, on="time")
```

For panel data, give the frame group-specific columns (e.g. `"store_a_weight"`,
`"store_b_weight"`) or a single `"weight"` column applied to all groups. A key
with no matching row raises `ValueError`.

## 3. Compose Multiple Weights

To combine recency and seasonal effects, use
[`CompositeWeighter`](/pages/api/generated/yohou.weighting.CompositeWeighter/).
Its components are **named `(name, weighter)` tuples** (the same convention as
`FeaturePipeline` and the voting ensembles), which keeps every sub-weighter's
parameters addressable for tuning. By default it multiplies the component
weights element-wise:

```python
from yohou.weighting import CompositeWeighter

weighter = CompositeWeighter([
    ("decay", ExponentialDecayWeighter(half_life=365)),
    ("seasonal", SeasonalEmphasisWeighter(seasonality=12, emphasis=2.0)),
])
```

Pass `combination="mean"` to average the components instead of multiplying, and
`weights=[...]` to give per-component exponents (under `"multiply"`) or mixing
coefficients (under `"mean"`):

```python
weighter = CompositeWeighter(
    [("decay", ExponentialDecayWeighter(half_life=365)),
     ("seasonal", SeasonalEmphasisWeighter(seasonality=12))],
    combination="mean",
    weights=[2.0, 1.0],
)
```

## 4. Weight Part of the Calendar

To train or score on specific hours of each day, or specific days of each week,
use [`SegmentWeighter`](/pages/api/generated/yohou.weighting.SegmentWeighter/)
with a [`CalendarSegmenter`](/pages/api/generated/yohou.segmenting.CalendarSegmenter/).
Rows inside the segment get `inside` (default 1.0) and all other rows get
`outside` (default 0.0), so the defaults act as a calendar mask:

```python
from yohou.segmenting import CalendarSegmenter
from yohou.weighting import SegmentWeighter

business_hours = SegmentWeighter(CalendarSegmenter("hour", values=range(8, 19)))
```

A window across midnight is a plain list, such as `values=[22, 23, 0, 1]`. For
local hours on UTC timestamps, pass `time_zone="Europe/Paris"` to the
segmenter. To combine conditions, such as weekdays during business hours, wrap
two segmenters in an
[`IntersectionSegmenter`](/pages/api/generated/yohou.segmenting.IntersectionSegmenter/):

```python
from yohou.segmenting import IntersectionSegmenter

weekday_business_hours = SegmentWeighter(IntersectionSegmenter([
    ("days", CalendarSegmenter("day_of_week", values=[1, 2, 3, 4, 5])),
    ("hours", CalendarSegmenter("hour", values=range(8, 19))),
]))
```

For graded weights instead of a mask, multiply two segment weighters. Here peak
hours count three times as much as other business hours, and nights count a
little:

```python
weighter = CompositeWeighter([
    ("business", SegmentWeighter(CalendarSegmenter("hour", values=range(8, 19)), outside=0.2)),
    ("peak", SegmentWeighter(CalendarSegmenter("hour", values=[17, 18]), outside=1.0, inside=3.0)),
])
```

`SegmentWeighter` reads calendar time, so it works in the `time_weighter` and
`vintage_weighter` slots. A `step_weighter` receives integer forecasting steps
and raises a `ValueError`.

**In a reduction forecaster, use `sample_weight_alignment="matched_step"`.** A
segment weighter emphasizes specific target times, which is the case `"matched_step"`
is built for: each per-step model is weighted by its own target timestamps (see
[§5](#5-apply-weights-during-training)). It requires `reduction_strategy="direct"`
or `"dir-rec"`; under `"multi-output"` it warns and falls back to `"first_step"`.
If the segment is so narrow that some step has no weighted training row, fitting
raises a `ValueError` naming the step. With a collapse value such as `"first_step"`,
a training row whose first target falls outside the segment gets weight 0 and is
effectively dropped; under `"mean_step"` such a row is down-weighted by the share
of its targets outside the segment.

## 5. Apply Weights During Training

Weighting is a **constructor parameter** of the forecaster, not an argument to
`fit`. Pass a weighter to the `time_weighter` slot of a reduction forecaster:

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    time_weighter=ExponentialDecayWeighter(half_life=365),
)
forecaster.fit(y_train, forecasting_horizon=12)
```

The forecaster converts time weights to sklearn `sample_weight` internally.
Because each training sample spans multiple forecast steps, a sample carries one
weight per target timestamp. With the default `"multi-output"` strategy those
weights must be collapsed into one weight per sample, and the
`sample_weight_alignment` constructor parameter controls how:

```python
forecaster = PointReductionForecaster(
    estimator=Ridge(),
    time_weighter=ExponentialDecayWeighter(half_life=365),
    sample_weight_alignment="mean_step",
)
forecaster.fit(y_train, forecasting_horizon=12)
```

The default is `"first_step"`. A `vintage_weighter` slot is available for
per-vintage weighting. See [Weighting](../explanation/weighting.md) for a full
comparison of alignment strategies.

If your weighter emphasizes specific target times, such as a season, use a
per-step strategy with `sample_weight_alignment="matched_step"`. Each per-step
model is then weighted by its own target timestamps:

```python
from yohou.weighting import SeasonalEmphasisWeighter

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    reduction_strategy="direct",
    time_weighter=SeasonalEmphasisWeighter(seasonality=12, emphasis=3.0),
    sample_weight_alignment="matched_step",
)
forecaster.fit(y_train, forecasting_horizon=12)
```

With a collapse value such as `"first_step"`, every per-step model receives the
weight of the first target timestamp, so the model for step 12 emphasizes the
rows eleven steps before the marked season instead of the rows that predict it.
For an `ExponentialDecayWeighter` on equally spaced timestamps you will see no
difference, because each step's weights are a constant multiple of the first
step's. `"matched_step"` requires `reduction_strategy="direct"` or `"dir-rec"`;
with `"multi-output"` it falls back to `"first_step"` with a warning. If a step
ends up with no weighted row for some panel group, for example because a
`LookupWeighter` is non-zero only near the start of the series, fitting raises a
`ValueError` naming the step: that step has nothing to learn from, so widen the
weighter or use a collapse value.

## 6. Apply Weights During Scoring

Scorers carry their weighting on `__init__` too. Pass a weighter to
`time_weighter` to weight per-timestep errors:

```python
from yohou.metrics import MeanAbsoluteError

scorer = MeanAbsoluteError(time_weighter=ExponentialDecayWeighter(half_life=365))
scorer.fit(y_train)
weighted_score = scorer.score(y_test, y_pred)
```

Scorers also expose `step_weighter` and `vintage_weighter` for multi-vintage
predictions. A scorer only exposes the weighter slots it supports. For example,
`MedianAbsoluteError` has no `time_weighter` parameter, so an unsupported
weighter is rejected at construction rather than at `score` time. See
[Multi-vintage Scoring](multi-vintage-scoring.md) for details.

Because the weighting lives on the scorer instance, a weighted scorer is a valid
[`cross_validate`](/pages/api/generated/yohou.model_selection.cross_validate/)
objective with no per-call weight argument:

```python
from yohou.model_selection import cross_validate

cross_validate(forecaster, y_train, forecasting_horizon=12, scoring=scorer)
```

## 7. Tune Your Weighting

Because weighters are constructor parameters, their settings are searchable
hyperparameters addressed with the `__` syntax. Tune the decay half-life, and
even the decay basis, directly:

```python
from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import GridSearchCV, ExpandingWindowSplitter

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    time_weighter=ExponentialDecayWeighter(half_life=365),
)

search = GridSearchCV(
    forecaster,
    param_grid={
        "time_weighter__half_life": [90, 180, 365, 730],
        "time_weighter__scale": ["elapsed", "position"],
    },
    scoring=MeanAbsoluteError(),
    cv=ExpandingWindowSplitter(n_splits=5, test_size=12),
)
search.fit(y_train, forecasting_horizon=12)
```

`sample_weight_alignment="weighted_mean_step"` mixes a sample's per-step weights
with coefficients that decay by `sample_weight_horizon_decay` per step, and that
rate is a searchable parameter too. It changes the weights only for a weighter
whose profile is not geometric, so tune it with a seasonal or table weighter
rather than an exponential decay:

```python
forecaster = PointReductionForecaster(
    estimator=Ridge(),
    time_weighter=SeasonalEmphasisWeighter(seasonality=12, emphasis=3.0),
    sample_weight_alignment="weighted_mean_step",
)

search = GridSearchCV(
    forecaster,
    param_grid={
        "time_weighter__seasonality": [6, 12],
        "sample_weight_horizon_decay": [0.1, 0.5, 1.0],
    },
    scoring=MeanAbsoluteError(),
    cv=ExpandingWindowSplitter(n_splits=5, test_size=12),
)
search.fit(y_train, forecasting_horizon=12)
```

Components of a `CompositeWeighter` are reachable through their names
(`time_weighter__decay__half_life`), and you can search over whole weighter
instances by listing them as grid values:

```python
search = GridSearchCV(
    forecaster,
    param_grid={"time_weighter": [
        ExponentialDecayWeighter(half_life=180),
        LinearDecayWeighter(window_size=100),
    ]},
    scoring=MeanAbsoluteError(),
    cv=ExpandingWindowSplitter(n_splits=5, test_size=12),
)
search.fit(y_train, forecasting_horizon=12)
```

Weighters recompute on each fold's own key series, so recency is always relative
to that fold's most-recent key.

## 8. Customize Weights for Panel Data

Built-in weighters are panel-aware automatically: for panel data the forecaster
calls `compute_weights(key, group_name)` once per group, on that group's own key
series, so each group is weighted relative to its own most-recent key. The
built-ins ignore `group_name` (every group gets the same profile shape).

To give different groups different *parameters*, write a small `BaseWeighter`
subclass that dispatches on `group_name`:

```python
import polars as pl
from yohou.weighting import BaseWeighter, ExponentialDecayWeighter

class PerStoreDecay(BaseWeighter):
    def compute_weights(self, key: pl.Series, group_name: str | None = None) -> pl.Series:
        half_life = 180 if group_name == "store_a" else 365
        return ExponentialDecayWeighter(half_life=half_life).compute_weights(key, group_name)

forecaster = PointReductionForecaster(estimator=Ridge(), time_weighter=PerStoreDecay())
forecaster.fit(y_train, forecasting_horizon=12)
```

For global (non-panel) data, `group_name` is `None`.

## 9. Visualize the Weight Profile

[`plot_time_weight`](/pages/api/generated/yohou.plotting.plot_time_weight/)
shows weights over time. Call the weighter's `compute_weights` on the key series
to build the expected DataFrame:

```python
from yohou.plotting import plot_time_weight

weighter = ExponentialDecayWeighter(half_life=365)
weights = weighter.compute_weights(y_train["time"])
weights_df = y_train.select("time").with_columns(time_weight=weights)

plot_time_weight(weights_df)
```

If your weight column has a different name, pass `weight_column="my_col"`. To
disable the filled area under the curve, pass `fill=False`.

## See Also

- [Weighting](../explanation/weighting.md) for the conceptual overview of weighter types, alignment strategies, and normalization
- [Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md) for using scorers with and without weights
- [Model Selection](../explanation/model-selection.md) for tuning weighter parameters as hyperparameters
- [Handle Long Series](handle-long-series.md) for limiting history length as an alternative to down-weighting old data
- [Multi-vintage Scoring](multi-vintage-scoring.md) for `step_weighter` and `vintage_weighter` in context
- [API Reference: yohou.weighting](/pages/api/weighting/) for the full parameter listing
- [API Reference: yohou.segmenting](/pages/api/segmenting/) for the calendar segmenters `SegmentWeighter` takes
