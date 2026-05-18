# Weighting

A single cross-validation score treats every observation as equally informative
about future forecaster performance. This is rarely a good assumption. A
retailer cares more about forecast accuracy in the next quarter than in quarters
from two years ago, because consumer behavior shifts and the model that performs
well now is the one that matters. An energy forecaster cares more about
peak-demand hours than off-peak ones, because the cost of a miss is asymmetric.
Weighting encodes these priorities: it tells yohou which parts of the data
should count more when fitting a model and when evaluating one.

## Fit-Time and Score-Time Weighting

The same weight concept applies at two distinct points in the modeling workflow,
and they mean different things.

**Fit-time weighting** shapes what the model learns. When you pass `time_weight`
to a forecaster, the underlying sklearn estimator receives a `sample_weight`
array during `fit`. Training samples with higher weight contribute
proportionally more to the loss function, so the model's parameters are pulled
toward fitting those periods well, at the expense of lower-weight periods. This
is a genuine model change: two forecasters trained on the same data with
different weights can produce substantially different predictions, not just
different evaluation scores.

The important subtlety for reduction forecasters is that training samples are
not individual time steps but rows of the tabularized feature matrix, each
spanning a window of the time series. A `time_weight` defined per timestamp must
be collapsed into a per-sample weight. The `sample_weight_alignment` parameter
controls this collapse (see [The Alignment Problem](#the-alignment-problem)
below).

**Score-time weighting** shapes how performance is summarized. When you pass
`time_weight` to a scorer, it changes the weighted average of per-timestep
errors that produces the final metric value. This is pure aggregation: the
model's predictions are unchanged, but the metric emphasizes errors that
correspond to high-weight periods.

Score-time weighting is the right tool when you want model selection to favor
configurations that perform well on the periods you care about, without
committing to those periods being literally more important during training.
Fit-time weighting goes further, making the model actually better on those
periods (at the cost of being worse on others).

## The Three Weight Parameters

yohou exposes three independent weight parameters, each targeting a different
axis of the evaluation or training data.

| Parameter | Controls | Fit time | Score time | Axis |
|---|---|---|---|---|
| `time_weight` | Recency or seasonal emphasis | Yes (via `sample_weight`) | Yes (weighted metric) | Individual timestamps |
| `vintage_weight` | Forecast-origin emphasis | Yes (via `sample_weight`) | Yes (weighted metric) | Forecast origins |
| `step_weight` | Horizon-step emphasis | No | Yes (weighted metric) | Forecast steps $1 \ldots h$ |

**`time_weight`** is the most commonly used parameter. It assigns importance to
individual timestamps. An exponentially decaying weight gives full credit to
the most recent observations and geometrically less to older ones, reflecting
the assumption that recent dynamics are more representative of the future. The
`half_life` parameter controls the rate of decay: a short half-life aggressively
de-emphasizes history, while a long one produces nearly uniform weighting.

Beyond recency, `time_weight` also supports seasonal emphasis via the
[`seasonal_emphasis_weight`](/pages/api/generated/yohou.utils.weighting.seasonal_emphasis_weight/)
function, which up-weights timestamps at specific positions within the seasonal
cycle. A retailer preparing for year-end might weight December observations more
heavily to favor models that excel in the hardest-to-predict season. Seasonal
emphasis is not a separate parameter: it produces a callable that you pass as
`time_weight`, or combine with a recency function through
[`compose_weights`](/pages/api/generated/yohou.utils.weighting.compose_weights/).

**`vintage_weight`** shifts the focus from individual time steps to forecast
origins. A weight that emphasizes recent forecast origins expresses a belief
that the model's recent forecasting behavior predicts its future behavior better
than its performance from months ago. This is particularly relevant for models
deployed in rolling-refit regimes, where each vintage corresponds to a distinct
forecast origin date.

**`step_weight`** focuses the score on specific forecast horizons. A
`step_weight` that gives full credit to step 1 and zero to steps 2 through 7
evaluates the model purely as a one-step-ahead predictor. A weight that
emphasizes the final step tests whether accuracy holds across the full horizon.
The right emphasis depends on how forecasts are consumed: if downstream systems
only use the one-step-ahead value, optimize for that step. Because `step_weight`
only affects evaluation (not training), it is a score-time-only parameter.

## Weight Input Formats

All three weight parameters accept the same three input formats, giving
flexibility in how you specify weights.

**Callable.** A function that receives a `pl.Series` of keys (timestamps,
steps, or vintage times) and returns a `pl.Series` of weights. The built-in
functions
[`exponential_decay_weight`](/pages/api/generated/yohou.utils.weighting.exponential_decay_weight/),
[`linear_decay_weight`](/pages/api/generated/yohou.utils.weighting.linear_decay_weight/),
and [`seasonal_emphasis_weight`](/pages/api/generated/yohou.utils.weighting.seasonal_emphasis_weight/)
all return callables in this format. For panel data, a callable can optionally
accept a second `group_name` parameter to produce group-specific weights.

**`pl.DataFrame`.** A two-column frame with the join column (`"time"`,
`"forecasting_step"`, or `"vintage_time"`) and a `"weight"` column. For panel
data, group-specific columns (e.g., `"store_a_weight"`) take precedence over a
global `"weight"` column.

**`dict`.** A mapping from key values to weights. Timestamps not present in the
dict receive a default weight of 1.0. The special `"*"` key overrides the
default (e.g., `{"*": 0.0, 1: 2.0}` gives step 1 weight 2.0 and all others
weight 0.0).

## The Alignment Problem

Reduction forecasters convert a time series into a supervised learning table
where each row (sample) spans a prediction window of `forecasting_horizon` time
steps. A per-timestamp weight function produces one weight per time step, but
sklearn's `sample_weight` needs one weight per row. The
`sample_weight_alignment` parameter defines how that many-to-one collapse works.

Five strategies are available:

| Strategy | Collapse rule | Good when |
|---|---|---|
| `"first_step"` (default) | Weight of the first target timestamp in the window | You care most about the immediate next step |
| `"mean_step"` | Simple average across all target timestamps | All horizon steps are equally important |
| `"weighted_mean_step"` | Exponentially-decayed average (nearer steps weighted more) | Recent steps matter more, with smooth decay |
| `"max_weight_step"` | Maximum weight in the window | The model should focus on the most important step in each window |
| `"min_weight_step"` | Minimum weight in the window | Conservative: a window counts only if even its least-important step is weighted |

For `"weighted_mean_step"`, the decay within the window follows:

$$w_i = \exp(-0.5 \cdot i), \quad i = 0, 1, \ldots, h-1$$

where $i$ is the step index within the window. These internal weights are
normalized to sum to 1 before being applied to the per-timestamp weights.

The choice of strategy matters most when the weight function varies sharply
within a prediction window. With slowly varying weights (long half-life, wide
seasonal patterns), all strategies produce similar results. With rapidly changing
weights (short half-life, spike emphasis), the strategies can diverge enough to
change model selection outcomes.

**Example.** Consider a 14-day forecast window where the time weight drops
steeply. With `"first_step"`, the window's importance is determined by its
nearest target date, so the model focuses on getting day 1 right. With
`"mean_step"`, the importance is the average over all 14 target days, diluting
the emphasis on any single step.

Note that `vintage_weight` does not require alignment. Each training sample
has a single forecast origin, so the vintage weight maps directly to a
per-sample weight without collapsing.

## Composition and Normalization

When multiple weight parameters are provided, their resolved arrays are combined
multiplicatively:

$$w_{\text{combined}}(t) = w_{\text{time}}(t) \times w_{\text{vintage}}(t)$$

After multiplication, the combined weights are renormalized so that their sum
equals the number of samples:

$$\hat{w} = w \cdot \frac{n}{\sum_i w_i}$$

This normalization preserves the effective learning rate of the underlying
sklearn estimator. Without it, a set of weights that averages to 0.5 would
halve the effective learning rate, changing the model's regularization behavior
in hard-to-predict ways.

The [`compose_weights`](/pages/api/generated/yohou.utils.weighting.compose_weights/)
utility performs the same multiplicative combination for weight *functions*
(as opposed to weight arrays). It applies each function in sequence and
multiplies their outputs element-wise, producing a single callable that
expresses all priorities simultaneously. This is how you combine, for example,
an exponential decay with a seasonal emphasis into a single `time_weight`
callable.

## Panel-Aware Weights

Weight functions can produce different weights for each panel group. A callable
with a two-parameter signature `(time: pl.Series, group_name: str)` is
automatically detected as panel-aware. When yohou evaluates panel data, it calls
this function once per group, passing the group's time series and its name. This
lets you encode group-specific priorities: for example, weighting recent data
more aggressively for volatile groups while using gentler decay for stable ones.

DataFrame weights support panel awareness through group-specific columns. A
column named `"{group_name}_weight"` (e.g., `"store_a_weight"`) takes
precedence over a global `"weight"` column, letting you assign different weight
profiles per group within a single DataFrame.

## Zero-Weight Filtering

When scorers encounter zero-weight observations, they pre-filter those rows
before computing the metric. This means zero-weighted periods are excluded
entirely from evaluation, not just scaled to zero. If all weights resolve to
zero, the scorer raises a `ValueError` rather than returning a degenerate result.
This behavior makes zero-weight useful as a hard mask: setting a weight to 0.0
removes that observation from scoring completely.

## Connections

For practical recipes on creating and applying weights, see [How to Use Time Weighting](../how-to/time-weighting.md). [Model Selection](model-selection.md) explains how weights flow through [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) and [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/) via metadata routing. [Metadata Routing](metadata-routing.md) covers the `set_fit_request` and `set_score_request` API that controls which weight parameters each component receives. [Forecast Accuracy](forecast-accuracy.md) discusses how stepwise and vintagewise aggregation relate to weighted scoring. The full API is documented in the [yohou.utils.weighting reference](/pages/api/utils/#weighting).
