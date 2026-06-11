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

**Fit-time weighting** shapes what the model learns. When you configure a
forecaster with a `time_weighter`, the underlying sklearn estimator receives a
`sample_weight` array during `fit`. Training samples with higher weight contribute
proportionally more to the loss function, so the model's parameters are pulled
toward fitting those periods well, at the expense of lower-weight periods. This
is a genuine model change: two forecasters trained on the same data with
different weights can produce substantially different predictions, not just
different evaluation scores.

The important subtlety for reduction forecasters is that training samples are
not individual time steps but rows of the tabularized feature matrix, each
spanning a window of the time series. The weighter's per-timestamp weights must
be collapsed into a per-sample weight. The `sample_weight_alignment` constructor
parameter controls this collapse (see [The Alignment Problem](#the-alignment-problem)
below).

**Score-time weighting** shapes how performance is summarized. When you configure
a scorer with a `time_weighter`, it changes the weighted average of per-timestep
errors that produces the final metric value. This is pure aggregation: the
model's predictions are unchanged, but the metric emphasizes errors that
correspond to high-weight periods.

Score-time weighting is the right tool when you want model selection to favor
configurations that perform well on the periods you care about, without
committing to those periods being literally more important during training.
Fit-time weighting goes further, making the model actually better on those
periods (at the cost of being worse on others).

## The Three Weighter Slots

yohou exposes three independent weighter slots, each holding a `BaseWeighter`
estimator and targeting a different axis of the evaluation or training data.

| Slot | Controls | Fit time | Score time | Axis |
|---|---|---|---|---|
| `time_weighter` | Recency or seasonal emphasis | Yes (via `sample_weight`) | Yes (weighted metric) | Individual timestamps |
| `vintage_weighter` | Forecast-origin emphasis | Yes (via `sample_weight`) | Yes (weighted metric) | Forecast origins |
| `step_weighter` | Horizon-step emphasis | No | Yes (weighted metric) | Forecast steps $1 \ldots h$ |

**`time_weighter`** is the most commonly used slot. It assigns importance to
individual timestamps. An
[`ExponentialDecayWeighter`](/pages/api/generated/yohou.weighting.weighters.ExponentialDecayWeighter/)
gives full credit to the most recent observations and geometrically less to
older ones, reflecting the assumption that recent dynamics are more
representative of the future. Its `half_life` parameter controls the rate of
decay: a short half-life aggressively de-emphasizes history, while a long one
produces nearly uniform weighting.

Beyond recency, the time slot also supports seasonal emphasis via
[`SeasonalEmphasisWeighter`](/pages/api/generated/yohou.weighting.weighters.SeasonalEmphasisWeighter/),
which up-weights timestamps at specific positions within the seasonal cycle. A
retailer preparing for year-end might weight December observations more heavily
to favor models that excel in the hardest-to-predict season. To express recency
*and* seasonality at once, combine the two with a
[`CompositeWeighter`](/pages/api/generated/yohou.weighting.weighters.CompositeWeighter/)
and assign it to `time_weighter`.

**`vintage_weighter`** shifts the focus from individual time steps to forecast
origins. A weight that emphasizes recent forecast origins expresses a belief
that the model's recent forecasting behavior predicts its future behavior better
than its performance from months ago. This is particularly relevant for models
deployed in rolling-refit regimes, where each vintage corresponds to a distinct
forecast origin date.

**`step_weighter`** focuses the score on specific forecast horizons. A step
weighter that gives full credit to step 1 and zero to steps 2 through 7
evaluates the model purely as a one-step-ahead predictor. A weighter that
emphasizes the final step tests whether accuracy holds across the full horizon.
The right emphasis depends on how forecasts are consumed: if downstream systems
only use the one-step-ahead value, optimize for that step. Because the step slot
only affects evaluation (not training), it is a score-time-only slot.

## Weighters Are Estimators

Every weighter is a scikit-learn estimator deriving from
[`BaseWeighter`](/pages/api/generated/yohou.weighting.weighters.BaseWeighter/): it
maps a key series (timestamps, steps, or vintage times) to a series of
non-negative weights through `compute_weights(key, group_name=None)`, declares
its tunable parameters in `_parameter_constraints`, and is configured on the host
estimator's `__init__`. Because the configuration *is* a constructor parameter,
its knobs are introspectable, clonable, and searchable (`time_weighter__half_life`).

The built-in weighters cover the common strategies:

| Weighter | Strategy |
|---|---|
| [`ExponentialDecayWeighter`](/pages/api/generated/yohou.weighting.weighters.ExponentialDecayWeighter/) | Geometric recency decay; `scale` selects an elapsed-time or rank-position basis |
| [`LinearDecayWeighter`](/pages/api/generated/yohou.weighting.weighters.LinearDecayWeighter/) | Linear ramp from oldest to newest, optionally zeroed beyond `max_steps` |
| [`SeasonalEmphasisWeighter`](/pages/api/generated/yohou.weighting.weighters.SeasonalEmphasisWeighter/) | Up-weights keys in phase with the latest seasonal position |
| [`LookupWeighter`](/pages/api/generated/yohou.weighting.weighters.LookupWeighter/) | Explicit per-key weights from a `dict`; absent keys get the tunable `default` |
| [`TableWeighter`](/pages/api/generated/yohou.weighting.weighters.TableWeighter/) | Weights resolved by joining the key series to a `pl.DataFrame` |
| [`CompositeWeighter`](/pages/api/generated/yohou.weighting.weighters.CompositeWeighter/) | Combines named sub-weighters by product or mean |

`LookupWeighter` and `TableWeighter` replace the former raw-`dict` and
raw-`pl.DataFrame` weight inputs, turning them into first-class tunable
estimators. The `default` parameter of `LookupWeighter` (the weight for keys
absent from the mapping) replaces the old `"*"` wildcard and is itself a
hyperparameter.

## The Alignment Problem

Reduction forecasters convert a time series into a supervised learning table
where each row (sample) spans a prediction window of `forecasting_horizon` time
steps. A per-timestamp weighter produces one weight per time step, but sklearn's
`sample_weight` needs one weight per row. The `sample_weight_alignment`
constructor parameter defines how that many-to-one collapse works.

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

Note that the `vintage_weighter` does not require alignment. Each training
sample has a single forecast origin, so the vintage weight maps directly to a
per-sample weight without collapsing.

## Composition and Normalization

When both a time and a vintage weighter are provided, their resolved arrays are
combined multiplicatively:

$$w_{\text{combined}}(t) = w_{\text{time}}(t) \times w_{\text{vintage}}(t)$$

After multiplication, the combined weights are renormalized so that their sum
equals the number of samples:

$$\hat{w} = w \cdot \frac{n}{\sum_i w_i}$$

This normalization preserves the effective learning rate of the underlying
sklearn estimator. Without it, a set of weights that averages to 0.5 would
halve the effective learning rate, changing the model's regularization behavior
in hard-to-predict ways.

[`CompositeWeighter`](/pages/api/generated/yohou.weighting.weighters.CompositeWeighter/)
performs an analogous combination at the *weighter* level (as opposed to combining
already-resolved arrays across slots). It holds named sub-weighters and combines
their outputs by element-wise product (the default) or weighted mean, producing
a single weighter that expresses all priorities simultaneously. This is how you
combine, for example, an exponential decay with a seasonal emphasis into one
`time_weighter`. Because the sub-weighters are named tuples exposed through
sklearn's `_BaseComposition`, each component's parameters remain addressable for
tuning (`time_weighter__decay__half_life`).

## Panel-Aware Weights

Weighters are panel-aware through their `compute_weights(key, group_name)`
signature. When yohou trains or evaluates panel data, it calls the weighter once
per group, passing that group's key series and its name, so each group is
weighted relative to its own most-recent key. The built-in weighters ignore
`group_name` (every group receives the same profile shape). To encode
group-specific *parameters* (for example, weighting recent data more
aggressively for volatile groups while using gentler decay for stable ones),
write a small `BaseWeighter` subclass whose `compute_weights` branches on
`group_name`.

`TableWeighter` supports panel awareness through group-specific columns. A
column named `"{group_name}_weight"` (e.g., `"store_a_weight"`) takes
precedence over a global `"weight"` column, letting you assign different weight
profiles per group within a single frame.

## Zero-Weight Filtering

When scorers encounter zero-weight observations, they pre-filter those rows
before computing the metric. This means zero-weighted periods are excluded
entirely from evaluation, not just scaled to zero. If all weights resolve to
zero, the scorer raises a `ValueError` rather than returning a degenerate result.
This behavior makes zero-weight useful as a hard mask: setting a weight to 0.0
removes that observation from scoring completely.

## Connections

For practical recipes on creating and applying weighters, see [How to Use Time Weighting](../how-to/time-weighting.md). Because weighters are constructor parameters, [Model Selection](model-selection.md) explains how their settings become ordinary tunable hyperparameters that [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) and [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/) clone and vary directly (e.g. `time_weighter__half_life`); no metadata routing is involved. The forecaster still converts the resolved weights into the sklearn `sample_weight` it forwards to the wrapped estimator; [Metadata Routing](metadata-routing.md) covers that routed-metadata machinery. [Forecast Accuracy](forecast-accuracy.md) discusses how stepwise and vintagewise aggregation relate to weighted scoring. The full API is documented in the [yohou.weighting reference](/pages/api/weighting/).
