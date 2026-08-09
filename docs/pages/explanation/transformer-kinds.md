# Transformer Kinds

Yohou transformers come in three kinds, and the kind is a property of the frame a transformer consumes and produces rather than of what it computes. An actual-kind transformer ([`BaseActualTransformer`](/pages/api/generated/yohou.base.BaseActualTransformer/)) operates on a single-axis frame: one `"time"` column and one row per timestamp, a single series marching forward. A forecast-kind transformer ([`BaseForecastTransformer`](/pages/api/generated/yohou.base.BaseForecastTransformer/)) operates on an `X_forecast` frame, which carries two time axes, `vintage_time` (when a forecast was issued) and `time` (what it forecasts). A step-kind transformer ([`BaseStepTransformer`](/pages/api/generated/yohou.base.BaseStepTransformer/)) operates on the derived step frame, the wide frame a forecaster builds internally in which each exogenous column becomes `{base}_step_1` through `{base}_step_H`, holding that column's values over the horizon ahead of each row.

Every transformer declares its kind through a `kind` tag whose value is `"actual"`, `"forecast"`, or `"step"`. Leaf transformers stamp the tag statically through their base class; the composition classes derive theirs from their children. Almost everything in Yohou is actual-kind, and that is the default a transformer gets if it says nothing.

Each kind has its own forecaster slot, and the tag is what keeps them apart: `actual_transformer` takes actual-kind, `forecast_transformer` takes forecast-kind, `step_transformer` takes step-kind. Putting a transformer in the wrong slot raises at fit rather than being quietly accepted.

## Why Two Kinds

The two axes make an `X_forecast` frame a different object, not merely a wider one. A single-axis frame is one series. An `X_forecast` frame is a stack of short series, one per vintage, each covering its own forecast horizon and each overlapping the others in `time`. The 6:00 AM weather forecast and the 9:30 AM forecast both say something about noon, so `time` alone does not identify a row.

That difference is why a single-axis transformer cannot simply be pointed at an `X_forecast` frame. Consider a transformer that computes a first difference. On one series, "the previous row" is unambiguous and means one timestep back. On a vintage stack, the row above may belong to a different vintage entirely, so the difference would silently subtract one forecast from a neighbouring one and produce a number that corresponds to nothing. The frame would be accepted and the output would be wrong, which is the worst available outcome. The kind tag exists so that the mismatch is caught rather than computed.

## Why the Step Frame Is a Third Kind

The step frame is the odd one out, because unlike the `X_forecast` frame it is not a different *shape*. It carries a single `"time"` index, exactly as a single-axis frame does, and one row per observation. Structurally, an actual-kind transformer could consume it without complaint.

That is precisely why it needs its own kind. What separates a step frame from an ordinary feature frame is not its index but what its columns mean: `temp_step_1` through `temp_step_48` are one variable seen at 48 points along the horizon, not 48 unrelated variables. A transformer written to summarise that block looks for the `_step_h` pattern. Point it at an `X_actual` frame and it finds nothing to summarise, so it emits nothing and the model silently loses features nobody removed. An error message is recoverable; a feature that quietly is not there is the kind of problem found months later in a model that underperforms for no visible reason. The kind tag turns the second into the first.

The step frame is also the only place in the pipeline where "the H values ahead of this observation" exist as one aligned row, which is what makes transformations along the horizon expressible at all. A forecast-kind transformer cannot substitute, because it sees one vintage at a time and is anchored to `vintage_time`, whereas the quantity a modeller usually wants ("the minimum temperature over the next 48 hours, as of now") is anchored to the observation time. Those coincide only when a fresh vintage is issued at every observation. And `X_future`, the deterministic known-future channel, never passes through a forecast transformer at all.

Step transformers are stateless in the `observe`/`rewind` sense, and unavoidably so: the step frame is rebuilt from scratch at every observe and every predict, so there is no buffer for memory to accumulate in. Leaf step transformers therefore do not define the memory API, and a step-kind composition raises if it is called.

[How to Reduce Forecast Step Features](../how-to/reduce-step-features.md) covers what to do with the kind in practice.

## Kind and Statefulness Are Orthogonal

Statefulness is Yohou's other transformer axis: a stateful transformer keeps a bounded buffer of recent rows and declares how many it needs through its `observation_horizon`, while a stateless transformer's output depends only on its fitted parameters and the current input. That axis is independent of kind, and all four combinations exist:

|              | Stateless                          | Stateful                                    |
| ------------ | ---------------------------------- | ------------------------------------------- |
| **Actual**   | scaling, log transforms, calendar features | lags, rolling statistics, filters   |
| **Forecast** | lifted stateless transformers      | lifted lags and differences, within a vintage |
| **Step**     | every step transformer             | none; the frame is rebuilt each call        |

The forecast-and-stateful cell rewards a moment's care, because it is easy to talk yourself out of it. State means memory: a buffer of the rows immediately preceding the current one, which is meaningful only where "preceding" is well defined and the history is contiguous. It is tempting to conclude that the vintage axis offers neither, since vintages are separate short series and the rows before a given vintage's first row belong to a different vintage.

That is true of *cross-vintage* memory and false of everything else. A vintage is internally contiguous: its rows are the forecast steps at the series interval, one after another. Within a vintage, "the previous row" is exactly as well defined as it is in any single series. So a lag or a difference computed inside one vintage is ordinary, and only a lag that reaches *across* a boundary is meaningless.

[`PerVintageActualTransformer`](/pages/api/generated/yohou.compose.PerVintageActualTransformer/) therefore accepts stateful inner transformers. It fits a fresh clone on each vintage's own rows, so a wrapped lag can only ever see that vintage's history, and the cross-vintage reach that would be meaningless is structurally impossible rather than merely discouraged.

The distinction to hold onto is between two senses of "stateful". A stateful *inner* keeps a buffer while computing one vintage. A stateful *estimator*, in the sense the `observe`/`rewind` API means, carries a buffer **between calls**. Forecast-kind transformers are stateless in the second sense and therefore have no `observe`/`rewind`: `PerVintageActualTransformer` refits every vintage on every `transform`, so nothing survives the call, whatever the inner does inside it. Serving a single fresh vintage is just a one-group input, no different in kind from a frame holding a year of them.

The cost of a stateful inner is rows: it consumes its `observation_horizon` from the **start** of every vintage, which are the nearest-term forecast steps. That is the same trade a lag makes on any series, and [How to Transform Features on the Forecast Channel](../how-to/transform-forecast-features.md) covers what it means in practice.

## Lifting Rather Than Reimplementing

Yohou ships one concrete forecast-kind transformer, [`PerVintageActualTransformer`](/pages/api/generated/yohou.compose.PerVintageActualTransformer/), against a couple of dozen actual-kind ones. That ratio is a design position rather than a gap in coverage.

The alternative would have been a parallel catalog: a vintage-aware scaler, a vintage-aware function transformer, a vintage-aware imputer, each duplicating the logic of its single-axis twin and each able to drift from it. Instead, `PerVintageActualTransformer` wraps an actual transformer and applies it to each vintage independently, grouping by `vintage_time`, fitting a fresh clone on each group's single-axis slice, transforming it, and restacking the results. Because every vintage is transformed using only its own rows, the order-dependent operations that would otherwise bleed across vintage boundaries stay contained. One wrapper lifts the whole catalog, and the wrapped transformers remain the same objects that run on `X_actual`.

Lifting is therefore the normal way to transform an `X_forecast` frame, not a stopgap. `BaseForecastTransformer` is the extension point beneath it: subclass it when an operation is genuinely about the vintage structure itself and so cannot be expressed as a per-vintage application of a single-axis transformer. Comparing each vintage against the one before it, for example, is inherently cross-vintage and has no single-axis equivalent to lift. Everything expressible per vintage should be lifted instead.

## Kind in Composition

The composition classes are polymorphic in kind rather than fixed to one. A [`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/) of forecast-kind branches is itself forecast-kind, and one of actual-kind branches is actual-kind. A composition must be homogeneous: mixing the two in a single container is rejected, because the container would have to align a one-axis frame against a two-axis one and there is no sensible answer. [Feature Pipelines](feature-pipelines.md) covers how composition derives its kind, aligns its branches, and reports the mismatch.

Kind also determines where a transformer may be attached, because the forecaster's slots are named for the kinds they take. `target_transformer` and `actual_transformer` process single-axis data, so they accept actual-kind transformers only. `forecast_transformer` takes the vintage-indexed `X_forecast` frame, so it accepts forecast-kind transformers only, and applies them before the step columns are derived.

The constructor surface therefore reads as the taxonomy does: one slot per kind, plus `target_transformer`, which shares the actual kind with `actual_transformer` and is distinguished from it by role instead. Putting a transformer in the wrong slot raises a `ValueError` naming the slot, and the message points at the one that would take it. The check reads the kind tag rather than the base class, because a composition declares its kind by tag: a `FeatureUnion` of forecast transformers is structurally a `BaseActualTransformer` while reporting `kind="forecast"`, and it belongs in `forecast_transformer` all the same.

## Connections

- [Preprocessing](preprocessing.md) covers actual-kind transformers in depth, including the observation horizon and the stateful lifecycle that only that kind has.
- [Feature Pipelines](feature-pipelines.md) covers composition, including how a container derives its kind from its children and why homogeneity is required.
- [Exogenous Features](exogenous-features.md) explains the three exogenous channels and what makes `X_forecast` vintage-indexed in the first place.
- [Core Concepts](core-concepts.md) places both transformer bases in the wider estimator hierarchy.
- [Extending Yohou](extending-yohou.md) discusses when subclassing a base is warranted over composing what exists.
