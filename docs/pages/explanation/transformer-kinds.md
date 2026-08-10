# Transformer Kinds

Yohou transformers come in three kinds, and the kind is a property of the frame a transformer consumes and produces rather than of what it computes.

| Kind | Base class | Frame | Index |
| --- | --- | --- | --- |
| actual | [`BaseActualTransformer`](/pages/api/generated/yohou.base.BaseActualTransformer/) | a single series marching forward, one row per timestamp | `time` |
| forecast | [`BaseForecastTransformer`](/pages/api/generated/yohou.base.BaseForecastTransformer/) | an `X_forecast` frame, a stack of short per-vintage series | `vintage_time`, `time` |
| step | [`BaseStepTransformer`](/pages/api/generated/yohou.base.BaseStepTransformer/) | the derived step frame a forecaster builds internally, where each exogenous column becomes `{base}_step_1` through `{base}_step_H` | `time` |

Every transformer declares its kind through a `kind` tag whose value is `"actual"`, `"forecast"`, or `"step"`. Leaf transformers stamp the tag statically through their base class; the composition classes derive theirs from their children. Almost everything in Yohou is actual-kind, and that is the default a transformer gets if it says nothing.

The two non-default kinds exist for different reasons, and holding the difference in mind makes the rest of this page easier to follow. The forecast kind is a **structural** distinction: its frame carries an index axis a single-axis transformer cannot read. The step kind is a **semantic** one: its frame is indexed exactly like an actual frame, and what sets it apart is what its columns mean. The next two sections take them in turn.

## Why the Vintage Axis Needs Its Own Kind

Carrying both `vintage_time` and `time` makes an `X_forecast` frame a different object, not merely a wider one. A single-axis frame is one series. An `X_forecast` frame is a stack of short series, one per vintage, each covering its own forecast horizon and each overlapping the others in `time`. The 6:00 AM weather forecast and the 9:30 AM forecast both say something about noon, so `time` alone does not identify a row.

That difference is why a single-axis transformer cannot simply be pointed at an `X_forecast` frame. Consider a transformer that computes a first difference. On one series, "the previous row" is unambiguous and means one timestep back. On a vintage stack, the row above may belong to a different vintage entirely, so the difference would silently subtract one forecast from a neighbouring one and produce a number that corresponds to nothing. The frame would be accepted and the output would be wrong, which is the worst available outcome. The kind tag exists so that the mismatch is caught rather than computed.

## Why the Step Frame Needs a Kind of Its Own

The step frame is the odd one out, because unlike the `X_forecast` frame it is not a different shape. It carries a single `"time"` index, exactly as a single-axis frame does, and one row per observation. Structurally, an actual-kind transformer could consume it without complaint.

That is precisely why it needs its own kind. What separates a step frame from an ordinary feature frame is not its index but what its columns mean. The columns `temp_step_1` through `temp_step_48` are one variable seen at 48 points along the horizon, not 48 unrelated variables. A transformer written to reduce that block looks for the `_step_h` pattern; point it at an `X_actual` frame and it finds nothing to reduce, so it emits nothing and the model quietly loses features nobody removed. An error message is recoverable. A feature that is silently absent is the kind of problem found months later, in a model that underperforms for no visible reason. The kind tag turns the second into the first.

The step frame is also the only place in the pipeline where "the H values ahead of this observation" exist as one aligned row, which is what makes a transformation along the horizon expressible at all. A forecast-kind transformer cannot substitute. It sees one vintage at a time and is anchored to `vintage_time`, whereas the quantity a modeller usually wants ("the minimum temperature over the next 48 hours, as of now") is anchored to the observation time. Those coincide only when a fresh vintage is issued at every observation. And `X_future`, the deterministic known-future channel, never passes through a forecast transformer at all, so for that channel the step frame is the only seam there is.

[How to Reduce Forecast Step Features](../how-to/reduce-step-features.md) covers what to do with the kind in practice.

## Kind and Statefulness Are Orthogonal

Statefulness is Yohou's other transformer axis: a stateful transformer keeps a bounded buffer of recent rows and declares how many it needs through its `observation_horizon`, while a stateless transformer's output depends only on its fitted parameters and the current input. The two axes are independent, though not every cell is occupied:

|              | Stateless                                  | Stateful                                      |
| ------------ | ------------------------------------------ | --------------------------------------------- |
| **Actual**   | scaling, log transforms, calendar features | lags, rolling statistics, filters             |
| **Forecast** | lifted stateless transformers              | lifted lags and differences, within a vintage |
| **Step**     | every step transformer                     | no such thing, see below                      |

The forecast-and-stateful cell rewards a moment's care, because it is easy to talk yourself out of it. State means memory: a buffer of the rows immediately preceding the current one, which is meaningful only where "preceding" is well defined and the history is contiguous. It is tempting to conclude that the vintage axis offers neither, since vintages are separate short series and the rows before a given vintage's first row belong to a different vintage.

That is true of *cross-vintage* memory and false of everything else. A vintage is internally contiguous: its rows are the forecast steps at the series interval, one after another. Within a vintage, "the previous row" is exactly as well defined as it is in any single series. So a lag or a difference computed inside one vintage is ordinary, and only a lag that reaches *across* a boundary is meaningless.

[`PerVintageActualTransformer`](/pages/api/generated/yohou.compose.PerVintageActualTransformer/) therefore accepts stateful inner transformers. It fits a fresh clone on each vintage's own rows, so a wrapped lag can only ever see that vintage's history, and the cross-vintage reach that would be meaningless is structurally impossible rather than merely discouraged.

The step-and-stateful cell is empty for a harder reason, and the emptiness is a property of the pipeline rather than a decision. A forecaster rebuilds the step frame from scratch at every `observe` and every `predict`, deriving it afresh from `X_future` and `X_forecast`. Nothing carries over between those calls, so there is no buffer for memory to live in. Step transformers are stateless because there is nothing available to be stateful about: leaf step transformers do not define the memory API at all, and a step-kind composition raises if one of its methods is called.

The distinction to hold onto is between two senses of "stateful". A stateful *inner* keeps a buffer while computing one vintage. A stateful *estimator*, in the sense the `observe`/`rewind` API means, carries a buffer **between calls**. Neither forecast-kind nor step-kind transformers are stateful in the second sense, so neither has `observe`/`rewind`. `PerVintageActualTransformer` refits every vintage on every `transform`, so nothing survives the call, whatever the inner does inside it. Serving a single fresh vintage is just a one-group input, no different in kind from a frame holding a year of them.

The cost of a stateful inner is rows: it consumes its `observation_horizon` from the **start** of every vintage, which are the nearest-term forecast steps. That is the same trade a lag makes on any series, and [How to Transform Features on the Forecast Channel](../how-to/transform-forecast-features.md) covers what it means in practice.

## Lifting Rather Than Reimplementing

Yohou ships one concrete forecast-kind transformer and three step-kind ones, against a couple of dozen actual-kind ones. That ratio is a design position rather than a gap in coverage.

The alternative would have been a parallel catalog per kind: a vintage-aware scaler, a vintage-aware function transformer, a vintage-aware imputer, then the same again for the step axis, each duplicating the logic of its single-axis twin and each able to drift from it. Instead each non-default kind is served by a wrapper that lifts an existing transformer onto its axis, and the wrapped objects stay exactly what they were.

What each kind lifts *from* differs, and the difference follows from the frame:

- [`PerVintageActualTransformer`](/pages/api/generated/yohou.compose.PerVintageActualTransformer/) lifts a Yohou **actual transformer** onto the vintage axis. It groups by `vintage_time`, fits a fresh clone on each group's single-axis slice, transforms it, and restacks. Each vintage is a legitimate single-axis frame, so the whole actual catalog applies to it unchanged, and the order-dependent operations that would otherwise bleed across vintage boundaries stay contained.
- [`StepColumnReducer`](/pages/api/generated/yohou.preprocessing.StepColumnReducer/) and [`StepFrameReducer`](/pages/api/generated/yohou.preprocessing.StepFrameReducer/) lift an **sklearn transformer** onto the horizon axis. A step block is not a time series at all once it is isolated: it is a plain numeric matrix, one row per observation and one column per horizon step, which is exactly the shape sklearn consumes. So the catalog worth lifting there is scikit-learn's, and a scaler, a projection, or a `FunctionTransformer` all apply without Yohou reimplementing any of them.

Lifting is therefore the normal way to work on either non-default kind, not a stopgap. The base classes are the extension point beneath it: subclass one when an operation is genuinely about the structure of that frame and so cannot be expressed as an application of an existing transformer. Comparing each vintage against the one before it is inherently cross-vintage and has no single-axis equivalent to lift, so it belongs under `BaseForecastTransformer`. `StepAggregator` is the step-side counterpart, reducing a block by a rule rather than by a fitted estimator, which is why it subclasses the base directly instead of wrapping anything. Everything else should be lifted.

## Kind in Composition

The composition classes are polymorphic in kind rather than fixed to one. A [`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/) of forecast-kind branches is itself forecast-kind, one of step-kind branches is step-kind, and one of actual-kind branches is actual-kind. A composition must be homogeneous: mixing kinds in a single container is rejected with a `ValueError` naming the members of each kind present.

For actual and forecast the restriction is structural, since the container would have to align a one-axis frame against a two-axis one and there is no sensible answer. For step it is semantic instead. A step frame and an actual frame align on `"time"` identically, so a mixed container would combine them without error and produce a frame in which half the columns mean something the consuming transformer does not expect. Homogeneity is enforced the same way whichever kinds are mixed, by tag, which is what makes the rule uniform even though the underlying reasons differ. [Feature Pipelines](feature-pipelines.md) covers how composition derives its kind, aligns its branches, and reports the mismatch.

Kind also determines where a transformer may be attached, because the forecaster's slots are named for the kinds they take:

| Slot | Consumes | Kind |
| --- | --- | --- |
| `target_transformer` | the target series | actual |
| `actual_transformer` | the feature frame built from `y` and `X_actual` | actual |
| `forecast_transformer` | `X_forecast`, before step columns are derived | forecast |
| `step_transformer` | the derived step frame, before it joins the design matrix | step |

The constructor surface therefore reads as the taxonomy does: one slot per kind, plus `target_transformer`, which shares the actual kind with `actual_transformer` and is distinguished from it by role instead. Putting a transformer in the wrong slot raises a `ValueError` naming the slot, and the message points at the one that would take it. The check reads the kind tag rather than the base class, because a composition declares its kind by tag: a `FeatureUnion` of forecast transformers is structurally a `BaseActualTransformer` while reporting `kind="forecast"`, and it belongs in `forecast_transformer` all the same.

The order of the last two slots is not arbitrary. `forecast_transformer` runs on the vintage-indexed frame, before step columns are derived from it; `step_transformer` runs on the result of that derivation. A transformation can therefore be expressed on whichever side of the pivot suits it, and the two are not interchangeable, because only the second sees the horizon laid out against the observation time.

## Connections

- [Preprocessing](preprocessing.md) covers actual-kind transformers in depth, including the observation horizon and the stateful lifecycle that only that kind has.
- [Feature Pipelines](feature-pipelines.md) covers composition, including how a container derives its kind from its children and why homogeneity is required.
- [Exogenous Features](exogenous-features.md) explains the three exogenous channels and what makes `X_forecast` vintage-indexed in the first place.
- [How to Transform Features on the Forecast Channel](../how-to/transform-forecast-features.md) and [How to Reduce Forecast Step Features](../how-to/reduce-step-features.md) are the practical guides for the two non-default kinds.
- [Core Concepts](core-concepts.md) places the transformer bases in the wider estimator hierarchy.
- [Extending Yohou](extending-yohou.md) discusses when subclassing a base is warranted over composing what exists.
