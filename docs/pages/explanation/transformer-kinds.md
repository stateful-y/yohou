# Transformer Kinds

Yohou transformers come in two kinds, and the kind is a property of the frame shape a transformer consumes and produces rather than of what it computes. An actual-kind transformer ([`BaseActualTransformer`](/pages/api/generated/yohou.base.transformer.BaseActualTransformer/)) operates on a single-axis frame: one `"time"` column and one row per timestamp, a single series marching forward. A forecast-kind transformer ([`BaseForecastTransformer`](/pages/api/generated/yohou.base.forecast_transformer.BaseForecastTransformer/)) operates on an `X_forecast` frame, which carries two time axes, `vintage_time` (when a forecast was issued) and `time` (what it forecasts).

Every transformer declares its kind through a `kind` tag whose value is `"actual"` or `"forecast"`. Leaf transformers stamp the tag statically through their base class; the composition classes derive theirs from their children. Almost everything in Yohou is actual-kind, and that is the default a transformer gets if it says nothing.

## Why Two Kinds

The two axes make an `X_forecast` frame a different object, not merely a wider one. A single-axis frame is one series. An `X_forecast` frame is a stack of short series, one per vintage, each covering its own forecast horizon and each overlapping the others in `time`. The 6:00 AM weather forecast and the 9:30 AM forecast both say something about noon, so `time` alone does not identify a row.

That difference is why a single-axis transformer cannot simply be pointed at an `X_forecast` frame. Consider a transformer that computes a first difference. On one series, "the previous row" is unambiguous and means one timestep back. On a vintage stack, the row above may belong to a different vintage entirely, so the difference would silently subtract one forecast from a neighbouring one and produce a number that corresponds to nothing. The frame would be accepted and the output would be wrong, which is the worst available outcome. The kind tag exists so that the mismatch is caught rather than computed.

## Kind and Statefulness Are Orthogonal

Statefulness is Yohou's other transformer axis: a stateful transformer keeps a bounded buffer of recent rows and declares how many it needs through its `observation_horizon`, while a stateless transformer's output depends only on its fitted parameters and the current input. That axis is independent of kind, which gives four combinations, of which three are populated:

|              | Stateless                          | Stateful                          |
| ------------ | ---------------------------------- | --------------------------------- |
| **Actual**   | scaling, log transforms, calendar features | lags, rolling statistics, filters |
| **Forecast** | lifted stateless transformers      | structurally impossible           |

The empty cell is the interesting one, and it is empty for a reason rather than for want of implementation. State on the time axis means memory: a buffer of the rows immediately preceding the current one, which is only meaningful if "preceding" is well defined and the history is contiguous. The vintage axis offers neither. Vintages are separate short series, and the rows before a given vintage's first row belong to a different vintage. There is no contiguous history for a buffer to hold, so there is nothing for an `observation_horizon` to count.

This is why [`PerVintageActualTransformer`](/pages/api/generated/yohou.compose.per_vintage.PerVintageActualTransformer/) accepts only stateless inner transformers, rejecting anything that measures a non-zero `observation_horizon` after fitting. The rejection is not a conservative guard around an unfinished feature. A stateful transformer on the vintage axis is asking for memory that the data shape cannot supply.

Forecast-kind transformers are therefore stateless as a class, which is also why they have no `observe` and `rewind` methods. Those exist to manage a memory buffer, and there is none to manage. Serving a single fresh vintage is just a one-group input, no different in kind from a frame holding a year of them.

## Lifting Rather Than Reimplementing

Yohou ships one concrete forecast-kind transformer, [`PerVintageActualTransformer`](/pages/api/generated/yohou.compose.per_vintage.PerVintageActualTransformer/), against a couple of dozen actual-kind ones. That ratio is a design position rather than a gap in coverage.

The alternative would have been a parallel catalog: a vintage-aware scaler, a vintage-aware function transformer, a vintage-aware imputer, each duplicating the logic of its single-axis twin and each able to drift from it. Instead, `PerVintageActualTransformer` wraps a stateless actual transformer and applies it to each vintage independently, grouping by `vintage_time`, handing each group's single-axis slice to the wrapped transformer, and restacking the results. Because every vintage is transformed using only its own rows, the order-dependent operations that would otherwise bleed across vintage boundaries stay contained. One wrapper lifts the whole catalog, and the wrapped transformers remain the same objects that run on `X_actual`.

Lifting is therefore the normal way to transform an `X_forecast` frame, not a stopgap. `BaseForecastTransformer` is the extension point beneath it: subclass it when an operation is genuinely about the vintage structure itself and so cannot be expressed as a per-vintage application of a single-axis transformer. Comparing each vintage against the one before it, for example, is inherently cross-vintage and has no single-axis equivalent to lift. Everything expressible per vintage should be lifted instead.

## Kind in Composition

The composition classes are polymorphic in kind rather than fixed to one. A [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/) of forecast-kind branches is itself forecast-kind, and one of actual-kind branches is actual-kind. A composition must be homogeneous: mixing the two in a single container is rejected, because the container would have to align a one-axis frame against a two-axis one and there is no sensible answer. [Feature Pipelines](feature-pipelines.md) covers how composition derives its kind, aligns its branches, and reports the mismatch.

Kind also determines where a transformer may be attached. A forecaster's `feature_transformer` and `target_transformer` slots process single-axis data, so they accept actual-kind transformers only. Frames on the forecast channel are transformed before they are handed to the forecaster.

## Connections

- [Preprocessing](preprocessing.md) covers actual-kind transformers in depth, including the observation horizon and the stateful lifecycle that only that kind has.
- [Feature Pipelines](feature-pipelines.md) covers composition, including how a container derives its kind from its children and why homogeneity is required.
- [Exogenous Features](exogenous-features.md) explains the three exogenous channels and what makes `X_forecast` vintage-indexed in the first place.
- [Core Concepts](core-concepts.md) places both transformer bases in the wider estimator hierarchy.
- [Extending Yohou](extending-yohou.md) discusses when subclassing a base is warranted over composing what exists.
