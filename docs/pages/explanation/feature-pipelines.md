# Feature Pipelines

Scikit-Learn's [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) and [`FeatureUnion`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html) are the standard tools for transformer
composition in the Python ML ecosystem. Yohou provides its own counterparts
([`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/), [`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/), [`ColumnTransformer`](/pages/api/generated/yohou.compose.ColumnTransformer/)) because time series
transformers carry obligations that Scikit-Learn's containers do not enforce.
The most important is the `observation_horizon`: a stateful transformer needs a
certain number of past observations to produce valid output. When transformers are
composed, the container must propagate that requirement so that forecasters know
how much history to retain. Scikit-Learn's `Pipeline` has no concept of temporal
lookback and would discard this information. The second obligation is the `"time"`
column: time series DataFrames carry a timestamp column that must pass through
every transformer step without being treated as a regular feature. The third is
statefulness: each transformer may maintain an internal memory buffer through
`observe` and `rewind`, and the composition classes propagate those calls to every
component. Yohou's composition classes handle all three automatically.

Three patterns cover the common cases. Sequential composition
([`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/)) chains steps in order. Parallel composition
([`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/), [`ColumnTransformer`](/pages/api/generated/yohou.compose.ColumnTransformer/)) runs steps independently on the same
data and concatenates the results.

For composing forecasters rather than transformers, see
[Forecaster Composition](forecaster-composition.md).

## FeaturePipeline

[`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/) chains transformers sequentially: each transformer's output feeds
into the next. This is the time series equivalent of sklearn's
[`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), but it respects the temporal contract.

Use [`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/) when preprocessing steps must execute in order, such as
imputation followed by scaling followed by lag feature extraction.

```python
from yohou.compose import FeaturePipeline
from yohou.preprocessing import SimpleTimeImputer, StandardScaler, LagTransformer

transformer = FeaturePipeline(steps=[
    ("impute", SimpleTimeImputer()),
    ("scale", StandardScaler()),
    ("lags", LagTransformer(lag=[1, 2, 3])),
])
```

Order matters here. Imputing after scaling would leave NaN gaps in the scaled
data; computing lag features before scaling would mix raw and scaled values.
[`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/) enforces this ordering while propagating `observation_horizon`
and state requirements from each step.

[`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/) also supports `inverse_transform`: it applies each step's
inverse in reverse order, which is needed when a forecaster must map predictions
back through a `target_transformer`. This requires every step in the pipeline to
be invertible.

The optional `memory` parameter accepts a `joblib.Memory` instance (or a path
string) to cache fitted transformers, which can speed up repeated fits during
hyperparameter search.

## FeatureUnion

[`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/) runs multiple transformers in parallel and concatenates their outputs
column-wise. This is useful when you want features from different sources: lag
features alongside rolling statistics alongside calendar features.

```python
from yohou.compose import FeatureUnion
from yohou.preprocessing import LagTransformer, RollingStatisticsTransformer

transformer = FeatureUnion(transformer_list=[
    ("lags", LagTransformer(lag=[1, 7, 14])),
    ("rolling", RollingStatisticsTransformer(window_size=7)),
])
```

Each transformer receives the same input independently, so there is no ordering
dependency. The `transformer_weights` parameter allows scaling each transformer's
output by a factor, and `n_jobs` enables parallel execution across transformers.

When `verbose_feature_names_out=True` (the default), output columns are prefixed
with the transformer name using a single underscore separator (for example,
`lags_sales`). For panel data the prefix is inserted after the group separator, so
`store_1__sales` becomes `store_1__lags_sales` rather than `lags_store_1__sales`.
This preserves the panel column convention.

## ColumnTransformer

[`ColumnTransformer`](/pages/api/generated/yohou.compose.ColumnTransformer/) applies different transformers to different column subsets. This
is the time series analogue of sklearn's
[`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html),
adapted for polars DataFrames with a time column. It is useful when different
features need different preprocessing: numeric columns might need scaling while
categorical columns need encoding.

```python
from yohou.compose import ColumnTransformer
from yohou.preprocessing import FunctionTransformer, StandardScaler

transformer = ColumnTransformer(transformers=[
    ("num", StandardScaler(), ["temperature", "humidity"]),
    ("cat", FunctionTransformer(), ["weather_type"]),
], remainder="passthrough")
```

Columns not matched by any transformer are handled by the `remainder` parameter:
`"drop"` (the default) discards them, `"passthrough"` keeps them unchanged, or
a transformer instance applies a default transformation.

Like [`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/), [`ColumnTransformer`](/pages/api/generated/yohou.compose.ColumnTransformer/) supports `n_jobs` for parallel
execution, `transformer_weights` for scaling outputs, and
`verbose_feature_names_out` for panel-aware column prefixing.

Internally, [`ColumnTransformer`](/pages/api/generated/yohou.compose.ColumnTransformer/) strips the `"time"` column before routing data
to individual transformers, then reattaches it in the final output. This prevents
column index mismatches that would occur if sklearn's internal bookkeeping tried
to track the time column. On a forecast-kind composition it protects `vintage_time`
the same way, since that column is part of the index rather than a feature to route.

## Kind Homogeneity

All three containers are polymorphic in transformer kind: each works on single-axis
actual data, on vintage-indexed forecast data, or on the derived step frame, and
none is fixed to one. A container
does not declare its kind. It derives it from its children, so a
[`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/) of forecast-kind branches is itself forecast-kind and can nest
inside another forecast-kind container. See [Transformer Kinds](transformer-kinds.md)
for what the three kinds are and why the distinctions exist.

Deriving a kind from the children only works if the children agree, which is why a
composition must be homogeneous. Mixing kinds in one container raises a `ValueError`
naming the members of each kind present. The restriction
is not bookkeeping: the container would have to concatenate a frame indexed by `"time"`
with one indexed by `vintage_time` and `time`, and there is no correct way to line those
up. One row per timestamp cannot be matched against a stack of vintages that each say
something different about that timestamp.

Kind also decides how a container combines its branches, because it decides what
identifies a row. Actual-kind and step-kind containers align their branches on
`"time"`. A forecast-kind container aligns on `vintage_time` and `time` together,
since `"time"` alone does not identify a row once several vintages are in play. Composition is
otherwise index-agnostic: the same container code runs for both, reading the join keys
from the kind it derived.

## Observation Horizon Propagation

Each transformer declares its `observation_horizon`: the number of past rows it
needs to produce valid output. Stateless transformers have `observation_horizon == 0`.
A `LagTransformer(lag=[1, 3])` has `observation_horizon == 3` (the maximum lag).
A `RollingStatisticsTransformer(window_size=7)` has `observation_horizon == 7`.

When you compose transformers, the combined `observation_horizon` depends on the
composition pattern:

**[`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/)**: the combined `observation_horizon` is the **sum** across
all steps. Each step's `transform` drops its first `observation_horizon` rows
from the output, so the next step receives fewer rows. The pipeline must receive
enough input for every step to produce at least one valid row after all the
dropoffs accumulate. A pipeline of [`StandardScaler`](/pages/api/generated/yohou.preprocessing.StandardScaler/) (0) followed by
`LagTransformer(lag=[1, 2])` (2) has a combined `observation_horizon` of 2.
A pipeline of `LagTransformer(lag=[1, 2])` (2) followed by
`RollingStatisticsTransformer(window_size=3)` (3) has a combined horizon of 5.

**[`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/)**: the combined `observation_horizon` is the **maximum** across
all transformers. Each branch receives the same input, so the bottleneck is the
branch that needs the most history.

**[`ColumnTransformer`](/pages/api/generated/yohou.compose.ColumnTransformer/)**: the combined `observation_horizon` is the **maximum**
across all transformers applied to their respective columns, following the same
logic as [`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/).

Forecasters read `observation_horizon` from their `actual_transformer` and
`target_transformer` to determine how much history to retain in `_y_observed`
and `_X_observed`. Nesting deeply pipelined transformers increases the memory
footprint because the pipeline's summed horizon grows with each step. Keeping
individual horizons short is advisable for memory-constrained deployments.

## State Propagation

Stateful transformers maintain an internal buffer of recent observations so that
`observe_transform` can produce valid output on new data without reprocessing the
entire history. These are the transformer-level analogues of the forecaster
`observe`/`rewind` lifecycle described in
[Core Concepts](core-concepts.md#observation-horizon): `observe_transform` combines
observe-then-transform, and `rewind_transform` rolls the buffer back to an earlier
time and retransforms from there. The composition classes propagate `observe`,
`rewind`, `observe_transform`, and `rewind_transform` calls to every component:

[`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/) forwards these calls **sequentially** through each step. When
`observe_transform` is called, the first step observes and transforms the new
data, then its output is passed to the next step, and so on. `rewind_transform`
replays the full input from scratch, discards the warmup rows, and resets each
step's buffer.

[`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/) and [`ColumnTransformer`](/pages/api/generated/yohou.compose.ColumnTransformer/) forward these calls **in parallel** to
all child transformers. Each transformer manages its own buffer independently.
The results are aligned by the `"time"` column before horizontal concatenation,
so transformers with different `observation_horizon` values still produce a
consistent output.

`observe` also validates temporal continuity: new data must follow directly from
the last observed timestamp. This prevents silent gaps that would corrupt
stateful features.

## Composability

Sequential and parallel patterns compose freely within a kind. A [`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/) can
be a step inside a [`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/), nested to any depth, so long as every
transformer in the structure shares one kind. An actual-kind combination can then serve
as the `actual_transformer` or `target_transformer` for a forecaster:

```python
from yohou.compose import FeaturePipeline, FeatureUnion
from yohou.preprocessing import StandardScaler, LagTransformer, RollingStatisticsTransformer

transformer = FeaturePipeline(steps=[
    ("scale", StandardScaler()),
    ("features", FeatureUnion(transformer_list=[
        ("lags", LagTransformer(lag=[1, 2, 3, 7])),
        ("rolling", RollingStatisticsTransformer(window_size=7, statistics=["mean", "std"])),
    ])),
])
```

Every step here is actual-kind, so the structure is homogeneous and the whole is
actual-kind too. The observation horizon of this nested structure is the sum of the
pipeline's steps: 0 ([`StandardScaler`](/pages/api/generated/yohou.preprocessing.StandardScaler/)) + max(7, 7) ([`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/)) = 7. State
propagation, feature naming, and panel-aware prefixing all carry through the nesting
without any additional configuration.

Both of those slots process single-axis data, so both accept actual-kind transformers
only, and passing a forecast-kind composition to either is an error rather than a
no-op. That rejection is not a dead end. A forecast-kind composition has a slot of its
own, `forecast_transformer`, which applies it to the vintage-indexed `X_forecast` frame
before the step columns are derived. The three slots divide by the shape each one
consumes: `target_transformer` and `actual_transformer` read a single `time` axis,
`forecast_transformer` reads `(vintage_time, time)`.

So a composition goes wherever its kind goes, and its kind is derived from its children.
The error you get from putting one in the wrong slot names the slot that would have
accepted it. See
[How to Transform Features on the Forecast Channel](../how-to/transform-forecast-features.md).

## Connections

[Preprocessing](preprocessing.md) covers the individual transformers used inside pipelines, including how `observe` and `rewind` state works on each transformer. [Transformer Kinds](transformer-kinds.md) explains the actual, forecast, and step kinds that composition requires to be homogeneous. [Forecaster Composition](forecaster-composition.md) discusses composing forecasters rather than transformers. Stationarity transforms that can serve as steps inside a [`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/) are described in [Stationarity](stationarity.md).

For practical recipes, see [How to Compose Feature Pipelines](../how-to/compose-feature-pipelines.md). The compose API is documented in the [yohou.compose reference](/pages/api/compose/).
