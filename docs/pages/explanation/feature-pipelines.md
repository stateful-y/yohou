# Feature Pipelines

Yohou's transformer composition classes let you combine multiple preprocessing
steps into a single transformer that respects the temporal contract: the `"time"`
column passes through unchanged, `observation_horizon` requirements are tracked
correctly, and `observe`/`rewind` state propagates to every step in the
composition.

Three patterns cover the common cases. Sequential patterns (`FeaturePipeline`)
chain steps in order. Parallel patterns (`FeatureUnion`, `ColumnTransformer`)
run steps on the same data independently and concatenate the results.

For composing forecasters rather than transformers, see
[Forecaster Composition](forecaster-composition.md).

## Sequential Patterns

### [`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/)

Chains transformers sequentially: each transformer's output feeds into the next.
This is the time-series equivalent of sklearn's `Pipeline`, but it respects the
temporal contract.

Use `FeaturePipeline` when preprocessing steps must execute in order, such as
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
`FeaturePipeline` enforces this ordering while propagating `observation_horizon`
and `memory` requirements from each step.

## Parallel Patterns

### [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/)

Runs multiple transformers in parallel and concatenates their outputs column-wise.
This is useful when you want features from different sources: lag features
alongside rolling statistics alongside calendar features.

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

### [`ColumnTransformer`](/pages/api/generated/yohou.compose.column_transformer.ColumnTransformer/)

Applies different transformers to different column subsets. Columns not matched by
any transformer are handled by the `remainder` parameter (drop, passthrough, or a
default transformer).

This is the time-series analogue of sklearn's `ColumnTransformer`, adapted for
polars DataFrames with a time column. It is useful when different features need
different preprocessing: numeric columns might need scaling while categorical
columns need encoding.

## Observation Horizon Propagation

Each transformer declares its `observation_horizon`: the number of past rows it
needs to produce valid output. Stateless transformers have `observation_horizon == 0`.
A `LagTransformer(lag=[1, 3])` has `observation_horizon == 3` (the maximum lag).
A `RollingStatisticsTransformer(window_size=7)` has `observation_horizon == 7`.

When you compose transformers, the combined `observation_horizon` is derived from
the composition pattern:

**`FeaturePipeline`**: the combined `observation_horizon` is the **sum** across
all steps. This is because each step's output loses its first
`observation_horizon_step` rows, so the next step needs to receive more rows to
produce any valid output at all. A pipeline of `StandardScaler` (0) followed by
`LagTransformer(lag=[1, 2])` (2) has a combined `observation_horizon` of 2.
A pipeline of `LagTransformer(lag=[1, 2])` (2) followed by
`RollingStatisticsTransformer(window_size=3)` (3) has a combined horizon of 5.

**`FeatureUnion`**: the combined `observation_horizon` is the **maximum** across
all transformers, since each branch receives the same input and the bottleneck is
the branch that needs the most history.

**`ColumnTransformer`**: the combined `observation_horizon` is the **maximum**
across all transformers applied to their respective columns.

Forecasters read `observation_horizon` from their `feature_transformer` and
`target_transformer` to determine how much history to retain in `_y_observed`
and `_X_observed`. This means nesting deeply pipelined transformers increases the
memory footprint; keeping horizons short is advisable for memory-constrained
deployments.

## Sequential vs. Parallel

The choice between sequential and parallel patterns depends on data flow:

**Sequential** (`FeaturePipeline`): each step depends on the previous step's output.
Use when operations must happen in order. The canonical example is imputation
followed by scaling followed by feature engineering: each step assumes the previous
one has already run.

**Parallel** (`FeatureUnion`, `ColumnTransformer`): steps are independent. Use when
different transformations operate on the same input independently and their results
should be merged. The canonical example is combining lag features, rolling statistics,
and calendar features: each branch produces a different type of information from the
same time series, and all are needed as model inputs.

These patterns compose freely. A `FeatureUnion` can be a step inside a
`FeaturePipeline`, and the combined transformer can serve as the
`feature_transformer` for any forecaster:

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

This composability is the reason the module provides separate building blocks
rather than a single monolithic pipeline class: smaller pieces combine more
flexibly.

## Connections

- [Preprocessing](preprocessing.md): transformers used inside pipelines, and how
  `observe`/`rewind` state works on individual transformers
- [Forecaster Composition](forecaster-composition.md): composing forecasters
  rather than transformers
- [Stationarity](stationarity.md): stationarity transforms that can be used as
  steps inside a `FeaturePipeline`
- [API Reference: yohou.compose](/pages/api/compose/)
