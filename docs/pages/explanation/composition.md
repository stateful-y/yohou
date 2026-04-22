# Composition and Pipelines

Most real-world forecasting workflows combine multiple operations: scaling,
lag feature extraction, seasonal decomposition, and the forecasting model itself.
Yohou's composition module provides building blocks for assembling these operations
into coherent pipelines that respect the temporal structure of the data.

The composition patterns fall into two categories: sequential (one operation feeds
into the next) and parallel (multiple operations run independently and their results
are combined). Each pattern is a full forecaster or transformer that can be used
anywhere a single one is expected, which means compositions can be nested
arbitrarily.

## Sequential Patterns

### [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/)

Decomposes a time series into additive components by fitting forecasters in
sequence. Each forecaster models the residuals left by all previous forecasters,
and the final prediction is the sum of all component predictions.

$$\hat{y}_t = \hat{f}_1(t) + \hat{f}_2(t) + \cdots + \hat{f}_k(t)$$

The `forecasters` parameter takes a list of `(name, forecaster)` tuples:

```python
from yohou.compose import DecompositionPipeline
from yohou.stationarity import PolynomialTrendForecaster
from yohou.point import SeasonalNaive

pipeline = DecompositionPipeline(forecasters=[
    ("trend", PolynomialTrendForecaster(degree=1)),
    ("seasonality", SeasonalNaive(seasonality=12)),
])
```

The first forecaster fits the raw data and produces a trend forecast. The second
receives the residuals (original minus trend) and models what remains. This is the
classic decompose-forecast-recompose pattern, but expressed as a chain of
forecasters rather than explicit transformer-forecaster pairs.

For multiplicative decomposition, pass a `target_transformer=LogTransformer()`.
This transforms the target into log-space where multiplication becomes addition,
applies the additive pipeline, and back-transforms the result. The
`store_residuals=True` option saves intermediate residuals in
`pipeline.residuals_` for diagnostic inspection.

### [`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/)

Chains transformers sequentially: each transformer's output feeds into the next.
This is the time-series equivalent of sklearn's `Pipeline`, but it respects the
temporal contract (the `"time"` column passes through unchanged, and
`observation_horizon` / `memory` state is managed correctly).

Use `FeaturePipeline` when preprocessing steps must execute in order, such as
imputation followed by scaling followed by lag feature extraction.

```python
from yohou.compose import FeaturePipeline
from yohou.preprocessing import SimpleTimeImputer, StandardScaler, LagTransformer

transformer = FeaturePipeline(steps=[
    ("impute", SimpleTimeImputer()),
    ("scale", StandardScaler()),
    ("lags", LagTransformer(lags=[1, 2, 3])),
])
```

Order matters here. Imputing after scaling would leave NaN gaps in the scaled
data; computing lag features before scaling would mix raw and scaled values.
`FeaturePipeline` enforces this ordering while propagating `observation_horizon`
and `memory` requirements from each step to determine the total data the pipeline
needs.

## Parallel Patterns

### [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/)

Runs multiple transformers in parallel and concatenates their outputs column-wise.
This is useful when you want features from different sources: lag features alongside
rolling statistics alongside calendar features.

```python
from yohou.compose import FeatureUnion
from yohou.preprocessing import LagTransformer, RollingStatisticsTransformer

transformer = FeatureUnion(transformer_list=[
    ("lags", LagTransformer(lags=[1, 7, 14])),
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

## Forecaster-Level Composition

The patterns above compose transformers. These next patterns compose forecasters
themselves, addressing situations where a single forecaster cannot handle the full
problem.

### [`ColumnForecaster`](/pages/api/generated/yohou.compose.column_forecaster.ColumnForecaster/)

Assigns different forecasters to different target columns. Each entry in the
`forecasters` list is a `(name, forecaster, columns)` tuple. Predictions from all
forecasters are concatenated horizontally.

This is useful when columns have fundamentally different characteristics. A
slow-moving trend variable might work best with a linear model while a volatile
signal needs gradient boosting. Forcing a single model to handle both can produce
mediocre predictions for each. The `remainder` parameter controls what happens
to columns not claimed by any forecaster: drop them, pass them through, or assign
a default forecaster.

### [`ForecastedFeatureForecaster`](/pages/api/generated/yohou.compose.forecasted_feature_forecaster.ForecastedFeatureForecaster/)

A two-stage forecaster for the common scenario where exogenous features are
available during training but not at prediction time. It chains a
`feature_forecaster` that predicts future `X` values with a `target_forecaster`
that uses those predicted features to forecast `y`.

The `strategy` parameter controls how to handle the distribution shift between
real and forecasted features during training:

- `"actual"`: trains the target forecaster on real `X`. Simple but creates a
  mismatch since prediction uses forecasted `X`.
- `"predicted"`: splits the data and trains the target forecaster on predicted
  `X`, avoiding the shift.
- `"rewind"`: fits the feature forecaster on all data, rewinds, then predicts
  `X` for target training. This uses all data for feature learning while still
  avoiding the shift.

### [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/)

Fits a separate forecaster instance per panel group rather than a single global
model. This is appropriate when groups have fundamentally different dynamics (e.g.,
different products with unrelated demand patterns) and a global model would blur
the distinctions. The trade-off is that each group trains on only its own data,
which can be a problem for groups with short histories. Global models share
information across groups at the cost of missing group-specific patterns.

## Sequential vs. Parallel

The choice between sequential and parallel patterns depends on data flow:

- **Sequential** (`FeaturePipeline`, `DecompositionPipeline`): Each step depends on
  the previous step's output. Use when operations must happen in order.
- **Parallel** (`FeatureUnion`, `ColumnTransformer`): Steps are independent. Use
  when different transformations operate on different aspects of the data and their
  results should be concatenated.

These patterns compose freely. A `FeatureUnion` can be a step inside a
`FeaturePipeline`, and the combined transformer can serve as the
`feature_transformer` for any forecaster. This composability is the reason the
module exists as separate building blocks rather than as a single monolithic
pipeline class: smaller pieces combine more flexibly.

### Related pages

- [Preprocessing](preprocessing.md): transformers used inside pipelines
- [Advanced Topics](advanced.md): how `observe` and `rewind` propagate through composites
- [Custom Estimators](../how-to/custom-estimators.md): building custom components
- [API Reference: yohou.compose](../api/compose.md)
