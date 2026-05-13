# Forecaster Composition

Yohou provides four classes that compose forecasters into larger forecasting
structures. These operate at the forecaster level: each component is itself a
full forecaster with `fit`/`predict`/`observe`/`rewind` lifecycle, not a
transformer or preprocessing step.

Forecaster composition addresses situations where a single forecaster cannot
handle the full problem: multiple additive components in the data, multiple
target columns with different dynamics, features that need to be forecast before
the target can be forecast, or panel groups with fundamentally different patterns.

For composing transformers (feature pipelines, scaling chains, lag features), see
[Feature Pipelines](feature-pipelines.md).

## [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/)

Decomposes a time series into additive components by fitting forecasters in
sequence. Each forecaster models the residuals left by all previous forecasters,
and the final prediction is the sum of all component predictions:

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
classic decompose-forecast-recompose pattern, expressed as a chain of forecasters
rather than explicit transformer-forecaster pairs.

For multiplicative decomposition, pass `target_transformer=LogTransformer()`.
This transforms the target into log-space where multiplication becomes addition,
applies the additive pipeline, and back-transforms the result. The
`store_residuals=True` option saves intermediate residuals in
`pipeline.residuals_` for diagnostic inspection.

## [`ColumnForecaster`](/pages/api/generated/yohou.compose.column_forecaster.ColumnForecaster/)

Assigns different forecasters to different target columns. Each entry in the
`forecasters` list is a `(name, forecaster, columns)` tuple. Predictions from all
forecasters are concatenated horizontally.

This is useful when columns have fundamentally different characteristics. A
slow-moving trend variable might work best with a linear model while a volatile
signal needs gradient boosting. Forcing a single model to handle both can produce
mediocre predictions for each. The `remainder` parameter controls what happens
to columns not claimed by any forecaster: drop them, pass them through, or assign
a default forecaster.

## [`ForecastedFeatureForecaster`](/pages/api/generated/yohou.compose.forecasted_feature_forecaster.ForecastedFeatureForecaster/)

A two-stage forecaster for the scenario where exogenous features are available
during training but not at prediction time. It chains a `feature_forecaster` that
predicts future feature values with a `target_forecaster` that uses those predicted
features to forecast `y`.

The core challenge is a training distribution shift: the target forecaster will be
used at prediction time with forecasted (imperfect) feature values, but during
training the real feature values are available. Training on real features and
predicting with forecasted ones can degrade accuracy. The `strategy` parameter
controls how this is handled:

- `"actual"`: trains the target forecaster on real features. Simple but creates a
  mismatch since prediction uses forecasted values.
- `"predicted"`: splits the data and trains the target forecaster on predicted
  features, avoiding the shift.
- `"rewind"`: fits the feature forecaster on all data, rewinds, then predicts
  features for target training. This uses all data for feature learning while still
  avoiding the distribution shift.

For the data-shaping perspective on exogenous features (the three types `X_actual`,
`X_future`, `X_forecast`, and step-indexed columns), see
[Exogenous Features](exogenous-features.md).

## [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/)

Fits a separate forecaster instance per panel group rather than a single global
model. This is appropriate when groups have fundamentally different dynamics (for
example, products with unrelated demand patterns) and a global model would blur
the distinctions. The trade-off is that each group trains on only its own data,
which can be a problem for groups with short histories. Global models share
information across groups at the cost of missing group-specific patterns.

## State Propagation Through Composite Forecasters

Every forecaster maintains two observation buffers: `_y_observed` and
`_X_observed`. Calling `observe()` appends new data to these buffers (with time
continuity validation), while `rewind()` replaces them with the last
`observation_horizon` rows without validation. Composite forecasters build on
this by dispatching `observe` and `rewind` to their sub-components in patterns
that mirror how `fit` and `predict` flow.

**`DecompositionPipeline`** sequences in the same order as training. When
`observe()` is called, it transforms the incoming data through any target or
feature transformers, then iterates through forecasters in order: each one
predicts, computes residuals, and observes those residuals. This preserves the
additive decomposition contract where each stage works on what previous stages
left behind.

**`ColumnForecaster`** dispatches by column subset. Each sub-forecaster observes
only its assigned columns of `y`, but all receive the full `X_actual` unmodified.
Column splitting applies exclusively to the target.

**`ForecastedFeatureForecaster`** chains in two stages. The feature forecaster
observes `X_actual` as its target (it has no exogenous data of its own), then
the target forecaster observes `y` with `X_future` containing the feature
forecasts. This maintains the two-stage relationship: the feature forecaster
learns to predict exogenous features, and the target forecaster uses those
features as known-ahead inputs.

**`LocalPanelForecaster`** dispatches `observe` to each group's sub-forecaster
with the rows belonging to that group. Each instance maintains independent
observation buffers.

For the metadata routing infrastructure that enables these operations to flow
through search and cross-validation objects, see [Advanced Topics](advanced.md).

## Connections

- [Feature Pipelines](feature-pipelines.md): composing transformers rather than
  forecasters
- [Exogenous Features](exogenous-features.md): the three exogenous parameter types
  and step-indexed columns, which `ForecastedFeatureForecaster` is designed around
- [Ensemble Forecasting](ensemble-forecasting.md): combining forecasters by voting
  rather than by decomposition or column assignment
- [Advanced Topics](advanced.md): metadata routing and the reduction architecture
  that underlies all composite forecasters
- [API Reference: yohou.compose](/pages/api/compose/)
