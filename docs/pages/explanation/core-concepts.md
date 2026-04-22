# Core Concepts

Yohou turns time series forecasting into a supervised learning problem while preserving
temporal structure. Rather than inventing a new estimator API, it extends scikit-learn's
familiar fit/predict interface with a small set of time-aware operations (`observe`,
`rewind`, and composite methods like `observe_predict`) so that any sklearn regressor
can power a forecaster. This page explains the concepts that make that bridge work.


## The Time Column Contract

Every polars DataFrame that flows through yohou must contain a `"time"` column of
datetime type. This single convention is what separates time series data from plain
tabular data, and yohou enforces it at every entry point.

The contract is simple:

- **`y`** is the target time series, containing the values you want to forecast. It has a `"time"`
  column plus one or more numeric value columns.
- **`X`** is the exogenous feature matrix, containing known-in-advance variables such as holidays,
  promotions, or weather. It also has a `"time"` column aligned with `y`.

Transformers preserve the `"time"` column through `transform()` and `inverse_transform()`.
The time column passes through unchanged, while value columns are modified. This makes
transformers composable, letting you chain them without losing temporal context.

Forecasters produce predictions with two time columns:

- `"vintage_time"`: the last timestamp the forecaster observed before making the prediction.
- `"time"`: the future timestamps being forecast.

A *vintage* is a single forecast origin: one call to `predict` or `observe_predict`
from a specific observation point. During rolling evaluation, each vintage corresponds
to one step of the walk-forward loop. Grouping predictions by `vintage_time` lets you
analyze how accuracy evolves as the forecaster sees more data (see
[Forecast Accuracy: Vintage-based Evaluation](forecast-accuracy.md#vintage-based-evaluation)).

The `"vintage_time"` column exists because the same forecaster can generate predictions
from different observation points during rolling evaluation. It anchors each prediction
to the information available when it was made.


## Polars-native Design

Yohou uses polars DataFrames end-to-end. There is no conversion to pandas or NumPy in
the core library (the reduction layer converts to NumPy only at the boundary where data
enters an sklearn regressor).

Polars brings several advantages for time series work:

- **Strict typing**: Column dtypes are enforced, not inferred. A Float64 column stays
  Float64 through transformations, and type mismatches surface as errors rather than
  silent coercions.
- **Expression-based API**: Polars expressions like `pl.col("value").shift(1)` and
  selectors like `cs.numeric()` make column operations explicit and composable.
  `cs.by_name("time")` appears frequently for excluding the time column from numeric
  operations.
- **Performance**: Polars executes operations in Rust with automatic parallelism. For
  the kind of grouped, windowed, and rolling operations common in time series
  preprocessing, this matters.
- **Datetime handling**: Polars natively distinguishes between regular intervals
  (`Duration` type for "1h", "1d") and calendar intervals (`"1mo"`, `"1y"` where month
  lengths vary). Yohou's
  [`check_interval_consistency`](/pages/api/generated/yohou.utils.validation.check_interval_consistency/)
  validates that time series have uniform spacing using this machinery.

Code within yohou's `src/` directory uses polars idioms consistently: selector-based
column selection, expression chaining, and `pl.concat` for combining DataFrames. If you
are coming from pandas, the main adjustment is thinking in expressions rather than
index-based operations.


## The sklearn Bridge

Yohou's central design decision is extending scikit-learn's `BaseEstimator` rather than
replacing it. Every forecaster and transformer inherits from `BaseEstimator`, gaining
`get_params()`, `set_params()`, cloning, and HTML representation for free. On top of
this, yohou adds time series methods.

The standard `fit` and `predict` methods work like their sklearn counterparts, with
one important difference: the forecasting horizon is specified at `fit` time because
reduction-based forecasters need to know how many steps ahead to tabularize. The
horizon at `predict` time can differ from the fit horizon because the model applies
recursively to reach further into the future (see
[Advanced Topics](advanced.md#recursive-prediction) for how this works internally).

The time series extensions are `observe` and `rewind`. Together they implement a
sliding-window memory model that makes rolling evaluation efficient. As new data
arrives, `observe` updates the forecaster's internal buffers without the cost of
retraining. `rewind` resets those buffers to a fixed-size window. The composite
methods `observe_predict` and `observe_predict_interval` combine observation and
prediction into a single atomic call, which is the most common operation during
rolling evaluation.

Interval-specific methods (`predict_interval`, `observe_predict_interval`) live on
[`BaseIntervalForecaster`](/pages/api/generated/yohou.interval.base.BaseIntervalForecaster/)
rather than on `BaseForecaster`, and class-probability methods
(`predict_class_proba`, `observe_predict_class_proba`) live on
[`BaseClassProbaForecaster`](/pages/api/generated/yohou.class_proba.base.BaseClassProbaForecaster/).
This keeps the base class focused on point prediction while allowing specialized
forecasters to add their prediction types.

**Metadata routing** is enabled automatically when yohou is imported. The `__init__.py`
module calls `set_config(enable_metadata_routing=True)` and registers custom composite
methods so that sklearn's routing machinery can handle `observe_transform`,
`observe_predict`, and other combined operations. Parameters like `time_weight` flow
through pipelines and compositions without manual wiring. See
[Advanced Topics](advanced.md#metadata-routing) for the full list of registered methods.

For transformers, the pattern mirrors forecasters.
[`BaseTransformer`](/pages/api/generated/yohou.base.base.BaseTransformer/) extends
`BaseEstimator` with `observe` and `rewind` for memory management. The composite
`observe_transform` method transforms using pre-existing memory, then updates state.
`rewind_transform` applies the full transformation (which internally drops the first
`observation_horizon` rows for stateful transformers), then rewinds the state.

This design means yohou components work with sklearn utilities like `clone()`,
[`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) (via yohou's time-series-aware wrapper), and `Pipeline` composition.
See [Model Selection](model-selection.md) for details on cross-validation.


## Observation Horizon

The `observation_horizon` property is how yohou components declare their memory
requirements. It answers: "how many past time steps does this component need to see
before it can produce output?"

For a transformer like [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/)
with `lags=[1, 7]`, the observation horizon is 7 because it needs at least 7 prior rows to
compute all requested lags. For a
[`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/)
transformer with `period=12`, the observation horizon is 12.

Forecasters compute their observation horizon as the maximum across their constituent
transformers. A forecaster with a target transformer needing 7 rows and a feature
transformer needing 12 rows has an observation horizon of 12. This composition happens
automatically. The `observation_horizon` property on
[`BaseForecaster`](/pages/api/generated/yohou.base.forecaster.BaseForecaster/)
walks the transformer tree and returns the maximum.

The observation horizon drives the memory management pattern:

1. `observe()` appends new data to internal buffers (`_X_observed`, `_y_observed`).
2. `observe()` then calls `rewind()`, which trims those buffers to exactly
   `observation_horizon` rows.
3. The result is a fixed-size sliding window that always contains just enough history for
   the next operation.

Stateless transformers (like scaling or log transforms) have an observation horizon of 0.
They need no memory and their `observe`/`rewind` operations are essentially no-ops on the
data dimension. Stateful transformers (like lag features or seasonal differencing) have a
positive observation horizon and maintain a rolling buffer of recent data.

This distinction is reflected in the tags system:
`TransformerTags.stateful` is `True` when a transformer has a nonzero observation
horizon. Forecasters inherit statefulness from their transformers: if any attached
transformer is stateful, the forecaster is stateful too.


## Univariate, Multivariate, and Panel Data

Yohou handles three data shapes through a naming convention rather than separate APIs.

**Univariate** data has a single target column:

```python
y = pl.DataFrame({"time": dates, "sales": [100, 110, 120, ...]})
```

**Multivariate** data has multiple target columns with no special naming:

```python
y = pl.DataFrame({
    "time": dates,
    "temperature": [20.1, 21.3, ...],
    "humidity": [0.65, 0.70, ...],
})
```

**Panel data** uses the `{entity}__{variable}` double-underscore convention to encode
multiple related time series:

```python
y = pl.DataFrame({
    "time": dates,
    "store_1__sales": [100, 110, ...],
    "store_2__sales": [150, 160, ...],
    "store_1__returns": [5, 3, ...],
    "store_2__returns": [8, 6, ...],
})
```

The [`inspect_panel`](/pages/api/generated/yohou.utils.panel.inspect_panel/) function
parses these names, returning global columns (no `__`) and a dictionary of panel groups.
In the example above, it would find groups `"store_1"` and `"store_2"`, each with
columns `"sales"` and `"returns"`.

Forecasters handle panel data through the `panel_strategy` parameter:

- **`"global"`** (default): Detects panel groups automatically. Each group gets its own
  transformer instances (independent state, observation buffers) but shares a single
  fitted model. This is the pooled-model approach: per-group features, global parameters.
- **`"multivariate"`**: Skips panel detection entirely. The `__`-prefixed columns are
  treated as ordinary wide-format columns. One transformer and one model see the full
  DataFrame, enabling cross-group feature interactions.

Panel group names can be passed to `observe`, `predict`, and `rewind` via the
`groups` parameter to operate on a subset of groups, which is useful for scenarios
where new data arrives for some entities but not others.


## Tags System

Yohou estimators declare their capabilities through a structured tag system. Each
estimator implements `__sklearn_tags__()` returning a [`Tags`](/pages/api/generated/yohou.utils.tags.Tags/)
dataclass that contains nested tag groups for different estimator types:
[`ForecasterTags`](/pages/api/generated/yohou.utils.tags.ForecasterTags/),
[`TransformerTags`](/pages/api/generated/yohou.utils.tags.TransformerTags/),
[`ScorerTags`](/pages/api/generated/yohou.utils.tags.ScorerTags/), and
[`SplitterTags`](/pages/api/generated/yohou.utils.tags.SplitterTags/).

Tags exist to make estimator capabilities machine-readable. They serve three roles:

**Validation**: The testing framework reads tags to determine which checks apply. A
forecaster with `ignores_exogenous=True` skips checks that require exogenous features.
A scorer with `lower_is_better=True` flips the comparison direction in search objects.
This means adding a new estimator automatically gets the right subset of the 27
forecaster checks or 11 scorer checks without maintaining explicit test lists.

**Composition**: Composite estimators inspect child tags to wire data flow correctly.
[`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/)
checks whether sub-forecasters use target transformers to decide how to pass residuals.
The `observation_horizon` property aggregates transformer statefulness tags to compute
the total memory requirement.

**Discovery**: The [`all_estimators()`](/pages/api/generated/yohou.utils.discovery.all_estimators/)
function reads `estimator_type` tags to filter components by kind. This powers the
auto-generated API pages and lets you build registries programmatically.

Some tags are set statically in a class definition (like `ignores_exogenous`), while
others are computed dynamically. For example, `ForecasterTags.stateful` is derived at
runtime by checking whether any attached transformer has a nonzero observation horizon.
This dynamic derivation means you cannot tell from a class definition alone whether a
forecaster is stateful. It depends on how it is configured.

Yohou supports three prediction types:

- **Point predictions** (`predict()`): a single numeric value per timestep
- **Interval predictions** (`predict_interval()`): lower/upper bounds per coverage rate
- **Class-probability predictions** (`predict_class_proba()`): probability distributions over categorical classes

For more on how these pieces connect in practice, see the pages on
[Forecasting](forecasting.md), [Interval Forecasting](interval-forecasting.md),
[Class-Probability Forecasting](class-probability-forecasting.md),
[Preprocessing](preprocessing.md), and [Model Selection](model-selection.md).
