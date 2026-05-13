# Extending Yohou

Yohou is designed to be extended with custom forecasters, transformers, and
scorers. This page explains the architecture that makes extension possible and
helps you decide when extending is the right approach.

## Composing vs Extending

Before writing a custom component, consider whether Yohou's built-in composition
tools already solve the problem. `PointReductionForecaster` wraps any sklearn
regressor as a time series forecaster. `FeaturePipeline` and `FeatureUnion`
chain transformers. `VotingPointForecaster` combines multiple forecasters.
These tools handle observation tracking, panel dispatch, and temporal structure
automatically.

Extending (writing a new base class subclass) is the right choice when:

- You need a forecasting algorithm that cannot be expressed as an sklearn
  estimator applied to tabularized features (for example, a state-space model or
  exponential smoothing).
- You need a transformer with custom temporal logic, such as one that requires a
  specific number of past observations to compute its output.
- You need a scoring function with domain-specific aggregation beyond the
  built-in stepwise/vintagewise/componentwise/groupwise modes.

In most cases, composition gets you there. Extending gives you full control at the
cost of implementing more methods and maintaining compatibility with the rest of
the framework.

## Base Class Architecture

Yohou's base classes bridge scikit-learn's estimator protocol with time series
operations. The key addition is the **observe/rewind lifecycle**: after fitting,
a forecaster can receive new observations via `observe()`, update its internal
state, and produce updated predictions. `rewind()` rolls back to the last
checkpoint. This lifecycle is managed entirely by the base class. Your custom implementation only needs to override `fit` (calling
`super().fit(...)` for validation and setup) and provide `_predict_one`.

### Forecasters

All forecasters inherit from a common ancestor that provides:

- **Validation**: data shape, time column presence, dtype checks
- **Panel dispatch**: automatic detection of `group__column` naming and
  per-group model management
- **Transformer composition**: `target_transformer` and `feature_transformer`
  parameters for wrapping transforms around the forecasting step
- **Observation tracking**: `_y_observed` maintains the most recent
  `observation_horizon` rows, updated by `observe()`
- **Metadata routing**: `time_weight`, `vintage_weight`, and other metadata flow through
  `set_fit_request()` / `set_score_request()` following sklearn's protocol

The three forecaster base classes (`BasePointForecaster`,
`BaseIntervalForecaster`, `BaseClassProbaForecaster`) share this machinery.
They differ in prediction output format: a single-value DataFrame, a
lower/upper bound DataFrame per coverage rate, or a probability distribution
DataFrame per class.

### Transformers

`BaseTransformer` extends sklearn's transformer protocol with:

- **Observation horizon**: stateful transformers declare how many past rows they
  need via the `observation_horizon` property. The pipeline respects this when
  slicing data during streaming prediction.
- **Inverse transform with warmup**: `_inverse_transform(X_t, X_p)` receives
  both the transformed data and warmup rows, enabling stateful reversal of
  operations like differencing.
- **Feature name tracking**: `get_feature_names_out()` propagates column names
  through composition chains.

Transformers declare their nature via tags: `stateful` (needs observation
history), `invertible` (supports `inverse_transform`).

## The Tag System

Tags are class-level dictionaries that describe component capabilities. The base
class merges `_tags` from all classes in the MRO, with the most-derived class
winning on conflicts. This means a subclass can override a parent's tag without
modifying the parent.

```python
class MyForecaster(BasePointForecaster):
    _tags = {"ignores_exogenous": True, "stateful": True}
```

Tags serve multiple purposes:

- **Validation shortcircuits**: a forecaster tagged `ignores_exogenous=True` skips
  exogenous feature processing entirely, avoiding unnecessary work.
- **Composition decisions**: pipelines use `stateful` and `observation_horizon` to
  determine how much historical data to keep in memory.
- **Discovery**: `all_estimators(type_filter=...)` uses tags to find components
  matching specific criteria.
- **Test generation**: the check generators inspect tags to include or skip
  checks (for example, interval-specific checks run only for interval
  forecasters).

For cases where tags depend on constructor parameters or child estimators (common
in reduction forecasters or ensembles), override `__sklearn_tags__()` as a method
instead of using the class-level `_tags` dict.

## Integration Packages

Yohou uses a workspace packages pattern for integrations that bring in heavy
or specialized dependencies. These are separate Python packages that live in
the `packages/` directory of the repository and depend on yohou as a core
library. Keeping them separate avoids bloating yohou's dependency tree with
large frameworks like PyTorch or Optuna that most users do not need.

**yohou-optuna** (`packages/yohou-optuna/`) integrates
[Optuna](https://optuna.org/) for Bayesian hyperparameter optimization. It
provides search classes that follow the same API as yohou's built-in
[`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/)
and
[`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/),
so switching between grid search and Optuna-based search requires changing only
the search object. This API compatibility is possible because all search classes
inherit from the same
[`BaseSearchCV`](/pages/api/generated/yohou.model_selection.search.BaseSearchCV/)
base class and implement a single `_run_search` method.

**yohou-nixtla** (`packages/yohou-nixtla/`) wraps the
[Nixtla](https://nixtla.io/) ecosystem (statsforecast, mlforecast, and
neuralforecast) as yohou-compatible forecasters. This gives access to
classical statistical models (ARIMA, ETS), ML models (LightGBM via mlforecast),
and deep learning models (N-BEATS, PatchTST) through yohou's standard
fit/predict interface.

Both packages follow yohou's API conventions: they accept polars DataFrames,
support panel data, participate in metadata routing, and work with yohou's
cross-validation and scoring infrastructure. This consistency is the main
benefit of the extension pattern over using those libraries directly.

## Connections

The extension architecture mirrors scikit-learn's estimator protocol deliberately.
If you have written a custom sklearn estimator, the patterns are similar: declare
parameters in `__init__`, store them as attributes, implement the core methods.
The additions are temporal: `observation_horizon`, `observe`, `rewind`, and panel
dispatch are specific to time series.

The [Custom Estimator Reference](/pages/api/custom-estimators/) provides the
complete API for all component types, including code templates and test
generators. The [How to Create a Custom Point Forecaster](/pages/how-to/custom-estimators/)
walks through a concrete forecaster example from start to finish, and the
[How to Create a Custom Scorer](/pages/how-to/creating-a-scorer/) covers
implementing custom evaluation metrics.
