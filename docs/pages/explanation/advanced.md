# Advanced Topics

This page covers the internal mechanisms and extension points that sit beneath
yohou's user-facing API. Understanding these topics is not required for basic
usage, but they become important when composing complex forecasting pipelines,
routing metadata through nested estimators, or extending yohou with new
integrations.

## Metadata Routing

Yohou enables scikit-learn's metadata routing globally the moment you
`import yohou`. This is not optional; it happens in `__init__.py` via
`sklearn.set_config(enable_metadata_routing=True)`. The reason is that yohou
introduces methods that sklearn does not know about (`observe`, `rewind`,
`predict_interval`, `predict_class_proba`, and their composites), and these
methods need to participate in sklearn's metadata routing infrastructure for
pipelines and search objects to work correctly.

Specifically, yohou registers seven additional methods as routable:

- `observe_transform` (composite: `observe` + `transform`)
- `rewind_transform` (composite: `rewind` + `transform`)
- `observe_predict` (composite: `observe` + `predict`)
- `predict_interval` (simple method)
- `observe_predict_interval` (composite: `observe` + `predict_interval`)
- `predict_class_proba` (simple method)
- `observe_predict_class_proba` (composite: `observe` + `predict_class_proba`)

The "composite" designation matters. When sklearn routes metadata for
`observe_predict`, it knows to split the incoming parameters and forward them to
both `observe` and `predict` individually. This means a `time_weight` or
`vintage_weight` parameter requested by a forecaster's `predict` method will
arrive correctly even when the caller uses `observe_predict` through a search
object or pipeline.

The practical effect: when
[`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/)
calls `observe_predict` during cross-validation, parameters like `time_weight`
and `vintage_weight` flow through to the right method on the right estimator
without manual intervention. This is the same mechanism sklearn uses for
`fit_transform` and `fit_predict`, extended to yohou's time series operations.

For a complete overview of weight types and formats, see
[Weighting](weighting.md).

## State Propagation in Composite Estimators

Yohou's [composition patterns](composition.md) let you build complex forecasting
pipelines from simpler components. Under the hood, these composites must
propagate `observe` and `rewind` operations to all sub-estimators while
maintaining correct state and data flow.

Every forecaster maintains two observation buffers: `_y_observed` and
`_X_observed`. Calling `observe()` appends new data to these buffers (with time
continuity validation), while `rewind()` replaces them with the last
`observation_horizon` rows without validation, allowing arbitrary reset windows.
Composite estimators build on this by dispatching these operations to their
sub-components in patterns that mirror how `fit` and `predict` flow.

[`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/)
sequences in the same order as training. When `observe()` is called, it
transforms the incoming data through any target/feature transformers, then
iterates through forecasters: each one predicts, computes residuals, and
observes those residuals. This preserves the additive decomposition contract
where each stage works on what previous stages left behind.

[`ColumnForecaster`](/pages/api/generated/yohou.compose.column_forecaster.ColumnForecaster/)
dispatches by column subset. Each sub-forecaster observes only its assigned
columns of `y`, but all receive the full `X` unmodified. Column splitting
applies exclusively to the target.

[`ForecastedFeatureForecaster`](/pages/api/generated/yohou.compose.forecasted_feature_forecaster.ForecastedFeatureForecaster/)
chains in two stages. The feature forecaster observes `X` as its target (it has
no exogenous data of its own), then the target forecaster observes `y` with `X`.
This maintains the two-stage relationship: the feature forecaster learns to
predict exogenous features, and the target forecaster uses those features.

## The Reduction Architecture

The [Forecasting](forecasting.md) page explains how reduction strategies
(multi-output, direct, dir-rec) convert time series into supervised learning
problems. This section covers the internal architecture that makes that possible.

[`BaseReductionForecaster`](/pages/api/generated/yohou.base.reduction.BaseReductionForecaster/)
sits between yohou's base classes and the concrete
[`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/)
and
[`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/).
It handles the shared machinery that both prediction types need: tabularization,
strategy dispatch, weight alignment, and recursive prediction. The concrete
classes add only their prediction-type-specific logic on top.

### Weight to Sample Weight Conversion

Time weights are defined per-timestamp, but tabularized training samples span
multiple timestamps. The `sample_weight_alignment` parameter controls how
per-timestamp weights collapse into per-sample weights:

- `"first_step"`: use the weight of the first target timestamp
- `"mean_step"`: average across all target timestamps
- `"weighted_mean_step"`: weighted average favoring earlier timestamps
- `"max_weight_step"`: use the maximum weight across target timestamps
- `"min_weight_step"`: use the minimum weight

The choice matters when time weights vary sharply. `"first_step"` is fastest but
ignores later timestamps entirely. `"mean_step"` smooths out local variation,
which may or may not be desirable depending on whether the weight signal is meant
to emphasize specific events or broad regions.

`vintage_weight` follows a simpler path: each vintage maps 1:1 to tabularized
samples (one row per forecast origin), so no alignment is needed.
`vintage_weight` and `time_weight` are combined multiplicatively into a single
`sample_weight` array before calling `estimator.fit()`.

### Recursive Prediction

When the requested forecasting horizon exceeds the horizon the model was trained
on, `BaseReductionForecaster` applies itself recursively. It creates a deep copy
of the forecaster, then loops in steps of `fit_forecasting_horizon_`: predict one
chunk, observe those predictions to advance the state, predict the next chunk,
and so on until the full horizon is covered. The final output is trimmed to the
exact requested length.

This means any reduction forecaster can produce arbitrarily long forecasts,
but accuracy typically degrades as errors compound across recursive steps.
Models trained with longer horizons reduce the number of recursive steps needed.

## Extending Yohou

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

## Discovery API

The `yohou.utils.discovery` module provides programmatic introspection of all
registered components. This is the same machinery that powers the documentation
system's auto-generated API pages, and it is useful for building registries,
running systematic tests across all estimators, or creating dynamic UIs.

[`all_estimators()`](/pages/api/generated/yohou.utils.discovery.all_estimators/)
crawls the package and returns all non-abstract `BaseEstimator` subclasses.
It accepts a `type_filter` parameter to select specific types:
`"forecaster"`, `"point"`, `"interval"`, `"class_proba"`, `"transformer"`,
`"splitter"`, `"scorer"`, `"point_scorer"`, `"interval_scorer"`,
`"class_proba_scorer"`, or `"conformity_scorer"`.

The filtering works through sklearn's tag system. Each estimator's
`__sklearn_tags__()` declares its type, and the discovery function reads those
tags to classify components. This means extension packages like yohou-optuna
and yohou-nixtla are automatically discoverable as long as they are installed
and their estimators properly declare tags.

[`all_displays()`](/pages/api/generated/yohou.utils.discovery.all_displays/)
and
[`all_functions()`](/pages/api/generated/yohou.utils.discovery.all_functions/)
complement `all_estimators()` by covering display classes and public functions
respectively.

**See also**: [Core Concepts](core-concepts.md) for the base class hierarchy
and time series method lifecycle. [Composition and Pipelines](composition.md)
for the user-facing composition API. [Forecasting](forecasting.md) covers the
reduction approach in more detail. [Model Selection](model-selection.md)
explains cross-validation and hyperparameter search.
[Class-Probability Forecasting](class-probability-forecasting.md) covers
categorical prediction.
