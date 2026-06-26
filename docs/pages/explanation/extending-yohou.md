# Extending Yohou

Yohou is designed to be extended with custom forecasters, transformers, scorers,
and splitters. This page explains the architecture that makes extension possible
and the contract each extension point enforces.

## Composing vs Subclassing

Yohou exposes two ways to add new behavior: composition using built-in
container classes, and extension by subclassing a base class. These are not
alternatives competing for the same use cases; they represent different levels
of the abstraction hierarchy.

**Composition** tools ([`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/), [`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/), [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/),
[`VotingPointForecaster`](/pages/api/generated/yohou.ensemble.voting_point.VotingPointForecaster/)) assemble existing components into new configurations.
They delegate all framework concerns (observation tracking, panel dispatch, temporal
structure) to the components they contain. The user provides a regressor or a list of
forecasters; the container handles the rest. Composition works when the algorithm
can be expressed as a scikit-learn estimator applied to tabularized features, or as a
combination of existing forecasters.

**Subclassing** a base class inserts entirely new algorithms into the framework. A custom
[`BasePointForecaster`](/pages/api/generated/yohou.point.base.BasePointForecaster/) subclass can implement a state-space model, exponential smoothing,
or any other method that cannot be expressed as regression on a feature matrix. A custom
[`BaseTransformer`](/pages/api/generated/yohou.base.transformer.BaseTransformer/) can implement temporal logic with a fixed lookback window that the
framework tracks. A custom scorer can aggregate error in a domain-specific way. In each
case, the subclass implements a small set of methods; the base class enforces the
contract that makes the component interoperable with the rest of the system.

Understanding which layer to work at shapes how much code you write and what the
framework manages on your behalf. Composing is faster and yields full framework
compatibility automatically. Subclassing gives complete algorithmic control at the cost
of implementing more methods and maintaining compatibility explicitly.

## The Observe/Rewind Lifecycle

The central addition Yohou makes to Scikit-Learn's estimator protocol is the
**observe/rewind lifecycle**. After fitting, a forecaster (or transformer) can
receive new observations via `observe()`, update its internal state, and produce
updated predictions without refitting. `rewind()` rolls back to the state before
the last `observe()` call.

This lifecycle enables rolling-window evaluation and streaming use cases. The
base class manages the `_y_observed` buffer (bounded by `observation_horizon`)
and propagates `observe()` and `rewind()` calls to nested transformers
automatically. A custom subclass does not need to implement these methods unless
it maintains additional internal state beyond what the base class tracks.

## Extension Points

### Forecasters

All forecasters inherit from a common `BaseForecaster` that provides:

- **Validation**: data shape, time column presence, dtype checks.
- **Panel dispatch**: automatic detection of `group__column` naming and
  per-group transformer management (see [Panel Dispatch](#panel-dispatch) below).
- **Transformer composition**: `target_transformer` and `feature_transformer`
  parameters for wrapping transforms around the forecasting step.
- **Observation tracking**: `_y_observed` maintains the most recent
  `observation_horizon` rows, updated by `observe()`.
- **Metadata routing**: `sample_weight` (resolved from the `time_weighter` and
  `vintage_weighter` constructor parameters by the framework) flows through
  `set_fit_request()` following sklearn's protocol. Weighters are configured at
  construction time, not passed as metadata (see [Weighting](weighting.md)).

The three forecaster base classes ([`BasePointForecaster`](/pages/api/generated/yohou.point.base.BasePointForecaster/),
[`BaseIntervalForecaster`](/pages/api/generated/yohou.interval.base.BaseIntervalForecaster/), [`BaseClassProbaForecaster`](/pages/api/generated/yohou.class_proba.base.BaseClassProbaForecaster/)) share this machinery.
They differ in prediction output format: a single-value DataFrame, a
lower/upper bound DataFrame per coverage rate, or a probability distribution
DataFrame per class.

A subclass must implement `_predict_one(groups, **params)`, which produces
predictions from the current observation horizon. Calling `super().fit(...)` in
`fit()` sets up all fitted attributes: `interval_` (detected time interval),
`fit_forecasting_horizon_`, `groups_` (panel group names), column schemas, and
transformer instances.

### Transformers

[`BaseTransformer`](/pages/api/generated/yohou.base.transformer.BaseTransformer/) extends sklearn's transformer protocol with:

- **Observation horizon**: stateful transformers declare how many past rows they
  need via the `observation_horizon` property. Pipelines respect this when
  slicing data during streaming prediction.
- **Inverse transform with warmup**: `_inverse_transform(X_t, X_p)` receives
  both the transformed data and warmup rows, enabling stateful reversal of
  operations like differencing.
- **Feature name tracking**: `get_feature_names_out()` propagates column names
  through composition chains.

A subclass implements `_fit(X)` and `_transform(X)`. If the transformer is
invertible, it also implements `_inverse_transform(X)`. Stateful transformers
set `self._observation_horizon` during fit (typically to a value derived from
constructor parameters such as `window_size` or `seasonality`). Stateless
transformers leave it at the default of zero, which causes `observe()` and
`rewind()` to be no-ops.

### Scorers

The scorer hierarchy ([`BasePointScorer`](/pages/api/generated/yohou.metrics.base.BasePointScorer/),
[`BaseIntervalScorer`](/pages/api/generated/yohou.metrics.base.BaseIntervalScorer/),
[`BaseClassProbaScorer`](/pages/api/generated/yohou.metrics.base.BaseClassProbaScorer/)) follows a template method pattern. The base class orchestrates a
multi-stage aggregation pipeline:

1. Compute raw per-timestep, per-component errors via `_compute_raw_errors()`.
2. Apply optional time, step, and vintage weights.
3. Collapse dimensions according to `aggregation_method` (`"stepwise"`,
   `"vintagewise"`, `"componentwise"`, `"groupwise"`, and for interval scorers,
   `"coveragewise"`).

A custom scorer subclass only needs to implement `_compute_raw_errors(y_truth, y_pred)`,
which returns a DataFrame of raw error values with the same shape as the inputs.
Everything else (weighting, aggregation, panel handling) is managed by the base class.

### Splitters

[`BaseSplitter`](/pages/api/generated/yohou.model_selection.split.BaseSplitter/) provides the interface for time series cross-validation strategies.
Built-in splitters include `ExpandingWindowSplitter` (growing train set) and
`SlidingWindowSplitter` (fixed-size sliding window).

A custom splitter implements three methods: `split(y, X_actual)` (yields
train/test index tuples), `_iter_test_indices(y, X_actual)` (generates test
indices), and `get_n_splits(y, X_actual)` (returns the number of folds).

## The Tag System

Tags are structured dataclass attributes that describe component capabilities.
They drive validation shortcuts, composition decisions, and test generation.
Each estimator exposes its tags through `__sklearn_tags__()`, which returns a
`Tags` object with nested dataclasses for each component type.

For example, a forecaster that does not use exogenous features and is
intrinsically stateful declares:

```python
class MyForecaster(BasePointForecaster):
    _tags = {"requires_exogenous": False, "stateful": True}
```

Tags are organized into groups:

- **ForecasterTags**: `forecaster_type`, `stateful`, `uses_reduction`,
  `supports_panel_data`, `supports_time_weight`, `requires_exogenous`,
  `tracks_observations`, and others.
- **TransformerTags**: `stateful`, `invertible`, `preserves_dtype`.
- **ScorerTags**: `prediction_type`, `lower_is_better`, `requires_calibration`.
- **SplitterTags**: `splitter_type`, `supports_panel_data`,
  `produces_non_overlapping_tests`.
- **InputTags**: `requires_time_column`, `allow_nan`.

### Tag Resolution via MRO

Tags follow Python's Method Resolution Order. The `__sklearn_tags__()` method
collects `_tags` dictionaries from every class in the hierarchy, starting from
the most-base class and working toward the most-derived one. When the same key
appears in multiple classes, the most-derived class wins.

```python
class Base(BasePointForecaster):
    _tags = {"stateful": False, "requires_exogenous": False}

class Child(Base):
    _tags = {"stateful": True}  # overrides Base; requires_exogenous inherited
```

`Child` ends up with `stateful=True` (overridden) and
`requires_exogenous=False` (inherited from `Base`). No manual merging is needed.

Some tags are computed dynamically. For instance, the base forecaster
automatically sets `stateful=True` when `target_transformer` or
`feature_transformer` has `stateful=True` in its own tags. A subclass that is
intrinsically stateful (independent of its transformers) overrides
`__sklearn_tags__()` to set `forecaster_tags.stateful = True` directly.

For the full list of available tags and how they interact with discovery and
testing, see [Tags](../reference/tags.md).

## Parameter Constraints

Every estimator can declare a `_parameter_constraints` dictionary that maps
constructor parameter names to lists of valid types or validators. These
constraints are checked automatically at the start of `fit()`.

```python
from sklearn.utils._param_validation import Interval, StrOptions

class MyForecaster(BasePointForecaster):
    _parameter_constraints = {
        "window_size": [Interval(int, 1, None, closed="left")],
        "strategy": [StrOptions({"mean", "last"})],
    }
```

Constraints merge along the MRO via `__init_subclass__()`, so a subclass
inherits all parent constraints and can override specific entries. This gives
every extension automatic input validation without writing manual checks.

## Panel Dispatch

Panel data (multiple related time series) is identified by the `group__column`
naming convention in DataFrame columns. The base forecaster detects panel groups
at fit time and manages per-group state automatically.

When `supports_panel_data=True` (the default for forecasters):

1. **Fit**: the base class detects panel groups from column names, fits
   transformers independently per group, pools the transformed data, and fits
   the estimator on the pooled result.
2. **Predict**: per-group transformers are applied, predictions are generated,
   and output is reconstructed with panel structure.
3. **Observe/Rewind**: observation buffers and transformer states are managed
   per group.

This means a custom forecaster that implements `_predict_one()` receives
already-transformed, panel-aware data and does not need to handle group logic
itself.

## Systematic Test Suite

Each extension point has a corresponding test generator that yields a
comprehensive set of checks:

- `_yield_yohou_forecaster_checks`: 40+ checks covering fit attributes,
  prediction structure, observe/rewind, panel support, metadata routing,
  and type-specific output validation.
- `_yield_yohou_transformer_checks`: 26+ checks covering transformation
  structure, feature names, statefulness, invertibility, and memory bounding.
- `_yield_yohou_scorer_checks`: 11+ checks covering aggregation methods,
  weighting, coverage rates, and parameter validation.
- `_yield_yohou_splitter_checks`: 8+ checks covering index validity,
  fold count consistency, and panel support.

The generators inspect each estimator's tags to decide which checks apply. A
stateless transformer skips the observation buffer checks; a forecaster with
`requires_exogenous=False` skips exogenous feature tests. Running the full
generator against a custom extension validates framework contract compliance
automatically.

## Integration Pattern

Any scikit-learn compatible estimator can be passed directly to a reduction
forecaster (e.g., `PointReductionForecaster(estimator=MyRegressor())`). For
algorithms that cannot be expressed as supervised learning on tabularized
features, subclassing the appropriate base class and running the test generator
is the recommended integration path. The subclass inherits automatic
compatibility with cross-validation, scoring, composition, and panel
infrastructure.

For integrations that require heavy dependencies (e.g., PyTorch, CatBoost),
separate packages can subclass Yohou's base classes while keeping the core
dependency tree small. See [Extensions](../reference/extensions.md) for the
current list of official and community integrations.

## Connections

The extension architecture mirrors scikit-learn's estimator protocol
deliberately. If you have written a custom scikit-learn estimator, the patterns
are familiar: declare parameters in `__init__`, store them as attributes,
implement the core methods. The additions are temporal: `observation_horizon`,
`observe`, `rewind`, and panel dispatch are specific to time series.

The how-to guides cover each component type with code templates and test
generators:

- [Create a Point Forecaster](/pages/how-to/create-a-point-forecaster/)
- [Create an Interval Forecaster](/pages/how-to/create-an-interval-forecaster/)
- [Create a Class-Probability Forecaster](/pages/how-to/create-a-class-proba-forecaster/)
- [Create a Transformer](/pages/how-to/create-a-transformer/)
- [Create a Custom Scorer](/pages/how-to/create-a-scorer/)

Related explanation pages:

- [Reduction Forecasting](reduction-forecasting.md) for how composition turns regressors into forecasters
- [Forecaster Composition](forecaster-composition.md) for combining multiple forecasters
- [Panel Data](panel-data.md) for the panel dispatch architecture in depth
- [Metadata Routing](metadata-routing.md) for how weights and parameters flow through nested estimators
