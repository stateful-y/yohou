# Metadata Routing

Metadata routing is the mechanism by which parameters like `time_weight`,
`vintage_weight`, and `step_weight` flow from a top-level call (such as
[`GridSearchCV.fit()`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/))
down through nested estimators to the objects that actually use them. Without
it, there would be no way for a pipeline or search object to know which of its
child estimators should receive a given parameter.

Yohou builds on
[scikit-learn's metadata routing infrastructure](https://scikit-learn.org/stable/metadata_routing.html),
extending it with time series specific methods. Routing is enabled globally the
moment you `import yohou` (via
`sklearn.set_config(enable_metadata_routing=True)` in `__init__.py`), so there
is nothing to configure manually.

## Routable Metadata Parameters

Three weight parameters can be routed through yohou's estimator hierarchy:

- **`time_weight`**: weights observations by their position in time. Accepted as
  a callable, a DataFrame with a `"time"` column, or a dict.
- **`vintage_weight`**: weights observations by when they were first observed
  (their vintage time). Accepted in the same formats as `time_weight`.
- **`step_weight`**: weights per-step forecasting errors during scoring.
  Accepted as a callable, a DataFrame with a `"forecasting_step"` column, or a
  dict.

Reduction forecasters consume `time_weight` and `vintage_weight` during
`fit()`, converting them into a single `sample_weight` vector that is passed to
the wrapped sklearn estimator. Scorers consume all three weight types in
`score()`, applying them when aggregating errors across time, vintage, and
forecasting step dimensions.

## Consumers and Routers

Sklearn's routing model has two roles:

- A **consumer** is an object that accepts and uses metadata in one of its
  methods.
- A **router** is a meta-estimator that forwards metadata to its children
  without necessarily using it itself.

An object can be both. A
[`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/)
is a consumer (it uses `time_weight` and `vintage_weight` in `fit()`) and also
a router (it forwards `fit` metadata to its wrapped sklearn estimator via
`get_metadata_routing()`).

### Consumers

| Class | Methods | Accepted metadata |
|---|---|---|
| [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) | `fit` | `time_weight`, `vintage_weight` |
| [`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/) | `fit` | `time_weight`, `vintage_weight` |
| Point and interval scorers | `score` | `time_weight`, `vintage_weight`, `step_weight` |
| Transformers (e.g. `StandardScaler`) | `fit`, `transform` | `time_weight` |

### Routers

| Router | Children | Routed methods |
|---|---|---|
| [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) / [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/) | forecaster, scorer, splitter | `fit`, `predict`, `predict_interval`, `predict_class_proba`, `observe_predict`, `observe_predict_interval`, `observe_predict_class_proba`, `score`, `split` |
| [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) | named sub-forecasters, `target_transformer`, `feature_transformer` | `fit`, `predict`, `observe_predict`, `transform` |
| [`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/) | sequential steps | `fit`, `fit_transform`, `transform`, `inverse_transform`, `score` (final step only) |
| [`ColumnTransformer`](/pages/api/generated/yohou.compose.column_transformer.ColumnTransformer/) | per-column transformers | `fit`, `fit_transform`, `transform` |
| [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/) | wrapped forecaster | `fit`, `predict`, `predict_interval`, `observe_predict`, `observe_predict_interval` |
| `BaseReductionForecaster` | wrapped sklearn estimator | `fit` |

## The Request API

By default, no metadata is forwarded anywhere. Each consumer must explicitly
**request** the parameters it wants using `set_{method}_request()` methods. This
prevents silent misrouting: if metadata is passed to a router but no child has
requested it, sklearn raises an error.

```python
from yohou.point import PointReductionForecaster
from sklearn.linear_model import Ridge

forecaster = PointReductionForecaster(estimator=Ridge())

# Request time_weight and vintage_weight in fit
forecaster.set_fit_request(time_weight=True, vintage_weight=True)
```

The request values are:

- `True`: the method requests this parameter. If provided, it will be forwarded;
  if not provided, no error is raised.
- `False`: the method explicitly does not want this parameter, even if the caller
  provides it.
- `None` (default): the router will raise an error if this parameter is passed.
  This forces users to make an explicit choice, preventing accidental omissions.
- A **string**: an alias. The caller uses the alias name and the router remaps
  it to the parameter the consumer expects. This allows different consumers to
  receive different values for identically named parameters.

Scorers use `set_score_request()`:

```python
from yohou.metrics import MeanAbsoluteError

scorer = MeanAbsoluteError()
scorer.set_score_request(time_weight=True, vintage_weight=True, step_weight=True)
```

Transformers use `set_fit_request()` and `set_transform_request()`:

```python
from yohou.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.set_fit_request(time_weight=True)
scaler.set_transform_request(time_weight=True)
```

### Aliasing

Aliases let two consumers receive different values for a parameter that shares
the same name. For example, if a forecaster and a scorer each need a different
`time_weight`, the caller can pass them under separate names:

```python
forecaster.set_fit_request(time_weight="train_weight")
scorer.set_score_request(time_weight="eval_weight")

search.fit(y=train, train_weight=w_train, eval_weight=w_eval)
```

The router remaps `train_weight` to the forecaster's `time_weight` and
`eval_weight` to the scorer's `time_weight`.

## Yohou's Extended Method Registry

Sklearn knows how to route metadata for its own methods (`fit`, `predict`,
`transform`, `score`). Yohou introduces methods that sklearn does not know
about, so it registers them at import time by adding to sklearn's internal
method registries (`SIMPLE_METHODS`, `METHODS`, and `COMPOSITE_METHODS`).

Seven additional methods are registered as routable:

| Method | Type | Decomposes into |
|---|---|---|
| `observe_transform` | composite | `observe` + `transform` |
| `rewind_transform` | composite | `rewind` + `transform` |
| `observe_predict` | composite | `observe` + `predict` |
| `predict_interval` | simple | |
| `observe_predict_interval` | composite | `observe` + `predict_interval` |
| `predict_class_proba` | simple | |
| `observe_predict_class_proba` | composite | `observe` + `predict_class_proba` |

The composite decomposition is what makes this work seamlessly. When
[`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/)
calls `observe_predict` during cross-validation, sklearn's routing
infrastructure splits the incoming parameters and forwards them to both
`observe` and `predict` individually. A `time_weight` parameter requested by a
forecaster's `predict` method will arrive correctly even when the caller uses
`observe_predict`. This is the same mechanism sklearn uses for `fit_transform`
and `fit_predict`, extended to yohou's time series operations.

Note that `observe` itself is not independently routable. It is a memory
management operation that only participates in routing as part of composite
methods.

## How Routers Forward Metadata

Each yohou router implements a `get_metadata_routing()` method that defines a
routing table mapping caller methods to callee methods on its children. When a
router receives a method call with extra parameters, it calls
`process_routing()` to look up which child requested what and dispatches
accordingly.

For example, when `GridSearchCV.fit()` is called with `time_weight=weights`,
the flow is:

1. `process_routing(self, "fit", time_weight=weights)` inspects the routing
   table.
2. It finds that the forecaster requested `time_weight` in `fit` and the scorer
   requested it in `score`.
3. It returns a dictionary keyed by child name, with each child's parameters
   grouped by method.
4. The router calls each child's method with the appropriate subset of
   parameters.

If a parameter is passed but no child has requested it, `process_routing()`
raises an error. If a child requested a parameter but the caller did not
provide it, the child simply does not receive it (no error).

## Putting It Together

A complete example showing metadata flowing through a search object:

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster
from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import GridSearchCV, ExpandingWindowSplitter

forecaster = PointReductionForecaster(estimator=Ridge())
forecaster.set_fit_request(time_weight=True)

scorer = MeanAbsoluteError()
scorer.set_score_request(time_weight=True)

search = GridSearchCV(
    forecaster=forecaster,
    param_grid={"observation_horizon": [5, 10]},
    cv=ExpandingWindowSplitter(n_splits=3),
    scoring=scorer,
)

# time_weight flows to both forecaster.fit() and scorer.score()
search.fit(y=train, time_weight=weights)
```

Inside `PointReductionForecaster.fit()`, the received `time_weight` is
converted into a `sample_weight` vector and forwarded to the sklearn
estimator's `fit()` method. Inside `MeanAbsoluteError.score()`, the same
`time_weight` is used to produce a weighted average of the errors across time.

If `set_fit_request(time_weight=True)` were omitted, `GridSearchCV` would raise
an error explaining that `time_weight` was passed but not explicitly requested
by the forecaster. This fail-safe ensures metadata never silently disappears.

## Connections

[Core Concepts](core-concepts.md) covers the base class hierarchy, the
observe/rewind lifecycle, and the sklearn bridge that underpins metadata
routing. [Forecaster Composition](forecaster-composition.md) explains how
`observe` and `rewind` propagate through composite forecasters and how state is
managed in pipelines. [Model Selection](model-selection.md) describes
cross-validation and hyperparameter search, where metadata routing ensures
parameters reach the right estimators. The weight types (`time_weight`,
`vintage_weight`, `step_weight`) that flow through the routing infrastructure
are discussed in [Weighting](weighting.md). [Extending Yohou](extending-yohou.md)
covers how custom components participate in the routing infrastructure through
tags and base class conventions.

For practical recipes on routing weights through search and composition, see
[How to Tune Hyperparameters](../how-to/tune-hyperparameters.md) and
[How to Use Time Weighting](../how-to/time-weighting.md).
