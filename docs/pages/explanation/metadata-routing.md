# Metadata Routing

Metadata routing is the mechanism by which parameters like `coverage_rates` and
`groups` flow from a top-level call (such as
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

The metadata a caller can supply at a top-level call includes:

- **`coverage_rates`**: the list of interval coverage levels, routed to an
  interval forecaster's `predict_interval` (and `fit`).
- **`groups`**: panel group names, used by `predict`/`observe_predict` to
  operate on a subset of panel groups.

Any consumer can additionally request its own arbitrary metadata key; the
routing infrastructure is generic and not limited to the parameters above.

### `sample_weight` is not a caller-supplied parameter

`sample_weight` rides the same machinery but you never pass it. A reduction
forecaster resolves its configured weighters (`time_weighter`,
`vintage_weighter`) into a `sample_weight` array, wires the request on its
wrapped estimator, and forwards the array so it reaches the estimator's `fit`.
It is produced and consumed entirely inside the framework. To weight training,
configure a weighter on the forecaster's `__init__` rather than routing
`sample_weight` by hand (see [Weighting](weighting.md)).

## Consumers and Routers

Sklearn's routing model has two roles:

- A **consumer** is an object that accepts and uses metadata in one of its
  methods.
- A **router** is a meta-estimator that forwards metadata to its children
  without necessarily using it itself.

An object can be both. An
[`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/)
is a consumer of `coverage_rates`, which it uses directly in its
`predict_interval` method, and at the same time a router that forwards `fit`
metadata to its wrapped sklearn estimator.

### Consumers

| Class | Methods | Accepted metadata |
|---|---|---|
| Wrapped sklearn estimator (e.g. `Ridge`) | `fit` | `sample_weight` |
| [`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/) | `fit`, `predict_interval` | `coverage_rates` |
| Panel forecasters | `predict`, `observe_predict` | `groups` |
| Pipeline transformers / custom consumers | `fit`, `transform` | any requested key |

### Routers

| Router | Children | Routed methods |
|---|---|---|
| [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) / [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/) | forecaster, scorer, splitter | `fit`, `predict`, `predict_interval`, `predict_class_proba`, `observe_predict`, `observe_predict_interval`, `observe_predict_class_proba`, `score`, `split` |
| [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) | named sub-forecasters, `target_transformer`, `actual_transformer` | `fit`, `predict`, `observe_predict`, `transform` |
| [`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/) | sequential steps | `fit`, `fit_transform`, `transform`, `inverse_transform`, `score` (final step only) |
| [`ColumnTransformer`](/pages/api/generated/yohou.compose.column_transformer.ColumnTransformer/) | per-column transformers | `fit`, `fit_transform`, `transform` |
| [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/) | wrapped forecaster | `fit`, `predict`, `predict_interval`, `observe_predict`, `observe_predict_interval` |
| `BaseReductionForecaster` | wrapped sklearn estimator | `fit` |

## The Request API

By default, no metadata is forwarded anywhere. Each consumer must explicitly
**request** the parameters it wants using `set_{method}_request()` methods. This
prevents silent misrouting: if metadata is passed to a router but no child has
requested it, sklearn raises an error.

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

Each routable method has its own request setter. An interval forecaster, for
example, requests `coverage_rates` on its `predict_interval` method:

```python
from sklearn.linear_model import Ridge
from yohou.interval import IntervalReductionForecaster

forecaster = IntervalReductionForecaster(estimator=Ridge())
forecaster.set_predict_interval_request(coverage_rates=True)
```

Transformers that consume a metadata key expose both `set_fit_request()` and
`set_transform_request()`, so the key can be requested independently in each
method.

### Aliasing

Aliases let two consumers receive different values for a parameter that shares
the same name. For example, if two consumers each need a different `my_metadata`,
the caller can pass them under separate names:

```python
consumer_a.set_fit_request(my_metadata="meta_a")
consumer_b.set_fit_request(my_metadata="meta_b")

router.fit(X, y, meta_a=value_a, meta_b=value_b)
```

The router remaps `meta_a` to `consumer_a`'s `my_metadata` and `meta_b`
to `consumer_b`'s `my_metadata`.

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
`observe` and `predict` individually. A `groups` parameter requested by a
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

For example, when `GridSearchCV.fit()` is called with
`coverage_rates=[0.8, 0.95]`, the flow is:

1. `process_routing(self, "fit", coverage_rates=...)` inspects the routing table.
2. It finds that the interval forecaster requested `coverage_rates` in `fit`.
3. It returns a dictionary keyed by child name, with each child's parameters
   grouped by method.
4. The router calls each child's method with the appropriate subset of
   parameters.

If a parameter is passed but no child has requested it, `process_routing()`
raises an error. If a child requested a parameter but the caller did not
provide it, the child simply does not receive it (no error).

## Putting It Together

On every nested call the three pieces combine: each consumer's **request**
declares what it wants, the router's `get_metadata_routing()` defines the
**table**, and `process_routing()` performs the **dispatch**. The clearest way to
see them work as a unit is to follow the one parameter that travels this
machinery in everyday use: the `sample_weight` a reduction forecaster forwards to
its wrapped estimator. The forecaster drives the dispatch; the caller supplies no
metadata.

```text
forecaster.fit(y, forecasting_horizon=7)         ← caller passes no metadata kwarg
      │
      │  forecaster resolves its time_weighter → sample_weight array
      │  estimator.set_fit_request(sample_weight=True)        (the request)
      ▼
process_routing(self, "fit", sample_weight=…)                (the dispatch)
      │  consults the routing table from get_metadata_routing()
      ▼
Ridge.fit(X_tab, y_tab, sample_weight=…)                     (arrives at the consumer)
```

The same dispatch carries `coverage_rates` to an interval forecaster's
`predict_interval` and `groups` to a panel forecaster's `predict`; only the key
and the method differ.

For the full request API and routing edge cases, see
[scikit-learn's metadata routing guide](https://scikit-learn.org/stable/metadata_routing.html).

## Connections

[Core Concepts](core-concepts.md) covers the base class hierarchy, the
observe/rewind lifecycle, and the sklearn bridge that underpins metadata
routing. [Forecaster Composition](forecaster-composition.md) explains how
`observe` and `rewind` propagate through composite forecasters and how state is
managed in pipelines. [Model Selection](model-selection.md) describes
cross-validation and hyperparameter search, where metadata routing ensures
parameters reach the right estimators. Time-axis weighting, configured with
weighter estimators on `__init__` rather than routed as metadata, is discussed
in [Weighting](weighting.md), which also explains how a forecaster turns its
weighters into the `sample_weight` that flows through this infrastructure.
[Extending Yohou](extending-yohou.md) covers how custom components participate in
the routing infrastructure through tags and base class conventions.

For practical recipes on tuning and composition, see
[How to Tune Hyperparameters](../how-to/tune-hyperparameters.md) and
[How to Use Time Weighting](../how-to/time-weighting.md).
