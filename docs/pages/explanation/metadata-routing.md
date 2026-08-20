# Metadata Routing

Metadata routing is the mechanism by which parameters like `coverage_rates` and
`groups` flow from a top-level call (such as
[`GridSearchCV.fit()`](/pages/api/generated/yohou.model_selection.GridSearchCV/))
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
- **`stride`**: the walk-forward cadence of the observe-predict loop,
  declarable on each response method (`predict`, `predict_interval`,
  `predict_class_proba`) so a search can route it into its inner loop.
- **`strategy`** (`predict_interval` only) and **`predict_transformed`**
  (`predict` only): routable where the family's walk-forward accepts them.

Any consumer can additionally request its own arbitrary metadata key; the
routing infrastructure is generic and not limited to the parameters above.

### The three-layer model

A key is routable end to end only when three independent layers line up:

1. **Declarable**: the key exists on the callee's request, either scraped
   from the method signature or declared with a `__metadata_request__*`
   class attribute (how `stride` exists on methods whose signature never
   takes it).
2. **Routed**: a router maps the calling method onto that callee. The search
   maps `fit` onto `fit` and all three response methods, so fit-passed
   metadata can reach whichever bucket the scorers resolve.
3. **Consumed**: something reads the bucket into a method that accepts the
   key. The search's inner loop reads the response-method bucket and splats
   it into the dispatched observe method.

A key failing any layer silently goes nowhere, which is why the effective
per-class key sets are pinned by a conformance test.

### The substring contract

sklearn resolves `__metadata_request__{method}` class attributes by substring
over mangled attribute names, so a declaration for `predict_interval` also
places its keys (unset) on the `predict` of any class that defines one. yohou
treats this as a stated contract rather than an accident: the pinned tables in
the conformance suite record every effective key, leaks included, and the
request discipline below pairs them off.

### The per-callee request discipline

A key passed to a search's `fit` is validated against every mapped callee
that carries it. Set the request `True` on the response method your scorers
resolve and explicitly `False` on every other carrier:

```python
forecaster.set_predict_interval_request(stride=True)   # interval scoring
forecaster.set_predict_request(stride=False)           # leaked sibling carrier
search.fit(y, forecasting_horizon=48, stride=24)
```

An unpaired carrier raises an error naming the method and the
`set_*_request` call that resolves it. `cross_validate` and
`cross_val_score` deliberately bypass routing: they take `predict_stride`
and `predict_forecasting_horizon` as explicit parameters, so no request
declarations are involved on that path.

### Known limits in yohou-optuna

Two follow-ups live in the external `yohou-optuna` package rather than here:
its per-trial objective re-splits without the routed splitter bucket (the
grid searches pass it; the bucket is empty today, so the divergence is
latent until a splitter gains a real metadata key), and its fit would raise
reading the scorer bucket under `scoring=None` (unreachable from callers
that always configure scoring).

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
[`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.IntervalReductionForecaster/)
is a consumer of `coverage_rates`, which it uses directly in its
`predict_interval` method, and at the same time a router that forwards `fit`
metadata to its wrapped sklearn estimator.

### Consumers

| Class | Methods | Accepted metadata |
|---|---|---|
| Wrapped sklearn estimator (e.g. `Ridge`) | `fit` | `sample_weight` |
| [`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.IntervalReductionForecaster/) | `fit`, `predict_interval` | `coverage_rates` |
| Panel forecasters | `predict`, `observe_predict` | `groups` |
| Pipeline transformers / custom consumers | `fit`, `transform` | any requested key |

### Routers

| Router | Children | Routed methods |
|---|---|---|
| [`GridSearchCV`](/pages/api/generated/yohou.model_selection.GridSearchCV/) / [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.RandomizedSearchCV/) | forecaster, scorer, splitter | `fit`, `predict`, `predict_interval`, `predict_class_proba`, `observe_predict`, `observe_predict_interval`, `observe_predict_class_proba`, `score`, `split` |
| [`DecompositionPipeline`](/pages/api/generated/yohou.compose.DecompositionPipeline/) | named sub-forecasters, `target_transformer`, `actual_transformer` | `fit`, `predict`, `observe_predict`, `transform` |
| [`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/) | sequential steps | `fit`, `fit_transform`, `transform`, `inverse_transform`, `score` (final step only) |
| [`ColumnTransformer`](/pages/api/generated/yohou.compose.ColumnTransformer/) | per-column transformers | `fit`, `fit_transform`, `transform` |
| [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.LocalPanelForecaster/) | wrapped forecaster | `fit`, `predict`, `predict_interval`, `observe_predict`, `observe_predict_interval` |
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
[`GridSearchCV`](/pages/api/generated/yohou.model_selection.GridSearchCV/)
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
