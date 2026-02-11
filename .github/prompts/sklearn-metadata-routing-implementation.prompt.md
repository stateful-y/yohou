---
description: "Complete guide to sklearn metadata routing in Yohou. Covers bootstrap behavior, composite methods, router vs consumer pattern. Use when working with metadata propagation in pipelines."
---

# sklearn Metadata Routing Implementation

## Bootstrap Behavior

Yohou **automatically enables** sklearn metadata routing on import (`src/yohou/__init__.py`):

```python
from sklearn import set_config
set_config(enable_metadata_routing=True)  # Global state change
```

### Registered Composite Methods

4 custom composite methods are registered (NOT just 2):

```python
SIMPLE_METHODS.extend([
    "update_transform",
    "update_predict",
    "predict_interval",
    "update_predict_interval",
])
COMPOSITE_METHODS["update_transform"] = ["update", "transform"]
COMPOSITE_METHODS["update_predict"] = ["update", "predict"]
COMPOSITE_METHODS["update_predict_interval"] = ["update", "predict_interval"]
```

**Critical**: `update()` is NOT a routable method—metadata only flows to the data-processing methods (`transform`, `predict`, `predict_interval`).

---

## Data vs Metadata

| Parameter | Routed? | Rationale |
|-----------|---------|-----------|
| `y` (pl.DataFrame) | ❌ | Primary target data |
| `X` (pl.DataFrame) | ❌ | Exogenous features |
| `forecasting_horizon` (int) | Explicit param | Can be routed |
| `time_weight` (Callable/DataFrame) | ✅ | Time-based weighting (consumed by reduction forecasters and scorers) |
| Custom params | ✅ | User-defined metadata |

---

## Router vs Consumer Pattern

| Class | Role | Overrides `get_metadata_routing()`? |
|-------|------|-------------------------------------|
| `BaseTransformer` | Consumer | Rarely (only if wraps estimators) |
| `BaseForecaster` | Router | Usually (wraps transformers/estimators) |
| `BaseScorer` | Consumer | Never |

---

## Implementation Status

### ✅ Completed

- MetadataRouter `owner=self` (not class name) — 6 files
- `_raise_for_params()` validation in all meta-estimators
- Direct Bunch access pattern: `routed_params[name].fit`
- Worker functions accept `params: Bunch` argument
- Removed all `if _routing_enabled():` conditionals (routing always on)
- 31 passing tests in `tests/test_metadata_routing.py`

### ⚠️ Deferred

- `transform_input` feature (advanced pipeline metadata transformation)
- `_get_metadata_for_step()` helper (using simpler direct Bunch access)

---

## Key Design Decisions

1. **Routing always enabled**: No conditionals—`set_config()` on import
2. **No `transform_input`**: Direct parameter extraction chosen over complex transformation
3. **`update()` no `**params`**: Memory management only, not data processing
4. **Direct Bunch access**: `routed_params[name].fit` over helper methods
5. **Explicit `time_weight`**: Declared in signature for API discoverability

## Test Utilities (`src/yohou/testing/metadata_routing.py`)

```python
from yohou.testing.metadata_routing import record_metadata, check_recorded_metadata

estimator = record_metadata(MyForecaster(), "fit")
estimator.fit(y, X, sample_weight=[...])
check_recorded_metadata(estimator, "fit", sample_weight=[...])
```

6 utilities: `record_metadata`, `record_metadata_not_default`, `_Registry`, `check_recorded_metadata`, `assert_request_is_empty`, `assert_request_equal`
