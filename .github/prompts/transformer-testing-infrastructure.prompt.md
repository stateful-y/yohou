---
description: "Testing patterns for Yohou time series transformers. 21 check functions covering fit/transform contracts, memory management, tag validation. Use when writing or debugging transformer tests."
---

# Transformer Testing Infrastructure

## Check Functions — `src/yohou/testing/transformer.py` (21 total)

### Core Yohou Checks (11)

| Check | Validates |
|-------|-----------|
| `check_fit_sets_attributes` | feature_names_in_, n_features_in_, _observation_horizon set after fit() |
| `check_observation_horizon_not_fitted` | NotFittedError before fit() |
| `check_observation_horizon_after_fit` | observation_horizon is non-negative int after fit() |
| `check_reset_updates_memory` | reset() updates _X_observed to last observation_horizon rows |
| `check_update_concatenates_memory` | update() appends data and maintains horizon size |
| `check_update_transform_equivalence` | update().transform() == fit().transform() for same final state |
| `check_insufficient_data_raises` | Error when data length < observation_horizon |
| `check_transform_output_structure` | Output has "time" column and valid structure |
| `check_feature_names_out_match` | get_feature_names_out() matches output columns |
| `check_inverse_transform_identity` | inverse_transform(transform(X)) ≈ X |
| `check_panel_data_support` | Handles struct columns (panel data) correctly |

### Tag System Checks (3)

| Check | Validates |
|-------|-----------|
| `check_tags_accessible_before_fit` | __sklearn_tags__() accessible pre-fit |
| `check_tags_static_after_fit` | Tags don't change after fit() |
| `check_tags_match_capabilities` | Tags match actual behavior (stateful↔observation_horizon, invertible↔inverse_transform) |

### Enhanced sklearn Checks (7)

| Check | Validates |
|-------|-----------|
| `check_transformers_unfitted_stateless` | Stateless transformers work without fit() |
| `check_transformer_preserve_dtypes` | Dtype preservation through transform |
| `check_fit_idempotent` | Multiple fits yield consistent results |
| `check_inverse_transform_round_trip` | Enhanced round-trip validation |
| `check_fit_transform_equivalence` | fit_transform() == fit().transform() |
| `check_memory_bounded` | Memory doesn't grow unbounded |
| `check_transformer_methods_call_check_is_fitted` | Methods call check_is_fitted() |

---

## Generator Usage

```python
from yohou.testing import _yield_yohou_transformer_checks

for check_name, check_func, check_kwargs in _yield_yohou_transformer_checks(
    transformer_fitted, X_train, y_train, X_test, y_test,
    tags={"stateful": True, "invertible": False}
):
    check_func(transformer_fitted, **check_kwargs)
```

**Tags**: `stateful` (bool, observation_horizon > 0), `invertible` (bool, has inverse_transform)

---

## Key Concepts

- **Stateful vs stateless**: Stateful transformers have `observation_horizon > 0` and maintain `_X_observed` buffer
- **Memory management**: `update()` appends → `reset()` trims to last `observation_horizon` rows
- **All DataFrames**: polars with mandatory `"time"` column (both input and output)

## Test File Layout

```
tests/preprocessing/test_stationarization.py  # Differencing transformers
tests/preprocessing/test_window.py            # Window transformers (lag, rolling)
tests/test_pipeline.py                        # FeaturePipeline, FeatureUnion, ColumnTransformer
```
