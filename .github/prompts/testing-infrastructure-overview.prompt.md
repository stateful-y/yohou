---
description: "Overview of Yohou's systematic testing infrastructure: 86 check functions, 5 generators, 6 metadata routing utilities. Use when writing tests for any estimator."
---

# Testing Infrastructure Overview

## Quick Reference

**86 check functions** across 10 modules + **5 generators** + **6 metadata routing utilities**.

| Component | Check Functions | Generator |
|-----------|----------------|-----------|
| Transformers | 21 | `_yield_yohou_transformer_checks` |
| Forecasters | 25 (13+2+5+2+3) | `_yield_yohou_forecaster_checks` |
| Scorers | 11 | `_yield_yohou_scorer_checks` |
| Splitters | 8 | `_yield_yohou_splitter_checks` |
| Search CVs | 19 | `_yield_yohou_search_checks` |
| Common | 2 | — |
| **Total** | **86** | **5** |

---

## Check Functions by Module

### Transformer (`src/yohou/testing/transformer.py`) — 21 checks

**Core (11)**: `check_fit_sets_attributes`, `check_observation_horizon_not_fitted`, `check_observation_horizon_after_fit`, `check_reset_updates_memory`, `check_update_concatenates_memory`, `check_update_transform_equivalence`, `check_insufficient_data_raises`, `check_transform_output_structure`, `check_feature_names_out_match`, `check_inverse_transform_identity`, `check_panel_data_support`

**Tag System (3)**: `check_tags_accessible_before_fit`, `check_tags_static_after_fit`, `check_tags_match_capabilities`

**Enhanced sklearn (7)**: `check_transformers_unfitted_stateless`, `check_transformer_preserve_dtypes`, `check_fit_idempotent`, `check_inverse_transform_round_trip`, `check_fit_transform_equivalence`, `check_memory_bounded`, `check_transformer_methods_call_check_is_fitted`

### Forecaster (`src/yohou/testing/forecaster.py`) — 13 common checks

`check_fit_sets_forecaster_attributes`, `check_forecaster_not_fitted_error`, `check_predict_time_columns`, `check_update_extends_observations`, `check_reset_replaces_observations`, `check_reset_propagates_to_transformers`, `check_forecasting_horizon_validation`, `check_prediction_types_property`, `check_clone_preserves_forecaster_params`, `check_forecaster_tags_accessible_before_fit`, `check_forecaster_tags_static_after_fit`, `check_forecaster_tags_match_capabilities`, `check_forecaster_methods_call_check_is_fitted`

### Point (`src/yohou/testing/point.py`) — 2 checks

`check_point_prediction_structure`, `check_point_prediction_types`

### Interval (`src/yohou/testing/interval.py`) — 5 checks

`check_interval_prediction_columns`, `check_interval_bounds`, `check_interval_prediction_types`, `check_coverage_rates_parameter`, `check_coverage_rates_validation`

### Reduction (`src/yohou/testing/reduction.py`) — 2 checks

`check_estimator_parameter`, `check_reduction_strategy`

### Panel (`src/yohou/testing/panel.py`) — 3 checks

`check_panel_data`, `check_panel_single_group`, `check_panel_invalid_group_raises`

### Scorer (`src/yohou/testing/scorer.py`) — 11 checks

**Tag System (3)**: `check_scorer_tags_accessible_before_fit`, `check_scorer_tags_static_after_fit`, `check_scorer_tags_match_capabilities`

**Functionality (7)**: `check_scorer_prediction_type_compatibility`, `check_scorer_lower_is_better`, `check_scorer_aggregation_methods`, `check_scorer_panel_subselection`, `check_scorer_component_subselection`, `check_scorer_coverage_rate_subselection`, `check_scorer_methods_call_check_is_fitted`

**Validation (1)**: `check_scorer_parameter_validation`

### Splitter (`src/yohou/testing/splitter.py`) — 8 checks

**Tag System (3)**: `check_splitter_tags_accessible_before_fit`, `check_splitter_tags_static_after_fit`, `check_splitter_tags_match_capabilities`

**Functionality (4)**: `check_splitter_produces_valid_indices`, `check_splitter_n_splits_consistency`, `check_splitter_non_overlapping_tests`, `check_splitter_panel_data_support`

**Validation (1)**: `check_splitter_parameter_constraints`

### Search (`src/yohou/testing/search.py`) — 19 checks

**Common (12)**: `check_search_fit_sets_attributes`, `check_search_not_fitted_error`, `check_search_cv_results_structure`, `check_search_refit_false_no_forecaster`, `check_search_predict_delegates`, `check_search_update_delegates`, `check_search_reset_delegates`, `check_search_score_delegates`, `check_search_multimetric_scoring`, `check_search_return_train_score`, `check_search_error_score_handling`, `check_search_clone_preserves_params`

**Grid-specific (2)**: `check_grid_search_exhaustive`, `check_grid_search_param_grid_validation`

**Randomized-specific (3)**: `check_randomized_search_n_iter`, `check_randomized_search_reproducibility`, `check_randomized_search_distributions`

**Panel (2)**: `check_search_panel_data`, `check_search_method_availability`

### Common (`src/yohou/testing/common.py`) — 2 checks

`check_metadata_routing_default_request`, `check_metadata_routing_get_metadata_routing`

---

## Generator Pattern

All generators yield `(check_name, check_func, check_kwargs)` tuples driven by tags:

```python
from yohou.testing import _yield_yohou_forecaster_checks

for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(
    forecaster_fitted, y_train, X_train, y_test, X_test,
    tags={"forecaster_type": "point", "uses_reduction": True}
):
    if check_name not in expected_failures:
        check_func(forecaster_fitted, **check_kwargs)
```

---

## Test File Template

```python
import pytest
from sklearn.base import clone
from yohou.testing import _yield_yohou_{component}_checks

@pytest.mark.parametrize(
    "estimator,tags,expected_failures",
    [(MyEstimator(param=val), {"key": True}, [])],
)
def test_estimator_checks(estimator, tags, expected_failures, y_X_factory):
    y, X = y_X_factory(length=100)
    estimator_fitted = clone(estimator)
    estimator_fitted.fit(y[:80], X[:80], forecasting_horizon=5)
    for check_name, check_func, check_kwargs in _yield_yohou_{component}_checks(
        estimator_fitted, y[:80], X[:80], y[80:], X[80:], tags=tags
    ):
        if check_name in set(expected_failures):
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(estimator_fitted, **check_kwargs)
```

---

## Metadata Routing Utilities (`src/yohou/testing/metadata_routing.py`)

6 utilities: `record_metadata`, `record_metadata_not_default`, `_Registry`, `check_recorded_metadata`, `assert_request_is_empty`, `assert_request_equal`

## File Layout

```
src/yohou/testing/           # Check function library
├── __init__.py              # Exports
├── generators.py            # 5 generators
├── transformer.py           # 21 checks
├── forecaster.py            # 13 checks
├── point.py                 # 2 checks
├── interval.py              # 5 checks
├── reduction.py             # 2 checks
├── panel.py                 # 3 checks
├── scorer.py                # 11 checks
├── splitter.py              # 8 checks
├── search.py                # 19 checks
├── common.py                # 2 checks
└── metadata_routing.py      # 6 utilities
```
