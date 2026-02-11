---
description: "Testing patterns for GridSearchCV and RandomizedSearchCV. 19 check functions for hyperparameter search validation. Use when writing or debugging search CV tests."
---

# Search CV Testing Infrastructure

## Components

- **GridSearchCV**: Exhaustive parameter search over all combinations
- **RandomizedSearchCV**: Random sampling from parameter distributions (n_iter controls)

---

## Check Functions — `src/yohou/testing/search.py` (19 total)

### Common Search CV Checks (12)

| Check | Validates |
|-------|-----------|
| `check_search_fit_sets_attributes` | Required attributes after fit() |
| `check_search_not_fitted_error` | NotFittedError before fit() |
| `check_search_cv_results_structure` | cv_results_ dictionary structure |
| `check_search_refit_false_no_forecaster` | refit=False: no best_forecaster_ |
| `check_search_predict_delegates` | predict() delegates to best_forecaster_ |
| `check_search_update_delegates` | update() delegates to best_forecaster_ |
| `check_search_reset_delegates` | reset() delegates to best_forecaster_ |
| `check_search_score_delegates` | score() uses internal scorer |
| `check_search_multimetric_scoring` | Multi-metric dict scoring works |
| `check_search_return_train_score` | return_train_score=True adds keys |
| `check_search_error_score_handling` | error_score parameter works |
| `check_search_clone_preserves_params` | clone() preserves parameters |

### GridSearchCV-Specific (2)

| Check | Validates |
|-------|-----------|
| `check_grid_search_exhaustive` | Evaluates all param combinations |
| `check_grid_search_param_grid_validation` | param_grid format validation |

### RandomizedSearchCV-Specific (3)

| Check | Validates |
|-------|-----------|
| `check_randomized_search_n_iter` | n_iter controls evaluation count |
| `check_randomized_search_reproducibility` | random_state ensures reproducibility |
| `check_randomized_search_distributions` | scipy.stats distributions work |

### Panel Data (2)

| Check | Validates |
|-------|-----------|
| `check_search_panel_data` | panel_group_names propagation |
| `check_search_method_availability` | @available_if logic with refit |

---

## Generator Usage

```python
from yohou.testing import _yield_yohou_search_checks

for check_name, check_func, check_kwargs in _yield_yohou_search_checks(
    search_cv, y_train, X_train, y_test, X_test,
    tags={"search_type": "grid", "refit": True, "multimetric": False}
):
    check_func(search_cv, **check_kwargs)
```

**Tags**: `search_type` ("grid"/"randomized"), `refit` (bool), `multimetric` (bool), `supports_panel_data` (bool)

**Generator logic**: Always yields common checks + conditional on refit (delegation vs no-forecaster) + search-type specific + panel if detected.

## Test File

```
tests/model_selection/test_search.py    # All search CV tests
```
