# Yohou Search CV Testing Infrastructure

## Overview

This guide documents the systematic testing infrastructure for hyperparameter search classes (`GridSearchCV` and `RandomizedSearchCV`) that implement cross-validated parameter optimization for forecasters.

### Key Features

1. **Check Generator Pattern**: Systematic check generation via `_yield_yohou_search_checks()` generator function

2. **Tag-Driven Selection**: Checks automatically selected based on search CV properties (search_type, refit, multimetric, supports_panel_data)

3. **Comprehensive Validation**: 19 total check functions covering fit/predict contracts, delegation, scoring, refit behavior, exhaustive/random search, reproducibility, and panel data support

4. **sklearn-Compatible**: Follows sklearn's BaseSearchCV patterns with metadata routing, cross-validation, and multi-metric scoring

5. **Panel Data Support**: Full support for panel time series with group prefixes (e.g., `sales__store_1`)

### Architecture

```
src/yohou/testing/
├── search.py           # 19 check functions for search CV validation
├── generators.py       # _yield_yohou_search_checks() generator
└── __init__.py         # Public API exports

tests/model_selection/
└── test_search.py      # Systematic tests using generator pattern
```

---

## Core Components

### 1. Check Functions Module (`src/yohou/testing/search.py`)

**Purpose**: Provides reusable validation functions that test specific aspects of search CV behavior

**Check Categories**:
- **Common Search CV Checks** (12 functions): Core validation applicable to all search CVs
- **GridSearchCV-Specific** (2 functions): Exhaustive search validation
- **RandomizedSearchCV-Specific** (3 functions): Sampling and reproducibility
- **Panel Data Checks** (2 functions): Panel data and method availability

**Key Patterns**:
- All functions raise `AssertionError` on failure (never return bool)
- Signature: `check_*(search_cv, y, X, **kwargs)`
- Use `clone()` to avoid mutating inputs
- Include descriptive error messages with actual vs expected values

---

### 2. Check Generator (`src/yohou/testing/generators.py`)

**Function**: `_yield_yohou_search_checks(search_cv, y_train, X_train, y_test, X_test, tags=None)`

**Purpose**: Dynamically generates applicable checks based on search CV properties

**How It Works**:
1. Infers or accepts tags about search CV type and configuration
2. Yields core checks applicable to all search CVs
3. Conditionally yields checks based on `refit` setting (delegation vs no-forecaster checks)
4. Adds search_type-specific checks (grid exhaustive or randomized sampling)
5. Includes panel data checks if panel groups detected
6. Returns tuples of `(check_name, check_function, check_kwargs)`

**Tag System**:
- `search_type`: "grid" | "randomized" - Type of search algorithm
- `refit`: bool - Whether best_forecaster_ is created and fitted
- `multimetric`: bool - Whether scoring is dict with multiple metrics
- `supports_panel_data`: bool - Whether search CV handles panel data (always True)

**Generator Logic**:
```python
# Common checks (always)
yield fit_sets_attributes
yield not_fitted_error
yield cv_results_structure
yield clone_preserves_params
yield error_score_handling

# Conditional on refit
if refit:
    yield predict_delegates
    yield update_delegates  # if enough test data
    yield reset_delegates   # if enough test data
    yield score_delegates
else:
    yield refit_false_no_forecaster

# Method availability
yield method_availability

# Multi-metric
if multimetric:
    yield multimetric_scoring

# Return train score (parameterized)
yield return_train_score

# Search type specific
if search_type == "grid":
    yield grid_search_exhaustive
    yield grid_search_param_grid_validation
elif search_type == "randomized":
    yield randomized_search_n_iter
    if random_state is not None:
        yield randomized_search_reproducibility
    yield randomized_search_distributions

# Panel data
if panel_data_detected:
    yield search_panel_data
```

---

### 3. Test File Structure (`tests/model_selection/test_search.py`)

**Purpose**: Systematic test suite using generator pattern to validate search CV classes

**Structure**:
1. **Parametrized Systematic Tests**: Main test using generator with multiple search CV configurations
2. **Panel Data Tests**: Dedicated tests for panel time series
3. **Edge Case Tests**: Single-value grids, n_iter exceeding space, X=None scenarios
4. **Multi-Metric Tests**: Best score selection and score return type
5. **Return Train Score Tests**: Train score key validation
6. **Error Handling Tests**: error_score parameter behavior

**Key Patterns**:
- Use `y_X_factory` for regular data, `y_X_panel_factory` for panel data
- Clone search CV before fitting to avoid mutations
- Use `expected_failures` list for known limitations
- Test both GridSearchCV and RandomizedSearchCV in parametrized tests

---

## Complete Check Function Reference

### Common Search CV Checks (12 functions)

#### 1. `check_search_fit_sets_attributes`
**Purpose**: Validates that `fit()` creates all required fitted attributes

**Checks**:
- `cv_results_` (dict): Cross-validation results
- `best_params_` (dict): Best parameter combination
- `best_score_` (numeric): Best cross-validation score
- `best_index_` (int): Index of best parameters in cv_results_
- `scorer_`: Scorer object or dict of scorers
- `n_splits_` (int): Number of CV splits
- `multimetric_` (bool): Whether multi-metric scoring used
- `best_forecaster_` (when refit=True): Refitted forecaster
- `refit_time_` (when refit=True): Time to refit best forecaster

**Usage**:
```python
from yohou.testing import check_search_fit_sets_attributes

check_search_fit_sets_attributes(search_cv, y_train, X_train, forecasting_horizon=3)
```

**Common Failures**:
- Missing attribute after fit
- Wrong attribute type (e.g., best_score_ not numeric)

---

#### 2. `check_search_not_fitted_error`
**Purpose**: Validates that accessing fitted attributes before `fit()` raises `NotFittedError`

**Checks**:
- `check_is_fitted()` raises NotFittedError before fit
- `predict()` raises NotFittedError or AttributeError before fit

**Usage**:
```python
from yohou.testing import check_search_not_fitted_error

check_search_not_fitted_error(search_cv, y_train, X_train)
```

---

#### 3. `check_search_cv_results_structure`
**Purpose**: Validates `cv_results_` dictionary structure and consistency

**Checks**:
- `"params"` key exists and is list of dicts
- Non-empty params list
- For single metric:
  - `"mean_test_score"` exists with correct length
  - `"rank_test_score"` exists with correct length
- For each split:
  - `"split{n}_test_score"` exists with correct length

**Usage**:
```python
from yohou.testing import check_search_cv_results_structure

check_search_cv_results_structure(search_cv, y_train, X_train, forecasting_horizon=3)
```

**Example cv_results_ structure**:
```python
{
    "params": [{"seasonality": 1}, {"seasonality": 5}, {"seasonality": 10}],
    "mean_test_score": [0.5, 0.3, 0.4],
    "rank_test_score": [3, 1, 2],
    "split0_test_score": [0.6, 0.4, 0.5],
    "split1_test_score": [0.4, 0.2, 0.3],
    # ... additional keys for train scores, times, etc.
}
```

---

#### 4. `check_search_refit_false_no_forecaster`
**Purpose**: Validates that `refit=False` doesn't create `best_forecaster_`

**Checks**:
- `best_params_` and `best_score_` still set
- `best_forecaster_` NOT created
- `refit_time_` NOT created
- `predict()` raises AttributeError

**Usage**:
```python
from yohou.testing import check_search_refit_false_no_forecaster

search_cv.refit = False
check_search_refit_false_no_forecaster(search_cv, y_train, X_train, forecasting_horizon=3)
```

**Note**: Temporarily sets refit=False, then restores original value

---

#### 5. `check_search_predict_delegates`
**Purpose**: Validates that `predict()` correctly delegates to `best_forecaster_.predict()`

**Checks**:
- Predictions from search CV match predictions from best_forecaster_ directly
- Uses polars `assert_frame_equal()` for DataFrame comparison

**Usage**:
```python
from yohou.testing import check_search_predict_delegates

check_search_predict_delegates(search_cv, y_train, y_test, X_train, X_test)
```

**Requires**: `refit=True` (check only runs for refitted search CVs)

---

#### 6. `check_search_update_delegates`
**Purpose**: Validates that `update()` correctly delegates to `best_forecaster_.update()`

**Checks**:
- `observed_time_` changes after update
- For panel data: All group observed times increase
- For non-panel: Single observed_time increases

**Usage**:
```python
from yohou.testing import check_search_update_delegates

y_update = y_test[:3]
X_update = X_test[:3] if X_test is not None else None
check_search_update_delegates(search_cv, y_train, y_update, X_train, X_update)
```

**Requires**: Fitted search CV with refit=True

---

#### 7. `check_search_reset_delegates`
**Purpose**: Validates that `reset()` correctly delegates to `best_forecaster_.reset()`

**Checks**:
- `observed_time_` changes after reset (may increase or decrease depending on data)
- For panel data: All group observed times change
- For non-panel: Single observed_time changes

**Usage**:
```python
from yohou.testing import check_search_reset_delegates

y_reset = y_test[:10]
X_reset = X_test[:10] if X_test is not None else None
check_search_reset_delegates(search_cv, y_train, y_reset, X_train, X_reset)
```

**Requires**: Fitted search CV with refit=True and sufficient test data (≥10 rows)

---

#### 8. `check_search_score_delegates`
**Purpose**: Validates that `score()` uses internal scorer correctly

**Checks**:
- For single metric: Returns numeric score
- For multi-metric: Returns dict with all metric names
- No NaN scores

**Usage**:
```python
from yohou.testing import check_search_score_delegates

check_search_score_delegates(search_cv, y_train, y_test, X_train, X_test)
```

---

#### 9. `check_search_multimetric_scoring`
**Purpose**: Validates multi-metric scoring with dict scorer

**Checks**:
- `multimetric_` flag is True
- All metric names in cv_results_ with `mean_test_{name}` and `rank_test_{name}` keys
- `scorer_` is dict
- `best_score_` matches refit metric at best_index_

**Usage**:
```python
from yohou.testing import check_search_multimetric_scoring

scoring = {"mae": MeanAbsoluteError(), "rmse": RootMeanSquaredError()}
search_cv = GridSearchCV(..., scoring=scoring, refit="mae")
check_search_multimetric_scoring(search_cv, y_train, X_train, forecasting_horizon=3)
```

**Requires**: `search_cv.scoring` must be dict

---

#### 10. `check_search_return_train_score`
**Purpose**: Validates that `return_train_score=True` adds train score keys

**Checks**:
- `"mean_train_score"` exists when return_train_score=True
- `"split{n}_train_score"` exists for each split

**Usage**:
```python
from yohou.testing import check_search_return_train_score

search_cv.return_train_score = True
check_search_return_train_score(search_cv, y_train, X_train, forecasting_horizon=3)
```

**Note**: Temporarily sets return_train_score=True, then restores original

---

#### 11. `check_search_error_score_handling`
**Purpose**: Validates that `error_score` parameter handles failing fits correctly

**Checks**:
- With `error_score=np.nan`, fit completes without raising (failed fits get NaN scores)
- `cv_results_` still created

**Usage**:
```python
from yohou.testing import check_search_error_score_handling

search_cv.error_score = np.nan
check_search_error_score_handling(search_cv, y_train, X_train, forecasting_horizon=3)
```

**Note**: This check may pass trivially if all parameters are valid

---

#### 12. `check_search_clone_preserves_params`
**Purpose**: Validates that sklearn `clone()` preserves search CV parameters

**Checks**:
- Parameter names match between original and clone
- Forecaster type matches
- Scorer type matches (dict or single scorer)
- Fitted attributes NOT cloned (clone is unfitted)

**Usage**:
```python
from yohou.testing import check_search_clone_preserves_params

check_search_clone_preserves_params(search_cv)  # Works on fitted or unfitted
```

---

### GridSearchCV-Specific Checks (2 functions)

#### 13. `check_grid_search_exhaustive`
**Purpose**: Validates that GridSearchCV evaluates all parameter combinations

**Checks**:
- Actual number of candidates matches expected combinations
- Handles single dict param_grid: product of all value lengths
- Handles list of dicts param_grid: sum of products

**Usage**:
```python
from yohou.testing import check_grid_search_exhaustive

param_grid = {"seasonality": [1, 5, 10], "method": ["naive", "average"]}
search = GridSearchCV(..., param_grid=param_grid)
search.fit(y_train, X_train, forecasting_horizon=3)
check_grid_search_exhaustive(search, y_train, X_train, forecasting_horizon=3)
# Expected: 3 * 2 = 6 combinations
```

**Example**:
```python
# Single grid
param_grid = {"alpha": [0.1, 1.0, 10.0], "fit_intercept": [True, False]}
# Combinations: 3 * 2 = 6

# Multiple grids
param_grid = [
    {"alpha": [0.1, 1.0], "fit_intercept": [True]},   # 2 * 1 = 2
    {"alpha": [10.0], "fit_intercept": [True, False]} # 1 * 2 = 2
]
# Total: 2 + 2 = 4 combinations
```

---

#### 14. `check_grid_search_param_grid_validation`
**Purpose**: Validates `param_grid` format (dict or list of dicts)

**Checks**:
- param_grid is dict or list
- All keys are strings
- All values are lists or tuples
- For list of dicts: each element is dict with valid structure

**Usage**:
```python
from yohou.testing import check_grid_search_param_grid_validation

check_grid_search_param_grid_validation(search_cv)  # Works before or after fit
```

---

### RandomizedSearchCV-Specific Checks (3 functions)

#### 15. `check_randomized_search_n_iter`
**Purpose**: Validates that `n_iter` controls number of evaluations

**Checks**:
- Actual number of candidates equals `n_iter`

**Usage**:
```python
from yohou.testing import check_randomized_search_n_iter

search = RandomizedSearchCV(..., n_iter=10)
search.fit(y_train, X_train, forecasting_horizon=3)
check_randomized_search_n_iter(search, y_train, X_train, forecasting_horizon=3)
```

**Note**: If `n_iter` exceeds parameter space, sklearn may deduplicate samples (behavior varies)

---

#### 16. `check_randomized_search_reproducibility`
**Purpose**: Validates that `random_state` produces reproducible samples

**Checks**:
- Two fits with same random_state produce identical parameter lists
- Scores are identical (via `np.testing.assert_array_equal`)

**Usage**:
```python
from yohou.testing import check_randomized_search_reproducibility

search = RandomizedSearchCV(..., random_state=42)
check_randomized_search_reproducibility(search, y_train, X_train, forecasting_horizon=3)
```

**Requires**: `search_cv.random_state` must not be None

---

#### 17. `check_randomized_search_distributions`
**Purpose**: Validates that scipy.stats distributions work for sampling

**Checks**:
- Fit completes successfully with scipy distributions
- Parameters are sampled (non-empty params list)

**Usage**:
```python
from scipy.stats import uniform
from yohou.testing import check_randomized_search_distributions

param_distributions = {"alpha": uniform(loc=0.01, scale=1.0)}
search = RandomizedSearchCV(..., param_distributions=param_distributions)
check_randomized_search_distributions(search, y_train, X_train, forecasting_horizon=3)
```

**Note**: Requires `scipy` in test dependencies

---

### Panel Data Checks (2 functions)

#### 18. `check_search_panel_data`
**Purpose**: Validates that `panel_group_names` parameter propagates correctly

**Checks**:
- Predictions with `panel_group_names` include requested groups
- Panel prefixes present in prediction columns

**Usage**:
```python
from yohou.testing import check_search_panel_data

panel_group_names = ["sales", "inventory"]
check_search_panel_data(
    search_cv, y_train_panel, y_test_panel, X_train_panel, X_test_panel,
    panel_group_names=panel_group_names
)
```

**Example Panel Data**:
```python
y_panel = pl.DataFrame({
    "time": [...],
    "sales__store_1": [...],
    "sales__store_2": [...],
    "inventory__store_1": [...],
    "inventory__store_2": [...],
})
```

---

#### 19. `check_search_method_availability`
**Purpose**: Validates `@available_if` decorator logic with refit setting

**Checks**:
- When refit=True: predict() is callable
- When refit=False: predict() raises AttributeError

**Usage**:
```python
from yohou.testing import check_search_method_availability

check_search_method_availability(search_cv, y_train, X_train, forecasting_horizon=3)
```

**Note**: Tests both refit=True and refit=False by cloning and modifying

---

## How to Test a Search CV

### Step 1: Identify Search CV Type and Configuration

Determine:
- **Search algorithm**: GridSearchCV or RandomizedSearchCV
- **Refit setting**: True (creates best_forecaster_) or False (results only)
- **Scoring**: Single metric or multi-metric dict
- **Panel data**: Does it handle panel prefixes?

### Step 2: Define Tags

Tags control which checks are generated:

```python
tags = {
    "search_type": "grid",          # or "randomized"
    "refit": True,                  # or False
    "multimetric": False,           # or True if scoring is dict
    "supports_panel_data": True,    # Always True for search CVs
}
```

### Step 3: Write Parametrized Test

Use the check generator pattern:

```python
import pytest
from sklearn.base import clone
from yohou.model_selection import GridSearchCV
from yohou.testing import _yield_yohou_search_checks

@pytest.mark.parametrize(
    "search_cv_class,params,tags,expected_failures",
    [
        (
            GridSearchCV,
            {
                "param_grid": {"seasonality": [1, 5, 10]},
                "scoring": MeanAbsoluteError(),
                "cv": 2,
                "refit": True,
            },
            {"search_type": "grid", "refit": True, "multimetric": False},
            [],  # No expected failures
        ),
    ],
)
def test_search_cv_systematic_checks(
    search_cv_class, params, tags, expected_failures, y_X_factory
):
    """Run systematic checks on search CV classes."""
    # Generate data
    y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)
    y_train, y_test = y[:80], y[80:]
    X_train, X_test = (X[:80], X[80:]) if X is not None else (None, None)

    # Create and fit search CV
    forecaster = SeasonalNaive()
    search_cv = search_cv_class(forecaster=forecaster, **params)
    search_cv_fitted = clone(search_cv)
    search_cv_fitted.fit(y_train, X_train, forecasting_horizon=3)

    # Run all generated checks
    expected_failures_set = set(expected_failures)
    for check_name, check_func, check_kwargs in _yield_yohou_search_checks(
        search_cv_fitted,
        y_train,
        X_train,
        y_test,
        X_test,
        tags=tags,
    ):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(search_cv_fitted, **check_kwargs)
```

### Step 4: Add Search-Specific Tests

Beyond systematic checks, add tests for specific behaviors:

```python
def test_grid_search_exhaustive_combinations(y_X_factory):
    """Test that all parameter combinations are evaluated."""
    y, X = y_X_factory(length=100)
    
    search = GridSearchCV(
        forecaster=SeasonalNaive(),
        param_grid={"seasonality": [1, 5, 10]},
        scoring=MeanAbsoluteError(),
        cv=2,
    )
    search.fit(y[:80], X[:80] if X is not None else None, forecasting_horizon=3)
    
    # Should evaluate exactly 3 combinations
    assert len(search.cv_results_["params"]) == 3
```

---

## Common Patterns

### Expected Failures Pattern

Document known limitations while maintaining systematic testing:

```python
@pytest.mark.parametrize(
    "search_cv_class,params,tags,expected_failures",
    [
        (
            GridSearchCV,
            {...},
            {"search_type": "grid"},
            ["check_search_panel_data"],  # Known limitation: panel support incomplete
        ),
    ],
)
def test_search_cv_checks(search_cv_class, params, tags, expected_failures, y_X_factory):
    # ... run checks with expected_failures
```

### Panel Data Testing Pattern

Use `y_X_panel_factory` for generating panel data:

```python
def test_search_cv_panel_data(y_X_panel_factory):
    """Test search CV with panel data."""
    y_panel, X_panel = y_X_panel_factory(
        n_groups=2, length=80, n_targets=1, n_features=2
    )
    
    search = GridSearchCV(...)
    search.fit(y_panel[:60], X_panel[:60] if X_panel else None, forecasting_horizon=3)
    
    # Predict with panel data
    y_pred = search.predict(forecasting_horizon=3)
    
    # Check panel structure
    assert any("__" in col for col in y_pred.columns)
```

### Multi-Metric Testing Pattern

Test multi-metric scoring with dict:

```python
def test_multimetric_scoring(y_X_factory):
    """Test multi-metric scoring."""
    y, X = y_X_factory(length=100)
    
    search = GridSearchCV(
        forecaster=SeasonalNaive(),
        param_grid={"seasonality": [1, 5, 10]},
        scoring={
            "mae": MeanAbsoluteError(),
            "rmse": RootMeanSquaredError(),
        },
        cv=2,
        refit="mae",  # Refit on MAE
    )
    search.fit(y[:80], X[:80] if X else None, forecasting_horizon=3)
    
    # Check that both metrics are in cv_results_
    assert "mean_test_mae" in search.cv_results_
    assert "mean_test_rmse" in search.cv_results_
    
    # Check that best_score_ matches refit metric
    expected_score = search.cv_results_["mean_test_mae"][search.best_index_]
    assert abs(search.best_score_ - expected_score) < 1e-6
```

### scipy Distribution Testing Pattern

Test RandomizedSearchCV with scipy distributions:

```python
from scipy.stats import uniform, randint

def test_scipy_distributions(y_X_factory):
    """Test RandomizedSearchCV with scipy distributions."""
    y, X = y_X_factory(length=100)
    
    search = RandomizedSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions={
            "estimator__alpha": uniform(loc=0.01, scale=1.0),
            "lags": randint(low=1, high=10),
        },
        n_iter=10,
        scoring=MeanAbsoluteError(),
        cv=2,
        random_state=42,
    )
    search.fit(y[:80], X[:80] if X else None, forecasting_horizon=3)
    
    # Check that parameters were sampled
    assert len(search.cv_results_["params"]) == 10
```

---

## Known Issues and Limitations

### 1. RandomizedSearchCV Deduplication

**Issue**: When `n_iter` exceeds parameter space for discrete distributions, sklearn may deduplicate samples, resulting in fewer evaluations than requested.

**Example**:
```python
# Only 3 possible values
param_distributions = {"seasonality": [1, 5, 10]}
n_iter = 10

# May evaluate only 3 unique combinations instead of 10
```

**Workaround**: Use continuous distributions or increase parameter space.

### 2. Multimetric Metadata Routing

**Issue**: Multi-metric scoring with dict scorers may trigger metadata routing errors in sklearn.

**Error**: `ValueError: not enough values to unpack (expected 2, got 1)`

**Status**: Under investigation. May be sklearn bug or yohou scorer implementation issue.

### 3. scipy Distribution Integer Casting

**Issue**: scipy distributions return floats, but some forecaster parameters require integers.

**Example**:
```python
# uniform returns floats like 5.7
param_distributions = {"seasonality": uniform(loc=1, scale=10)}

# Causes error if seasonality must be int
```

**Workaround**: Use discrete distributions like `randint` or cast in forecaster.

### 4. X=None Validation

**Issue**: Some validation functions don't handle `X=None` correctly.

**Status**: Partially fixed in `validate_forecaster_data`, but some edge cases remain.

---

## Testing Checklist

When testing a new search CV class:

- [ ] Run systematic checks via `_yield_yohou_search_checks()`
- [ ] Test with single metric scoring
- [ ] Test with multi-metric scoring (dict)
- [ ] Test with `refit=True` (delegation checks)
- [ ] Test with `refit=False` (no forecaster checks)
- [ ] Test with panel data (if supported)
- [ ] Test X=None scenario (forecast-only)
- [ ] Test return_train_score=True
- [ ] Test error_score parameter
- [ ] Test clone() preserves parameters
- [ ] For GridSearchCV: Test exhaustive evaluation
- [ ] For RandomizedSearchCV: Test n_iter, random_state, scipy distributions
- [ ] Document expected failures if any
- [ ] Add search-specific behavioral tests

---

## Integration with CI/CD

### Pre-commit Hooks

Search tests run automatically via pre-commit hooks:

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: pytest-search
      name: Search CV Tests
      entry: uv run pytest tests/model_selection/test_search.py
      language: system
      pass_filenames: false
```

### GitHub Actions

Search tests run on all PRs:

```yaml
# .github/workflows/tests-os-coverage.yml
- name: Test search CVs
  run: uv run pytest tests/model_selection/test_search.py -v --cov=src/yohou/model_selection
```

---

## Appendix: Example Test File Structure

Complete example showing recommended test organization:

```python
"""Systematic tests for GridSearchCV and RandomizedSearchCV."""

import pytest
from scipy.stats import uniform
from sklearn.base import clone

from yohou.metrics import MeanAbsoluteError, RootMeanSquaredError
from yohou.model_selection import GridSearchCV, RandomizedSearchCV
from yohou.point_forecaster import SeasonalNaive
from yohou.testing import _yield_yohou_search_checks


# ============================================================================
# Parametrized Systematic Tests
# ============================================================================

@pytest.mark.parametrize(
    "search_cv_class,params,tags,expected_failures",
    [
        # GridSearchCV with single metric
        (GridSearchCV, {...}, {...}, []),
        # GridSearchCV with multi-metric
        (GridSearchCV, {...}, {...}, []),
        # RandomizedSearchCV
        (RandomizedSearchCV, {...}, {...}, []),
    ],
)
def test_search_cv_systematic_checks(
    search_cv_class, params, tags, expected_failures, y_X_factory
):
    """Run systematic checks on search CV classes."""
    # Implementation here...


# ============================================================================
# Panel Data Tests
# ============================================================================

def test_search_cv_panel_data(y_X_panel_factory):
    """Test search CV with panel data."""
    # Implementation here...


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_single_value_param_grid(y_X_factory):
    """Test degenerate case with single-value param_grid."""
    # Implementation here...


# ============================================================================
# Multi-Metric Tests
# ============================================================================

def test_multimetric_best_score_selection(y_X_factory):
    """Test best_score_ corresponds to refit metric."""
    # Implementation here...


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_error_score_handling(y_X_factory):
    """Test error_score parameter."""
    # Implementation here...
```

---

## Summary

The search CV testing infrastructure provides:

1. **19 check functions** covering all search CV behaviors
2. **Tag-driven generator** for automatic check selection
3. **Systematic test pattern** reducing boilerplate
4. **Panel data support** with dedicated fixtures and checks
5. **sklearn compatibility** following BaseSearchCV patterns
6. **Comprehensive documentation** with usage examples

This infrastructure ensures search CV classes are thoroughly tested with minimal code duplication, making it easy to add new search algorithms or validate existing implementations.
