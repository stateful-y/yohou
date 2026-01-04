# Yohou Forecaster Testing Infrastructure
## A Guide to Testing Time Series Forecasters

**Based on**: scikit-learn regressor/classifier testing patterns, pytest fixtures, and time series-specific requirements

---

## Overview

Yohou's forecaster testing infrastructure provides comprehensive, reusable testing for time series forecasters (point and interval). It adapts patterns from scikit-learn's regressor/classifier testing framework while accommodating time series-specific requirements (forecasting_horizon, update/reset, single DataFrame for X, dual time columns in predictions).

### Key Features

1. **Check Generator Pattern**: Systematic check generation via `_yield_yohou_forecaster_checks()` (similar to sklearn's `_yield_regressor_checks`)

2. **Pytest Fixtures Architecture**: Factory fixtures for flexible data generation (y, X tuples), forecaster registry with tags

3. **Point vs Interval Forecasters**: Distinct validation for point predictions vs interval predictions with coverage rates

4. **Comprehensive Validation**: 20+ check functions covering fit/predict contracts, time column validation, observation buffer management, and composition patterns

5. **Analytical Tests**: Exact numerical validation for PointReductionForecaster with LinearRegression on known processes (linear trends, AR(1))

6. **Composition Classes**: Dedicated testing for ColumnForecaster composition patterns

---

## How It Works

## File Organization

The testing infrastructure is organized following pytest conventions:

```
tests/
├── conftest.py                         # Global fixtures (y_X_factory, forecaster_registry)
├── estimator_checks.py                 # Check function library (28 functions + generator)
├── test_estimator_checks.py            # Meta-tests for check functions
├── forecaster/
│   └── test_composition.py             # Tests for composition classes
├── point_forecaster/
│   ├── test_naive.py                   # Tests for SeasonalNaive
│   ├── test_reduction.py               # Tests for PointReductionForecaster + analytical tests
│   └── test_cross_learning.py          # Cross-learning tests for point forecasters
└── interval_forecaster/
    ├── test_reduction.py               # Tests for IntervalReductionForecaster
    └── test_cross_learning.py          # Cross-learning tests for interval forecasters
```

**Organization Principles**:
- Test files mirror source structure: `tests/{module}/test_{file}.py` for `src/yohou/{module}/{file}.py`
- Example: `src/yohou/point_forecaster/naive.py` → `tests/point_forecaster/test_naive.py`
- Check functions in `estimator_checks.py` are validated by actual usage in forecaster tests
- Shared testing infrastructure lives at `tests/` root
- Single global `conftest.py` sufficient - module-specific fixtures not needed

---

## Core Components

The testing infrastructure consists of three main components that work together to provide comprehensive forecaster validation.

### 1. Check Functions Library (`tests/estimator_checks.py`)

**Purpose**: Reusable library of 28 validation functions that test forecaster contracts

**Key Characteristics**:
- All functions raise `AssertionError` on failure (never return bool)
- Comprehensive docstrings with parameters and error conditions
- Functions accept `(forecaster, y, X, **kwargs)` signature
- Handle single X DataFrame (exogenous features) with "time" column

**Check Categories**:

**Common Forecaster Checks (8 functions)**:
1. **`check_fit_sets_forecaster_attributes`** - Validates fit() sets required attributes (fit_forecasting_horizon_, interval_, local_group_names_, local_y_columns_, local_X_columns_, _y_observed, _X_observed)
2. **`check_forecaster_not_fitted_error`** - Ensures NotFittedError before fit() when accessing fitted attributes
3. **`check_predict_time_columns`** - Validates predictions have "observed_time" and "predicted_time" columns
4. **`check_update_extends_observations`** - Tests update() properly extends _y_observed, _X_observed buffers
5. **`check_reset_replaces_observations`** - Tests reset() replaces observation buffers correctly (_y_observed, _X_observed)
6. **`check_forecasting_horizon_validation`** - Ensures forecasting_horizon < 1 raises ValueError
7. **`check_prediction_types_property`** - Validates prediction_types returns correct set ({"point"}, {"interval"}, or both)
8. **`check_clone_preserves_forecaster_params`** - sklearn's clone() preserves init parameters (enhanced to handle deeply nested estimators)

**Point Forecaster Checks (2 functions)**:
10. **`check_point_prediction_structure`** - Validates output has observed_time, predicted_time, and target columns only (no interval columns)
11. **`check_point_prediction_types`** - Ensures prediction_types == {"point"}

**Interval Forecaster Checks (4 functions)**:
12. **`check_interval_prediction_columns`** - Validates {col}_lower_{rate} and {col}_upper_{rate} format (handles both global and panel data with struct columns)
13. **`check_interval_bounds`** - Ensures upper >= lower for all coverage rates and time steps (handles struct columns by unnesting)
14. **`check_interval_prediction_types`** - Validates prediction_types contains "interval"
15. **`check_coverage_rates_parameter`** - Validates coverage_rates is list of floats in (0, 1)

**Reduction Forecaster Checks (2 functions)**:
16. **`check_estimator_parameter`** - Validates estimator is sklearn BaseEstimator
17. **`check_reduction_strategy`** - Validates reduction_strategy parameter exists

**Cross-Learning Forecaster Checks (3 functions)**:
18. **`check_cross_learning_panel_data`** - Validates cross_learning_group=None predicts all groups in panel data
19. **`check_cross_learning_single_group`** - Validates cross_learning_group filters to specified struct column
20. **`check_cross_learning_invalid_group_raises`** - Validates ValueError raised for invalid cross_learning_group

**Composition Class Checks (8 functions)**:
21. **`check_column_forecaster_column_selection`** - Validates column selectors (str, list, slice, callable) work correctly
22. **`check_column_forecaster_remainder_drop`** - Tests remainder='drop' excludes unspecified columns
23. **`check_column_forecaster_remainder_passthrough`** - Tests remainder='passthrough' uses default forecaster
24. **`check_column_forecaster_remainder_custom`** - Tests custom remainder forecaster
25. **`check_column_forecaster_parallel_execution`** - Validates n_jobs parameter works correctly
26. **`check_column_forecaster_column_order_preserved`** - Ensures prediction columns match input X column order
27. **`check_target_transformed_forecaster_inverse`** - Tests inverse transformation round-trip
28. **`check_target_transformed_forecaster_check_inverse_warning`** - Validates check_inverse warns when transformation isn't reversible

### 2. Check Generator (`tests/estimator_checks.py`)

**Function**: `_yield_yohou_forecaster_checks(forecaster, y_train, X_train, y_test, X_test, tags=None)`

**Purpose**: Dynamically generates applicable checks based on forecaster properties, eliminating boilerplate test code

**How It Works**:
1. Examines forecaster tags (forecaster_type, uses_reduction, is_column_forecaster, is_target_transformed, etc.)
2. Yields core checks that apply to all forecasters
3. Conditionally yields checks based on forecaster type (point vs interval)
4. Adds reduction-specific checks if uses_reduction=True
5. Adds composition-specific checks if is_column_forecaster or is_target_transformed=True
6. Returns tuples of `(check_name, check_function, check_kwargs)`

**Tag System**:
- `forecaster_type`: "point" | "interval" - Type of predictions
- `uses_reduction`: bool - Whether forecaster uses sklearn estimator
- `supports_panel_data`: bool - Whether forecaster handles struct columns
- `is_column_forecaster`: bool - Whether forecaster is ColumnForecaster (triggers composition checks)

**Benefits**:
- Average of 10-12 checks generated per forecaster automatically
- Adding new forecaster requires only registry entry
- Clear documentation of expected failures
- Systematic coverage without code duplication

### 3. Pytest Fixtures (`tests/conftest.py`)

**Purpose**: Centralized test data generation for forecasters

**Data Generation Fixtures** (1 fixture):
- `y_X_factory` - Factory function returning (y, X) tuples with configurable length, n_features, seed

**Configuration Fixtures** (1 fixture):
- `forecaster_registry` - Dict mapping forecaster names to instances, tags, and expected_failures (planned, not yet implemented)

**Fixture Usage Pattern**:
```python
def test_example(y_X_factory):
    # Generate train/test data
    y, X = y_X_factory(length=100, seed=42)
    y_train, y_test = y[:80], y[80:]
    X_train, X_test = X[:80], X[80:]

    # Fit and test forecaster
    forecaster.fit(y_train, X_train, forecasting_horizon=3)
    y_pred = forecaster.predict(forecasting_horizon=3, X=X_test)
```

---

## How to Test a Forecaster

### Step 1: Add to Forecaster Registry (Future)

Add your forecaster to the `forecaster_registry` fixture in `tests/conftest.py`:

```python
"MyForecaster": {
    "forecaster": MyForecaster(param=value),
    "expected_failed_checks": [],  # Or list check names that fail
    "tags": {
        "forecaster_type": "point",
        "uses_reduction": True,
        "uses_transformers": True,
    },
}
```

### Step 2: Create Test File

Create `tests/{module}/test_{file}.py` mirroring the source file location.

### Step 3: Write Parametrized Test

Use the check generator pattern:

```python
import sys
from pathlib import Path
import pytest
from sklearn.base import clone

from yohou.{module} import MyForecaster

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from estimator_checks import _yield_yohou_forecaster_checks

@pytest.mark.parametrize(
    "forecaster,tags,expected_failures",
    [
        (
            MyForecaster(param=value1),
            {"forecaster_type": "point", "uses_reduction": True},
            [],
        ),
        (
            MyForecaster(param=value2),
            {"forecaster_type": "point", "uses_reduction": True},
            [],
        ),
    ],
)
def test_my_forecaster_checks(forecaster, tags, expected_failures, y_X_factory):
    """Run systematic checks on MyForecaster."""
    # Generate data
    y, X = y_X_factory(length=100, seed=42)
    y_train, y_test = y[:80], y[80:]
    X_train, X_test = X[:80], X[80:]

    # Fit forecaster
    forecaster_fitted = clone(forecaster)
    forecaster_fitted.fit(y_train, X_train, forecasting_horizon=3)

    # Run all generated checks
    expected_failures_set = set(expected_failures)
    for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(
        forecaster_fitted, y_train, X_train,
        y_test, X_test, tags=tags
    ):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(forecaster_fitted, **check_kwargs)
```

### Step 4: Add Forecaster-Specific Tests

Below the parametrized test, add any forecaster-specific behavior tests:

```python
def test_my_forecaster_specific_behavior(y_X_factory):
    y, X = y_X_factory(length=50)
    forecaster = MyForecaster(param=value)
    forecaster.fit(y, X, forecasting_horizon=3)

    y_pred = forecaster.predict(forecasting_horizon=3, X=X)

    # Test specific behavior
    assert some_condition
```

---

## Test Results & Status

### Current State

**Forecaster Tests**:
- `tests/point_forecaster/test_naive.py`: 11/11 passing (2 check generator + 9 seasonality-specific)
- `tests/point_forecaster/test_reduction.py`: 13/13 passing (2 check generator + 9 existing + 2 analytical)
- `tests/point_forecaster/test_cross_learning.py`: 5/5 passing (1 check generator + 4 behavior-specific)
- `tests/interval_forecaster/test_reduction.py`: 9/9 passing
- `tests/interval_forecaster/test_cross_learning.py`: 5/5 passing (1 check generator with 1 expected failure + 4 behavior-specific)

### Validated Forecasters

**SeasonalNaive** (Point Forecaster):
- Check generator validates 9 checks automatically
- Seasonality-specific tests validate correct seasonal predictions
- No expected failures - all checks pass

**PointReductionForecaster** (Reduction + Point Forecaster):
- Check generator validates 11 checks automatically
- Analytical tests with LinearRegression validate exact numerical behavior
- Cross-learning tests validate panel data handling with `supports_panel_data=True` tag
- No expected failures - all checks pass

**IntervalReductionForecaster** (Reduction + Interval Forecaster):
- Check generator validates 15+ checks automatically
- Cross-learning tests validate panel data handling with struct columns
- Expected failure: `check_interval_bounds` with default QuantileRegressor (known issue - doesn't guarantee monotonic bounds)

**Decomposition Forecasters** (yohou.decomposition):
- **PolynomialTrendForecaster**: Check generator + analytical tests (linear/quadratic trend recovery)
- **ExponentialTrendForecaster**: Check generator + validation tests (positive values required)
- **SeasonalityForecaster**: Check generator + pattern-based tests (naive/average/median methods)
- **FourierSeasonalityForecaster**: Check generator + harmonic analysis tests (sine wave recovery)
- **Decomposer** (Meta-Forecaster): Check generator with special handling for list-of-tuples parameters
  - Expected failures: `check_update_extends_observations`, `check_reset_replaces_observations` (complex residual-based update logic)

---

## Implementation Insights

### Forecaster vs Transformer Differences

**Fitted Attributes**:
- **Transformers**: `feature_names_in_`, `n_features_in_`, `_observation_horizon`, `_X_observed`
- **Forecasters**: `fit_forecasting_horizon_`, `interval_`, `local_group_names_`, `local_y_columns_`, `local_X_columns_`, `_y_observed`, `_X_observed`, `_X_t_observed`

**Lifecycle Methods**:
- **Transformers**: `fit(X, y=None)`, `transform(X)`, `update(X)`, `reset(X)`
- **Forecasters**: `fit(y, X, forecasting_horizon)`, `predict(forecasting_horizon, X)`, `update(y, X)`, `reset(y, X)`

**Time Columns**:
- **Transformers**: Input/output both have "time" column
- **Forecasters**: Predictions add "observed_time" (when forecast was made) and "predicted_time" (time step being predicted)

**Composition Classes**:
- **ColumnForecaster**: Applies different forecasters to different X columns, concatenates predictions

### Analytical Testing Pattern

For forecasters with analytical solutions (e.g., PointReductionForecaster + LinearRegression), add exact numerical tests:

```python
def test_linear_regression_ar1_process():
    """Test PointReductionForecaster with LinearRegression on AR(1) process.

    For an AR(1) process y_t = phi * y_{t-1} + c, LinearRegression with default lag=1
    should recover the exact parameters and produce exact one-step-ahead predictions.
    """
    # Create AR(1) process: y_t = 0.8 * y_{t-1} + 5
    phi = 0.8
    c = 5.0

    # Generate series
    values = [10.0]  # Initial value
    for i in range(1, length):
        values.append(phi * values[-1] + c)

    y = pl.DataFrame({"time": time, "value": values})

    # Fit forecaster
    forecaster = PointReductionForecaster(estimator=LinearRegression())
    forecaster.fit(y_train, forecasting_horizon=1)

    # Check fitted coefficients match analytical solution
    fitted_estimator = forecaster.estimator_
    np.testing.assert_allclose(fitted_estimator.coef_[0], phi, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fitted_estimator.intercept_, c, rtol=1e-10, atol=1e-10)
```

### Clone Parameter Validation

The `check_clone_preserves_forecaster_params` function handles nested estimators correctly, including special handling for meta-estimators:

```python
# For nested estimators like LinearRegression, check type and params
if hasattr(orig_val, "get_params"):
    assert type(orig_val) == type(cloned_val), \
        f"Parameter {key}: different types"
    assert orig_val.get_params() == cloned_val.get_params(), \
        f"Parameter {key}: different estimator params"

# For list of (name, estimator) tuples (meta-estimators like Decomposer, Pipeline)
elif isinstance(orig_val, list) and len(orig_val) > 0 and isinstance(orig_val[0], tuple):
    for i, (orig_item, cloned_item) in enumerate(zip(orig_val, cloned_val)):
        orig_name, orig_est = orig_item
        cloned_name, cloned_est = cloned_item
        
        # Names should match exactly
        assert orig_name == cloned_name
        
        # Estimators should be different instances but same type
        assert type(orig_est) == type(cloned_est)
        assert orig_est is not cloned_est  # Verify cloning happened
```

This prevents false failures when comparing meta-estimator instances like Decomposer or ColumnForecaster.

---

## Check Function Implementation Details

### Fit Attributes Validation

**Purpose**: Verify fit() sets all required forecaster attributes

**Validation**:
- `fit_forecasting_horizon_` matches input parameter
- `interval_` is a timedelta object
- `local_group_names_`, `local_y_columns_`, `local_X_columns_` are set
- `_y_observed` and `_X_observed` buffers exist

**Note**: All features in `X` are expected to be known ex-ante. For ex-post features, use `ColumnForecaster`.

### Composition Class Validation

**Purpose**: Verify ColumnForecaster implement composition patterns correctly

**ColumnForecaster Validation**:
- Column selectors (str, list, slice, callable) properly index X columns
- Remainder strategies ('drop', 'passthrough', custom forecaster) work as expected
- Parallel execution with `n_jobs` produces same results as sequential
- Prediction column order matches input X column order
- Panel data (struct columns) handled correctly

### Time Column Validation

**Purpose**: Ensure predictions have dual time columns for alignment

**Validation**:
- "observed_time" column exists (when forecast was made)
- "predicted_time" column exists (time step being predicted)
- Both columns have datetime dtype
- Number of rows matches forecasting_horizon

**Usage**: These dual time columns enable:
- Tracking forecast vintage (when was this forecast made?)
- Aligning predictions with actual observations for scoring
- Rolling window evaluation patterns

### Reduction Strategy Validation

**Purpose**: Check reduction forecasters properly handle sklearn estimators

**Validation**:
- `estimator` parameter exists and is sklearn BaseEstimator
- After fit(), `estimator_` attribute set with fitted estimator
- For PointReductionForecaster, default estimator is Ridge()
- `reduction_strategy` parameter exists (default: "multi-output")

---

## Success Metrics

✅ **Current achievements**:

1. **Coverage**: All tested forecasters pass 100% of applicable checks
   - SeasonalNaive: 9/9 checks passing
   - PointReductionForecaster: 11/11 checks passing

2. **Maintainability**: Check generator pattern eliminates test boilerplate
   - Single parametrized test covers all systematic checks
   - Adding forecaster-specific tests is straightforward

3. **Clarity**: Test failures provide actionable error messages
   - All assertions include descriptive messages with actual vs expected values
   - Check names clearly indicate what's being validated

4. **Analytical Validation**: Exact tests for known processes
   - LinearRegression on AR(1) process validates coefficient recovery
   - Linear trend test validates recursive prediction accuracy

5. **Documentation**: Check functions have comprehensive docstrings
   - All 16+ check functions documented with NumPy style
   - Ex-ante vs ex-post distinction clearly explained

📋 **Remaining work**:

1. Complete interval forecaster testing (`test_reduction.py` for SplitConformalForecaster)
2. Implement forecaster_registry fixture in conftest.py
3. Add meta-tests to validate check functions catch failures
4. Document additional forecaster types as they're implemented

---

## Common Patterns

### Import Pattern for Test Files

```python
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from estimator_checks import _yield_yohou_forecaster_checks
```

**Why**: Tests are in subdirectories (`point_forecaster/`, `interval_forecaster/`) but need to import from `tests/estimator_checks.py`.

### Data Splitting Pattern

```python
def test_my_forecaster(y_X_factory):
    # Generate data
    y, X = y_X_factory(length=100, seed=42)

    # Split maintaining temporal order
    y_train, y_test = y[:80], y[80:]
    X_train, X_test = X[:80], X[80:]
```

**Critical**: Always maintain temporal order when splitting - no shuffling for time series data.

### Expected Failures Pattern

```python
@pytest.mark.parametrize(
    "forecaster,tags,expected_failures",
    [
        (
            MyForecaster(),
            {"forecaster_type": "point"},
            ["check_reset_propagates_to_transformers"],  # Known limitation
        ),
    ],
)
def test_my_forecaster_checks(forecaster, tags, expected_failures, y_X_factory):
    # ...
    expected_failures_set = set(expected_failures)
    for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(...):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(forecaster_fitted, **check_kwargs)
```

**Purpose**: Document known limitations while maintaining systematic testing.

### Cross-Learning Testing Pattern

**Overview**: Cross-learning enables training on all groups (e.g., all stores) but predicting for specific groups only. This is tested via the `supports_panel_data=True` tag.

**Check Generator Integration**:
```python
@pytest.mark.parametrize(
    "forecaster,tags,expected_failures",
    [
        (
            PointReductionForecaster(
                estimator=LinearRegression(),
                feature_transformer=LagTransformer(lag=[1, 2])
            ),
            {"forecaster_type": "point", "uses_reduction": True, "supports_panel_data": True},
            [],
        ),
    ],
)
def test_point_reduction_cross_learning_checks(forecaster, tags, expected_failures, panel_data):
    """Run systematic cross-learning checks on PointReductionForecaster with panel data."""
    y = panel_data["y"]
    y_train, y_test = y[:80], y[80:]

    forecaster_fitted = clone(forecaster)
    forecaster_fitted.fit(y_train, X=None, forecasting_horizon=3)

    # Run all generated checks (including cross-learning checks)
    for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(
        forecaster_fitted, y_train, None, y_test, None, tags=tags
    ):
        if check_name in set(expected_failures):
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(forecaster_fitted, **check_kwargs)
```

**Cross-Learning Check Functions**:
1. **`check_cross_learning_panel_data`**: Validates `cross_learning_group=None` predicts all groups
2. **`check_cross_learning_single_group`**: Validates filtering to specific struct column
3. **`check_cross_learning_invalid_group_raises`**: Validates error handling for invalid groups

**Panel Data Structure**:
- Single struct column: `{"time": ..., "stores": pl.DataFrame({"store_0": ..., "store_1": ..., ...})}`
- Framework constraint: All struct columns must have same field names
- Interval forecasters nest interval bounds within structs: `store_0_lower_0.1` inside `"stores"` struct

**Key Considerations**:
- `cross_learning_group` operates on struct column level, not field level (e.g., `"stores"` not `"store_0"`)
- For interval forecasters, unnest struct columns to access interval bounds
- Expected failure for `IntervalReductionForecaster` with default estimator: `check_interval_bounds` (QuantileRegressor doesn't guarantee monotonic bounds)

---

## References

- sklearn source: `sklearn/utils/estimator_checks.py` (regressor/classifier checks)
- sklearn patterns: `sklearn/tests/test_common.py` (parametrize_with_checks)
- Transformer testing: `.github/copilot_plans/transformer-testing-infrastructure.md`
- yohou architecture: `.github/copilot-instructions.md` (base classes, data flow)
