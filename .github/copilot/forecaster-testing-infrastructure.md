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

4. **Comprehensive Validation**: 25 total check functions covering fit/predict contracts, time column validation, observation buffer management, tag system validation (13 common + 2 point + 5 interval + 2 reduction + 3 panel), plus 2 metadata routing checks

5. **Parameter Validation Tests**: Dedicated tests ensure `forecasting_horizon` and `coverage_rates` are always validated in fit(), predict(), and update methods

6. **Analytical Tests**: Exact numerical validation for PointReductionForecaster with LinearRegression on known processes (linear trends, AR(1))

7. **Composition Classes**: Dedicated testing for ColumnForecaster composition patterns

**Related Guides**:
- [Splitter Testing Infrastructure](splitter-testing-infrastructure.md): Cross-validation splitter testing
- [Scorer Testing Infrastructure](scorer-testing-infrastructure.md): Metrics/scorers testing
- [Transformer Testing Infrastructure](transformer-testing-infrastructure.md): Transform testing
- [sklearn Metadata Routing Implementation](sklearn-metadata-routing-implementation.md): Metadata routing patterns

---

## How It Works

## File Organization

The testing infrastructure is organized following pytest conventions:

```
tests/
├── conftest.py                         # Global fixtures (y_X_factory, data generators, forecaster_registry)
├── test_estimator_checks.py            # sklearn compatibility tests
├── test_parameter_validation.py        # Tests for forecasting_horizon and coverage_rates validation
├── forecaster/
│   └── test_composition.py             # Tests for composition classes
├── point_forecaster/
│   ├── test_naive.py                   # Tests for SeasonalNaive
│   ├── test_reduction.py               # Tests for PointReductionForecaster + analytical tests
│   └── test_panel.py                   # Cross-learning tests for point forecasters
└── interval_forecaster/
    ├── test_reduction.py               # Tests for IntervalReductionForecaster
    └── test_panel.py                   # Cross-learning tests for interval forecasters

src/yohou/testing/                       # Check function library
├── __init__.py                          # Exports all check functions
├── generators.py                        # Check generators (_yield_yohou_forecaster_checks, _yield_yohou_transformer_checks)
├── common.py                            # Metadata routing checks (2 functions)
├── forecaster.py                        # Common forecaster checks (12 functions)
├── point.py                             # Point forecaster checks (2 functions)
├── interval.py                          # Interval forecaster checks (5 functions)
├── reduction.py                         # Reduction forecaster checks (2 functions)
├── panel.py                             # Panel data checks (3 functions)
├── transformer.py                       # Transformer checks (21 functions)
└── metadata_routing.py                  # Metadata routing test utilities
```

**Organization Principles**:
- Test files mirror source structure: `tests/{module}/test_{file}.py` for `src/yohou/{module}/{file}.py`
- Example: `src/yohou/point_forecaster/naive.py` → `tests/point_forecaster/test_naive.py`
- Check functions organized in `src/yohou/testing/` as a reusable library
- Shared testing infrastructure lives at `tests/` root
- Single global `conftest.py` sufficient - module-specific fixtures not needed

---

## Core Components

The testing infrastructure consists of three main components that work together to provide comprehensive forecaster validation.

### 1. Check Functions Library (`src/yohou/testing/`)

**Purpose**: Reusable library of 25 validation functions that test forecaster contracts organized by module (forecaster.py, point.py, interval.py, reduction.py, panel.py, common.py)

**Key Characteristics**:
- All functions raise `AssertionError` on failure (never return bool)
- Comprehensive docstrings with parameters and error conditions
- Functions accept `(forecaster, y, X, **kwargs)` signature
- Handle single X DataFrame (exogenous features) with "time" column
- Parameter validation checks ensure `forecasting_horizon >= 1` and `coverage_rates in (0, 1]` are enforced
- Support both global data and panel data with prefixed columns (e.g., "sales__store_1")

**Check Categories**:

**Common Forecaster Checks (13 functions)**:
1. **`check_fit_sets_forecaster_attributes`** - Validates fit() sets required attributes (fit_forecasting_horizon_, interval_, panel_group_names_, local_y_schema_, local_X_schema_, global_X_schema_, _y_observed, _X_observed, _X_t_observed)
2. **`check_forecaster_not_fitted_error`** - Ensures NotFittedError before fit() when accessing fitted attributes
3. **`check_predict_time_columns`** - Validates predictions have "observed_time" and "time" columns (note: "predicted_time" was renamed to "time")
4. **`check_update_extends_observations`** - Tests update() properly extends _y_observed, _X_observed buffers
5. **`check_reset_replaces_observations`** - Tests reset() replaces observation buffers correctly (_y_observed, _X_observed)
6. **`check_reset_propagates_to_transformers`** - Tests reset() cascades to nested transformers
7. **`check_forecasting_horizon_validation`** - Ensures forecasting_horizon < 1 raises ValueError in fit() and predict()
8. **`check_prediction_types_property`** - Validates prediction_types returns correct set ({"point"}, {"interval"}, or both)
9. **`check_clone_preserves_forecaster_params`** - sklearn's clone() preserves init parameters (enhanced to handle deeply nested estimators)
10. **`check_forecaster_tags_accessible_before_fit`** - Validates __sklearn_tags__() is accessible before fit() (tags are static capabilities)
11. **`check_forecaster_tags_static_after_fit`** - Ensures tag values don't change after fit() (forecaster_type, stateful, uses_reduction, supports_panel_data)
12. **`check_forecaster_tags_match_capabilities`** - Verifies tags accurately reflect actual behavior (forecaster_type vs prediction_types, uses_reduction vs estimator, transformer usage)
13. **`check_forecaster_methods_call_check_is_fitted`** - Validates methods (predict, update, reset) call check_is_fitted()

**Point Forecaster Checks (2 functions)**:
14. **`check_point_prediction_structure`** - Validates output has observed_time, predicted_time, and target columns only (no interval columns)
15. **`check_point_prediction_types`** - Ensures prediction_types == {"point"}

**Interval Forecaster Checks (5 functions)**:
16. **`check_interval_prediction_columns`** - Validates {col}_lower_{rate} and {col}_upper_{rate} format (handles both global and panel data with prefixed columns)
17. **`check_interval_bounds`** - Ensures upper >= lower for all coverage rates and time steps (handles panel data with prefixed columns)
18. **`check_interval_prediction_types`** - Validates prediction_types contains "interval"
19. **`check_coverage_rates_parameter`** - Validates coverage_rates is list of floats in (0, 1)
20. **`check_coverage_rates_validation`** - Ensures invalid coverage_rates (≤0, >1) raise ValueError in fit() and predict_interval()

**Reduction Forecaster Checks (2 functions)**:
21. **`check_estimator_parameter`** - Validates estimator is sklearn BaseEstimator
22. **`check_reduction_strategy`** - Validates reduction_strategy parameter exists

**Panel Data Forecaster Checks (3 functions)**:
23. **`check_panel_data`** - Validates panel_group_names=None predicts all groups in panel data
24. **`check_panel_single_group`** - Validates panel_group_names filters to specified list of group prefixes
25. **`check_panel_invalid_group_raises`** - Validates ValueError raised for invalid panel_group_names (list containing invalid groups)
24. **`check_column_forecaster_column_selection`** - Validates column selectors (str, list, slice, callable) work correctly for target columns
25. **`check_column_forecaster_remainder_drop`** - Tests remainder='drop' excludes unspecified columns (not yet implemented)
26. **`check_column_forecaster_remainder_passthrough`** - Tests remainder='passthrough' uses default forecaster (not yet implemented)
27. **`check_column_forecaster_remainder_custom`** - Tests custom remainder forecaster (not yet implemented)
28. **`check_column_forecaster_parallel_execution`** - Validates n_jobs parameter works correctly
29. **`check_column_forecaster_column_order_preserved`** - Ensures prediction columns match input target column order
30. **`check_target_transformed_forecaster_inverse`** - Tests inverse transformation round-trip (not yet implemented)
31. **`check_target_transformed_forecaster_check_inverse_warning`** - Validates check_inverse warns when transformation isn't reversible (not yet implemented)

### 2. Check Generator (`src/yohou/testing/generators.py`)

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
import pytest
from sklearn.base import clone

from yohou.{module} import MyForecaster
# Import from yohou.testing module
from yohou.testing import _yield_yohou_forecaster_checks

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

### Current State (After API Reorganization)

**Forecaster Tests** (as of reorganization):
- `tests/point_forecaster/test_naive.py`: Tests for SeasonalNaive (pattern-based forecaster)
- `tests/point_forecaster/test_reduction.py`: Tests for PointReductionForecaster (reduction-based)
- `tests/point_forecaster/test_panel.py`: Cross-learning tests for panel data
- `tests/interval_forecaster/test_reduction.py`: Tests for IntervalReductionForecaster
- `tests/interval_forecaster/test_panel.py`: Cross-learning tests for interval forecasters
- `tests/decomposition/test_*.py`: Tests for decomposition forecasters (PolynomialTrend, Seasonality, FourierSeasonality, ExponentialTrend, Decomposer)
- `tests/forecaster/test_composition.py`: Tests for ColumnForecaster (not yet implemented)

**Note**: After the API reorganization, tests need to be re-run to verify compatibility with the new structure.

### Validated Forecasters

**SeasonalNaive** (`src/yohou/point_forecaster/naive.py`):
- Extends `BasePointForecaster` directly
- Check generator validates 9+ checks automatically
- Seasonality-specific tests validate correct seasonal predictions
- Supports panel data with prefixed columns

**PointReductionForecaster** (`src/yohou/point_forecaster/reduction.py`):
- Extends both `BaseReductionForecaster` and `BasePointForecaster`
- Check generator validates 11+ checks automatically
- Analytical tests with LinearRegression validate exact numerical behavior
- Cross-learning tests validate panel data handling with prefixed columns
- Supports `time_weight` parameter for sample weighting

**IntervalReductionForecaster** (`src/yohou/interval_forecaster/reduction.py`):
- Extends both `BaseReductionForecaster` and `BaseIntervalForecaster`
- Check generator validates 15+ checks automatically
- Cross-learning tests validate panel data handling with prefixed columns
- Supports quantile regression for prediction intervals

**SplitConformalForecaster** (`src/yohou/interval_forecaster/split_conformal.py`):
- Extends `BaseIntervalForecaster` directly
- NOT yet reorganized - still uses old API patterns
- Provides conformal prediction intervals with calibration
- Supports similarity-weighted conformal prediction

**Decomposition Forecasters** (`yohou.decomposition`):
- **PolynomialTrendForecaster**: Extends `BasePointForecaster`, fits polynomial trends
- **PatternSeasonalityForecaster**: Extends `BasePointForecaster`, pattern-based seasonality (naive/average/median)
- **FourierSeasonalityForecaster**: Extends `BasePointForecaster`, Fourier basis with ElasticNet
- **Decomposer** (Meta-Forecaster): Extends `BasePointForecaster` and `_BaseComposition`
  - Sequential decomposition into trend + seasonality + residual components
  - NOT reorganized - remains in decomposition module as a meta-forecaster
  - Supports `store_residuals=True` for inspecting intermediate residuals

**ColumnForecaster** (`src/yohou/forecaster/composition.py`):
- Extends `BaseForecaster` and `_BaseComposition`
- Applies different forecasters to different target columns
- Supports parallel execution via `n_jobs`
- NOT yet fully tested - composition checks need implementation

**SearchCV** (`src/yohou/model_selection/search.py`):
- Meta-forecaster wrapper extending `BaseForecaster`
- Implements dynamic method availability via `_search_forecaster_has(attr)` pattern
- Conditionally exposes `predict()`, `predict_interval()`, `update_predict()`, `update_predict_interval()` based on `best_forecaster_.prediction_types`
- All methods accept `panel_group_names` parameter (list of group prefixes or None for all)
- Follows sklearn's `GridSearchCV`/`RandomizedSearchCV` pattern for meta-estimator method delegation
- Systematic tests validate: method availability before/after fit, delegation to best_forecaster_, panel data support

---

## Implementation Insights

### Forecaster vs Transformer Differences

**Fitted Attributes**:
- **Transformers**: `feature_names_in_`, `n_features_in_`, `_observation_horizon`, `_X_observed`
- **Forecasters**: `fit_forecasting_horizon_`, `interval_`, `panel_group_names_`, `local_y_schema_`, `local_X_schema_`, `global_X_schema_`, `local_y_t_schema_`, `local_X_t_schema_`, `_y_observed`, `_X_observed`, `_X_t_observed`, `target_transformer_`, `feature_transformer_`

**Schema Attributes**:
- `local_y_schema_`, `local_X_schema_`: Schemas with **unprefixed** column names for original data
- `local_y_t_schema_`, `local_X_t_schema_`: Schemas with **unprefixed** column names for transformed data
- `global_X_schema_`: Schema for global X columns (appear alongside panel groups)

**Lifecycle Methods**:
- **Transformers**: `fit(X, y=None)`, `transform(X)`, `update(X)`, `reset(X)`
- **Forecasters**: `fit(y, X, forecasting_horizon)`, `predict(X, forecasting_horizon, panel_group_names)`, `update(y, X, panel_group_names)`, `reset(y, X, panel_group_names)`
  - Note: `panel_group_names` is a **list of strings** (e.g., `["sales", "inventory"]`) or `None` for all groups

**Time Columns**:
- **Transformers**: Input/output both have "time" column
- **Forecasters**: Predictions add "observed_time" (when forecast was made) and "time" (time step being predicted)

**Panel Data Representation**:
- Internal: `dict[str, pl.DataFrame]` with unprefixed column names (e.g., {"sales": df_with_cols["time", "store_1", "store_2"]})
- External (predictions): Prefixed column names (e.g., "sales__store_1", "sales__store_2")

**Note**: All features in `X` are expected to be known ex-ante (in advance). For ex-post features (observed after the fact), use `ColumnForecaster` to forecast them first.

**Composition Classes**:
- **ColumnForecaster** (`src/yohou/forecaster/composition.py`): Applies different forecasters to different target columns, concatenates predictions horizontally
- **Decomposer** (`src/yohou/decomposition/decomposer.py`): Applies forecasters sequentially to residuals, sums predictions vertically

**Lifecycle Methods**:
- **Transformers**: `fit(X, y=None)`, `transform(X)`, `update(X)`, `reset(X)`
- **Forecasters**: `fit(y, X, forecasting_horizon)`, `predict(forecasting_horizon, X)`, `update(y, X)`, `reset(y, X)`

**Time Columns**:
- **Transformers**: Input/output both have "time" column
- **Forecasters**: Predictions add "observed_time" (when forecast was made) and "predicted_time" (time step being predicted)

**Composition Classes**:
- **ColumnForecaster**: Applies different forecasters to different target columns, concatenates predictions
- **SearchCV** (`src/yohou/model_selection/search.py`): Meta-forecaster wrapper for hyperparameter optimization
  - Uses `_search_forecaster_has(attr)` pattern following sklearn's `_estimator_has` for conditional method availability
  - Methods only available when `refit=True` and after fitting
  - Supports both point and interval forecasters through `prediction_types` checking
  - All methods accept `panel_group_names` parameter (list of group prefixes)

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

### Meta-Forecaster Testing Pattern

**Meta-forecasters** like SearchCV and Decomposer wrap other forecasters and delegate method calls. They require special testing considerations:

**SearchCV Testing Pattern**:
```python
def test_search_cv_method_availability():
    """Test methods only available after fit with refit=True."""
    search = SearchCV(forecaster=PointReductionForecaster(), refit=True)

    # Before fit: methods should not be available
    assert not hasattr(search, 'predict')

    # After fit: methods available based on prediction_types
    search.fit(y, X, forecasting_horizon=3)
    assert hasattr(search, 'predict')  # Point forecaster
    assert not hasattr(search, 'predict_interval')  # Not interval forecaster

def test_search_cv_interval_forecaster():
    """Test interval-specific methods with interval forecaster."""
    search = SearchCV(forecaster=IntervalReductionForecaster(), refit=True)
    search.fit(y, X, forecasting_horizon=3)

    # Should have interval methods
    assert hasattr(search, 'predict_interval')
    assert hasattr(search, 'update_predict_interval')
```

**Key testing principles**:
- Test method availability before/after fit
- Test delegation to `best_forecaster_` with correct parameters
- Test `panel_group_names` parameter propagation (list of strings)
- Test both point and interval forecasters
- Verify `_search_forecaster_has(attr)` logic with different forecaster types

### Clone Parameter Validation

The `check_clone_preserves_forecaster_params` function handles nested estimators correctly, including special handling for meta-estimators:

```python
# For nested estimators like LinearRegression, check type and params
if hasattr(orig_val, "get_params"):
    assert type(orig_val) == type(cloned_val), \
        f"Parameter {key}: different types"
    assert orig_val.get_params() == cloned_val.get_params(), \
        f"Parameter {key}: different estimator params"

# For list of (name, estimator) tuples (meta-estimators like Decomposer, FeaturePipeline)
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
- `panel_group_names_`, `local_y_columns_`, `local_X_columns_` are set
- `_y_observed` and `_X_observed` buffers exist

### Composition Class Validation

**Purpose**: Verify ColumnForecaster and Decomposer implement composition patterns correctly

**ColumnForecaster Validation**:
- Column selectors (str, list, slice, callable) properly index target columns
- Parallel execution with `n_jobs` produces same results as sequential
- Prediction column order matches input target column order
- Panel data (prefixed columns) handled correctly across different forecasters

**Decomposer Validation**:
- Sequential residual modeling: each component models residuals from previous components
- Final prediction equals sum of all component predictions
- `store_residuals=True` captures intermediate residuals for inspection
- Supports both additive (default) and multiplicative (via LogTransform) decomposition

### Time Column Validation

**Purpose**: Ensure predictions have dual time columns for alignment

**Validation**:
- "observed_time" column exists (when forecast was made)
- "time" column exists (time step being predicted - note: renamed from "predicted_time")
- Both columns have datetime dtype
- Number of rows matches forecasting_horizon

**Usage**: These dual time columns enable:
- Tracking forecast vintage (when was this forecast made?)
- Aligning predictions with actual observations for scoring
- Rolling window evaluation patterns

### Forecasting Horizon and Coverage Rates Validation

**Purpose**: Ensure all forecasters validate parameters at fit/predict time

**Checks**:
- **`check_forecasting_horizon_validation`**: Tests that `forecasting_horizon < 1` raises ValueError during fit()
  - Validates error message mentions "forecasting_horizon" or "positive"
  - Tests both `forecasting_horizon=0` and `forecasting_horizon=-1`

- **`check_coverage_rates_validation`**: Tests that invalid coverage_rates raise ValueError during fit() and predict_interval()
  - Validates `coverage_rates=[0.0]` (boundary - invalid)
  - Validates `coverage_rates=[1.5]` (above 1 - invalid)
  - Validates `coverage_rates=[-0.5]` (negative - invalid)
  - Tests both fit() and predict_interval() enforce validation

**Implementation Details**:
- Base classes (`BasePointForecaster`, `BaseIntervalForecaster`) implement `_validate_fit_params()` and `_validate_predict_params()`
- `_validate_predict_params()` resolves None defaults then delegates to `_validate_fit_params()` for DRY
- All concrete forecasters call validation at start of fit() method
- predict() and update_predict() methods in base classes automatically call `_validate_predict_params()`

**Test Coverage** (`tests/test_parameter_validation.py`):
- `test_point_forecaster_horizon_validation`: Tests PointReductionForecaster and SeasonalNaive
- `test_interval_forecaster_horizon_validation`: Tests IntervalReductionForecaster and SplitConformalForecaster
- `test_interval_forecaster_coverage_rates_validation`: Tests IntervalReductionForecaster
- `test_predict_validates_horizon`: Verifies predict() enforces validation
- `test_predict_interval_validates_coverage_rates`: Verifies predict_interval() enforces validation

**Validation Rules**:
- `forecasting_horizon`: Must be integer `>= 1`
- `coverage_rates`: List of floats, each in `(0, 1]` (exclusive 0, inclusive 1)
  - Default: `[0.95]` for interval forecasters

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
from yohou.testing import _yield_yohou_forecaster_checks
```

**Why**: Check functions are organized in `src/yohou/testing/` and exported through the module's `__init__.py` for easy access.

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
def test_point_reduction_panel_checks(forecaster, tags, expected_failures, panel_data):
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
1. **`check_panel_data`**: Validates `panel_group_names=None` predicts all groups
2. **`check_panel_single_group`**: Validates filtering to specific struct column
3. **`check_panel_invalid_group_raises`**: Validates error handling for invalid groups

**Panel Data Structure**:
- Single struct column: `{"time": ..., "stores": pl.DataFrame({"store_0": ..., "store_1": ..., ...})}`
- Framework constraint: All struct columns must have same field names
- Interval forecasters nest interval bounds within structs: `store_0_lower_0.1` inside `"stores"` struct

**Key Considerations**:
- `panel_group_names` operates on group prefix level using list of strings (e.g., `["sales", "inventory"]`), not individual series suffixes
- For interval forecasters, unnest struct columns to access interval bounds
- Expected failure for `IntervalReductionForecaster` with default estimator: `check_interval_bounds` (QuantileRegressor doesn't guarantee monotonic bounds)

---

## References

- sklearn source: `sklearn/utils/estimator_checks.py` (regressor/classifier checks)
- sklearn patterns: `sklearn/tests/test_common.py` (parametrize_with_checks)
- Transformer testing: `.github/copilot/transformer-testing-infrastructure.md`
- yohou architecture: `.github/copilot-instructions.md` (base classes, data flow)
