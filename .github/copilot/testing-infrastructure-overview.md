# Yohou Testing Infrastructure Overview
## Complete Guide to Systematic Testing

**Last Updated**: January 20, 2026

---

## Quick Reference

Yohou provides **86 check functions** organized across **10 testing modules** plus **5 generator functions** and **6 metadata routing utilities** for comprehensive systematic validation of time series estimators.

| Component Type | Check Functions | Generator Functions | Metadata Utilities |
|----------------|-----------------|--------------------|--------------------|
| **Transformers** | 21 | 1 | - |
| **Forecasters** | 25 total (13+2+5+2+3) | 1 | - |
| **Scorers** | 11 | 1 | - |
| **Splitters** | 8 | 1 | - |
| **Search CVs** | 19 | 1 | - |
| **Common** | 2 | - | 6 |
| **Total** | **86** | **5** | **6** |

---

## Testing Module Breakdown

### 1. Transformer Testing (`src/yohou/testing/transformer.py`)

**21 check functions** organized in 3 categories:

**Core Yohou Checks (11 functions)**:
- `check_fit_sets_attributes` - Validates fit() sets required attributes
- `check_observation_horizon_not_fitted` - NotFittedError before fit()
- `check_observation_horizon_after_fit` - Valid non-negative integer after fit()
- `check_reset_updates_memory` - Verifies reset() updates _X_observed
- `check_update_concatenates_memory` - Ensures update() maintains horizon
- `check_update_transform_equivalence` - update().transform() == fit().transform()
- `check_insufficient_data_raises` - Error when data length < observation_horizon
- `check_transform_output_structure` - Valid output with "time" column
- `check_feature_names_out_match` - get_feature_names_out() matches output
- `check_inverse_transform_identity` - Round-trip test
- `check_panel_data_support` - Handles struct columns correctly

**Tag System Checks (3 functions)**:
- `check_tags_accessible_before_fit` - Tags accessible pre-fit
- `check_tags_static_after_fit` - Tags don't change after fit
- `check_tags_match_capabilities` - Tags reflect actual behavior

**Enhanced sklearn Checks (7 functions)**:
- `check_transformers_unfitted_stateless` - Stateless work without fit()
- `check_transformer_preserve_dtypes` - Dtype preservation
- `check_fit_idempotent` - Multiple fits yield consistent results
- `check_inverse_transform_round_trip` - Enhanced round-trip validation
- `check_fit_transform_equivalence` - fit_transform() == fit().transform()
- `check_memory_bounded` - Memory doesn't grow unbounded
- `check_transformer_methods_call_check_is_fitted` - Methods call check_is_fitted()

**Generator**: `_yield_yohou_transformer_checks(transformer, X_train, y_train, X_test, y_test, tags)`

**Detailed Guide**: [Transformer Testing Infrastructure](transformer-testing-infrastructure.md)

---

### 2. Forecaster Testing (`src/yohou/testing/forecaster.py`, `point.py`, `interval.py`, `reduction.py`, `panel.py`)

**25 total check functions** organized by module:

#### Common Forecaster Checks (13 functions - `forecaster.py`):
- `check_fit_sets_forecaster_attributes` - Required attributes after fit()
- `check_forecaster_not_fitted_error` - NotFittedError before fit()
- `check_predict_time_columns` - "observed_time" and "time" columns
- `check_update_extends_observations` - update() extends buffers
- `check_reset_replaces_observations` - reset() replaces buffers
- `check_reset_propagates_to_transformers` - Cascades to nested transformers
- `check_forecasting_horizon_validation` - horizon >= 1 enforcement
- `check_prediction_types_property` - Correct prediction_types set
- `check_clone_preserves_forecaster_params` - clone() preserves params
- `check_forecaster_tags_accessible_before_fit` - Tags pre-fit
- `check_forecaster_tags_static_after_fit` - Tags unchanging
- `check_forecaster_tags_match_capabilities` - Tags match behavior
- `check_forecaster_methods_call_check_is_fitted` - Methods call check_is_fitted()

#### Point Forecaster Checks (2 functions - `point.py`):
- `check_point_prediction_structure` - No interval columns
- `check_point_prediction_types` - prediction_types == {"point"}

#### Interval Forecaster Checks (5 functions - `interval.py`):
- `check_interval_prediction_columns` - {col}_lower_{rate}, {col}_upper_{rate}
- `check_interval_bounds` - upper >= lower for all rates
- `check_interval_prediction_types` - "interval" in prediction_types
- `check_coverage_rates_parameter` - List of floats in (0, 1)
- `check_coverage_rates_validation` - Invalid rates raise ValueError

#### Reduction Forecaster Checks (2 functions - `reduction.py`):
- `check_estimator_parameter` - sklearn BaseEstimator validation
- `check_reduction_strategy` - Strategy parameter exists

#### Panel Data Checks (3 functions - `panel.py`):
- `check_panel_data` - panel_group_names=None predicts all groups
- `check_panel_single_group` - Filters to specified groups
- `check_panel_invalid_group_raises` - ValueError for invalid groups

**Generator**: `_yield_yohou_forecaster_checks(forecaster, y_train, X_train, y_test, X_test, tags)`

**Detailed Guide**: [Forecaster Testing Infrastructure](forecaster-testing-infrastructure.md)

---

### 3. Scorer Testing (`src/yohou/testing/scorer.py`)

**11 check functions** in 3 categories:

**Tag System Checks (3 functions)**:
- `check_scorer_tags_accessible_before_fit` - Tags accessible pre-fit
- `check_scorer_tags_static_after_fit` - Tags unchanging
- `check_scorer_tags_match_capabilities` - Tags match behavior

**Functionality Checks (7 functions)**:
- `check_scorer_prediction_type_compatibility` - Compatibility with forecaster outputs
- `check_scorer_lower_is_better` - Correct lower_is_better property
- `check_scorer_aggregation_methods` - mean/median/sum aggregation
- `check_scorer_panel_subselection` - Filter by panel_group_names
- `check_scorer_component_subselection` - Filter by component_names
- `check_scorer_coverage_rate_subselection` - Filter by coverage_rates
- `check_scorer_methods_call_check_is_fitted` - Methods call check_is_fitted()

**Parameter Validation (1 function)**:
- `check_scorer_parameter_validation` - Input parameter constraints

**Generator**: `_yield_yohou_scorer_checks(scorer, y_truth, y_pred, tags)`

**Detailed Guide**: [Scorer Testing Infrastructure](scorer-testing-infrastructure.md)

---

### 4. Splitter Testing (`src/yohou/testing/splitter.py`)

**8 check functions** in 3 categories:

**Tag System Checks (3 functions)**:
- `check_splitter_tags_accessible_before_fit` - Tags accessible pre-fit
- `check_splitter_tags_static_after_fit` - Tags unchanging
- `check_splitter_tags_match_capabilities` - Tags match behavior

**Functionality Checks (4 functions)**:
- `check_splitter_produces_valid_indices` - Valid train/test indices
- `check_splitter_n_splits_consistency` - get_n_splits() matches actual splits
- `check_splitter_non_overlapping_tests` - Test sets don't overlap
- `check_splitter_panel_data_support` - Handles panel data correctly

**Parameter Validation (1 function)**:
- `check_splitter_parameter_constraints` - Constraint validation via _parameter_constraints

**Generator**: `_yield_yohou_splitter_checks(splitter, y, X, tags)`

**Detailed Guide**: [Splitter Testing Infrastructure](splitter-testing-infrastructure.md)

---

### 5. Search CV Testing (`src/yohou/testing/search.py`)

**19 check functions** in 4 categories:

**Common Search CV Checks (12 functions)**:
- `check_search_fit_sets_attributes` - Required attributes after fit()
- `check_search_not_fitted_error` - NotFittedError before fit()
- `check_search_cv_results_structure` - cv_results_ dictionary structure
- `check_search_refit_false_no_forecaster` - refit=False behavior
- `check_search_predict_delegates` - predict() delegates to best_forecaster_
- `check_search_update_delegates` - update() delegates to best_forecaster_
- `check_search_reset_delegates` - reset() delegates to best_forecaster_
- `check_search_score_delegates` - score() uses internal scorer
- `check_search_multimetric_scoring` - Multi-metric dict scoring
- `check_search_return_train_score` - return_train_score=True adds keys
- `check_search_error_score_handling` - error_score parameter handling
- `check_search_clone_preserves_params` - clone() preserves parameters

**GridSearchCV-Specific Checks (2 functions)**:
- `check_grid_search_exhaustive` - Evaluates all combinations
- `check_grid_search_param_grid_validation` - param_grid format validation

**RandomizedSearchCV-Specific Checks (3 functions)**:
- `check_randomized_search_n_iter` - n_iter controls evaluations
- `check_randomized_search_reproducibility` - random_state reproducibility
- `check_randomized_search_distributions` - scipy.stats distributions work

**Panel Data Checks (2 functions)**:
- `check_search_panel_data` - panel_group_names parameter propagation
- `check_search_method_availability` - @available_if logic with refit

**Generator**: `_yield_yohou_search_checks(search_cv, y_train, X_train, y_test, X_test, tags)`

**Detailed Guide**: [Search Testing Infrastructure](search-testing-infrastructure.md)

---

### 6. Common Checks (`src/yohou/testing/common.py`)

**2 metadata routing check functions**:
- `check_metadata_routing_default_request` - Default request is empty
- `check_metadata_routing_get_metadata_routing` - get_metadata_routing() works

**Applicability**: All estimators with metadata routing enabled (forecasters, transformers, scorers, splitters)

**Detailed Guide**: [sklearn Metadata Routing Implementation](sklearn-metadata-routing-implementation.md)

---

### 7. Metadata Routing Test Utilities (`src/yohou/testing/metadata_routing.py`)

**6 utilities** for metadata routing validation:

**Recording Utilities**:
- `record_metadata(obj, method)` - Decorator to track metadata calls
- `record_metadata_not_default(obj, method)` - Track non-default metadata
- `_Registry` class - Custom list for recording method calls

**Assertion Utilities**:
- `check_recorded_metadata(obj, method, record, **expected)` - Validate metadata
- `assert_request_is_empty(metadata_request)` - Check empty request
- `assert_request_equal(request, dictionary)` - Compare requests

**Usage Pattern**:
```python
from yohou.testing.metadata_routing import record_metadata, check_recorded_metadata

# Wrap estimator methods to record metadata
estimator = record_metadata(MyForecaster(), "fit")

# Use estimator
estimator.fit(y, X, sample_weight=[...])

# Validate metadata was passed correctly
check_recorded_metadata(estimator, "fit", sample_weight=[...])
```

---

## Generator Functions

All generator functions follow the same pattern: dynamically yield applicable check functions based on estimator tags.

### Pattern

```python
def _yield_yohou_{component}_checks(
    estimator, 
    *data,  # Component-specific data (X, y, etc.)
    tags: dict[str, Any] | None = None
) -> Generator[tuple[str, Callable, dict], None, None]:
    """Generate applicable checks based on tags.
    
    Yields
    ------
    check_name : str
        Name of the check function
    check_func : Callable
        The check function to call
    check_kwargs : dict
        Keyword arguments to pass to check_func
    """
```

### Available Generators

1. **`_yield_yohou_transformer_checks`** - Transformer validation (21 checks)
2. **`_yield_yohou_forecaster_checks`** - Forecaster validation (25 checks)
3. **`_yield_yohou_scorer_checks`** - Scorer validation (11 checks)
4. **`_yield_yohou_splitter_checks`** - Splitter validation (8 checks)
5. **`_yield_yohou_search_checks`** - Search CV validation (19 checks)

### Usage Pattern

```python
from yohou.testing import _yield_yohou_forecaster_checks

# Generate and run checks
for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(
    forecaster_fitted, y_train, X_train, y_test, X_test,
    tags={"forecaster_type": "point", "uses_reduction": True}
):
    if check_name not in expected_failures:
        check_func(forecaster_fitted, **check_kwargs)
```

### Benefits

1. **Automatic Coverage**: Generates 8-15 checks per estimator automatically
2. **Tag-Driven**: Only yields checks relevant to estimator capabilities
3. **No Boilerplate**: Single parametrized test covers all systematic checks
4. **Clear Failures**: Expected failures documented explicitly
5. **Maintainable**: Adding new check auto-applies to all estimators

---

## File Organization

All testing infrastructure follows pytest conventions with centralized check function library:

```
tests/
├── conftest.py                          # Global fixtures (factories, registries)
├── test_estimator_checks.py             # sklearn compatibility tests
├── test_parameter_validation.py         # Parameter validation tests
├── forecaster/
│   └── test_composition.py              # ColumnForecaster, GridSearchCV/RandomizedSearchCV
├── point_forecaster/
│   ├── test_naive.py                    # Pattern-based forecasters
│   ├── test_reduction.py                # Reduction-based forecasters
│   └── test_panel.py                    # Cross-learning tests
├── interval_forecaster/
│   ├── test_reduction.py                # Interval reduction forecasters
│   ├── test_split_conformal.py          # Conformal prediction
│   └── test_panel.py                    # Cross-learning tests
├── decomposition/
│   ├── test_trend.py                    # Trend forecasters
│   ├── test_seasonality.py              # Seasonality forecasters
│   └── test_decomposer.py               # Meta-forecaster
├── preprocessing/
│   ├── test_stationarization.py         # Differencing transformers
│   └── test_window.py                   # Window transformers
├── model_selection/
│   ├── test_split.py                    # Cross-validation splitters
│   └── test_search.py                   # GridSearchCV, RandomizedSearchCV
├── metrics/
│   ├── test_point.py                    # Point metrics
│   ├── test_interval.py                 # Interval metrics
│   └── test_conformity.py               # Conformity scores
└── utils/
    └── test_*.py                        # Utility function tests

src/yohou/testing/                       # Check function library (86 functions)
├── __init__.py                          # Exports all public functions
├── generators.py                        # 5 generator functions
├── transformer.py                       # 21 transformer checks
├── forecaster.py                        # 13 common forecaster checks
├── point.py                             # 2 point forecaster checks
├── interval.py                          # 5 interval forecaster checks
├── reduction.py                         # 2 reduction forecaster checks
├── panel.py                             # 3 panel data checks
├── scorer.py                            # 11 scorer checks
├── splitter.py                          # 8 splitter checks
├── search.py                            # 19 search CV checks
├── common.py                            # 2 metadata routing checks
└── metadata_routing.py                  # 6 metadata utilities
```

**Organization Principles**:
1. Test files mirror source structure: `tests/{module}/test_{file}.py`
2. Check functions live in reusable library: `src/yohou/testing/`
3. Single global `conftest.py` for shared fixtures
4. Module-specific fixtures only when necessary

---

## Testing Workflow

### Step 1: Implement Estimator

Create your estimator following base class patterns:
- **Transformers**: Extend `BaseTransformer`
- **Point Forecasters**: Extend `BasePointForecaster`
- **Interval Forecasters**: Extend `BaseIntervalForecaster`
- **Scorers**: Extend `BaseScorer`
- **Splitters**: Extend `BaseSplitter`

### Step 2: Create Test File

Create `tests/{module}/test_{name}.py` mirroring source location:
```python
import pytest
from sklearn.base import clone
from yohou.{module} import MyEstimator
from yohou.testing import _yield_yohou_{component}_checks

@pytest.mark.parametrize(
    "estimator,tags,expected_failures",
    [
        (
            MyEstimator(param=value),
            {"capability1": True, "capability2": False},
            [],  # Or list of check names expected to fail
        ),
    ],
)
def test_my_estimator_checks(estimator, tags, expected_failures, data_fixture):
    """Run systematic checks on MyEstimator."""
    # Setup data
    # ...
    
    # Fit estimator
    estimator_fitted = clone(estimator)
    estimator_fitted.fit(...)
    
    # Run generated checks
    expected_failures_set = set(expected_failures)
    for check_name, check_func, check_kwargs in _yield_yohou_{component}_checks(
        estimator_fitted, ..., tags=tags
    ):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(estimator_fitted, **check_kwargs)
```

### Step 3: Add Component-Specific Tests

Below systematic checks, add tests for specific behaviors:
```python
def test_my_estimator_specific_behavior(data_fixture):
    """Test specific behavior unique to this estimator."""
    estimator = MyEstimator(param=value)
    # Test specific logic
    assert specific_condition
```

### Step 4: Run Tests

```bash
# Run all tests for module
uv run pytest tests/{module}/test_{name}.py -v

# Run single test
uv run pytest tests/{module}/test_{name}.py::test_name -v

# Run with debugging
uv run pytest tests/{module}/test_{name}.py::test_name --pdb
```

---

## Tag System

Tags drive generator functions to yield only applicable checks. All estimators must implement `__sklearn_tags__()` returning a `Tags` object.

### Common Tags

**All Estimators**:
- `estimator_type`: "transformer" | "forecaster" | "scorer" | "splitter"
- `requires_fit`: bool (always True for yohou estimators)

**Transformers**:
- `stateful`: bool - Has observation_horizon > 0
- `invertible`: bool - Has inverse_transform method

**Forecasters**:
- `forecaster_type`: "point" | "interval"
- `uses_reduction`: bool - Uses sklearn estimator
- `supports_panel_data`: bool - Handles panel data (prefixed columns)
- `uses_transformers`: bool - Has target/feature transformers

**Scorers**:
- `prediction_types`: set[str] - {"point", "interval", "both"}
- `multi_output`: bool - Handles multiple target columns
- `requires_positive_y`: bool - Needs positive values

**Splitters**:
- `supports_panel_data`: bool - Handles panel data
- `expanding_window`: bool - Expanding vs rolling window

### Tag Validation

All components include tag validation checks:
- `check_{component}_tags_accessible_before_fit` - Tags work pre-fit
- `check_{component}_tags_static_after_fit` - Tags don't change
- `check_{component}_tags_match_capabilities` - Tags reflect actual behavior

---

## Success Metrics

The testing infrastructure achieves these goals:

✅ **Coverage**: 65 check functions cover all estimator contracts
✅ **Maintainability**: Generator pattern eliminates test boilerplate
✅ **Clarity**: Descriptive assertions with expected vs actual values
✅ **Performance**: Session fixtures and parallelization for speed
✅ **Documentation**: Every check function has NumPy-style docstring

**Test Suite Performance**:
- Full suite: ~90 seconds (with coverage)
- Fast mode (no coverage): ~15 seconds
- Per-module tests: 1-5 seconds

---

## Common Patterns

### Expected Failures

Document known limitations explicitly:
```python
expected_failures = [
    "check_memory_bounded",  # Known: unbounded memory for streaming
]
```

### Panel Data Testing

Use panel data fixtures for cross-learning tests:
```python
def test_panel_forecaster(panel_data_fixture):
    y = panel_data_fixture["y"]  # Has prefixed columns: "sales__store_1", etc.
    forecaster.fit(y, forecasting_horizon=3)
    
    # Predict all groups
    y_pred_all = forecaster.predict(forecasting_horizon=3)
    
    # Predict specific groups
    y_pred_subset = forecaster.predict(
        forecasting_horizon=3, 
        panel_group_names=["sales"]  # List of group prefixes
    )
```

### Metadata Routing Validation

Test metadata propagation through pipelines:
```python
from yohou.testing.metadata_routing import record_metadata, check_recorded_metadata

# Wrap pipeline
pipeline = FeaturePipeline([
    ("transformer", record_metadata(MyTransformer(), "transform")),
    ("forecaster", record_metadata(MyForecaster(), "fit")),
])

# Use with metadata
pipeline.fit(y, X, sample_weight=[...])

# Validate propagation
check_recorded_metadata(pipeline.named_steps["forecaster"], "fit", sample_weight=[...])
```

---

## References

**Detailed Guides**:
- [Transformer Testing Infrastructure](transformer-testing-infrastructure.md) - 20 checks, 400+ lines
- [Forecaster Testing Infrastructure](forecaster-testing-infrastructure.md) - 24 checks, 720+ lines
- [Scorer Testing Infrastructure](scorer-testing-infrastructure.md) - 10 checks, 800+ lines
- [Splitter Testing Infrastructure](splitter-testing-infrastructure.md) - 8 checks, 730+ lines

**Implementation Guides**:
- [sklearn Metadata Routing Implementation](sklearn-metadata-routing-implementation.md) - Complete routing infrastructure
- [Creating New Forecasters](creating-new-forecasters.md) - Step-by-step guide
- [Architecture & Core Concepts](architecture-and-core-concepts.md) - Base classes and data flow

**sklearn References**:
- `sklearn/utils/estimator_checks.py` - Regressor/classifier checks
- `sklearn/tests/test_common.py` - parametrize_with_checks pattern
- `sklearn/utils/_testing.py` - MinimalTransformer/MinimalRegressor
