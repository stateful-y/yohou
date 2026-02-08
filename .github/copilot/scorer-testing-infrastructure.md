# Scorer Testing Infrastructure

**Status**: ✅ Implemented (January 2026)

## Overview

This document describes the systematic testing infrastructure for **forecasting metrics (scorers)** in yohou. Scorers evaluate forecast quality by comparing predictions against ground truth.

**Philosophy**: Function-based testing with `@pytest.mark.parametrize` using systematic check functions and generators, eliminating class-based test duplication.

**What This Guide Covers**:
- Scorer tag system and validation patterns
- 11 systematic check functions for scorer validation
- Test generators for automated test discovery
- Migration guide from class-based to function-based tests
- Real examples from production scorers

**Related Guides**:
- [Forecaster Testing Infrastructure](forecaster-testing-infrastructure.md): Base patterns
- [Splitter Testing Infrastructure](splitter-testing-infrastructure.md): Cross-validation testing
- [Transformer Testing Infrastructure](transformer-testing-infrastructure.md): Transform testing

---

## How It Works

### Scorer Types in Yohou

Yohou scorers fall into two categories based on prediction type:

1. **Point Scorers**: Evaluate point forecasts (single values)
   - Examples: MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError
   - Input: `y_truth` (actual values), `y_pred` (predicted values)
   - Output: Scalar score (lower is better for error metrics)

2. **Interval Scorers**: Evaluate prediction intervals (ranges)
   - Examples: IntervalWidth, IntervalCoverage, PinballLoss
   - Input: `y_truth`, `y_pred` with `_lower` and `_upper` columns
   - Output: Scalar score (depends on metric)

### Tag System

**ScorerTags** in `src/yohou/utils/tags.py`:

```python
@dataclass
class ScorerTags:
    prediction_type: Literal["point", "interval"] | None = None
    lower_is_better: bool = True
    requires_calibration: bool = False
```

**Tag Meanings**:
- `prediction_type`: Type of predictions this scorer evaluates ("point" or "interval")
- `lower_is_better`: Whether lower scores are better (True for error metrics, False for R²)
- `requires_calibration`: Whether scorer needs fit() with calibration data (e.g., conformity scorers)

### Testing Pattern

**Before: Class-based tests (legacy pattern)**
```python
class TestMeanAbsoluteError:
    def test_tags_accessible(self):
        tags = MeanAbsoluteError.__sklearn_tags__()
        assert tags.scorer_tags.prediction_type == "point"
    
    def test_score_basic(self):
        scorer = MeanAbsoluteError()
        score = scorer.score(y_truth, y_pred)
        assert score >= 0
    
    def test_aggregation_methods(self):
        scorer = MeanAbsoluteError()
        # Test each aggregation method
        ...

class TestMeanSquaredError:
    def test_tags_accessible(self):
        # Duplicate code
        pass
    
    def test_score_basic(self):
        # Duplicate code with slight variations
        pass
```

**After: Function-based tests (current pattern)**
```python
from yohou.testing import _yield_yohou_scorer_checks

@pytest.mark.parametrize(
    "check_name,check_func,check_kwargs",
    list(_yield_yohou_scorer_checks(
        MeanAbsoluteError(),
        y_truth,
        y_pred
    ))
)
def test_mae_systematic(check_name, check_func, check_kwargs):
    """Systematic tests for MeanAbsoluteError."""
    check_func(**check_kwargs)
```

**Benefits**:
- **DRY**: Single parametrized test function vs. many duplicated test classes
- **Coverage**: Automatically runs all applicable checks
- **Maintenance**: Add check once, applies to all scorers
- **Readability**: Clear check names in pytest output

---

## Core Components

### 1. Systematic Check Functions

**Location**: `src/yohou/testing/scorer.py` (446 lines, 10 functions)

#### Tag Validation Checks (3 functions)

```python
def check_scorer_tags_accessible_before_fit(scorer_class):
    """Validate tags accessible on class without instantiation.
    
    Ensures __sklearn_tags__() works on the class itself.
    """
    tags = scorer_class.__sklearn_tags__()
    assert tags.estimator_type == "scorer"
    assert tags.scorer_tags is not None
```

```python
def check_scorer_tags_static_after_fit(scorer, y_truth, y_pred):
    """Validate tags don't change after fit().
    
    Tags should be static metadata, not affected by data.
    """
    tags_before = scorer.__sklearn_tags__()
    scorer.score(y_truth, y_pred)
    tags_after = scorer.__sklearn_tags__()
    assert tags_before == tags_after
```

```python
def check_scorer_tags_match_capabilities(scorer, y_truth, y_pred, expected_tags):
    """Validate tags match actual behavior.
    
    Example expected_tags:
        {"prediction_type": "point", "lower_is_better": True}
    """
    tags = scorer.__sklearn_tags__()
    for tag_name, expected_value in expected_tags.items():
        actual_value = getattr(tags.scorer_tags, tag_name)
        assert actual_value == expected_value
```

#### Functionality Checks (6 functions)

```python
def check_scorer_prediction_type_compatibility(scorer, y_truth, y_pred):
    """Validate scorer works with appropriate prediction format.
    
    Point scorers: y_pred has columns like "value"
    Interval scorers: y_pred has columns like "value_lower", "value_upper"
    """
    tags = scorer.__sklearn_tags__()
    
    if tags.scorer_tags.prediction_type == "point":
        # Should work with point predictions
        score = scorer.score(y_truth, y_pred)
        assert isinstance(score, (int, float))
    elif tags.scorer_tags.prediction_type == "interval":
        # Should work with interval predictions
        assert any("_lower" in col for col in y_pred.columns)
        assert any("_upper" in col for col in y_pred.columns)
        score = scorer.score(y_truth, y_pred)
        assert isinstance(score, (int, float))
```

```python
def check_scorer_lower_is_better(scorer):
    """Validate lower_is_better tag is boolean.
    
    Critical for GridSearchCV/RandomizedSearchCV optimization direction.
    """
    tags = scorer.__sklearn_tags__()
    assert isinstance(tags.scorer_tags.lower_is_better, bool)
```

```python
def check_scorer_aggregation_methods(scorer, y_truth, y_pred, aggregation_methods):
    """Validate all aggregation methods work (parametric).
    
    Tests: ["timewise", "componentwise", "timewise_componentwise"]
    
    Each method should:
    - Return numeric score
    - Be deterministic (same inputs → same output)
    """
    for method in aggregation_methods:
        scorer_with_agg = scorer.set_params(aggregation_method=[method])
        score = scorer_with_agg.score(y_truth, y_pred)
        assert isinstance(score, (int, float))
        
        # Deterministic check
        score2 = scorer_with_agg.score(y_truth, y_pred)
        assert score == score2
```

```python
def check_scorer_panel_subselection(scorer, y_truth_panel, y_pred_panel, panel_group_names):
    """Validate panel group subselection works.
    
    When scorer is initialized with panel_group_names=["sales"],
    it should only score series with "sales" prefix.
    """
    scorer_subset = scorer.set_params(panel_group_names=panel_group_names)
    score = scorer_subset.score(y_truth_panel, y_pred_panel)
    assert isinstance(score, (int, float))
```

```python
def check_scorer_component_subselection(scorer, y_truth, y_pred, component_names):
    """Validate component subselection works.
    
    When scorer is initialized with component_names=["value"],
    it should only score the "value" column.
    """
    scorer_subset = scorer.set_params(component_names=component_names)
    score = scorer_subset.score(y_truth, y_pred)
    assert isinstance(score, (int, float))
```

```python
def check_scorer_coverage_rate_subselection(scorer, y_truth, y_pred_interval, coverage_rates):
    """Validate coverage rate subselection works (interval scorers only).
    
    When scorer is initialized with coverage_rates=[0.9],
    it should only score the 0.9 coverage interval.
    """
    scorer_subset = scorer.set_params(coverage_rates=coverage_rates)
    score = scorer_subset.score(y_truth, y_pred_interval)
    assert isinstance(score, (int, float))
```

#### Parameter Validation Check (1 parametric function)

```python
def check_scorer_parameter_validation(scorer_class, param_name, invalid_value, error_match):
    """Validate parameter constraints are enforced.
    
    Tests that invalid parameter values raise errors with expected message.
    
    Example:
        check_scorer_parameter_validation(
            MeanAbsoluteError,
            "panel_group_names",
            ["nonexistent"],
            "panel_group_names"
        )
    """
    scorer = scorer_class()
    with pytest.raises((ValueError, TypeError), match=error_match):
        scorer.set_params(**{param_name: invalid_value})
        # Trigger validation
        scorer.score(y_truth, y_pred)
```

**Common Parameter Constraints**:
- `panel_group_names`: Must exist in data
- `component_names`: Must exist in data
- `aggregation_method`: Must be in ["timewise", "componentwise", "timewise_componentwise"]
- `coverage_rates`: Must be in (0, 1) for interval scorers

### 2. Test Generator

**Location**: `src/yohou/testing/generators.py` (lines ~636-757)

```python
def _yield_yohou_scorer_checks(scorer, y_truth, y_pred, tags=None):
    """Generate applicable checks for a scorer.
    
    Auto-detects scorer capabilities from tags and
    yields appropriate check functions.
    
    Parameters
    ----------
    scorer : BaseScorer instance
        Scorer to test (can be fitted or unfitted)
    y_truth : pl.DataFrame
        Ground truth with "time" column
    y_pred : pl.DataFrame
        Predictions with "observed_time" and "time" columns
    tags : dict, optional
        Expected tag overrides for validation
    
    Yields
    ------
    (check_name, check_func, check_kwargs) : tuple
        Check name, function, and bound arguments
    
    Example
    -------
    >>> scorer = MeanAbsoluteError()
    >>> for name, func, kwargs in _yield_yohou_scorer_checks(scorer, y_truth, y_pred):
    ...     print(name)
    ...     func(**kwargs)
    check_scorer_tags_accessible_before_fit
    check_scorer_tags_static_after_fit
    ...
    """
    # Always yield tag checks
    yield ("check_scorer_tags_accessible_before_fit",
           check_scorer_tags_accessible_before_fit,
           {"scorer_class": type(scorer)})
    
    yield ("check_scorer_tags_static_after_fit",
           check_scorer_tags_static_after_fit,
           {"scorer": scorer, "y_truth": y_truth, "y_pred": y_pred})
    
    # Conditional tag validation
    if tags:
        yield ("check_scorer_tags_match_capabilities",
               check_scorer_tags_match_capabilities,
               {"scorer": scorer, "y_truth": y_truth, "y_pred": y_pred,
                "expected_tags": tags})
    
    # Functionality checks
    yield ("check_scorer_prediction_type_compatibility",
           check_scorer_prediction_type_compatibility,
           {"scorer": scorer, "y_truth": y_truth, "y_pred": y_pred})
    
    yield ("check_scorer_lower_is_better",
           check_scorer_lower_is_better,
           {"scorer": scorer})
    
    # Aggregation methods (parametric)
    for method in ["timewise", "componentwise"]:
        yield (f"check_scorer_aggregation_methods[{method}]",
               check_scorer_aggregation_methods,
               {"scorer": scorer, "y_truth": y_truth, "y_pred": y_pred,
                "aggregation_methods": [method]})
    
    # Panel subselection (if panel data)
    _, panel_groups = inspect_locality(y_truth)
    if panel_groups:
        yield ("check_scorer_panel_subselection",
               check_scorer_panel_subselection,
               {"scorer": scorer, "y_truth_panel": y_truth, "y_pred_panel": y_pred,
                "panel_group_names": list(panel_groups.keys())[:1]})
    
    # Component subselection
    component_names = [col for col in y_truth.columns if col != "time"]
    if component_names:
        yield ("check_scorer_component_subselection",
               check_scorer_component_subselection,
               {"scorer": scorer, "y_truth": y_truth, "y_pred": y_pred,
                "component_names": component_names[:1]})
    
    # Coverage rate subselection (interval scorers only)
    scorer_tags = scorer.__sklearn_tags__()
    if scorer_tags.scorer_tags.prediction_type == "interval":
        # Check if y_pred has interval columns
        if any("_lower" in col for col in y_pred.columns):
            yield ("check_scorer_coverage_rate_subselection",
                   check_scorer_coverage_rate_subselection,
                   {"scorer": scorer, "y_truth": y_truth, "y_pred_interval": y_pred,
                    "coverage_rates": [0.9]})
    
    # Parameter validation (parametric)
    param_tests = [
        ("panel_group_names", ["nonexistent"], "panel_group_names"),
        ("component_names", ["nonexistent"], "component_names"),
        ("aggregation_method", [["invalid"]], "aggregation_method"),
    ]
    
    for param_name, invalid_value, error_match in param_tests:
        yield (f"check_scorer_parameter_validation[{param_name}]",
               check_scorer_parameter_validation,
               {"scorer_class": type(scorer),
                "param_name": param_name,
                "invalid_value": invalid_value,
                "error_match": error_match})
```

**Key Features**:
- **Auto-detection**: Inspects `__sklearn_tags__()` to determine capabilities
- **Conditional yielding**: Only yields coverage_rate check for interval scorers
- **Parametric expansion**: Generates separate checks for each aggregation method
- **Data-driven**: Detects panel data and components from input DataFrames
- **Extensible**: Add new checks by appending yield statements

### 3. Usage in Test Files

**Pattern 1: Meta-testing (tests/testing/test_scorer.py)**

```python
"""Tests for scorer check functions themselves."""

def test_check_scorer_tags_accessible_before_fit():
    """Validate the check function works."""
    check_scorer_tags_accessible_before_fit(MeanAbsoluteError)
    check_scorer_tags_accessible_before_fit(MeanSquaredError)
```

**Pattern 2: Systematic scorer testing (tests/metrics/test_point.py)**

```python
"""Tests for point scorer implementations."""

@pytest.fixture
def y_truth():
    return pl.DataFrame({
        "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
        "value": range(10),
    })

@pytest.fixture
def y_pred():
    return pl.DataFrame({
        "observed_time": [datetime(2020, 1, 1)] * 10,
        "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
        "value": [i + 0.1 for i in range(10)],
    })

@pytest.mark.parametrize(
    "check_name,check_func,check_kwargs",
    list(_yield_yohou_scorer_checks(
        MeanAbsoluteError(),
        pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
            "value": range(10),
        }),
        pl.DataFrame({
            "observed_time": [datetime(2020, 1, 1)] * 10,
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
            "value": [i + 0.1 for i in range(10)],
        })
    ))
)
def test_mae_systematic(check_name, check_func, check_kwargs):
    """Systematic checks for MeanAbsoluteError."""
    check_func(**check_kwargs)
```

**Pattern 3: Specific scorer logic tests**

```python
def test_mae_perfect_prediction(y_truth):
    """Test MAE is zero for perfect predictions."""
    scorer = MeanAbsoluteError()
    y_pred = y_truth.with_columns(pl.lit(datetime(2020, 1, 1)).alias("observed_time"))
    score = scorer.score(y_truth, y_pred)
    assert score == 0.0

def test_mse_penalizes_large_errors():
    """Test MSE penalizes large errors more than MAE."""
    y_truth = pl.DataFrame({
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        "value": [0.0, 0.0],
    })
    y_pred = pl.DataFrame({
        "observed_time": [datetime(2020, 1, 1)] * 2,
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        "value": [1.0, 2.0],
    })
    
    mae = MeanAbsoluteError().score(y_truth, y_pred)
    mse = MeanSquaredError().score(y_truth, y_pred)
    
    # MSE = (1^2 + 2^2) / 2 = 2.5
    # MAE = (1 + 2) / 2 = 1.5
    assert mse > mae
```

---

## Implementation Details

### Scorer Base Class

**BaseScorer** (`src/yohou/metrics/base.py`):

```python
class BaseScorer:
    """Base class for all yohou scorers."""
    
    def __sklearn_tags__(self):
        return Tags(
            estimator_type="scorer",
            requires_fit=False,
            scorer_tags=ScorerTags(
                prediction_type=self._get_prediction_type(),
                lower_is_better=self._get_lower_is_better(),
            ),
        )
    
    def score(self, y_truth, y_pred, **params):
        """Compute score."""
        # Validate parameters
        self._validate_parameters(y_truth, y_pred, **params)
        
        # Compute metric
        return self._score_impl(y_truth, y_pred)
    
    def _validate_parameters(self, y_truth, y_pred, **params):
        """Validate panel_group_names, component_names, coverage_rates, aggregation_method."""
        # Lines 151-280 in base.py
        # Checks:
        # - panel_group_names exist in data
        # - component_names exist in data
        # - coverage_rates in (0, 1)
        # - aggregation_method valid
        pass
```

### Point Scorer Example

**MeanAbsoluteError** (`src/yohou/metrics/point.py`):

```python
class MeanAbsoluteError(BaseScorer):
    """Mean Absolute Error for point forecasts.
    
    MAE = mean(|y_truth - y_pred|)
    """
    
    def __sklearn_tags__(self):
        return Tags(
            estimator_type="scorer",
            requires_fit=False,
            scorer_tags=ScorerTags(
                prediction_type="point",
                lower_is_better=True,  # Lower MAE is better
            ),
        )
    
    def _score_impl(self, y_truth, y_pred):
        """Compute MAE."""
        # Extract value columns (exclude time columns)
        truth_values = y_truth.select(~cs.by_name("time"))
        pred_values = y_pred.select(~cs.by_name(["observed_time", "time"]))
        
        # Compute absolute error
        abs_error = (truth_values - pred_values).abs()
        
        # Aggregate according to aggregation_method
        return self._aggregate(abs_error)
```

### Interval Scorer Example

**IntervalCoverage** (`src/yohou/metrics/interval.py`):

```python
class IntervalCoverage(BaseScorer):
    """Interval coverage rate.
    
    Coverage = proportion of actuals within prediction intervals.
    """
    
    def __sklearn_tags__(self):
        return Tags(
            estimator_type="scorer",
            requires_fit=False,
            scorer_tags=ScorerTags(
                prediction_type="interval",
                lower_is_better=False,  # Higher coverage is better
            ),
        )
    
    def _score_impl(self, y_truth, y_pred_interval):
        """Compute coverage."""
        # Extract lower and upper bounds
        lower_cols = [col for col in y_pred_interval.columns if col.endswith("_lower")]
        upper_cols = [col for col in y_pred_interval.columns if col.endswith("_upper")]
        
        # Check if truth is within interval
        in_interval = (
            (y_truth >= y_pred_interval.select(lower_cols)) &
            (y_truth <= y_pred_interval.select(upper_cols))
        )
        
        # Compute proportion
        return in_interval.mean()
```

### Key Design Decisions

1. **Parameter Validation**: Centralized in `BaseScorer._validate_parameters()`
   - Validates panel_group_names, component_names, coverage_rates, aggregation_method
   - Raises informative errors with parameter names in message
   - Lines 151-280 in `src/yohou/metrics/base.py`

2. **Aggregation Methods**: Flexible scoring granularity
   - `"timewise"`: Average across time, separate per component
   - `"componentwise"`: Average across components, separate per time
   - `"timewise_componentwise"`: Single scalar (average everything)

3. **Subselection**: Filter data before scoring
   - `panel_group_names`: Score only specific panel groups
   - `component_names`: Score only specific components
   - `coverage_rates`: Score only specific coverage levels (interval scorers)

4. **No Calibration Yet**: `requires_calibration=False` for all current scorers
   - Future: Conformity-based interval scorers may need calibration

---

## How to Test a New Scorer

### Step 1: Implement `__sklearn_tags__()`

```python
class MyCustomScorer(BaseScorer):
    """My custom metric."""
    
    def __sklearn_tags__(self):
        return Tags(
            estimator_type="scorer",
            requires_fit=False,
            scorer_tags=ScorerTags(
                prediction_type="point",  # or "interval"
                lower_is_better=True,     # or False
            ),
        )
    
    def _score_impl(self, y_truth, y_pred):
        """Compute custom metric."""
        # Your logic here
        return score
```

### Step 2: Create Test File

```python
# tests/metrics/test_my_custom_scorer.py
from datetime import datetime, timedelta
import polars as pl
import pytest
from yohou.metrics import MyCustomScorer
from yohou.testing import _yield_yohou_scorer_checks

@pytest.fixture
def y_truth():
    return pl.DataFrame({
        "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
        "value": range(10),
    })

@pytest.fixture
def y_pred():
    return pl.DataFrame({
        "observed_time": [datetime(2020, 1, 1)] * 10,
        "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
        "value": [i + 0.1 for i in range(10)],
    })

@pytest.mark.parametrize(
    "check_name,check_func,check_kwargs",
    lambda: list(_yield_yohou_scorer_checks(
        MyCustomScorer(),
        pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
            "value": range(10),
        }),
        pl.DataFrame({
            "observed_time": [datetime(2020, 1, 1)] * 10,
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
            "value": [i + 0.1 for i in range(10)],
        })
    ))
)
def test_my_custom_scorer_systematic(check_name, check_func, check_kwargs):
    """Systematic checks for MyCustomScorer."""
    check_func(**check_kwargs)

def test_my_custom_scorer_specific_behavior(y_truth, y_pred):
    """Test MyCustomScorer specific logic."""
    scorer = MyCustomScorer()
    score = scorer.score(y_truth, y_pred)
    # Test custom behavior
```

### Step 3: Add Custom Checks (if needed)

```python
# In src/yohou/testing/scorer.py
def check_scorer_custom_property(scorer, y_truth, y_pred):
    """Validate custom scorer property."""
    # Your validation logic
    pass

# Export in src/yohou/testing/__init__.py
__all__ = [
    # ... existing exports ...
    "check_scorer_custom_property",
]

# Add to generator in src/yohou/testing/generators.py
def _yield_yohou_scorer_checks(scorer, y_truth, y_pred, tags=None):
    # ... existing checks ...
    
    # Conditional check for custom scorers
    if isinstance(scorer, MyCustomScorer):
        yield ("check_scorer_custom_property",
               check_scorer_custom_property,
               {"scorer": scorer, "y_truth": y_truth, "y_pred": y_pred})
```

---

## Success Metrics

### Coverage Goals

- **Tag Coverage**: All scorers implement `__sklearn_tags__()`
- **Check Coverage**: All 10 check functions tested in `tests/testing/test_scorer.py`
- **Systematic Coverage**: All production scorers use `_yield_yohou_scorer_checks()`

### Current Status (January 2026)

✅ **Implemented**:
- Point scorers: MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError, MeanAbsolutePercentageError, MedianAbsoluteError
- Interval scorers: IntervalWidth, IntervalCoverage, PinballLoss
- 10 systematic check functions
- 1 test generator with auto-detection
- Meta-tests in `tests/testing/test_scorer.py`

📋 **Remaining**:
- Add systematic tests to `tests/metrics/test_point.py` and `tests/metrics/test_interval.py`
- Refactor `tests/metrics/test_parameter_validation.py` (534 lines, 5 classes) to function-based

### Testing Infrastructure Health

**Check Function Quality**:
- ✅ Pure functions (no side effects except AssertionError)
- ✅ Single responsibility (one check per function)
- ✅ Clear error messages
- ✅ Parametric validation checks

**Generator Quality**:
- ✅ Auto-detection from tags
- ✅ Conditional yielding based on prediction_type
- ✅ Parametric expansion for aggregation methods
- ✅ Data-driven detection of panel groups and components
- ✅ Extensible design (add new checks easily)

**Test Coverage**:
- ✅ Meta-tests verify check functions work
- ⏳ Systematic tests in production test files (in progress)
- ⏳ Legacy test refactoring (in progress)

---

## Migration Example: test_parameter_validation.py

**Before: Class-based (534 lines, 5 classes, 21+ methods)**

```python
class TestMeanAbsoluteError:
    def test_invalid_panel_group_names(self):
        scorer = MeanAbsoluteError(panel_group_names=["nonexistent"])
        with pytest.raises(ValueError, match="panel_group_names"):
            scorer.score(y_truth, y_pred)
    
    def test_invalid_component_names(self):
        scorer = MeanAbsoluteError(component_names=["nonexistent"])
        with pytest.raises(ValueError, match="component_names"):
            scorer.score(y_truth, y_pred)
    
    # ... many more methods ...

class TestMeanSquaredError:
    def test_invalid_panel_group_names(self):
        # Duplicate code
        scorer = MeanSquaredError(panel_group_names=["nonexistent"])
        with pytest.raises(ValueError, match="panel_group_names"):
            scorer.score(y_truth, y_pred)
    
    # ... many more duplicate methods ...

# 3 more classes for other scorers...
```

**After: Function-based (~50 lines)**

```python
@pytest.mark.parametrize(
    "scorer_class",
    [MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError, MeanAbsolutePercentageError]
)
@pytest.mark.parametrize(
    "param_name,invalid_value,error_match",
    [
        ("panel_group_names", ["nonexistent"], "panel_group_names"),
        ("component_names", ["nonexistent"], "component_names"),
        ("aggregation_method", [["invalid"]], "aggregation_method"),
    ]
)
def test_scorer_parameter_validation(scorer_class, param_name, invalid_value, error_match, y_truth, y_pred):
    """Test parameter validation for all point scorers."""
    check_scorer_parameter_validation(scorer_class, param_name, invalid_value, error_match)
```

**Result**: 90% reduction in code, automatic coverage of new scorers.

---

## References

### Related Files

- **Tag System**: `src/yohou/utils/tags.py` (ScorerTags dataclass)
- **Base Class**: `src/yohou/metrics/base.py` (BaseScorer with _validate_parameters)
- **Point Scorers**: `src/yohou/metrics/point.py` (MAE, MSE, RMSE, MAPE, etc.)
- **Interval Scorers**: `src/yohou/metrics/interval.py` (Coverage, Width, Pinball)
- **Check Functions**: `src/yohou/testing/scorer.py` (10 functions)
- **Generator**: `src/yohou/testing/generators.py` (_yield_yohou_scorer_checks)
- **Meta-tests**: `tests/testing/test_scorer.py`
- **Production tests**: `tests/metrics/test_point.py`, `tests/metrics/test_interval.py`
- **Legacy tests**: `tests/metrics/test_parameter_validation.py` (to be refactored)

### Key Patterns

1. **Function-based testing**: `@pytest.mark.parametrize` with generators
2. **Auto-detection**: Tags drive which checks apply
3. **Systematic validation**: Consistent checks across all scorers
4. **Parametric expansion**: One check function → many test cases
5. **Data-driven**: Detect panel groups and components from input DataFrames

### Future Work

1. **Calibration Support**:
   - Implement scorers with `requires_calibration=True`
   - Add `check_scorer_calibration` to generator
   - Test calibration data flow

2. **Custom Aggregation**:
   - Allow user-defined aggregation functions
   - New check: `check_scorer_custom_aggregation`

3. **Multi-horizon Scoring**:
   - Score different horizons separately
   - New tag: `supports_horizon_subselection`
   - New check: `check_scorer_horizon_subselection`

---

**Last Updated**: January 17, 2026  
**Implementation**: Steps 1-5 complete, Step 6 in progress  
**Status**: Production-ready, active development
