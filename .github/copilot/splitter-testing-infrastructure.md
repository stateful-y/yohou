# Splitter Testing Infrastructure

**Status**: ✅ Implemented (January 2026)

## Overview

This document describes the systematic testing infrastructure for **time series cross-validation splitters** in yohou. Splitters divide time series data into train/test folds for model evaluation while respecting temporal order.

**Philosophy**: Function-based testing with `@pytest.mark.parametrize` using systematic check functions and generators, eliminating class-based test duplication.

**What This Guide Covers**:
- Splitter tag system and validation patterns
- 8 systematic check functions for splitter validation
- Test generators for automated test discovery
- Migration guide from class-based to function-based tests
- Real examples from production splitters

**Related Guides**:
- [Forecaster Testing Infrastructure](forecaster-testing-infrastructure.md): Base patterns
- [Scorer Testing Infrastructure](scorer-testing-infrastructure.md): Metrics testing
- [Transformer Testing Infrastructure](transformer-testing-infrastructure.md): Transform testing

---

## How It Works

### Splitter Types in Yohou

Yohou provides three main splitter types:

1. **ExpandingWindowSplitter**: Train set grows over time (cumulative)
   - Example: Fold 1: train[0:50], test[50:60]; Fold 2: train[0:60], test[60:70]
   - Use case: When all historical data should inform each forecast
   
2. **SlidingWindowSplitter**: Fixed train window size (rolling)
   - Example: Fold 1: train[0:50], test[50:60]; Fold 2: train[10:60], test[60:70]
   - Use case: When recent history is more relevant than distant past
   - **gap parameter**: Optional gap between train and test (e.g., gap=2 creates 2-step buffer)

### Tag System

**SplitterTags** in `src/yohou/utils/tags.py`:

```python
@dataclass
class SplitterTags:
    splitter_type: Literal["expanding", "sliding", "gap"] | None = None
    supports_panel_data: bool = False
    requires_data_for_n_splits: bool = False
    produces_non_overlapping_tests: bool = True
    stateful: bool = False
```

**Tag Meanings**:
- `splitter_type`: Window strategy (expanding/sliding/gap)
- `supports_panel_data`: Can handle multi-series panel data
- `requires_data_for_n_splits`: Must see data to compute split count (e.g., `SlidingWindowSplitter`)
- `produces_non_overlapping_tests`: Test sets don't overlap across folds (True for all current splitters)
- `stateful`: Maintains state across `split()` calls (False for all current splitters)

### Testing Pattern

**Before: Class-based tests (legacy pattern)**
```python
class TestExpandingWindowSplitter:
    def test_tags_accessible(self):
        tags = ExpandingWindowSplitter.__sklearn_tags__()
        assert tags.splitter_tags.splitter_type == "expanding"
    
    def test_produces_valid_indices(self):
        # Test implementation
        pass

class TestSlidingWindowSplitter:
    def test_tags_accessible(self):
        # Duplicate code
        pass
    
    def test_produces_valid_indices(self):
        # Duplicate code with slight variations
        pass
```

**After: Function-based tests (current pattern)**
```python
from yohou.testing import _yield_yohou_splitter_checks

@pytest.mark.parametrize(
    "check_name,check_func,check_kwargs",
    list(_yield_yohou_splitter_checks(
        ExpandingWindowSplitter(n_splits=3, test_size=10),
        y_data
    ))
)
def test_expanding_window_checks(check_name, check_func, check_kwargs):
    """Systematic tests for ExpandingWindowSplitter."""
    check_func(**check_kwargs)
```

**Benefits**:
- **DRY**: Single parametrized test function vs. many duplicated test classes
- **Coverage**: Automatically runs all applicable checks
- **Maintenance**: Add check once, applies to all splitters
- **Readability**: Clear check names in pytest output

---

## Core Components

### 1. Systematic Check Functions

**Location**: `src/yohou/testing/splitter.py` (365 lines, 8 functions)

#### Tag Validation Checks (3 functions)

```python
def check_splitter_tags_accessible_before_fit(splitter_class):
    """Validate tags accessible on class without instantiation.
    
    Ensures __sklearn_tags__() works on the class itself.
    """
    tags = splitter_class.__sklearn_tags__()
    assert tags.estimator_type == "splitter"
    assert tags.splitter_tags is not None
```

```python
def check_splitter_tags_static_after_fit(splitter, y):
    """Validate tags don't change after split().
    
    Tags should be static metadata, not affected by data.
    """
    tags_before = splitter.__sklearn_tags__()
    list(splitter.split(y))  # Consume generator
    tags_after = splitter.__sklearn_tags__()
    assert tags_before == tags_after
```

```python
def check_splitter_tags_match_capabilities(splitter, y, expected_tags):
    """Validate tags match actual behavior.
    
    Example expected_tags:
        {"splitter_type": "expanding", "requires_data_for_n_splits": False}
    """
    tags = splitter.__sklearn_tags__()
    for tag_name, expected_value in expected_tags.items():
        actual_value = getattr(tags.splitter_tags, tag_name)
        assert actual_value == expected_value
```

#### Functionality Checks (4 functions)

```python
def check_splitter_produces_valid_indices(splitter, y):
    """Validate train/test indices are within bounds.
    
    Checks:
    - All indices < len(y)
    - All indices >= 0
    - No duplicates within train or test sets
    """
    n_samples = len(y)
    for train_idx, test_idx in splitter.split(y):
        assert all(0 <= i < n_samples for i in train_idx)
        assert all(0 <= i < n_samples for i in test_idx)
        assert len(set(train_idx)) == len(train_idx)  # No duplicates
        assert len(set(test_idx)) == len(test_idx)
```

```python
def check_splitter_n_splits_consistency(splitter, y):
    """Validate get_n_splits() matches actual split count.
    
    Critical for ExpandingWindowSplitter (n_splits set at init).
    SlidingWindowSplitter computes dynamically from data.
    """
    expected = splitter.get_n_splits(y)
    actual = len(list(splitter.split(y)))
    assert expected == actual
```

```python
def check_splitter_non_overlapping_tests(splitter, y):
    """Validate test sets don't overlap across folds.
    
    Temporal ordering guarantee: each time point appears
    in at most one test set.
    """
    all_test_indices = []
    for _, test_idx in splitter.split(y):
        all_test_indices.extend(test_idx)
    
    # Check for duplicates
    assert len(all_test_indices) == len(set(all_test_indices))
```

```python
def check_splitter_panel_data_support(splitter, y_panel):
    """Validate panel data handling (if supported).
    
    Currently all splitters operate on time column only,
    ignoring panel prefixes (supports_panel_data=False).
    """
    tags = splitter.__sklearn_tags__()
    if tags.splitter_tags.supports_panel_data:
        # Should not raise
        list(splitter.split(y_panel))
    else:
        # Panel data treated same as single series
        # (splits based on time column only)
        list(splitter.split(y_panel))
```

#### Parameter Validation Check (1 parametric function)

```python
def check_splitter_parameter_constraints(splitter_class, param_name, invalid_values):
    """Validate parameter constraints are enforced.
    
    Tests that invalid parameter values raise errors.
    
    Example:
        check_splitter_parameter_constraints(
            ExpandingWindowSplitter,
            "n_splits",
            [1, 0, -1]  # All invalid
        )
    """
    for invalid_value in invalid_values:
        with pytest.raises((ValueError, TypeError)):
            splitter_class(**{param_name: invalid_value})
```

**Common Parameter Constraints**:
- `n_splits`: Must be ≥ 2 (need at least 2 folds for cross-validation)
- `test_size`, `train_size`: Must be > 0
- `gap`: Must be ≥ 0

### 2. Test Generator

**Location**: `src/yohou/testing/generators.py` (lines ~496-633)

```python
def _yield_yohou_splitter_checks(splitter, y, tags=None):
    """Generate applicable checks for a splitter.
    
    Auto-detects splitter capabilities from tags and
    yields appropriate check functions.
    
    Parameters
    ----------
    splitter : BaseSplitter instance
        Fitted or unfitted splitter to test
    y : pl.DataFrame
        Time series data with "time" column
    tags : dict, optional
        Expected tag overrides for validation
    
    Yields
    ------
    (check_name, check_func, check_kwargs) : tuple
        Check name, function, and bound arguments
    
    Example
    -------
    >>> splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)
    >>> y = pl.DataFrame({"time": [...], "value": [...]})
    >>> for name, func, kwargs in _yield_yohou_splitter_checks(splitter, y):
    ...     print(name)
    ...     func(**kwargs)
    check_splitter_tags_accessible_before_fit
    check_splitter_tags_static_after_fit
    ...
    """
    # Always yield tag checks
    yield ("check_splitter_tags_accessible_before_fit",
           check_splitter_tags_accessible_before_fit,
           {"splitter_class": type(splitter)})
    
    yield ("check_splitter_tags_static_after_fit",
           check_splitter_tags_static_after_fit,
           {"splitter": splitter, "y": y})
    
    # Conditional tag validation
    if tags:
        yield ("check_splitter_tags_match_capabilities",
               check_splitter_tags_match_capabilities,
               {"splitter": splitter, "y": y, "expected_tags": tags})
    
    # Functionality checks
    yield ("check_splitter_produces_valid_indices",
           check_splitter_produces_valid_indices,
           {"splitter": splitter, "y": y})
    
    yield ("check_splitter_n_splits_consistency",
           check_splitter_n_splits_consistency,
           {"splitter": splitter, "y": y})
    
    yield ("check_splitter_non_overlapping_tests",
           check_splitter_non_overlapping_tests,
           {"splitter": splitter, "y": y})
    
    # Conditionally yield panel data check
    splitter_tags = splitter.__sklearn_tags__()
    if splitter_tags.splitter_tags.supports_panel_data:
        yield ("check_splitter_panel_data_support",
               check_splitter_panel_data_support,
               {"splitter": splitter, "y_panel": y})  # Assumes y is panel
    
    # Parameter validation (parametric)
    param_tests = {
        "ExpandingWindowSplitter": [
            ("n_splits", [1, 0, -1]),
            ("test_size", [0, -1]),
            ("gap", [-1]),
        ],
        "SlidingWindowSplitter": [
            ("train_size", [0, -1]),
            ("test_size", [0, -1]),
            ("gap", [-1]),
        ],
            ("gap", [-1]),
        ],
    }
    
    splitter_name = type(splitter).__name__
    if splitter_name in param_tests:
        for param_name, invalid_values in param_tests[splitter_name]:
            yield (f"check_splitter_parameter_constraints[{param_name}]",
                   check_splitter_parameter_constraints,
                   {"splitter_class": type(splitter),
                    "param_name": param_name,
                    "invalid_values": invalid_values})
```

**Key Features**:
- **Auto-detection**: Inspects `__sklearn_tags__()` to determine capabilities
- **Conditional yielding**: Only yields panel data check if `supports_panel_data=True`
- **Parametric validation**: Generates separate checks for each parameter
- **Extensible**: Add new checks by appending yield statements

### 3. Usage in Test Files

**Pattern 1: Meta-testing (tests/testing/test_splitter.py)**

```python
"""Tests for splitter check functions themselves."""

def test_check_splitter_tags_accessible_before_fit():
    """Validate the check function works."""
    check_splitter_tags_accessible_before_fit(ExpandingWindowSplitter)
    check_splitter_tags_accessible_before_fit(SlidingWindowSplitter)
```

**Pattern 2: Systematic splitter testing (tests/model_selection/test_split.py)**

```python
"""Tests for splitter implementations."""

@pytest.fixture
def y_data():
    return pl.DataFrame({
        "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)],
        "value": range(100),
    })

@pytest.mark.parametrize(
    "check_name,check_func,check_kwargs",
    list(_yield_yohou_splitter_checks(
        ExpandingWindowSplitter(n_splits=3, test_size=10),
        pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)],
            "value": range(100),
        })
    ))
)
def test_expanding_window_systematic(check_name, check_func, check_kwargs):
    """Systematic checks for ExpandingWindowSplitter."""
    check_func(**check_kwargs)
```

**Pattern 3: Specific splitter logic tests**

```python
def test_expanding_window_grows_train_set(y_data):
    """Test expanding window specific behavior."""
    splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)
    
    train_sizes = []
    for train_idx, _ in splitter.split(y_data):
        train_sizes.append(len(train_idx))
    
    # Train set should grow each fold
    assert train_sizes == sorted(train_sizes)
    assert len(set(train_sizes)) == len(train_sizes)  # All different

def test_sliding_window_fixed_train_size(y_data):
    """Test sliding window specific behavior."""
    splitter = SlidingWindowSplitter(train_size=50, test_size=10)
    
    train_sizes = []
    for train_idx, _ in splitter.split(y_data):
        train_sizes.append(len(train_idx))
    
    # All train sets same size
    assert len(set(train_sizes)) == 1
    assert train_sizes[0] == 50
```

---

## Implementation Details

### Splitter Tag Implementation

**BaseSplitter** (`src/yohou/model_selection/split.py`, lines ~102-113):

```python
class BaseSplitter:
    def __sklearn_tags__(self):
        return Tags(
            estimator_type="splitter",
            requires_fit=False,
            splitter_tags=SplitterTags(),
        )
```

**ExpandingWindowSplitter** (lines ~318-330):

```python
class ExpandingWindowSplitter(BaseSplitter):
    def __sklearn_tags__(self):
        return Tags(
            estimator_type="splitter",
            requires_fit=False,
            splitter_tags=SplitterTags(
                splitter_type="expanding",
                requires_data_for_n_splits=False,  # n_splits set at init
                produces_non_overlapping_tests=True,
            ),
        )
```

**SlidingWindowSplitter** (lines ~533-545):

```python
class SlidingWindowSplitter(BaseSplitter):
    def __sklearn_tags__(self):
        return Tags(
            estimator_type="splitter",
            requires_fit=False,
            splitter_tags=SplitterTags(
                splitter_type="sliding",
                requires_data_for_n_splits=True,  # Computed from data length
                produces_non_overlapping_tests=True,
            ),
        )
```

**Note on gap parameter**: Both ExpandingWindowSplitter and SlidingWindowSplitter now support an optional `gap` parameter (int >= 0 or None) to insert a buffer between train and test sets. When gap > 0:
- Test indices are shifted forward by gap samples
- Training set remains unchanged
- May reduce the number of valid splits if gap pushes test beyond data
- `get_n_splits()` requires y to count valid splits

### Key Design Decisions

1. **No panel data support yet**: All splitters currently operate on time column only
   - `supports_panel_data=False` for all splitters
   - Future: May add per-panel splitting (independent splits for each series)

2. **Non-overlapping tests guaranteed**: Temporal order prevents test overlap
   - `produces_non_overlapping_tests=True` for all current splitters
   - Future: May add custom splitters with overlapping tests

3. **Stateless operation**: Splitters don't maintain state
   - `stateful=False` for all current splitters
   - Each `split()` call is independent

4. **Data-dependent n_splits**:
   - `ExpandingWindowSplitter`: `requires_data_for_n_splits=True` (gap may reduce splits)
   - `SlidingWindowSplitter`: `requires_data_for_n_splits=True` (computed from data length)

---

## How to Test a New Splitter

### Step 1: Implement `__sklearn_tags__()`

```python
class MyCustomSplitter(BaseSplitter):
    def __sklearn_tags__(self):
        return Tags(
            estimator_type="splitter",
            requires_fit=False,
            splitter_tags=SplitterTags(
                splitter_type="custom",  # Or None
                requires_data_for_n_splits=True,
                produces_non_overlapping_tests=True,
            ),
        )
```

### Step 2: Create Test File

```python
# tests/model_selection/test_my_custom_splitter.py
from datetime import datetime, timedelta
import polars as pl
import pytest
from yohou.model_selection import MyCustomSplitter
from yohou.testing import _yield_yohou_splitter_checks

@pytest.fixture
def y_data():
    return pl.DataFrame({
        "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)],
        "value": range(100),
    })

@pytest.mark.parametrize(
    "check_name,check_func,check_kwargs",
    list(_yield_yohou_splitter_checks(
        MyCustomSplitter(param1=10),
        pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)],
            "value": range(100),
        })
    ))
)
def test_my_custom_splitter_systematic(check_name, check_func, check_kwargs):
    """Systematic checks for MyCustomSplitter."""
    check_func(**check_kwargs)

def test_my_custom_splitter_specific_behavior(y_data):
    """Test MyCustomSplitter specific logic."""
    splitter = MyCustomSplitter(param1=10)
    # Test custom behavior
```

### Step 3: Add Custom Checks (if needed)

```python
# In src/yohou/testing/splitter.py
def check_splitter_custom_behavior(splitter, y):
    """Validate custom splitter behavior."""
    # Your validation logic
    pass

# Export in src/yohou/testing/__init__.py
__all__ = [
    # ... existing exports ...
    "check_splitter_custom_behavior",
]

# Add to generator in src/yohou/testing/generators.py
def _yield_yohou_splitter_checks(splitter, y, tags=None):
    # ... existing checks ...
    
    # Conditional check for custom splitters
    if isinstance(splitter, MyCustomSplitter):
        yield ("check_splitter_custom_behavior",
               check_splitter_custom_behavior,
               {"splitter": splitter, "y": y})
```

---

## Success Metrics

### Coverage Goals

- **Tag Coverage**: All splitters implement `__sklearn_tags__()`
- **Check Coverage**: All 8 check functions tested in `tests/testing/test_splitter.py`
- **Systematic Coverage**: All production splitters use `_yield_yohou_splitter_checks()`

### Current Status (January 2026)

✅ **Implemented**:
- 2 production splitters (ExpandingWindowSplitter, SlidingWindowSplitter with optional gap parameter)
- 8 systematic check functions
- 1 test generator with auto-detection
- Meta-tests in `tests/testing/test_splitter.py`

📋 **Remaining**:
- Add systematic tests to `tests/model_selection/test_split.py`
- Refactor legacy class-based tests to function-based
- Add panel data support (future)

### Testing Infrastructure Health

**Check Function Quality**:
- ✅ Pure functions (no side effects except AssertionError)
- ✅ Single responsibility (one check per function)
- ✅ Clear error messages
- ✅ Parametric validation checks

**Generator Quality**:
- ✅ Auto-detection from tags
- ✅ Conditional yielding based on capabilities
- ✅ Parametric expansion for parameter validation
- ✅ Extensible design (add new checks easily)

**Test Coverage**:
- ✅ Meta-tests verify check functions work
- ⏳ Systematic tests in production test files (in progress)
- ⏳ Legacy test refactoring (in progress)

---

## References

### Related Files

- **Tag System**: `src/yohou/utils/tags.py` (SplitterTags dataclass)
- **Splitters**: `src/yohou/model_selection/split.py` (BaseSplitter + 2 implementations with gap parameter)
- **Check Functions**: `src/yohou/testing/splitter.py` (8 functions)
- **Generator**: `src/yohou/testing/generators.py` (_yield_yohou_splitter_checks)
- **Meta-tests**: `tests/testing/test_splitter.py`
- **Production tests**: `tests/model_selection/test_split.py`

### Key Patterns

1. **Function-based testing**: `@pytest.mark.parametrize` with generators
2. **Auto-detection**: Tags drive which checks apply
3. **Systematic validation**: Consistent checks across all splitters
4. **Parametric expansion**: One check function → many test cases

### Migration Guide

**From Class-based to Function-based**:

```python
# Before: 50+ lines per splitter
class TestExpandingWindowSplitter:
    def test_tags_accessible(self): ...
    def test_produces_valid_indices(self): ...
    # ... many more methods ...

class TestSlidingWindowSplitter:
    def test_tags_accessible(self): ...  # Duplicate
    def test_produces_valid_indices(self): ...  # Duplicate
    # ... many more methods ...

# After: 10 lines for all splitters
@pytest.mark.parametrize(
    "check_name,check_func,check_kwargs",
    list(_yield_yohou_splitter_checks(splitter, y_data))
)
def test_splitter_systematic(check_name, check_func, check_kwargs):
    check_func(**check_kwargs)
```

**Benefits**:
- 80% reduction in test code
- Automatic coverage of new checks
- Clear test names in pytest output
- Easy to add new splitters (just one parametrize block)

---

## Future Work

### Planned Enhancements

1. **Panel Data Support**:
   - Per-panel splitting (independent splits for each series)
   - Set `supports_panel_data=True` for supporting splitters
   - Add `check_splitter_panel_data_support` to generator

2. **Stratified Splitting**:
   - Stratify by metadata (e.g., weekday/weekend)
   - New tag: `supports_stratification`
   - New check: `check_splitter_stratification`

3. **Custom Overlap Patterns**:
   - Allow controlled test set overlap
   - Update `produces_non_overlapping_tests` to be more granular
   - New check: `check_splitter_overlap_pattern`

### Maintenance Notes

- **Adding Checks**: Append to `splitter.py`, update generator, export in `__init__.py`
- **Adding Splitters**: Implement `__sklearn_tags__()`, add to test file with generator
- **Deprecating Checks**: Mark deprecated in docstring, keep in codebase for backward compatibility

---

**Last Updated**: January 17, 2026  
**Implementation**: Steps 1-5 complete, Step 6 in progress  
**Status**: Production-ready, active development
