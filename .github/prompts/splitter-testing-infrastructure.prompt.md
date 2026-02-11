---
description: "Testing patterns for Yohou time series cross-validation splitters. 8 check functions for expanding/sliding window splitters. Use when writing or debugging splitter tests."
---

# Splitter Testing Infrastructure

## Splitter Types

- **ExpandingWindowSplitter**: Train set grows (cumulative). Use when all history should inform forecasts.
- **SlidingWindowSplitter**: Fixed train window (rolling). Use when recent data is more relevant. Supports `gap` parameter for buffer between train/test.

---

## Tag System (`SplitterTags`)

```python
@dataclass
class SplitterTags:
    splitter_type: Literal["expanding", "sliding", "gap"] | None = None
    supports_panel_data: bool = False
    requires_data_for_n_splits: bool = False
    produces_non_overlapping_tests: bool = True
    stateful: bool = False
```

---

## Check Functions — `src/yohou/testing/splitter.py` (8 total)

### Tag System (3)

| Check | Validates |
|-------|-----------|
| `check_splitter_tags_accessible_before_fit` | Tags accessible pre-fit |
| `check_splitter_tags_static_after_fit` | Tags don't change after fit |
| `check_splitter_tags_match_capabilities` | Tags reflect actual behavior |

### Functionality (4)

| Check | Validates |
|-------|-----------|
| `check_splitter_produces_valid_indices` | Valid train/test indices from split() |
| `check_splitter_n_splits_consistency` | get_n_splits() matches actual split count |
| `check_splitter_non_overlapping_tests` | Test sets don't overlap across folds |
| `check_splitter_panel_data_support` | Handles panel data correctly |

### Parameter Validation (1)

| Check | Validates |
|-------|-----------|
| `check_splitter_parameter_constraints` | _parameter_constraints enforced |

---

## Generator Usage

```python
from yohou.testing import _yield_yohou_splitter_checks

for check_name, check_func, check_kwargs in _yield_yohou_splitter_checks(
    splitter, y, X,
    tags={"splitter_type": "expanding", "supports_panel_data": False}
):
    check_func(**check_kwargs)
```

## Test File

```
tests/model_selection/test_split.py    # All splitter tests
```
