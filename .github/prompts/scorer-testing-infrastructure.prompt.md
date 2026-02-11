---
description: "Testing patterns for Yohou forecasting metrics/scorers. 11 check functions for point and interval scorers. Use when writing or debugging scorer tests."
---

# Scorer Testing Infrastructure

## Scorer Types

- **Point Scorers**: Evaluate point forecasts (MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError). Lower is better for error metrics.
- **Interval Scorers**: Evaluate prediction intervals (IntervalWidth, IntervalCoverage, PinballLoss). Input has `_lower`/`_upper` columns.

---

## Tag System (`ScorerTags`)

```python
@dataclass
class ScorerTags:
    prediction_type: Literal["point", "interval"] | None = None
    lower_is_better: bool = True
    requires_calibration: bool = False
```

---

## Check Functions — `src/yohou/testing/scorer.py` (11 total)

### Tag System (3)

| Check | Validates |
|-------|-----------|
| `check_scorer_tags_accessible_before_fit` | Tags accessible pre-fit |
| `check_scorer_tags_static_after_fit` | Tags don't change after fit |
| `check_scorer_tags_match_capabilities` | Tags match actual behavior |

### Functionality (7)

| Check | Validates |
|-------|-----------|
| `check_scorer_prediction_type_compatibility` | Compatibility with forecaster outputs |
| `check_scorer_lower_is_better` | Correct lower_is_better property |
| `check_scorer_aggregation_methods` | mean/median/sum aggregation works |
| `check_scorer_panel_subselection` | Filter by panel_group_names |
| `check_scorer_component_subselection` | Filter by component_names |
| `check_scorer_coverage_rate_subselection` | Filter by coverage_rates |
| `check_scorer_methods_call_check_is_fitted` | Methods call check_is_fitted() |

### Parameter Validation (1)

| Check | Validates |
|-------|-----------|
| `check_scorer_parameter_validation` | Input parameter constraints enforced |

---

## Generator Usage

```python
from yohou.testing import _yield_yohou_scorer_checks

for check_name, check_func, check_kwargs in _yield_yohou_scorer_checks(
    scorer, y_truth, y_pred,
    tags={"prediction_type": "point", "lower_is_better": True}
):
    check_func(scorer, **check_kwargs)
```

## Test Files

```
tests/metrics/test_point.py        # Point metrics
tests/metrics/test_interval.py     # Interval metrics
tests/metrics/test_conformity.py   # Conformity scores
```
