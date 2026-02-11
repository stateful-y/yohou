---
description: "Comprehensive testing guide for Yohou forecasters. 25 check functions across 5 modules (forecaster.py, point.py, interval.py, reduction.py, panel.py). Use when writing or debugging forecaster tests."
---

# Forecaster Testing Infrastructure

## Check Functions (25 total)

### Common Forecaster Checks — `src/yohou/testing/forecaster.py` (13 functions)

| Check | Validates |
|-------|-----------|
| `check_fit_sets_forecaster_attributes` | Required attributes after fit() (fit_forecasting_horizon_, interval_, panel_group_names_, local_y_schema_, local_X_schema_, etc.) |
| `check_forecaster_not_fitted_error` | NotFittedError before fit() |
| `check_predict_time_columns` | Predictions have "observed_time" and "time" columns |
| `check_update_extends_observations` | update() extends observation buffers |
| `check_reset_replaces_observations` | reset() replaces buffers with last horizon rows |
| `check_reset_propagates_to_transformers` | reset() cascades to nested transformers |
| `check_forecasting_horizon_validation` | forecasting_horizon >= 1 enforcement |
| `check_prediction_types_property` | Correct prediction_types set |
| `check_clone_preserves_forecaster_params` | clone() preserves parameters |
| `check_forecaster_tags_accessible_before_fit` | Tags accessible pre-fit |
| `check_forecaster_tags_static_after_fit` | Tags don't change after fit |
| `check_forecaster_tags_match_capabilities` | Tags match actual behavior |
| `check_forecaster_methods_call_check_is_fitted` | Methods call check_is_fitted() |

### Point Checks — `src/yohou/testing/point.py` (2 functions)

| Check | Validates |
|-------|-----------|
| `check_point_prediction_structure` | No interval columns in output |
| `check_point_prediction_types` | prediction_types == {"point"} |

### Interval Checks — `src/yohou/testing/interval.py` (5 functions)

| Check | Validates |
|-------|-----------|
| `check_interval_prediction_columns` | {col}_lower_{rate}, {col}_upper_{rate} naming |
| `check_interval_bounds` | upper >= lower for all rates |
| `check_interval_prediction_types` | "interval" in prediction_types |
| `check_coverage_rates_parameter` | List of floats in (0, 1) |
| `check_coverage_rates_validation` | Invalid rates raise ValueError |

### Reduction Checks — `src/yohou/testing/reduction.py` (2 functions)

| Check | Validates |
|-------|-----------|
| `check_estimator_parameter` | sklearn BaseEstimator accepted |
| `check_reduction_strategy` | Strategy parameter exists |

### Panel Checks — `src/yohou/testing/panel.py` (3 functions)

| Check | Validates |
|-------|-----------|
| `check_panel_data` | panel_group_names=None predicts all groups |
| `check_panel_single_group` | Filters to specified groups only |
| `check_panel_invalid_group_raises` | ValueError for invalid group names |

---

## Generator Usage

```python
from yohou.testing import _yield_yohou_forecaster_checks

for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(
    forecaster_fitted, y_train, X_train, y_test, X_test,
    tags={"forecaster_type": "point", "uses_reduction": True}
):
    check_func(forecaster_fitted, **check_kwargs)
```

**Tags**: `forecaster_type` ("point"/"interval"), `uses_reduction` (bool), `supports_panel_data` (bool), `uses_transformers` (bool)

---

## Test File Layout

```
tests/point_forecaster/test_naive.py         # SeasonalNaive
tests/point_forecaster/test_reduction.py      # PointReductionForecaster + analytical
tests/point_forecaster/test_panel.py          # Cross-learning (point)
tests/interval_forecaster/test_reduction.py   # IntervalReductionForecaster
tests/interval_forecaster/test_panel.py       # Cross-learning (interval)
tests/decomposition/test_trend.py             # Trend forecasters
tests/decomposition/test_seasonality.py       # Seasonality forecasters
tests/decomposition/test_decomposer.py        # Meta-forecaster
tests/test_parameter_validation.py            # forecasting_horizon, coverage_rates
```
