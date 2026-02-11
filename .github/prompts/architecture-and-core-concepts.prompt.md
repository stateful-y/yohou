---
description: "Reference guide for Yohou's architecture, class hierarchy, data flow, panel data, and metadata routing. Use when working on base classes, understanding data flow, or modifying core infrastructure."
---

# Yohou Architecture & Core Concepts

## Project Philosophy

Yohou bridges sklearn's tabular ML ecosystem with time series forecasting by treating forecasting as a supervised learning reduction problem while maintaining temporal structure.

**Key Design Principles**:
- **sklearn-Compatible**: All forecasters/transformers extend sklearn base classes
- **Polars-First**: Built on polars DataFrames for performance
- **Time Series Extensions**: Adds `update`, `reset`, `update_predict` methods for incremental learning
- **Panel Data Native**: Supports both global and local (panel) time series via prefixed column names

**Critical Bootstrap Behavior**: Yohou automatically enables sklearn's metadata routing on import (`set_config(enable_metadata_routing=True)` in `src/yohou/__init__.py`). This is a global state change. It also registers custom composite methods: `update_transform`, `update_predict`, `predict_interval`, `update_predict_interval`.

---

## Data Flow

### DataFrame Structure

All data uses **polars DataFrames** with a mandatory `"time"` column (datetime type):
- **`y`**: Target time series (what to forecast)
- **`X`**: Exogenous features (known ex-ante — in advance)

**Time Column Preservation**:
- Transformers: Input/output both have `"time"` column
- Forecasters: Predictions have `"observed_time"` and `"time"` columns
  - `"observed_time"`: Last observation time used for prediction
  - `"time"`: Predicted time steps

---

## Class Hierarchy (`src/yohou/base.py`)

### 1. BaseTransformer (extends `sklearn.base.TransformerMixin`)
- **Must Implement**: `fit(X, y)`, `transform(X)`, `update(X)`, `reset(X)`, `get_feature_names_out()`
- **Key Attributes**: `_X_observed` (stateful buffer), `observation_horizon` property, `feature_names_in_`, `n_features_in_`
- Memory pattern: `update()` appends then calls `reset()` to maintain fixed-size window

### 2. BaseForecaster
- Handles `target_transformer` and `feature_transformer` composition
- Manages observation buffers: `_y_observed`, `_X_observed`, `_X_t_observed`
- Panel data setup via `_set_input_attributes()`
- **Signature difference from sklearn**: `fit(y, X, forecasting_horizon)` — horizon required at fit time

### 3. BaseReductionForecaster
- Converts time series to tabular format via `tabularize()` (creates lag features)
- Fits sklearn estimators on lagged features → recursive multi-step predictions
- `estimator` parameter: any sklearn regressor

### 4. BaseWrapper
- Wraps non-sklearn classes into sklearn estimators for compatibility

### Forecaster Types
- **BasePointForecaster**: `prediction_types = {"point"}`
- **BaseIntervalForecaster**: `prediction_types = {"interval"}` or `{"point", "interval"}`

### Meta-Forecasters
- **Decomposer** (`src/yohou/decomposition/decomposer.py`): Sequential decomposition (trend + season + residual), additive by default
- **ColumnForecaster** (`src/yohou/forecaster/composition.py`): Different forecasters per column with parallel execution

---

## Time Series Methods

- `fit(y, X, forecasting_horizon)`: Train on historical data
- `update(y, X, panel_group_names)`: Add new observations without retrain (incremental learning)
- `predict(forecasting_horizon, X, panel_group_names)`: Generate forecasts (can differ from fit horizon)
- `predict_interval(forecasting_horizon, X, coverage_rates, panel_group_names)`: Generate interval forecasts
- `update_predict(y, X, panel_group_names)`: Combined update + predict (atomic)
- `update_predict_interval(y, X, coverage_rates, panel_group_names)`: Combined update + predict_interval
- `reset(y, X, panel_group_names)`: Reset memory to last `observation_horizon` rows

---

## Panel Data & Locality (`src/yohou/utils/panel.py`)

**Naming Convention**: `{group_name}__{suffix}` for prefixed columns.

```python
y = pl.DataFrame({
    "time": [...],
    "sales__store_1": [100, 110, ...],  # Prefix: sales, Suffix: store_1
    "sales__store_2": [150, 160, ...],
})

from yohou.utils.panel import inspect_locality, get_group_df
global_names, panel_groups = inspect_locality(y)
# Returns: ([], {"sales": ["sales__store_1", "sales__store_2"]})

sales_data = get_group_df(y, "sales", schema)
# Returns: DataFrame with columns ["time", "store_1", "store_2"]
```

**Forecaster Panel Attributes** (after fit):
- `panel_group_names_`: List of group prefixes or `None` for global data
- `local_y_schema_`, `local_X_schema_`: Schemas with unprefixed column names
- Internal storage: `dict[str, pl.DataFrame]` with unprefixed columns

---

## FeaturePipeline (`src/yohou/pipeline.py`)

- `observation_horizon` computed as sum (FeaturePipeline) or max (FeatureUnion) of steps
- All components must implement `update()` and `reset()`
- Also provides: `FeatureUnion`, `ColumnTransformer`

---

## Metadata Routing

Enabled automatically on import. Custom composite methods registered:
- `update_transform` → routes to `["update", "transform"]`
- `update_predict` → routes to `["update", "predict"]`
- `update_predict_interval` → routes to `["update", "predict_interval"]`

**What is NOT routed**: `y`, `X` (primary data), `update()` (memory management only)
**What IS routed via `**params`**: `time_weight`, custom metadata

---

## Metrics & Scoring (`src/yohou/metrics/base.py`)

All metrics extend `BaseScorer` with `prediction_types` property and `score(y_truth, y_pred)` method.

## Hyperparameter Search (`src/yohou/model_selection/search.py`)

`GridSearchCV` and `RandomizedSearchCV`: sklearn-compatible, wraps any `BaseForecaster`. Methods like `predict()`, `predict_interval()` only available after fitting when `refit=True`, controlled via `_search_forecaster_has(attr)` pattern.

## Key Utilities

- `yohou.utils.tabularization.tabularize(df, lags)`: Create lagged features
- `yohou.utils.panel.inspect_locality(df)`: Parse global/local columns
- `yohou.utils.validation.check_interval_consistency(df)`: Validate time spacing
