# Yohou Architecture & Core Concepts

**Purpose**: Deep dive into Yohou's architecture, class hierarchy, data flow, and design philosophy.

---

## Project Philosophy

Yohou bridges sklearn's tabular ML ecosystem with time series forecasting by treating forecasting as a supervised learning reduction problem while maintaining temporal structure.

**Key Design Principles**:
- **sklearn-Compatible**: All forecasters/transformers extend sklearn base classes
- **Polars-First**: Built on polars DataFrames for performance
- **Time Series Extensions**: Adds `update`, `reset`, `update_predict` methods for incremental learning
- **Panel Data Native**: Supports both global and local (panel) time series via prefixed column names

**Critical Bootstrap Behavior**: Yohou automatically enables sklearn's metadata routing on import (`set_config(enable_metadata_routing=True)` in `src/yohou/__init__.py`). This is a global state change that enables metadata propagation through pipelines and cross-validation.

---

## Data Flow

### DataFrame Structure

All data uses **polars DataFrames** with a mandatory `"time"` column (datetime type):

```python
import polars as pl
from datetime import datetime

# Basic time series
y = pl.DataFrame({
    "time": pl.datetime_range(start=datetime(2020, 1, 1),
                               end=datetime(2020, 12, 31),
                               interval="1d", eager=True),
    "sales": [100, 105, 110, ...]  # Target variable
})

# With exogenous features
X = pl.DataFrame({
    "time": [...],  # Same time index as y
    "temperature": [20, 22, 19, ...],
    "is_holiday": [0, 0, 1, ...]
})
```

### Feature Types

- **`y`**: Target time series (what to forecast)
- **`X`**: Exogenous features (known in advance, e.g., holidays, planned promotions, weather forecasts)

**Critical Note**: All features in `X` are expected to be known ex-ante (in advance). For ex-post features (observed after the fact), use `ColumnForecaster` to forecast them first.

### Time Column Preservation

**Transformers**: Input/output both have `"time"` column
```python
X_transformed = transformer.transform(X)  # Still has "time" column
```

**Forecasters**: Predictions add dual time columns for alignment
```python
y_pred = forecaster.predict(forecasting_horizon=7)
# Columns: "observed_time", "time", <target_columns>
# - "observed_time": Last observation time used for prediction
# - "time": Predicted time steps (7 future points)
```

---

## Class Hierarchy

### Core Base Classes (`src/yohou/base.py`)

#### 1. BaseTransformer
Extends `sklearn.base.TransformerMixin`

**Must Implement**:
- `fit(X, y)`: Train transformer
- `transform(X)`: Apply transformation
- `update(X)`: Extend observation buffer (incremental learning)
- `reset(X)`: Replace observation buffer
- `get_feature_names_out()`: Return output column names

**Key Attributes**:
- `_X_observed`: Stateful buffer for windowing operations (last `observation_horizon` rows)
- `observation_horizon` property: Raises `NotFittedError` before fit for stateful transformers
- `feature_names_in_`, `n_features_in_`: Auto-set by `fit()`

**Memory Management**:
```python
# fit() initializes observation buffer
transformer.fit(X)
transformer._X_observed  # Last observation_horizon rows of X

# update() extends memory
transformer.update(X_new)  # Appends X_new, keeps last observation_horizon rows

# reset() replaces memory
transformer.reset(X_latest)  # Replaces with last observation_horizon rows of X_latest
```

#### 2. BaseForecaster
Base for all forecasters (point and interval)

**Key Responsibilities**:
- Handles `target_transformer` and `feature_transformer` composition
- Manages observation buffers: `_y_observed`, `_X_observed`, `_X_t_observed`
- Panel data setup via `_set_input_attributes()`
- Recursive prediction infrastructure

**Critical Methods**:
- `_pre_fit(y, X, forecasting_horizon)`: Pre-processes inputs, applies transformers, sets up buffers
- `_add_time_columns(y_pred)`: Adds "observed_time" and "time" columns to predictions
- `input_features` parameter: Controls what feature_transformer sees ("X" | "y_t|X" | "y|X")

**Signature Difference from sklearn**:
```python
# Forecasters require horizon at fit time (not predict time)
forecaster.fit(y, X, forecasting_horizon=7)  # Horizon is required
forecaster.predict(forecasting_horizon=3)     # Can predict different horizon
```

#### 3. BaseReductionForecaster
Forecasting via sklearn regressors (supervised learning reduction)

**Key Responsibilities**:
- Converts time series to tabular format via `tabularize()` (creates lag features)
- Fits sklearn estimators on lagged features
- Generates recursive predictions for multi-step forecasts

**Must Provide**:
- `estimator` parameter: Any sklearn regressor (e.g., `Ridge`, `RandomForest`)
- `reduction_strategy`: "direct" (separate model per step) or "multi-output" (single model)

**How It Works**:
```python
# 1. Tabularize: Create lag features
X_lags = tabularize(y, lags=[1, 2, 3])
# y_t   y_t-1  y_t-2  y_t-3
# 100   95     90     85
# 105   100    95     90
# ...

# 2. Fit estimator
estimator.fit(X_lags, y_target)

# 3. Recursive prediction
y_pred_t1 = estimator.predict(latest_lags)
y_pred_t2 = estimator.predict([y_pred_t1, latest_lags[0], latest_lags[1]])
# Continue recursively...
```

**Panel Data Support**: Handles prefixed columns (e.g., "sales__store_1", "sales__store_2")

#### 4. BaseWrapper
Wraps non-sklearn classes into sklearn estimators

**Purpose**: Make similarity measures and other classes sklearn-compatible
**Provides**: `get_params()`, `set_params()` for sklearn compatibility

---

### Forecaster Type Hierarchy

#### 5. BasePointForecaster (`src/yohou/point_forecaster/base.py`)
Base for point (deterministic) forecasts

**Required Property**:
- `prediction_types` → returns `{"point"}`

**Key Method**:
- `predict()` with `predict_transformed` parameter

**Concrete Implementations**:
- `SeasonalNaive`: Pattern-based seasonal forecasting
- `PointReductionForecaster`: Reduction to sklearn regressors

#### 6. BaseIntervalForecaster (`src/yohou/interval_forecaster/base.py`)
Base for interval (probabilistic) forecasts

**Required Property**:
- `prediction_types` → returns `{"interval"}` or `{"point", "interval"}`

**Key Features**:
- Handles `coverage_rates` parameter (e.g., `[0.8, 0.9, 0.95]`)
- Includes `BaseSimilarity` base class for similarity-weighted conformal prediction
- Prediction columns: `{col}_lower_{rate}`, `{col}_upper_{rate}`

**Concrete Implementations**:
- `SplitConformalForecaster`: Conformal prediction wrapper
- `IntervalReductionForecaster`: Reduction with quantile regression

---

### Meta-Forecasters

#### 7. Decomposer (`src/yohou/decomposition/decomposer.py`)
Sequential decomposition meta-forecaster

**Key Pattern**: Additive decomposition
```python
# Each component models residuals from previous components
residuals = y
for forecaster in [trend_model, seasonal_model]:
    forecaster.fit(residuals, X, forecasting_horizon)
    residuals = residuals - forecaster.predict(...)

# Final prediction = sum of all component predictions
y_pred = trend_pred + seasonal_pred + residual_pred
```

**Features**:
- `store_residuals=True`: Inspect intermediate residuals in `self.residuals_`
- `target_transformer=LogTransform()`: Multiplicative decomposition (additive in log-space)
- Extends `_BaseComposition` for sklearn compatibility

**Example**:
```python
from yohou.decomposition import Decomposer, PolynomialTrendForecaster, PatternSeasonalityForecaster

forecaster = Decomposer([
    ("trend", PolynomialTrendForecaster(degree=2)),
    ("season", PatternSeasonalityForecaster(seasonality=7, method="average")),
])
```

#### 8. ColumnForecaster (`src/yohou/forecaster/composition.py`)
Apply different forecasters to different columns

**Use Cases**:
- Different models for different product categories
- Heterogeneous multi-variate forecasting
- Ensemble approaches with column-specific models

**Key Features**:
- Column selectors: `str`, `list[str]`, `slice`, or `callable`
- Parallel execution via `n_jobs` parameter
- Preserves column order in predictions

**Example**:
```python
from yohou.forecaster.composition import ColumnForecaster
from yohou.point_forecaster import PointReductionForecaster, SeasonalNaive

forecaster = ColumnForecaster([
    ("sales_model", PointReductionForecaster(estimator=Ridge()), ["sales"]),
    ("inventory_model", SeasonalNaive(seasonality=7), ["inventory"])
], n_jobs=-1)  # Parallel execution
```

---

## Time Series-Specific Methods

Standard sklearn lifecycle extended with streaming/incremental learning:

### fit(y, X, forecasting_horizon)
Train on historical data

**Forecasters**: Horizon is required at fit time (unlike sklearn's predict-time horizon)
```python
forecaster.fit(y, X, forecasting_horizon=7)  # Must specify horizon
```

**Transformers**: Follow sklearn convention (`y` optional)
```python
transformer.fit(X, y)  # y is optional
```

### update(y, X, panel_group_names)
Add new observations **without full retrain** (incremental learning)

**What it does**:
- Updates internal memory buffers (`_X_observed`, `_y_observed`, etc.)
- Does NOT refit models - use for streaming/online scenarios
- `panel_group_names`: Optional list of group prefixes to update (for panel data)

**Pattern**: `update()` appends then calls `reset()` to maintain fixed-size window

### predict(forecasting_horizon, X, panel_group_names)
Generate forecasts

**Flexibility**: Can predict different horizon than fit (applies model recursively)
```python
forecaster.fit(y, X, forecasting_horizon=1)   # Fit for 1-step
forecaster.predict(forecasting_horizon=12)     # Predict 12 steps (recursive)
```

**Panel data**: `panel_group_names` controls which groups to predict (list of strings or None for all)

### update_predict(y, X, panel_group_names)
Combined update + predict (atomic operation)

**Common use**: Rolling evaluation, online forecasting
```python
for batch in data_stream:
    y_pred = forecaster.update_predict(y=batch, forecasting_horizon=7)
```

### reset(y, X, panel_group_names)
Reset memory/observation horizon to last `observation_horizon` rows

**Use case**: "Rewind" forecaster state without refitting
```python
forecaster.reset(y_recent, X_recent)  # Only keeps last observation_horizon rows
```

---

## Panel Data & Locality (`src/yohou/utils/panel.py`)

### Critical Concept: Prefixed Column Names

`inspect_locality()` distinguishes **global** vs **local (panel)** time series using prefixed column names.

**Naming Convention**: `{group_name}__{suffix}` pattern
```python
import polars as pl

# Panel data example: Multiple stores
y = pl.DataFrame({
    "time": [...],
    "sales__store_1": [100, 110, 120, ...],  # Prefixed columns
    "sales__store_2": [150, 160, 170, ...],
    "sales__store_3": [200, 210, 220, ...],
})

from yohou.utils.panel import inspect_locality
global_names, panel_groups = inspect_locality(y)
# Returns: ([], {"sales": ["sales__store_1", "sales__store_2", "sales__store_3"]})
```

**Requirements**:
- Group name: Common prefix identifying the time series group (e.g., "sales", "inventory")
- Suffix: Unique identifier for each series in the group (e.g., "store_1", "store_2")
- All columns in a group must have the **same set of suffixes**

### Key Utilities

**`inspect_locality(df)`**: Returns `(global_columns, panel_groups_dict)`

**`get_group_df(df, group_name, schema)`**: Extracts a single group's data with unprefixed columns
```python
from yohou.utils.panel import get_group_df

# Extract "sales" group with unprefixed columns
sales_data = get_group_df(y, "sales", schema)
# Returns DataFrame with columns: ["time", "store_1", "store_2", "store_3"]
```

### Forecaster Panel Data Attributes

After fitting with panel data:
- `panel_group_names_`: List of group prefixes (e.g., `["sales", "inventory"]`) or `None` for global data
- `local_y_schema_`, `local_X_schema_`: Schemas with **unprefixed** column names (after group extraction)
- `local_y_t_schema_`, `local_X_t_schema_`: Schemas for **transformed** data (unprefixed)
- `global_X_schema_`: Schema for global X columns that appear alongside panel groups

### Internal Representation

During fit/predict:
- Panel data stored as `dict[str, pl.DataFrame]` internally
- Each dict entry has unprefixed columns for easier processing
- Predictions reconstructed with prefixed columns for output consistency

### Why Prefixed Columns?

Maintains polars' native column model while enabling clear grouping semantics. Avoids complexity of nested struct columns while preserving vectorized operations.

---

## FeaturePipeline (`src/yohou/pipeline.py`)

Custom sklearn Pipeline supporting time series operations:

```python
from yohou.pipeline import FeaturePipeline

pipeline = FeaturePipeline([
    ('lag', LagTransformer(lags=[1, 2, 3])),
    ('forecaster', PointReductionForecaster())
])
```

**Key Differences from sklearn**:
- `observation_horizon` computed as sum (FeaturePipeline) or max (FeatureUnion) of steps
- All components must implement `update()` and `reset()`
- Handles struct columns (panel data) in concat operations

**Also Provides**:
- `FeatureUnion`: Parallel feature extraction (max observation_horizon)
- `ColumnTransformer`: Apply different transformers to different column subsets

---

## Metadata Routing (`src/yohou/__init__.py`)

### Automatic Setup

**Critical infrastructure**: Yohou uses sklearn's metadata routing system to propagate auxiliary parameters through pipelines and nested estimators.

```python
# This happens automatically when you import yohou
from sklearn import set_config
set_config(enable_metadata_routing=True)  # Global enable

# Registers custom composite methods
from sklearn.utils.metadata_routing import SIMPLE_METHODS, COMPOSITE_METHODS
SIMPLE_METHODS.extend(["update_transform", "update_predict"])
COMPOSITE_METHODS["update_transform"] = ["update", "transform"]  # Routes only to transform
COMPOSITE_METHODS["update_predict"] = ["update", "predict"]      # Routes only to predict
```

### What is Routed

- ❌ **NOT routed**: `y`, `X` (primary time series data - always explicit parameters)
- ✅ **Routed via `**params`**: `time_weight`, custom metadata
- ⚠️ **`forecasting_horizon`**: Explicit parameter (not in `**params`) but CAN be routed if needed
- ⚠️ **`update()`**: Does NOT accept `**params` (memory management only, no metadata)

### Implementation Pattern

**All forecasters/transformers**:
- Have `**params` in method signatures (`fit`, `transform`, `predict`)
- Use `@_fit_context` decorator on `fit()` methods for automatic routing

**Routers** (FeaturePipeline, forecasters with transformers):
- Implement `get_metadata_routing()` → returns `MetadataRouter`

**Consumers** (simple transformers, scorers):
- Just accept `**params` for future extensibility

**Current Status**: Infrastructure 100% complete. Actual metadata consumption (e.g., `time_weight` → `sample_weight` conversion) not yet implemented. See `.github/copilot/sklearn-metadata-routing-implementation.md` for full details.

---

## Metrics & Scoring (`src/yohou/metrics/base.py`)

All metrics extend `BaseScorer`:

**Required Methods**:
- `prediction_types` property: `{"point"}` or `{"interval"}`
- `score(y_truth, y_pred)`: Returns pl.DataFrame with scores
- `_validate_inputs()`: Aligns y_truth (with "time") and y_pred (with "observed_time"/"time")

**Optional Methods**:
- `fit(y_train)`: For scale-dependent metrics (most are stateless)

**Time Alignment Pattern**: Forecasters produce predictions with dual time columns for tracking when observation was made vs. predicted time step.

**Usage in SearchCV**:
```python
from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import SearchCV

search = SearchCV(
    forecaster=PointReductionForecaster(),
    scoring=MeanAbsoluteError(),  # Single metric
    # Or multi-metric: scoring={"mae": MeanAbsoluteError(), "rmse": RMSE()}
    ...
)
```

---

## Hyperparameter Search (`src/yohou/model_selection/search.py`)

### SearchCV: Optuna-Based Cross-Validation

**Key Features**:
- Wraps any `BaseForecaster` with Optuna's trial-based optimization
- Uses time series CV splits (via `cv` parameter)
- Supports multi-metric evaluation and custom scoring functions
- **Dynamic method availability**: Methods like `predict()`, `predict_interval()` only available after fitting when `refit=True`

**Key Components**:
- `Sampler`: Wrapper for Optuna samplers (default: `TPESampler`)
- `Storage`: Optional persistent storage for optimization history (e.g., `RDBStorage`)
- `n_warmup_trials`: Random search trials before sampler kicks in
- `n_trials`: Number of Optuna optimization trials

### Method Availability Pattern

```python
from sklearn.utils.metaestimators import available_if

def _search_forecaster_has(attr):
    """Helper following sklearn's _estimator_has pattern."""
    def check(self):
        if not self.refit:
            return False
        if not hasattr(self, "best_forecaster_"):
            return False
        if attr in {"point", "interval"}:
            return attr in self.best_forecaster_.prediction_types
        return hasattr(self.best_forecaster_, attr)
    return check

# Methods only available if best_forecaster_ supports them
@available_if(_search_forecaster_has("point"))
def predict(self, forecasting_horizon=None, X=None, **params):
    return self.best_forecaster_.predict(forecasting_horizon, X, **params)

@available_if(_search_forecaster_has("interval"))
def predict_interval(self, forecasting_horizon=None, X=None, **params):
    return self.best_forecaster_.predict_interval(forecasting_horizon, X, **params)
```

**Pattern**: Enables proper support for point, interval, and hybrid forecasters (both point+interval)

### Usage

```python
from yohou.model_selection import SearchCV
from yohou.metrics import MeanAbsoluteError
import optuna

search = SearchCV(
    forecaster=PointReductionForecaster(),
    param_distributions={
        "estimator__alpha": optuna.distributions.FloatDistribution(0.01, 1.0),
        "feature_transformer__lag": optuna.distributions.IntDistribution(1, 10)
    },
    scoring=MeanAbsoluteError(),
    n_warmup_trials=5,
    n_trials=20,
    refit=True  # Refits on full data with best params
)

search.fit(y, X, forecasting_horizon=3)
y_pred = search.predict(forecasting_horizon=3, X=X_future)

# Panel data support
y_pred = search.predict(forecasting_horizon=3, X=X_future,
                        panel_group_names=["sales"])  # List of group prefixes
```

**Critical Notes**:
- Param names follow sklearn convention: `step__param` for pipelines
- Always returns `best_forecaster_`, `best_params_`, `cv_results_`
- `panel_group_names` parameter is a **list of strings** or `None` for all groups
- All methods accept `panel_group_names`: `predict`, `predict_interval`, `update`, `update_predict`, `reset`

---

## Key Utilities

**Data Manipulation**:
- `yohou.utils.tabularization.tabularize(df, lags)`: Create lagged features
- `yohou.utils.polars.concat_struct(items, how)`: Merge panel data struct columns

**Panel Data**:
- `yohou.utils.panel.inspect_locality(df)`: Parse global/local columns
- `yohou.utils.panel.get_group_df(df, group_name, schema)`: Extract group with unprefixed columns

**Validation**:
- `yohou.utils.validation.check_interval_consistency(df)`: Validate time spacing
- `yohou.utils.validation.check_inputs(y, X)`: Validate all inputs have matching intervals
