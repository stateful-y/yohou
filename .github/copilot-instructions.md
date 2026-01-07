# GitHub Copilot Instructions for Yohou

## Project Overview
Yohou is a scikit-learn-compatible time series forecasting framework built on **polars** for data manipulation. It extends sklearn's API with time series-specific operations (`update`, `reset`) and supports both point and interval (native or conformal) forecasting with panel data capabilities.

**Philosophy**: Bridge sklearn's tabular ML ecosystem with time series forecasting by treating forecasting as a supervised learning reduction problem while maintaining temporal structure.

**Critical Bootstrap Behavior**: Yohou automatically enables sklearn's metadata routing on import (`set_config(enable_metadata_routing=True)` in `src/yohou/__init__.py`). This is a global state change that enables metadata propagation through pipelines and cross-validation. It also registers custom composite methods (`update_transform`, `update_predict`) with sklearn's routing system.

## Architecture & Core Concepts

### Data Flow
All data uses **polars DataFrames** with a mandatory `"time"` column (datetime type). Two feature types:
- `y`: Target time series (what to forecast)
- `X`: Exogenous features (known in advance, e.g., holidays, planned promotions, weather forecasts)

**Note**: All features in `X` are expected to be known ex-ante (in advance). For ex-post features (observed after the fact), use `ColumnForecaster` to forecast them first.

**Critical**: Time columns are preserved differently across components:
- Transformers: Input/output have "time" column
- Forecasters: Predictions add "observed_time" and "predicted_time" columns for alignment

### Class Hierarchy

**Core Base Classes** (`src/yohou/base.py`):

1. **BaseTransformer** (extends `sklearn.base.TransformerMixin`)
   - Must implement: `fit`, `transform`, `update`, `reset`, `get_feature_names_out`
   - Maintains stateful `_X_observed` for windowing operations (last `observation_horizon` rows)
   - `observation_horizon` property: Raises `NotFittedError` before fit for stateful transformers
   - `fit()` auto-sets `feature_names_in_`, `n_features_in_`, `_X_observed`
   - `update()` extends memory, `reset()` replaces it (both take last `observation_horizon` rows)

2. **BaseForecaster** (base for all forecasters)
   - Handles `target_transformer` and `feature_transformer` composition
   - `_set_input_attributes()` enables panel data (local vs. global time series)
   - Stores `_y_observed`, `_X_observed`, `_X_t_observed` for recursive prediction
   - Signature: `fit(y, X, forecasting_horizon)` - note horizon at fit time
   - Provides `input_features` parameter: "X" | "y_t|X" | "y|X" to control feature_transformer input

3. **BaseReductionForecaster** (forecasting via sklearn regressors, in `base.py`)
   - Converts time series to supervised learning via `tabularize()` (creates lag features)
   - `_estimator_fit_one()`: Single-horizon supervised learning fit
   - `_estimator_predict_one()`: Generates one-step predictions (recursive for multi-step)
   - Supports panel data with prefixed columns (e.g., "sales__store_1", "sales__store_2")
   - Must provide `estimator` param (any sklearn regressor)
   - Supports `reduction_strategy`: "direct" (separate model per step) or "multi-output" (single model)

4. **BaseWrapper** (wraps classes into sklearn estimators, in `base.py`)
   - Base class for wrapping any class into a sklearn-compatible estimator
   - Provides `get_params()` and `set_params()` for sklearn compatibility
   - Used for wrapping similarity measures and other non-sklearn components

**Forecaster Type Hierarchy**:

5. **BasePointForecaster** (`src/yohou/point_forecaster/base.py`)
   - Base for all point forecasters
   - `prediction_types` property returns `{"point"}`
   - Provides `predict()` with `predict_transformed` parameter
   - Concrete implementations: `SeasonalNaive`, `PointReductionForecaster`

6. **BaseIntervalForecaster** (`src/yohou/interval_forecaster/base.py`)
   - Base for all interval forecasters
   - `prediction_types` property returns `{"interval"}` or `{"point", "interval"}`
   - Handles `coverage_rates` parameter for prediction intervals
   - Includes `BaseSimilarity` base class for similarity-weighted conformal prediction
   - Concrete implementations: `SplitConformalForecaster`, `IntervalReductionForecaster`

**Meta-Forecasters** (not reorganized):

7. **Decomposer** (`src/yohou/decomposition/decomposer.py`)
   - Meta-forecaster for sequential decomposition
   - Decomposes time series into additive components (trend + seasonality + residual)
   - Fits forecasters sequentially: each models residuals from previous components
   - Final prediction = sum of all component predictions
   - **Key pattern**: `residuals = y - forecaster1.predict()`, then fit forecaster2 on residuals
   - Use `target_transformer=LogTransform()` for multiplicative decomposition (additive in log-space)
   - Extends `_BaseComposition` for sklearn compatibility with nested estimators
   - Example: `Decomposer([("trend", PolynomialTrend()), ("season", SeasonalNaive())])`
   - `store_residuals=True` enables inspection of intermediate residuals in `self.residuals_`

8. **ColumnForecaster** (`src/yohou/forecaster/composition.py`)
   - Applies different forecasters to different columns of the target time series
   - Takes list of `(name, forecaster, columns)` tuples
   - Supports parallel execution via `n_jobs` parameter
   - Useful for heterogeneous multi-variate forecasting
   - Example: Different models for different product categories

### Time Series-Specific Methods
**Standard sklearn lifecycle extended:**
- `fit(y, X, forecasting_horizon)`: Train on historical data
  - Forecasters: Horizon is required at fit time (unlike sklearn's predict-time horizon)
  - Transformers: `fit(X, y)` follows sklearn convention (`y` optional)
- `update(y, X, panel_group_names)`: Add new observations **without full retrain** (incremental learning)
  - Updates internal memory buffers (`_X_observed`, `_y_observed`, etc.)
  - Does NOT refit models - use for streaming/online scenarios
  - `panel_group_names`: Optional list of group prefixes to update (for panel data)
- `predict(forecasting_horizon, X, panel_group_names)`: Generate forecasts
  - Can predict different horizon than fit (applies model recursively)
  - `panel_group_names`: Optional list of group prefixes to predict (for panel data)
- `update_predict(y, X, panel_group_names)`: Combined update + predict (atomic operation, common in rolling evaluation)
- `reset(y, X, panel_group_names)`: Reset memory/observation horizon to last `observation_horizon` rows
  - Used to "rewind" forecaster state without refitting
  - `panel_group_names`: Optional list of group prefixes to reset (for panel data)

**Memory management pattern**: `update()` appends then calls `reset()` to maintain fixed-size window.

### FeaturePipeline (`src/yohou/pipeline.py`)
Custom sklearn FeaturePipeline/FeatureUnion/ColumnTransformer supporting time series operations:
```python
from yohou.pipeline import FeaturePipeline
FeaturePipeline([
    ('lag', LagTransformer(lag=[1, 2, 3])),
    ('forecaster', PointReductionForecaster())
])
```
**Key differences from sklearn**:
- `observation_horizon` computed as sum (FeaturePipeline) or max (FeatureUnion) of steps
- All components must implement `update()` and `reset()`
- Handles struct columns (panel data) in concat operations

### Panel Data & Locality (`src/yohou/utils/panel.py`)
**Critical concept**: `inspect_locality()` distinguishes global vs. local (panel) time series using **prefixed column names**.

- **Global**: Regular columns apply to all time series (e.g., single univariate series, shared features)
- **Local/Panel**: Columns with group prefixes separated by `__` (double underscore)
  ```python
  # Panel data example: Prefixed columns for multiple stores
  y = pl.DataFrame({
      "time": [...],
      "sales__store_1": [100, 110, 120, ...],
      "sales__store_2": [150, 160, 170, ...],
      "sales__store_3": [200, 210, 220, ...],
  })
  global_names, panel_groups = inspect_locality(y)
  # Returns: ([], {"sales": ["sales__store_1", "sales__store_2", "sales__store_3"]})
  ```

**Naming Convention**: `{group_name}__{suffix}` pattern for panel columns
- Group name: Common prefix identifying the time series group (e.g., "sales", "inventory")
- Suffix: Unique identifier for each series in the group (e.g., "store_1", "store_2")
- All columns in a group must have the **same set of suffixes**

**Key Utilities**:
- `inspect_locality(df)`: Returns `(global_columns, panel_groups_dict)`
- `get_group_df(df, group_name, schema)`: Extracts a single group's data with unprefixed columns
- Example: `get_group_df(y, "sales", schema)` returns DataFrame with columns `["time", "store_1", "store_2", "store_3"]`

**Forecaster Panel Data Attributes**:
- `panel_group_names_`: List of group prefixes (e.g., `["sales", "inventory"]`) or `None` for global data
- `local_y_schema_`, `local_X_schema_`: Schemas with **unprefixed** column names (after group extraction)
- `local_y_t_schema_`, `local_X_t_schema_`: Schemas for **transformed** data (unprefixed)
- `global_X_schema_`: Schema for global X columns that appear alongside panel groups

**Internal Representation**:
- During fit/predict, panel data is stored as `dict[str, pl.DataFrame]` internally
- Each dict entry has unprefixed columns for easier processing
- Predictions are reconstructed with prefixed columns for output consistency

**Why prefixed columns?** Maintains polars' native column model while enabling clear grouping semantics. Avoids complexity of nested struct columns while preserving vectorized operations.

### Metrics & Scoring (`src/yohou/metrics/base.py`)
All metrics extend `BaseScorer`:
- `prediction_types` property: `{"point"}` or `{"interval"}`
- `_validate_inputs()`: Aligns y_truth (with "time") and y_pred (with "observed_time"/"predicted_time")
- `fit(y_train)`: Optional for scale-dependent metrics (most are stateless)
- `score(y_truth, y_pred)`: Returns pl.DataFrame with scores
- Used in `SearchCV` for hyperparameter optimization

**Time alignment pattern**: Forecasters produce predictions with dual time columns for tracking when observation was made vs. predicted time step.

### Hyperparameter Search (`src/yohou/model_selection/search.py`)
**SearchCV**: Optuna-based cross-validation for time series hyperparameter tuning.
- Wraps any `BaseForecaster` with Optuna's trial-based optimization
- Uses time series CV splits (via `cv` parameter)
- Supports multi-metric evaluation and custom scoring functions
- **Dynamic method availability**: Methods like `predict()`, `predict_interval()`, `update_predict()`, etc. are only available after fitting when `refit=True`

**Key components**:
- `Sampler`: Wrapper for Optuna samplers (default: `TPESampler`)
- `Storage`: Optional persistent storage for optimization history (e.g., `RDBStorage`)
- `n_warmup_trials`: Random search trials before sampler kicks in
- `n_trials`: Number of Optuna optimization trials
- `_search_forecaster_has(attr)`: Helper function following sklearn's `_estimator_has` pattern for conditional method availability

**Method Availability Pattern**:
```python
from sklearn.utils.metaestimators import available_if

# Methods only available if best_forecaster_ supports them
@available_if(_search_forecaster_has("point"))  # Checks prediction_types
def predict(self, forecasting_horizon=None, X=None, **params):
    return self.best_forecaster_.predict(forecasting_horizon, X, **params)

@available_if(_search_forecaster_has("interval"))  # Checks prediction_types
def predict_interval(self, forecasting_horizon=None, X=None, **params):
    return self.best_forecaster_.predict_interval(forecasting_horizon, X, **params)
```

**Critical implementation details**:
- `_search_forecaster_has(attr)` checks both `refit=True` and `best_forecaster_` existence
- For "point" or "interval" attributes, checks `best_forecaster_.prediction_types`
- For other attributes, checks `hasattr(best_forecaster_, attr)`
- Pattern matches sklearn's `GridSearchCV`/`RandomizedSearchCV` approach
- Enables proper support for point, interval, and hybrid forecasters (both point+interval)

**Usage pattern**:
```python
from yohou.model_selection import SearchCV
from yohou.metrics import MAE
import optuna

search = SearchCV(
    forecaster=PointReductionForecaster(),
    param_distributions={
        "estimator__alpha": optuna.distributions.FloatDistribution(0.01, 1.0),
        "feature_transformer__lag": optuna.distributions.IntDistribution(1, 10)
    },
    scoring=MAE(),
    n_warmup_trials=5,
    n_trials=20,
    refit=True  # Refits on full data with best params
)
search.fit(y, X, forecasting_horizon=3)
y_pred = search.predict(forecasting_horizon=3, X=X_future)

# Panel data support with panel_group_names (list of group prefixes)
y_panel = pl.DataFrame({
    "time": [...],
    "sales__store_1": [...],
    "sales__store_2": [...],
})
search.fit(y_panel, X, forecasting_horizon=3)
# Predict specific groups only
y_pred = search.predict(forecasting_horizon=3, X=X_future, 
                        panel_group_names=["sales"])  # List of groups
# Or predict all groups (default)
y_pred_all = search.predict(forecasting_horizon=3, X=X_future)  # panel_group_names=None
```

**Critical notes**:
- Param names follow sklearn convention: `step__param` for pipelines
- Always returns `best_forecaster_`, `best_params_`, `cv_results_`
- Integrates with sklearn's metadata routing for CV splits
- `panel_group_names` parameter is a **list of strings** (e.g., `["sales", "inventory"]`) or `None` for all groups
- All methods (`predict`, `predict_interval`, `update`, `update_predict`, `update_predict_interval`, `reset`) accept `panel_group_names`
- Methods delegate directly to `best_forecaster_` after optimization completes

### Metadata Routing (`src/yohou/__init__.py`, detailed in `.github/copilot_plans/`)
**Critical infrastructure**: Yohou uses sklearn's metadata routing system to propagate auxiliary parameters through pipelines and nested estimators.

**Automatic setup on import**:
```python
# This happens automatically when you import yohou
from sklearn import set_config
set_config(enable_metadata_routing=True)  # Global enable

# Registers custom composite methods
SIMPLE_METHODS.extend(["update_transform", "update_predict"])
COMPOSITE_METHODS["update_transform"] = ["update", "transform"]  # Metadata routes only to transform
COMPOSITE_METHODS["update_predict"] = ["update", "predict"]      # Metadata routes only to predict
```

**What is routed vs NOT routed**:
- ❌ NOT routed: `y`, `X` (primary time series data - always explicit parameters)
- ✅ Routed via `**params`: `time_weight`, custom metadata
- ⚠️ `forecasting_horizon` is an explicit parameter (not in `**params`) but CAN be routed if needed
- ⚠️ `update()` does NOT accept `**params` (memory management only, no metadata)

**Implementation pattern**:
- All forecasters/transformers have `**params` in method signatures (`fit`, `transform`, `predict`)
- Routers (FeaturePipeline, forecasters with transformers) implement `get_metadata_routing()` → returns `MetadataRouter`
- Consumers (simple transformers, scorers) just accept `**params` for future extensibility
- Use `@_fit_context` decorator on `fit()` methods for automatic routing

**Current status**: Infrastructure 100% complete. Actual metadata consumption (e.g., `time_weight` → `sample_weight` conversion) not yet implemented. See `.github/copilot_plans/sklearn-metadata-routing-implementation.md` for full details.

## Composition Classes (`src/yohou/forecaster/composition.py`)

### ColumnForecaster
Applies different forecasters to different columns of the target time series.

**Use Cases**:
- Different modeling strategies for different products/categories
- Heterogeneous multi-variate forecasting
- Ensemble approaches with column-specific models

**Usage Pattern**:
```python
from yohou.forecaster.composition import ColumnForecaster
from yohou.point_forecaster import PointReductionForecaster, SeasonalNaive
from sklearn.linear_model import Ridge

forecaster = ColumnForecaster([
    ("sales_model", PointReductionForecaster(estimator=Ridge()), ["sales"]),
    ("inventory_model", SeasonalNaive(seasonality=7), ["inventory"])
], n_jobs=-1)  # Parallel execution

forecaster.fit(y, X, forecasting_horizon=3)
y_pred = forecaster.predict(forecasting_horizon=3, X=X_future)
```

**Key Features**:
- Column selectors: `str`, `list[str]`, `slice`, or `callable`
- Parallel execution with `n_jobs` parameter
- Preserves column order in predictions
- Extends `_BaseComposition` for sklearn compatibility

## Developer Workflow

### Environment & Dependencies
- **Package manager**: `uv` (fast Python package installer/resolver)
- **Dependency groups** in `pyproject.toml`: `dev`, `docs`, `tests`, `fix`, `examples`
- Install dev environment: `uv sync` (syncs all groups)
- **Examples framework**: `marimo` (reactive Python notebooks in `examples/`)
  - NOT traditional Jupyter notebooks - marimo is a reactive notebook system
  - Run examples: `marimo edit examples/air_passengers_tutorial.py`

### Nox Sessions (`noxfile.py` at project root)
**Critical**: Always use `uvx nox` (not plain `nox`) to leverage uv's automatic tool management. Nox is configured to use `uv` as the default venv backend.

**Available sessions** (default: `fix`, `test`, `docs`):
- `test`: Pytest with coverage, runs doctests and unit tests
  - Outputs: `coverage.{python}.xml`, `junit.{python}.xml`, HTML coverage in temp dir
  - Uses `coverage run --parallel-mode` for parallel execution
- `fix`: Pre-commit hooks (ruff linter/formatter + ty type checking + interrogate)
  - Runs all pre-commit hooks with `--all-files`
- `docs`: Build MkDocs documentation (output in `site/`)
- `serve_docs`: Local docs server on `localhost:8080` with live reload
- `deploy_docs`: Deploy to GitHub Pages via `mkdocs gh-deploy`

### Code Quality Tools
- **Linter/Formatter**: Ruff (100 char line length, target py3.12)
  - Auto-fix: `uvx ruff check --fix .`
  - Format: `uvx ruff format .`
- **Type Checker**: `ty` (Rust-based, replaces mypy - NOT mypy compliant)
  - Run: `uvx ty check src` (do NOT use mypy commands)
  - Pre-commit enforces ty checking
  - Note: `ty` uses different inference rules than mypy/pyright
- **Docstring Coverage**: `interrogate` requires 100% coverage (see `pyproject.toml`)
  - Excludes: tests, examples, `_version.py`, private/magic methods
- **Pre-commit hooks**: Defined in `.pre-commit-config.yaml`
  - Run manually: `uvx nox -s fix` or `pre-commit run --all-files`
  - Auto-runs on git commit (includes ty, ruff, interrogate)
  - Hooks: check-yaml, check-merge-conflict, end-of-file-fixer, trailing-whitespace, interrogate, ruff, ruff-format, ty

## Creating New Forecasters

### Step-by-Step Guide to Roll a New Forecaster

This guide walks through creating a new forecaster in Yohou, using real examples from the codebase.

#### 1. Choose Forecaster Type & Location

**Decision tree**:
- **Pattern-based/Statistical**: `src/yohou/point_forecaster/` (e.g., naive, seasonality, trend models)
- **ML-based reduction**: Extend `PointReductionForecaster` or `BaseReductionForecaster`
- **Interval forecasting**: `src/yohou/interval_forecaster/` (extends `BaseIntervalForecaster`)

**File naming**: Use descriptive names like `polynomial_trend.py`, `seasonality.py`, `fourier_seasonality.py`

#### 2. Implement Core Structure

**Minimum requirements** for a point forecaster:

```python
"""Module docstring describing the forecaster."""

import numbers  # For _parameter_constraints

import polars as pl
from pydantic import StrictInt  # For strict type validation
from sklearn.base import _fit_context  # REQUIRED: For automatic parameter validation
from sklearn.utils._param_validation import Interval  # For range constraints

from yohou.base import BaseTransformer  # For _parameter_constraints
from .base import BasePointForecaster  # Or BaseIntervalForecaster


class MyForecaster(BasePointForecaster):
    """Class docstring with NumPy-style documentation.

    Parameters
    ----------
    param1 : type
        Description of parameter.
    target_transformer : BaseTransformer, optional
        Transformer for target variable (standard across forecasters).

    Attributes
    ----------
    fitted_attr_ : type
        Fitted attributes end with underscore (sklearn convention).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.point_forecaster import MyForecaster
    >>>
    >>> # Create example data
    >>> time = pl.datetime_range(
    ...     start=datetime(2020, 1, 1),
    ...     end=datetime(2020, 2, 1),
    ...     interval="1d",
    ...     eager=True
    ... )
    >>> y = pl.DataFrame({"time": time, "value": range(len(time))})
    >>>
    >>> # Fit and predict
    >>> forecaster = MyForecaster(param1=10)
    >>> forecaster.fit(y, forecasting_horizon=5)
    MyForecaster(param1=10)
    >>> y_pred = forecaster.predict(forecasting_horizon=5)

    Notes
    -----
    - Implementation notes, limitations, assumptions
    - References to papers/algorithms if applicable

    """

    # sklearn parameter validation - REQUIRED for all forecasters
    _parameter_constraints: dict = {
        **BasePointForecaster._parameter_constraints,  # Inherit parent constraints
        "param1": [Interval(numbers.Integral, 1, None, closed="left")],  # Integer ≥ 1
        # Use Interval for range validation (min, max, closed="left|right|both|neither")
    }

    def __init__(
        self,
        param1: int,
        target_transformer: BaseTransformer | None = None,
    ):
        """Initialize forecaster.

        Parameters
        ----------
        param1 : int
            Description.
        target_transformer : BaseTransformer, optional
            Transformer for target variable.

        """
        super().__init__(target_transformer=target_transformer)
        self.param1 = param1
        # DO NOT validate parameters here - validation happens at fit time via @_fit_context
        # Only store parameters in __init__

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,  # For metadata routing (always include)
    ) -> "MyForecaster":
        """Fit forecaster to historical data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with "time" column (datetime type).
        X : pl.DataFrame, optional
            Exogenous features with "time" column (not used by all forecasters).
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.
        **params : dict
            Additional metadata (routed via sklearn's metadata routing).

        Returns
        -------
        self
            Fitted forecaster.

        """
        # ALWAYS call _pre_fit first - handles transformers, validation, panel data
        y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)

        # Domain-specific validation (after automatic validation from @_fit_context)
        # Example: Check relationships between parameters that aren't expressible in _parameter_constraints
        # if self.param1 > len(y_t):
        #     raise ValueError(f"param1 ({self.param1}) cannot exceed data length ({len(y_t)})")

        # Your fitting logic here
        # - y_t, X_t are already transformed via target_transformer/feature_transformer
        # - self._y_observed, self._X_observed are set by _pre_fit
        # - self.panel_group_names_, self.local_y_columns_ set if panel data

        # Store fitted parameters with trailing underscore
        self.fitted_attr_ = self._compute_something(y_t)

        return self

    def predict(
        self,
        forecasting_horizon: StrictInt | None = None,
        X: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate forecasts.

        Parameters
        ----------
        forecasting_horizon : int, optional
            Number of steps to forecast. If None, uses horizon from fit().
        X : pl.DataFrame, optional
            Future exogenous features (must have forecasting_horizon rows).
        **params : dict
            Additional metadata.

        Returns
        -------
        pl.DataFrame
            Predictions with columns: "observed_time", "time", <target_columns>
            - "observed_time": Last observation time used for prediction
            - "time": Predicted time steps

        """
        # Handle horizon (use fit horizon if not specified)
        if forecasting_horizon is None:
            forecasting_horizon = self._forecasting_horizon

        # Your prediction logic here
        # Use self._y_observed, self._X_observed for context
        y_pred = self._generate_predictions(forecasting_horizon)

        # CRITICAL: Always add time columns before returning
        return self._add_time_columns(y_pred)

    def _add_time_columns(self, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Add observed_time and time columns to predictions.

        This is inherited from BaseForecaster - handles:
        - observed_time: Last time in self._y_observed
        - time: Future time steps based on forecasting_horizon

        """
        return super()._add_time_columns(y_pred)
```

#### 3. Add Parameter Constraints

**Critical**: All forecasters MUST implement `_parameter_constraints` for sklearn validation:

```python
import numbers
from sklearn.utils._param_validation import Interval
from yohou.base import BaseTransformer

_parameter_constraints: dict = {
    **ParentForecaster._parameter_constraints,  # Inherit parent
    "int_param": [Interval(numbers.Integral, 1, None, closed="left")],  # Integer ≥ 1
    "float_param": [Interval(numbers.Real, 0.0, 1.0, closed="both")],   # Float in [0, 1]
    "positive_float": [Interval(numbers.Real, 0.0, None, closed="neither")],  # Float > 0
    "optional_param": [Interval(numbers.Real, 0.0, 1.0, closed="both"), None],  # Optional
    "transformer_param": [BaseTransformer, None],  # Optional transformer
    "string_param": [str],                         # String parameters
}
```

**Common constraint patterns**:
- `Interval(numbers.Integral, min, max, closed)`: Range-constrained integer
  - `closed="left"`: `[min, max)` - min ≤ value < max
  - `closed="right"`: `(min, max]` - min < value ≤ max
  - `closed="both"`: `[min, max]` - min ≤ value ≤ max
  - `closed="neither"`: `(min, max)` - min < value < max
  - Use `None` for min/max to leave unbounded (e.g., `1, None` means ≥ 1)
- `Interval(numbers.Real, min, max, closed)`: Range-constrained float (includes integers)
- `[Type, None]`: Optional parameters (e.g., `[numbers.Real, None]`)
- `[str]`: String parameters (sklearn validates type only)
- Inherit parent constraints: `**BasePointForecaster._parameter_constraints`

**Validation timing**:
- **Automatic validation** at fit time via `@_fit_context` decorator (type + range checks)
- **Domain-specific validation** in `fit()` body after automatic validation (e.g., data-dependent checks)
- **NO validation in `__init__`** - only store parameters there

**Real-world examples**:
```python
# Example 1: FourierSeasonalityForecaster
_parameter_constraints: dict = {
    **_BaseSeasonalityForecaster._parameter_constraints,
    "n_harmonics": [Interval(numbers.Integral, 1, None, closed="left")],  # ≥ 1
    "alpha": [Interval(numbers.Real, 0.0, None, closed="neither")],        # > 0
    "l1_ratio": [Interval(numbers.Real, 0.0, 1.0, closed="both")],         # [0, 1]
}

# Domain validation in fit() - Nyquist limit check
if self.n_harmonics > self._seasonality // 2:
    raise ValueError(...)

# Example 2: PolynomialTrendForecaster
_parameter_constraints: dict = {
    **BasePointForecaster._parameter_constraints,
    "degree": [Interval(numbers.Integral, 0, None, closed="left")],  # ≥ 0
}

# Example 3: SeasonalityForecaster with string enum
_parameter_constraints: dict = {
    **_BaseSeasonalityForecaster._parameter_constraints,
    "method": [str],  # Type validation only
}

# Custom validation in fit() for allowed values
if self.method not in ["naive", "average", "median"]:
    raise ValueError(f"Invalid method: {self.method}")
```

**Required imports**:
```python
import numbers
from sklearn.base import _fit_context
from sklearn.utils._param_validation import Interval
```

#### 4. Handle Panel Data (if applicable)

If your forecaster should support panel data (multiple time series in struct columns):

```python
def fit(self, y, X, forecasting_horizon, **params):
    y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)

    # Check if panel data
    if self.panel_group_names_ is not None:
        # Panel data: self.local_y_columns_ has field names
        # Process each series separately or vectorize
        for col_name in self.local_y_columns_:
            # Extract series, fit separately
            pass
    else:
        # Global data: regular columns
        pass
```

**Testing panel data**: Use `panel_time_series_factory` fixture (see "Testing Patterns").

#### 5. Write Comprehensive Tests

**Test file structure**: `tests/point_forecaster/test_<forecaster_name>.py`

**Minimum test coverage**:

```python
"""Tests for MyForecaster."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from yohou.point_forecaster import MyForecaster


def test_my_forecaster_basic_fit_predict():
    """Test basic fit and predict workflow."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=49),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": range(50)})

    forecaster = MyForecaster(param1=10)
    forecaster.fit(y[:30], forecasting_horizon=5)

    y_pred = forecaster.predict(forecasting_horizon=5)

    # Validate output structure
    assert len(y_pred) == 5
    assert "observed_time" in y_pred.columns
    assert "time" in y_pred.columns
    assert "value" in y_pred.columns


def test_my_forecaster_parameter_validation():
    """Test parameter validation."""
    with pytest.raises(ValueError, match="param1 must be positive"):
        MyForecaster(param1=0)


def test_my_forecaster_different_horizons():
    """Test prediction with different horizons."""
    # ... test predicting different horizon than fit


def test_my_forecaster_panel_data(panel_time_series_factory):
    """Test with panel data."""
    y_panel = panel_time_series_factory(length=50, n_series=3, seed=42)

    forecaster = MyForecaster(param1=10)
    forecaster.fit(y_panel[:30], forecasting_horizon=5)

    y_pred = forecaster.predict(forecasting_horizon=5)
    # Validate panel structure preserved


def test_my_forecaster_update_predict():
    """Test update_predict method."""
    # ... test incremental learning workflow
```

**Run tests**: `uv run pytest tests/point_forecaster/test_my_forecaster.py -v`

#### 6. Add Docstring Examples (Doctests)

**Critical**: All public methods need docstring examples that actually run:

```python
def predict(self, forecasting_horizon=None, X=None, **params):
    """Generate forecasts.

    Examples
    --------
    >>> # Setup
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.point_forecaster import MyForecaster
    >>> time = pl.datetime_range(
    ...     start=datetime(2020, 1, 1),
    ...     end=datetime(2020, 1, 30),
    ...     interval="1d",
    ...     eager=True
    ... )
    >>> y = pl.DataFrame({"time": time, "value": range(30)})
    >>>
    >>> # Fit and predict
    >>> forecaster = MyForecaster(param1=5)
    >>> forecaster.fit(y, forecasting_horizon=3)
    MyForecaster(param1=5)
    >>> y_pred = forecaster.predict(forecasting_horizon=3)
    >>> len(y_pred)
    3

    """
```

**Run doctests**: `uv run pytest --doctest-modules src/yohou/point_forecaster/my_forecaster.py`

#### 7. Update Module Exports

Add to `src/yohou/point_forecaster/__init__.py`:

```python
from .my_forecaster import MyForecaster

__all__ = [
    # ... existing exports
    "MyForecaster",
]
```

#### 8. Quality Checks Checklist

Before committing, ensure:

- [ ] **Linting**: `uvx ruff check src/yohou/point_forecaster/my_forecaster.py`
- [ ] **Formatting**: `uvx ruff format src/yohou/point_forecaster/my_forecaster.py`
- [ ] **Type checking**: `uvx ty check src/yohou/point_forecaster/my_forecaster.py`
- [ ] **Docstring coverage**: `uvx interrogate src/yohou/point_forecaster/my_forecaster.py` (100% required)
- [ ] **Tests pass**: `uv run pytest tests/point_forecaster/test_my_forecaster.py`
- [ ] **Doctests pass**: `uv run pytest --doctest-modules src/yohou/point_forecaster/my_forecaster.py`
- [ ] **Pre-commit**: `uvx nox -s fix` (runs all quality checks)

#### 9. Real-World Examples from Codebase

**Pattern-based forecaster** (`src/yohou/point_forecaster/seasonality.py`):
- Stores seasonal pattern in `_extract_pattern()`
- Repeats/averages pattern in `_predict_from_pattern()`
- Validates sufficient data (at least 2 cycles)
- Supports 3 methods via `method` parameter

**Model-based forecaster** (`src/yohou/point_forecaster/fourier_seasonality.py`):
- Uses sklearn model (ElasticNet) for fitting
- Builds feature matrix in `_build_fourier_features()`
- Stores fitted models in `fourier_coefficients_` dict
- Parameters: `alpha`, `l1_ratio`, `n_harmonics`

**Trend forecaster** (`src/yohou/point_forecaster/polynomial_trend.py`):
- Fits polynomial with `numpy.polyfit()`
- Extrapolates via `numpy.polyval()`
- Simple stateless prediction
- Single parameter: `degree`

#### 10. Common Pitfalls & Solutions

**Problem**: Predictions don't have time columns
- **Solution**: Always call `self._add_time_columns(y_pred)` before returning

**Problem**: Panel data not handled
- **Solution**: Check `self.panel_group_names_` and iterate over `self.local_y_columns_`

**Problem**: Transformers not applied
- **Solution**: Use `_pre_fit()` to get transformed data (`y_t`, `X_t`)

**Problem**: Tests fail with type errors
- **Solution**: Use `StrictInt` from pydantic for integer params, validate in `__init__`

**Problem**: Doctests fail with repr mismatches
- **Solution**: Use exact repr format: `MyForecaster(param1=5)` not `MyForecaster(param1=5, ...)`

**Problem**: Linting fails on imports
- **Solution**: Order imports: stdlib → third-party → local, use `uvx ruff check --fix`

**Problem**: 100% docstring coverage not met
- **Solution**: Add NumPy-style docstrings to ALL public methods, classes, modules

#### 11. Advanced: Reduction Forecasters

For ML-based forecasters using sklearn estimators, extend `BaseReductionForecaster`:

```python
from yohou.point_forecaster.reduction import PointReductionForecaster

# Use directly with any sklearn regressor
forecaster = PointReductionForecaster(
    estimator=RandomForestRegressor(n_estimators=100),
    lags=[1, 2, 3, 7, 14],
    reduction_strategy="direct"  # or "multi-output"
)
```

**When to create custom reduction forecaster**:
- Need custom lag/feature engineering beyond `lags` parameter
- Want to bundle specific estimator with preset hyperparameters
- Implementing novel forecasting algorithm that reduces to supervised learning

**Example**: See `src/yohou/point_forecaster/reduction.py` for full implementation.

## Coding Conventions

### Type Annotations (Strictly Enforced)
```python
from pydantic import StrictInt, StrictFloat
import polars as pl

def forecast(y: pl.DataFrame, horizon: StrictInt) -> pl.DataFrame:
    # Use pydantic types for validation (StrictInt prevents float->int coercion)
    ...
```

### Polars Patterns
```python
import polars.selectors as cs  # Always import as cs

# Column selection: Exclude time column
df.select(~cs.by_name("time"))

# Accessing struct columns (panel data)
df_local = df[["time", "sales"]].unnest("sales")  # Flattens struct to columns

# Validate time consistency
from yohou.utils.validation import check_interval_consistency
interval = check_interval_consistency(df)  # Returns timedelta
```

### Transformers Pattern
```python
class MyTransformer(BaseTransformer):
    @property
    def observation_horizon(self) -> int:
        return self._window_size  # Must return observation horizon

    def fit(self, X: pl.DataFrame, y: Optional[pl.DataFrame] = None) -> "MyTransformer":
        self.reset(X)  # Initialize _X_observed
        # Store fitted state
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        # Use self._X_observed for context
        return transformed_X

    def update(self, X: pl.DataFrame) -> "MyTransformer":
        # Update _X_observed with new data
        self.reset(pl.concat([self._X_observed, X]))
        return self
```

### Forecasters Pattern (Reduction)
```python
class MyForecaster(BaseReductionForecaster, BasePointForecaster):
    def fit(self, y, X, forecasting_horizon):
        # Pre-fit handles transformers and sets up observation buffers
        y_t, X_t = BasePointForecaster._pre_fit(
            self, y=y, X=X,
            forecasting_horizon=forecasting_horizon
        )
        # Tabularize and fit sklearn estimator
        self.estimator_ = self._estimator_fit_one(y_t, X_t, forecasting_horizon)
        return self

    def _predict_one(self) -> pl.DataFrame:
        # Generate one-step (or multi-step) prediction
        y_pred = self._estimator_predict_one(self.estimator_)
        return self._add_time_columns(y_pred)
```

### Docstrings (NumPy Style - Strictly Enforced)
```python
def fit(self, X: pl.DataFrame) -> "MyClass":
    """Fits the transformer and returns it.

    Parameters
    ----------
    X : pl.DataFrame
        Input time series with "time" column.

    Returns
    -------
    self

    """
```
**Critical**: All docstrings MUST use **NumPy style**, enforced by `interrogate` at 100% coverage.
- Coverage requirements in `pyproject.toml`: `fail-under = 100`
- Excludes: tests, examples, `_version.py`, private/magic/init methods
- Ignores nested classes but NOT nested functions
- Run check: `uvx interrogate src/yohou` or via `uvx nox -s fix` (pre-commit hooks)

## Testing Patterns

### Test Infrastructure (`tests/estimator_checks.py`)
Systematic validation via **check functions** pattern (inspired by sklearn):
```python
from estimator_checks import _yield_yohou_transformer_checks

# Generate all applicable checks based on tags
for check_name, check_func, check_kwargs in _yield_yohou_transformer_checks(
    transformer_fitted, X_train, X_test, tags={"invertible": True, "stateful": True}
):
    check_func(transformer_fitted, **check_kwargs)
```

**Three check categories**:
1. **Core Yohou Checks** (12 functions): Time series-specific validation
   - `check_fit_sets_attributes()`: Validates `feature_names_in_`, `n_features_in_`, `_observation_horizon`
   - `check_observation_horizon_not_fitted()`: Ensures `NotFittedError` before fit for stateful transformers
   - `check_transform_preserves_time()`: Validates "time" column preservation
   - `check_update_extends_memory()`: Tests `update()` behavior
   - `check_reset_replaces_memory()`: Tests `reset()` behavior
2. **Enhanced sklearn Checks** (6 functions): Adapted from sklearn patterns
   - `check_transformer_clonable()`: Tests `clone()` compatibility
   - `check_fit_idempotent()`: Multiple fits don't change behavior
   - `check_get_feature_names_out()`: Feature name propagation
3. **Check Generator**: `_yield_yohou_transformer_checks()` dynamically generates applicable checks based on tags

**Tags control check selection**:
- `invertible: bool` - Whether transformer implements `inverse_transform()`
- `stateful: bool` - Whether `observation_horizon > 0`

### Test Fixtures (`tests/conftest.py`)
**Data generation factories** (all return callables):
```python
def test_example(time_series_factory):
    X = time_series_factory(length=50, n_features=2, seed=42)
    # Returns pl.DataFrame with "time" column + feature columns
```

Key fixtures:
- `time_series_factory()`: Generates global time series with datetime "time" column
- `panel_time_series_factory()`: Creates struct columns for panel data
- `edge_case_datasets_factory()`: Minimal length, single feature, etc.
- `base_time_series`: Session-scoped immutable dataset (performance optimization)
- `transformer_registry`: Fixture with metadata and expected failures per transformer

**Dummy classes for testing**:
- `SimpleTransformer`: Identity transformer with configurable `observation_horizon`
- `StatelessTransformer`: `observation_horizon=0`, works without fitting
- `InvertibleTransformer`: Has perfect `inverse_transform()`

### Test Organization
- Tests in `tests/` mirror `src/yohou/` structure
- **Run tests locally**: `uv run pytest` (quick, no coverage) or `uvx nox -s test` (full coverage report)
- **Run specific test**: `uv run pytest tests/decomposition/test_polynomial_trend.py -v`
- Use sample data with `pl.datetime_range` for time columns:
  ```python
  time = pl.DataFrame({
      "time": pl.datetime_range(start=datetime(2021, 12, 16),
                                 end=datetime(2021, 12, 16, 0, 0, 21),
                                 interval="1s", eager=True)
  })
  ```
- Test both global and local (panel) data scenarios
- Use `sklearn.model_selection.train_test_split` with `shuffle=False` for time series splits
- Parametrize tests heavily with `@pytest.mark.parametrize`

**Pytest configuration** (`pyproject.toml`):
- `--doctest-modules`: Runs doctests in source code
- `testpaths = ["src", "tests"]`: Tests both unit tests and inline doctests

## Key Utilities
- `yohou.utils.tabularization.tabularize(df, lags)`: Create lagged features
- `yohou.utils.polars.inspect_locality(df)`: Parse global/local columns
- `yohou.utils.polars.concat_struct(items, how)`: Merge panel data struct columns
- `yohou.utils.validation.check_interval_consistency(df)`: Validate time spacing
- `yohou.utils.validation.check_inputs(y, X)`: Validate all inputs have matching intervals

## Decomposition Module (`src/yohou/decomposition/`)

**Architecture pattern**: Time series decomposition into sequential additive components.

### Available Decomposition Forecasters
1. **Trend Forecasters**:
   - `PolynomialTrendForecaster`: Fits polynomial trend via `numpy.polyfit()`, extrapolates with `numpy.polyval()`
   - `ExponentialTrendForecaster`: Exponential growth/decay patterns

2. **Seasonality Forecasters**:
   - `SeasonalityForecaster`: Pattern-based seasonality (naive, average, median methods)
   - `FourierSeasonalityForecaster`: Fourier basis functions with ElasticNet fitting

3. **Meta-Forecaster**:
   - `Decomposer`: Orchestrates sequential decomposition workflow

### Decomposer Usage Pattern
```python
from yohou.decomposition import Decomposer, PolynomialTrendForecaster, SeasonalityForecaster
from yohou.preprocessing import LogTransform

# Additive decomposition: trend + seasonality + residual
forecaster = Decomposer([
    ("trend", PolynomialTrendForecaster(degree=2)),
    ("season", SeasonalityForecaster(seasonality=7, method="average")),
])
forecaster.fit(y, forecasting_horizon=7)
y_pred = forecaster.predict(forecasting_horizon=7)

# Multiplicative decomposition (additive in log-space)
forecaster_mult = Decomposer(
    [("trend", PolynomialTrendForecaster(degree=1)),
     ("season", SeasonalityForecaster(seasonality=12))],
    target_transformer=LogTransform()
)

# Inspect intermediate residuals
forecaster_inspect = Decomposer(
    [("trend", ...), ("season", ...)],
    store_residuals=True
)
forecaster_inspect.fit(y, forecasting_horizon=7)
trend_residuals = forecaster_inspect.residuals_["trend"]
season_residuals = forecaster_inspect.residuals_["season"]
```

### Key Implementation Details
- **Sequential fitting**: Each component models residuals from all previous components
  ```python
  # Inside Decomposer.fit():
  residuals = y_t  # Start with target
  for name, forecaster in self.forecasters:
      forecaster.fit(residuals, X_t, forecasting_horizon)
      y_pred_train = forecaster.predict(...)
      residuals = residuals - y_pred_train  # Update residuals
  ```
- **Prediction aggregation**: Final prediction = sum of all component predictions
- **Inheritance**: Extends `_BaseComposition` (sklearn's meta-estimator base class)
- **Metadata routing**: Properly routes params to nested forecasters via `process_routing()`
- **Validation**: All component forecasters must be point forecasters (no interval forecasters)

### Common Patterns
- **Classical decomposition**: Trend → Seasonal → Residual (use naive/baseline for residual)
- **STL-style**: Polynomial trend + Fourier seasonality
- **Hierarchical**: Multiple levels of seasonality (daily + weekly + yearly)
- **Debugging**: Enable `store_residuals=True` to inspect what each component captures



## Plans & Documentation
- **Copilot Plans Directory**: `.github/copilot_plans/` contains detailed implementation plans
  - `transformer-testing-infrastructure.md`: Comprehensive transformer testing guide
  - `forecaster-testing-infrastructure.md`: Forecaster testing patterns
  - `monthly-interval-support.md`: Feature implementation plan
- When asked to create a new plan, save it in `.github/copilot_plans/`
- Plans use markdown format with detailed specifications, rationale, and implementation steps

## Contributing Workflow
- **Branch naming**: `<type>/<description>` where type is `feature|fix|docs|tests`
- **Commits**: Must be signed off (`git commit -s`)
- **PR titles**: Follow [Conventional Commits](https://www.conventionalcommits.org/) format
- **Pre-commit hooks**: Run automatically on commit or manually with `uvx nox -s fix`
- **CI validation**: All nox sessions must pass (test, fix, docs)
- **Nox configuration**: `noxfile.py` at project root with uv backend
- **Code compatibility**: Target Python 3.12+, cross-platform (Windows, macOS, Linux)
- **License**: All contributions under BSD License

## Examples & Marimo Notebooks

Yohou uses **marimo** for interactive examples - a reactive notebook system where cells automatically re-run when dependencies change.

### Running Examples
```bash
# Start marimo notebook server
marimo edit examples/air_passengers_tutorial.py

# Or run all cells non-interactively
marimo run examples/air_passengers_tutorial.py
```

### Marimo Notebook Pattern
Examples follow this structure (see `examples/air_passengers_tutorial.py`):
```python
import marimo

app = marimo.App(width="medium")

@app.cell
def _():
    # Import cells - define dependencies
    import polars as pl
    from yohou.point_forecaster import PointReductionForecaster
    return (pl, PointReductionForecaster)

@app.cell
def _(pl):
    # Data loading cell - uses imports from previous cell
    df = pl.read_csv("data.csv")
    return (df,)

@app.cell
def _(mo):
    # Interactive controls with marimo UI
    horizon_slider = mo.ui.slider(start=1, stop=24, value=12,
                                   label="Forecast Horizon")
    return (horizon_slider,)
```

**Key patterns**:
- Each cell is a function decorated with `@app.cell`
- Return values in tuples: `return (var1, var2)`
- Cells automatically re-run when dependencies change (reactive execution)
- Use `mo.ui.*` for interactive controls (sliders, dropdowns, etc.)
- Use `mo.md(r"""...""")` for markdown documentation cells
- Plotly figures render automatically without explicit `.show()`

### Example Topics Covered
- `air_passengers_tutorial.py`: Full forecasting workflow
  - Baseline models (SeasonalNaive)
  - Preprocessing pipelines (LogTransform, SeasonalDifferencing, LagTransformer)
  - Interactive parameter exploration with sliders
  - Hyperparameter optimization with SearchCV + Optuna
  - Incremental learning with `update_predict()`
  - Visualization with plotly

## CI/CD & GitHub Actions

### Workflow Files (`.github/workflows/`)
1. **`tests-os-coverage.yml`**: Comprehensive coverage testing
   - Matrix: Python 3.12+ across Windows/macOS/Linux
   - Uses `uv` for fast dependency installation
   - **CI workflow**: `uvx nox -s test` (installs nox as uv tool first)
   - Concurrent execution with auto-cancellation on new pushes

2. **`lint.yml`**: Code quality checks
   - **CI workflow**: `uvx nox -s fix`
   - Validates ruff + ty + interrogate (100% docstring coverage)

3. **`release.yml`**: PyPI package publishing
   - Automated on version tags (e.g., `v0.1.0`)
   - Builds with hatchling + hatch-vcs
   - Publishes to PyPI with trusted publishing

### CI Debugging Tips
- **Local development**: Always use `uvx nox` (uv's automatic tool runner)
  ```bash
  uvx nox -s test  # Runs same tests as CI
  uvx nox -s fix   # Runs same quality checks as CI
  ```
- **CI vs Local difference**: CI uses `uv tool install nox` + `nox`, local uses `uvx nox` (automatic)
- **Coverage failures**: Check `coverage.{python}.xml` or HTML report in temp dir
- **Cross-platform issues**: Use `pathlib` over string paths, avoid shell-specific commands
- **Dependency conflicts**: Run `uv sync` to regenerate lockfile

## Custom Metrics & Scoring

### Creating Custom Metrics
Extend `BaseScorer` for time series-specific evaluation:

```python
from yohou.metrics.base import BasePointScorer
import polars as pl
import polars.selectors as cs

class MAPE(BasePointScorer):
    """Mean Absolute Percentage Error."""

    @property
    def prediction_types(self) -> set[str]:
        return {"point"}

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> float:
        # Align predictions with ground truth (removes time columns)
        y_truth, y_pred = self._validate_inputs(y_truth, y_pred)

        # Compute metric on numeric columns (excluding time)
        mape = (
            ((y_truth.select(cs.numeric()) - y_pred.select(cs.numeric())).abs()
             / y_truth.select(cs.numeric()).abs())
            .mean()
            .to_numpy()[0, 0]
        )
        return float(mape * 100)
```

### Metric Categories
1. **Point Forecasting** (`BasePointScorer`):
   - `MAE`, `MSE`, `RMSE` (in `src/yohou/metrics/point.py`)
   - Returns: `float` (scalar error)

2. **Interval Forecasting** (`BaseIntervalScorer`):
   - Coverage, width, calibration metrics
   - Returns: `pl.DataFrame` with per-horizon scores

3. **Conformity Scores** (`BaseConformityScorer`):
   - `Residual`, `NormalizedResidual`, `GammaResidual`, `QuantileResidual`
   - Used internally by conformal predictors
   - Returns: `pl.DataFrame` with conformity values

### Using Metrics in SearchCV
```python
from yohou.metrics import MAE, MSE
from yohou.model_selection import SearchCV

# Single metric
search = SearchCV(forecaster=model, scoring=MAE(), ...)

# Multi-metric (returns dict of scores)
search = SearchCV(
    forecaster=model,
    scoring={"mae": MAE(), "mse": MSE()},
    refit="mae",  # Which metric to use for best model selection
    ...
)
```

**Critical**: Metrics must implement `score(y_truth, y_pred)` where:
- `y_truth`: Has "time" column
- `y_pred`: Has "observed_time" and "time" columns (forecaster output format)
- `_validate_inputs()` handles time alignment automatically

## Debugging Time Series Issues

### Common Patterns & Solutions

**1. Observation Horizon Errors**
```python
# Problem: NotFittedError when accessing observation_horizon
transformer.observation_horizon  # Fails if not fitted

# Solution: Always fit before accessing (for stateful transformers)
transformer.fit(X)
print(transformer.observation_horizon)  # Now works

# Or check if stateless (observation_horizon=0)
if hasattr(transformer, '_observation_horizon') and transformer._observation_horizon == 0:
    # Stateless transformer, can use without fitting
    pass
```

**2. Time Column Mismatches**
```python
# Problem: Predictions don't align with test data
y_pred = forecaster.predict(forecasting_horizon=3)
# y_pred has "observed_time" + "time" columns

# Solution: Use metric's _validate_inputs() for alignment
from yohou.metrics import MAE
scorer = MAE()
y_truth, y_pred_aligned = scorer._validate_inputs(y_test, y_pred)
```

**3. Panel Data Column Access**
```python
# Problem: Can't access individual series in panel data
y_panel = pl.DataFrame({
    "time": [...],
    "sales__store_1": [100, 110, ...],  # Prefixed columns
    "sales__store_2": [150, 160, ...],
})

# Solution 1: Use get_group_df to extract a single group with unprefixed columns
from yohou.utils.panel import get_group_df
y_store1 = get_group_df(y_panel, group_name="sales", schema={"store_1": pl.Float64, "store_2": pl.Float64})
# Returns DataFrame with columns: time, store_1, store_2

# Solution 2: Inspect locality to understand structure
from yohou.utils.panel import inspect_locality
global_cols, panel_groups = inspect_locality(y_panel)
print(panel_groups)  # {'sales': ['sales__store_1', 'sales__store_2']}

# Solution 3: Direct column selection if you know the names
y_store1_only = y_panel.select(["time", "sales__store_1"])
```

**4. Memory Management in Streaming**
```python
# Problem: Transformer memory grows unbounded
for batch in data_stream:
    forecaster.update(batch)  # Memory keeps growing

# Solution: update() automatically calls reset() to maintain fixed window
# Observation horizon is maintained at transformer's configured size
transformer.observation_horizon  # e.g., 12 steps
# _X_observed always keeps last 12 rows, no matter how many updates
```

**5. Debugging Recursive Predictions**
```python
# Problem: Multi-step predictions diverge or explode
forecaster.fit(y_train, forecasting_horizon=1)
y_pred = forecaster.predict(forecasting_horizon=12)  # Predicts 12 steps

# Debug: Check intermediate predictions
# Forecaster applies model recursively: pred_t+1 → pred_t+2 → ... → pred_t+12
# Enable logging in BaseReductionForecaster._predict_one() to see each step

# Common issues:
# - Lag features not updated properly between steps
# - Transformations not inverted correctly (check target_transformer)
# - Model extrapolating beyond training distribution
```

**6. Polars Expression Debugging**
```python
# Problem: Complex polars expression fails silently
df.select(pl.col("value").rolling_mean(window_size=12))

# Solution: Build incrementally and inspect
result = df.select([
    pl.col("time"),
    pl.col("value"),
    pl.col("value").rolling_mean(window_size=12).alias("rolling_mean")
])
print(result)  # Verify intermediate results

# Use .explain() for query plans
print(df.lazy().select(pl.col("value").mean()).explain())
```

### Debugging Tools
- **Check fitted attributes**: `sklearn.utils.validation.check_is_fitted(obj, 'attr_name')`
- **Inspect data shapes**: `print(f"Shape: {df.shape}, Columns: {df.columns}")`
- **Validate time consistency**: `yohou.utils.validation.check_interval_consistency(df)`
- **Test transformers in isolation**: Use `tests/conftest.py` dummy transformers as templates
- **Pytest with verbose**: `uv run pytest -vv tests/path/to/test.py::test_name`
- **Coverage gaps**: `uvx nox -s test` then open `htmlcov/index.html`
- **Rerun failed tests only**: `uv run pytest --lf` (last-failed)
- **Step through with debugger**: `uv run pytest --pdb tests/path/to/test.py::test_name`

## Performance & Optimization

### Panel Data Performance
**Prefixed columns are efficient for panel data** - use column selection wisely:
```python
# ❌ Slow: Select all panel columns individually
result = df.select([
    "time",
    "sales__store_1",
    "sales__store_2",
    # ... hundreds more
])

# ✅ Fast: Use regex or column selectors
import polars.selectors as cs
result = df.select([
    cs.by_name("time"),
    cs.matches("^sales__.*"),  # Select all sales columns
])

# ✅ Efficient: Use get_group_df for processing a single group
from yohou.utils.panel import get_group_df
sales_data = get_group_df(df, "sales", local_y_schema)
# Returns unprefixed columns for easier manipulation
```

### Memory Optimization
1. **Use observation_horizon wisely**: Higher horizons = more memory
   ```python
   # For transformers that need last 12 observations
   LagTransformer(lags=[1, 2, 3])  # observation_horizon = 3
   LagTransformer(lags=[1, 12, 24])  # observation_horizon = 24 (stores more)
   ```

2. **Streaming updates**: `update()` maintains fixed-size windows
   ```python
   # Memory stays constant regardless of updates
   for new_data in stream:
       forecaster.update(new_data)  # Only keeps last observation_horizon rows
   ```

3. **Lazy evaluation**: Use polars lazy API for large datasets
   ```python
   # Eager: Loads all data immediately
   df = pl.read_csv("large_file.csv")

   # Lazy: Optimized query plan, streams results
   df = pl.scan_csv("large_file.csv").filter(...).collect()
   ```

### Parallel Execution
SearchCV, ColumnForecaster and pipelines support parallel execution via `n_jobs`:
```python
from yohou.model_selection import SearchCV
from yohou.pipeline import FeatureUnion
from yohou.forecaster.composition import ColumnForecaster

# Parallel hyperparameter search (across CV folds)
search = SearchCV(forecaster=model, n_jobs=-1, ...)  # Use all cores

# Parallel feature engineering
features = FeatureUnion([
    ('lags', LagTransformer([1, 2, 3])),
    ('rolling', RollingMeanTransformer(window=12)),
], n_jobs=2)  # Compute both transformers in parallel

# Parallel column forecasting
forecaster = ColumnForecaster([
    ("sales", PointReductionForecaster(), ["sales"]),
    ("inventory", SeasonalNaive(), ["inventory"]),
], n_jobs=-1)  # Forecast columns in parallel
```

**Notes**:
- `n_jobs=-1`: Use all available cores
- `n_jobs=None` or `n_jobs=1`: Sequential execution (easier to debug)
- Overhead exists for small datasets - benchmark before parallelizing
- Batch optimization in SearchCV: Uses `effective_n_jobs()` for Optuna trials

### Profiling Large Datasets
```bash
# Time-based profiling
uv run python -m cProfile -o profile.stats your_script.py
uv run python -c "import pstats; p=pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

# Memory profiling
uv run python -m memory_profiler your_script.py

# Polars-specific: Check query plans
import polars as pl
df.lazy().select([...]).explain()  # See optimization plan
df.lazy().select([...]).show_graph()  # Visual query graph
```

### Optimization Checklist
- ✅ Use prefixed column names for panel data (natural polars column model)
- ✅ Keep `observation_horizon` as small as possible
- ✅ Use lazy polars API for large files (`scan_csv`, `scan_parquet`)
- ✅ Enable `n_jobs` for SearchCV, FeatureUnion, and ColumnForecaster (if dataset is large)
- ✅ Profile before optimizing - measure actual bottlenecks
- ✅ Consider data types: use `pl.Int32` vs `pl.Int64` when appropriate
- ✅ Use column selectors (`cs.matches()`, `cs.starts_with()`) for panel data
- ❌ Don't select panel columns one-by-one (use regex patterns)
- ❌ Don't parallelize small datasets (overhead > benefit)
- ❌ Don't load entire dataset into memory if streaming is possible
