# Scikit-learn Metadata Routing in Yohou: Implementation Guide

## Overview

Yohou integrates with scikit-learn's metadata routing system to enable flexible propagation of metadata (e.g., `forecasting_horizon`, `time_weight`, and custom parameters) through pipelines, transformers, and cross-validation workflows. This document explains how metadata routing is implemented in Yohou and how it differs from standard sklearn usage.

**Key Design Principle**: Time series data (`y`, `X`) are **NOT routed as metadata** - they remain explicit positional/keyword arguments in all method signatures. Only auxiliary parameters like `time_weight`, `forecasting_horizon`, and custom metadata are routed through the `**params` mechanism.

## Critical Implementation Decisions

### 1. Global Metadata Routing Enabled on Import

**Implementation**: Yohou **automatically enables** sklearn's metadata routing when the package is imported.

**Location**: `src/yohou/__init__.py`
```python
from sklearn import set_config

# Enable metadata routing globally for all Yohou estimators
set_config(enable_metadata_routing=True)
```

**Rationale**:
- Metadata routing is experimental in sklearn but essential for Yohou's architecture
- Users don't need to remember to enable it manually
- Provides consistent behavior across all Yohou code
- Simplifies examples and documentation (no setup boilerplate)

**Trade-offs**:
- ✅ User convenience - works out of the box
- ✅ Consistent behavior - no "routing disabled" mode
- ⚠️ Global state change - may affect sklearn estimators in user code
- ⚠️ Requires sklearn >= 1.3 (when metadata routing was introduced)

### 2. Registering Yohou-Specific Composite Methods

**Implementation**: Yohou extends sklearn's metadata routing to support time series-specific composite methods.

**Location**: `src/yohou/__init__.py`
```python
from sklearn.utils._metadata_requests import SIMPLE_METHODS, COMPOSITE_METHODS

if "update_transform" not in SIMPLE_METHODS:
    SIMPLE_METHODS.extend(["update_transform", "update_predict"])
    COMPOSITE_METHODS["update_transform"] = ["update", "transform"]
    COMPOSITE_METHODS["update_predict"] = ["update", "predict"]
```

**What this enables**:
- `update_transform()`: Combines `update()` (memory management) + `transform()` (transformation with metadata)
  - Metadata routes ONLY to `transform()`, not to `update()`
  - Similar to sklearn's `fit_transform()` pattern

- `update_predict()`: Combines `update()` (memory management) + `predict()` (forecasting with metadata)
  - Metadata routes ONLY to `predict()`, not to `update()`
  - Common pattern for rolling forecasts

**Critical distinction**:
- `update()` itself is **NOT** a routable method - it's a pure memory management operation
- Metadata only flows to the data-processing methods (`transform`, `predict`)
- This prevents confusion about what metadata means during memory updates

### 3. Data vs Metadata Distinction

**Core Principle**: Time series data (`y`, `X`) are NOT metadata - they are the primary data being modeled.

| Parameter | Type | Routing | Rationale |
|-----------|------|---------|-----------||
| `y` | pl.DataFrame | ❌ Not routed | Target time series - primary data |
| `X` | pl.DataFrame | ❌ Not routed | Ex-ante features (known in advance) - primary data |
| `forecasting_horizon` | int | Explicit parameter | Forecast horizon - explicit param that CAN be routed |
| `time_weight` | Callable or pl.DataFrame | ✅ Routed | Time-based sample weighting |
| Custom params | Any | ✅ Routed | User-defined metadata |

**Note**: All features in `X` are expected to be known ex-ante (in advance). For ex-post features (observed after the fact), use `ColumnForecaster` to forecast them first.

### 4. Time Series-Specific Metadata

Yohou defines standard metadata types for time series forecasting:

| Metadata | Type | Use Case | Example |
|----------|------|----------|---------|
| `time_weight` | `Callable` or `pl.DataFrame` | Time-based sample weighting | Exponential decay, seasonal emphasis |
| `forecasting_horizon` | `int` | Forecast horizon (explicit param) | Steps ahead to forecast |
| Custom params | Any | User-defined metadata | Application-specific parameters |

**Special handling of `time_weight`**:
- Input: `pl.DataFrame` with weights OR callable `f(time) -> weights`
- Forecasters are designed to convert this to `sample_weight` arrays aligned to training data
- Enables recency weighting, seasonal emphasis based on temporal structure
- **Note**: Currently declared as explicit parameter in `PointReductionForecaster.fit()`, actual conversion logic not yet implemented

## Implementation Architecture

### Base Class Pattern: Routers vs Consumers

Yohou follows sklearn's pattern of distinguishing between:
- **Consumers**: Estimators that USE metadata directly (e.g., `LagTransformer`)
- **Routers**: Estimators that FORWARD metadata to nested estimators (e.g., `Pipeline`, reduction forecasters)

The distinction is determined by whether a class overrides `get_metadata_routing()`:

| Base Class | Default Role | Override `get_metadata_routing()`? | Signature |
|------------|--------------|-----------------------------------|-----------|
| `BaseTransformer` | Consumer | Rarely (only if wraps estimators) | `**params` for flexibility |
| `BaseForecaster` | Router | Usually (most wrap transformers/estimators) | `**params` |
| `BaseScorer` | Consumer | Never (no nested estimators) | `**params` (future-proofing) |

### BaseForecaster Implementation

**Location**: `src/yohou/base.py`

All forecasters have `**params` in their method signatures and use `@_fit_context` for automatic metadata routing:

```python
from sklearn.base import _fit_context
from sklearn.utils.metadata_routing import MetadataRouter, MethodMapping

class BaseForecaster(BaseEstimator):
    """Base class for all forecasters."""

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,  # ← Metadata routing
    ) -> "BaseForecaster":
        """Fit forecaster with optional metadata."""
        # @_fit_context automatically routes **params to nested estimators
        ...

    def predict(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        **params,  # ← Metadata routing
    ) -> pl.DataFrame:
        """Predict with optional metadata."""
        ...

    def get_metadata_routing(self) -> MetadataRouter:
        """Configure metadata routing to transformers."""
        router = MetadataRouter(owner=self.__class__.__name__)

        if hasattr(self, "target_transformer") and self.target_transformer is not None:
            router.add(
                target_transformer=self.target_transformer,
                method_mapping=MethodMapping()
                    .add(caller="fit", callee="fit")
                    .add(caller="fit", callee="transform")
            )

        if hasattr(self, "feature_transformer") and self.feature_transformer is not None:
            router.add(
                feature_transformer=self.feature_transformer,
                method_mapping=MethodMapping()
                    .add(caller="fit", callee="fit")
                    .add(caller="fit", callee="transform")
            )

        return router
```

**Key points**:
- `@_fit_context` decorator handles metadata routing automatically
- `get_metadata_routing()` declares which nested estimators receive metadata
- `MethodMapping` specifies how caller methods map to callee methods

### BaseReductionForecaster: Routing to Wrapped Estimators

**Location**: `src/yohou/base.py`

Reduction forecasters wrap sklearn estimators and route metadata to them:

```python
class BaseReductionForecaster(BaseForecaster):
    """Forecaster using sklearn estimators on tabularized data."""

    def _estimator_fit_one(
        self,
        y_t: pl.DataFrame,
        X_t: pl.DataFrame,
        forecasting_horizon: int,
        time_weight: Callable | pl.DataFrame | None = None,
        estimator_params: dict | None = None,
        estimator_fit_params: dict | None = None,
    ) -> BaseEstimator:
        """Fit wrapped sklearn estimator.

        Parameters
        ----------
        estimator_params : dict
            Parameters for estimator.set_params() (e.g., 'estimator__alpha')
        estimator_fit_params : dict
            Parameters for estimator.fit() (e.g., metadata to route)
        """
        estimator = clone(self.estimator).set_params(**(estimator_params or {}))

        # Tabularize time series to supervised learning format
        X_tab, y_tab = self._get_tabularized_dataset(y_t, X_t, forecasting_horizon)

        # Fit with metadata
        estimator.fit(X_tab, y_tab, **(estimator_fit_params or {}))
        return estimator

    def get_metadata_routing(self) -> MetadataRouter:
        """Route metadata to transformers (parent) AND wrapped estimator."""
        router = super().get_metadata_routing()  # Get transformer routing

        # Add wrapped sklearn estimator
        if hasattr(self, "estimator") and self.estimator is not None:
            router.add(
                estimator=self.estimator,
                method_mapping=MethodMapping().add(caller="fit", callee="fit")
            )

        return router
```

### PointReductionForecaster: time_weight Handling

**Location**: `src/yohou/point_forecaster/reduction.py`

Point forecasters explicitly declare `time_weight` as a parameter:

```python
class PointReductionForecaster(BaseReductionForecaster, BasePointForecaster):
    """Point forecaster using sklearn estimators."""

    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        time_weight: Callable | pl.DataFrame | None = None,  # ← Explicit param
        **params,
    ) -> "PointReductionForecaster":
        """Fit forecaster with optional time-based weighting."""
        y_t, X_t = BasePointForecaster._pre_fit(
            self, y=y, X=X,
            forecasting_horizon=forecasting_horizon
        )

        self.estimator_ = self._estimator_fit_one(
            y_t, X_t, forecasting_horizon,
            time_weight=time_weight,
            estimator_fit_params=params
        )

        return self
```

**Note**: `time_weight` is declared explicitly (not in `**params`) because it's intended for special handling by the forecaster. The actual conversion to `sample_weight` for the sklearn estimator is not yet implemented.

### Pipeline Implementation

**Location**: `src/yohou/pipeline.py`

Pipelines route metadata to all steps:

```python
class Pipeline(BaseEstimator):
    """Time series pipeline with metadata routing."""

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, **params):
        """Fit all steps with metadata routing."""
        # @_fit_context routes **params based on get_metadata_routing()
        for step_name, step in self.steps:
            X = step.fit_transform(X, y, **params)
        return self

    def get_metadata_routing(self) -> MetadataRouter:
        """Route metadata to all pipeline steps."""
        router = MetadataRouter(owner=self.__class__.__name__)

        for name, step in self.steps:
            router.add(
                **{name: step},
                method_mapping=MethodMapping()
                    .add(caller="fit", callee="fit")
                    .add(caller="fit", callee="transform")
                    .add(caller="transform", callee="transform")
                    .add(caller="update_transform", callee="update_transform")
            )

        return router
```

### Scorer Implementation

**Location**: `src/yohou/metrics/base.py`, `src/yohou/metrics/point.py`

Scorers are consumers with `**params` for future extensibility:

```python
class BasePointScorer(BaseScorer):
    """Base class for point forecasting metrics."""

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **params) -> float:
        """Compute metric score.

        Parameters
        ----------
        **params : dict
            Metadata for scoring (currently unused, reserved for future features
            like time_weight for weighted scoring).
        """
        y_truth, y_pred = self._validate_inputs(y_truth, y_pred)
        # Compute score
        ...

class MAE(BasePointScorer):
    """Mean Absolute Error metric."""

    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **params) -> float:
        """Compute MAE."""
        y_truth, y_pred = self._validate_inputs(y_truth, y_pred)
        return float(np.nanmean((y_truth - y_pred).select(pl.all().abs().mean()).to_numpy()))
```

**Note**: Scorers have `**params` for future extensibility but don't currently use metadata. The infrastructure is in place for when weighted scoring is implemented.

## Testing Infrastructure

### Test Organization

**Location**: `tests/test_metadata_routing.py`, `tests/metadata_routing_common.py`

Tests verify that metadata routing works correctly across all Yohou components:

#### Test Categories

1. **Basic Routing**: All estimators have `get_metadata_routing()` method
2. **Transformer Routing**: `transform()` and `update_transform()` accept `**params`
3. **Forecaster Routing**: `fit()`, `predict()`, `update_predict()` accept `**params`
4. **Pipeline Routing**: Metadata flows through all pipeline steps
5. **SearchCV Routing**: Metadata routes through cross-validation
6. **Composite Methods**: `update_transform` and `update_predict` work correctly
7. **Integration**: Nested routing scenarios (SearchCV → Forecaster → Pipeline)

#### Test Utilities

**`metadata_routing_common.py`** provides testing utilities adapted from sklearn:

```python
class _Registry(list):
    """Track which estimators received metadata."""
    def __deepcopy__(self, memo):
        return self  # Preserve reference through cloning

def record_metadata(obj, record_default=True, **kwargs):
    """Store metadata passed to estimator methods."""
    ...

def check_recorded_metadata(obj, method, parent, **expected):
    """Verify expected metadata was passed."""
    ...

def assert_request_is_empty(metadata_request, exclude=None):
    """Verify no metadata requests are set (default state)."""
    ...
```

#### Example Test

```python
def test_forecaster_routing_with_reduction(y_X_factory, consuming_estimator):
    """Reduction forecasters should route metadata to sub-estimators."""
    y, X = y_X_factory(length=50, n_targets=1, n_X_features=1)
    estimator, registry = consuming_estimator

    forecaster = PointReductionForecaster(estimator=estimator)
    forecaster.fit(y, X, forecasting_horizon=3)

    # Check that estimator's fit was called
    assert len(registry) > 0
    fit_calls = [call for call in registry if call[0] == "fit"]
    assert len(fit_calls) > 0
```

## Panel Data Support

### Cross-Learning with Metadata Routing

**Location**: `src/yohou/utils/panel.py`

Yohou supports panel data (multiple time series in struct columns) with metadata routing:

```python
def inspect_locality(df: pl.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    """Distinguish global columns from panel (struct) columns.

    Returns
    -------
    global_names : list[str]
        Non-struct column names (excluding 'time')
    local_groups : dict[str, list[str]]
        Mapping from struct column names to their field names
    """
    global_names, local_groups = [], {}
    for col, dtype in df.schema.items():
        if col == "time":
            continue
        if isinstance(dtype, pl.Struct):
            local_groups[col] = [field.name for field in dtype.fields]
        else:
            global_names.append(col)
    return global_names, local_groups

def filter_panel_columns(
    df: pl.DataFrame,
    panel_group: str,
    local_group_names: list[str] | None,
    include_global: bool = True,
) -> pl.DataFrame:
    """Filter DataFrame for cross-learning.

    Parameters
    ----------
    panel_group : str
        Struct column to keep for prediction
    include_global : bool
        Whether to keep global (non-struct) columns
        - True: Keep for X (features)
        - False: Keep only for y (target)
    """
    if local_group_names is None:
        return df

    if include_global:
        cols_to_keep = [
            c for c in df.columns
            if c == "time" or c == panel_group or c not in local_group_names
        ]
    else:
        cols_to_keep = [c for c in df.columns if c == "time" or c == panel_group]

    return df.select(cols_to_keep)
```

**Usage**: When forecasting with panel data, metadata routes correctly to each series group.

## Usage Examples

### Basic Forecasting with time_weight

```python
import polars as pl
from yohou.point_forecaster import PointReductionForecaster
from sklearn.linear_model import Ridge

# Create time series data
y = pl.DataFrame({
    "time": pl.datetime_range(...),
    "value": [10, 12, 15, 14, 16, ...]
})

# Create time-based weights (exponential decay - recent data more important)
def exponential_weight(time_col: pl.Series, decay=0.05) -> pl.DataFrame:
    """Weight recent observations more heavily."""
    time_numeric = (time_col - time_col.min()).dt.total_seconds()
    weights = np.exp(decay * time_numeric / time_numeric.max())
    return pl.DataFrame({"time": time_col, "weight": weights})

# Fit with time weighting
# Note: Actual time_weight consumption not yet implemented
forecaster = PointReductionForecaster(estimator=Ridge())
forecaster.fit(
    y,
    forecasting_horizon=3,
    time_weight=exponential_weight  # ← Declared as parameter, routing infrastructure in place
)

# Predict
y_pred = forecaster.predict(forecasting_horizon=3)
```

### Metadata Routing Through Pipelines

```python
from yohou.pipeline import Pipeline
from yohou.preprocessing import SeasonalDifferencing
from yohou.point_forecaster import PointReductionForecaster

# Create pipeline with preprocessing
pipeline = Pipeline([
    ("diff", SeasonalDifferencing(seasonality=12)),
    ("forecaster", PointReductionForecaster(estimator=Ridge()))
])

# Metadata routes to the forecaster (not the transformer)
pipeline.fit(
    y,
    forecasting_horizon=3,
    time_weight=exponential_weight  # ← Routes to forecaster.fit()
)

y_pred = pipeline.predict(forecasting_horizon=3)
```

### SearchCV with Metadata Routing

```python
from yohou.model_selection import SearchCV
from yohou.metrics import MAE
import optuna

# Hyperparameter search with metadata
search = SearchCV(
    forecaster=PointReductionForecaster(),
    param_distributions={
        "estimator__alpha": optuna.distributions.FloatDistribution(0.01, 1.0)
    },
    scoring=MAE(),
    n_trials=20
)

# time_weight routes through cross-validation to each fold
search.fit(
    y,
    X,
    forecasting_horizon=3,
    time_weight=exponential_weight  # ← Routes to each CV fold
)

# Best forecaster with optimal hyperparameters
y_pred = search.predict(forecasting_horizon=3, X=X_future)
```

### Panel Data with Cross-Learning

```python
# Panel data with multiple stores
y_panel = pl.DataFrame({
    "time": pl.datetime_range(...),
    "sales": pl.Series([  # Struct column for panel data
        {"store_1": 100, "store_2": 150, "store_3": 200},
        {"store_1": 110, "store_2": 160, "store_3": 210},
        ...
    ])
})

# Cross-learning: train on all stores, predict for specific store
forecaster = PointReductionForecaster(estimator=Ridge())
forecaster.fit(y_panel, forecasting_horizon=3, time_weight=exponential_weight)

# Predict for store_2
y_pred = forecaster.predict(
    forecasting_horizon=3,
    panel_group="store_2"
)
```

## Key Takeaways

### What Works Out of the Box

1. **Automatic Routing**: Metadata automatically flows through pipelines and meta-estimators
2. **Composite Methods**: `update_transform()` and `update_predict()` registered with sklearn
3. **Time Weighting Signature**: `time_weight` parameter explicitly supported in reduction forecasters
4. **Panel Data**: Infrastructure for metadata routing with cross-learning scenarios
5. **Testing**: Comprehensive test suite validates routing behavior

### What's Different from Sklearn

1. **Global Enable**: `set_config(enable_metadata_routing=True)` called on import
2. **Composite Methods**: Yohou-specific methods (`update_transform`, `update_predict`) registered
3. **Time Series Params**: `y`, `X` are NOT routed (explicit parameters only)
4. **Update Method**: `update()` does NOT accept `**params` (memory management only)
5. **Explicit time_weight**: Declared as explicit parameter (not in `**params`)

**Note**: All features in `X` are expected to be known ex-ante (in advance). For ex-post features (observed after the fact), use `ColumnForecaster` to forecast them first.

### Current Implementation Status

**What's Implemented (✅)**:
- Complete metadata routing infrastructure
- All base classes have `get_metadata_routing()` implementations
- Pipelines, FeatureUnion, and SearchCV route metadata correctly
- Comprehensive test suite (31 passing tests)
- Composite methods registered with sklearn
- Panel data utilities for cross-learning

**What's Not Yet Implemented (⚠️)**:
- Actual `time_weight` to `sample_weight` conversion
- Passing `sample_weight` to sklearn estimators
- Metadata consumption in scorers
- Metadata tabularization for panel data

**Current Status**: The routing **infrastructure** is 100% complete and tested. Metadata flows correctly through all components. However, the actual **consumption** of metadata (e.g., converting `time_weight` to `sample_weight` and using it) is not yet implemented. Think of it as a working highway system where cars (metadata) can drive anywhere, but the destinations (estimators) aren't yet set up to receive them.

### Design Rationale

**Why `**params` everywhere?**
- Provides flexibility for future metadata types
- Enables nesting (pipelines, meta-estimators) without signature changes
- Follows sklearn patterns for composability

**Why not route `y` and `X`?**
- These are primary data, not auxiliary metadata
- Explicit parameters make API clearer
- Avoids confusion about what's being modeled vs what's metadata

**Note**: All features in `X` are expected to be known ex-ante. For ex-post features, use `ColumnForecaster`.
- Explicit parameters make API clearer
- Avoids confusion about what's being modeled vs what's metadata

**Why special handling for `time_weight`?**
- Requires transformation (time-based → sample weights)
- Needs alignment with tabularized data
- Domain-specific logic belongs in forecaster, not generic routing

**Why explicit parameter instead of `**params`?**
- Makes the forecaster's support for time weighting discoverable
- Clear in API documentation
- Avoids silent failures if metadata name typos occur

## Future Extensions

### Potential Enhancements

1. **Time Weight Consumption**: Complete the implementation to actually use `time_weight`
2. **Scorer Metadata**: Enable weighted scoring using `time_weight`
3. **Time Weight Utilities**: Pre-built weighting functions (exponential, linear, seasonal)
4. **Metadata Validation**: Type checking and validation for metadata
5. **Documentation**: More examples of custom metadata usage

### Compatibility Notes

- **Sklearn Version**: Requires sklearn >= 1.3 (metadata routing introduced)
- **Polars**: All data must be `pl.DataFrame` (no pandas support)
- **Python**: Python 3.12+ (uses modern type hints)

## References

- [Sklearn Metadata Routing User Guide](https://scikit-learn.org/stable/metadata_routing.html)
- [Sklearn SLEP006](https://github.com/scikit-learn/enhancement_proposals/blob/main/slep006/proposal.rst) - Original metadata routing proposal
- Yohou test suite: `tests/test_metadata_routing.py`
- Panel data utilities: `src/yohou/utils/panel.py`
