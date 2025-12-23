# GitHub Copilot Instructions for Yohou

## Project Overview
Yohou is a scikit-learn-compatible time series forecasting framework built on **polars** for data manipulation. It extends sklearn's API with time series-specific operations (`update`, `reset`) and supports both point and interval (native or conformal) forecasting with panel data capabilities.

**Philosophy**: Bridge sklearn's tabular ML ecosystem with time series forecasting by treating forecasting as a supervised learning reduction problem while maintaining temporal structure.

## Architecture & Core Concepts

### Data Flow
All data uses **polars DataFrames** with a mandatory `"time"` column (datetime type). Three feature types:
- `y`: Target time series (what to forecast)
- `X_ante`: Ex-ante features (known in advance, e.g., holidays, planned promotions)
- `X_post`: Ex-post features (observed after the fact, e.g., actual weather, traffic)

**Critical**: Time columns are preserved differently across components:
- Transformers: Input/output have "time" column
- Forecasters: Predictions add "observed_time" and "predicted_time" columns for alignment

### Class Hierarchy (`src/yohou/base.py`)
1. **BaseTransformer** (extends `sklearn.base.TransformerMixin`)
   - Must implement: `fit`, `transform`, `update`, `reset`, `get_feature_names_out`
   - Maintains stateful `_X_observed` for windowing operations (last `observation_horizon` rows)
   - `observation_horizon` property: Raises `NotFittedError` before fit for stateful transformers
   - `fit()` auto-sets `feature_names_in_`, `n_features_in_`, `_X_observed`
   - `update()` extends memory, `reset()` replaces it (both take last `observation_horizon` rows)

2. **BaseForecaster** (base for all forecasters)
   - Handles `target_transformer` and `feature_transformer` composition
   - `_set_local_groups()` enables panel data (local vs. global time series)
   - Stores `_y_observed`, `_X_ante_observed` for recursive prediction
   - Signature: `fit(y, X_ante, X_post, forecasting_horizon)` - note horizon at fit time

3. **BaseReductionForecaster** (forecasting via sklearn regressors)
   - Converts time series to supervised learning via `tabularize()` (creates lag features)
   - `_estimator_fit_one()`: Single-horizon supervised learning fit
   - `_estimator_predict_one()`: Generates one-step predictions (recursive for multi-step)
   - Supports panel data via struct columns (see "Locality" below)
   - Must provide `estimator` param (any sklearn regressor)

4. **Point vs Interval Forecasters**
   - `BasePointForecaster` → `PointReductionForecaster` (standard forecasting)
   - `BaseIntervalForecaster` → `SplitConformalForecaster` (conformal prediction intervals)
   - Both extend `BaseForecaster` but handle different `prediction_types`

### Time Series-Specific Methods
**Standard sklearn lifecycle extended:**
- `fit(y, X_ante, X_post, forecasting_horizon)`: Train on historical data
  - Forecasters: Horizon is required at fit time (unlike sklearn's predict-time horizon)
  - Transformers: `fit(X, y)` follows sklearn convention (`y` optional)
- `update(y, X_ante, X_post)`: Add new observations **without full retrain** (incremental learning)
  - Updates internal memory buffers (`_X_observed`, `_y_observed`, etc.)
  - Does NOT refit models - use for streaming/online scenarios
- `predict(forecasting_horizon, X_ante, X_post)`: Generate forecasts
  - Can predict different horizon than fit (applies model recursively)
- `update_predict()`: Combined update + predict (atomic operation, common in rolling evaluation)
- `reset(X)`: Reset memory/observation horizon to last `observation_horizon` rows
  - Used to "rewind" transformer state without refitting

**Memory management pattern**: `update()` appends then calls `reset()` to maintain fixed-size window.

### Pipeline (`src/yohou/pipeline.py`)
Custom sklearn Pipeline/FeatureUnion/ColumnTransformer supporting time series operations:
```python
from yohou.pipeline import Pipeline
Pipeline([
    ('lag', LagTransformer(lag=[1, 2, 3])),
    ('forecaster', PointReductionForecaster())
])
```
**Key differences from sklearn**:
- `observation_horizon` computed as sum (Pipeline) or max (FeatureUnion) of steps
- All components must implement `update()` and `reset()`
- Handles struct columns (panel data) in concat operations

### Panel Data & Locality (`src/yohou/utils/polars.py`)
**Critical concept**: `inspect_locality()` distinguishes global vs. local (panel) time series.
- **Global**: Columns apply to all time series (e.g., single univariate series, shared features)
- **Local**: Struct columns containing different series (e.g., sales per store)
  ```python
  # Panel data example: pl.Struct with field per store
  y = pl.DataFrame({
      "time": [...],
      "sales": pl.Series([
          {"store_1": 100, "store_2": 150},
          {"store_1": 110, "store_2": 160},
          ...
      ])
  })
  global_names, local_groups = inspect_locality(y)
  # Returns: ([], {"sales": ["store_1", "store_2"]})
  ```
- Forecasters automatically handle both via `local_group_names_` and `local_y_names_`
- Use `concat_struct()` for vertical/horizontal concatenation preserving struct columns
- Access panel data: `df.unnest("sales")` flattens struct to separate columns

**Why structs?** Polars structs enable efficient panel data storage while maintaining type consistency and allowing vectorized operations across groups.

### Reduction Strategies (README mentions but not yet in code)
Framework designed for Recursive, Direct, Multi-output, DirRec strategies via `BaseReductionForecaster`.

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

**Key components**:
- `Sampler`: Wrapper for Optuna samplers (default: `TPESampler`)
- `Storage`: Optional persistent storage for optimization history (e.g., `RDBStorage`)
- `n_warmup_trials`: Random search trials before sampler kicks in
- `n_trials`: Number of Optuna optimization trials

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
search.fit(y, X_ante, X_post, forecasting_horizon=3)
y_pred = search.predict(forecasting_horizon=3, X_ante=X_ante_future)
```

**Critical notes**:
- Param names follow sklearn convention: `step__param` for pipelines
- Always returns `best_forecaster_`, `best_params_`, `cv_results_`
- Integrates with sklearn's metadata routing for CV splits

## Developer Workflow

### Environment & Dependencies
- **Package manager**: `uv` (fast Python package installer)
- **Dependency groups** in `pyproject.toml`: `dev`, `docs`, `tests`, `fix`
- Install dev environment: `uv sync` (syncs all groups)

### Nox Sessions (All commands use `uvx nox -s <session>`)
Critical: Always use `uvx nox` (not plain `nox`) to leverage uv backend:
- `test`: Pytest with coverage (default session)
- `fix`: Pre-commit hooks (ruff linter/formatter + ty type checking)
- `docs`: Build MkDocs documentation
- `serve_docs`: Local docs server on `localhost:8080`
- `deploy_docs`: Deploy to GitHub Pages

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
  - Run manually: `uvx nox -s fix`
  - Auto-runs on git commit (includes ty, ruff, interrogate)

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
    def fit(self, y, X_ante, X_post, forecasting_horizon):
        # Pre-fit handles transformers and sets up observation buffers
        y_t, X_t = BasePointForecaster._pre_fit(
            self, y=y, X_ante=X_ante, X_post=X_post,
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

### Docstrings (Google Style)
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
**Critical**: Docstrings use **NumPy style** (not Google), enforced by `interrogate` at 100% coverage.
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
- Run tests: `uvx nox -s test` (includes coverage, doctests, and unit tests)
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
- `yohou.utils.validation.check_inputs(y, X_ante, X_post)`: Validate all inputs have matching intervals


## Plans & Documentation
- **Testing Transformer Infrastructure**: See `.github/copilot_plans/transformer-testing-infrastructure.md` for comprehensive transformer testing plan using sklearn patterns and pytest fixtures
- When asked to create a new plan, save it in `.github/copilot_plans/`
