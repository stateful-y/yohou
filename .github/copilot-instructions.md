# GitHub Copilot Instructions for Yohou

## Project Overview
Yohou is a scikit-learn-compatible time series forecasting framework built on **polars** for data manipulation. It extends sklearn's API with time series-specific operations (`update`, `reset`) and supports both point and interval (native or conformal) forecasting with panel data capabilities.

**Philosophy**: Bridge sklearn's tabular ML ecosystem with time series forecasting by treating forecasting as a supervised learning reduction problem while maintaining temporal structure.

## Architecture & Core Concepts

### Data Flow
All data uses **polars DataFrames** with a mandatory `"time"` column (datetime type). Three feature types:
- `y`: Target time series (what to forecast)
- `X_post`: Ex-ante features (known in advance, e.g., holidays, planned promotions)
- `X_ante`: Ex-post features (observed after the fact, e.g., actual weather, traffic)

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
   - Stores `_y_observed`, `_X_post_observed` for recursive prediction
   - Signature: `fit(y, X_post, X_ante, forecasting_horizon)` - note horizon at fit time

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
- `fit(y, X_post, X_ante, forecasting_horizon)`: Train on historical data
  - Forecasters: Horizon is required at fit time (unlike sklearn's predict-time horizon)
  - Transformers: `fit(X, y)` follows sklearn convention (`y` optional)
- `update(y, X_post, X_ante)`: Add new observations **without full retrain** (incremental learning)
  - Updates internal memory buffers (`_X_observed`, `_y_observed`, etc.)
  - Does NOT refit models - use for streaming/online scenarios
- `predict(forecasting_horizon, X_post, X_ante)`: Generate forecasts
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
search.fit(y, X_post, X_ante, forecasting_horizon=3)
y_pred = search.predict(forecasting_horizon=3, X_post=X_post_future)
```

**Critical notes**:
- Param names follow sklearn convention: `step__param` for pipelines
- Always returns `best_forecaster_`, `best_params_`, `cv_results_`
- Integrates with sklearn's metadata routing for CV splits

## Developer Workflow

### Environment & Dependencies
- **Package manager**: `uv` (fast Python package installer/resolver)
- **Dependency groups** in `pyproject.toml`: `dev`, `docs`, `tests`, `fix`, `examples`
- Install dev environment: `uv sync` (syncs all groups)
- **Examples framework**: `marimo` (reactive Python notebooks in `examples/`)
  - NOT traditional Jupyter notebooks - marimo is a reactive notebook system
  - Run examples: `marimo edit examples/air_passengers_tutorial.py`

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
    def fit(self, y, X_post, X_ante, forecasting_horizon):
        # Pre-fit handles transformers and sets up observation buffers
        y_t, X_t = BasePointForecaster._pre_fit(
            self, y=y, X_post=X_post, X_ante=X_ante,
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
- Run tests: `uvx nox -s test` (includes coverage, doctests, and unit tests)
  - Alternative: `uv run pytest` (quicker for local testing, no coverage report)
  - Note: `uvx nox` uses uv backend for virtual environment management
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
- `yohou.utils.validation.check_inputs(y, X_post, X_ante)`: Validate all inputs have matching intervals


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
   - Matrix: Python 3.12 across Windows/macOS/Linux
   - Uses `uv` for fast dependency installation
   - Runs `uvx nox -s test` for full test suite
   - Concurrent execution with auto-cancellation on new pushes

2. **`lint.yml`**: Code quality checks
   - Runs `uvx nox -s fix` (ruff + ty + interrogate)
   - Validates docstring coverage at 100%

3. **`release.yml`**: PyPI package publishing
   - Automated on version tags (e.g., `v0.1.0`)
   - Builds with hatchling + hatch-vcs
   - Publishes to PyPI with trusted publishing

### CI Debugging Tips
- **Local CI simulation**: Use `uvx nox` to run same commands as CI
  ```bash
  uvx nox -s test  # Same as CI test step
  uvx nox -s fix   # Same as CI lint step
  ```
- **Coverage failures**: Check `coverage.xml` or HTML report in `htmlcov/`
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

**3. Panel Data Struct Access**
```python
# Problem: Can't access individual series in panel data
y_panel = pl.DataFrame({
    "time": [...],
    "sales": [{"store_1": 100, "store_2": 150}, ...]  # Struct column
})

# Solution: Unnest struct to separate columns
y_unnested = y_panel.unnest("sales")
# Now has columns: time, store_1, store_2

# Or inspect locality
from yohou.utils.polars import inspect_locality
global_cols, local_groups = inspect_locality(y_panel)
print(local_groups)  # {'sales': ['store_1', 'store_2']}
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

## Performance & Optimization

### Panel Data Performance
**Polars structs are optimized for panel data** - avoid unnesting unless necessary:
```python
# ❌ Slow: Unnest then process
y_panel = df.unnest("sales")  # Expands to many columns
result = y_panel.select([...])  # Processes all columns

# ✅ Fast: Operate on struct directly
result = df.select([
    pl.col("time"),
    pl.col("sales").struct.field("store_1"),  # Access single field
])

# ✅ Faster: Batch operations on structs
result = df.select([
    pl.col("time"),
    pl.col("sales").struct.rename_fields(["s1", "s2"]),  # Rename all fields at once
])
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
SearchCV and pipelines support parallel execution via `n_jobs`:
```python
from yohou.model_selection import SearchCV
from yohou.pipeline import FeatureUnion

# Parallel hyperparameter search (across CV folds)
search = SearchCV(forecaster=model, n_jobs=-1, ...)  # Use all cores

# Parallel feature engineering
features = FeatureUnion([
    ('lags', LagTransformer([1, 2, 3])),
    ('rolling', RollingMeanTransformer(window=12)),
], n_jobs=2)  # Compute both transformers in parallel
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
- ✅ Use polars structs for panel data (avoid unnecessary unnesting)
- ✅ Keep `observation_horizon` as small as possible
- ✅ Use lazy polars API for large files (`scan_csv`, `scan_parquet`)
- ✅ Enable `n_jobs` for SearchCV and FeatureUnion (if dataset is large)
- ✅ Profile before optimizing - measure actual bottlenecks
- ✅ Consider data types: use `pl.Int32` vs `pl.Int64` when appropriate
- ❌ Don't unnest panel data unless you need all series
- ❌ Don't parallelize small datasets (overhead > benefit)
- ❌ Don't load entire dataset into memory if streaming is possible
