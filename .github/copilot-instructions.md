# GitHub Copilot Instructions for Yohou

## Project Overview
Yohou is a scikit-learn-compatible time series forecasting framework built on **polars** for data manipulation. It extends sklearn's API with time series-specific operations (`update`, `reset`) and supports both point and interval (native or conformal) forecasting with panel data capabilities.

## Architecture & Core Concepts

### Data Flow
All data uses **polars DataFrames** with a mandatory `"time"` column. Three feature types:
- `y`: Target time series
- `X_ante`: Ex-ante features (known in advance, e.g., holidays, planned promotions)
- `X_post`: Ex-post features (observed after the fact, e.g., actual weather, traffic)

### Class Hierarchy (`src/yohou/base.py`)
1. **BaseTransformer** (extends `sklearn.base.TransformerMixin`)
   - Must implement: `fit`, `transform`, `update`, `reset`, `get_feature_names_out`
   - Maintains stateful `_X_observed` for windowing operations
   - `observation_horizon` property defines observation horizon length

2. **BaseForecaster** (base for all forecasters)
   - Handles `target_transformer` and `feature_transformer` composition
   - `_set_local_groups()` enables panel data (local vs. global time series)
   - Stores `_y_observed`, `_X_ante_observed`, `_X_post_observed` for recursive prediction

3. **BaseReductionForecaster** (forecasting via sklearn regressors)
   - Tabularizes time series using `tabularize()` to create lag features
   - `_estimator_fit_one()`: Converts to supervised learning problem
   - `_estimator_predict_one()`: Generates predictions from tabular features
   - Supports panel data via struct columns (see "Locality" below)

4. **Point vs Interval Forecasters**
   - `BasePointForecaster` → `PointReductionForecaster` (standard forecasting)
   - `BaseIntervalForecaster` → `SplitConformalForecaster` (conformal prediction intervals)

### Time Series-Specific Methods
- `fit(y, X_ante, X_post, forecasting_horizon)`: Train on historical data
- `update(y, X_ante, X_post)`: Add new observations without full retrain (incremental learning)
- `predict(forecasting_horizon, X_ante, X_post)`: Generate forecasts
- `update_predict()`: Combined update + predict (common in rolling forecast evaluation)
- `reset(X)`: Reset memory/observation horizon to last `observation_horizon` rows

### Pipeline (`src/yohou/pipeline.py`)
Custom sklearn Pipeline supporting `update` method for time series:
```python
Pipeline([
    ('lag', LagTransformer(lag=[1, 2, 3])),
    ('forecaster', PointReductionForecaster())
])
```

### Panel Data & Locality (`src/yohou/utils/polars.py`)
**Critical concept**: `inspect_locality()` distinguishes global vs. local (panel) time series.
- **Global**: Columns apply to all time series (e.g., single univariate series)
- **Local**: Struct columns containing different series (e.g., sales per store)
  ```python
  # Panel data example: pl.Struct with field per store
  y = pl.DataFrame({"time": [...], "sales": pl.Struct({"store_1": [...], "store_2": [...]})})
  ```
- Forecasters automatically handle both via `local_group_names_` and `local_y_names_`

### Reduction Strategies (README mentions but not yet in code)
Framework designed for Recursive, Direct, Multi-output, DirRec strategies via `BaseReductionForecaster`.

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
- `fix`: Pre-commit hooks (ruff linter/formatter + mypy type checking)
- `docs`: Build MkDocs documentation
- `serve_docs`: Local docs server on `localhost:8080`
- `deploy_docs`: Deploy to GitHub Pages

### Code Quality Tools
- **Linter/Formatter**: Ruff (100 char line length, target py3.12)
- **Type Checker**: mypy with `--strict` mode (enforced via pre-commit)
- **Pre-commit hooks**: Defined in `.pre-commit-config.yaml`
  - Run manually: `uvx nox -s fix`
  - Auto-runs on git commit

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

## Testing Patterns
- Tests in `tests/` mirror `src/yohou/` structure
- Use sample data with `pl.datetime_range` for time columns:
  ```python
  time = pl.DataFrame({
      "time": pl.datetime_range(start=datetime(2021, 12, 16),
                                 end=datetime(2021, 12, 16, 0, 0, 21),
                                 interval="1s", eager=True)
  })
  ```
- Test both global and local (panel) data scenarios

## Key Utilities
- `yohou.utils.tabularization.tabularize(df, lags)`: Create lagged features
- `yohou.utils.polars.inspect_locality(df)`: Parse global/local columns
- `yohou.utils.polars.concat_struct(items, how)`: Merge panel data struct columns
- `yohou.utils.validation.check_interval_consistency(df)`: Validate time spacing
