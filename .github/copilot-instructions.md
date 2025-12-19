# GitHub Copilot Instructions for Yohou

## Project Overview
Yohou is a time series forecasting package built on **scikit-learn** and **polars**. It provides a scikit-learn compatible API for forecasting, supporting both point and interval (conformal) prediction.

## Architecture & Core Concepts

### Data Structures
- **Polars DataFrames**: All data manipulation uses `polars`. Avoid `pandas` unless strictly necessary for interoperability.
- **Time Column**: The time index is expected to be a column named `"time"`.
- **Feature Types**:
  - `y`: Target time series.
  - `X_ante`: Ex-ante features (known in advance, e.g., holidays).
  - `X_post`: Ex-post features (known only after the fact, e.g., weather measurements).

### Class Hierarchy
- **Base Classes** (`src/yohou/base.py`):
  - `BaseTransformer`: Extends `sklearn.base.TransformerMixin`. Must implement `fit`, `transform`, `update`, and `reset`.
  - `BaseForecaster`: Base for all forecasters. Handles `target_transformer` and `feature_transformer`.
- **Forecasting**:
  - `BasePointForecaster` (`src/yohou/point_forecaster/base.py`): For point predictions.
  - `BaseIntervalForecaster` (`src/yohou/interval_forecaster/base.py`): For conformal prediction intervals.
- **Pipeline** (`src/yohou/pipeline.py`): Custom `Pipeline` implementation that supports the `update` method for time series.

### Key Methods
- `fit(y, X_ante=None, X_post=None, forecasting_horizon=1)`: Train the model.
- `update(y, X_ante=None, X_post=None)`: Update the model with new observations without full retraining.
- `predict(forecasting_horizon, X_ante=None, X_post=None)`: Generate forecasts.
- `reset(X)`: Reset the transformer/forecaster memory (observation horizon).

## Developer Workflow

### Dependency Management
- Uses **uv** for package management.
- `pyproject.toml` defines dependencies and groups (`dev`, `docs`, `tests`, `fix`).

### Build & Test (Nox)
Use `nox` for all development tasks.
- **Run Tests**: `uvx nox -s tests_coverage` (runs pytest with coverage).
- **Lint & Format**: `uvx nox -s fix` (runs pre-commit hooks: ruff, black).
- **Build Docs**: `uvx nox -s docs` (builds MkDocs).
- **Serve Docs**: `uvx nox -s serve_docs`.

### Coding Conventions
- **Type Hints**: Strictly use type hints. Use `pydantic` types where appropriate (e.g., `StrictInt`).
  ```python
  def my_func(x: pl.DataFrame, horizon: StrictInt) -> pl.DataFrame: ...
  ```
- **Docstrings**: Google style docstrings.
- **Polars Selectors**: Use `polars.selectors` (imported as `cs`) for column selection.
  ```python
  df.select(~cs.by_name("time"))
  ```
- **Locality**: Be aware of `inspect_locality` for handling local vs. global time series data.

## Common Patterns

### Implementing a Forecaster
Inherit from `BasePointForecaster` or `BaseIntervalForecaster`. Implement `_predict_one` or `fit`/`predict` logic.
```python
class MyForecaster(BasePointForecaster):
    def _predict_one(self) -> pl.DataFrame:
        # Implementation using self._y_observed
        ...
```

### Transformers
Transformers must maintain state via `memory_size` and handle `reset`/`update`.
```python
class MyTransformer(BaseTransformer):
    @property
    def memory_size(self):
        return 10  # Example

    def fit(self, X, y=None):
        self.reset(X)
        # ... fitting logic
        return self
```
