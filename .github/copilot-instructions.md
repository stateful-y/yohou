# GitHub Copilot Instructions for Yohou

## Project Overview

Yohou is a scikit-learn-compatible time series forecasting framework built on **polars** for high-performance data manipulation. It extends sklearn's API with time series-specific operations (`update`, `reset`, `update_predict`) and supports both point and interval forecasting with native panel data capabilities.

**Philosophy**: Bridge sklearn's tabular ML ecosystem with time series forecasting by treating forecasting as a supervised learning reduction problem while maintaining temporal structure.

**Python Support**: 3.12, 3.13, 3.14 (per pyproject.toml)

**Note**: While README mentions multi-DataFrame support via Narwhals, this is planned but not yet implemented (see TODO.md).

**Key Dependencies**:
- **polars** (≥0.20): Primary DataFrame library (via `import polars as pl`)
- **scikit-learn** (≥1.6.0): ML estimator integration with metadata routing
- **marimo** (≥0.19.0): Reactive notebook system for examples
- **pydantic** (≥2.6): Type validation (use `StrictInt`, `StrictFloat` for strict typing)
- **plotly** (≥5.19): Interactive visualizations

**Quick Links to Detailed Guides**:

| Guide | Purpose | Size |
|-------|---------|------|
| [Architecture & Core Concepts](.github/copilot/architecture-and-core-concepts.md) | Class hierarchy, data flow, panel data, metadata routing | 560 lines |
| [Creating New Forecasters](.github/copilot/creating-new-forecasters.md) | Step-by-step guide with real examples | 596 lines |
| [Developer Workflow & Tools](.github/copilot/developer-workflow-and-tools.md) | Commands, testing, debugging, CI/CD | 602 lines |
| [Testing Infrastructure Overview](.github/copilot/testing-infrastructure-overview.md) | Complete testing system (86 checks, 5 generators, 6 utilities) | 580 lines |
| [Forecaster Testing Infrastructure](.github/copilot/forecaster-testing-infrastructure.md) | Comprehensive testing guide with check functions | 717 lines |
| [Transformer Testing Infrastructure](.github/copilot/transformer-testing-infrastructure.md) | Testing patterns for transformers | 400 lines |
| [Splitter Testing Infrastructure](.github/copilot/splitter-testing-infrastructure.md) | Cross-validation splitter testing patterns | 733 lines |
| [Scorer Testing Infrastructure](.github/copilot/scorer-testing-infrastructure.md) | Metrics/scorer testing patterns | 803 lines |
| [Search Testing Infrastructure](.github/copilot/search-testing-infrastructure.md) | GridSearchCV and RandomizedSearchCV testing | 900 lines |
| [sklearn Metadata Routing Implementation](.github/copilot/sklearn-metadata-routing-implementation.md) | Complete metadata routing infrastructure | 814 lines |
| [Monthly Interval Support](.github/copilot/monthly-interval-support.md) | Variable-length time intervals (monthly, quarterly, yearly) | 715 lines |
| [Time Weighting Implementation](.github/copilot/time-weighting-implementation.md) | Time-based weighting for scorers and forecasters (planned) | 893 lines |

---

## Critical Concepts (Quick Reference)

### Bootstrap Behavior

**Critical**: Yohou automatically enables sklearn's metadata routing on import:
```python
# This happens in src/yohou/__init__.py
from sklearn import set_config
set_config(enable_metadata_routing=True)  # Global state change

# Registers custom composite methods for metadata routing
SIMPLE_METHODS.extend(["update_transform", "update_predict", "predict_interval", "update_predict_interval"])
COMPOSITE_METHODS["update_transform"] = ["update", "transform"]
COMPOSITE_METHODS["update_predict"] = ["update", "predict"]
COMPOSITE_METHODS["update_predict_interval"] = ["update", "predict_interval"]
```

### Data Flow

All data uses **polars DataFrames** with mandatory `"time"` column (datetime type):
- `y`: Target time series (what to forecast)
- `X`: Exogenous features (known ex-ante - in advance)

**Time Column Preservation**:
- Transformers: Input/output have `"time"` column
- Forecasters: Predictions have `"observed_time"` and `"time"` columns

**📖 Full Details**: See [Architecture & Core Concepts](.github/copilot/architecture-and-core-concepts.md)

### Class Hierarchy (Simplified)

**Core Base Classes** (`src/yohou/base.py`):
1. **BaseTransformer**: Stateful windowing with `update()`, `reset()`, `observation_horizon`
2. **BaseForecaster**: Handles transformers, observation buffers, panel data
3. **BaseReductionForecaster**: Forecasting via sklearn regressors (tabularize → fit → recursive predict)

**Forecaster Types**:
- **BasePointForecaster**: `prediction_types = {"point"}`
- **BaseIntervalForecaster**: `prediction_types = {"interval"}` or both

**Meta-Forecasters** (`src/yohou/forecaster/composition.py`, `src/yohou/decomposition/`):
- **Decomposer**: Sequential decomposition (trend + season + residual)
- **ColumnForecaster**: Different forecasters per column (parallel execution)

**📖 Full Details**: See [Architecture & Core Concepts](.github/copilot/architecture-and-core-concepts.md)

### Time Series Methods

Standard sklearn lifecycle extended:
- `fit(y, X, forecasting_horizon)`: Train on historical data
  - Forecasters: Horizon is required at fit time (unlike sklearn's predict-time horizon)
  - Transformers: `fit(X, y)` follows sklearn convention (`y` optional)
- `update(y, X, panel_group_names)`: Add new observations **without full retrain** (incremental learning)
  - Updates internal memory buffers (`_X_observed`, `_y_observed`, etc.)
  - Does NOT refit models - use for streaming/online scenarios
  - `panel_group_names`: Optional list of group prefixes to update (for panel data)
- `predict(forecasting_horizon, X, panel_group_names)`: Generate point forecasts
  - Can predict different horizon than fit (applies model recursively)
  - `panel_group_names`: Optional list of group prefixes to predict (for panel data)
- `predict_interval(forecasting_horizon, X, coverage_rates, panel_group_names)`: Generate interval forecasts
  - Available on interval forecasters (BaseIntervalForecaster subclasses)
  - `coverage_rates`: List of coverage levels (e.g., [0.9, 0.95] for 90% and 95% intervals)
- `update_predict(y, X, panel_group_names)`: Combined update + predict (atomic operation, common in rolling evaluation)
- `update_predict_interval(y, X, coverage_rates, panel_group_names)`: Combined update + predict_interval for intervals
- `reset(y, X, panel_group_names)`: Reset memory/observation horizon to last `observation_horizon` rows
  - Used to "rewind" forecaster state without refitting
  - `panel_group_names`: Optional list of group prefixes to reset (for panel data)

**Memory management pattern**: `update()` appends then calls `reset()` to maintain fixed-size window.

### Panel Data & Locality

**Critical concept**: Prefixed column names distinguish global vs. local (panel) time series.

```python
# Panel data example
y = pl.DataFrame({
    "time": [...],
    "sales__store_1": [100, 110, ...],  # Prefix: sales, Suffix: store_1
    "sales__store_2": [150, 160, ...],
})
```

**Key Utilities**:
- `inspect_locality(df)`: Returns `(global_columns, panel_groups_dict)`
- `get_group_df(df, group_name, schema)`: Extracts group with unprefixed columns

**📖 Full Details**: See [Architecture & Core Concepts](.github/copilot/architecture-and-core-concepts.md) - Panel Data section

### Metrics & Scoring

All metrics extend `BaseScorer` with `prediction_types` property and `score(y_truth, y_pred)` method.

**📖 Full Details**: See [Architecture & Core Concepts](.github/copilot/architecture-and-core-concepts.md) - Metrics section

### Hyperparameter Search

**GridSearchCV and RandomizedSearchCV**: sklearn-compatible hyperparameter search for time series.

```python
from yohou.model_selection import RandomizedSearchCV
from scipy.stats import uniform

search = RandomizedSearchCV(
    forecaster=PointReductionForecaster(),
    param_distributions={"estimator__alpha": uniform(0.01, 1.0)},
    scoring=MeanAbsoluteError(),
    n_iter=20
)
search.fit(y, X, forecasting_horizon=3)
y_pred = search.predict(forecasting_horizon=3)
```

**📖 Full Details**: See [Architecture & Core Concepts](.github/copilot/architecture-and-core-concepts.md) - Hyperparameter Search section

### Metadata Routing

**Critical**: sklearn metadata routing enabled automatically on import. All methods accept `**params`.

**Planned Feature**: `time_weight` parameter is declared in method signatures but **conversion to `sample_weight` not yet implemented** (see TODO.md, `.github/copilot/time-weighting-implementation.md`).

**📖 Full Details**: [sklearn Metadata Routing Implementation](.github/copilot/sklearn-metadata-routing-implementation.md)

---

## Developer Workflow

**📖 Complete Guide**: [Developer Workflow & Tools](.github/copilot/developer-workflow-and-tools.md)

### Quick Commands

**Testing**:
```bash
uv run pytest                          # Run all tests (fast, no coverage)
uv run pytest tests/path/test.py -v   # Run specific test file
uv run pytest --pdb tests/...          # Debug with pdb
uvx nox -s test                        # Full test suite with coverage
```

**Note**: Test suite may have failures during active development. Check terminal history for latest test results.

**Code Quality**:
```bash
uvx ruff check --fix .                 # Lint and auto-fix
uvx ruff format .                      # Format code
uvx ty check src                       # Type check (NOT mypy!)
uvx interrogate src/yohou              # Docstring coverage (100% required)
uvx nox -s fix                         # Run all quality checks (pre-commit)
```

**Documentation**:
```bash
uvx nox -s docs                        # Build docs
uvx nox -s serve_docs                  # Live docs server (localhost:8080)
```

### Environment Setup

```bash
uv sync                                # Install all dependencies
uv sync --group tests                  # Install specific group
```

**Critical**: Always use `uvx nox` (not plain `nox`) - automatic tool management via uv.

### Common Debugging Commands

```bash
# Debug failing tests interactively
uv run pytest tests/path/test.py::test_name --pdb

# Run tests with verbose output
uv run pytest tests/path/test.py -vv -s

# Show fixtures available for a test
uv run pytest tests/path/test.py --fixtures

# Run only tests that failed last time
uv run pytest --lf

# Run tests matching a pattern
uv run pytest -k "test_pattern"
```

### Marimo Notebooks (Interactive Examples)

**Critical**: Examples in `examples/*.py` are **Marimo reactive notebooks**, NOT regular Python scripts.

**Marimo notebooks**:
- Stored as `.py` files (e.g., `air_passengers_tutorial.py`, `decomposition_tutorial.py`, `panel_data_tutorial.py`)
- Contain special `marimo.App()` structure with `@app.cell` decorators
- Reactive execution: cells automatically re-run when dependencies change
- Each cell has unique ID (e.g., `#VSC-a7628b23`) and language tag (`mo-python`, `markdown`)

**Running notebooks**:
```bash
# Launch interactive notebook in browser (primary workflow)
uv run marimo edit examples/air_passengers_tutorial.py

# Run as script (non-interactive, executes all cells)
uv run marimo run examples/air_passengers_tutorial.py

# Run as regular Python file (for debugging/testing)
uv run python examples/air_passengers_tutorial.py
```

**Debugging patterns**:

1. **Interactive debugging in browser** (preferred for reactive issues):
   ```bash
   # Launch with debug mode
   uv run marimo edit examples/air_passengers_tutorial.py

   # In notebook: Add print statements or mo.output.append() to cells
   # Use mo.debug() to inspect reactive dependencies
   # Check cell execution order in UI (numbers show dependency graph)
   ```

2. **Python debugger** (for logic errors):
   ```bash
   # Run with pdb
   uv run python -m pdb examples/air_passengers_tutorial.py

   # Or add breakpoint() inside @app.cell decorator:
   @app.cell
   def __():
       forecaster = PointReductionForecaster()
       breakpoint()  # Execution will pause here
       return forecaster,
   ```

3. **Isolate failing cells**:
   ```python
   # Copy failing cell code to a separate test.py file
   # Run without reactive framework:
   uv run python test.py

   # Or create minimal reproduction in pytest:
   def test_marimo_issue(y_X_factory):
       y, X = y_X_factory(length=100)
       # Paste cell logic here
   ```

4. **Check reactive dependencies**:
   ```bash
   # In marimo edit UI:
   # - Hover over variable names to see which cells define/use them
   # - Click cell IDs to navigate dependency graph
   # - Use "Show graph" button to visualize execution flow

   # Common issue: Circular dependencies (marimo will error)
   # Fix: Refactor to one-way data flow
   ```

5. **Session state debugging**:
   ```bash
   # Session files stored in examples/__marimo__/session/
   # Delete if notebook behaves unexpectedly:
   rm -rf examples/__marimo__/session/air_passengers_tutorial.py.json

   # Then relaunch:
   uv run marimo edit examples/air_passengers_tutorial.py
   ```

6. **Polars/Yohou specific debugging**:
   ```python
   # Add these debug cells to inspect data:
   @app.cell
   def __(y):
       # Check DataFrame structure
       mo.output.append(y.head())
       mo.output.append(y.schema)
       return

   @app.cell
   def __(forecaster):
       # Inspect fitted forecaster state
       from sklearn.utils.validation import check_is_fitted
       check_is_fitted(forecaster)
       mo.output.append(f"Observation horizon: {forecaster._observation_horizon}")
       return
   ```

7. **Common pitfalls**:
   - **Forgetting return statement**: Cells must `return variable,` to expose to other cells
   - **Name collisions**: Variables from different cells with same name cause conflicts
   - **Non-deterministic code**: Random seeds should be set in cell that uses them
   - **Side effects**: Avoid mutating shared state (use pure functions)

**Key differences from Jupyter**:
- No hidden state: cell order doesn't matter, only dependencies
- Deterministic execution: same inputs → same outputs always
- Git-friendly: plain Python files, not JSON
- Session files in `__marimo__/session/` store runtime state (gitignored)
- Auto-save: Changes persist immediately
- No cell output stored in file: Only code is versioned

**When to use**:
- Interactive demonstrations and tutorials
- Exploratory data analysis with yohou
- Testing forecaster behavior with visualizations
- Rapid prototyping with immediate feedback

**When NOT to use**:
- Unit tests (use pytest in `tests/`)
- Production code (use regular Python modules)
- CI/CD scripts (notebooks are for interactive exploration)
- Performance benchmarks (reactive overhead affects timing)

---

## Creating New Forecasters

**📖 Complete Guide**: [Creating New Forecasters](.github/copilot/creating-new-forecasters.md)

### Quick Checklist

1. **Choose forecaster type**: Point/interval, pattern-based/ML-based
2. **Implement core structure**: Class with `_parameter_constraints`, `fit()`, `predict()`
3. **Add parameter constraints**: Use `Interval` for validation
4. **Handle panel data** (if applicable): Check `self.panel_group_names_`
5. **Write tests**: Use `tests/<module>/test_<name>.py`
6. **Add doctests**: NumPy-style docstrings with runnable examples
7. **Update exports**: Add to `__init__.py`
8. **Quality checks**: `uvx nox -s fix` before committing

### Minimal Example

```python
import numbers
from sklearn.base import _fit_context
from sklearn.utils._param_validation import Interval
from .base import BasePointForecaster

class MyForecaster(BasePointForecaster):
    """Docstring with NumPy style."""

    _parameter_constraints: dict = {
        **BasePointForecaster._parameter_constraints,
        "param1": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(self, param1: int, target_transformer=None):
        super().__init__(target_transformer=target_transformer)
        self.param1 = param1

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y, X=None, forecasting_horizon=1, **params):
        y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)
        # Your fitting logic
        return self

    def predict(self, forecasting_horizon=None, X=None, **params):
        # Your prediction logic
        return self._add_time_columns(y_pred)
```

---

## Testing Patterns

**📖 Complete Guides**:
- [Testing Infrastructure Overview](.github/copilot/testing-infrastructure-overview.md) - **Start here**: Complete system with 86 checks
- [Forecaster Testing Infrastructure](.github/copilot/forecaster-testing-infrastructure.md)
- [Transformer Testing Infrastructure](.github/copilot/transformer-testing-infrastructure.md)
- [Splitter Testing Infrastructure](.github/copilot/splitter-testing-infrastructure.md)
- [Scorer Testing Infrastructure](.github/copilot/scorer-testing-infrastructure.md)
- [Search Testing Infrastructure](.github/copilot/search-testing-infrastructure.md)

### Systematic Check Functions

```python
from yohou.testing import _yield_yohou_forecaster_checks

for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(
    forecaster_fitted, y_train, X_train, y_test, X_test,
    tags={"forecaster_type": "point", "uses_reduction": False}
)::
    if check_name not in expected_failures:
        check_func(forecaster_fitted, **check_kwargs)
```

### Test Fixtures

```python
def test_my_forecaster(y_X_factory):
    y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)
    forecaster = MyForecaster(param1=10)
    forecaster.fit(y[:80], X[:80], forecasting_horizon=5)
    y_pred = forecaster.predict(forecasting_horizon=5)
```

---

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

# Validate time consistency
from yohou.utils.validation import check_interval_consistency
interval = check_interval_consistency(df)  # Returns timedelta
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

**Critical**: All docstrings MUST use **NumPy style** (NOT Google style), enforced by `interrogate` at 100% coverage.
- Coverage requirements in `pyproject.toml`: `fail-under = 100`
- Excludes: tests, examples, `_version.py`, private/magic/init methods
- Ignores nested classes but NOT nested functions
- Run check: `uvx interrogate src/yohou` or via `uvx nox -s fix` (pre-commit hooks)

### Error Handling Patterns
```python
# Import validation utilities from sklearn
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

# Use _parameter_constraints for automatic validation
_parameter_constraints = {
    "alpha": [Interval(numbers.Real, 0, None, closed="neither")],
    "strategy": [StrOptions({"direct", "recursive"})],
}

# Check fitted state before predict/transform
check_is_fitted(self, ["_observation_horizon", "_X_observed"])
```

### Validation Patterns for Time Series Data

**Critical**: Handle single-row DataFrames (single-step predictions):
```python
# check_interval_consistency requires ≥2 rows
if len(df) >= 2:
    interval = check_interval_consistency(df)
else:
    interval = None  # Single-step prediction, skip interval validation

# check_continuity accepts None for expected_interval (skips validation)
check_continuity(df_p, df_n, expected_interval=interval, check_intervals=(interval is not None))
```

**Common Pattern**: For transformers with `observation_horizon`, validate data sufficiency before operations:
```python
if len(X) < self.observation_horizon:
    raise ValueError(
        f"Not enough data to reset: {len(X)} rows provided, "
        f"but observation_horizon={self.observation_horizon}. "
        f"Provide at least {self.observation_horizon} rows."
    )
```

---

## Key File Locations

**Source Code**:
- `src/yohou/base.py`: Core base classes (BaseTransformer, BaseForecaster, BaseReductionForecaster)
- `src/yohou/point_forecaster/`: Point forecasters (naive, reduction)
- `src/yohou/interval_forecaster/`: Interval forecasters (conformal, reduction)
- `src/yohou/decomposition/`: Decomposition components (trend, seasonality, decomposer)
- `src/yohou/forecaster/`: Composition forecasters (ColumnForecaster)
- `src/yohou/preprocessing/`: Transformers (stationarization, windowing)
- `src/yohou/pipeline.py`: FeaturePipeline, FeatureUnion, ColumnTransformer
- `src/yohou/metrics/`: Scorers (point, interval, conformity)
- `src/yohou/model_selection/`: GridSearchCV, RandomizedSearchCV, cross-validation utilities
- `src/yohou/analysis/`: Visualization tools (plotly-based)

**Tests**:
- `tests/conftest.py`: Global fixtures (y_X_factory, data generators)
- `src/yohou/testing/`: Reusable check functions for systematic testing
  - `generators.py`: `_yield_yohou_forecaster_checks()`, `_yield_yohou_transformer_checks()`
  - `forecaster.py`, `point.py`, `interval.py`, `reduction.py`, `panel.py`: Specific check functions
- `tests/<module>/test_<file>.py`: Test files mirror source structure

**Configuration**:
- `pyproject.toml`: Dependencies, tool config (ruff, ty, interrogate, pytest)
- `noxfile.py`: Nox sessions (test, fix, docs)
- `.pre-commit-config.yaml`: Pre-commit hooks
- `.github/workflows/`: CI/CD workflows

**Documentation**:
- `.github/copilot/`: Detailed implementation plans and guides
- `docs/`: MkDocs documentation source
- `examples/`: Marimo notebooks (reactive examples)
---

## CI/CD Workflows

**Critical**: All workflows use `uv` for dependency management and run via `nox` sessions.

### Automated Checks on Push/PR

**Lint Workflow** (`.github/workflows/lint.yml`):
- Runs: `nox -s fix` (ruff, ty, interrogate)
- Python: 3.12 on ubuntu-latest
- Triggers: Push to main, all PRs
- Concurrency: Cancels previous runs on new push

**Test Workflow** (`.github/workflows/tests-os-coverage.yml`):
- Runs: `nox -s test` with full coverage
- Matrix: Python 3.12 on [ubuntu, macos, windows]
- Coverage: Uploaded to Codecov (optional)
- Triggers: Push to main, all PRs
- Uses `covdefaults` for coverage config

**Release Workflow** (`.github/workflows/release.yml`):
- Trigger: Git tags matching `v*.*.*` pattern
- Steps:
  1. Build: `uv build --sdist --wheel`
  2. Validation: `uvx twine check dist/*`
  3. PyPI publish: Trusted publishing (OIDC, no tokens)
- Environment: `pypi` (requires repository secrets)

### Local Pre-commit

**Before committing**:
```bash
uvx nox -s fix  # Runs all quality checks (same as CI)
```

Pre-commit hooks (`.pre-commit-config.yaml`):
- `ruff check --fix`: Linting
- `ruff format`: Code formatting
- `ty check src`: Type checking (NOT mypy)
- `interrogate`: Docstring coverage (100% required)

**Critical Patterns**:
- All CI uses `uv python install --python-preference only-managed` for reproducibility
- Nox sessions use `uv sync` with dependency groups
- Windows requires special PATH setup for `.local/bin`
- Coverage combines parallel runs via `coverage combine`
