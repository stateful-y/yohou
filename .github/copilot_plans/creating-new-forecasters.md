# Creating New Forecasters: Complete Guide

**Purpose**: Step-by-step guide to implementing new forecasters in Yohou with real examples from the codebase.

---

## Quick Decision Tree

**Choose your forecaster type**:
- **Pattern-based/Statistical** → `src/yohou/point_forecaster/` (e.g., naive, seasonality, trend models)
- **ML-based reduction** → Extend `PointReductionForecaster` or `BaseReductionForecaster`
- **Interval forecasting** → `src/yohou/interval_forecaster/` (extends `BaseIntervalForecaster`)

**File naming**: Use descriptive names like `polynomial_trend.py`, `seasonality.py`, `fourier_seasonality.py`

---

## Step 1: Implement Core Structure

### Minimum Requirements for Point Forecaster

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

---

## Step 2: Add Parameter Constraints

### Critical: _parameter_constraints

All forecasters MUST implement `_parameter_constraints` for sklearn validation:

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

### Common Constraint Patterns

**Integer constraints**:
```python
"degree": [Interval(numbers.Integral, 0, None, closed="left")]  # ≥ 0
"lag": [Interval(numbers.Integral, 1, None, closed="left")]     # ≥ 1
```

**Float constraints**:
```python
"alpha": [Interval(numbers.Real, 0.0, 1.0, closed="both")]      # [0, 1]
"rate": [Interval(numbers.Real, 0.0, None, closed="neither")]   # > 0
```

**Interval closed parameter**:
- `closed="left"`: `[min, max)` - min ≤ value < max
- `closed="right"`: `(min, max]` - min < value ≤ max
- `closed="both"`: `[min, max]` - min ≤ value ≤ max
- `closed="neither"`: `(min, max)` - min < value < max
- Use `None` for min/max to leave unbounded (e.g., `1, None` means ≥ 1)

**Optional parameters**:
```python
"transformer": [BaseTransformer, None]
"estimator": [RegressorMixin, None]
```

**List/string parameters** (type-only validation):
```python
"harmonics": [list]  # Validate contents in fit()
"method": [str]      # Validate allowed values in fit()
```

### Validation Timing

1. **Automatic validation** at fit time via `@_fit_context` decorator (type + range checks)
2. **Domain-specific validation** in `fit()` body after automatic validation
3. **NO validation in `__init__`** - only store parameters there

### Real-World Examples

**Example 1: FourierSeasonalityForecaster**
```python
_parameter_constraints: dict = {
    **_BaseSeasonalityForecaster._parameter_constraints,
    "harmonics": [list],  # Type validation only (list of int)
    "estimator": [RegressorMixin],  # Any sklearn regressor
}

# Domain validation in fit() - Validate list contents and constraints
if not self.harmonics:
    raise ValueError("harmonics list cannot be empty")
if any(h <= 0 for h in self.harmonics):
    raise ValueError("All harmonics must be positive integers")
if max(self.harmonics) > self.seasonality // 2:
    raise ValueError(f"Maximum harmonic ({max(self.harmonics)}) cannot exceed Nyquist limit ({self.seasonality // 2})")
```

**Example 2: PolynomialTrendForecaster**
```python
_parameter_constraints: dict = {
    **BasePointForecaster._parameter_constraints,
    "degree": [Interval(numbers.Integral, 0, None, closed="left")],  # ≥ 0
}
```

**Example 3: PatternSeasonalityForecaster with string enum**
```python
_parameter_constraints: dict = {
    **_BaseSeasonalityForecaster._parameter_constraints,
    "method": [str],  # Type validation only
}

# Custom validation in fit() for allowed values
if self.method not in ["naive", "average", "median"]:
    raise ValueError(f"Invalid method: {self.method}")
```

### Required Imports

```python
import numbers
from sklearn.base import _fit_context
from sklearn.utils._param_validation import Interval
```

---

## Step 3: Handle Panel Data (if applicable)

If your forecaster should support panel data (multiple time series with prefixed columns):

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

### Testing Panel Data

Use `panel_time_series_factory` fixture or `y_X_factory` with panel data enabled:

```python
@pytest.mark.parametrize("model_panel", [False, True])
def test_forecaster_model_panel(panel_time_series_factory, model_panel):
    y_panel = panel_time_series_factory(length=100, n_series=3, seed=42)
    forecaster = PolynomialTrendForecaster(degree=1, model_panel=model_panel)
    forecaster.fit(y_panel[:80], forecasting_horizon=5)

    # Type check: Verify estimator_ structure
    if model_panel:
        assert isinstance(forecaster.estimator_, dict)  # Per-group: dict of estimators
    else:
        assert isinstance(forecaster.estimator_, Pipeline)  # Pooled: single estimator
```

---

## Step 4: Write Comprehensive Tests

### Test File Structure

**Location**: `tests/point_forecaster/test_<forecaster_name>.py`

### Minimum Test Coverage

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

### Run Tests

```bash
# Run specific test file
uv run pytest tests/point_forecaster/test_my_forecaster.py -v

# Run single test
uv run pytest tests/point_forecaster/test_my_forecaster.py::test_my_forecaster_basic_fit_predict -v

# Run with debugger
uv run pytest tests/point_forecaster/test_my_forecaster.py::test_my_forecaster_basic_fit_predict --pdb
```

### Test Patterns for Estimator-Based Forecasters

When testing Fourier/model-based forecasters, use zero-regularization for exact recovery:

```python
from sklearn.linear_model import ElasticNet

forecaster = FourierSeasonalityForecaster(
    seasonality=12,
    harmonics=[1],
    estimator=ElasticNet(alpha=0.0, l1_ratio=0.0)  # No regularization for testing
)
```

---

## Step 5: Add Docstring Examples (Doctests)

### Critical: Runnable Examples

All public methods need docstring examples that actually run:

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

### Run Doctests

```bash
# Run doctests in specific file
uv run pytest --doctest-modules src/yohou/point_forecaster/my_forecaster.py

# Run all doctests
uvx nox -s test  # Includes doctests
```

---

## Step 6: Update Module Exports

Add to `src/yohou/point_forecaster/__init__.py`:

```python
from .my_forecaster import MyForecaster

__all__ = [
    # ... existing exports
    "MyForecaster",
]
```

---

## Step 7: Quality Checks Checklist

Before committing, ensure:

- [ ] **Linting**: `uvx ruff check src/yohou/point_forecaster/my_forecaster.py`
- [ ] **Formatting**: `uvx ruff format src/yohou/point_forecaster/my_forecaster.py`
- [ ] **Type checking**: `uvx ty check src/yohou/point_forecaster/my_forecaster.py`
- [ ] **Docstring coverage**: `uvx interrogate src/yohou/point_forecaster/my_forecaster.py` (100% required)
- [ ] **Tests pass**: `uv run pytest tests/point_forecaster/test_my_forecaster.py`
- [ ] **Doctests pass**: `uv run pytest --doctest-modules src/yohou/point_forecaster/my_forecaster.py`
- [ ] **Pre-commit**: `uvx nox -s fix` (runs all quality checks)

---

## Real-World Examples from Codebase

### Pattern-Based Forecaster

**`src/yohou/point_forecaster/seasonality.py`**:
- Stores seasonal pattern in `_extract_pattern()`
- Repeats/averages pattern in `_predict_from_pattern()`
- Validates sufficient data (at least 2 cycles)
- Supports 3 methods via `method` parameter

### Model-Based Forecaster

**`src/yohou/decomposition/seasonality.py` - `FourierSeasonalityForecaster`**:
- Uses sklearn regressor (default `ElasticNet()`) for fitting
- Builds feature matrix in `_build_fourier_features()`
- Stores fitted models in `estimator_` attribute (Pipeline or dict for panel data)
- Parameters: `seasonality` (float), `harmonics` (list[int]), `estimator` (RegressorMixin)

### Trend Forecaster

**`src/yohou/point_forecaster/polynomial_trend.py`**:
- Fits polynomial with `numpy.polyfit()`
- Extrapolates via `numpy.polyval()`
- Simple stateless prediction
- Single parameter: `degree`

---

## Common Pitfalls & Solutions

### Problem: Predictions don't have time columns
**Solution**: Always call `self._add_time_columns(y_pred)` before returning

### Problem: Panel data not handled
**Solution**: Check `self.panel_group_names_` and iterate over `self.local_y_columns_`

### Problem: Transformers not applied
**Solution**: Use `_pre_fit()` to get transformed data (`y_t`, `X_t`)

### Problem: Tests fail with type errors
**Solution**: Use `StrictInt` from pydantic for integer params, validate in `__init__`

### Problem: Doctests fail with repr mismatches
**Solution**: Use exact repr format: `MyForecaster(param1=5)` not `MyForecaster(param1=5, ...)`

### Problem: Linting fails on imports
**Solution**: Order imports: stdlib → third-party → local, use `uvx ruff check --fix`

### Problem: 100% docstring coverage not met
**Solution**: Add NumPy-style docstrings to ALL public methods, classes, modules

---

## Advanced: Reduction Forecasters

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
