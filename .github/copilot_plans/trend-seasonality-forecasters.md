# Trend and Seasonality Forecasters Implementation Plan

## Overview

This plan outlines the implementation of specialized forecasters for decomposition-based forecasting:

**Trend Forecasters**:
- **PolynomialTrendForecaster**: Polynomial regression (linear is degree=1 special case)
- **ExponentialTrendForecaster**: Exponential growth/decay patterns  
- **MovingAverageTrendForecaster**: Moving average baseline

**Seasonality Forecasters**:
- **SeasonalityForecaster**: Pattern-based seasonality (naive/average/median)
- **FourierSeasonalityForecaster**: Fourier series representation

All forecasters follow Yohou's architecture patterns and integrate with the existing forecaster testing infrastructure.

## Design Philosophy

**Core Principle**: These forecasters focus on single components of time series decomposition (trend or seasonality) rather than full forecasting. They're designed to:
1. Work as standalone forecasters for component-only predictions
2. Compose with ColumnForecaster for additive/multiplicative decomposition workflows
3. Serve as baselines for more complex models
4. Enable interpretable, explainable forecasting

**Design Decision: Separate Classes per Method**:
- Each trend extraction method has its own class (not a single class with `method` parameter)
- Enables method-specific parameters and validation
- Clearer API and documentation
- Follows sklearn convention (e.g., `LinearRegression`, `Ridge`, `Lasso` are separate)
- Linear trend is polynomial with degree=1, so only `PolynomialTrendForecaster` needed

**Use Cases**:
- Baseline models for benchmarking
- Component extraction for residual analysis
- Educational examples of decomposition-based forecasting
- Building blocks for ensemble methods

## Architecture

### Class Hierarchy

```
BaseForecaster
└── BasePointForecaster
    ├── PolynomialTrendForecaster (new)
    ├── ExponentialTrendForecaster (new)
    ├── MovingAverageTrendForecaster (new)
    └── _BaseSeasonalityForecaster (new, abstract)
        ├── SeasonalityForecaster (new)
        └── FourierSeasonalityForecaster (new)
```

**Key Design Decisions**:
- All are **point forecasters** (not reduction forecasters - no sklearn estimator wrapped)
- All are **stateless** for trend/seasonality patterns (fitted patterns stored, not raw data)
- All support **panel data** via struct columns
- None use feature transformers (X is optional, reserved for future external regressors)
- **_BaseSeasonalityForecaster** provides common logic for seasonality forecasters:
  - Time-to-phase conversion (handles irregular intervals)
  - Phase tracking for wrap-around prediction
  - Seasonality validation
  - Reduces code duplication between pattern-based and Fourier methods

### File Structure

```
src/yohou/point_forecaster/
├── __init__.py                          # Update to export new classes
├── base.py                              # No changes needed
├── naive.py                             # Reference implementation
├── reduction.py                         # Reference implementation
├── polynomial_trend.py                  # New: PolynomialTrendForecaster
├── exponential_trend.py                 # New: ExponentialTrendForecaster
├── moving_average_trend.py              # New: MovingAverageTrendForecaster
├── _base_seasonality.py                 # New: _BaseSeasonalityForecaster (abstract)
├── seasonality.py                       # New: SeasonalityForecaster
└── fourier_seasonality.py               # New: FourierSeasonalityForecaster

tests/point_forecaster/
├── __init__.py
├── test_naive.py                        # Reference for testing patterns
├── test_reduction.py                    # Reference for testing patterns
├── test_polynomial_trend.py             # New: PolynomialTrendForecaster tests
├── test_exponential_trend.py            # New: ExponentialTrendForecaster tests
├── test_moving_average_trend.py         # New: MovingAverageTrendForecaster tests
├── test_seasonality.py                  # New: SeasonalityForecaster tests
└── test_fourier_seasonality.py          # New: FourierSeasonalityForecaster tests
```

**Note**: `_BaseSeasonalityForecaster` is not exported (leading underscore indicates internal use only). It provides shared functionality for both seasonality forecaster implementations.

## Implementations

### 1. PolynomialTrendForecaster

**File**: `src/yohou/point_forecaster/polynomial_trend.py`

**Key Features**:
- Fits polynomial of specified degree using least squares
- Linear trend is special case with degree=1
- Efficient implementation using numpy.polyfit
- Extrapolates trend recursively for multi-step forecasting

**Implementation Sketch**:
```python
class PolynomialTrendForecaster(BasePointForecaster):
    """Polynomial trend forecasting (degree=1 for linear)."""
    
    def __init__(self, degree: StrictInt = 1, target_transformer=None):
        super().__init__(target_transformer=target_transformer)
        self.degree = degree
    
    def fit(self, y, X=None, forecasting_horizon=1, **params):
        y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)
        self.coefficients_ = self._fit_polynomial(y_t)
        return self
    
    def _fit_polynomial(self, y):
        # Convert time to numeric, fit polynomial for each column
        # Store coefficients [a_0, a_1, ..., a_degree]
        pass
    
    def _predict_one(self, **params):
        # Evaluate polynomial at next time point
        pass
```

**Parameters**:
- `degree`: Polynomial degree (1=linear, 2=quadratic, etc.)
- `target_transformer`: Optional transformer (e.g., LogTransform)

**Attributes**:
- `coefficients_`: DataFrame with polynomial coefficients per column

### 2. ExponentialTrendForecaster

**File**: `src/yohou/point_forecaster/exponential_trend.py`

**Key Features**:
- Fits exponential trend: y = a * exp(b*t)
- Transforms to linear via log: log(y) = log(a) + b*t
- Requires all positive values
- Good for growth/decay processes

**Implementation Sketch**:
```python
class ExponentialTrendForecaster(BasePointForecaster):
    """Exponential trend forecasting: y = a*exp(b*t)."""
    
    def __init__(self, target_transformer=None):
        super().__init__(target_transformer=target_transformer)
    
    def fit(self, y, X=None, forecasting_horizon=1, **params):
        y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)
        self._validate_positive(y_t)
        self.coefficients_ = self._fit_exponential(y_t)
        return self
    
    def _validate_positive(self, y):
        # Raise ValueError if any values <= 0
        pass
    
    def _fit_exponential(self, y):
        # Fit linear to log(y), store [a, b]
        pass
    
    def _predict_one(self, **params):
        # Evaluate a * exp(b*t) at next time point
        pass
```

**Parameters**:
- `target_transformer`: Optional transformer

**Attributes**:
- `coefficients_`: DataFrame with [a, b] per column

**Validation**:
- Raises ValueError if any y values are non-positive

### 3. MovingAverageTrendForecaster

**File**: `src/yohou/point_forecaster/moving_average_trend.py`

**Key Features**:
- Uses moving average of last N observations as constant trend
- Simple baseline, sensitive to recent fluctuations
- Produces constant (flat) forecasts

**Implementation Sketch**:
```python
class MovingAverageTrendForecaster(BasePointForecaster):
    """Moving average trend (constant forecast)."""
    
    def __init__(self, window_size: StrictInt = 12, target_transformer=None):
        super().__init__(target_transformer=target_transformer)
        self.window_size = window_size
    
    def fit(self, y, X=None, forecasting_horizon=1, **params):
        y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)
        self.trend_value_ = self._compute_moving_average(y_t)
        return self
    
    def _compute_moving_average(self, y):
        # Compute mean of last window_size rows for each column
        pass
    
    def _predict_one(self, **params):
        # Return constant trend_value_
        pass
```

**Parameters**:
- `window_size`: Number of recent observations to average
- `target_transformer`: Optional transformer

**Attributes**:
- `trend_value_`: DataFrame with constant trend value per column

### 4. _BaseSeasonalityForecaster (Abstract Base Class)

**File**: `src/yohou/point_forecaster/_base_seasonality.py`

**Purpose**: Provides shared logic for all seasonality-based forecasters (pattern-based and Fourier-based).

**Key Features**:
- Time-to-phase conversion: Maps datetime to position within seasonal cycle
- Phase tracking: Maintains current position for wrap-around prediction
- Data validation: Ensures sufficient data for given seasonality
- Interval handling: Works with irregular time intervals (converts to index-based phases)

**Abstract Methods**:
- `_extract_pattern()`: Must be implemented by subclasses to extract seasonal pattern
- `_predict_from_pattern()`: Must be implemented to generate predictions from pattern

**Implementation Sketch**:
```python
from abc import abstractmethod
import polars as pl
from yohou.point_forecaster.base import BasePointForecaster
from yohou.utils.validation import check_interval_consistency


class _BaseSeasonalityForecaster(BasePointForecaster):
    """Abstract base class for seasonality forecasters.
    
    Provides common infrastructure for pattern-based and Fourier-based
    seasonality forecasting.
    
    Parameters
    ----------
    seasonality : StrictInt
        Length of seasonal cycle (number of time steps).
    target_transformer : BaseTransformer, optional
        Transformer applied to target before forecasting.
    
    """
    
    def __init__(self, seasonality: StrictInt, target_transformer=None):
        super().__init__(target_transformer=target_transformer)
        self.seasonality = seasonality
    
    def _validate_sufficient_data(self, y: pl.DataFrame) -> None:
        """Validates that y has at least one complete seasonal cycle.
        
        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        
        Raises
        ------
        ValueError
            If y has fewer than seasonality rows.
        
        """
        if len(y) < self.seasonality:
            raise ValueError(
                f"Insufficient data: need at least {self.seasonality} observations "
                f"(one seasonal cycle), got {len(y)}"
            )
    
    def _time_to_phase(self, time_col: pl.Series) -> pl.Series:
        """Converts time column to seasonal phase indices.
        
        Handles irregular intervals by computing phase based on row position
        relative to first observation.
        
        Parameters
        ----------
        time_col : pl.Series
            Time column (datetime type).
        
        Returns
        -------
        pl.Series
            Integer phase indices in range [0, seasonality).
        
        """
        # Compute row indices relative to first observation
        row_indices = pl.arange(0, len(time_col), eager=True)
        # Map to seasonal phases with wrap-around
        phases = row_indices % self.seasonality
        return phases
    
    def _get_time_indices(self, forecasting_horizon: int) -> pl.Series:
        """Generates phase indices for future predictions.
        
        Continues from current position (_y_observed length) and wraps around
        seasonal cycle.
        
        Parameters
        ----------
        forecasting_horizon : int
            Number of steps to predict.
        
        Returns
        -------
        pl.Series
            Phase indices for next forecasting_horizon steps.
        
        """
        current_position = len(self._y_observed)
        future_indices = pl.arange(
            current_position,
            current_position + forecasting_horizon,
            eager=True
        )
        return future_indices % self.seasonality
    
    @abstractmethod
    def _extract_pattern(self, y: pl.DataFrame) -> pl.DataFrame:
        """Extracts seasonal pattern from training data.
        
        Must be implemented by subclasses.
        
        Parameters
        ----------
        y : pl.DataFrame
            Transformed target time series.
        
        Returns
        -------
        pl.DataFrame
            Seasonal pattern (length = seasonality).
        
        """
        pass
    
    @abstractmethod
    def _predict_from_pattern(self, forecasting_horizon: int, **params) -> pl.DataFrame:
        """Generates predictions from stored seasonal pattern.
        
        Must be implemented by subclasses.
        
        Parameters
        ----------
        forecasting_horizon : int
            Number of steps to predict.
        **params : dict
            Additional parameters (e.g., for metadata routing).
        
        Returns
        -------
        pl.DataFrame
            Predictions for next forecasting_horizon steps.
        
        """
        pass
    
    def _predict_one(self, **params) -> pl.DataFrame:
        """Generates predictions by delegating to _predict_from_pattern.
        
        Parameters
        ----------
        **params : dict
            Additional parameters (e.g., for metadata routing).
        
        Returns
        -------
        pl.DataFrame
            Predictions with time columns added.
        
        """
        y_pred = self._predict_from_pattern(
            forecasting_horizon=self._forecasting_horizon,
            **params
        )
        return self._add_time_columns(y_pred)
```

**Notes**:
- `_time_to_phase()` uses row indices (not datetime arithmetic) to handle irregular intervals
- `_get_time_indices()` continues from `len(_y_observed)` to maintain position across updates
- Subclasses only need to implement pattern extraction and prediction logic

### 5. SeasonalityForecaster

**File**: `src/yohou/point_forecaster/seasonality.py`

**Key Features**:
- Extracts seasonal pattern and repeats it
- Three aggregation methods: naive (last cycle), average, median
- Handles multiple complete cycles
- Inherits phase tracking from _BaseSeasonalityForecaster

**Implementation Sketch**:
```python
class SeasonalityForecaster(_BaseSeasonalityForecaster):
    """Pattern-based seasonality forecasting.
    
    Parameters
    ----------
    seasonality : StrictInt
        Length of seasonal cycle.
    method : Literal["naive", "average", "median"], default="average"
        Aggregation method for multiple cycles.
    target_transformer : BaseTransformer, optional
        Transformer applied to target.
    
    """
    
    def __init__(
        self,
        seasonality: StrictInt,
        method: Literal["naive", "average", "median"] = "average",
        target_transformer=None
    ):
        super().__init__(seasonality=seasonality, target_transformer=target_transformer)
        self.method = method
    
    def fit(self, y, X=None, forecasting_horizon=1, **params):
        y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)
        self._validate_sufficient_data(y_t)
        self.seasonal_pattern_ = self._extract_pattern(y_t)
        return self
    
    def _validate_sufficient_data(self, y):
        # naive needs 1 cycle, average/median need 2 cycles
        pass
    
    def _extract_pattern(self, y):
        # Reshape into cycles, aggregate per method
        # Return pattern with seasonal_index column
        pass
    
    def _predict_one(self, **params):
        # Look up next position in cycle (modulo seasonality)
        pass
```

**Parameters**:
- `seasonality`: Seasonal period length
- `method`: "naive" (last cycle), "average", "median"
- `target_transformer`: Optional transformer

**Attributes**:
- `seasonal_pattern_`: DataFrame with pattern (length=seasonality)

**Data Requirements**:
- Minimum 1 cycle for naive
- Minimum 2 cycles for average/median

### 6. FourierSeasonalityForecaster

**File**: `src/yohou/point_forecaster/fourier_seasonality.py`

**Key Features**:
- Represents seasonality using Fourier series
- Flexible: captures non-integer periods and multiple seasonalities
- Fits coefficients via least squares
- Inherits phase tracking from _BaseSeasonalityForecaster

**Implementation Sketch**:
```python
import numpy as np
from sklearn.linear_model import LinearRegression


class FourierSeasonalityForecaster(_BaseSeasonalityForecaster):
    """Fourier series seasonality forecasting.
    
    Parameters
    ----------
    seasonality : float
        Seasonal period length (can be non-integer).
    n_harmonics : StrictInt, default=3
        Number of Fourier harmonics to use.
    target_transformer : BaseTransformer, optional
        Transformer applied to target.
    
    """
    
    def __init__(
        self,
        seasonality: float,
        n_harmonics: StrictInt = 3,
        target_transformer=None
    ):
        super().__init__(seasonality=seasonality, target_transformer=target_transformer)
        self.n_harmonics = n_harmonics
    
    def fit(self, y, X=None, forecasting_horizon=1, **params):
        y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)
        self._validate_sufficient_data(y_t)
        self.fourier_coefficients_ = self._extract_pattern(y_t)
        return self
    
    def _extract_pattern(self, y: pl.DataFrame) -> dict:
        """Fits Fourier coefficients for each column.
        
        Parameters
        ----------
        y : pl.DataFrame
            Training data.
        
        Returns
        -------
        dict
            Dictionary mapping column names to fitted LinearRegression models.
        
        """
        coefficients = {}
        time_col = pl.arange(0, len(y), eager=True)
        
        for col_name in y.columns:
            if col_name == "time":
                continue
            
            # Build Fourier feature matrix
            X_fourier = self._build_fourier_features(time_col)
            
            # Fit linear regression
            y_values = y[col_name].to_numpy()
            model = LinearRegression()
            model.fit(X_fourier, y_values)
            coefficients[col_name] = model
        
        return coefficients
    
    def _build_fourier_features(self, time_indices: pl.Series) -> np.ndarray:
        """Constructs Fourier feature matrix.
        
        Parameters
        ----------
        time_indices : pl.Series
            Time step indices.
        
        Returns
        -------
        np.ndarray
            Shape (n_samples, 2 * n_harmonics) with sin/cos features.
        
        """
        t = time_indices.to_numpy()
        features = []
        
        for k in range(1, self.n_harmonics + 1):
            features.append(np.sin(2 * np.pi * k * t / self.seasonality))
            features.append(np.cos(2 * np.pi * k * t / self.seasonality))
        
        return np.column_stack(features)
    
    def _predict_from_pattern(self, forecasting_horizon: int, **params) -> pl.DataFrame:
        """Generates predictions using Fourier coefficients.
        
        Parameters
        ----------
        forecasting_horizon : int
            Number of steps to predict.
        **params : dict
            Additional parameters.
        
        Returns
        -------
        pl.DataFrame
            Predictions without time columns.
        
        """
        # Get future time indices
        current_position = len(self._y_observed)
        future_indices = pl.arange(
            current_position,
            current_position + forecasting_horizon,
            eager=True
        )
        
        # Build Fourier features for future times
        X_future = self._build_fourier_features(future_indices)
        
        # Predict for each column
        predictions = {}
        for col_name, model in self.fourier_coefficients_.items():
            predictions[col_name] = model.predict(X_future)
        
        return pl.DataFrame(predictions)
```

**Parameters**:
- `seasonality`: Seasonal period (can be non-integer)
- `n_harmonics`: Number of Fourier terms (more = more flexible)
- `target_transformer`: Optional transformer

**Attributes**:
- `fourier_coefficients_`: Dictionary mapping column names to fitted `LinearRegression` models

**Advantages over SeasonalityForecaster**:
- Handles non-integer seasonality (e.g., 365.25 days/year)
- Smooth seasonal curves
- Can represent multiple seasonalities simultaneously (by adding harmonics)

**Note**: Unlike pattern-based approach, Fourier representation is continuous and differentiable, making it suitable for optimization-based methods.

## Testing Infrastructure Integration

All forecasters follow the established testing pattern from `tests/point_forecaster/test_naive.py`.

### Common Test Structure

```python
import sys
from pathlib import Path
import polars as pl
import pytest
from sklearn.base import clone

from yohou.point_forecaster import <ForecasterClass>

sys.path.insert(0, str(Path(__file__).parent.parent))
from estimator_checks import _yield_yohou_forecaster_checks


@pytest.mark.parametrize(
    "forecaster,tags,expected_failures",
    [
        # Test cases with different parameters
    ],
)
def test_forecaster_checks(forecaster, tags, expected_failures, y_X_factory):
    """Run systematic checks."""
    y, X = y_X_factory(length=100, n_targets=1, n_X_features=0, seed=42)
    y_train, y_test = y[:80], y[80:]
    
    forecaster_fitted = clone(forecaster)
    forecaster_fitted.fit(y_train, forecasting_horizon=3)
    
    for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(
        forecaster_fitted, y_train, None, y_test, None, tags=tags
    ):
        if check_name not in expected_failures:
            check_func(forecaster_fitted, **check_kwargs)


def test_analytical_case():
    """Test on known analytical case."""
    pass


def test_panel_data(panel_time_series_factory):
    """Test with panel data."""
    pass
```

### Test Coverage

Each forecaster needs:
1. **Systematic checks**: Via `_yield_yohou_forecaster_checks`
2. **Analytical tests**: Known processes with exact solutions
3. **Edge cases**: Boundary conditions, error handling
4. **Panel data tests**: Struct column handling

### Example Analytical Tests

**PolynomialTrendForecaster**:
- Linear (degree=1): Perfect linear series y=2t+5 should forecast exactly
- Quadratic (degree=2): Perfect quadratic y=0.5t²+2t+1 should fit closely

**ExponentialTrendForecaster**:
- Perfect exponential y=10*exp(0.05t) should fit closely
- Non-positive values should raise ValueError

**MovingAverageTrendForecaster**:
- All forecasts should be constant (equal to moving average)

**SeasonalityForecaster**:
- Naive: Should repeat last cycle exactly
- Average: Should average across cycles
- Wrap-around: Should cycle correctly for multi-step predictions

**FourierSeasonalityForecaster**:
- Perfect sine wave should be captured with n_harmonics=1
- Complex patterns need more harmonics

## Integration Steps

### Step 1: Update `__init__.py`

**File**: `src/yohou/point_forecaster/__init__.py`

```python
"""Point forecasting methods."""

from yohou.point_forecaster.base import BasePointForecaster
from yohou.point_forecaster.exponential_trend import ExponentialTrendForecaster
from yohou.point_forecaster.fourier_seasonality import FourierSeasonalityForecaster
from yohou.point_forecaster.moving_average_trend import MovingAverageTrendForecaster
from yohou.point_forecaster.naive import SeasonalNaive
from yohou.point_forecaster.polynomial_trend import PolynomialTrendForecaster
from yohou.point_forecaster.reduction import PointReductionForecaster
from yohou.point_forecaster.seasonality import SeasonalityForecaster

__all__ = [
    "BasePointForecaster",
    "ExponentialTrendForecaster",
    "FourierSeasonalityForecaster",
    "MovingAverageTrendForecaster",
    "PointReductionForecaster",
    "PolynomialTrendForecaster",
    "SeasonalNaive",
    "SeasonalityForecaster",
]
```

### Step 2: Run Tests

```bash
# Run all point forecaster tests
uv run pytest tests/point_forecaster/ -v

# Run specific tests
uv run pytest tests/point_forecaster/test_polynomial_trend.py -v
uv run pytest tests/point_forecaster/test_exponential_trend.py -v
uv run pytest tests/point_forecaster/test_moving_average_trend.py -v
uv run pytest tests/point_forecaster/test_seasonality.py -v
uv run pytest tests/point_forecaster/test_fourier_seasonality.py -v

# Run with coverage
uvx nox -s test
```

### Step 3: Run Quality Checks

```bash
# Linting and formatting
uvx nox -s fix

# Type checking
uvx ty check src/yohou/point_forecaster/

# Docstring coverage
uvx interrogate src/yohou/point_forecaster/
```

## Use Cases and Examples

### Example 1: Linear Trend Baseline

```python
from yohou.point_forecaster import PolynomialTrendForecaster

# Fit linear trend (degree=1)
forecaster = PolynomialTrendForecaster(degree=1)
forecaster.fit(y_train, forecasting_horizon=12)
y_pred = forecaster.predict(forecasting_horizon=12)
```

### Example 2: Exponential Growth

```python
from yohou.point_forecaster import ExponentialTrendForecaster

# Forecast exponential growth (e.g., viral spread, compound growth)
forecaster = ExponentialTrendForecaster()
forecaster.fit(y_train, forecasting_horizon=30)
y_pred = forecaster.predict(forecasting_horizon=30)
```

### Example 3: Seasonal Decomposition

```python
from yohou.forecaster import ColumnForecaster
from yohou.point_forecaster import (
    PolynomialTrendForecaster,
    SeasonalityForecaster
)

# Decompose into trend + seasonality
forecaster = ColumnForecaster(
    forecasters=[
        ("trend", PolynomialTrendForecaster(degree=2), ["sales"]),
        ("seasonal", SeasonalityForecaster(seasonality=12), ["sales_detrended"]),
    ]
)

forecaster.fit(y, forecasting_horizon=12)
y_pred = forecaster.predict(forecasting_horizon=12)
```

### Example 4: Fourier Seasonality for Complex Patterns

```python
from yohou.point_forecaster import FourierSeasonalityForecaster

# Capture complex seasonal patterns (e.g., multiple peaks)
forecaster = FourierSeasonalityForecaster(
    seasonality=365.25,  # Yearly with leap years
    n_harmonics=5  # More harmonics = more flexibility
)
forecaster.fit(y_train, forecasting_horizon=30)
y_pred = forecaster.predict(forecasting_horizon=30)
```

### Example 5: Baseline Comparison

```python
from yohou.point_forecaster import (
    PolynomialTrendForecaster,
    ExponentialTrendForecaster,
    MovingAverageTrendForecaster,
    SeasonalityForecaster
)
from yohou.metrics import MAE

baselines = {
    "Linear": PolynomialTrendForecaster(degree=1),
    "Quadratic": PolynomialTrendForecaster(degree=2),
    "Exponential": ExponentialTrendForecaster(),
    "MA(12)": MovingAverageTrendForecaster(window_size=12),
    "Seasonal": SeasonalityForecaster(seasonality=12, method="average"),
}

for name, forecaster in baselines.items():
    forecaster.fit(y_train, forecasting_horizon=12)
    y_pred = forecaster.predict(forecasting_horizon=12)
    score = MAE().score(y_test, y_pred)
    print(f"{name}: MAE = {score:.2f}")
```

## Implementation Checklist

### Trend Forecasters

- [ ] **PolynomialTrendForecaster** (`polynomial_trend.py`)
  - [ ] Implement class with degree parameter
  - [ ] Time to numeric conversion
  - [ ] Polynomial fitting via numpy.polyfit
  - [ ] Coefficient storage and prediction
  - [ ] NumPy docstrings (100% coverage)
  - [ ] Type hints with pydantic
  - [ ] Panel data support (automatic via parent)

- [ ] **ExponentialTrendForecaster** (`exponential_trend.py`)
  - [ ] Implement class
  - [ ] Positive value validation
  - [ ] Log-linear fitting
  - [ ] Exponential prediction
  - [ ] NumPy docstrings (100% coverage)
  - [ ] Type hints with pydantic
  - [ ] Panel data support

- [ ] **MovingAverageTrendForecaster** (`moving_average_trend.py`)
  - [ ] Implement class with window_size parameter
  - [ ] Moving average computation
  - [ ] Constant forecast generation
  - [ ] NumPy docstrings (100% coverage)
  - [ ] Type hints with pydantic
  - [ ] Panel data support

### Seasonality Forecasters

- [ ] **SeasonalityForecaster** (`seasonality.py`)
  - [ ] Implement class with seasonality and method parameters
  - [ ] Data sufficiency validation
  - [ ] Cycle extraction logic (naive/average/median)
  - [ ] Position tracking for wrap-around
  - [ ] NumPy docstrings (100% coverage)
  - [ ] Type hints with pydantic
  - [ ] Panel data support

- [ ] **FourierSeasonalityForecaster** (`fourier_seasonality.py`)
  - [ ] Implement class with seasonality and n_harmonics parameters
  - [ ] Fourier feature matrix construction
  - [ ] Linear regression for coefficient fitting
  - [ ] Fourier series evaluation
  - [ ] NumPy docstrings (100% coverage)
  - [ ] Type hints with pydantic
  - [ ] Panel data support

### Tests

- [ ] **test_polynomial_trend.py**
  - [ ] Systematic checks via `_yield_yohou_forecaster_checks`
  - [ ] Linear analytical test (degree=1 on y=2t+5)
  - [ ] Quadratic analytical test (degree=2 on y=0.5t²+2t+1)
  - [ ] Panel data test
  - [ ] Edge cases (degree=0, very high degree)

- [ ] **test_exponential_trend.py**
  - [ ] Systematic checks
  - [ ] Exponential analytical test
  - [ ] Non-positive value error test
  - [ ] Panel data test

- [ ] **test_moving_average_trend.py**
  - [ ] Systematic checks
  - [ ] Constant forecast test
  - [ ] Different window sizes
  - [ ] Panel data test

- [ ] **test_seasonality.py**
  - [ ] Systematic checks for each method
  - [ ] Naive exact repetition test
  - [ ] Average aggregation test
  - [ ] Median robustness test
  - [ ] Wrap-around test
  - [ ] Insufficient data error test
  - [ ] Panel data test

- [ ] **test_fourier_seasonality.py**
  - [ ] Systematic checks
  - [ ] Sine wave test (n_harmonics=1)
  - [ ] Complex pattern test (higher harmonics)
  - [ ] Non-integer seasonality test
  - [ ] Panel data test

### Integration

- [ ] Update `src/yohou/point_forecaster/__init__.py`
- [ ] Update `src/yohou/__init__.py`
- [ ] Run all tests: `uvx nox -s test`
- [ ] Run quality checks: `uvx nox -s fix`
- [ ] Verify coverage ≥ 90% for new files

### Documentation

- [ ] Create marimo example: `examples/trend_forecasting.py`
- [ ] Create marimo example: `examples/seasonal_forecasting.py`
- [ ] Create marimo example: `examples/decomposition_baselines.py`
- [ ] Add to API reference documentation
- [ ] Add to user guide (baseline models section)

## Future Enhancements

### Trend Forecasters
1. **Damped Trends**: Add damping parameter for exponential trends
2. **Robust Regression**: Use robust estimators (RANSAC, Huber) for outlier resistance
3. **Change Point Detection**: Detect and handle trend breaks
4. **STL Trend**: Use LOESS for robust, flexible trend extraction

### Seasonality Forecasters
1. **Multiple Seasonalities**: Combine multiple seasonal periods
2. **Dynamic Seasonality**: Allow seasonal pattern to evolve over time
3. **Seasonal Strength Metric**: Measure seasonal component strength
4. **Automatic Seasonality Detection**: Auto-detect period from data

### Composition
1. **TrendSeasonalForecaster**: Built-in additive/multiplicative decomposition
2. **EnsembleDecomposition**: Weighted combination of methods
3. **Hierarchical Seasonality**: Nested seasonal patterns (daily within weekly)

## References

- Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: principles and practice* (3rd ed.). Chapter 3: Time series decomposition.
- Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics*, 6(1), 3-73.
- Harvey, A. C. (1990). *Forecasting, structural time series models and the Kalman filter*. Cambridge University Press.
- Yohou Architecture: `.github/copilot-instructions.md`
- Testing Infrastructure: `.github/copilot_plans/forecaster-testing-infrastructure.md`
