---
description: "Step-by-step guide for implementing new forecasters in Yohou. Use when creating a new point or interval forecaster class."
---

# Creating New Forecasters

## Quick Decision Tree

- **Pattern-based/Statistical** → Extend `BasePointForecaster` in `src/yohou/point_forecaster/`
- **ML-based reduction** → Extend `PointReductionForecaster` or `BaseReductionForecaster`
- **Interval forecasting** → Extend `BaseIntervalForecaster` in `src/yohou/interval_forecaster/`

---

## Minimal Forecaster Template

```python
"""Module docstring."""

import numbers
import polars as pl
from pydantic import StrictInt
from sklearn.base import _fit_context
from sklearn.utils._param_validation import Interval
from yohou.base import BaseTransformer
from .base import BasePointForecaster


class MyForecaster(BasePointForecaster):
    """NumPy-style docstring required.

    Parameters
    ----------
    param1 : int
        Description.
    target_transformer : BaseTransformer, optional
        Transformer for target variable (applied before forecasting).
    feature_transformer : BaseTransformer, optional
        Transformer for exogenous features X (applied before forecasting).

    Attributes
    ----------
    fitted_attr_ : type
        Description of fitted attribute (trailing underscore required).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 2, 1), interval="1d", eager=True)
    >>> y = pl.DataFrame({"time": time, "value": range(len(time))})
    >>> forecaster = MyForecaster(param1=10)
    >>> forecaster.fit(y, forecasting_horizon=5)
    MyForecaster(param1=10)
    >>> y_pred = forecaster.predict(forecasting_horizon=5)
    >>> len(y_pred)
    5
    """

    _parameter_constraints: dict = {
        **BasePointForecaster._parameter_constraints,
        "param1": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        param1: int,
        target_transformer: BaseTransformer | None = None,
        feature_transformer: BaseTransformer | None = None,
    ):
        super().__init__(
            target_transformer=target_transformer,
            feature_transformer=feature_transformer,
        )
        self.param1 = param1
        # DO NOT validate parameters here — validation happens at fit time via @_fit_context

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "MyForecaster":
        """Fit forecaster.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with "time" column.
        X : pl.DataFrame, optional
            Exogenous features with "time" column.
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        self

        """
        y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)
        # Your fitting logic using y_t, X_t (already transformed)
        # Must set at least one fitted attribute with trailing underscore
        self.fitted_attr_ = ...  # Example: self.model_, self.coefficients_, etc.
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
            Number of steps ahead to forecast. If None, uses value from fit().
        X : pl.DataFrame, optional
            Exogenous features for forecast period (must have "time" column).
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Predictions with "observed_time", "time", and target columns.

        """
        if forecasting_horizon is None:
            forecasting_horizon = self._forecasting_horizon
        # Your prediction logic
        y_pred = ...  # Must be pl.DataFrame with target columns (no time yet)
        return self._add_time_columns(y_pred)  # CRITICAL: adds observed_time + time
```

---

## Parameter Constraints

All forecasters MUST define `_parameter_constraints` for sklearn validation:

```python
_parameter_constraints: dict = {
    **ParentForecaster._parameter_constraints,
    "int_param": [Interval(numbers.Integral, 1, None, closed="left")],     # Integer ≥ 1
    "float_param": [Interval(numbers.Real, 0.0, 1.0, closed="both")],      # Float in [0, 1]
    "positive_float": [Interval(numbers.Real, 0.0, None, closed="neither")], # Float > 0
    "optional_param": [Interval(numbers.Real, 0.0, 1.0, closed="both"), None],
    "transformer_param": [BaseTransformer, None],
}
```

Validation timing: (1) automatic at fit via `@_fit_context`, (2) domain-specific in `fit()` body, (3) **never** in `__init__`.

---

## Panel Data Support

```python
def fit(self, y, X, forecasting_horizon, **params):
    y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)
    if self.panel_group_names_ is not None:
        for col_name in self.local_y_columns_:
            pass  # Process each series
    else:
        pass  # Global data
```

---

## Checklist Before Committing

1. `uvx ruff check --fix src/yohou/<module>/<file>.py`
2. `uvx ruff format src/yohou/<module>/<file>.py`
3. `uvx ty check src/yohou/<module>/<file>.py`
4. `uvx interrogate src/yohou/<module>/<file>.py` (docstring coverage)
5. `uv run pytest tests/<module>/test_<file>.py -v`
6. `uv run pytest --doctest-modules src/yohou/<module>/<file>.py`
7. `uvx nox -s fix` (all quality checks)
8. Add to `__init__.py` exports

## Fitted Attributes

All forecasters MUST set at least one fitted attribute (trailing underscore `_`) in `fit()`:

```python
def fit(self, y, X, forecasting_horizon, **params):
    y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)

    # Base class sets these automatically:
    # - self._forecasting_horizon
    # - self._observation_horizon
    # - self._y_observed (last observation_horizon rows)
    # - self._X_observed (if X provided)
    # - self.panel_group_names_ (if panel data)
    # - self.local_y_columns_ (if panel data)

    # Your forecaster MUST set custom fitted attributes:
    self.model_ = ...           # Fitted model/estimator
    self.coefficients_ = ...    # Model parameters
    self.last_values_ = ...     # State for recursive prediction
    # etc.

    return self
```

**sklearn's `check_is_fitted()` will verify these exist before `predict()`.**

## Common Pitfalls

- **Missing time columns**: Always call `self._add_time_columns(y_pred)` before returning from `predict()`
- **Panel data not handled**: Check `self.panel_group_names_` and iterate `self.local_y_columns_`
- **Transformers not applied**: Use `_pre_fit()` to get transformed data (`y_t`, `X_t`), NOT raw `y`, `X`
- **No fitted attributes**: Must set at least one attribute with trailing `_` in `fit()`
- **Doctest repr mismatches**: Use exact repr format: `MyForecaster(param1=5)`
- **Mutable default args**: Use `None` and set default in method body, not `[]` or `{}`
- **Parameter validation in `__init__`**: sklearn validates at `fit()` time, NOT construction time

## Real-World Examples to Study

**Pattern-based forecasters** (no ML, statistical patterns):
- `src/yohou/point_forecaster/naive.py` - NaiveForecaster (simplest example)
- `src/yohou/point_forecaster/seasonality.py` - PatternSeasonalityForecaster
- `src/yohou/decomposition/trend.py` - PolynomialTrendForecaster

**Model-based forecasters** (ML/statistical models):
- `src/yohou/decomposition/seasonality.py` - FourierSeasonalityForecaster
- `src/yohou/point_forecaster/reduction.py` - PointReductionForecaster (sklearn integration)
- `src/yohou/interval_forecaster/reduction.py` - IntervalReductionForecaster

**Meta-forecasters** (combine other forecasters):
- `src/yohou/decomposition/decomposer.py` - Decomposer (sequential combination)
- `src/yohou/forecaster/composition.py` - ColumnForecaster (parallel combination)
