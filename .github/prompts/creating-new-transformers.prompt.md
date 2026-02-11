---
description: "Step-by-step guide for implementing new transformers in Yohou. Use when creating a new time series transformer class."
---

# Creating New Transformers

## Quick Decision Tree

- **Stateful** (needs observation_horizon) → Set `self._observation_horizon` in `fit()`
- **Stateless** (no memory) → Don't set `_observation_horizon`
- **Windowing/Lag features** → Extend `LagTransformer` or use `tabularize()`
- **Stationarization** → See `src/yohou/preprocessing/stationarization.py`

---

## Minimal Transformer Template

```python
"""Module docstring."""

import numbers
import polars as pl
from pydantic import StrictInt
from sklearn.base import _fit_context
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted
from yohou.base import BaseTransformer
from yohou.utils import validate_transformer_data


class MyTransformer(BaseTransformer):
    """NumPy-style docstring required.

    Parameters
    ----------
    param1 : int
        Description.

    Attributes
    ----------
    fitted_attr_ : type
        Description of fitted attribute (trailing underscore required).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 2, 1), interval="1d", eager=True)
    >>> X = pl.DataFrame({"time": time, "value": range(len(time))})
    >>> transformer = MyTransformer(param1=10)
    >>> transformer.fit(X)
    MyTransformer(param1=10)
    >>> X_t = transformer.transform(X)
    >>> "time" in X_t.columns
    True
    """

    _parameter_constraints: dict = {
        **BaseTransformer._parameter_constraints,
        "param1": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(self, param1: int):
        self.param1 = param1
        # DO NOT call super().__init__() — BaseTransformer has no __init__
        # DO NOT validate parameters here — validation happens at fit time

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "MyTransformer":
        """Fit transformer.

        Parameters
        ----------
        X : pl.DataFrame
            Feature time series with "time" column.
        y : pl.DataFrame, optional
            Target time series (for API compatibility, often unused).
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        self

        """
        # Validate input data
        X = validate_transformer_data(self, X=X, reset=True)

        # OPTIONAL: Set observation horizon for stateful transformers
        # self._observation_horizon = 10  # Keep last 10 observations

        # Call parent fit (stores schema, memory, etc.)
        BaseTransformer.fit(self, X, y, **params)

        # Your fitting logic
        self.fitted_attr_ = ...  # Must set at least one fitted attribute

        return self

    def transform(self, X: pl.DataFrame, **params) -> pl.DataFrame:
        """Transform input time series.

        Parameters
        ----------
        X : pl.DataFrame
            Feature time series with "time" column.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Transformed time series (MUST include "time" column).

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)

        # Your transformation logic
        X_t = ...  # Transform X (must preserve "time" column)

        return X_t

    def inverse_transform(self, X: pl.DataFrame, **params) -> pl.DataFrame:
        """Inverse transform (optional, only if transformation is reversible).

        Parameters
        ----------
        X : pl.DataFrame
            Transformed time series with "time" column.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Original scale time series.

        """
        check_is_fitted(self, ["fitted_attr_"])
        # Your inverse transformation logic
        X_inv = ...
        return X_inv
```

---

## Stateful vs. Stateless Transformers

### Stateless (No Memory)

**Example**: Scaling, polynomial features, Fourier features
```python
def fit(self, X, y=None, **params):
    X = validate_transformer_data(self, X=X, reset=True)
    # DO NOT set self._observation_horizon
    BaseTransformer.fit(self, X, y, **params)
    # Learn parameters (mean, std, etc.)
    self.mean_ = X.select(~cs.by_name("time")).mean()
    return self
```

### Stateful (Needs Memory)

**Example**: Lag features, differencing, rolling windows
```python
def fit(self, X, y=None, **params):
    X = validate_transformer_data(self, X=X, reset=True)
    # CRITICAL: Set observation_horizon BEFORE calling BaseTransformer.fit()
    self._observation_horizon = 5  # Keep last 5 observations
    BaseTransformer.fit(self, X, y, **params)
    # Base class automatically stores self._X_observed (last 5 rows)
    return self

def transform(self, X, **params):
    # Access memory via self._X_observed
    combined = pl.concat([self._X_observed, X])  # Prepend memory
    # Compute lag features using combined data
    X_t = ...
    return X_t
```

**Key insight**: `BaseTransformer.fit()` automatically stores `self._X_observed` if `_observation_horizon > 0`.

---

## Memory Management: `update()` and `reset()`

Stateful transformers inherit automatic memory management:

```python
# update() — Add new observations to memory
transformer.update(X_new)  # Appends then calls reset()

# reset() — Trim memory to last observation_horizon rows
transformer.reset(X_all)  # Keeps last N rows based on observation_horizon

# Memory is in self._X_observed
print(len(transformer._X_observed))  # Always == observation_horizon after reset()
```

**Pattern**: `update()` = append new data + `reset()` to maintain fixed-size window.

---

## Panel Data Support

Transformers automatically support panel data (prefixed columns):

```python
# Input: Panel data
X = pl.DataFrame({
    "time": [...],
    "sales__store_1": [...],  # Prefix: sales, Suffix: store_1
    "sales__store_2": [...],
})

# Transform each group independently
transformer.fit(X)
X_t = transformer.transform(X)

# Panel groups stored in self.panel_group_names_ (set by BaseTransformer)
# Access via yohou.utils.panel.inspect_locality(X)
```

**Critical**: Transformers process panel groups together (no explicit looping needed unless custom logic required).

---

## Parameter Constraints

```python
_parameter_constraints: dict = {
    **BaseTransformer._parameter_constraints,
    "int_param": [Interval(numbers.Integral, 1, None, closed="left")],  # int >= 1
    "float_param": [Interval(numbers.Real, 0.0, 1.0, closed="both")],   # float in [0, 1]
    "list_param": [list, "array-like"],                                 # list or array
    "callable_param": [callable, None],                                 # function or None
}
```

---

## Feature Names for Output

Implement `get_feature_names_out()` if transformation changes column count/names:

```python
from sklearn.utils.validation import _check_feature_names_in

def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
    """Get output feature names for transformation.

    Parameters
    ----------
    input_features : list of str or None, default=None
        Input features. If None, uses feature_names_in_.

    Returns
    -------
    list of str
        Transformed feature names.
    """
    input_features = _check_feature_names_in(self, input_features)
    # Example: Lag transformer adds suffixes
    feature_names = [f"{col}_lag_{self.lag}" for col in input_features]
    return feature_names
```

---

## Checklist Before Committing

1. `uvx ruff check --fix src/yohou/preprocessing/<file>.py`
2. `uvx ruff format src/yohou/preprocessing/<file>.py`
3. `uvx ty check src/yohou/preprocessing/<file>.py`
4. `uvx interrogate src/yohou/preprocessing/<file>.py` (docstring coverage)
5. `uv run pytest tests/preprocessing/test_<file>.py -v`
6. `uv run pytest --doctest-modules src/yohou/preprocessing/<file>.py`
7. `uvx nox -s fix` (all quality checks)
8. Add to `__init__.py` exports

---

## Common Pitfalls

- **Time column missing**: Always preserve `"time"` column in output
- **Stateful without observation_horizon**: Must set `self._observation_horizon` in `fit()` BEFORE calling `BaseTransformer.fit()`
- **Memory not used**: Stateful transformers should use `self._X_observed` in `transform()`
- **No fitted attributes**: Must set at least one attribute with trailing `_`
- **BaseTransformer.__init__() called**: BaseTransformer has no `__init__`, don't call `super().__init__()`
- **Validation order wrong**: Call `validate_transformer_data()` BEFORE setting `_observation_horizon`
- **Panel data broken**: Ensure transformation works with prefixed columns (test with panel fixtures)

---

## Real-World Examples to Study

**Stateless transformers**:
- `src/yohou/preprocessing/stationarization.py` - Differencing, detrending

**Stateful transformers** (with observation_horizon):
- `src/yohou/preprocessing/window.py` - LagTransformer (windowing/tabularization)

**Testing**:
- `tests/preprocessing/test_window.py` - Comprehensive transformer tests
- `src/yohou/testing/transformer.py` - Check functions for systematic testing
