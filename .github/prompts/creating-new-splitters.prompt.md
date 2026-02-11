---
description: "Step-by-step guide for implementing new time series cross-validation splitters in Yohou. Use when creating a new splitter class."
---

# Creating New Splitters

## Quick Decision Tree

- **Expanding window** (training set grows) → Extend `ExpandingWindowSplitter` pattern
- **Sliding window** (fixed training size) → Extend `SlidingWindowSplitter` pattern
- **Custom logic** → Extend `BaseSplitter` directly

---

## Minimal Splitter Template

```python
"""Module docstring."""

import numbers
from collections.abc import Iterator
from typing import Any

import numpy as np
import polars as pl
from sklearn.utils._param_validation import Interval

from yohou.model_selection import BaseSplitter
from yohou.utils import validate_splitter_data


class MySplitter(BaseSplitter):
    """NumPy-style docstring required.

    Parameters
    ----------
    n_splits : int, default=3
        Number of splits. Must be at least 2.
    test_size : int, optional
        Size of test set for each split.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
    >>> y = pl.DataFrame({"time": time, "value": range(100)})
    >>> splitter = MySplitter(n_splits=3, test_size=10)
    >>> splits = list(splitter.split(y))
    >>> len(splits)
    3
    >>> train, test = splits[0]
    >>> len(test)
    10
    """

    _parameter_constraints: dict = {
        **BaseSplitter._parameter_constraints,
        "n_splits": [Interval(numbers.Integral, 2, None, closed="left")],
        "test_size": [Interval(numbers.Integral, 1, None, closed="left"), None],
    }

    def __init__(self, n_splits: int = 3, test_size: int | None = None):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> Iterator[tuple[np.ndarray[Any, np.dtype[np.intp]], np.ndarray[Any, np.dtype[np.intp]]]]:
        """Generate indices to split time series data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with mandatory "time" column.
        X : pl.DataFrame, optional
            Exogenous features (for signature compatibility, not used in splitting logic).

        Yields
        ------
        train : ndarray
            Training set row indices for that split.
        test : ndarray
            Test set row indices for that split.

        """
        # Validate data
        y = validate_splitter_data(self, y=y, X=X)
        n_samples = len(y)

        # Generate test indices
        for test_indices in self._iter_test_indices(y, X):
            # Compute train indices (all indices before test start)
            train_indices = np.arange(0, test_indices[0])
            yield train_indices, test_indices

    def _iter_test_indices(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> Iterator[np.ndarray[Any, np.dtype[np.intp]]]:
        """Generate test indices for each split.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame, optional
            Exogenous features.

        Yields
        ------
        test : ndarray
            Test set indices for this split.

        """
        n_samples = len(y)
        test_size = self.test_size or n_samples // (self.n_splits + 1)

        # Generate test indices for each split
        for i in range(self.n_splits):
            test_start = n_samples - (self.n_splits - i) * test_size
            test_end = test_start + test_size
            yield np.arange(test_start, test_end)

    def get_n_splits(
        self,
        y: pl.DataFrame | None = None,
        X: pl.DataFrame | None = None,
    ) -> int:
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        y : pl.DataFrame, optional
            Target time series.
        X : pl.DataFrame, optional
            Exogenous features.

        Returns
        -------
        n_splits : int
            Number of splits.

        """
        return self.n_splits
```

---

## Expanding vs. Sliding Window Patterns

### Expanding Window (Training Set Grows)

```python
def split(self, y, X=None):
    n_samples = len(y)
    test_size = self.test_size or n_samples // (self.n_splits + 1)

    for i in range(self.n_splits):
        # Training set: [0, test_start)
        test_start = n_samples - (self.n_splits - i) * test_size
        test_end = test_start + test_size

        train_indices = np.arange(0, test_start)  # Grows with each split
        test_indices = np.arange(test_start, test_end)

        yield train_indices, test_indices
```

**Use case**: More data generally improves models (e.g., ARIMA, neural networks).

### Sliding Window (Fixed Training Size)

```python
def split(self, y, X=None):
    n_samples = len(y)
    test_size = self.test_size or n_samples // (self.n_splits + 1)
    train_size = self.train_size or n_samples - (self.n_splits * test_size)

    for i in range(self.n_splits):
        test_start = train_size + (i * test_size)
        test_end = test_start + test_size

        # Training set: fixed size, slides forward
        train_start = max(0, test_start - train_size)
        train_indices = np.arange(train_start, test_start)
        test_indices = np.arange(test_start, test_end)

        yield train_indices, test_indices
```

**Use case**: Stationary processes, concept drift, computational constraints.

---

## Gap Between Train and Test

Add `gap` parameter to simulate forecasting lag or prevent leakage:

```python
_parameter_constraints: dict = {
    **BaseSplitter._parameter_constraints,
    "gap": [Interval(numbers.Integral, 0, None, closed="left"), None],
}

def split(self, y, X=None):
    gap = self.gap or 0

    for i in range(self.n_splits):
        test_start = ...  # Compute test start
        test_end = test_start + self.test_size

        # Training set excludes last `gap` indices
        train_indices = np.arange(0, test_start - gap)
        test_indices = np.arange(test_start, test_end)

        yield train_indices, test_indices
```

**Example**: `gap=5` means 5 time steps between train end and test start.

---

## Panel Data Support

Splitters work with panel data automatically (split all groups together using row indices):

```python
# Panel data example
y = pl.DataFrame({
    "time": [...],
    "sales__store_1": [...],
    "sales__store_2": [...],
})

splitter = MySplitter(n_splits=3)
for train_idx, test_idx in splitter.split(y):
    y_train = y[train_idx]  # All groups, train period
    y_test = y[test_idx]    # All groups, test period
```

**No special handling needed** — row indices apply across all panel groups.

---

## Parameter Constraints

```python
_parameter_constraints: dict = {
    **BaseSplitter._parameter_constraints,
    "n_splits": [Interval(numbers.Integral, 2, None, closed="left")],
    "test_size": [Interval(numbers.Integral, 1, None, closed="left"), None],
    "train_size": [Interval(numbers.Integral, 1, None, closed="left"), None],
    "gap": [Interval(numbers.Integral, 0, None, closed="left"), None],
}
```

---

## Checklist Before Committing

1. `uvx ruff check --fix src/yohou/model_selection/split.py`
2. `uvx ruff format src/yohou/model_selection/split.py`
3. `uvx ty check src/yohou/model_selection/split.py`
4. `uvx interrogate src/yohou/model_selection/split.py` (docstring coverage)
5. `uv run pytest tests/model_selection/test_split.py -v`
6. `uv run pytest --doctest-modules src/yohou/model_selection/split.py`
7. `uvx nox -s fix` (all quality checks)
8. Add to `__init__.py` exports

---

## Common Pitfalls

- **Train/test overlap**: Ensure `test_start > train_end` (or `test_start >= train_end + gap`)
- **Non-temporal order**: Never shuffle indices — time series must maintain order
- **Off-by-one errors**: Use `np.arange(start, end)` carefully (end is exclusive)
- **Empty splits**: Validate that `n_samples` is sufficient for `n_splits * test_size + train_size`
- **Inconsistent test sizes**: Last split may have different size — document or adjust
- **Panel groups split differently**: Splitters use row indices, not per-group logic (by design)

---

## Real-World Examples to Study

**Built-in splitters**:
- `src/yohou/model_selection/split.py`:
  - `ExpandingWindowSplitter` - Training set grows
  - `SlidingWindowSplitter` - Fixed training window

**Testing**:
- `tests/model_selection/test_split.py` - Comprehensive splitter tests
- `src/yohou/testing/splitter.py` - Check functions for systematic testing (8 checks)

**Key checks to review**:
- `check_split_returns_non_overlapping_indices()` - Ensures no train/test overlap
- `check_split_preserves_temporal_order()` - Validates chronological order
- `check_get_n_splits_matches_actual()` - Confirms split count accuracy
