---
description: "Step-by-step guide for implementing new metrics (scorers) in Yohou. Use when creating a new point or interval scorer class."
---

# Creating New Scorers

## Quick Decision Tree

- **Point forecast metric** → Extend `BasePointScorer` in `src/yohou/metrics/point.py`
- **Interval forecast metric** → Extend `BaseIntervalScorer` in `src/yohou/metrics/interval.py`
- **Coverage/calibration metric** → See `src/yohou/metrics/conformity.py`

---

## Minimal Point Scorer Template

```python
"""Module docstring."""

import numbers
from typing import Callable

import polars as pl
from sklearn.base import _fit_context

from yohou.utils import validate_scorer_data
from .base import BasePointScorer


class MyMetric(BasePointScorer):
    """NumPy-style docstring required.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. Options:
        - "timewise": Aggregate across time, return per-component DataFrame
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar)
    panel_group_names : list of str or None, default=None
        List of panel group names to include in scoring. If None, all groups included.
    component_names : list of str or None, default=None
        List of component (target column) names to include. If None, all included.
    panel_group_weight : dict or None, default=None
        Dictionary mapping panel group names to weights. If None, equal weights.

    Attributes
    ----------
    lower_is_better : bool
        True if lower scores are better (e.g., MAE, RMSE).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [10.0, 20.0, 30.0]
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [12.0, 19.0, 28.0]
    ... })
    >>> metric = MyMetric()
    >>> score = metric.score(y_true, y_pred)
    >>> isinstance(score, float)
    True
    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
        # Add metric-specific parameters here
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ):
        super().__init__(
            aggregation_method=aggregation_method,
            panel_group_names=panel_group_names,
            component_names=component_names,
            panel_group_weight=panel_group_weight,
        )

    def score(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
        /,
        time_weight: Callable | pl.DataFrame | None = None,
        **params,
    ) -> float | pl.DataFrame:
        """Compute metric score.

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth values with "time" column.
        y_pred : pl.DataFrame
            Predicted values with "observed_time" and "time" columns.
        time_weight : callable or pl.DataFrame, optional
            Time-based weights for scoring:
            - DataFrame: Must have "time" and "{group}_weight" columns
            - Callable (1 param): Global weight function(y_truth) -> DataFrame
            - Callable (2 params): Panel-aware weight function(y_truth, group_name) -> DataFrame
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        float or pl.DataFrame
            Scalar score (if aggregation_method="all") or DataFrame with partial aggregations.

        """
        # Validate inputs
        y_truth, y_pred = validate_scorer_data(
            self,
            y_truth=y_truth,
            y_pred=y_pred,
        )

        # Compute per-timestep, per-component scores
        scores = self._compute_scores(y_truth, y_pred)

        # Apply time weighting and aggregate
        return self._aggregate_scores(
            scores=scores,
            y_truth=y_truth,
            y_pred=y_pred,
            time_weight=time_weight,
        )

    def _compute_scores(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute per-timestep, per-component scores (before aggregation).

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth values (already validated).
        y_pred : pl.DataFrame
            Predicted values (already validated).

        Returns
        -------
        pl.DataFrame
            Scores with "time" column and score columns for each component.

        """
        import polars.selectors as cs

        # Extract target columns (exclude time columns)
        target_cols = [c for c in y_truth.columns if c != "time"]

        # Compute element-wise metric
        # Example: Absolute error
        scores = pl.DataFrame({"time": y_pred["time"]})
        for col in target_cols:
            scores = scores.with_columns(
                (y_truth[col] - y_pred[col]).abs().alias(col)
            )

        return scores
```

---

## Interval Scorer Template

```python
from .base import BaseIntervalScorer


class MyIntervalMetric(BaseIntervalScorer):
    """Interval forecast metric.

    Evaluates prediction intervals (e.g., coverage, sharpness, calibration).

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Same as point scorers.
    panel_group_names : list of str or None, default=None
        Same as point scorers.
    component_names : list of str or None, default=None
        Same as point scorers.
    panel_group_weight : dict or None, default=None
        Same as point scorers.

    """

    def score(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
        /,
        time_weight: Callable | pl.DataFrame | None = None,
        **params,
    ) -> float | pl.DataFrame:
        """Compute interval metric score.

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth with "time" and target columns.
        y_pred : pl.DataFrame
            Interval predictions with "time", "observed_time", and columns like:
            - "target_lower_0.9", "target_upper_0.9" (90% interval)
            - "target_lower_0.95", "target_upper_0.95" (95% interval)
        time_weight : callable or pl.DataFrame, optional
            Time-based weights.
        **params : dict
            Metadata routing.

        Returns
        -------
        float or pl.DataFrame
            Metric score.

        """
        y_truth, y_pred = validate_scorer_data(self, y_truth=y_truth, y_pred=y_pred)

        # Extract intervals from y_pred
        # Convention: "{target}_lower_{coverage}", "{target}_upper_{coverage}"
        scores = self._compute_scores(y_truth, y_pred)

        return self._aggregate_scores(
            scores=scores,
            y_truth=y_truth,
            y_pred=y_pred,
            time_weight=time_weight,
        )

    def _compute_scores(self, y_truth, y_pred):
        # Example: Coverage rate (fraction of times truth is within interval)
        scores = pl.DataFrame({"time": y_pred["time"]})

        for col in self.component_names_:
            # Find interval columns for this component
            lower_cols = [c for c in y_pred.columns if c.startswith(f"{col}_lower_")]
            upper_cols = [c for c in y_pred.columns if c.startswith(f"{col}_upper_")]

            # Compute coverage for each interval level
            for lower_col, upper_col in zip(lower_cols, upper_cols):
                coverage_level = lower_col.split("_")[-1]
                in_interval = (
                    (y_truth[col] >= y_pred[lower_col]) &
                    (y_truth[col] <= y_pred[upper_col])
                ).cast(pl.Float64)
                scores = scores.with_columns(
                    in_interval.alias(f"{col}_coverage_{coverage_level}")
                )

        return scores
```

---

## Aggregation Patterns

All scorers inherit `_aggregate_scores()` from `BaseScorer`, which handles:

1. **Time weighting**: Applies `time_weight` to per-timestep scores
2. **Hierarchical aggregation**: Based on `aggregation_method`

### Aggregation Examples

```python
# Scalar (fully aggregated)
metric = MyMetric(aggregation_method="all")
score = metric.score(y_truth, y_pred)  # float

# Per-component (aggregate across time)
metric = MyMetric(aggregation_method="timewise")
scores = metric.score(y_truth, y_pred)  # DataFrame with one row per component

# Per-timestep (aggregate across components)
metric = MyMetric(aggregation_method="componentwise")
scores = metric.score(y_truth, y_pred)  # DataFrame with one row per timestep

# Panel-only aggregation (per-component per-timestep, groups aggregated)
metric = MyMetric(aggregation_method="groupwise")
scores = metric.score(y_truth, y_pred)  # DataFrame with component x timestep

# Multiple partial aggregations
metric = MyMetric(aggregation_method=["timewise", "componentwise"])
score = metric.score(y_truth, y_pred)  # Scalar or per-group DataFrame (panel data)
```

---

## Time Weighting Support

Scorers automatically support `time_weight` via `_aggregate_scores()`:

```python
# DataFrame weight (explicit per-group weights)
time_weight = pl.DataFrame({
    "time": [...],
    "sales__store_1_weight": [1.0, 1.5, 2.0, ...],
    "sales__store_2_weight": [1.0, 1.0, 1.0, ...],
})
score = metric.score(y_truth, y_pred, time_weight=time_weight)

# Callable weight (global, 1 parameter)
def recency_weight(y_truth):
    return pl.DataFrame({
        "time": y_truth["time"],
        "weight": np.linspace(0.5, 1.0, len(y_truth)),  # More recent = higher weight
    })
score = metric.score(y_truth, y_pred, time_weight=recency_weight)

# Callable weight (panel-aware, 2 parameters)
def group_specific_weight(y_truth, group_name):
    # Different weight strategy per group
    weights = {...}  # Custom logic
    return pl.DataFrame({"time": y_truth["time"], f"{group_name}_weight": weights})
score = metric.score(y_truth, y_pred, time_weight=group_specific_weight)
```

**Key**: `_aggregate_scores()` calls `_process_time_weights()` automatically — no manual implementation needed.

---

## Parameter Constraints

```python
_parameter_constraints: dict = {
    **BasePointScorer._parameter_constraints,
    # No additional constraints needed if using standard aggregation
}
```

---

## Checklist Before Committing

1. `uvx ruff check --fix src/yohou/metrics/<file>.py`
2. `uvx ruff format src/yohou/metrics/<file>.py`
3. `uvx ty check src/yohou/metrics/<file>.py`
4. `uvx interrogate src/yohou/metrics/<file>.py` (docstring coverage)
5. `uv run pytest tests/metrics/test_<file>.py -v`
6. `uv run pytest --doctest-modules src/yohou/metrics/<file>.py`
7. `uvx nox -s fix` (all quality checks)
8. Add to `__init__.py` exports

---

## Common Pitfalls

- **Aggregation not implemented**: `_compute_scores()` must return per-timestep, per-component scores
- **Time column missing**: Scores DataFrame must include `"time"` column
- **Wrong sign for `lower_is_better`**: Error metrics (MAE, RMSE) = `True`, accuracy metrics (R²) = `False`
- **Time weights not supported**: Don't override `_aggregate_scores()` unless necessary — inherit base behavior
- **Panel data not handled**: Base class handles panel groups automatically via `component_names_` filtering
- **Doctest repr issues**: Scores may be floats or DataFrames depending on `aggregation_method`

---

## Real-World Examples to Study

**Point metrics**:
- `src/yohou/metrics/point.py`:
  - `MeanAbsoluteError` - Simple absolute error metric
  - `MeanSquaredError` - Squared error metric
  - `RootMeanSquaredScaledError` - Scale-independent metric

**Interval metrics**:
- `src/yohou/metrics/interval.py`:
  - `CoverageScore` - Empirical coverage rate
  - `IntervalWidth` - Average interval width (sharpness)

**Conformity metrics**:
- `src/yohou/metrics/conformity.py`:
  - `ConformityScore` - Calibration diagnostics

**Testing**:
- `tests/metrics/test_point.py` - Point metric tests
- `tests/metrics/test_interval.py` - Interval metric tests
- `src/yohou/testing/scorer.py` - Check functions (11 checks)
