# How to Create a Custom Scorer

This guide shows how to implement a custom scoring metric for evaluating
forecasts. Use this when the built-in metrics (MAE, RMSE, MASE, etc.) do not
capture the evaluation criterion your domain requires.

!!! tip "Interactive version available"
    Try this guide as an interactive notebook:
    [View](/examples/custom_scorer/) · [Open in marimo](/examples/custom_scorer/edit/)

## Prerequisites

- Familiarity with the fit/score API ([Evaluate Forecast Accuracy](/pages/how-to/evaluate-forecast-accuracy/))
- Understanding of aggregation modes ([Point Metrics notebook](/examples/point_metrics/))

## 1. Implement `_compute_raw_errors`

Create a class that extends `BasePointScorer` and implement `_compute_raw_errors`.
This method receives two aligned DataFrames (time column already removed) and
returns per-timestep, per-component raw scores:

```python
import polars as pl
from yohou.metrics.base import BasePointScorer


class MaxAbsoluteError(BasePointScorer):
    """Maximum absolute error across the forecast horizon."""

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
    }

    _metric_name = "max_ae"

    def __init__(
        self,
        aggregation_method="all",
        groups=None,
        components=None,
    ):
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
        )

    def _compute_raw_errors(self, y_truth, y_pred):
        return (y_truth - y_pred).select(pl.all().abs())
```

`_metric_name` controls output column naming: the base class renames `score`
columns to your metric name (e.g., `target_0` becomes `target_0_max_ae`).

The base class `score()` method calls your `_compute_raw_errors`, then handles
validation, time weighting, aggregation, post-aggregate transforms, and column
renaming automatically.

## 2. Set Score Direction

`GridSearchCV` and `RandomizedSearchCV` use the `lower_is_better` property to
determine score direction. By default it is `True` (error metrics). For
accuracy-like metrics where higher is better, override `__sklearn_tags__`:

```python
class MyAccuracyMetric(BasePointScorer):

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.scorer_tags.lower_is_better = False
        return tags
```

## 3. Add Custom Parameters

If your metric has parameters beyond the standard ones, declare constraints
for automatic validation:

```python
import numbers
from yohou.utils._compat import Interval


class TrimmedMeanAbsoluteError(BasePointScorer):
    """MAE after trimming the top and bottom proportion of errors."""

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
        "trim_proportion": [
            Interval(numbers.Real, 0.0, 0.5, closed="neither")
        ],
    }

    _metric_name = "trimmed_mae"

    def __init__(
        self,
        trim_proportion=0.1,
        aggregation_method="all",
        groups=None,
        components=None,
    ):
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
        )
        self.trim_proportion = trim_proportion

    def _compute_raw_errors(self, y_truth, y_pred):
        errors = (y_truth - y_pred).select(pl.all().abs())

        # Trim extreme values per column
        n = len(errors)
        k = int(n * self.trim_proportion)
        return errors.select(
            pl.all().sort().slice(k, n - 2 * k)
        )
```

Every constructor parameter must be stored as an attribute with the same name
(scikit-learn convention). The `_parameter_constraints` dict is merged
automatically with the parent's constraints.

## 4. Add a Post-aggregate Transform

If your metric needs a final transformation after aggregation (e.g., RMSE
takes the square root of aggregated squared errors), override
`_post_aggregate`:

```python
import numpy as np


class RootMaxSquaredError(BasePointScorer):
    """Square root of the maximum squared error."""

    _metric_name = "root_max_se"

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
    }

    def __init__(self, aggregation_method="all", groups=None, components=None):
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
        )

    def _compute_raw_errors(self, y_truth, y_pred):
        return (y_truth - y_pred).select(pl.all().pow(2))

    def _post_aggregate(self, result):
        if isinstance(result, pl.DataFrame):
            numeric_cols = [c for c in result.columns if c != "time"]
            return result.with_columns([pl.col(c).sqrt() for c in numeric_cols])
        return float(np.sqrt(result))
```

## 5. Create an Interval Scorer

For metrics that evaluate prediction intervals, extend `BaseIntervalScorer`.
The predictions contain `{target}_lower_{rate}` and `{target}_upper_{rate}`
columns for each coverage rate:

```python
from yohou.metrics.base import BaseIntervalScorer


class MeanIntervalRange(BaseIntervalScorer):
    """Average width of prediction intervals."""

    _parameter_constraints: dict = {
        **BaseIntervalScorer._parameter_constraints,
    }

    _metric_name = "interval_range"

    def __init__(
        self,
        aggregation_method="all",
        coverage_rates=None,
        groups=None,
        components=None,
    ):
        super().__init__(
            aggregation_method=aggregation_method,
            coverage_rates=coverage_rates,
            groups=groups,
            components=components,
        )

    def _compute_raw_errors(self, y_truth, y_pred):
        upper_cols = [c for c in y_pred.columns if "_upper_" in c]
        lower_cols = [c for c in y_pred.columns if "_lower_" in c]

        widths = []
        for u, l in zip(upper_cols, lower_cols, strict=True):
            widths.append(
                (pl.col(u) - pl.col(l)).alias(u.replace("_upper_", "_width_"))
            )

        return y_pred.select(widths)
```

`BaseIntervalScorer` adds `coverage_rates` to the parameter set and
`"coveragewise"` as a valid aggregation method.

## 6. Test Your Scorer

Use the systematic check generator to validate API conformance. The pattern
matches how the built-in scorers are tested in `tests/metrics/test_point.py`:

```python
from datetime import datetime, timedelta

import polars as pl
import pytest
from conftest import run_checks as _run_checks_base
from yohou.metrics import MeanAbsoluteError  # replace with your scorer
from yohou.testing import _yield_yohou_scorer_checks


@pytest.fixture
def y_true_y_pred():
    base = datetime(2020, 1, 1)
    n = 10
    y_truth = pl.DataFrame({
        "time": [base + timedelta(days=i) for i in range(n)],
        "value": [float(i) for i in range(n)],
    })
    y_pred = pl.DataFrame({
        "vintage_time": [base - timedelta(days=1)] * n,
        "time": [base + timedelta(days=i) for i in range(n)],
        "value": [float(i) + 0.5 for i in range(n)],
    })
    return y_truth, y_pred


def run_checks(scorer, y_truth, y_pred):
    _run_checks_base(scorer, _yield_yohou_scorer_checks(scorer, y_truth, y_pred))


class TestMaxAbsoluteError:

    def test_systematic_checks(self, y_true_y_pred):
        y_truth, y_pred = y_true_y_pred
        scorer = MaxAbsoluteError()
        run_checks(scorer, y_truth, y_pred)

    def test_basic_score(self, y_true_y_pred):
        y_truth, y_pred = y_true_y_pred
        scorer = MaxAbsoluteError()
        scorer.fit(y_truth)
        score = scorer.score(y_truth, y_pred)
        assert isinstance(score, float)
        assert score >= 0
```

The generator runs 8 checks for point scorers covering tags, aggregation,
parameter validation, and fit state.

## Troubleshooting

**`NotFittedError` when calling `score()`**
:   Call `scorer.fit(y_train)` before `scorer.score(y_test, y_pred)`. Scorers
    need training data to infer the time interval and panel structure.

**Negative scores in `GridSearchCV.best_score_`**
:   scikit-learn negates scores for `lower_is_better=True` scorers. Recover
    the raw value with `raw_mae = -search.best_score_`.

**`_compute_raw_errors` receives unexpected columns**
:   The base class strips the time column before calling your method. Both
    `y_truth` and `y_pred` contain only numeric target columns.

## See Also

- [Custom Estimator Reference](/pages/api/custom-estimators/): full API for all component types
- [Extending Yohou](/pages/explanation/extending-yohou/): when to extend vs compose
- [Evaluate Forecast Accuracy](/pages/how-to/evaluate-forecast-accuracy/): using built-in metrics
- [Point Metrics](/examples/point_metrics/): interactive notebook showing all built-in point scorers
