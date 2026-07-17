# How to Create a Custom Scorer

This guide shows you how to implement a custom forecasting metric that plugs
into yohou's scoring pipeline. Use this when the built-in metrics don't
cover your evaluation needs. The guide follows the most common pattern
(a per-row decomposable point metric) end to end.

## Prerequisites

- yohou installed ([Getting Started](../tutorials/getting-started.md))
- Familiarity with the fit/score API ([Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md))

<!-- COMPANION_NOTEBOOKS -->

## 1. Subclass the Base

Create a class that extends [`BasePointScorer`](/pages/api/generated/yohou.metrics.BasePointScorer/) and implement
`_compute_raw_errors`. The base `score()` method handles weight resolution,
weight application, aggregation, vintage collapse, and column renaming
automatically.

Every scorer must define two class attributes:

- **`_metric_name`** (str): Controls output column names (e.g., `"mae"` produces `value__mae`).
- **`_parameter_constraints`** (dict): Merged with base class constraints for validation. Pass `**BasePointScorer._parameter_constraints` at minimum.

If your metric is one where higher is better (R², accuracy), set `_lower_is_better = False`.

```python
import polars as pl

from yohou.metrics.base import BasePointScorer
from yohou.weighting import BaseWeighter


class MeanAbsoluteError(BasePointScorer):

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
    }

    _metric_name = "mae"

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
        time_weighter: BaseWeighter | None = None,
        step_weighter: BaseWeighter | None = None,
        vintage_weighter: BaseWeighter | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
            time_weighter=time_weighter,
            step_weighter=step_weighter,
            vintage_weighter=vintage_weighter,
        )

    def _compute_raw_errors(
        self, y_truth: pl.DataFrame, y_pred: pl.DataFrame
    ) -> pl.DataFrame:
        """Return per-row absolute errors."""
        return (y_truth - y_pred).select(pl.all().abs())
```

`_compute_raw_errors` receives DataFrames with only value columns (no
`time` or `vintage_time`). Return a DataFrame with the same shape: one
row per timestep, one column per component.

Scorers that implement `_compute_raw_errors` inherit full weight support
(`time_weighter`, `step_weighter`, `vintage_weighter`) from
`BasePointScorer.score()`, provided you declare those parameters on `__init__`
and forward them to `super().__init__()` (shown below).

If you are evaluating prediction intervals instead of point predictions,
extend [`BaseIntervalScorer`](/pages/api/generated/yohou.metrics.BaseIntervalScorer/) and implement `_compute_raw_scores`.
See the [yohou.metrics API Reference](../api/metrics.md) for all base
class options.

## 2. Add Custom Parameters

If your metric needs additional configuration (e.g., an epsilon to
prevent division by zero), add a parameter to `__init__` and register
its constraint:

```python
import numbers

from yohou.utils._compat import Interval


class MeanAbsolutePercentageError(BasePointScorer):

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
        "epsilon": [Interval(numbers.Real, 0, None, closed="neither")],
    }

    _metric_name = "mape"

    def __init__(
        self,
        epsilon: float = 1e-8,
        aggregation_method: list[str] | str = "all",
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
        time_weighter: BaseWeighter | None = None,
        step_weighter: BaseWeighter | None = None,
        vintage_weighter: BaseWeighter | None = None,
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
            time_weighter=time_weighter,
            step_weighter=step_weighter,
            vintage_weighter=vintage_weighter,
        )
        self.epsilon = epsilon

    def _compute_raw_errors(self, y_truth, y_pred):
        pct_errors = {}
        for col in y_truth.columns:
            abs_errors = (y_truth[col] - y_pred[col]).abs()
            pct_errors[col] = (abs_errors / (y_truth[col].abs() + self.epsilon)) * 100.0
        return pl.DataFrame(pct_errors)
```

Available constraint validators include `Interval` (numeric range) and
`StrOptions` (string enum), both from `yohou.utils._compat`.

If your scorer has no custom parameters, skip this step.

## 3. Customize Aggregation

The base class collapses raw errors to a scalar using the mean. Override
these hooks when you need different behavior:

**Post-aggregation transform** (e.g., square root for RMSE):

```python
def _transform_scores(self, df):
    return df.select(pl.all().sqrt())
```

**Custom row aggregation** (e.g., max instead of mean):

```python
def _collapse_rows(self, df, context, dims):
    return self._collapse_rows_with(df, context, dims, agg_fn="max")
```

**Training-data statistics** (e.g., RMSSE needs seasonal naive errors):
override `fit()`, call `super().fit()`, and store computed statistics as
attributes ending in `_`.

If mean aggregation with no transform is sufficient, skip this step.

## 4. Register in `__init__.py`

!!! note
    This step applies only when contributing a scorer to the Yohou package itself. If you are building a scorer in your own package, skip this section.

Add your scorer to `src/yohou/metrics/__init__.py` so it is accessible
via `get_scorer()` and `make_scorer()`:

```python
from .point import MeanAbsoluteError

_SCORER_REGISTRY: dict[str, type[BaseScorer]] = {
    # ...existing entries...
    "mae": MeanAbsoluteError,
}
```

The registry key becomes the short name used by `get_scorer("mae")`.

## 5. Test Your Scorer

yohou provides check generators that validate API conformance (tag
accessibility, aggregation methods, parameter validation, fit/score
lifecycle, multi-vintage scoring):

```python
import polars as pl
from datetime import datetime
from conftest import run_checks
from yohou.testing import _yield_yohou_scorer_checks

y_truth = pl.DataFrame({
    "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    "value": [10.0, 20.0, 30.0],
})
y_pred = pl.DataFrame({
    "vintage_time": [datetime(2019, 12, 31)] * 3,
    "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    "value": [12.0, 19.0, 28.0],
})

scorer = MeanAbsoluteError()
scorer.fit(y_truth)

run_checks(None, _yield_yohou_scorer_checks(scorer, y_truth, y_pred))
```

Most scorer checks bundle the scorer inside their keyword arguments, so
`run_checks(None, ...)` lets the helper dispatch each check correctly. (Passing
the scorer positionally as well would raise `TypeError` for a duplicate
argument.)

The generator yields approximately 11 checks depending on scorer type.
[`check_scorer_multi_vintage`](/pages/api/generated/yohou.testing.check_scorer_multi_vintage/) automatically builds a 2-vintage dataset
from your test data and verifies the scorer produces a finite result.
Add your own tests for numerical correctness alongside the generated
checks.

## 6. Use in Cross-Validation

Because a custom scorer follows the same interface as the built-ins, it plugs
directly into hyperparameter search and cross-validation as a `scoring`
objective:

```python
from sklearn.linear_model import Ridge
from yohou.model_selection import GridSearchCV, ExpandingWindowSplitter
from yohou.point import PointReductionForecaster

search = GridSearchCV(
    PointReductionForecaster(estimator=Ridge()),
    param_grid={"estimator__alpha": [0.1, 1.0, 10.0]},
    scoring=MeanAbsoluteError(),
    cv=ExpandingWindowSplitter(n_splits=5, test_size=12),
)
search.fit(y_train, forecasting_horizon=12)
```

See [Tune Hyperparameters](tune-hyperparameters.md) for the full search
workflow.

## See Also

- [Create a Point Forecaster](create-a-point-forecaster.md): custom point forecasters
- [Create an Interval Forecaster](create-an-interval-forecaster.md): custom interval forecasters
- [Create a Class-Probability Forecaster](create-a-class-proba-forecaster.md): categorical outcome forecasters
- [Create a Transformer](create-a-transformer.md): custom preprocessing and feature engineering
- [yohou.metrics API Reference](../api/metrics.md) for the full list of
  built-in scorers and all base class options
- [yohou.testing API Reference](../api/testing.md) for check generators
