# How to Create a Custom Scorer

This guide shows you how to implement a custom forecasting metric that
integrates with yohou's scoring pipeline. Five base classes and three
implementation patterns are covered, from simplest (override one method)
to most flexible (full `score()` override). Every pattern plugs into
the same aggregation, weighting, and multi-vintage infrastructure.

## Prerequisites

- yohou installed ([Getting Started](../getting-started/index.md))
- Familiarity with scikit-learn's `BaseEstimator` conventions
- Understanding of yohou's forecasting data format (`time` column, optional
  `vintage_time` column)

## Choose a Base Class

Your scorer's base class determines the prediction format it expects and
the pipeline machinery it inherits:

| Base class | Prediction format | Abstract method | Use when |
|---|---|---|---|
| `BasePointScorer` | Numeric value columns | `_compute_raw_errors` | Evaluating point predictions |
| `BaseIntervalScorer` | `__lower` / `__upper` column pairs | `_compute_raw_scores` | Evaluating prediction intervals |
| `BaseClassProbaScorer` | Probability columns per class | `_compute_raw_errors` | Evaluating class probability predictions |
| `BaseHardLabelScorer` | Probability columns (argmaxed internally) | `_compute_metric_from_counts` | Evaluating hard classification labels |
| `BaseRankingScorer` | Probability columns per class | `_compute_ranking_metric` | Evaluating ranking quality (AUC) |

All base classes are importable from `yohou.metrics.base`.

## Required Class Attributes

Every scorer must define:

- **`_metric_name`** (str): Controls output column names when results are
  returned as a DataFrame (e.g., `"mae"` produces `value__mae`).
- **`_parameter_constraints`** (dict): Merged with the base class constraints
  for input validation. At minimum, pass through the parent constraints.

Optional attributes:

- **`_lower_is_better`** (bool, default `True`): Set to `False` for metrics
  where higher is better (R², accuracy, AUC). This propagates through
  `__sklearn_tags__()` and is used by model selection utilities.

## Pattern 1: Per-row Decomposable Metric

The simplest pattern. Override `_compute_raw_errors` to return per-row
per-component errors. The base `score()` method handles weight resolution,
weight application, row/component/group collapse, score transforms,
vintage aggregation, and column renaming automatically.

### 1. Define the class

```python
import polars as pl

from yohou.metrics.base import BasePointScorer


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
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
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

Pattern 1 scorers inherit full weight support (`time_weight`,
`step_weight`, `vintage_weight`) from `BasePointScorer.score()` with
no additional code.

### 2. Optional: add custom `__init__` parameters

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
    ) -> None:
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
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

### 3. Optional: add a post-aggregation transform

If your metric requires a transform after row aggregation (e.g., square root
for RMSE), override `_transform_scores`. This method receives a DataFrame
with value columns only (no metadata columns) and runs after component/group
collapse but before vintage collapse:

```python
class RootMeanSquaredError(BasePointScorer):

    _metric_name = "rmse"

    def _compute_raw_errors(self, y_truth, y_pred):
        return (y_truth - y_pred).select(pl.all().pow(2))

    def _transform_scores(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.select(pl.all().sqrt())
```

Because `_transform_scores` fires per vintage before the cross-vintage
mean, each vintage contributes its own $\sqrt{\text{MSE}}$ rather than
$\sqrt{\text{mean MSE across vintages}}$.

### 4. Optional: change the row aggregation function

If your metric uses a different aggregation (e.g., max instead of mean),
override `_collapse_rows`:

```python
class MaxAbsoluteError(BasePointScorer):

    _metric_name = "max_ae"

    def _compute_raw_errors(self, y_truth, y_pred):
        return (y_truth - y_pred).select(pl.all().abs())

    def _collapse_rows(self, df, context, dims):
        return self._collapse_rows_with(df, context, dims, agg_fn="max")
```

The `_collapse_rows_with` helper accepts any Polars aggregation function
name (`"mean"`, `"sum"`, `"max"`, `"min"`).

### 5. Optional: override `fit()` for training-data statistics

Some metrics need statistics from the training set (e.g., RMSSE needs
seasonal naive errors). Override `fit()`, call `super().fit()`, and store
the statistics as attributes ending in `_` (sklearn convention):

```python
from yohou.utils import validate_scorer_data
from yohou.utils._compat import _fit_context


class RootMeanSquaredScaledError(BasePointScorer):

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
        "seasonality": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    _metric_name = "rmsse"

    def __init__(self, seasonality=1, **kwargs):
        super().__init__(**kwargs)
        self.seasonality = seasonality

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y_train, *, forecaster=None, **params):
        super().fit(y_train, forecaster=forecaster, **params)
        y_values, _, _ = validate_scorer_data(
            self, y_train, y_pred=None, reset=True
        )
        self.scale_factors_ = self._compute_scale(y_values, self.seasonality)
        return self
```

Pass `["_is_fitted", "scale_factors_"]` to `check_is_fitted` in `score()`
to ensure both base and custom fitted state are verified.

Note: import `_fit_context` from `yohou.utils._compat`, not directly
from sklearn.

## Pattern 2: Whole-column Computation

For metrics that cannot be decomposed into independent per-row errors
(e.g., R², median, directional accuracy), override `score()` and use
the `_map_per_vintage` + `_aggregate_per_vintage_scores` utilities.

You must still define `_compute_raw_errors` (it is abstract on
`BasePointScorer`), but it will not be called when `score()` is fully
overridden. A simple stub is sufficient:

```python
def _compute_raw_errors(self, y_truth, y_pred):
    """Not used directly; score() is overridden."""
    return (y_truth - y_pred).select(pl.all().abs())
```

### 1. Override `score()` with a compute function

```python
import numpy as np
import polars as pl
from sklearn.utils.validation import check_is_fitted

from yohou.utils import validate_scorer_data
from yohou.metrics.base import BasePointScorer


class R2Score(BasePointScorer):

    _metric_name = "r2"
    _lower_is_better = False

    def _compute_raw_errors(self, y_truth, y_pred):
        return (y_truth - y_pred).select(pl.all().pow(2))

    def score(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
        /,
        vintage_weight=None,  # (1)
        **params,
    ) -> float | pl.DataFrame:
        self._reject_weights(**params)  # (2)
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, context = validate_scorer_data(
            self, y_truth, y_pred
        )

        context = self._resolve_vintage_weight_to_context(  # (3)
            context, vintage_weight
        )

        def _compute_r2(yt_slice, yp_slice):  # (4)
            r2_values = {}
            for col in yt_slice.columns:
                truth = yt_slice[col].to_numpy().astype(np.float64)
                pred = yp_slice[col].to_numpy().astype(np.float64)
                ss_res = np.sum((truth - pred) ** 2)
                ss_tot = np.sum((truth - np.mean(truth)) ** 2)
                r2_values[col] = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
            return pl.DataFrame(r2_values).select(yt_slice.columns)

        result = self._map_per_vintage(  # (5)
            y_truth, y_pred, context, _compute_r2
        )
        return self._aggregate_per_vintage_scores(result, context)  # (6)
```

Key steps:

1. **`vintage_weight` as a named parameter**: Python captures it before
   `**params`, so it is not rejected by `_reject_weights`.
2. **`_reject_weights(**params)`** raises `TypeError` if the caller passes
   `time_weight` or `step_weight` (these flow into `**params` because
   they are not named in the signature).
3. **`_resolve_vintage_weight_to_context`** resolves the `vintage_weight`
   argument into `context.vintage_weight` for use by the pipeline.
4. **Compute function** receives DataFrames with value columns only (no
   `time` or `vintage_time`). Return a single-row DataFrame with one column
   per component.
5. **`_map_per_vintage`** groups rows by `vintage_time` and calls the
   compute function per group. For single-vintage data (one unique
   `vintage_time` or no `vintage_time` column), it calls the function
   once with the full data.
6. **`_aggregate_per_vintage_scores`** handles component collapse, group
   collapse, `_transform_scores`, vintage collapse, finalization, and
   column renaming.

### 2. Handle edge cases via `None` returns

If a vintage slice has insufficient data, return `None` from the compute
function. `_map_per_vintage` skips that vintage and excludes it from the
cross-vintage aggregation:

```python
def _compute_mda(yt_slice, yp_slice):
    if len(yt_slice) < 2:
        return None  # Need at least 2 rows for directional diff
    mda_values = {}
    for col in yt_slice.columns:
        truth_diff = np.diff(yt_slice[col].to_numpy().astype(np.float64))
        pred_diff = np.diff(yp_slice[col].to_numpy().astype(np.float64))
        matches = (np.sign(truth_diff) == np.sign(pred_diff)).astype(np.float64)
        mda_values[col] = float(np.mean(matches))
    return pl.DataFrame(mda_values).select(yt_slice.columns)
```

If all vintages return `None`, `_map_per_vintage` raises `ValueError`.

## Pattern 3: Full Override (Classification and Ranking)

For metrics with unique aggregation needs (confusion matrix scorers,
ranking metrics), override `score()` entirely but delegate to
`_aggregate_per_vintage_scores` for the tail pipeline.

### Hard-label scorers

`BaseHardLabelScorer` subclasses override `_compute_metric_from_counts`.
The base class handles converting probabilities to hard labels via argmax,
building per-row TP/FP/FN indicator columns, summing them per vintage
(via `_collapse_rows_with(..., agg_fn="sum")`), and calling your method
with the aggregated confusion counts.

The `counts` DataFrame has three columns: `"tp"`, `"fp"`, `"fn"` (each
row corresponds to a temporal group):

```python
import polars as pl

from yohou.metrics.base import BaseHardLabelScorer


class Precision(BaseHardLabelScorer):

    _metric_name = "precision"
    _lower_is_better = False

    def _compute_metric_from_counts(self, counts: pl.DataFrame) -> pl.DataFrame:
        """Compute precision = TP / (TP + FP)."""
        return counts.select(
            pl.when(pl.col("tp") + pl.col("fp") == 0)
            .then(self.zero_division)
            .otherwise(pl.col("tp") / (pl.col("tp") + pl.col("fp")))
            .alias("value")
        )
```

`BaseHardLabelScorer` provides two extra `__init__` parameters:

- **`average`** (`"macro"`, `"micro"`, or `"weighted"`): Class averaging
  strategy. `"micro"` aggregates TP/FP/FN across all classes before
  computing the metric; `"macro"` and `"weighted"` compute per-class first.
- **`zero_division`** (float, default `0.0`): Value to return when the
  metric denominator is zero.

### Ranking scorers

`BaseRankingScorer` subclasses override `_compute_ranking_metric`, which
receives numpy arrays for a single class (one-vs-rest) and returns a
float. The base class handles per-vintage grouping, one-vs-rest
decomposition, class averaging, and sample weight slicing:

```python
import numpy as np
from sklearn.metrics import roc_auc_score

from yohou.metrics.base import BaseRankingScorer


class ROCAuC(BaseRankingScorer):

    _metric_name = "roc_auc"
    _lower_is_better = False

    def _compute_ranking_metric(
        self,
        y_true_binary: np.ndarray,
        y_proba: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """Compute ROC AUC via sklearn."""
        return float(roc_auc_score(
            y_true_binary, y_proba, sample_weight=sample_weight
        ))
```

`BaseRankingScorer` provides an `average` parameter (`"macro"` or
`"weighted"`). Unlike hard-label scorers, `"micro"` is not supported
because ranking metrics require per-class probability vectors.

Both hard-label and ranking scorers inherit full weight support
(`time_weight`, `step_weight`, `vintage_weight`) from their base
classes.

## Interval Scorers

Interval scorers follow Pattern 1 but with an additional coverage rate
dimension.

### 1. Override `_compute_raw_scores`

`BaseIntervalScorer` uses `_compute_raw_scores` instead of
`_compute_raw_errors`. The method receives two extra parameters:

```python
import polars as pl

from yohou.metrics.base import BaseIntervalScorer


class EmpiricalCoverage(BaseIntervalScorer):

    _parameter_constraints: dict = {
        **BaseIntervalScorer._parameter_constraints,
    }

    _metric_name = "coverage"
    _lower_is_better = False

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        coverage_rates: list[float] | dict[float, float] | None = None,
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ) -> None:
        agg_list = aggregation_method
        if aggregation_method == "all":
            agg_list = [
                "stepwise", "vintagewise", "componentwise",
                "groupwise", "coveragewise",
            ]
        super().__init__(
            aggregation_method=agg_list,
            coverage_rates=coverage_rates,
            groups=groups,
            components=components,
        )

    def _compute_raw_scores(
        self, y_truth, y_pred, coverage_rates, target_columns
    ):
        """Compute per-row empirical coverage indicators."""
        frames = []
        for rate in coverage_rates:
            rate_data = {}
            for col in target_columns:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"
                if lower_col in y_pred.columns and upper_col in y_pred.columns:
                    in_interval = (
                        (y_truth[col] >= y_pred[lower_col])
                        & (y_truth[col] <= y_pred[upper_col])
                    )
                    rate_data[col] = in_interval.cast(pl.Float64)
            frames.append(
                pl.DataFrame(rate_data)
                .with_columns(pl.lit(rate).alias("coverage_rate"))
            )
        return pl.concat(frames)
```

Key differences from point scorers:

- **`coverage_rates`** (list of float): The coverage rates extracted from
  the prediction column names (e.g., 0.5 and 0.9 from
  `value_lower_0.5` / `value_upper_0.9`).
- **`target_columns`** (list of str): The base target column names
  (e.g., `["value"]`).
- **Return format**: The returned DataFrame must include a
  `"coverage_rate"` column. Rows = `n_timesteps × n_rates`.

### 2. Expand `"all"` to include `"coveragewise"`

Interval scorers have a `"coveragewise"` aggregation dimension. When
`aggregation_method="all"`, expand it to include `"coveragewise"` in
`__init__` before calling `super().__init__()`, as shown above.

### 3. Interval scorers with custom aggregation

If your interval scorer needs to override coverage rate collapse (e.g.,
CRPS uses trapezoidal integration), override `_collapse_coverage_rates`
and/or `_aggregate_scores`. Add custom parameters with constraints:

```python
from yohou.utils._compat import StrOptions


class ContinuousRankedProbabilityScore(BaseIntervalScorer):

    _parameter_constraints: dict = {
        **BaseIntervalScorer._parameter_constraints,
        "integration": [StrOptions({"mean", "trapezoidal"})],
        "coverage_rates": [list, None],  # dict not supported for CRPS
    }

    _metric_name = "crps"

    def __init__(self, integration="mean", **kwargs):
        self.integration = integration
        super().__init__(**kwargs)
```

## Class-Probability Scorers

Class-probability scorers follow Pattern 1 with additional data handling
for probability columns.

### 1. Override `_compute_raw_errors`

The method receives `y_truth` (true class labels, no time column) and
`y_pred` (probability columns, no time column). Use the helper methods
`_extract_target_columns` and `_extract_class_proba_columns` to parse
the column naming convention (`{target}_proba_{class}`):

```python
import numpy as np
import polars as pl

from yohou.metrics.base import BaseClassProbaScorer


class LogLoss(BaseClassProbaScorer):

    _metric_name = "log_loss"

    def _compute_raw_errors(self, y_truth, y_pred):
        target_cols = self._extract_target_columns(y_truth)
        scores_dict = {}
        for target_col in target_cols:
            proba_cols, class_labels = self._extract_class_proba_columns(
                y_pred, target_col
            )
            true_labels = y_truth[target_col].cast(pl.String)
            per_row = []
            for row_idx in range(len(y_truth)):
                true_label = true_labels[row_idx]
                label_idx = (
                    class_labels.index(true_label)
                    if true_label in class_labels else None
                )
                if label_idx is not None:
                    prob = float(y_pred[proba_cols[label_idx]][row_idx])
                    prob = np.clip(prob, 1e-15, 1 - 1e-15)
                    per_row.append(-np.log(prob))
                else:
                    per_row.append(-np.log(1e-15))
            scores_dict[target_col] = per_row
        return pl.DataFrame(scores_dict)
```

`BaseClassProbaScorer.score()` validates that probabilities are finite
and in [0, 1] before calling `_compute_raw_errors`, so you can rely on
valid inputs. It also provides full weight support (`time_weight`,
`step_weight`, `vintage_weight`) automatically.

## Weight Support

### Pattern 1: automatic

`BasePointScorer.score()`, `BaseIntervalScorer.score()`, and
`BaseClassProbaScorer.score()` accept `time_weight`, `step_weight`,
and `vintage_weight` and handle them automatically:

- `time_weight` and `step_weight` are resolved and applied at the row
  level via `_pre_filter_zero_weights` and `_apply_weights`.
- `vintage_weight` is resolved into `context.vintage_weight` and applied
  during `_collapse_vintage_dimension` (weighted mean across vintages).

No additional code is needed.

### Pattern 2: vintage_weight only

Most whole-column metrics reject `time_weight` and `step_weight` because
the computation is not decomposable per row. Support `vintage_weight` by
naming it in the `score()` signature:

```python
def score(self, y_truth, y_pred, /, vintage_weight=None, **params):
    self._reject_weights(**params)
    # ...
    context = self._resolve_vintage_weight_to_context(context, vintage_weight)
```

Since `vintage_weight` is named, Python captures it before `**params`.
`_reject_weights` then raises if `time_weight` or `step_weight` appear
in `**params`.

### Pattern 2: full weight support

If your whole-column metric supports all three weight types, use
`_pre_filter_zero_weights` instead of `_resolve_vintage_weight_to_context`:

```python
def score(self, y_truth, y_pred, /,
          time_weight=None, step_weight=None, vintage_weight=None,
          **params):
    check_is_fitted(self, ["_is_fitted"])
    y_truth, y_pred, context = validate_scorer_data(self, y_truth, y_pred)

    y_truth, y_pred, context, tw, sw, _ = self._pre_filter_zero_weights(
        y_truth, y_pred, context, time_weight, step_weight, vintage_weight,
    )
    # tw/sw are resolved numpy arrays (or dicts for panel-aware weights)
    # vintage_weight is absorbed into context.vintage_weight; _ is always None
    # Apply tw/sw manually in your compute function if needed
    ...
```

### Weight format options

All three weight types accept the same formats:

| Format | Example | Behavior |
|---|---|---|
| Callable (1-param) | `lambda ts: pl.Series([1.0, 2.0, 1.0])` | Called with the key series, returns weight series |
| Callable (2-param) | `lambda ts, group: pl.Series([...])` | Panel-aware: called per group |
| DataFrame | `pl.DataFrame({"time": [...], "weight": [...]})` | Joined on the key column |
| Dict | `{"2020-01-01": 1.0, "*": 0.5}` | Keys are matched to values; `"*"` is the default |

The key column depends on the weight type: `"time"` for `time_weight`,
`"forecasting_step"` for `step_weight`, `"vintage_time"` for
`vintage_weight`.

## Register in `__init__.py`

Add your scorer to `src/yohou/metrics/__init__.py`:

```python
from .my_module import MyCustomScorer

_SCORER_REGISTRY: dict[str, type[BaseScorer]] = {
    # ...existing entries...
    "my_metric": MyCustomScorer,
}
```

The registry key becomes the short name used by `get_scorer("my_metric")`.

## Test Your Scorer

yohou provides check generators that validate API conformance. The
generators test tag accessibility, aggregation method handling, parameter
validation, fit/score lifecycle, and multi-vintage scoring:

```python
import polars as pl
from datetime import datetime
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

scorer = MyCustomScorer()
scorer.fit(y_truth)

for check_name, check_func, check_kwargs in _yield_yohou_scorer_checks(
    scorer, y_truth, y_pred
):
    check_func(scorer, **check_kwargs)
```

The `check_scorer_multi_vintage` check automatically builds a 2-vintage
dataset from your test data and verifies the scorer produces a finite
result. Add your own tests for numerical correctness.

## Aggregation Pipeline Reference

All scorers converge on `_aggregate_per_vintage_scores` for the tail
pipeline. The full sequence for Pattern 1 point scorers:

1. `_pre_filter_zero_weights`: Resolve all weight arguments. Remove rows
   where any weight is zero. Store per-vintage weights in
   `context.vintage_weight`.
2. `_compute_raw_errors`: Per-row per-component error computation.
3. `_apply_weights`: Multiply scores by normalized `time_weight` and
   `step_weight`. Vintage weights are not applied here (they act at the
   cross-vintage level).
4. `_aggregate_scores`:
    1. `_collapse_coverage_rates` (interval scorers only)
    2. `_collapse_rows` (stepwise and/or vintagewise row collapse,
       preserving per-vintage rows for the next stage)
    3. `_aggregate_per_vintage_scores`:
        1. `_collapse_components` (weighted average across components)
        2. `_collapse_groups` (weighted average across panel groups)
        3. `_transform_scores` (e.g., sqrt for RMSE)
        4. `_collapse_vintage_dimension` (mean or weighted mean across
           vintages using `context.vintage_weight`)
        5. `_finalize` (attach row labels, convert 1x1 results to scalar)
        6. `_rename_metric_columns` (rename output columns to use
           `_metric_name`)

For Pattern 2 scorers, the pipeline starts at step 4.3 (the compute
function replaces steps 1 through 4.2).

For hard-label scorers, step 4.2 uses `agg_fn="sum"` (not `"mean"`),
followed by `_compute_metric_from_counts` and class averaging before
entering `_aggregate_per_vintage_scores`.

## See Also

- [yohou.metrics API Reference](../api/metrics.md) for the full list of
  built-in scorers
- [yohou.testing API Reference](../api/testing.md) for check generators
