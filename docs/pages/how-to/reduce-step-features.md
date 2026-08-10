# How to Reduce Forecast Step Features

This guide shows you how to shrink the columns a forecaster derives from `X_future` and `X_forecast`, using the `step_transformer` slot. Use it when a forward window gives your model far more columns than it needs: a 48-hour weather forecast over four variables becomes 192 columns, where the model usually wants twelve numbers.

## Prerequisites

- Familiarity with exogenous features ([How to Use Exogenous Features](exogenous-features.md))
- Understanding of the three transformer kinds ([Transformer Kinds](../explanation/transformer-kinds.md))

<!-- COMPANION_NOTEBOOKS -->


A forecaster turns each exogenous column into a block of step columns: `temp` becomes `temp_step_1` through `temp_step_H`, one per horizon step. The `step_transformer` slot applies a transformer to that block before it joins the design matrix, and the resulting columns carry no horizon index, so every per-step estimator sees them.

## Summarise a Block with StepAggregator

[`StepAggregator`](/pages/api/generated/yohou.preprocessing.StepAggregator/) replaces each block with one column per aggregation:

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster
from yohou.preprocessing import StepAggregator

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    step_transformer=StepAggregator(aggregations=("min", "max", "mean")),
)
forecaster.fit(y, forecasting_horizon=48, X_future=X_future)
# temp_step_1 .. temp_step_48  ->  temp_step_min, temp_step_max, temp_step_mean
```

The available aggregations are `"min"`, `"max"`, `"mean"`, `"std"`, and `"sum"`. The set is closed; for anything else, see [Apply an Arbitrary Reduction](#apply-an-arbitrary-reduction) below.

## Keep the Raw Steps as Well

The slot replaces the block rather than adding to it. To keep both, compose:

```python
from yohou.compose import FeatureUnion

step_transformer = FeatureUnion([
    ("raw", "passthrough"),
    ("agg", StepAggregator(aggregations=("mean",))),
])
```

The design matrix then carries `raw_temp_step_1 .. raw_temp_step_48` and `agg_temp_step_mean`, since `FeatureUnion` prefixes each branch's output with the branch name.

Compose in parallel, not in sequence. Chaining two step transformers in a `FeaturePipeline` does not work, and the library says so rather than failing quietly: the first stage's output is horizon-agnostic by construction, so the second finds no `{base}_step_{h}` blocks left to reduce and raises. If you want two reductions of the same raw block, put them in a `FeatureUnion` as above.

## Use Different Aggregations per Variable

Route by column name with a [`ColumnTransformer`](/pages/api/generated/yohou.compose.ColumnTransformer/). Event indicators usually want a count; weather usually wants extremes. Each branch takes an explicit list of the step columns it handles:

```python
from yohou.compose import ColumnTransformer

step_transformer = ColumnTransformer([
    ("weather", StepAggregator(aggregations=("min", "max")),
     [f"temp_step_{h}" for h in range(1, 49)]),
    ("events", StepAggregator(aggregations=("sum",)),
     [f"holiday_step_{h}" for h in range(1, 49)]),
])
# -> weather_temp_step_min, weather_temp_step_max, events_holiday_step_sum
```

Two things to note. Column selection is by name, so the lists must span the horizon you fitted with; a comprehension over `range(1, H + 1)` is the readable way to write that. And `ColumnTransformer` prefixes each branch's output with the branch name, which is why the columns above read `weather_temp_step_min` rather than `temp_step_min`. The prefix does not disturb anything: the name still ends in a non-numeric suffix, so it is still classified as horizon-agnostic.

Step column names are unique across `X_future` and `X_forecast`, because a collision between the two sources is refused at fit, so selecting by name is always unambiguous.

## Learn a Reduction Instead of Fixing One

[`StepColumnReducer`](/pages/api/generated/yohou.preprocessing.StepColumnReducer/) lifts any scikit-learn transformer onto the step axis, fitting one clone per variable. Each variable's block becomes an `(n_observations, H)` table:

```python
from sklearn.decomposition import PCA
from yohou.preprocessing import StepColumnReducer

step_transformer = StepColumnReducer(reducer=PCA(n_components=3))
# temp_step_1 .. temp_step_48  ->  temp_step_c0, temp_step_c1, temp_step_c2
```

A wrapped `StandardScaler` standardises each step position independently, which preserves the shape of the horizon profile. A wrapped reducer compresses each variable's profile on its own terms, without mixing variables.

[`StepFrameReducer`](/pages/api/generated/yohou.preprocessing.StepFrameReducer/) instead fits one estimator over every step column of every variable, which captures structure across correlated channels. It needs a `prefix`, because its output describes no single variable:

```python
from yohou.preprocessing import StepFrameReducer

step_transformer = StepFrameReducer(reducer=PCA(n_components=4), prefix="wx")
# every step column  ->  wx_step_c0 .. wx_step_c3
```

Choose per-variable when you want to keep track of which variable a feature came from, and whole-frame when the variables are correlated and you care more about total width.

### Fixed Output Width Is Required

Both wrappers refuse an inner estimator whose output width depends on the data, such as `PCA(n_components=0.95)` or `PCA(n_components=None)`. Under panel data a forecaster fits one step transformer per group and derives a single column schema from the first group, so groups producing different widths would break. Pass a positive integer.

## Apply an Arbitrary Reduction

Anything outside `StepAggregator`'s vocabulary goes through `StepColumnReducer` with a `FunctionTransformer`, so there is one extension mechanism rather than two:

```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer

p90 = FunctionTransformer(lambda a: np.percentile(a, 90, axis=1, keepdims=True))
step_transformer = StepColumnReducer(reducer=p90)
```

## Handle Partial Coverage

A block is partially covered when the forecast data does not reach the full horizon, which happens routinely: the newest usable vintage may be older than the observation point. The missing steps arrive as nulls, and a warning at fit reports how far the coverage reached.

`StepAggregator` takes a `null_policy`. `"ignore"` (the default) summarises over the steps that carry a value; `"propagate"` yields null for any row with a missing step:

```python
StepAggregator(aggregations=("mean",), null_policy="ignore")
```

Be aware of what `"ignore"` means: a mean over 12 steps and a mean over 48 steps are different quantities sharing a column name, and the model cannot tell them apart. When coverage varies, turn on the companion column so it can:

```python
StepAggregator(aggregations=("mean",), emit_coverage=True)
# adds temp_step_n_covered, the number of steps that contributed to each row
```

It is off by default because under full coverage it is a constant column.

The wrappers have no null policy of their own, since that decision belongs to the inner estimator. If the inner estimator cannot handle missing values, the fit raises with the variable named and its coverage reported. Compose imputation in to fix it:

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

step_transformer = StepColumnReducer(
    reducer=Pipeline([("impute", SimpleImputer()), ("reduce", PCA(n_components=3))])
)
```

## Panel Data

Under `panel_strategy="global"`, each group gets its own fitted step transformer. One consequence is worth knowing before you rely on it: a **global** (unprefixed) exogenous column is folded into every group's data and comes back per group, so `holiday` becomes `A__holiday_step_mean` and `B__holiday_step_mean` rather than a single shared column.

For arithmetic aggregation this is invisible, since every copy holds the same number. For a fitted reduction it is not: each group's estimator is fitted on that group's data, so the same shared input yields different values per group. That is a defensible modelling choice, per-group normalisation of a shared signal, but it is a choice you are making. It matches how the `forecast_transformer` slot already behaves.

`StepFrameReducer` goes further and blends global columns into its components, so a shared channel cannot be recovered downstream at all. Use `StepColumnReducer` when that matters.

## Related Diagnostics

Reducing a block silences the rank-deficiency warning for that variable, because the warning is keyed on whether the original step columns still reach the estimator. This holds for any step transformer, including one that only rescales: the diagnostic cannot tell whether the redundancy was actually removed, only that the block was consumed. Forecasters without a `step_transformer` are unaffected.

The coverage warning is unaffected either way. It is measured before the transform runs, and it reports the measurement without recommending an estimator, so it will not point you at the options on this page.

## See Also

- [Transformer Kinds](../explanation/transformer-kinds.md) for why the step frame is its own kind
- [How to Transform Features on the Forecast Channel](transform-forecast-features.md) for transforming `X_forecast` before step columns are derived
- [How to Use Exogenous Features](exogenous-features.md) for the `X_actual`, `X_future`, and `X_forecast` channels
