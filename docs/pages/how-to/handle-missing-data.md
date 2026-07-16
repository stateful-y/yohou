# How to Handle Missing Data

Time series often contain gaps from sensor outages, market holidays, or
irregular reporting schedules. This guide shows you how to fill those gaps before
passing data through a forecasting pipeline.

## Prerequisites

- Familiarity with the pipeline API ([Getting Started](../tutorials/getting-started.md))
- Understanding of stateful transformers ([Preprocessing](../explanation/preprocessing.md))

<!-- COMPANION_NOTEBOOKS -->

## Place Imputation Before Stateful Transformers

Imputation belongs **before** stateful transformers such as
[`LagTransformer`](/pages/api/generated/yohou.preprocessing.LagTransformer/) or
[`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.SeasonalDifferencing/).
Stateful transformers maintain look-back windows, so if a gap reaches them they
propagate `null` values through the window and the regressor receives incomplete
feature rows. Place imputation first in the pipeline to ensure every downstream
component sees complete data.

```python
from yohou.compose import FeaturePipeline
from yohou.preprocessing.imputation import SimpleTimeImputer
from yohou.stationarity import SeasonalDifferencing
from yohou.preprocessing import LagTransformer

pipeline = FeaturePipeline([
    ("impute", SimpleTimeImputer(method="linear")),
    ("diff", SeasonalDifferencing(seasonality=7)),
    ("lags", LagTransformer(lag=[1, 2, 7])),
])
```

## Interpolate or Fill with SimpleTimeImputer

[`SimpleTimeImputer`](/pages/api/generated/yohou.preprocessing.SimpleTimeImputer/)
offers time-aware gap-filling methods.

If your series is smooth and continuously varying (temperature, price), use
linear interpolation:

```python
from yohou.preprocessing.imputation import SimpleTimeImputer

imputer = SimpleTimeImputer(method="linear")
```

If your values persist at their last reading until explicitly updated (inventory
levels, status flags), use forward fill:

```python
imputer = SimpleTimeImputer(method="forward")
```

If you need to handle both leading and trailing gaps, use `"fill_both"` which
applies a forward fill followed by a backward fill:

```python
imputer = SimpleTimeImputer(method="fill_both")
```

### Limit Consecutive Fills

When gaps span long stretches, unbounded filling can mask data quality issues.
Set `limit` to cap the number of consecutive `null` values that get filled,
leaving longer gaps as `null` for downstream inspection:

```python
imputer = SimpleTimeImputer(method="forward", limit=3)
```

## Fill with a Global Statistic

When temporal order does not matter and you just need a fast baseline, use
[`SimpleImputer`](/pages/api/generated/yohou.preprocessing.SimpleImputer/),
a wrapper around sklearn's imputer. It replaces gaps with a column-wide
statistic (mean, median, most frequent, or a constant) and ignores the time
axis entirely:

```python
from yohou.preprocessing.imputation import SimpleImputer

imputer = SimpleImputer(strategy="mean")
```

Prefer this only when the series has no trend or seasonality to exploit;
otherwise the time-aware methods above produce more faithful fills.

## Fill from Seasonal Patterns with SeasonalImputer

[`SeasonalImputer`](/pages/api/generated/yohou.preprocessing.SeasonalImputer/)
borrows from the same position in adjacent cycles rather than drawing a line
between neighbouring observations. This is appropriate when the series has a
pronounced seasonal shape and the gaps fall at a predictable point in the cycle
(for example, weekends in a daily series with a strong weekly pattern).

Set `period` to the cycle length. If you prefer robustness to outliers, switch
`fill_method` to `"seasonal_median"`:

```python
from yohou.preprocessing.imputation import SeasonalImputer

# Weekly seasonality, median aggregation
imputer = SeasonalImputer(period=7, fill_method="seasonal_median")
```

## Use KNN Imputation for Complex Patterns

[`TransformedSpaceKNNImputer`](/pages/api/generated/yohou.preprocessing.TransformedSpaceKNNImputer/)
performs nearest-neighbour imputation and is appropriate when the series has
complex structure that neither interpolation nor seasonal borrowing captures
well. It is also the most computationally expensive, so prefer the simpler
strategies unless they leave visible imputation artefacts.

Pass a transformer to define the feature space used for neighbour search.
For example, a `LagTransformer` makes neighbours lag-feature vectors
(temporally similar windows) rather than individual time points:

```python
from yohou.preprocessing import LagTransformer
from yohou.preprocessing.imputation import TransformedSpaceKNNImputer

imputer = TransformedSpaceKNNImputer(
    n_neighbors=5,
    weights="distance",
    transformer=LagTransformer(lag=[1, 2, 3]),
)
```

## Apply Custom Imputation Logic

For domain-specific rules, wrap any callable in
[`FunctionTransformer`](/pages/api/generated/yohou.preprocessing.FunctionTransformer/)
and place it at the same position in the pipeline:

`FunctionTransformer` strips the `time` column before calling the function
and reattaches it afterwards, so operate only on the data columns:

```python
import polars as pl
from yohou.preprocessing.function import FunctionTransformer

def clamp_and_fill(df: pl.DataFrame) -> pl.DataFrame:
    """Forward-fill, then replace remaining nulls with column medians."""
    return df.select(pl.all().forward_fill().fill_null(pl.all().median()))

imputer = FunctionTransformer(func=clamp_and_fill)
```

## Skip NaN Instances During Training

When your estimator cannot handle NaN natively (e.g. `LinearRegression`,
`Ridge`, `SVR`), you can skip the imputation step entirely and let the
reduction forecaster drop any training row that contains NaN:

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    nan_handling="drop",
)
forecaster.fit(y=y_with_gaps, forecasting_horizon=3)
```

The forecaster emits a warning reporting how many rows were removed.
If the gaps are sparse, this is often simpler than building an imputation
pipeline and avoids introducing imputation artifacts into the training signal.

For tree-based estimators that handle NaN natively (LightGBM, XGBoost,
CatBoost, `HistGradientBoostingRegressor`), keep the default
`nan_handling="pass"` and let the estimator learn split decisions around
missing values directly:

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from yohou.point import PointReductionForecaster

forecaster = PointReductionForecaster(
    estimator=HistGradientBoostingRegressor(),
    nan_handling="pass",  # default, tree handles NaN internally
)
forecaster.fit(y=y_with_gaps, forecasting_horizon=3)
```

!!! tip
    Use `nan_handling="drop"` as a quick baseline when gaps are rare.
    Switch to a proper imputation transformer (see sections above) when
    you need to preserve every training sample or when the gap pattern
    is systematic.

## See Also

- [Handle Outliers](handle-outliers.md) for detecting and treating anomalous values
  before or after imputation.
- [Preprocessing](../explanation/preprocessing.md) for the conceptual model of
  stateful vs. stateless transformers and how the pipeline contract works.
- [Compose Feature Pipelines](compose-feature-pipelines.md) for combining
  imputers with other transformers.
- [Use Preprocessing Transformers](use-preprocessing-transformers.md) for
  `FunctionTransformer`, sklearn scalers, and window transformers.
