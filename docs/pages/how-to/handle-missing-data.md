# How to Handle Missing Data

Time series often contain gaps from sensor outages, market holidays, or
irregular reporting schedules. This guide shows you how to fill those gaps before
passing data through a forecasting pipeline.

## Prerequisites

- Familiarity with the pipeline API ([Getting Started](../tutorials/getting-started.md))
- Understanding of stateful transformers ([Preprocessing](../explanation/preprocessing.md))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Place Imputation Before Stateful Transformers

Imputation belongs **before** stateful transformers such as
[`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/) or
[`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/).
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

[`SimpleTimeImputer`](/pages/api/generated/yohou.preprocessing.imputation.SimpleTimeImputer/)
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

## Fill from Seasonal Patterns with SeasonalImputer

[`SeasonalImputer`](/pages/api/generated/yohou.preprocessing.imputation.SeasonalImputer/)
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

[`TransformedSpaceKNNImputer`](/pages/api/generated/yohou.preprocessing.imputation.TransformedSpaceKNNImputer/)
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
[`FunctionTransformer`](/pages/api/generated/yohou.preprocessing.function.FunctionTransformer/)
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

## See Also

- [Handle Outliers](handle-outliers.md) for detecting and treating anomalous values
  before or after imputation.
- [Preprocessing](../explanation/preprocessing.md) for the conceptual model of
  stateful vs. stateless transformers and how the pipeline contract works.
- [Compose Feature Pipelines](compose-feature-pipelines.md) for combining
  imputers with other transformers.
- [Use Preprocessing Transformers](use-preprocessing-transformers.md) for
  `FunctionTransformer`, sklearn scalers, and window transformers.
