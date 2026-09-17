# How to Use Preprocessing Transformers

This guide shows you how to prepare features for a forecasting model using Yohou's preprocessing transformers. Use these when you need to create lag features, compute rolling or seasonal statistics, give each forecast step its own seasonal features, scale values, wrap custom logic, or apply different transformations to different columns.

## Prerequisites

- Familiarity with the fit/predict API ([Getting Started](../tutorials/getting-started.md))
- Understanding of feature pipelines ([Feature Pipelines](../explanation/feature-pipelines.md))

<!-- COMPANION_NOTEBOOKS -->

## Create Lag Features with LagTransformer

[`LagTransformer`](/pages/api/generated/yohou.preprocessing.LagTransformer/) creates lagged copies of each value column, producing autoregressive inputs for a forecaster. Output columns follow the pattern `{col}_lag_{k}`:

```python
from yohou.preprocessing import LagTransformer

lags = LagTransformer(lag=[1, 3, 6, 12])
lags.fit(y_train)
y_lagged = lags.transform(y_train)
```

The transformer's `observation_horizon` equals the largest lag, since that many past rows are needed to produce a complete output:

```python
print(lags.observation_horizon)  # 12
```

If your series has a strong seasonal pattern, average several seasonal multiples of a lag by lagging once and taking a seasonal rolling mean (see [Compute Rolling Statistics](#compute-rolling-statistics)):

```python
from yohou.compose import FeaturePipeline
from yohou.preprocessing import LagTransformer, RollingStatisticsTransformer

# Average lags 12, 24, 36 (3 yearly cycles for monthly data)
mean_lags = FeaturePipeline([
    ("lag", LagTransformer(lag=12)),
    ("mean", RollingStatisticsTransformer(window_size=3, seasonality=12)),
])
```

This replaces `MeanLagTransformer(lag=12, n_lags=3)`, which was removed, and accepts any other statistic in place of the mean.

## Compute Rolling Statistics

[`RollingStatisticsTransformer`](/pages/api/generated/yohou.preprocessing.RollingStatisticsTransformer/) computes rolling aggregates over a sliding window. Available statistics: `mean`, `std`, `min`, `max`, `median`, `sum`, `var`, `q25`, `q75`:

```python
from yohou.preprocessing import RollingStatisticsTransformer

rolling = RollingStatisticsTransformer(
    window_size=12, statistics=["mean", "std"]
)
rolling.fit(y_train)
y_rolled = rolling.transform(y_train)
```

Output columns follow the pattern `{col}_{statistic}` (e.g., `value_mean`, `value_std`). The first `window_size - 1` rows are dropped because they contain incomplete windows:

```python
print(rolling.observation_horizon)  # 11  (window_size - 1)
```

To summarise the same position in past seasons rather than consecutive rows, set `seasonality`. The window then holds `window_size` values spaced `seasonality` rows apart, and output columns carry the season length (`{col}_s{seasonality}_{statistic}`):

```python
# Mean and spread of the same hour over the last 7 days, for hourly data
same_hour = RollingStatisticsTransformer(
    window_size=7, seasonality=24, statistics=["mean", "std"]
)
same_hour.fit(y_train)
print(same_hour.get_feature_names_out())  # ['value_s24_mean', 'value_s24_std']
print(same_hour.observation_horizon)      # 144  ((window_size - 1) * seasonality)
```

For custom aggregation logic, use [`SlidingWindowFunctionTransformer`](/pages/api/generated/yohou.preprocessing.SlidingWindowFunctionTransformer/) with any callable:

```python
import numpy as np
from yohou.preprocessing import SlidingWindowFunctionTransformer

# Coefficient of variation over a 7-step window
cv = SlidingWindowFunctionTransformer(
    func=lambda x: np.std(x) / np.mean(x), window_size=7
)
```

## Give Each Forecast Step Its Seasonal Statistics

A reduction forecaster builds one feature row at the forecast origin and uses it for every step, so a seasonal statistic computed at the origin describes the origin's hour, not the hour each step predicts. [`HorizonRollingStatisticsTransformer`](/pages/api/generated/yohou.preprocessing.HorizonRollingStatisticsTransformer/) instead emits one column per step, `{col}_s{seasonality}_{statistic}_step_{h}`, holding the statistic over the most recent `n_seasons` values at that step's position in the season. With `statistics="mean"` the columns equal a [`MeanSeasonalNaive`](/pages/api/generated/yohou.point.MeanSeasonalNaive/) forecast, step for step.

Put it in the forecaster's `actual_transformer` and use the `"direct"` strategy with `step_feature_alignment="matched"`, so the model for step `h` reads only its own step's columns:

```python
from sklearn.linear_model import Ridge
from yohou.compose import FeatureUnion
from yohou.point import PointReductionForecaster
from yohou.preprocessing import HorizonRollingStatisticsTransformer, LagTransformer

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    actual_transformer=FeatureUnion([
        ("lags", LagTransformer(lag=[1, 24])),
        ("seasonal", HorizonRollingStatisticsTransformer(
            seasonality=24, n_seasons=7, statistics=["mean", "std"]
        )),
    ]),
    reduction_strategy="direct",
    step_feature_alignment="matched",
)
forecaster.fit(y_train, forecasting_horizon=48)
```

You do not pass the horizon to the transformer: the forecaster routes its own `forecasting_horizon` to it as fit metadata, so the two always agree. To fit the transformer on its own, call `fit(X, forecasting_horizon=48)`.

Keep these rules in mind:

- **Feed it the series unlagged.** Lagging its input by `d` rows shifts the position every step describes by `d`, so step 4 would carry another hour's profile. Only a lag that is a whole number of seasons (24, 48, ... for a daily season) is safe.
- **Make it the last step of a pipeline.** A transformer after it would rename or shift its step columns; the forecaster rejects a fit where the declared step columns are missing.
- **Refit to forecast further.** Its columns cover the fit horizon only, so predicting beyond that horizon raises.

With `reduction_strategy="multi-output"` or `step_feature_alignment="all"`, every model receives all `H` step columns, and fit warns about it.

## Scale and Normalize Values

Use [`StandardScaler`](/pages/api/generated/yohou.preprocessing.StandardScaler/) to normalize value columns to zero mean and unit variance. Yohou's native scaler wrappers work directly with polars DataFrames, preserving the `"time"` column automatically:

```python
from yohou.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(y_train)
y_scaled = scaler.transform(y_train)
```

Other built-in scalers: [`MinMaxScaler`](/pages/api/generated/yohou.preprocessing.MinMaxScaler/), [`RobustScaler`](/pages/api/generated/yohou.preprocessing.RobustScaler/), [`MaxAbsScaler`](/pages/api/generated/yohou.preprocessing.MaxAbsScaler/). All support `inverse_transform` for reversing the scaling during prediction.

If you need an sklearn transformer that doesn't have a native wrapper (e.g., a custom encoder), use [`SklearnTransformer`](/pages/api/generated/yohou.preprocessing.SklearnTransformer/) to adapt it:

```python
from sklearn.preprocessing import KBinsDiscretizer
from yohou.preprocessing import SklearnTransformer

discretizer = SklearnTransformer(
    transformer=KBinsDiscretizer, n_bins=5, encode="ordinal"
)
```

## Wrap Custom Functions with FunctionTransformer

[`FunctionTransformer`](/pages/api/generated/yohou.preprocessing.FunctionTransformer/) wraps a plain Python function into a transformer that works inside a pipeline:

```python
import polars as pl
from yohou.preprocessing import FunctionTransformer

def log_transform(df):
    return df.with_columns(pl.all().exclude("time").log())

def exp_transform(df):
    return df.with_columns(pl.all().exclude("time").exp())

transformer = FunctionTransformer(func=log_transform, inverse_func=exp_transform)
transformer.fit(y_train)
y_log = transformer.transform(y_train)
```

Providing `inverse_func` lets target transformers reverse the operation during prediction. If the function is not invertible, omit it.

## Select Columns with ColumnTransformer

[`ColumnTransformer`](/pages/api/generated/yohou.compose.ColumnTransformer/) applies different transformers to different column subsets. Use this when a multivariate series needs distinct treatment per column:

```python
from yohou.compose import ColumnTransformer
from yohou.preprocessing import LagTransformer, RollingStatisticsTransformer

ct = ColumnTransformer(
    transformers=[
        ("lags", LagTransformer(lag=[1, 2, 3]), ["temperature"]),
        ("rolling", RollingStatisticsTransformer(window_size=7), ["humidity"]),
    ],
    remainder="drop",
)

ct.fit(y_train)
y_features = ct.transform(y_train)
```

Set `remainder="passthrough"` to keep columns not assigned to any transformer in the output.

## See Also

- [How to Compose Feature Pipelines](compose-feature-pipelines.md) for chaining transformers sequentially and in parallel
- [How to Clean and Resample Time Series](clean-and-resample.md) for data preparation before feature engineering
- [Reduction Forecasting](../explanation/reduction-forecasting.md) for how step columns reach per-step models
- [Preprocessing API Reference](/pages/api/preprocessing/) for full parameter documentation
