# How to Transform Features on the Forecast Channel

This guide shows you how to apply transformers to an `X_forecast` frame, using [`PerVintageActualTransformer`](/pages/api/generated/yohou.compose.per_vintage.PerVintageActualTransformer/). Use this when your exogenous features arrive as *forecasts* (carrying `vintage_time` and `time`) and you need to derive features from them, for example computing net load from load, wind, and solar forecasts before feeding them to a forecaster's `X_forecast` channel.

## Prerequisites

- Familiarity with transformers ([How to Use Preprocessing Transformers](use-preprocessing-transformers.md))
- Understanding of forecast vintages ([How to Work with Forecast Vintages](forecast-vintages.md))

## Why a Forecast Transformer

A regular transformer (a [`BaseActualTransformer`](/pages/api/generated/yohou.base.transformer.BaseActualTransformer/)) operates on a single-axis frame with one `"time"` column. An `X_forecast` frame carries **two** time axes, `vintage_time` (when the forecast was issued) and `time` (what it forecasts), so it is a stack of short per-vintage series that a single-axis transformer cannot consume.

[`PerVintageActualTransformer`](/pages/api/generated/yohou.compose.per_vintage.PerVintageActualTransformer/) bridges the gap: it wraps a **stateless** actual transformer and applies it to each vintage independently, so order-dependent operations never bleed across vintage boundaries.

## Derive a Feature per Vintage

Wrap any stateless transformer. Here a [`FunctionTransformer`](/pages/api/generated/yohou.preprocessing.function.FunctionTransformer/) computes net load from load and wind forecasts:

```python
import polars as pl
from yohou.preprocessing import FunctionTransformer
from yohou.compose import PerVintageActualTransformer

net_load = PerVintageActualTransformer(
    FunctionTransformer(
        func=lambda df: df.select((pl.col("load") - pl.col("wind")).alias("net_load")),
        feature_names_out=lambda self, names: ["net_load"],
    )
)

X_forecast_t = net_load.fit_transform(X_forecast)
# columns: ["vintage_time", "time", "net_load"]
```

The `vintage_time` and `time` index columns are preserved, and each vintage's `net_load` is computed from only that vintage's rows.

!!! note "The wrapped transformer must be stateless"
    `PerVintageActualTransformer` requires the wrapped transformer to measure `observation_horizon == 0` after fitting. A stateful transformer (a lag or rolling window) needs contiguous memory that the discontinuous vintage axis cannot provide, so it is rejected with a `ValueError`.

## Compose Several Forecast Transformers

The composition estimators accept forecast transformers too. A [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/) of forecast transformers is itself forecast-kind and aligns its branches on `(vintage_time, time)`:

```python
from yohou.compose import FeatureUnion

features = FeatureUnion([
    ("net_load", net_load),
    ("ramp", PerVintageActualTransformer(
        FunctionTransformer(
            func=lambda df: df.select(pl.col("load").diff().alias("ramp")),
            feature_names_out=lambda self, names: ["ramp"],
        )
    )),
])
```

A composition must be **homogeneous in kind**: mixing actual and forecast transformers in one `FeatureUnion`, `FeaturePipeline`, or `ColumnTransformer` raises a `ValueError`.

Equivalently, you can *compose-then-lift* — build a union of actual transformers and lift the whole thing once:

```python
PerVintageActualTransformer(FeatureUnion([
    ("net_load", FunctionTransformer(func=..., feature_names_out=...)),
    ("ramp", FunctionTransformer(func=..., feature_names_out=...)),
]))
```

## Feed the Result to a Forecaster

Pass the transformed frame through a forecaster's `X_forecast` channel:

```python
forecaster.fit(y_train, X_forecast=features.fit_transform(X_forecast))
```

Forecast transformers are for the `X_forecast` channel only. Passing one as a forecaster's `feature_transformer` (which processes the single-axis `X_actual`) raises an error, because that slot requires an actual-kind transformer.

## Related

- [How to Work with Forecast Vintages](forecast-vintages.md)
- [How to Compose Feature Pipelines](compose-feature-pipelines.md)
- [How to Use Exogenous Features](exogenous-features.md)
