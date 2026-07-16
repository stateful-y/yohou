# How to Transform Features on the Forecast Channel

This guide shows you how to apply transformers to an `X_forecast` frame, using [`PerVintageActualTransformer`](/pages/api/generated/yohou.compose.PerVintageActualTransformer/). Use this when your exogenous features arrive as *forecasts* (carrying `vintage_time` and `time`) and you need to derive features from them, for example computing net load from load, wind, and solar forecasts before feeding them to a forecaster's `X_forecast` channel.

## Prerequisites

- Familiarity with transformers ([How to Use Preprocessing Transformers](use-preprocessing-transformers.md))
- Understanding of forecast vintages ([How to Work with Forecast Vintages](forecast-vintages.md))

<!-- COMPANION_NOTEBOOKS -->


An `X_forecast` frame carries two time axes, `vintage_time` and `time`, so an ordinary single-axis transformer cannot consume it. [`PerVintageActualTransformer`](/pages/api/generated/yohou.compose.PerVintageActualTransformer/) wraps an actual transformer and fits and applies it to each vintage independently. For why the two kinds exist, see [Transformer Kinds](../explanation/transformer-kinds.md).

## Derive a Feature per Vintage

Wrap any actual transformer. Here a [`FunctionTransformer`](/pages/api/generated/yohou.preprocessing.FunctionTransformer/) computes net load from load and wind forecasts:

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

## Fit a Transformer per Vintage

`net_load` learns nothing, so per-vintage handling only keeps its rows aligned. The wrapper earns its keep when the inner *fits* from data: each vintage is fitted on its own rows, so a scaler standardizes each forecast trajectory to its own scale, and an imputer fills each vintage's gaps from that vintage's own values.

```python
from yohou.preprocessing import SimpleImputer

# each vintage's missing values are filled with that vintage's own mean
impute = PerVintageActualTransformer(SimpleImputer(strategy="mean"))
X_forecast_t = impute.fit_transform(X_forecast)
```

This is leakage-free: a vintage's output depends only on its own rows, all of which are known at that vintage's `vintage_time`. It is a different operation from normalizing `X_forecast` against the *training* distribution; for that, put the scaler in the estimator pipeline (`estimator=make_pipeline(StandardScaler(), model)`) so it fits on the tabularized training matrix instead.

!!! note "Per-vintage fits are small"
    A vintage spans roughly the forecasting horizon, so a per-vintage fit sees few rows. This is fine for imputation and scaling; a transformer needing many points to fit well (a high-resolution quantile transform, say) is a poor fit for the per-vintage axis.

!!! warning "Per-vintage fitting is slow on large frames"
    Fitting once per vintage costs far more than a single shared fit: roughly 640 times slower on a 500-vintage, 12000-row frame (0.76s against 0.001s). The cost is the per-vintage revalidation, not the arithmetic. Transform an `X_forecast` frame once and reuse the result rather than calling this inside a backtest loop.

!!! note "The tail of a forecast frame is dropped"
    A forecast frame usually ends in vintages of one or two rows, where the horizon runs off the end of the series. A single-row vintage has no per-vintage statistic to compute, so `PerVintageActualTransformer` drops vintages with fewer than two rows and emits a `UserWarning` naming how many. The surviving vintages are unaffected.

!!! note "A stateful transformer costs you the earliest steps"
    A stateful wrapped transformer (a lag, a difference, a rolling window) is supported: each vintage is fitted on its own rows, so its history never reaches across a vintage boundary. It does consume its `observation_horizon` from the **start** of every vintage, and those are the nearest-term forecast steps, usually the ones you care about most. If it consumes the whole vintage you get a `ValueError` rather than an empty frame.

    The loss spreads. A [`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/) keeps only the index rows common to every branch, so one lag branch truncates its siblings' earliest steps too.

!!! warning "A lifted lag is within a vintage, not across vintages"
    `PerVintageActualTransformer(LagTransformer(lag=1))` gives you the forecast for step `h-1` **issued at the same `vintage_time`**: the ramp along one forecast trajectory. It does **not** give you the previous vintage's forecast for the same `time`. That is a genuinely different operation, and it needs its own [`BaseForecastTransformer`](/pages/api/generated/yohou.base.BaseForecastTransformer/) rather than a lifted actual transformer.

## Compose Several Forecast Transformers

The composition estimators accept forecast transformers too. A [`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/) of forecast transformers is itself forecast-kind and aligns its branches on `(vintage_time, time)`:

```python
from yohou.compose import FeatureUnion

features = FeatureUnion([
    ("net_load", net_load),
    ("wind_share", PerVintageActualTransformer(
        FunctionTransformer(
            func=lambda df: df.select(
                (pl.col("wind") / pl.col("load")).alias("wind_share")
            ),
            feature_names_out=lambda self, names: ["wind_share"],
        )
    )),
])
```

Each branch is lifted independently, so a branch may be stateful. A branch computing a ramp (`pl.col("load").diff()`) works, but `FunctionTransformer` measures the `.diff()` as an `observation_horizon` of 1, so that branch drops each vintage's first step, and the union drops it from every other branch too.

A composition must be **homogeneous in kind**: mixing actual and forecast transformers in one `FeatureUnion`, `FeaturePipeline`, or `ColumnTransformer` raises a `ValueError`.

Equivalently, you can *compose-then-lift*, building a union of actual transformers and lifting the whole thing once:

```python
PerVintageActualTransformer(FeatureUnion([
    ("net_load", FunctionTransformer(func=..., feature_names_out=...)),
    ("wind_share", FunctionTransformer(func=..., feature_names_out=...)),
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
