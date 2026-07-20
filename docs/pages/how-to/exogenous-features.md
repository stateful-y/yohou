# How to Use Exogenous Features

This guide shows you how to pass external data (`X_actual`, `X_future`,
`X_forecast`) to forecasters and composition pipelines in production
scenarios.

## Prerequisites

- Yohou installed ([Installation](installation.md))
- Familiarity with the fit/predict lifecycle
  ([Exogenous Features Tutorial](../tutorials/exogenous-features.md))

<!-- COMPANION_NOTEBOOKS -->

## Classify Your Features

Before calling `fit()`, decide which parameter each feature belongs in.
See [About Exogenous Features](../explanation/exogenous-features.md) for
the full conceptual model.

| Question | Yes | No |
|---|---|---|
| Is it a measurement that can only be known after it happens? | `X_actual` | Continue |
| Can you compute it from the timestamp alone, with no external table? | `actual_transformer` | Continue |
| Does it come from an external model with an issuance time? | `X_forecast` | `X_future` |

The second question is the one that catches people. Day-of-week, hour of day,
and Fourier terms are all deterministic and all knowable for any future date,
which makes `X_future` look like the answer. It is not. Their value at the
observation point already determines their value at every horizon, so
`actual_transformer` hands the estimator the same information in a fraction of
the columns. Ask whether you need a *lookup table*, not whether the value is
*knowable*: a holiday calendar needs one, a day-of-week indicator does not.

If a feature is uncertain but has no vintage (a single "best guess"),
treat it as `X_future`. If you need multiple versions of that guess at
predict time, wrap it with a `vintage_time` column and use `X_forecast`.

The table routes the raw data. Features *derived* from that data are a separate
question, answered by the channel it landed in. A feature derived from an
`X_forecast` column (a wind-adjusted load, a scaled temperature, a ramp between
consecutive steps of one vintage) belongs in the forecaster's
`forecast_transformer` slot, not in caller code that transforms the frame before
passing it in. The slot makes the transform tunable through a search path, keeps
it attached across `clone`, and re-applies it on every `observe` and `rewind`
for you. See
[How to Transform Forecast Features](transform-forecast-features.md).

### Clock Features Versus Event Features

Both of these are calendar-related. Only one of them needs `X_future`.

```python
from yohou.compose import FeatureUnion
from yohou.point import PointReductionForecaster
from yohou.preprocessing import FourierFeatureTransformer, LagTransformer

forecaster = PointReductionForecaster(
    # Clock feature: weekly seasonality on hourly data. Computable from the
    # timestamp, so it belongs here and never touches X_future.
    actual_transformer=FeatureUnion([
        ("lags", LagTransformer([1, 2, 3])),
        ("weekly", FourierFeatureTransformer(seasonality=168.0, harmonics=[1, 2])),
    ]),
)

forecaster.fit(
    y=y_train,
    forecasting_horizon=24,
    # Event feature: not derivable from the timestamp, so it is windowed
    # forward into is_holiday_step_1 .. is_holiday_step_24.
    X_future=holiday_calendar,
)
```

Routing that Fourier block through `X_future` instead would window its four
columns into 96 step columns spanning the same four dimensions, and leave the
model's predictions unchanged.

## Pass Exogenous Features to a Forecaster

Supply any combination of the three parameters to `fit()` on a
[`PointReductionForecaster`](/pages/api/generated/yohou.point.PointReductionForecaster/).
At predict time, only `X_future` and `X_forecast` are accepted because
`X_actual` comes from the forecaster's stored observation window.

```python
from sklearn.ensemble import HistGradientBoostingRegressor

from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer

forecaster = PointReductionForecaster(
    estimator=HistGradientBoostingRegressor(),
    actual_transformer=LagTransformer([1, 2, 3]),
    reduction_strategy="direct",
)

forecaster.fit(
    y=y_train,
    X_actual=temperature,       # observation features (lagged internally)
    forecasting_horizon=24,
    X_future=holidays,          # event calendar, needs a lookup table
    X_forecast=weather_forecast, # vintage-indexed external predictions
)

pred = forecaster.predict(X_future=holidays, X_forecast=weather_forecast)
```

## Choose a Step Feature Alignment

When using the `"direct"` reduction strategy, `step_feature_alignment`
controls which step columns each horizon's estimator sees:

- `"all"` (default): every estimator sees all step columns
- `"matched"`: each estimator sees only the step column for its horizon
- `"cumulative"`: estimator for step $h$ sees step columns $1$ through $h$

```python
forecaster = PointReductionForecaster(
    estimator=HistGradientBoostingRegressor(),
    actual_transformer=LagTransformer([1, 2, 3]),
    reduction_strategy="direct",
    step_feature_alignment="matched",
)
```

If your `X_future` or `X_forecast` columns evolve meaningfully across steps
(e.g., temperature forecasts degrade with horizon), `"matched"` or
`"cumulative"` can reduce noise from distant step columns.

## Use Composition Forecasters

### [`ColumnForecaster`](/pages/api/generated/yohou.compose.ColumnForecaster/)

Each child forecaster receives all three exogenous parameters. Children that
don't use exogenous features ignore them:

```python
from yohou.compose import ColumnForecaster

forecaster = ColumnForecaster(
    forecasters=[
        ("demand", demand_forecaster),
        ("supply", supply_forecaster),
    ],
)

forecaster.fit(
    y=y_panel,
    X_actual=actuals,
    forecasting_horizon=24,
    X_future=holidays,
    X_forecast=weather,
)
```

### [`DecompositionPipeline`](/pages/api/generated/yohou.compose.DecompositionPipeline/)

All three parameters pass through to the residual forecaster after trend
and seasonality removal:

```python
from sklearn.ensemble import HistGradientBoostingRegressor

from yohou.compose import DecompositionPipeline
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer
from yohou.stationarity import PolynomialTrendForecaster

pipeline = DecompositionPipeline(
    forecasters=[
        ("trend", PolynomialTrendForecaster(degree=1)),
        ("residual", PointReductionForecaster(
            estimator=HistGradientBoostingRegressor(),
            actual_transformer=LagTransformer([1, 2, 3]),
            reduction_strategy="direct",
        )),
    ],
)

pipeline.fit(
    y=y_train,
    X_actual=X_actual_train,
    forecasting_horizon=H,
    X_future=holidays,
    X_forecast=weather,
)
```

### [`ForecastedFeatureForecaster`](/pages/api/generated/yohou.compose.ForecastedFeatureForecaster/)

Use `ForecastedFeatureForecaster` when you want Yohou to forecast the
exogenous feature itself. `X_actual` trains the feature forecaster (as its
target); its forecast then reaches the target forecaster through the
`X_forecast` channel as contemporaneous step columns. `X_future` passes
through to the target forecaster directly.

The `strategy` parameter controls the quality of the in-sample feature
forecast the target trains on: `"actual"` uses perfect-foresight (real)
values, while `"predicted"` and `"rewind"` use the feature forecaster's
rolling predictions so the target learns from inputs similar to what it sees
at predict time.

```python
from yohou.compose import ForecastedFeatureForecaster

fff = ForecastedFeatureForecaster(
    target_forecaster=price_forecaster,
    feature_forecaster=temperature_forecaster,
    strategy="rewind",
)

fff.fit(
    y=y_train,
    X_actual=X_actual_train,
    forecasting_horizon=H,
    X_future=holidays,
)

pred = fff.predict(X_future=holidays)
```

At predict time the feature forecaster runs first to forecast the exogenous
features, and that forecast is passed to the target forecaster as `X_forecast`.
Note that `observe` and `rewind` require `X_actual`, since the feature forecaster
needs new feature observations to advance in step with the target. See
[About Exogenous Features](../explanation/exogenous-features.md) for how the
`X_forecast` step columns and predict-time override work internally.

### Refresh the feature forecast less often than you predict

If the feature forecaster is expensive and you cannot re-run it every step in
production (for example you refresh it daily while predicting hourly), set
`feature_stride` to that refresh cadence. The feature forecast is then regenerated
every `feature_stride` steps and reused in between, both at fit and at serve, so the
target trains on features of the same age it sees in production:

```python
fff = ForecastedFeatureForecaster(
    target_forecaster=price_forecaster,
    feature_forecaster=temperature_forecaster,
    strategy="rewind",
    feature_stride=24,  # regenerate the feature forecast every 24 steps
)
fff.fit(y=y_train, X_actual=X_actual_train, forecasting_horizon=H)

# Walk forward: observe_predict regenerates the feature forecast every 24 steps
results = fff.observe_predict(y=y_test, X_actual=X_actual_test, stride=H)
```

`feature_stride` takes effect only through `observe_predict` (a bare `predict`
always produces a single fresh forecast). The default `feature_stride=1` regenerates
the forecast at every step.

## Update Observations with Exogenous Data

In a walk-forward loop, `observe_predict()` atomically observes new data
and produces the next forecast. Pass `X_actual` so the forecaster's
observation window stays current:

```python
results = forecaster.observe_predict(
    y=y_test,
    X_actual=X_actual_test,
    X_future=holidays_test,
    X_forecast=weather_test,
    stride=1,  # one forecast per time step
)
```

If you need finer control, call `observe()` and `predict()` separately:

```python
forecaster.observe(y=y_new, X_actual=X_actual_new)
pred = forecaster.predict(X_future=holidays_new, X_forecast=weather_new)
```

## As-of Vintage Selection

`X_forecast` uses **as-of (backward) matching**: for each observation time
$T$, the forecaster selects the latest vintage $V$ where $V \leq T$, then
extracts forecast values at $T + 1 \cdot \Delta t$ through
$T + H \cdot \Delta t$ from that vintage's rows. This means vintage times
do not need to align exactly with observation times.

### Sparse vintage schedules

External forecast providers often publish on a coarser schedule than your
observation frequency. For example, a weather model might issue forecasts
every 6 hours while you observe hourly. With as-of matching, each hourly
observation automatically picks up the most recent 6-hourly vintage:

```text
Vintages:     V0=00:00          V1=06:00          V2=12:00
              |                 |                 |
Observations: 00 01 02 03 04 05 06 07 08 09 10 11 12 ...
              ↑                 ↑
              uses V0            uses V1
```

Observation at 03:00 uses vintage V0 (00:00) because that is the latest
vintage at or before 03:00. Observation at 09:00 uses vintage V1 (06:00).

### Step alignment

Step columns are always relative to the **observation time**, not the
vintage time. For observation $T$ with a matched vintage $V$:

- `step_1` = forecast value at $T + 1 \cdot \Delta t$
- `step_2` = forecast value at $T + 2 \cdot \Delta t$
- ...
- `step_H` = forecast value at $T + H \cdot \Delta t$

If the vintage does not cover a particular target time (because the
forecast did not extend that far), the corresponding step column is null.

### Null step columns

Null step columns are expected in two situations:

1. **No vintage available**: the observation time is before all vintage
   times in `X_forecast`. All step columns are null for that row.
2. **Partial coverage**: the matched vintage's forecast horizon does not
   reach $T + h \cdot \Delta t$. Later step columns are null.

Tree-based estimators (XGBoost, LightGBM, HistGradientBoosting) handle
null features natively. For estimators that require complete data, set
`nan_handling="drop"` so rows with null step features are excluded from
training.

## Pickle and Restore

The three-parameter state (step column names, observation window) survives
pickle round-trips:

```python
import pickle

with open("forecaster.pkl", "wb") as f:
    pickle.dump(forecaster, f)

with open("forecaster.pkl", "rb") as f:
    restored = pickle.load(f)

# Multi-vintage predictions still work
pred = restored.predict(X_forecast=new_vintage)
```

## Troubleshooting

**Problem: `ValueError` about column name collisions**
: `X_future` and `X_forecast` produce step columns with the same name. Rename
  your source columns so they don't collide after `_step_` suffixing.

**Problem: `X_actual` passed to `predict()`**
: `predict()` does not accept `X_actual`. The forecaster uses its stored
  observation window instead. Call `observe()` to update it with new actuals
  before predicting.

**Problem: step columns missing at predict time**
: All `X_future` and `X_forecast` columns seen during `fit()` must also be
  present at `predict()` time with the same names.

**Problem: `UserWarning` about X_forecast covering fewer steps than the horizon**
: The forecast vintage covers fewer future timestamps than `forecasting_horizon`.
  This is normal for short-range forecasts or when the observation point has
  advanced past some forecast timestamps (e.g., after `observe()`). The missing
  step columns are filled with null. Tree-based estimators (XGBoost, LightGBM,
  HistGradientBoosting) handle null features natively. For estimators that do
  not support nulls, set `nan_handling="drop"` so null rows are excluded from
  training, or provide forecasts with full horizon coverage.

## See Also

- [Work with Forecast Vintages](forecast-vintages.md): X_forecast preparation,
  multi-vintage prediction, and walk-forward evaluation
- [About Exogenous Features](../explanation/exogenous-features.md): design
  rationale and internal mechanics
- [Exogenous Features Tutorial](../tutorials/exogenous-features.md): hands-on
  introduction
- [`window_forecasts` API Reference](../api/utils.md): utility for as-of
  vintage matching with step alignment
