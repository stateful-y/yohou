# How to Work with Forecast Vintages

This guide shows you how to prepare, align, and predict with `X_forecast`
(external model outputs stamped with an issuance time). Use this when your
features come from an upstream model that produces a new vintage at each
observation point.

## Prerequisites

- Yohou installed ([Installation](installation.md))
- Familiarity with `X_actual`, `X_future`, and `X_forecast` categories
  ([Use Exogenous Features](exogenous-features.md))
- A fitted forecaster using exogenous features

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Tasks Covered

- [Prepare the X_forecast table](#prepare-the-x_forecast-table) with `vintage_time` and `time` columns
- [Align vintages to observation times](#align-vintages-to-observation-times) for training
- [Predict with different vintages](#predict-with-different-vintages) at the same observation point
- [Run walk-forward evaluation](#run-walk-forward-evaluation-with-vintages) with `X_forecast`

## Prepare the X_forecast Table

`X_forecast` is a tidy Polars DataFrame with columns `vintage_time`, `time`,
and one or more feature columns. Both `vintage_time` and `time` must have
`pl.Date` or `pl.Datetime` dtype:

```python
import polars as pl
from datetime import datetime

X_forecast = pl.DataFrame({
    "vintage_time": [datetime(2024, 1, 1)] * 3 + [datetime(2024, 1, 2)] * 3,
    "time": [
        datetime(2024, 1, 2), datetime(2024, 1, 3), datetime(2024, 1, 4),
        datetime(2024, 1, 3), datetime(2024, 1, 4), datetime(2024, 1, 5),
    ],
    "temperature": [10.0, 11.0, 12.0, 15.0, 16.0, 17.0],
})
```

Each `vintage_time` value represents the observation point at which this
forecast was known. Multiple rows per vintage capture the forecast horizon
(step 1, step 2, etc.). Internally, the framework converts this tidy
format into step-indexed columns (`temperature_step_1`,
`temperature_step_2`, ...).

## Align Vintages to Observation Times

For training, provide one vintage per observation time. If your upstream model
issues forecasts on a different schedule than your observation frequency,
re-anchor each vintage to the matching observation time.

For example, if a weather model issues a forecast every day at 06:00 but your
observation time is 23:00 the previous day:

```python
from datetime import datetime
import polars as pl

# Raw weather forecast: issued 2024-01-15 06:00, covers 3 hours ahead
wx_raw = pl.DataFrame({
    "issue_time": [datetime(2024, 1, 15, 6)] * 3,
    "target_time": [
        datetime(2024, 1, 15, 7),
        datetime(2024, 1, 15, 8),
        datetime(2024, 1, 15, 9),
    ],
    "temperature": [5.2, 5.8, 6.1],
})

# Re-anchor to observation time (last settled price at 23:00 previous day)
wx_aligned = wx_raw.rename({
    "issue_time": "vintage_time",
    "target_time": "time",
}).with_columns(
    pl.lit(datetime(2024, 1, 14, 23)).alias("vintage_time"),
)
```

The mapping from issuance time to observation time is domain specific. The
logic depends on your data frequency, observation schedule, and business
rules.

## Predict with Different Vintages

After fitting, call `predict()` once per vintage. Each call re-derives step
columns temporarily without mutating forecaster state:

```python
# Two weather vintages at the same observation point
pred_6am = forecaster.predict(X_forecast=wx_6am)
pred_9am = forecaster.predict(X_forecast=wx_9am)

# Bare predict still uses data stored during fit
pred_baseline = forecaster.predict()
```

If you also want to override deterministic features at the same time:

```python
pred = forecaster.predict(
    X_future=updated_holidays,
    X_forecast=wx_latest,
)
```

!!! warning "Thread Safety"
    The column swap mechanism is not thread safe. For parallel multi-vintage
    predictions, use `copy.deepcopy(forecaster)` once per thread.

## Run Walk-Forward Evaluation with Vintages

The `observe_predict` loop accepts all three exogenous parameters. Pass
`X_forecast` covering the test range with one vintage per observation point,
then score the rolling predictions with
[`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.point.MeanAbsoluteError/):

```python
from copy import deepcopy
from yohou.metrics import MeanAbsoluteError

forecasting_horizon = 7

preds = deepcopy(forecaster).observe_predict(
    y=y_test,
    X_actual=X_actual_test,
    X_future=X_future_full,       # full range (deterministic)
    X_forecast=X_forecast_test,   # vintages covering the test range
    stride=forecasting_horizon,
)

scorer = MeanAbsoluteError()
scorer.fit(y_train)
score = scorer.score(y_test, preds)
```

`X_future` should cover the full time range because it is deterministic.
`X_forecast` only needs to cover the test range.

!!! tip
    Always `deepcopy` the forecaster before calling `observe_predict`. The
    method mutates internal state, so a copy preserves the original for
    further use.

## See Also

- [Use Exogenous Features](exogenous-features.md): core `X_actual` and `X_future` workflows
- [About Exogenous Features](../explanation/exogenous-features.md): design rationale and
  internal mechanics
- [Exogenous Features Tutorial](../tutorials/exogenous-features.md): hands-on introduction
- [Evaluate Forecasts with Multi-vintage Scoring](multi-vintage-scoring.md): scoring across
  forecast origins
- [`window_forecasts` API Reference](../api/utils.md): as-of vintage
  matching with step alignment
