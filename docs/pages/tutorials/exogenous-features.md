# Exogenous Features

In this tutorial, we will build a forecasting model that uses all three types
of exogenous features: actual observations, known-future indicators, and
external forecast vintages. We will fit the model on synthetic electricity
price data, produce predictions from two different weather forecast vintages,
and run a walk-forward evaluation.

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Prerequisites

- Completed [Forecasting Workflow](forecasting-workflow.md)

## 1. Create the Synthetic Data

We start with a simple scenario: hourly electricity prices that depend linearly
on temperature and a holiday indicator. This makes the relationships
transparent so we can verify the model captures them.

```python
from datetime import datetime, timedelta
import numpy as np
import polars as pl

rng = np.random.default_rng(42)
n = 200
times = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
time_series = pl.Series("time", times)
```

Now we create the three exogenous data sources:

```python
# X_actual: realized temperature (sinusoidal daily cycle + noise)
t = np.arange(n, dtype=float)
actual_temp = 15.0 + 5.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n)
X_actual = pl.DataFrame({"time": time_series, "temperature": actual_temp})

# X_future: holiday indicator (1.0 on Sundays, 0.0 otherwise)
holidays = np.array([1.0 if dt.weekday() == 6 else 0.0 for dt in times])
X_future = pl.DataFrame({"time": time_series, "is_holiday": holidays})

# y: price = 50 + 2·temperature + 10·is_holiday + noise
price = 50.0 + 2.0 * actual_temp + 10.0 * holidays + rng.normal(0, 0.1, n)
y = pl.DataFrame({"time": time_series, "price": price})
```

Notice that `X_actual` contains temperature readings (historical observations),
while `X_future` contains holiday indicators (deterministic, known for any
date).

## 2. Create Weather Forecasts (X_forecast)

External forecasts carry a `vintage_time` column that identifies when the
forecast was issued. We create training forecasts with one vintage per
observation time, covering the next `H` steps:

```python
H = 6  # forecast horizon
forecast_rows = []
for i in range(H, n):
    vt = times[i]
    for step in range(1, H + 1):
        if i + step < n:
            forecast_rows.append({
                "vintage_time": vt,
                "time": times[i + step],
                "wx_temp": float(actual_temp[i + step] + 0.5 + rng.normal(0, 0.3)),
            })
X_forecast = pl.DataFrame(forecast_rows)
```

Each row says: "at time `vintage_time`, the weather model predicted temperature
`wx_temp` for time `time`." The 0.5 bias represents the forecast's systematic
error.

## 3. Fit the Forecaster

We use a `PointReductionForecaster` with the `"direct"` strategy and
`HistGradientBoostingRegressor` (which handles null values from partial
forecast coverage):

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer

forecaster = PointReductionForecaster(
    estimator=HistGradientBoostingRegressor(max_iter=50, max_depth=3),
    feature_transformer=LagTransformer(lag=[1, 2, 3]),
    reduction_strategy="direct",
)

# Split train/test
train_size = 160
y_train, y_test = y[:train_size], y[train_size:]
X_actual_train = X_actual[:train_size]

forecaster.fit(
    y=y_train,
    X_actual=X_actual_train,
    forecasting_horizon=H,
    X_future=X_future,        # full range, deterministic
    X_forecast=X_forecast,     # training vintages
)
```

After fitting, the forecaster stores step columns from both `X_future`
and `X_forecast`:

```python
step_cols = sorted(forecaster._step_column_names_)
print(f"Step columns ({len(step_cols)}): {step_cols[:3]} ... {step_cols[-3:]}")
```

```text
Step columns (12): ['is_holiday_step_1', 'is_holiday_step_2', 'is_holiday_step_3'] ... ['wx_temp_step_4', 'wx_temp_step_5', 'wx_temp_step_6']
```

Notice that both holiday and weather columns were converted to step-indexed
format: one column per forecast step for each feature.

## 4. Predict with Multiple Vintages

Now we create two weather forecast vintages at the test boundary: one accurate
(small bias) and one biased (large bias):

```python
last_obs = times[train_size - 1]
test_times = times[train_size:train_size + H]

# Accurate vintage
X_forecast_accurate = pl.DataFrame({
    "vintage_time": [last_obs] * H,
    "time": test_times,
    "wx_temp": [float(actual_temp[train_size + i] + 0.1) for i in range(H)],
})

# Biased vintage
X_forecast_biased = pl.DataFrame({
    "vintage_time": [last_obs] * H,
    "time": test_times,
    "wx_temp": [float(actual_temp[train_size + i] + 5.0) for i in range(H)],
})
```

We call `predict()` once per vintage. Each call temporarily swaps the weather
step columns without changing the forecaster's internal state:

```python
pred_accurate = forecaster.predict(X_forecast=X_forecast_accurate)
pred_biased = forecaster.predict(X_forecast=X_forecast_biased)

print("Accurate vintage prices:", pred_accurate["price"].to_list()[:3])
print("Biased vintage prices:  ", pred_biased["price"].to_list()[:3])
```

The predictions differ because the weather forecasts differ. The more accurate
vintage should produce predictions closer to the true prices.

## 5. Walk-Forward Evaluation

The `observe_predict` loop steps through `y_test` one stride at a time,
observing new `X_actual` at each step. We pass the full `X_forecast` so the
forecaster can look up the appropriate vintage at each observation time:

```python
from yohou.metrics import MeanAbsoluteError

X_actual_test = X_actual[train_size:]
preds_with_wx = forecaster.observe_predict(
    y=y_test,
    X_actual=X_actual_test,
    X_future=X_future,
    X_forecast=X_forecast,
    stride=H,
)

scorer = MeanAbsoluteError()
scorer.fit(y_train)
print(f"Walk-forward MAE (with weather forecast): {scorer.score(y_test[:len(preds_with_wx)], preds_with_wx):.4f}")
```

```text
Walk-forward MAE (with weather forecast): 2.8083
```

To see how much the weather signal contributes, fit the same architecture
without `X_forecast` and compare:

```python
forecaster_no_wx = PointReductionForecaster(
    estimator=HistGradientBoostingRegressor(max_iter=50, max_depth=3),
    feature_transformer=LagTransformer(lag=[1, 2, 3]),
    reduction_strategy="direct",
)
forecaster_no_wx.fit(
    y=y_train,
    X_actual=X_actual_train,
    forecasting_horizon=H,
    X_future=X_future,
)

preds_no_wx = forecaster_no_wx.observe_predict(
    y=y_test,
    X_actual=X_actual_test,
    X_future=X_future,
    stride=H,
)
print(f"Walk-forward MAE (no  weather forecast): {scorer.score(y_test[:len(preds_no_wx)], preds_no_wx):.4f}")
```

```text
Walk-forward MAE (no  weather forecast): 3.7157
```

Notice that `X_future` (holiday indicator) covers the full time range in
both cases: it is deterministic and known for all dates, so no slicing is needed.

The weather signal reduces MAE from 3.72 to 2.81. The remaining gap to the
noise floor reflects the forecast bias (0.5°C) and uncertainty (std 0.3°C)
we deliberately built into `X_forecast`: a perfect weather model would close
it further.

## What You Built

You built a forecasting model that:

- Uses temperature readings (`X_actual`) for historical actual features
- Incorporates holiday calendars (`X_future`) as step-indexed known-future
  features
- Accepts weather forecast vintages (`X_forecast`) with `vintage_time`
- Produces different predictions for different forecast vintages
- Runs walk-forward evaluation with proper data separation

## Next Steps

- [How to Use Exogenous Features](../how-to/exogenous-features.md): production workflow
  recipes for multi-vintage prediction and composition
- [About Exogenous Features](../explanation/exogenous-features.md): design rationale,
  step-indexed columns, and cross-validation behavior
- [`PointReductionForecaster` API Reference](../api/point.md): full parameter
  documentation
