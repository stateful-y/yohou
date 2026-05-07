# Exogenous Features Tutorial

In this tutorial, we will build a forecasting model that uses all three types
of exogenous features: actual observations, known-future indicators, and
external forecast vintages. We will fit the model on synthetic electricity
price data, produce predictions from two different weather forecast vintages,
and run a walk-forward evaluation.

!!! tip "Interactive version available"
    Try this tutorial as an interactive notebook:
    [View](/examples/exogenous_features/) · [Open in marimo](/examples/point/exogenous_features/edit/)

## Prerequisites

- Yohou installed (`pip install yohou`)
- Basic familiarity with the fit/predict/observe lifecycle
  ([Core Concepts](core-concepts.md))

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
holidays = [1.0 if dt.weekday() == 6 else 0.0 for dt in times]
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
    feature_transformer=LagTransformer([1, 2, 3]),
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
print(f"Step columns: {len(forecaster._step_column_names_)}")
print(f"Some step columns: {sorted(forecaster._step_column_names_)[:6]}")
```

The output should look something like:

```text
Step columns: 12
Some step columns: ['is_holiday_step_1', 'is_holiday_step_2', ..., 'wx_temp_step_1', ...]
```

Notice that both holiday and weather columns were converted to step-indexed
format.

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

The `observe_predict` loop observes `X_actual` at each step and uses
`X_future` for step column derivation. Here we omit `X_forecast` so the
loop uses the training forecasts stored at fit time:

```python
from yohou.metrics import MeanAbsoluteError

X_actual_test = X_actual[train_size:]

preds = forecaster.observe_predict(
    y=y_test,
    X_actual=X_actual_test,
    X_future=X_future,
    stride=H,
)

scorer = MeanAbsoluteError()
scorer.fit(y_train)
score = scorer.score(y_test[:len(preds)], preds)
print(f"Walk-forward MAE: {score:.4f}")
```

Notice that `X_future` covers the full time range (it's deterministic and
available for all dates).

## What We Built

We built a forecasting model that:

- Uses temperature readings (`X_actual`) for lag-based features
- Incorporates holiday calendars (`X_future`) as step-indexed known-future
  features
- Accepts weather forecast vintages (`X_forecast`) with `vintage_time`
- Produces different predictions for different forecast vintages
- Runs walk-forward evaluation with proper data separation

## Next Steps

- [How to Use Exogenous Features](exogenous-howto.md): production workflow
  recipes for multi-vintage prediction and composition
- [About Exogenous Features](exogenous-features.md): design rationale,
  step-indexed columns, and cross-validation behavior
- [`PointReductionForecaster` API Reference](../api/point.md): full parameter
  documentation
