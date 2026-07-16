# Exogenous Features

In this tutorial, we will build a forecasting model that uses all three types of exogenous features: actual observations (`X_actual`), known future indicators (`X_future`), and external forecast vintages (`X_forecast`). We will fit the model on synthetic electricity price data, show that different weather forecast vintages produce different predictions, and run a walk-forward evaluation to measure the improvement from including weather forecasts.

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Prerequisites

- Completed [Forecasting Workflow](forecasting-workflow.md)

## 1. Load the Data

Start by generating a synthetic electricity dataset that includes all three exogenous types. [`make_exogenous_regression()`](/pages/api/generated/yohou.datasets.make_exogenous_regression/) creates hourly electricity prices driven by temperature (actual), a holiday indicator (future), and weather model forecasts (forecast vintage).

```python
from yohou.datasets import make_exogenous_regression

data = make_exogenous_regression()
y = data.y
X_actual = data.X_actual
X_future = data.X_future
X_forecast = data.X_forecast
H = 6

print("y:", y.shape)
print("X_actual:", X_actual.shape)
print("X_future:", X_future.shape)
print("X_forecast:", X_forecast.shape)
```

```text
y: (200, 2)
X_actual: (200, 2)
X_future: (200, 2)
X_forecast: (1143, 3)
```

`X_actual` contains realized temperature (known only after it occurs), `X_future` contains a holiday indicator (deterministic, known for all dates), and `X_forecast` carries weather forecasts with a `vintage_time` column identifying when each forecast was issued. See [Exogenous Features](../explanation/exogenous-features.md) for the full conceptual model.

## 2. Split and Fit

We split the data with [`train_test_split`](/pages/api/generated/yohou.model_selection.train_test_split/), which handles both row-indexed arrays and vintage-indexed `X_forecast` in a single call:

```python
from yohou.model_selection import train_test_split

y_train, y_test, X_actual_train, X_actual_test, X_forecast_train, _ = train_test_split(
    y, X_actual, test_size=40, X_forecast=X_forecast,
)
```

Now we build a [`PointReductionForecaster`](/pages/api/generated/yohou.point.PointReductionForecaster/) with the `"direct"` strategy and `HistGradientBoostingRegressor`, using a [`LagTransformer`](/pages/api/generated/yohou.preprocessing.LagTransformer/) to build lag features:

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer

forecaster = PointReductionForecaster(
    estimator=HistGradientBoostingRegressor(max_iter=50, max_depth=3),
    feature_transformer=LagTransformer(lag=[1, 2, 3]),
    reduction_strategy="direct",
)

forecaster.fit(
    y=y_train,
    X_actual=X_actual_train,
    forecasting_horizon=H,
    X_future=X_future,
    X_forecast=X_forecast_train,
)
```

After fitting, the forecaster stores step columns from both `X_future` and `X_forecast`:

```python
step_cols = sorted(forecaster._step_column_names_)
print(f"Step columns ({len(step_cols)}): {step_cols[:3]} ... {step_cols[-3:]}")
```

```text
Step columns (12): ['is_holiday_step_1', 'is_holiday_step_2', 'is_holiday_step_3'] ... ['wx_temp_step_4', 'wx_temp_step_5', 'wx_temp_step_6']
```

Notice that both holiday and weather columns were converted to step-indexed format: one column per forecast step for each feature.

## 3. Predict with Multiple Vintages

Now we create two weather forecast vintages at the test boundary: one accurate (small bias) and one biased (large bias). Each vintage covers the same future time steps but with different temperature values:

```python
import polars as pl

last_train_time = y_train["time"].item(-1)
test_times = y_test["time"].to_list()[:H]
actual_temp = X_actual_test["temperature"].to_list()[:H]

X_forecast_accurate = pl.DataFrame({
    "vintage_time": [last_train_time] * H,
    "time": test_times,
    "wx_temp": [t + 0.1 for t in actual_temp],
})

X_forecast_biased = pl.DataFrame({
    "vintage_time": [last_train_time] * H,
    "time": test_times,
    "wx_temp": [t + 5.0 for t in actual_temp],
})
```

We call `predict()` once per vintage:

```python
pred_accurate = forecaster.predict(X_forecast=X_forecast_accurate)
pred_biased = forecaster.predict(X_forecast=X_forecast_biased)

print("Accurate vintage prices:", [round(v, 2) for v in pred_accurate["price"].to_list()[:3]])
print("Biased vintage prices:  ", [round(v, 2) for v in pred_biased["price"].to_list()[:3]])
```

Notice that the predictions differ because the weather forecasts differ. The accurate vintage should produce values closer to the true prices.

## 4. Walk-Forward Evaluation

`observe_predict` steps through `y_test` one stride at a time, observing new `X_actual` and issuing fresh forecasts, with accuracy computed using [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.MeanAbsoluteError/):

```python
from yohou.metrics import MeanAbsoluteError

preds_with_wx = forecaster.observe_predict(
    y=y_test,
    X_actual=X_actual_test,
    X_future=X_future,
    X_forecast=X_forecast,
    stride=H,
)

scorer = MeanAbsoluteError()
scorer.fit(y_train)
score_with_wx = scorer.score(y_test[:len(preds_with_wx)], preds_with_wx)
print(f"Walk-forward MAE (with weather): {score_with_wx:.4f}")
```

```text
Walk-forward MAE (with weather): 2.8083
```

## 5. Measure the Value of Weather Forecasts

To see how much the weather signal contributes, we fit the same architecture without `X_forecast` and compare:

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
score_no_wx = scorer.score(y_test[:len(preds_no_wx)], preds_no_wx)
print(f"Walk-forward MAE (no weather):   {score_no_wx:.4f}")
```

```text
Walk-forward MAE (no weather):   3.7157
```

The weather signal reduces MAE from 3.72 to 2.81. Notice that `X_future` (holiday indicator) covers the full time range in both cases: it is deterministic and known for all dates, so no slicing is needed.

## What You Built

We built a forecasting model that accepts all three exogenous types, showed that different `X_forecast` vintages produce different predictions, and measured the improvement from including weather forecasts via walk-forward evaluation.

## Next Steps

- [Exogenous Features](../explanation/exogenous-features.md) for the conceptual model behind `X_actual`, `X_future`, and `X_forecast`
- [How to Use Exogenous Features](../how-to/exogenous-features.md) for production workflow recipes with multi-vintage prediction and composition
