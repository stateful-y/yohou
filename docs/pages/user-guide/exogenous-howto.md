# How to Use Exogenous Features

This guide shows you how to use `X_actual`, `X_future`, and `X_forecast` in
common production forecasting scenarios. Use this when you have external data
that should influence your forecasts.

!!! tip "Interactive version available"
    Try the multi-vintage recipe as an interactive notebook:
    [View](/examples/point/multi_vintage_forecasting/) · [Open in marimo](/examples/point/multi_vintage_forecasting/edit/)

## Prerequisites

- Yohou installed
- Familiarity with the fit/predict lifecycle
  ([Exogenous Features Tutorial](exogenous-tutorial.md))

---

## How to Classify Your Features

Before calling `fit()`, decide which parameter each feature belongs in:

| Question | Yes | No |
|---|---|---|
| Is it a measurement that can only be known after it happens? | `X_actual` | Continue |
| Is it deterministic and known for any future date? | `X_future` | Continue |
| Does it come from an external model with an issuance time? | `X_forecast` | N/A |

Common mappings:

- **Temperature readings, sensor data, realized demand** → `X_actual`
- **Holidays, day-of-week, scheduled events** → `X_future`
- **Weather model output, demand projections** → `X_forecast`

If a feature is uncertain but has no vintage (e.g., a single "best guess"),
treat it as `X_future`. If you need multiple versions of that guess at
predict time, wrap it with a `vintage_time` column and use `X_forecast`.

---

## How to Prepare X_forecast for Training

`X_forecast` requires a tidy table with columns `[vintage_time, time, col1, col2, ...]`.
Each `vintage_time` value represents when the forecast was issued.

**For training**, provide one vintage per observation time. If your weather model
issues a forecast every day at 06:00, re-anchor each forecast to the
corresponding observation time:

```python
import polars as pl

# Raw weather forecast: issued 2024-01-15 06:00, covers 24 hours
wx_raw = pl.DataFrame({
    "issue_time": [datetime(2024, 1, 15, 6)] * 24,
    "target_time": [datetime(2024, 1, 15, h) for h in range(24)],
    "temperature": [...]
})

# Re-anchor to observation time (last settled price at 23:00 previous day)
wx_aligned = wx_raw.rename({
    "issue_time": "vintage_time",
    "target_time": "time",
}).with_columns(
    pl.lit(datetime(2024, 1, 14, 23)).alias("vintage_time"),
)
```

The mapping from issuance time to observation time is domain-specific. Yohou
does not provide a utility for this because the logic depends on your data
frequency, observation schedule, and business rules.

---

## How to Predict with Multiple Vintages

After fitting, call `predict()` once per vintage. Each call swaps step columns
temporarily without changing internal state:

```python
# Two weather vintages at the same observation point
pred_6am = forecaster.predict(X_forecast=wx_6am_aligned)
pred_9am = forecaster.predict(X_forecast=wx_9am_aligned)

# State is unchanged: bare predict uses original stored data
pred_baseline = forecaster.predict()
```

If you also want to override known-future features:

```python
# Override both
pred = forecaster.predict(
    X_future=updated_holidays,
    X_forecast=wx_latest,
)
```

!!! warning "Thread Safety"
    The column-swap mechanism is not thread-safe. For parallel multi-vintage
    predictions, `copy.deepcopy(forecaster)` once per thread.

---

## How to Run Walk-Forward Evaluation with Exogenous Data

The `observe_predict` loop accepts all three parameters:

```python
preds = forecaster.observe_predict(
    y=y_test,
    X_actual=X_actual_test,
    X_future=X_future_full,       # full range, deterministic
    X_forecast=X_forecast_test,   # vintages covering the test range
    stride=forecasting_horizon,
)

scorer = MeanAbsoluteError()
scorer.fit(y_train)
score = scorer.score(y_test, preds)
```

`X_future` should cover the full time range (past and future) since it is
deterministic. `X_forecast` should cover the test range with appropriate
vintages.

---

## How to Use Exogenous Features with Composition Forecasters

### ColumnForecaster

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

### DecompositionPipeline

All three parameters pass through to the residual forecaster after trend
and seasonality removal:

```python
from yohou.compose import DecompositionPipeline
from yohou.stationarity import PolynomialTrendForecaster

pipeline = DecompositionPipeline(
    trend_forecaster=PolynomialTrendForecaster(degree=1),
    residual_forecaster=PointReductionForecaster(
        estimator=HistGradientBoostingRegressor(),
        feature_transformer=LagTransformer([1, 2, 3]),
        reduction_strategy="direct",
    ),
)

pipeline.fit(
    y=y_train,
    X_actual=X_actual_train,
    forecasting_horizon=H,
    X_future=holidays,
    X_forecast=weather,
)
```

### ForecastedFeatureForecaster

`X_actual` trains the feature forecaster (treated as its y) and provides
lag features for the target forecaster. `X_future` and `X_forecast` pass
through to the target forecaster directly.

At predict time, the feature forecaster is **not called**. Instead, the
target forecaster uses X_actual values stored in its observation window
(set during fit and updated by `observe`). This means the target
forecaster's X_actual lag features always reflect the latest observed
actuals, not forecasted values.

The `strategy` parameter controls what X_actual the target forecaster is
**trained** on: `"actual"` uses real values, `"predicted"` and `"rewind"`
use the feature forecaster's predictions so the target learns from inputs
similar to what it would see if actuals were unavailable.

```python
from yohou.compose import ForecastedFeatureForecaster

fff = ForecastedFeatureForecaster(
    target_forecaster=price_forecaster,
    feature_forecaster=temperature_forecaster,
    strategy="rewind",  # train target on predicted X_actual
)

# X_actual trains feature_forecaster (as y) and target_forecaster (as X_actual)
# X_future passes through to target_forecaster directly
fff.fit(
    y=y_train,
    X_actual=X_actual_train,
    forecasting_horizon=H,
    X_future=holidays,
)
```

---

## How to Pickle and Restore

The three-parameter state (step column names, stored raws) survives pickle
round-trips:

```python
import pickle

# Save
with open("forecaster.pkl", "wb") as f:
    pickle.dump(forecaster, f)

# Load
with open("forecaster.pkl", "rb") as f:
    restored = pickle.load(f)

# Multi-vintage predictions still work
pred = restored.predict(X_forecast=new_vintage)
```

---

## Troubleshooting

**Problem: `ValueError` about column name collisions**
: `X_future` and `X_forecast` produce step columns with the same name. Rename
  your source columns so they don't collide after `_step_` suffixing.

**Problem: Predictions don't change with different X_forecast vintages**
: Check that your X_forecast has the correct `vintage_time` value matching
  the forecaster's `observed_time_`. The vintage must be aligned to the
  observation point.

## See Also

- [About Exogenous Features](exogenous-features.md): design rationale and
  internal mechanics
- [Exogenous Features Tutorial](exogenous-tutorial.md): hands-on introduction
- [`pivot_forecasts` API Reference](../api/utils.md): utility for manual
  forecast pivoting
