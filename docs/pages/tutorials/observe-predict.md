# Observe/Predict Workflow

In this tutorial, we will walk through a two-year test set in six-month batches, updating forecasts as new data arrives. Along the way, we will compare a single-shot prediction against a full walk-forward loop and score them both with [`MeanAbsoluteScaledError`](/pages/api/generated/yohou.metrics.point.MeanAbsoluteScaledError/).

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Prerequisites

- Completed [Forecasting Workflow](forecasting-workflow.md)

## 1. Load the Data

We use the tourism monthly dataset and work with a single series: visitor arrivals renamed to `tourists`. We hold out the final 24 months as the test set and set a 6-month forecasting horizon:

```python
from yohou.datasets import fetch_tourism_monthly
from yohou.model_selection import train_test_split

bunch = fetch_tourism_monthly()
y = (
    bunch.frame
    .select("time", "T1__tourists")
    .rename({"T1__tourists": "tourists"})
    .drop_nulls()
)
print(f"Series length: {len(y)} months")

forecasting_horizon = 6
y_train, y_test = train_test_split(y, test_size=24)
print(f"Train: {len(y_train)} months, Test: {len(y_test)} months")
```

```text
Series length: 187 months
Train: 163 months, Test: 24 months
```

## 2. Fit the Forecaster

Now build a [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) with seasonal differencing and lag features. If the pipeline looks unfamiliar, see [Getting Started](getting-started.md) for a step-by-step walkthrough:

```python
from yohou.compose import FeaturePipeline
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer
from yohou.stationarity import SeasonalDifferencing
from sklearn.linear_model import Ridge

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    target_transformer=SeasonalDifferencing(seasonality=12),
    actual_transformer=FeaturePipeline([
        ("lags", LagTransformer(lag=[1, 2, 3, 12])),
    ]),
)
forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)
```

## 3. Single-Shot Predict

First, produce a single-shot forecast: one batch of predictions from the end of the training data, covering the first six months of the test period.

```python
y_pred_single = forecaster.predict(forecasting_horizon=forecasting_horizon)
print(y_pred_single)
```

```text
shape: (6, 3)
┌─────────────────────┬─────────────────────┬─────────────┐
│ vintage_time        ┆ time                ┆ tourists    │
│ ---                 ┆ ---                 ┆ ---         │
│ datetime[μs]        ┆ datetime[μs]        ┆ f64         │
╞═════════════════════╪═════════════════════╪═════════════╡
│ 1992-07-01 00:00:00 ┆ 1992-08-01 00:00:00 ┆ 6487.458017 │
│ 1992-07-01 00:00:00 ┆ 1992-09-01 00:00:00 ┆ 4226.827546 │
│ 1992-07-01 00:00:00 ┆ 1992-10-01 00:00:00 ┆ 3063.841085 │
│ 1992-07-01 00:00:00 ┆ 1992-11-01 00:00:00 ┆ 1956.573403 │
│ 1992-07-01 00:00:00 ┆ 1992-12-01 00:00:00 ┆ 2509.830942 │
│ 1992-07-01 00:00:00 ┆ 1993-01-01 00:00:00 ┆ 1916.446268 │
└─────────────────────┴─────────────────────┴─────────────┘
```

Notice that all six rows share the same `vintage_time` (July 1992). This tells you the forecaster used only training data up to that date to produce all six predictions at once.

## 4. Walk-Forward with observe_predict

Now that we have a baseline prediction, let's see how the forecaster performs when it can update itself with actual observations. [`observe_predict`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) steps through `y_test` in batches of size `stride`, observing actual values and issuing a fresh forecast after each batch. Setting `stride=forecasting_horizon` tiles the test set with no gaps or overlaps:

```python
y_pred_loop = forecaster.observe_predict(y=y_test, stride=forecasting_horizon)
print(f"Total predictions: {len(y_pred_loop)}")
print(f"Distinct vintage_times: {y_pred_loop['vintage_time'].n_unique()}")
print(y_pred_loop.head(12))
```

```text
Total predictions: 30
Distinct vintage_times: 5
shape: (12, 3)
┌─────────────────────┬─────────────────────┬─────────────┐
│ vintage_time        ┆ time                ┆ tourists    │
│ ---                 ┆ ---                 ┆ ---         │
│ datetime[μs]        ┆ datetime[μs]        ┆ f64         │
╞═════════════════════╪═════════════════════╪═════════════╡
│ 1992-07-01 00:00:00 ┆ 1992-08-01 00:00:00 ┆ 6487.458017 │
│ 1992-07-01 00:00:00 ┆ 1992-09-01 00:00:00 ┆ 4226.827546 │
│ 1992-07-01 00:00:00 ┆ 1992-10-01 00:00:00 ┆ 3063.841085 │
│ 1992-07-01 00:00:00 ┆ 1992-11-01 00:00:00 ┆ 1956.573403 │
│ 1992-07-01 00:00:00 ┆ 1992-12-01 00:00:00 ┆ 2509.830942 │
│ 1992-07-01 00:00:00 ┆ 1993-01-01 00:00:00 ┆ 1916.446268 │
│ 1993-01-01 00:00:00 ┆ 1993-02-01 00:00:00 ┆ 2087.752024 │
│ 1993-01-01 00:00:00 ┆ 1993-03-01 00:00:00 ┆ 2107.340529 │
│ 1993-01-01 00:00:00 ┆ 1993-04-01 00:00:00 ┆ 2261.252994 │
│ 1993-01-01 00:00:00 ┆ 1993-05-01 00:00:00 ┆ 2465.019067 │
│ 1993-01-01 00:00:00 ┆ 1993-06-01 00:00:00 ┆ 2642.654218 │
│ 1993-01-01 00:00:00 ┆ 1993-07-01 00:00:00 ┆ 3005.618428 │
└─────────────────────┴─────────────────────┴─────────────┘
```

Notice three things in the output:

1. There are 5 distinct `vintage_time` values, one for each batch of 6 predictions
2. The first 6 rows (vintage July 1992) are identical to the single-shot predict above
3. The second vintage (January 1993) starts after the first 6-month batch was observed, so the forecaster now has fresher context

The 5 vintages come from 24 test months divided into 6-month strides: $\lceil 24 / 6 \rceil + 1 = 5$ (the extra batch extends beyond the test window). Each batch contributes 6 predictions, producing $5 \times 6 = 30$ rows total.

After observing all 24 test months, the loop issues one final batch that extends beyond the test window. We will filter those out before scoring.

## 5. Score and Compare

Now let's score both approaches. First, filter predictions to the test period, then compare single-shot and walk-forward MASE:

```python
import polars as pl
from yohou.metrics import MeanAbsoluteScaledError

y_pred_in_test = y_pred_loop.filter(pl.col("time") <= y_test["time"][-1])

scorer = MeanAbsoluteScaledError(seasonality=12)
scorer.fit(y_train)
mase_loop = scorer.score(y_test, y_pred_in_test)
mase_single = scorer.score(y_test[:forecasting_horizon], y_pred_single)

print(f"Single-shot MASE (months 1 to 6):  {mase_single:.3f}")
print(f"Walk-forward MASE (all 24 months): {mase_loop:.3f}")
```

```text
Single-shot MASE (months 1 to 6):  0.752
Walk-forward MASE (all 24 months): 0.827
```

Notice that the single-shot score only covers the first six months, where the model has the freshest lag features. The walk-forward score spans all 24 months, giving a more representative picture of real-world performance.

## 6. Visualize the Predictions

Finally, plot all walk-forward prediction vintages against the actual test values:

```python
from yohou.plotting import plot_forecast

fig = plot_forecast(y_test, y_pred_in_test, y_train=y_train[-24:])
fig.show()
```

You should see each vintage as a separate colored segment overlaid on the actual test series, with the last 24 months of training history for context. See how successive batches pick up where the previous ones left off, covering the full two-year test window.

## What You Built

We used `observe_predict` to walk through a two-year test set in six-month batches, updating the forecaster's observation window at each stride:

- Produced a single-shot forecast with [`predict`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) and observed that all predictions share one `vintage_time`
- Ran [`observe_predict`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) to step through the test set, issuing fresh forecasts after each observed batch
- Scored both approaches with [`MeanAbsoluteScaledError`](/pages/api/generated/yohou.metrics.point.MeanAbsoluteScaledError/) and saw that walk-forward evaluation gives a fuller picture than a single holdout
- Visualized all prediction vintages against actuals with [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/)

## Next Steps

- [Exogenous Features](exogenous-features.md): Pass `X_actual`, `X_future`, and `X_forecast` into the observe/predict loop to incorporate weather forecasts and calendar information
- [Model Selection](../explanation/model-selection.md): Understand how to choose `stride` and `forecasting_horizon` for different deployment patterns
- [How to Use Forecast Vintages](../how-to/forecast-vintages.md): Prepare, align, and predict with `X_forecast` in a vintage workflow
