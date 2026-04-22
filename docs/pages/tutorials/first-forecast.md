# Your First Forecast

In this tutorial, we will build a sunspot forecaster that predicts solar activity two years into the future. Along the way, we will load and resample a real dataset, establish a seasonal baseline, build a reduction forecasting pipeline with stationarity transforms and lag features, compare three models side by side, and evaluate them with multiple metrics.

## Prerequisites

- Yohou installed (see [Getting Started](getting-started.md))
- A Python 3.11+ environment

## Load and Explore the Data

Sunspot numbers track solar activity and exhibit a well-known cycle of roughly 11 years. The dataset contains daily observations from 1818 to 2020.

```python
from yohou.datasets import fetch_sunspot

bunch = fetch_sunspot()
y_daily = bunch.frame
print(y_daily.head())
```

```text
shape: (5, 2)
┌─────────────────────┬────────────────┐
│ time                ┆ sunspot_number │
│ ---                 ┆ ---            │
│ datetime[μs]        ┆ f64            │
╞═════════════════════╪════════════════╡
│ 1818-01-08 00:00:00 ┆ 0.0            │
│ 1818-01-09 00:00:00 ┆ 0.0            │
│ 1818-01-10 00:00:00 ┆ 0.0            │
│ 1818-01-11 00:00:00 ┆ 0.0            │
│ 1818-01-12 00:00:00 ┆ 0.0            │
└─────────────────────┴────────────────┘
```

With 73,924 daily rows, working at monthly resolution is more practical. Yohou's [`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/) handles this:

```python
from yohou.preprocessing import Downsampler

downsampler = Downsampler(interval="1mo", aggregation="mean")
downsampler.fit(y_daily)
y = downsampler.transform(y_daily)
print(f"Monthly series: {len(y)} rows")
print(y.head())
```

```text
Monthly series: 2429 rows
shape: (5, 2)
┌─────────────────────┬────────────────┐
│ time                ┆ sunspot_number │
│ ---                 ┆ ---            │
│ datetime[μs]        ┆ f64            │
╞═════════════════════╪════════════════╡
│ 1818-01-01 00:00:00 ┆ 0.0            │
│ 1818-02-01 00:00:00 ┆ 0.208333       │
│ 1818-03-01 00:00:00 ┆ 0.0            │
│ 1818-04-01 00:00:00 ┆ 0.0            │
│ 1818-05-01 00:00:00 ┆ 0.0            │
└─────────────────────┴────────────────┘
```

## Train/Test Split

We hold out the last 24 months as our test set, covering two full years of solar activity to predict:

```python
y_train, y_test = y[:-24], y[-24:]
forecasting_horizon = len(y_test)
print(f"Train: {len(y_train)} months, Test: {forecasting_horizon} months")
```

```text
Train: 2405 months, Test: 24 months
```

## Step 1: Seasonal Baseline

A good starting point is [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/), which repeats values from one seasonal cycle ago. The sunspot cycle is roughly 11 years, or 132 months:

```python
from yohou.point import SeasonalNaive

baseline = SeasonalNaive(seasonality=132)
baseline.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_baseline = baseline.predict(forecasting_horizon=forecasting_horizon)
print(y_pred_baseline.head())
```

```text
shape: (5, 3)
┌─────────────────────┬─────────────────────┬────────────────┐
│ vintage_time        ┆ time                ┆ sunspot_number │
│ ---                 ┆ ---                 ┆ ---            │
│ datetime[μs]        ┆ datetime[μs]        ┆ f64            │
╞═════════════════════╪═════════════════════╪════════════════╡
│ 2018-05-01 00:00:00 ┆ 2018-06-01 00:00:00 ┆ …              │
│ 2018-05-01 00:00:00 ┆ 2018-07-01 00:00:00 ┆ …              │
│ 2018-05-01 00:00:00 ┆ 2018-08-01 00:00:00 ┆ …              │
│ 2018-05-01 00:00:00 ┆ 2018-09-01 00:00:00 ┆ …              │
│ 2018-05-01 00:00:00 ┆ 2018-10-01 00:00:00 ┆ …              │
└─────────────────────┴─────────────────────┴────────────────┘
```

Notice that predictions include both a `vintage_time` column (when the model last saw data) and a `time` column (the prediction timestamp). This is a Yohou convention across all forecasters.

## Step 2: Reduction Forecaster with Ridge

[`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) converts the forecasting problem into supervised learning. It tabularizes the time series, fits an sklearn regressor, and generates multi-step predictions:

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster

ridge_forecaster = PointReductionForecaster(estimator=Ridge())
ridge_forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_ridge = ridge_forecaster.predict(forecasting_horizon=forecasting_horizon)
```

This gives us a working forecaster, but it treats the raw sunspot numbers directly. Sunspot data has strong seasonal structure and non-stationarity, so we can do better.

## Step 3: Add Stationarity with SeasonalDifferencing

A [`target_transformer`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) removes seasonal patterns before the regressor sees the data. [`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/) subtracts each value from its counterpart one cycle ago, making the series closer to stationary. The forecaster automatically inverts this at prediction time.

Wrap the transformer in a [`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/) (the same pattern as sklearn's `Pipeline`):

```python
from yohou.compose import FeaturePipeline
from yohou.stationarity import SeasonalDifferencing

ridge_pipeline = PointReductionForecaster(
    estimator=Ridge(),
    target_transformer=FeaturePipeline([
        ("seasonal_diff", SeasonalDifferencing(seasonality=132)),
    ]),
)
ridge_pipeline.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_ridge_diff = ridge_pipeline.predict(forecasting_horizon=forecasting_horizon)
```

## Step 4: Add Lag Features

A [`feature_transformer`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) engineers input features for the regressor. [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/) creates autoregressive features from past values, letting the regressor learn patterns across multiple time steps:

```python
from yohou.preprocessing import LagTransformer

ridge_full = PointReductionForecaster(
    estimator=Ridge(),
    target_transformer=FeaturePipeline([
        ("seasonal_diff", SeasonalDifferencing(seasonality=132)),
    ]),
    feature_transformer=FeaturePipeline([
        ("lags", LagTransformer(lag=[1, 2, 3, 6, 12])),
    ]),
)
ridge_full.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_ridge_full = ridge_full.predict(forecasting_horizon=forecasting_horizon)
```

The lags `[1, 2, 3, 6, 12]` give the model access to the previous month, two and three months back, a half-year ago, and a full year ago.

## Step 5: Evaluate with Multiple Metrics

Now let's score all three models. Yohou scorers follow the same fit/score pattern. Call `fit` on training data first (some scaled metrics need the training scale), then `score` with test and predicted data:

```python
from yohou.metrics import MeanAbsoluteError, MeanSquaredError

mae = MeanAbsoluteError()
mse = MeanSquaredError()
mae.fit(y_train)
mse.fit(y_train)

results = {}
predictions = {
    "SeasonalNaive": y_pred_baseline,
    "Ridge": y_pred_ridge,
    "Ridge + Pipeline": y_pred_ridge_full,
}

for name, y_pred in predictions.items():
    results[name] = {
        "MAE": round(mae.score(y_test, y_pred), 2),
        "MSE": round(mse.score(y_test, y_pred), 2),
    }

for name, scores in results.items():
    print(f"{name:20s} MAE={scores['MAE']:.2f}  MSE={scores['MSE']:.2f}")
```

```text
SeasonalNaive        MAE=3.82  MSE=27.79
Ridge                MAE=31.90  MSE=1220.02
Ridge + Pipeline     MAE=4.48  MSE=30.26
```

The reduction forecaster with seasonal differencing and lag features should improve on the plain Ridge model. The Ridge model without a pipeline performs poorly because it tries to predict raw sunspot numbers without accounting for the 11-year seasonal cycle.

## Step 6: Visualize

[`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) accepts a dict of predictions to overlay multiple models on one chart:

```python
from yohou.plotting import plot_forecast

plot_forecast(
    y_test,
    {"SeasonalNaive": y_pred_baseline, "Ridge + Pipeline": y_pred_ridge_full},
    y_train=y_train,
    n_history=132,
    title="Sunspot Forecast Comparison",
    y_label="Monthly mean sunspot number",
)
```

The `n_history=132` parameter shows the last 11 years of training data for context. You should see the actual test values alongside both forecasts.

For a metric-level comparison, [`plot_score_summary`](/pages/api/generated/yohou.plotting.evaluation.plot_score_summary/) visualizes scores as a grouped bar chart:

```python
from yohou.plotting import plot_score_summary

plot_score_summary(
    {"MAE": mae, "MSE": mse},
    y_test,
    predictions,
    title="Model Comparison - Sunspot Forecasting",
)
```

## Step 7: Try a Stronger Model

The pipeline we built is regressor-agnostic. Swapping Ridge for a gradient boosting model takes one line. Since `HistGradientBoostingRegressor` does not support multi-output natively, we switch to `reduction_strategy="direct"`, which fits one model per forecast step:

```python
from sklearn.ensemble import HistGradientBoostingRegressor

hgb_forecaster = PointReductionForecaster(
    estimator=HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05),
    reduction_strategy="direct",
    target_transformer=FeaturePipeline([
        ("seasonal_diff", SeasonalDifferencing(seasonality=132)),
    ]),
    feature_transformer=FeaturePipeline([
        ("lags", LagTransformer(lag=[1, 2, 3, 6, 12])),
    ]),
)
hgb_forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_hgb = hgb_forecaster.predict(forecasting_horizon=forecasting_horizon)

results["HistGradientBoosting"] = {
    "MAE": round(mae.score(y_test, y_pred_hgb), 2),
    "MSE": round(mse.score(y_test, y_pred_hgb), 2),
}

for name, scores in results.items():
    print(f"{name:22s} MAE={scores['MAE']:.2f}  MSE={scores['MSE']:.2f}")
```

Now compare all four models visually:

```python
plot_forecast(
    y_test,
    {
        "SeasonalNaive": y_pred_baseline,
        "Ridge + Pipeline": y_pred_ridge_full,
        "HistGradientBoosting": y_pred_hgb,
    },
    y_train=y_train,
    n_history=132,
    title="Sunspot Forecast - All Models",
    y_label="Monthly mean sunspot number",
)
```

## What We Built

You have built a complete forecasting workflow from scratch. Along the way, you:

- Loaded and resampled a real-world dataset with `fetch_sunspot` and `Downsampler`
- Established a seasonal baseline with `SeasonalNaive`
- Built a reduction pipeline combining `SeasonalDifferencing` for stationarity and `LagTransformer` for autoregressive features
- Evaluated with `MeanAbsoluteError` and `MeanSquaredError`
- Swapped in a gradient boosting regressor with a single parameter change

The key pattern (`target_transformer` for stationarity, `feature_transformer` for feature engineering, and a pluggable sklearn `estimator`) works the same way for any time series and any regressor.

## Next Steps

- [Interval Forecasting](../explanation/interval-forecasting.md): prediction intervals and uncertainty quantification
- [Core Concepts](../explanation/core-concepts.md): observe/rewind, panel data, and metadata routing
- [Model Selection](../explanation/model-selection.md): evaluate with time series CV splitters
- [How to Evaluate Forecast Accuracy](../how-to/evaluate-forecast-accuracy.md): scoring, multi-metric comparison, and score visualization
- [Point Forecaster Examples](/pages/examples/point/): interactive notebooks with more dataset/model combinations
- [`PointReductionForecaster` API Reference](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/): full parameter documentation
- [Classification Forecasting](../how-to/classification-forecasting.md): forecast categorical targets
- [Ensemble Forecasting](../how-to/ensemble-forecasting.md): combine multiple forecasters
