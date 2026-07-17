# Getting Started

In this tutorial, we will install Yohou, load a real time series dataset, establish a seasonal baseline, build a reduction forecasting pipeline with lag and rolling features, compare models side by side, and evaluate them with multiple metrics. Along the way, we will encounter the core Yohou workflow: load, split, fit, predict, score, and plot.

<!-- COMPANION_NOTEBOOKS -->

## Prerequisites

- No prior experience with Yohou is required
- For conda, mamba, or development installation, see [Installation](../how-to/installation.md)

## 1. Install

We include the `plotting` extra so we can visualise the forecast later.

=== "uv"

    ```bash
    uv add "yohou[plotting]"
    ```

=== "pip"

    ```bash
    pip install "yohou[plotting]"
    ```

Verify the installation:

```python
import yohou

print(yohou.__version__)
```

You should see the installed version number printed.

## 2. Load and Resample the Data

Sunspot numbers track solar activity and exhibit a well-known cycle of roughly 11 years. The dataset contains daily observations from 1818 to 2020. Yohou ships with loaders that fetch datasets from [Monash/Zenodo](https://forecastingdata.org) and cache them locally. Each loader returns a `Bunch` object whose `.frame` attribute is a Polars DataFrame with a `time` column plus one or more value columns.

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
│ 1818-01-08 00:00:00 ┆ 65.0           │
│ 1818-01-09 00:00:00 ┆ 65.0           │
│ 1818-01-10 00:00:00 ┆ 65.0           │
│ 1818-01-11 00:00:00 ┆ 65.0           │
│ 1818-01-12 00:00:00 ┆ 65.0           │
└─────────────────────┴────────────────┘
```

With 73,924 daily rows, working at monthly resolution is more practical. Yohou's [`Downsampler`](/pages/api/generated/yohou.preprocessing.Downsampler/) handles this:

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
│ 1818-01-01 00:00:00 ┆ 60.458333      │
│ 1818-02-01 00:00:00 ┆ 33.857143      │
│ 1818-03-01 00:00:00 ┆ 39.0           │
│ 1818-04-01 00:00:00 ┆ 56.133333      │
│ 1818-05-01 00:00:00 ┆ 86.096774      │
└─────────────────────┴────────────────┘
```

## 3. Train/Test Split

Solar cycle 24 (2009 to 2019) was significantly weaker than cycle 23 (1996 to 2008). We will use this as our test period: we hold out the last 125 months (roughly the full cycle 24 era) with [`train_test_split`](/pages/api/generated/yohou.model_selection.train_test_split/) and forecast the first 24 of those months, covering cycle 24's early rise.

```python
from yohou.model_selection import train_test_split

y_train, y_future = train_test_split(y, test_size=125)
y_test = y_future.head(24)
forecasting_horizon = len(y_test)
print(f"Train: {len(y_train)} months, Test: {forecasting_horizon} months")
```

```text
Train: 2304 months, Test: 24 months
```

## 4. Seasonal Baseline

A good starting point is [`SeasonalNaive`](/pages/api/generated/yohou.point.SeasonalNaive/), which repeats values from one seasonal cycle ago. The sunspot cycle is roughly 11 years, or 132 months:

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
│ 2009-12-01 00:00:00 ┆ 2010-01-01 00:00:00 ┆ 86.0           │
│ 2009-12-01 00:00:00 ┆ 2010-02-01 00:00:00 ┆ 98.0           │
│ 2009-12-01 00:00:00 ┆ 2010-03-01 00:00:00 ┆ 103.548387     │
│ 2009-12-01 00:00:00 ┆ 2010-04-01 00:00:00 ┆ 93.566667      │
│ 2009-12-01 00:00:00 ┆ 2010-05-01 00:00:00 ┆ 149.612903     │
└─────────────────────┴─────────────────────┴────────────────┘
```

Notice that predictions include both a `vintage_time` column (when the model last saw data) and a `time` column (the prediction target). This is a Yohou convention across all forecasters.

Also notice the predicted values: 86, 98, 103, 93, 149 monthly sunspots. These are the actual values from January to May 1999, exactly 132 months earlier. At that time, solar cycle 23 was rising strongly toward its 2001 peak. But solar cycle 24 rose much more slowly. We will see how badly this assumption breaks down.

## 5. Reduction Forecaster with Ridge

[`PointReductionForecaster`](/pages/api/generated/yohou.point.PointReductionForecaster/) converts the forecasting problem into supervised learning. It tabularizes the time series, fits an sklearn regressor, and generates multi-step predictions. When no `feature_transformer` is provided, the forecaster uses the default tabularization, which creates a single lag feature from the most recent observation:

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster

ridge_plain = PointReductionForecaster(estimator=Ridge())
ridge_plain.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_ridge = ridge_plain.predict(forecasting_horizon=forecasting_horizon)
```

Notice that even without any explicit feature engineering, Ridge already captures the current level of solar activity, unlike SeasonalNaive which anchors to cycle 23's much stronger rise.

For targets that are categorical classes rather than continuous values, see the [Class-Probability Forecasting](class-proba-forecasting.md) tutorial for the equivalent pattern using a classifier estimator.

## 6. Add Lag Features

A [`feature_transformer`](/pages/api/generated/yohou.point.PointReductionForecaster/) engineers input features for the regressor. [`LagTransformer`](/pages/api/generated/yohou.preprocessing.LagTransformer/) creates autoregressive features from past values, letting the regressor learn patterns across multiple time steps. Wrap it in a [`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/) (the same pattern as sklearn's [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)):

```python
from yohou.compose import FeaturePipeline
from yohou.preprocessing import LagTransformer

ridge_lags = PointReductionForecaster(
    estimator=Ridge(),
    feature_transformer=FeaturePipeline([
        ("lags", LagTransformer(lag=[1, 2, 3, 6, 12])),
    ]),
)
ridge_lags.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_ridge_lags = ridge_lags.predict(forecasting_horizon=forecasting_horizon)
```

The lags `[1, 2, 3, 6, 12]` give the model access to the previous month, two and three months back, half a year ago, and a full year ago. Notice that `feature_transformer` adds input features for the regressor without modifying the target values.

## 7. Add Rolling Statistics

[`RollingStatisticsTransformer`](/pages/api/generated/yohou.preprocessing.RollingStatisticsTransformer/) adds rolling mean and standard deviation over a window. The rolling mean provides a smooth estimate of the current solar activity level, while the standard deviation captures recent volatility. Use [`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/) to combine it with the lag features in parallel:

```python
from yohou.compose import FeatureUnion
from yohou.preprocessing import RollingStatisticsTransformer

ridge_full = PointReductionForecaster(
    estimator=Ridge(),
    feature_transformer=FeaturePipeline([
        ("features", FeatureUnion([
            ("lags", LagTransformer(lag=list(range(1, 13)))),
            ("rolling", RollingStatisticsTransformer(window_size=12, statistics=["mean", "std"])),
        ])),
    ]),
)
ridge_full.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_ridge_full = ridge_full.predict(forecasting_horizon=forecasting_horizon)
print(y_pred_ridge_full.head(3))
```

```text
shape: (3, 3)
┌─────────────────────┬─────────────────────┬────────────────┐
│ vintage_time        ┆ time                ┆ sunspot_number │
│ ---                 ┆ ---                 ┆ ---            │
│ datetime[μs]        ┆ datetime[μs]        ┆ f64            │
╞═════════════════════╪═════════════════════╪════════════════╡
│ 2009-12-01 00:00:00 ┆ 2010-01-01 00:00:00 ┆ 14.465744      │
│ 2009-12-01 00:00:00 ┆ 2010-02-01 00:00:00 ┆ 14.410504      │
│ 2009-12-01 00:00:00 ┆ 2010-03-01 00:00:00 ┆ 16.006048      │
└─────────────────────┴─────────────────────┴────────────────┘
```

Notice that [`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/) combines transformers in parallel, stacking their output columns. [`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/) chains them sequentially. Together they give you the full sklearn composition vocabulary for feature engineering.

## 8. Evaluate with Multiple Metrics

Now let's score all four models. Scorers in Yohou are stateful: `scorer.fit(y_train)` stores the training data so that scale-dependent metrics like [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.MeanAbsoluteError/) can normalise correctly.

```python
from yohou.metrics import MeanAbsoluteError, MeanSquaredError

mae = MeanAbsoluteError()
mse = MeanSquaredError()
mae.fit(y_train)
mse.fit(y_train)

predictions = {
    "SeasonalNaive": y_pred_baseline,
    "Ridge": y_pred_ridge,
    "Ridge + Lags": y_pred_ridge_lags,
    "Ridge + Lags + Rolling": y_pred_ridge_full,
}

for name, y_pred in predictions.items():
    print(f"{name:25s} MAE={mae.score(y_test, y_pred):.2f}  MSE={mse.score(y_test, y_pred):.2f}")
```

```text
SeasonalNaive             MAE=102.15  MSE=12567.36
Ridge                     MAE=19.99   MSE=939.55
Ridge + Lags              MAE=14.98   MSE=559.97
Ridge + Lags + Rolling    MAE=13.73   MSE=465.88
```

SeasonalNaive fails badly because solar cycle 23 (11 years back) was much stronger than cycle 24 (which we are predicting). Each pipeline step reduces error: lag features add momentum and rolling statistics provide a stable trend estimate.

## 9. Visualize

[`plot_forecast`](/pages/api/generated/yohou.plotting.plot_forecast/) accepts a dict of predictions to overlay multiple models on one chart:

```python
from yohou.plotting import plot_forecast

plot_forecast(
    y_test,
    {"SeasonalNaive": y_pred_baseline, "Ridge + Lags + Rolling": y_pred_ridge_full},
    y_train=y_train,
    n_history=132,
    title="Sunspot Forecast Comparison",
    y_label="Monthly mean sunspot number",
)
```

The `n_history=132` parameter shows the last 11 years of training data for context. You should see SeasonalNaive dramatically overshooting the actual test values, while the Ridge pipeline tracks the real rise of cycle 24.

For a metric-level comparison, [`plot_score_summary`](/pages/api/generated/yohou.plotting.plot_score_summary/) visualizes scores as a grouped bar chart:

```python
from yohou.plotting import plot_score_summary

plot_score_summary(
    {"MAE": mae, "MSE": mse},
    y_test,
    predictions,
    title="Model Comparison: Sunspot Forecasting",
)
```

You should see SeasonalNaive with a dramatically taller bar than the Ridge pipeline models, reflecting the failure to account for the weaker cycle 24 amplitude.

## 10. Try a Stronger Model

The pipeline is regressor-agnostic. `ExtraTreesRegressor` supports multi-output natively, so it can replace Ridge without any other changes:

```python
from sklearn.ensemble import ExtraTreesRegressor

etrees = PointReductionForecaster(
    estimator=ExtraTreesRegressor(n_estimators=200, random_state=42),
    feature_transformer=FeaturePipeline([
        ("features", FeatureUnion([
            ("lags", LagTransformer(lag=list(range(1, 13)))),
            ("rolling", RollingStatisticsTransformer(window_size=12, statistics=["mean", "std"])),
        ])),
    ]),
)
etrees.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_etrees = etrees.predict(forecasting_horizon=forecasting_horizon)

predictions["ExtraTrees + Lags + Rolling"] = y_pred_etrees

for name, y_pred in predictions.items():
    print(f"{name:30s} MAE={mae.score(y_test, y_pred):.2f}  MSE={mse.score(y_test, y_pred):.2f}")
```

```text
SeasonalNaive                  MAE=102.15  MSE=12567.36
Ridge                          MAE=19.99   MSE=939.55
Ridge + Lags                   MAE=14.98   MSE=559.97
Ridge + Lags + Rolling         MAE=13.73   MSE=465.88
ExtraTrees + Lags + Rolling    MAE=13.83   MSE=393.91
```

Notice that ExtraTrees achieves similar MAE to Ridge but lower MSE, meaning fewer large prediction errors. The feature pipeline is identical: only the estimator changed. Linear models can be competitive when feature engineering is good, and tree-based models may offer lower variance on tail errors.

## What You Built

We installed Yohou and built a complete forecasting pipeline from scratch. Along the way, we:

- Loaded and resampled a real-world dataset with [`fetch_sunspot`](/pages/api/generated/yohou.datasets.fetch_sunspot/) and [`Downsampler`](/pages/api/generated/yohou.preprocessing.Downsampler/)
- Established a seasonal baseline with [`SeasonalNaive`](/pages/api/generated/yohou.point.SeasonalNaive/) and saw where it fails (cycle amplitude mismatch)
- Built a reduction pipeline step by step: [`LagTransformer`](/pages/api/generated/yohou.preprocessing.LagTransformer/) for autoregressive features, [`RollingStatisticsTransformer`](/pages/api/generated/yohou.preprocessing.RollingStatisticsTransformer/) for trend features, combined with [`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/)
- Evaluated with [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.MeanAbsoluteError/) and [`MeanSquaredError`](/pages/api/generated/yohou.metrics.MeanSquaredError/)
- Swapped in a tree-based regressor with a single parameter change

## Next Steps

- **[Forecasting Workflow](forecasting-workflow.md)**: Evaluate with cross-validation, hyperparameter search, and residual diagnostics
- **[Choose a Forecasting Method](../how-to/choose-forecasting-method.md)**: Compare estimator families and pick between reduction, naive baselines, and composition
- **[Class-Probability Forecasting](class-proba-forecasting.md)**: Forecast categorical outcomes as probability distributions
- **[Interval Forecasting](interval-forecasting.md)**: Produce prediction intervals with coverage guarantees
- **[Core Concepts](../explanation/core-concepts.md)**: Observe/rewind, panel data, and metadata routing
