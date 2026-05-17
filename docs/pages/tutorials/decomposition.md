# Decomposition

In this tutorial, we will build a [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) that separates a time series into trend, seasonality, and residual components. Along the way, we will fit a [`PolynomialTrendForecaster`](/pages/api/generated/yohou.stationarity.trend.PolynomialTrendForecaster/) and two seasonality forecasters ([`PatternSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.PatternSeasonalityForecaster/) and [`FourierSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.FourierSeasonalityForecaster/)), combine them with a [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) for residuals, and visualize each component.

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Prerequisites

- Completed [Getting Started](getting-started.md)

## 1. Prepare Data

We will use the Australian tourism dataset, a monthly series tracking tourist arrivals:

```python
from yohou.datasets import fetch_tourism_monthly
from yohou.model_selection import train_test_split

bunch = fetch_tourism_monthly(n_series=1)
y = bunch.frame
print(y.head())
```

```text
shape: (5, 2)
┌─────────────────────┬───────────┐
│ time                ┆ tourists  │
│ ---                 ┆ ---       │
│ datetime[μs]        ┆ f64       │
╞═════════════════════╪═══════════╡
│ 1979-01-01 00:00:00 ┆ 1149.87   │
│ 1979-02-01 00:00:00 ┆ 1053.8002 │
│ 1979-03-01 00:00:00 ┆ 1388.8798 │
│ 1979-04-01 00:00:00 ┆ 1783.3702 │
│ 1979-05-01 00:00:00 ┆ 1921.0252 │
└─────────────────────┴───────────┘
```

Now split the data, holding out the last 12 months as the test set using [`train_test_split`](/pages/api/generated/yohou.model_selection.split.train_test_split/):

```python
forecasting_horizon = 12
y_train, y_test = train_test_split(y, test_size=forecasting_horizon)
print(f"Train: {len(y_train)} months, Test: {len(y_test)} months")
```

```text
Train: 175 months, Test: 12 months
```

## 2. Model the Trend

[`PolynomialTrendForecaster`](/pages/api/generated/yohou.stationarity.trend.PolynomialTrendForecaster/) fits a polynomial to the time index. A degree-1 polynomial captures a linear trend:

```python
from yohou.stationarity import PolynomialTrendForecaster

trend = PolynomialTrendForecaster(degree=1)
trend.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_trend = trend.predict(forecasting_horizon=forecasting_horizon)
print(y_pred_trend.head(3))
```

```text
shape: (3, 3)
┌─────────────────────┬─────────────────────┬─────────────┐
│ vintage_time        ┆ time                ┆ tourists    │
│ ---                 ┆ ---                 ┆ ---         │
│ datetime[μs]        ┆ datetime[μs]        ┆ f64         │
╞═════════════════════╪═════════════════════╪═════════════╡
│ 1993-07-01 00:00:00 ┆ 1993-08-01 00:00:00 ┆ 3442.565913 │
│ 1993-07-01 00:00:00 ┆ 1993-09-01 00:00:00 ┆ 3451.141173 │
│ 1993-07-01 00:00:00 ┆ 1993-10-01 00:00:00 ┆ 3459.716433 │
└─────────────────────┴─────────────────────┴─────────────┘
```

Notice that the `tourists` values increase slowly from month to month, reflecting the upward linear trend.

## 3. Model the Seasonality

[`PatternSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.PatternSeasonalityForecaster/) averages historical values at each position in the seasonal cycle and repeats the pattern forward. We set `seasonality=12` for monthly data:

```python
from yohou.stationarity import PatternSeasonalityForecaster

seasonal = PatternSeasonalityForecaster(seasonality=12, method="average")
seasonal.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_seasonal = seasonal.predict(forecasting_horizon=forecasting_horizon)
print(y_pred_seasonal.head(3))
```

```text
shape: (3, 3)
┌─────────────────────┬─────────────────────┬─────────────┐
│ vintage_time        ┆ time                ┆ tourists    │
│ ---                 ┆ ---                 ┆ ---         │
│ datetime[μs]        ┆ datetime[μs]        ┆ f64         │
╞═════════════════════╪═════════════════════╪═════════════╡
│ 1993-07-01 00:00:00 ┆ 1993-08-01 00:00:00 ┆ 5626.263664 │
│ 1993-07-01 00:00:00 ┆ 1993-09-01 00:00:00 ┆ 3454.061586 │
│ 1993-07-01 00:00:00 ┆ 1993-10-01 00:00:00 ┆ 2333.7382   │
└─────────────────────┴─────────────────────┴─────────────┘
```

Notice the large swing between months: August (peak summer tourism) is much higher than October. That is the seasonal pattern at work.

[`FourierSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.FourierSeasonalityForecaster/) is an alternative that models seasonality with Fourier basis functions, producing smooth curves instead of repeating raw averages. The `harmonics` parameter controls how many sine/cosine pairs to include:

```python
from yohou.stationarity import FourierSeasonalityForecaster

fourier = FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2, 3])
fourier.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_fourier = fourier.predict(forecasting_horizon=forecasting_horizon)
print(y_pred_fourier.head(3))
```

```text
shape: (3, 3)
┌─────────────────────┬─────────────────────┬─────────────┐
│ vintage_time        ┆ time                ┆ tourists    │
│ ---                 ┆ ---                 ┆ ---         │
│ datetime[μs]        ┆ datetime[μs]        ┆ f64         │
╞═════════════════════╪═════════════════════╪═════════════╡
│ 1993-07-01 00:00:00 ┆ 1993-08-01 00:00:00 ┆ 5475.889171 │
│ 1993-07-01 00:00:00 ┆ 1993-09-01 00:00:00 ┆ 3493.766504 │
│ 1993-07-01 00:00:00 ┆ 1993-10-01 00:00:00 ┆ 2373.792979 │
└─────────────────────┴─────────────────────┴─────────────┘
```

Notice that the values are close to `PatternSeasonalityForecaster` but smoother. `FourierSeasonalityForecaster` is a good choice when the seasonal shape is gradual, or when the series is short and averaging over few cycles would be noisy. We will use `PatternSeasonalityForecaster` in the pipeline below, but you can swap it for `FourierSeasonalityForecaster`.

## 4. Build a DecompositionPipeline

Now that we have seen trend and seasonality individually, let's combine them. [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) chains forecasters in sequence: each one models the residuals left by all previous forecasters, and the final prediction is the sum of all components.

We add a [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) as the third stage to capture any structure remaining in the residuals after removing trend and seasonality:

```python
from sklearn.linear_model import Ridge
from yohou.compose import DecompositionPipeline, FeaturePipeline
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer

decomp = DecompositionPipeline(
    forecasters=[
        ("trend", PolynomialTrendForecaster(degree=1)),
        ("seasonality", PatternSeasonalityForecaster(seasonality=12)),
        ("residual", PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=FeaturePipeline([
                ("lags", LagTransformer(lag=list(range(1, 7)))),
            ]),
        )),
    ],
    store_residuals=True,
)
decomp.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_decomp = decomp.predict(forecasting_horizon=forecasting_horizon)
print(y_pred_decomp.head(3))
```

```text
shape: (3, 3)
┌─────────────────────┬─────────────────────┬─────────────┐
│ vintage_time        ┆ time                ┆ tourists    │
│ ---                 ┆ ---                 ┆ ---         │
│ datetime[μs]        ┆ datetime[μs]        ┆ f64         │
╞═════════════════════╪═════════════════════╪═════════════╡
│ 1993-07-01 00:00:00 ┆ 1993-08-01 00:00:00 ┆ 7377.468416 │
│ 1993-07-01 00:00:00 ┆ 1993-09-01 00:00:00 ┆ 5488.378737 │
│ 1993-07-01 00:00:00 ┆ 1993-10-01 00:00:00 ┆ 4231.479497 │
└─────────────────────┴─────────────────────┴─────────────┘
```

Notice that these combined predictions are higher than either the trend or seasonality predictions alone, because the pipeline sums all three component contributions.

## 5. Visualize Components

Each fitted forecaster inside the pipeline can produce its own predictions. We collect them into a dict and pass it to [`plot_decomposition`](/pages/api/generated/yohou.plotting.forecasting.plot_decomposition/), which displays each component as a separate subplot:

```python
from yohou.plotting import plot_decomposition

components = {}
for name, fc, *_ in decomp.forecasters_:
    components[name] = fc.predict(forecasting_horizon=forecasting_horizon)

fig = plot_decomposition(y_test, components)
fig.show()
```

The plot shows the trend, seasonality, and residual contributions separately. Check that the trend line rises gradually, that seasonality shows a repeating monthly pattern, and that the residuals are small relative to the other components.

## 6. Score the Pipeline

```python
from yohou.metrics import MeanAbsoluteError

mae = MeanAbsoluteError()
mae.fit(y_train)
score = mae.score(y_test, y_pred_decomp)
print(f"DecompositionPipeline MAE: {score:.2f}")
```

```text
DecompositionPipeline MAE: 950.31
```

## What You Built

We constructed a [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) that separates a time series into trend, seasonality, and residual components. Each component is modeled by a specialized forecaster, and the final prediction is their sum. We visualized the decomposition to verify that each stage captured the right structure.

## Next Steps

- [Forecaster Composition](../explanation/forecaster-composition.md) for the conceptual background on decomposition pipelines
- [Stationarity](../explanation/stationarity.md) for trend and seasonality transforms
- [Seasonal Analysis](seasonal-analysis.md) for visualizing and testing seasonal patterns
- [Handle Complex Seasonality](../how-to/handle-complex-seasonality.md) for multiple and non-integer seasonalities
