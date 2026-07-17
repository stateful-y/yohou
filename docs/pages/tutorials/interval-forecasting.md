# Interval Forecasting

In this tutorial, we will wrap a point forecaster with [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.SplitConformalForecaster/) to produce prediction intervals, visualize them as shaded bands, and measure how well the intervals cover the true values. Along the way, we will use `predict_interval`, [`plot_forecast`](/pages/api/generated/yohou.plotting.plot_forecast/), and [`EmpiricalCoverage`](/pages/api/generated/yohou.metrics.EmpiricalCoverage/).

## Prerequisites

- Completed [Getting Started](getting-started.md)

<!-- COMPANION_NOTEBOOKS -->

## 1. Load the Data

We use the tourism monthly dataset and extract a single series. The `T1__` prefix is the panel identifier (see [Core Concepts](../explanation/core-concepts.md)); we rename the column to plain `tourists`:

```python
from yohou.datasets import fetch_tourism_monthly

bunch = fetch_tourism_monthly()
y = (
    bunch.frame
    .select("time", "T1__tourists")
    .rename({"T1__tourists": "tourists"})
    .drop_nulls()
)
print(f"Series length: {len(y)} months")
```

```text
Series length: 187 months
```

We split into train and test, holding out 12 months:

```python
from yohou.model_selection import train_test_split

forecasting_horizon = 12
y_train, y_test = train_test_split(y, test_size=forecasting_horizon)
print(f"Train: {len(y_train)} months, Test: {forecasting_horizon} months")
```

```text
Train: 175 months, Test: 12 months
```

## 2. Build a Point Forecaster

We first build a reduction forecaster with lag features and seasonal differencing. This is the same pattern as [Getting Started](getting-started.md):

```python
from yohou.compose import FeaturePipeline
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer
from yohou.stationarity import SeasonalDifferencing
from sklearn.linear_model import Ridge

point_forecaster = PointReductionForecaster(
    estimator=Ridge(),
    target_transformer=SeasonalDifferencing(seasonality=12),
    feature_transformer=FeaturePipeline([
        ("lags", LagTransformer(lag=[1, 2, 3, 12])),
    ]),
)
```

## 3. Wrap with SplitConformalForecaster

[`SplitConformalForecaster`](/pages/api/generated/yohou.interval.SplitConformalForecaster/) takes any point forecaster and wraps it to produce prediction intervals. It holds out `calibration_size` training observations to measure forecasting errors, then constructs intervals wide enough to cover 95% of those errors:

```python
from yohou.interval import SplitConformalForecaster

conformal = SplitConformalForecaster(
    point_forecaster=point_forecaster,
    calibration_size=24,
)
conformal.fit(y_train, forecasting_horizon=forecasting_horizon)
```

Notice that the API follows the same `fit`/`predict` pattern as every other forecaster in Yohou.

## 4. Predict Intervals

Now we call `predict_interval` to get the lower and upper bounds. The default coverage rate is 0.95 (a 95% prediction interval):

```python
y_pred_int = conformal.predict_interval(forecasting_horizon=forecasting_horizon)
print(y_pred_int.head(4))
```

The output should look something like:

```text
shape: (4, 4)
┌─────────────────────┬─────────────────────┬────────────────────────┬───────────────────────┐
│ vintage_time        ┆ time                ┆ tourists_lower_0.95    ┆ tourists_upper_0.95   │
│ ---                 ┆ ---                 ┆ ---                    ┆ ---                   │
│ datetime[μs]        ┆ datetime[μs]        ┆ f64                    ┆ f64                   │
╞═════════════════════╪═════════════════════╪════════════════════════╪═══════════════════════╡
│ 1993-07-01 00:00:00 ┆ 1993-08-01 00:00:00 ┆ 6313.553540            ┆ 7096.259105           │
│ 1993-07-01 00:00:00 ┆ 1993-09-01 00:00:00 ┆ 3948.652368            ┆ 4699.295620           │
│ 1993-07-01 00:00:00 ┆ 1993-10-01 00:00:00 ┆ 2700.083519            ┆ 3408.338242           │
│ 1993-07-01 00:00:00 ┆ 1993-11-01 00:00:00 ┆ 1657.301340            ┆ 2393.047151           │
└─────────────────────┴─────────────────────┴────────────────────────┴───────────────────────┘
```

Notice the column names: `tourists_lower_0.95` and `tourists_upper_0.95`, derived from the target column name and the coverage rate. Each row gives the interval for one forecast step.

## 5. Visualize the Intervals

[`plot_forecast`](/pages/api/generated/yohou.plotting.plot_forecast/) detects the interval columns automatically and renders them as a shaded band:

```python
from yohou.plotting import plot_forecast

fig = plot_forecast(y_test, y_pred_int, y_train=y_train[-24:])
fig.show()
```

You should see the forecast line with a shaded 95% band around it. The training history appears on the left, and the test actuals overlay the forecast period.

## 6. Check Empirical Coverage

How many of the 12 test points actually fall inside the predicted intervals? [`EmpiricalCoverage`](/pages/api/generated/yohou.metrics.EmpiricalCoverage/) measures this:

```python
from yohou.metrics import EmpiricalCoverage

cov_scorer = EmpiricalCoverage()
cov_scorer.fit(y_train)
empirical_cov = cov_scorer.score(y_test, y_pred_int)
print(f"Target coverage:    0.95")
print(f"Empirical coverage: {empirical_cov:.3f}")
```

The output should look something like:

```text
Target coverage:   0.95
Empirical coverage: 0.750
```

The empirical coverage of 0.75 is below the 0.95 target. With only 12 test observations and 24 calibration points, this gap is expected: it is a small-sample artifact, not a model failure. As both the calibration set and the test set grow, the empirical coverage converges toward the target rate. See [Interval Forecasting](../explanation/interval-forecasting.md) for the theory behind this guarantee.

## What You Built

We wrapped a point forecaster with [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.SplitConformalForecaster/) to produce 95% prediction intervals, visualized them as shaded bands with [`plot_forecast`](/pages/api/generated/yohou.plotting.plot_forecast/), and measured the empirical coverage with [`EmpiricalCoverage`](/pages/api/generated/yohou.metrics.EmpiricalCoverage/). Any point forecaster in Yohou can be wrapped this way to produce calibrated intervals.

## Next Steps

- [Class-Probability Forecasting](class-proba-forecasting.md): Forecast categorical outcomes as probability distributions over classes
- [Forecasting Workflow](forecasting-workflow.md): Cross-validation, hyperparameter search, and residual diagnostics
- [Interval Forecasting](../explanation/interval-forecasting.md): Conformal prediction theory, coverage guarantees, and calibration sizing
- [How to Produce Prediction Intervals](../how-to/interval-forecasting.md): Practical recipes for configuring and scoring interval forecasters
