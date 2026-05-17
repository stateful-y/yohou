# Forecast Visualization

In this tutorial, we will compare two models' forecasts visually, add prediction intervals to quantify uncertainty, check whether those intervals are well-calibrated, inspect a decomposition to understand what each model component contributes, and plot time weights to see how the training emphasis is distributed.

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Prerequisites

- Completed [Getting Started](getting-started.md)
- Completed [Exploratory Visualization](exploratory-visualization.md)

## 1. Prepare Data and Models

```python
from sklearn.linear_model import Ridge
from yohou.compose import FeaturePipeline
from yohou.datasets import fetch_tourism_monthly
from yohou.model_selection import train_test_split
from yohou.plotting import plot_forecast
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.preprocessing import LagTransformer

bunch = fetch_tourism_monthly(n_series=1)
y = bunch.frame

forecasting_horizon = 12
y_train, y_test = train_test_split(y, test_size=forecasting_horizon)

baseline = SeasonalNaive(seasonality=12)
baseline.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_baseline = baseline.predict(forecasting_horizon=forecasting_horizon)

ridge = PointReductionForecaster(
    estimator=Ridge(),
    feature_transformer=FeaturePipeline([
        ("lags", LagTransformer(lag=list(range(1, 13)))),
    ]),
)
ridge.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_ridge = ridge.predict(forecasting_horizon=forecasting_horizon)
```

## 2. Single Forecast Plot

Start by plotting one model in isolation with [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) to see how its predictions align with the test data:

```python
fig = plot_forecast(y_test, y_pred_baseline, y_train=y_train[-24:])
fig.show()
```

The plot shows the training history on the left, the test actuals, and the forecast overlay. Look for systematic over- or under-prediction and whether the forecast captures the seasonal shape.

## 3. Multi-Model Comparison

Pass a dictionary of predictions to compare models side by side:

```python
fig = plot_forecast(
    y_test,
    {"SeasonalNaive": y_pred_baseline, "Ridge": y_pred_ridge},
    y_train=y_train[-24:],
)
fig.show()
```

Each model gets a distinct color. The legend lets you toggle individual models on and off. Notice where the two forecasts diverge: the Ridge model may track the test data more closely, suggesting it is the better candidate. But point forecasts alone do not tell us how confident the model is.

## 4. Prediction Intervals

Now that we can see which model tracks the test data better, let's quantify how uncertain those predictions are. Wrap the Ridge forecaster with [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/) to add prediction intervals:

```python
from yohou.interval import SplitConformalForecaster

conformal = SplitConformalForecaster(
    point_forecaster=ridge,
    calibration_size=24,
)
conformal.fit(
    y_train,
    forecasting_horizon=forecasting_horizon,
    coverage_rates=[0.90],
)
y_pred_int = conformal.predict_interval(forecasting_horizon=forecasting_horizon)

fig = plot_forecast(y_test, y_pred_int, y_train=y_train[-24:])
fig.show()
```

The prediction interval appears as a shaded band around the forecast line. Narrow bands indicate high confidence; wide bands warn that the model is uncertain about those time steps.

## 5. Calibration Diagram

The prediction intervals look reasonable visually, but do they actually achieve their claimed 90% coverage? [`plot_calibration`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/) checks this:

```python
from yohou.plotting import plot_calibration

fig = plot_calibration(y_pred_int, y_test)
fig.show()
```

Points close to the diagonal indicate well-calibrated intervals. If points fall below the diagonal, the model is overconfident (intervals are too narrow). If above, the intervals are conservative.

## 6. Decomposition Visualization

With the forecast comparison and calibration settled, let's look inside a structured model with [`plot_decomposition`](/pages/api/generated/yohou.plotting.forecasting.plot_decomposition/) to understand what each component contributes:

```python
from yohou.compose import DecompositionPipeline
from yohou.plotting import plot_decomposition
from yohou.stationarity import PatternSeasonalityForecaster, PolynomialTrendForecaster

decomp = DecompositionPipeline(
    forecasters=[
        ("trend", PolynomialTrendForecaster(degree=1)),
        ("seasonality", PatternSeasonalityForecaster(seasonality=12)),
    ],
    store_residuals=True,
)
decomp.fit(y_train, forecasting_horizon=forecasting_horizon)

components = {}
for name, fc, *_ in decomp.forecasters_:
    components[name] = fc.predict(forecasting_horizon=forecasting_horizon)

fig = plot_decomposition(y_test, components)
fig.show()
```

Each component appears as a separate subplot showing its contribution. Check that the trend captures the long-term direction without absorbing seasonal variation, and that the residuals look like noise rather than structured signal.

## 7. Time Weight Visualization

Finally, let's examine how the training data is weighted with [`plot_time_weight`](/pages/api/generated/yohou.plotting.forecasting.plot_time_weight/), since this affects which historical periods influence the model most:

```python
from yohou.plotting import plot_time_weight
from yohou.utils.weighting import exponential_decay_weight

weight_fn = exponential_decay_weight(half_life=30)
y_weighted = y_train.with_columns(
    weight_fn(y_train["time"]).alias("time_weight")
)
fig = plot_time_weight(y_weighted)
fig.show()
```

The plot shows the weight assigned to each training observation. Exponential decay concentrates weight on recent observations.

## What You Built

We followed a complete model comparison workflow: visually compared two forecasters, added prediction intervals to quantify uncertainty, verified calibration to ensure those intervals are trustworthy, decomposed a structured model to understand what each component captures, and examined time weights to see how historical emphasis is distributed. This sequence (compare, quantify, calibrate, decompose, weight) gives you a systematic way to evaluate any forecaster before deployment.

## Next Steps

- [Visualization](../explanation/visualization.md) for the conceptual overview of the plotting workflow
- [How to Visualize Scores](../how-to/visualize-scores.md) for scoring visualization patterns
