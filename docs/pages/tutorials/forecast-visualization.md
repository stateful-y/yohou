# Forecast Visualization

In this tutorial, we will compare two models' forecasts visually, add prediction intervals to quantify uncertainty, check whether those intervals are well-calibrated, inspect a decomposition to understand what each model component contributes, and plot time weights to see how the training emphasis is distributed.

## Prerequisites

- Completed [Getting Started](getting-started.md)
- Completed [Exploratory Visualization](exploratory-visualization.md)

<!-- COMPANION_NOTEBOOKS -->

## 1. Prepare Data and Models

We set up two forecasters: a [`SeasonalNaive`](/pages/api/generated/yohou.point.SeasonalNaive/) baseline and a [`PointReductionForecaster`](/pages/api/generated/yohou.point.PointReductionForecaster/) with Ridge regression and a [`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/) containing [`LagTransformer`](/pages/api/generated/yohou.preprocessing.LagTransformer/) features.

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
    actual_transformer=FeaturePipeline([
        ("lags", LagTransformer(lag=list(range(1, 13)))),
    ]),
)
ridge.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_ridge = ridge.predict(forecasting_horizon=forecasting_horizon)
```

## 2. Single Forecast Plot

Start by plotting one model in isolation with [`plot_forecast`](/pages/api/generated/yohou.plotting.plot_forecast/) to see how its predictions align with the test data:

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

Now that we can see which model tracks the test data better, let's quantify how uncertain those predictions are. Wrap the Ridge forecaster with [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.SplitConformalForecaster/) to add prediction intervals:

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

The prediction intervals look reasonable visually, but do they actually achieve their claimed 90% coverage? [`plot_calibration`](/pages/api/generated/yohou.plotting.plot_calibration/) checks this:

```python
from yohou.plotting import plot_calibration

fig = plot_calibration(y_pred_int, y_test)
fig.show()
```

Points close to the diagonal indicate well-calibrated intervals. If points fall below the diagonal, the model is overconfident (intervals are too narrow). If above, the intervals are conservative.

## 6. Decomposition Visualization

With the forecast comparison and calibration settled, let's look inside a structured model by building a [`DecompositionPipeline`](/pages/api/generated/yohou.compose.DecompositionPipeline/) with a [`PolynomialTrendForecaster`](/pages/api/generated/yohou.stationarity.PolynomialTrendForecaster/) and a [`PatternSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.PatternSeasonalityForecaster/), then visualizing it with [`plot_decomposition`](/pages/api/generated/yohou.plotting.plot_decomposition/) to understand what each component contributes:

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

Finally, let's examine how the training data is weighted with [`plot_time_weight`](/pages/api/generated/yohou.plotting.plot_time_weight/) by computing [`ExponentialDecayWeighter`](/pages/api/generated/yohou.weighting.ExponentialDecayWeighter/) weights and attaching them as a column, since this affects which historical periods influence the model most:

```python
from yohou.plotting import plot_time_weight
from yohou.weighting import ExponentialDecayWeighter

weighter = ExponentialDecayWeighter(half_life=30)
y_weighted = y_train.with_columns(
    weighter.compute_weights(y_train["time"]).alias("time_weight")
)
fig = plot_time_weight(y_weighted)
fig.show()
```

The plot shows the weight assigned to each training observation. Exponential decay concentrates weight on recent observations.

## 8. Categorical Forecast Visualization

[`plot_forecast`](/pages/api/generated/yohou.plotting.plot_forecast/) also handles categorical time series. When predictions contain `String` or `Categorical` columns, the plot renders step traces instead of continuous lines. Wrap a classifier with [`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.ClassProbaReductionForecaster/) and plot both hard predictions and probability distributions:

```python
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from yohou.class_proba import ClassProbaReductionForecaster

# Discretize the target into categories
y_cat = y.with_columns(
    pl.when(pl.col("tourists") < 2_000).then(pl.lit("low"))
    .when(pl.col("tourists") < 3_500).then(pl.lit("medium"))
    .otherwise(pl.lit("high"))
    .alias("demand")
).select("time", "demand")

y_cat_train, y_cat_test = train_test_split(y_cat, test_size=forecasting_horizon)

cls_forecaster = ClassProbaReductionForecaster(
    estimator=RandomForestClassifier(n_estimators=50, random_state=42),
)
cls_forecaster.fit(y_cat_train, forecasting_horizon=forecasting_horizon)

y_cat_pred = cls_forecaster.predict(forecasting_horizon=forecasting_horizon)
fig = plot_forecast(y_cat_test, y_cat_pred, y_train=y_cat_train[-24:])
fig.show()
```

If you also call `predict_class_proba()`, passing the result to `plot_forecast` renders a stacked area chart where each class is a coloured band whose height equals the predicted probability, with diamond markers showing the true class. See the companion notebook for the full interactive example.

## What You Built

We followed a complete model comparison workflow: visually compared two forecasters, added prediction intervals to quantify uncertainty, verified calibration to ensure those intervals are trustworthy, decomposed a structured model to understand what each component captures, and examined time weights to see how historical emphasis is distributed. This sequence (compare, quantify, calibrate, decompose, weight) gives you a systematic way to evaluate any forecaster before deployment.

## Next Steps

- [Interval Forecasting](interval-forecasting.md): Build and evaluate prediction intervals step by step
- [Interval Forecasting](../explanation/interval-forecasting.md): Coverage guarantees and calibration sizing
- [Visualization](../explanation/visualization.md) for the conceptual overview of the plotting workflow
- [How to Visualize Scores](../how-to/visualize-scores.md) for scoring visualization patterns
