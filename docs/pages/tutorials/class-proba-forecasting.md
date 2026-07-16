# Class-Probability Forecasting

In this tutorial, we will forecast air quality categories using [`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.ClassProbaReductionForecaster/), producing a probability distribution over four WHO air quality classes instead of a single numeric value. We will load a real dataset with hourly pollution readings, fit a classifier-backed forecaster, and evaluate the probabilistic output with the Brier score and accuracy.

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Prerequisites

- Completed [Getting Started](getting-started.md)

## 1. Load and Inspect the Data

The air quality classification dataset contains hourly PM2.5 readings from Beijing (2017 to 2019), labelled with one of four WHO air quality classes: good, moderate, unhealthy, and hazardous. Five pollutant features are available as `X_actual`: PM10, NO2, CO, O3, and SO2.

```python
from yohou.datasets import fetch_air_quality_classification

bunch = fetch_air_quality_classification()
y = bunch.y
X_actual = bunch.X_actual

print(f"Series length: {len(y)} hours")
print(f"Classes: {bunch.classes}")
print(f"Features: {bunch.feature_names}")
print(y.head(3))
```

```text
Series length: 10898 hours
Classes: ['good', 'hazardous', 'moderate', 'unhealthy']
Features: ['pm10', 'no2', 'co', 'o3', 'so2']
shape: (3, 2)
┌─────────────────────┬─────────────┐
│ time                ┆ air_quality │
│ ---                 ┆ ---         │
│ datetime[μs]        ┆ str         │
╞═════════════════════╪═════════════╡
│ 2017-01-01 14:00:00 ┆ hazardous   │
│ 2017-01-01 15:00:00 ┆ hazardous   │
│ 2017-01-01 16:00:00 ┆ hazardous   │
└─────────────────────┴─────────────┘
```

## 2. Train/Test Split

We split the data so that the last 24 hours form the test set (one full day ahead) using [`train_test_split`](/pages/api/generated/yohou.model_selection.train_test_split/):

```python
from yohou.model_selection import train_test_split

forecasting_horizon = 24

y_train, y_test, X_train, X_test = train_test_split(
    y, X_actual, test_size=forecasting_horizon
)

print(f"Train: {len(y_train)} hours, Test: {len(y_test)} hours")
print(f"Test class distribution:")
print(y_test["air_quality"].value_counts())
```

```text
Train: 10874 hours, Test: 24 hours
Test class distribution:
shape: (2, 2)
┌─────────────┬───────┐
│ air_quality ┆ count │
│ ---         ┆ ---   │
│ str         ┆ u32   │
╞═════════════╪═══════╡
│ good        ┆ 18    │
│ moderate    ┆ 6     │
└─────────────┴───────┘
```

## 3. Fit the Forecaster

[`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.ClassProbaReductionForecaster/) wraps any Scikit-Learn classifier that supports `predict_proba` and uses a reduction strategy to produce forecasts for each step in the horizon. We pass `X_actual` at fit time so the model learns to use pollutant readings as features:

```python
from yohou.class_proba import ClassProbaReductionForecaster
from yohou.compose import FeaturePipeline
from yohou.preprocessing import LagTransformer
from sklearn.ensemble import RandomForestClassifier

forecaster = ClassProbaReductionForecaster(
    estimator=RandomForestClassifier(n_estimators=50, random_state=42),
    feature_transformer=FeaturePipeline([
        ("lags", LagTransformer(lag=[1, 2, 3, 24])),
    ]),
)
forecaster.fit(y_train, forecasting_horizon=forecasting_horizon, X_actual=X_train)
```

[`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/) chains feature transformers sequentially, just like sklearn's [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). Here we use a single [`LagTransformer`](/pages/api/generated/yohou.preprocessing.LagTransformer/) that creates autoregressive features from 1, 2, 3, and 24 hours back.

## 4. Predict Class Probabilities

Calling `predict_class_proba` produces a probability distribution over all four classes for each hour in the forecast horizon. We pass `X_test` so the model can use pollutant readings from the test window:

```python
y_pred_proba = forecaster.predict_class_proba(
    forecasting_horizon=forecasting_horizon,
    X_actual=X_test,
)
print(y_pred_proba.columns)
print(y_pred_proba.head(4))
```

```text
['vintage_time', 'time', 'air_quality_proba_good', 'air_quality_proba_hazardous', 'air_quality_proba_moderate', 'air_quality_proba_unhealthy']
shape: (4, 6)
┌─────────────────────┬─────────────────────┬────────────────────────┬──────────────────────────┬────────────────────────┬──────────────────────────┐
│ vintage_time        ┆ time                ┆ air_quality_proba_good ┆ air_quality_proba_hazard ┆ air_quality_proba_mode ┆ air_quality_proba_unheal │
│ ---                 ┆ ---                 ┆ ---                    ┆ ous                      ┆ rate                   ┆ thy                      │
│ datetime[μs]        ┆ datetime[μs]        ┆ f64                    ┆ ---                      ┆ ---                    ┆ ---                      │
│                     ┆                     ┆                        ┆ f64                      ┆ f64                    ┆ f64                      │
╞═════════════════════╪═════════════════════╪════════════════════════╪══════════════════════════╪════════════════════════╪══════════════════════════╡
│ 2017-07-28 21:00:00 ┆ 2017-07-28 22:00:00 ┆ 0.02                   ┆ 0.0                      ┆ 0.8                    ┆ 0.18                     │
│ 2017-07-28 21:00:00 ┆ 2017-07-28 23:00:00 ┆ 0.1                    ┆ 0.0                      ┆ 0.66                   ┆ 0.24                     │
│ 2017-07-28 21:00:00 ┆ 2017-07-29 00:00:00 ┆ 0.0                    ┆ 0.0                      ┆ 0.72                   ┆ 0.28                     │
│ 2017-07-28 21:00:00 ┆ 2017-07-29 01:00:00 ┆ 0.04                   ┆ 0.04                     ┆ 0.64                   ┆ 0.28                     │
└─────────────────────┴─────────────────────┴────────────────────────┴──────────────────────────┴────────────────────────┴──────────────────────────┘
```

The first line shows the full column names. Polars may truncate long names in the table display, but the actual column names are `air_quality_proba_good`, `air_quality_proba_hazardous`, `air_quality_proba_moderate`, and `air_quality_proba_unhealthy`. All four probabilities sum to 1.0 for each row, forming a complete probability distribution over the outcome space.

!!! warning "Probabilities may be miscalibrated"
    The values in the `_proba_` columns sum to 1.0, but that does not guarantee
    they reflect true likelihoods. Most classifiers produce scores that are
    only approximately calibrated. For example, a predicted 0.8 for "moderate"
    does not necessarily mean the event occurs 80% of the time. If reliable
    probability estimates matter for your application, consider calibrating the
    underlying classifier (e.g., with [`CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)) before passing
    it to [`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.ClassProbaReductionForecaster/).

## 5. Evaluate

We score the probabilistic forecasts with [`BrierScore`](/pages/api/generated/yohou.metrics.BrierScore/), which measures calibration quality (lower is better), and with [`Accuracy`](/pages/api/generated/yohou.metrics.Accuracy/), which evaluates the argmax class prediction against the true label (higher is better). Both scorers require a `.fit(y_train)` call to infer the panel structure and time interval from the training data:

```python
from yohou.metrics import Accuracy, BrierScore

brier = BrierScore()
brier.fit(y_train)

accuracy = Accuracy()
accuracy.fit(y_train)

print(f"Brier score: {brier.score(y_test, y_pred_proba):.3f}  (lower is better)")
print(f"Accuracy:    {accuracy.score(y_test, y_pred_proba):.3f}  (higher is better)")
```

```text
Brier score: 0.521  (lower is better)
Accuracy:    0.667  (higher is better)
```

A Brier score near 0 means the predicted probabilities are concentrated around the correct class. An accuracy of 0.667 means the argmax class was correct for roughly two thirds of the 24-hour forecast horizon.

!!! note "Brier score is a proper scoring rule"
    The Brier score is preferred over accuracy for evaluating probabilistic
    forecasts because it rewards well-calibrated probabilities, not just correct
    argmax predictions. A model that predicts 51% for the correct class scores
    the same accuracy as one that predicts 99%, but the Brier score distinguishes
    them. See [Forecast Accuracy](../explanation/forecast-accuracy.md) for more
    on proper scoring rules.

## What You Built

We built a complete class-probability forecasting workflow. Along the way, we:

- Loaded a real-world air quality dataset with [`fetch_air_quality_classification`](/pages/api/generated/yohou.datasets.fetch_air_quality_classification/) and split it with [`train_test_split`](/pages/api/generated/yohou.model_selection.train_test_split/)
- Fit a [`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.ClassProbaReductionForecaster/) backed by a Random Forest classifier with lag features
- Called `predict_class_proba` to produce a four-class probability distribution for each hour
- Evaluated calibration with [`BrierScore`](/pages/api/generated/yohou.metrics.BrierScore/) and classification accuracy with [`Accuracy`](/pages/api/generated/yohou.metrics.Accuracy/)

## Next Steps

- [Class-Probability Forecasting](../explanation/class-probability-forecasting.md): Understand how the reduction strategy works under the hood and when to prefer probabilistic over hard-class outputs
- [Interval Forecasting](interval-forecasting.md): Apply a complementary approach to uncertainty quantification for continuous targets
- [How to Forecast with Class Probabilities](../how-to/class-probability-forecasting.md): Practical recipes for fitting, scoring, and interpreting class-probability forecasters
