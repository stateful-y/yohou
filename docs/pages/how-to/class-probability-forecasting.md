# How to Forecast with Class Probabilities

This guide shows you how to fit a class-probability forecaster, obtain per-class
probability predictions, and evaluate them with classification metrics. Reach
for this workflow when you need per-class probabilities (for risk-aware
decisions, calibration analysis, or proper scoring rules) rather than the hard
labels a point forecaster predicts for categorical targets.

## Prerequisites

- Familiarity with the fit-predict workflow ([Getting Started](../tutorials/getting-started.md))
- Familiarity with train/test evaluation ([Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Prepare Data and Train/Test Split

Use one of the built-in classification datasets, or prepare your own DataFrame
with a `"time"` column and one or more string-valued target columns. Split
the data before fitting so the evaluation later reflects true out-of-sample
performance:

```python
from yohou.datasets import fetch_air_quality_classification
from yohou.model_selection import train_test_split

data = fetch_air_quality_classification()
y = data.y
# DataFrame with "time" and "air_quality" columns
# air_quality values: "good", "moderate", "unhealthy", "hazardous"

y_train, y_test = train_test_split(y, test_size=24)
```

## Fit a Class-Probability Forecaster

[`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.reduction.ClassProbaReductionForecaster/) wraps any scikit-learn classifier that supports
`predict_proba()`. The default estimator is
[`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html);
any classifier with `fit()`, `predict()`, and `predict_proba()` works:

```python
from sklearn.ensemble import GradientBoostingClassifier
from yohou.class_proba import ClassProbaReductionForecaster

forecaster = ClassProbaReductionForecaster(
    estimator=GradientBoostingClassifier(n_estimators=100),
)
forecaster.fit(y_train, forecasting_horizon=24)
```

## Get Predictions

**Soft probabilities** (recommended for decision-making):

```python
y_proba = forecaster.predict_class_proba()
# Columns: vintage_time, time, air_quality_proba_good, air_quality_proba_moderate, ...
```

**Hard labels** (argmax of probabilities):

```python
y_pred = forecaster.predict()
# Columns: vintage_time, time, air_quality
```

## Evaluate with Classification Metrics

Proper scoring rules give reliable model comparisons because they reward
calibrated probabilities. Prefer [`LogLoss`](/pages/api/generated/yohou.metrics.class_proba.LogLoss/) or [`BrierScore`](/pages/api/generated/yohou.metrics.class_proba.BrierScore/) over
[`Accuracy`](/pages/api/generated/yohou.metrics.classification.Accuracy/) for model selection:

```python
from yohou.metrics import LogLoss, BrierScore, Accuracy

log_loss = LogLoss().fit(y_train).score(y_test, y_proba)
brier = BrierScore().fit(y_train).score(y_test, y_proba)
accuracy = Accuracy().fit(y_train).score(y_test, y_pred)
```

## Visualize Results

[`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) auto-detects categorical and probability columns:

```python
from yohou.plotting import plot_forecast

# Hard labels: step chart
plot_forecast(y_test, y_pred)

# Probabilities: stacked area chart
plot_forecast(y_test, y_proba)
```

Use [`plot_calibration`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/) to assess whether predicted probabilities match observed
frequencies:

```python
from yohou.plotting import plot_calibration

plot_calibration(y_proba, y_test)
```

## See Also

- [Class-Probability Forecasting](../explanation/class-probability-forecasting.md) for theory and mathematical details
- [Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md) for the complete metrics guide
- [API Reference: yohou.class_proba](../api/class_proba.md)
