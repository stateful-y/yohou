# How to Forecast Categorical Time Series

This guide shows how to forecast categorical (non-numeric) time series using
class-probability forecasters.

**Prerequisites**: Familiarity with the fit-predict workflow. See
[Your First Forecast](../tutorials/first-forecast.md) if needed.

## 1. Load a Classification Dataset

Use one of the built-in classification datasets, or prepare your own DataFrame with
a `"time"` column and one or more string-valued target columns:

```python
from yohou.datasets import fetch_air_quality_classification

data = fetch_air_quality_classification()
y = data.y
# DataFrame with "time" and "air_quality" columns
# air_quality values: "good", "moderate", "unhealthy", "hazardous"
```

## 2. Fit a Class-Probability Forecaster

[`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.reduction.ClassProbaReductionForecaster/) wraps any sklearn classifier that supports
`predict_proba()`:

```python
from sklearn.ensemble import GradientBoostingClassifier
from yohou.class_proba import ClassProbaReductionForecaster

forecaster = ClassProbaReductionForecaster(
    estimator=GradientBoostingClassifier(n_estimators=100),
)
forecaster.fit(y, forecasting_horizon=24)
```

The default estimator is `LogisticRegression`. Any classifier with `fit()`,
`predict()`, and `predict_proba()` works.

## 3. Get Predictions

**Soft probabilities** (recommended for decision-making):

```python
y_proba = forecaster.predict_class_proba()
# Columns: time, vintage_time, air_quality_proba_good, air_quality_proba_moderate, ...
```

**Hard labels** (argmax of probabilities):

```python
y_pred = forecaster.predict()
# Columns: time, vintage_time, air_quality
```

## 4. Evaluate with Classification Metrics

Use proper scoring rules for reliable model comparison:

```python
from yohou.metrics import LogLoss, BrierScore, Accuracy

y_test = y[-24:]  # last 24 observations as ground truth

log_loss = LogLoss().fit(y_test).score(y_test, y_proba)
brier = BrierScore().fit(y_test).score(y_test, y_proba)
accuracy = Accuracy().fit(y_test).score(y_test, y_proba)
```

Prefer [`LogLoss`](/pages/api/generated/yohou.metrics.class_proba.LogLoss/) or [`BrierScore`](/pages/api/generated/yohou.metrics.class_proba.BrierScore/) over [`Accuracy`](/pages/api/generated/yohou.metrics.class_proba.Accuracy/) for model selection. They are
proper scoring rules that reward calibrated probabilities.

## 5. Visualize Results

[`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) auto-detects categorical columns and probability columns:

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

plot_calibration(y_test, y_proba)
```

### Related pages

- [Class-Probability Forecasting](../explanation/class-probability-forecasting.md): theory and mathematical details
- [Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md): complete metrics guide
- [API Reference: yohou.class_proba](../api/class_proba.md)
- [Class-Probability Examples](../examples/class_proba.md)
