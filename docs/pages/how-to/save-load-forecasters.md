# How to Save and Load Forecasters

This guide shows you how to serialize trained forecasters to disk and load
them in a new session to produce predictions without retraining.

## Prerequisites

- A fitted forecaster ([Getting Started](../tutorials/getting-started.md))
- Understanding of the fit/observe/predict lifecycle ([Core Concepts](../explanation/core-concepts.md))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## 1. Save a Fitted Forecaster

Use `joblib` to serialize the forecaster after fitting. `joblib` handles large
NumPy arrays more efficiently than `pickle` and is the recommended approach
for scikit-learn compatible estimators:

```python
import joblib
from sklearn.linear_model import Ridge
from yohou.datasets import fetch_electricity_demand
from yohou.model_selection import train_test_split
from yohou.point import PointReductionForecaster

data = fetch_electricity_demand()
y_train, y_test = train_test_split(data.frame, test_size=48)

forecaster = PointReductionForecaster(estimator=Ridge())
forecaster.fit(y_train, forecasting_horizon=48)

joblib.dump(forecaster, "forecaster.joblib")
```

## 2. Load and Predict

Load the saved forecaster and call `predict` without calling `fit` again:

```python
import joblib

forecaster = joblib.load("forecaster.joblib")
y_pred = forecaster.predict(forecasting_horizon=48)
```

The loaded forecaster retains all fitted state: learned parameters,
observation history, and pipeline transformers.

## 3. Use pickle as an Alternative

Standard library `pickle` works for all yohou forecasters. Prefer `joblib`
when the estimator contains large arrays; use `pickle` when you want to avoid
the extra dependency:

```python
import pickle

# Save
with open("forecaster.pkl", "wb") as f:
    pickle.dump(forecaster, f)

# Load
with open("forecaster.pkl", "rb") as f:
    forecaster = pickle.load(f)
```

!!! warning "Security: never load untrusted pickle files"
    Both `pickle` and `joblib` files can execute arbitrary code when loaded.
    Only load files you created yourself or received from a trusted source.
    For sharing across trust boundaries, consider exporting predictions as
    CSV or Parquet instead of serialized objects.

## Version Compatibility

A saved forecaster is tied to the versions of yohou and its dependencies
(scikit-learn, polars) that were installed when the file was created. Loading
a forecaster saved with a different version may raise errors or produce
silent changes in behaviour. Pin versions in your deployment environment and
re-save the forecaster whenever you upgrade.

## See Also

- [Forecasting Workflow](../tutorials/forecasting-workflow.md) for the full
  training and evaluation pipeline that produces a fitted forecaster.
- [Core Concepts](../explanation/core-concepts.md) for the fit/observe/predict
  lifecycle and what state the forecaster carries after fitting.
- [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) API reference.
