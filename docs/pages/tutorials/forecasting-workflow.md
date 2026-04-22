# Forecasting Workflow

This tutorial covers the evaluation side of forecasting: cross-validation, hyperparameter search, and residual diagnostics. By the end you will have compared multiple models with temporal cross-validation and diagnosed model weaknesses through residual analysis.

## Prerequisites

- Completed [Your First Forecast](first-forecast.md)
- A Python 3.11+ environment

## Setup

We use the monthly tourism dataset (a different series than the sunspot data in [Your First Forecast](first-forecast.md)) and build two forecasters to compare:

```python
from sklearn.linear_model import Ridge
from yohou.compose import FeaturePipeline
from yohou.datasets import fetch_tourism_monthly
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.preprocessing import LagTransformer
from yohou.stationarity import SeasonalDifferencing

bunch = fetch_tourism_monthly()
y = bunch.frame.select("time", "T1__tourists").drop_nulls()

y_train = y.head(-24)
y_test = y.tail(24)

baseline = SeasonalNaive(seasonality=12)
baseline.fit(y_train, forecasting_horizon=24)
y_pred_baseline = baseline.predict(forecasting_horizon=24)

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    target_transformer=SeasonalDifferencing(seasonality=12),
    feature_transformer=FeaturePipeline([
        ("lags", LagTransformer(lag=[1, 2, 3, 12])),
    ]),
)
forecaster.fit(y_train, forecasting_horizon=24)
y_pred_ridge = forecaster.predict(forecasting_horizon=24)
```

For how this pipeline was built step by step, see [Your First Forecast](first-forecast.md).

## Score with Multiple Metrics

Start by scoring both models on a single train/test split:

```python
from yohou.metrics import MeanAbsoluteError, MeanAbsoluteScaledError

mae = MeanAbsoluteError()
mae.fit(y_train)
mase = MeanAbsoluteScaledError(seasonality=12)
mase.fit(y_train)

for name, y_pred in [("SeasonalNaive", y_pred_baseline), ("Ridge", y_pred_ridge)]:
    print(f"{name:15s} MAE={mae.score(y_test, y_pred):.2f}  MASE={mase.score(y_test, y_pred):.2f}")
```

A MASE below 1.0 means the model outperforms the seasonal naive baseline. But a single split can be misleading: the model might perform well (or poorly) by luck depending on which period was held out.

## Evaluate with Cross-Validation

[`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.ExpandingWindowSplitter/) creates multiple temporal train/test folds by growing the training window while keeping the test size fixed. This gives a more robust performance estimate:

```python
from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV

cv = ExpandingWindowSplitter(n_splits=3, test_size=24)

search = GridSearchCV(
    forecaster=forecaster,
    param_grid={},
    scoring=MeanAbsoluteScaledError(seasonality=12),
    cv=cv,
)
search.fit(y, forecasting_horizon=24)

print(f"CV Mean MASE: {search.best_score_:.2f}")
```

Passing an empty `param_grid` runs the forecaster with its current parameters across all folds. The `best_score_` attribute reports the mean score across folds.

## Hyperparameter Search

`GridSearchCV` also searches over parameter combinations. Define a grid using the nested parameter syntax (`estimator__param` to access regressor parameters, `feature_transformer__step_name__param` for pipeline steps):

```python
search = GridSearchCV(
    forecaster=forecaster,
    param_grid={
        "estimator__alpha": [0.1, 1.0, 10.0, 100.0],
    },
    scoring=MeanAbsoluteScaledError(seasonality=12),
    cv=cv,
)
search.fit(y, forecasting_horizon=24)

print(f"Best alpha: {search.best_params_}")
print(f"Best CV MASE: {search.best_score_:.2f}")
```

The best forecaster is available as `search.best_forecaster_` and can be used directly for predictions.

## Check Residuals

Good residuals resemble white noise: no autocorrelation, no trend, and roughly constant variance. Patterned residuals indicate the model is missing structure in the data:

```python
from yohou.plotting import plot_residuals

plot_residuals(y_pred_ridge, y_test, title="Ridge Residuals")
```

If the residual plot shows spikes at seasonal lags, the model is not fully capturing the seasonal pattern. If residuals trend upward or downward, the model is missing a trend component. See [Residual Diagnostics](/pages/explanation/residual-diagnostics/) for interpretation guidance.

## Compare Models Visually

Plot both forecasts against the actual test values to see where each model succeeds and fails:

```python
from yohou.plotting import plot_forecast

plot_forecast(
    y_test,
    {"SeasonalNaive": y_pred_baseline, "Ridge": y_pred_ridge},
    y_train=y_train,
    title="Model Comparison: Tourism Forecast",
    y_label="Monthly tourists",
)
```

## What We Built

You have evaluated forecasters using the tools that go beyond a single train/test split:

- Scored models with multiple metrics (`MeanAbsoluteError`, `MeanAbsoluteScaledError`) to get different perspectives on forecast quality
- Used `ExpandingWindowSplitter` and `GridSearchCV` for temporal cross-validation
- Searched over hyperparameters with nested parameter syntax
- Diagnosed model weaknesses through residual plots

These evaluation tools work with any Yohou forecaster (point, interval, or class-probability) and any scikit-learn compatible scorer.

## Next Steps

- [Interval Forecasting](../explanation/interval-forecasting.md): Add prediction intervals with `SplitConformalForecaster`
- [Panel Data](../how-to/panel-data.md): Work with multiple related series
- [Choose a Forecasting Method](../how-to/choose-forecasting-method.md): Try nonlinear regressors
- [Ensemble Forecasting](../how-to/ensemble-forecasting.md): Combine multiple forecasters
- [Forecast Accuracy](../explanation/forecast-accuracy.md): Understand when to use MAE, MASE, or percentage metrics
