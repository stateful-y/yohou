# Forecasting Workflow

In this tutorial, we will evaluate two forecasters using temporal cross-validation, search for the best hyperparameters, and inspect residuals to diagnose model weaknesses.

## Prerequisites

- Completed [Getting Started](getting-started.md)

<!-- COMPANION_NOTEBOOKS -->

## 1. Setup

We use the monthly tourism dataset: 187 months of visitor arrivals to a single Australian region (T1). First, load the data and define a 12-month forecasting horizon:

```python
from yohou.datasets import fetch_tourism_monthly
from yohou.model_selection import train_test_split

bunch = fetch_tourism_monthly()
y = (
    bunch.frame
    .select("time", "T1__tourists")
    .drop_nulls()
    .rename({"T1__tourists": "tourists"})
)

forecasting_horizon = 12
y_train, y_test = train_test_split(y, test_size=forecasting_horizon)
```

Next, fit a [`SeasonalNaive`](/pages/api/generated/yohou.point.SeasonalNaive/) baseline:

```python
from yohou.point import SeasonalNaive

baseline = SeasonalNaive(seasonality=12)
baseline.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_baseline = baseline.predict(forecasting_horizon=forecasting_horizon)
```

Now build a Ridge pipeline with [`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.SeasonalDifferencing/) and lag features. If the pipeline looks unfamiliar, see [Getting Started](getting-started.md) for a step-by-step walkthrough:

```python
from sklearn.linear_model import Ridge
from yohou.compose import FeaturePipeline
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer
from yohou.stationarity import SeasonalDifferencing

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    target_transformer=SeasonalDifferencing(seasonality=12),
    actual_transformer=FeaturePipeline([
        ("lags", LagTransformer(lag=[1, 2, 3, 12])),
    ]),
)
forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_ridge = forecaster.predict(forecasting_horizon=forecasting_horizon)
```

## 2. Score with Multiple Metrics

Now score both models on the single train/test split. Scorers in Yohou are stateful: call `fit(y_train)` first so that scale-dependent metrics like [`MeanAbsoluteScaledError`](/pages/api/generated/yohou.metrics.MeanAbsoluteScaledError/) can normalise correctly:

```python
from yohou.metrics import MeanAbsoluteError, MeanAbsoluteScaledError

mae = MeanAbsoluteError()
mae.fit(y_train)
mase = MeanAbsoluteScaledError(seasonality=12)
mase.fit(y_train)

for name, y_pred in [("SeasonalNaive", y_pred_baseline), ("Ridge", y_pred_ridge)]:
    print(f"{name:15s}  MAE={mae.score(y_test, y_pred):.2f}  MASE={mase.score(y_test, y_pred):.2f}")
```

```text
SeasonalNaive    MAE=302.05  MASE=1.65
Ridge            MAE=214.35  MASE=1.17
```

Notice that both MASE values are above 1.0, meaning neither model outperforms the seasonal naive baseline on this single holdout. Cross-validation across multiple folds will tell us whether this pattern holds.

## 3. Evaluate with Cross-Validation and Hyperparameter Search

[`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.ExpandingWindowSplitter/) creates multiple temporal train/test folds by growing the training window. [`GridSearchCV`](/pages/api/generated/yohou.model_selection.GridSearchCV/) evaluates each parameter combination across all folds and selects the best. We pass the full `y` so the cross-validation splitter can build multiple train/test folds from the complete history; passing only `y_train` would shrink each fold unnecessarily:

```python
from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV

cv = ExpandingWindowSplitter(n_splits=3, test_size=forecasting_horizon)

search = GridSearchCV(
    forecaster=forecaster,
    param_grid={"estimator__alpha": [0.1, 1.0, 10.0, 100.0]},
    scoring=MeanAbsoluteScaledError(seasonality=12),
    cv=cv,
)
search.fit(y, forecasting_horizon=forecasting_horizon)

print(f"Best params:  {search.best_params_}")
print(f"CV MASE:      {-search.best_score_:.2f}")
```

```text
Best params:  {'estimator__alpha': 0.1}
CV MASE:      0.87
```

Notice that `best_score_` is negative. Yohou follows scikit-learn's convention of negating scores so that higher is always better. The code above negates it inline; if you access `search.best_score_` directly elsewhere, remember to negate it to recover the actual MASE.

The CV MASE of 0.87 is below 1.0, confirming that Ridge consistently outperforms the seasonal naive baseline across all three folds. The single holdout was harder than average.

## 4. Inspect Residuals

Let's refit the best forecaster from the search on the training data and inspect what the model gets wrong with [`plot_residuals`](/pages/api/generated/yohou.plotting.plot_residuals/):

```python
from yohou.plotting import plot_residuals

best = search.best_forecaster_
best.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_tuned = best.predict(forecasting_horizon=forecasting_horizon)

plot_residuals(y_pred_tuned, y_test, title="Residuals: Ridge (Tuned)")
```

You should see a scatter of residuals over the test period. If the residuals cluster near zero with no obvious pattern, the model is capturing the main signal. Spikes at seasonal lags or a visible trend suggest missing structure. See [Residual Diagnostics](../explanation/residual-diagnostics.md) for a full interpretation guide.

## 5. Compare Models Visually

Now plot both forecasts against the actual test values:

```python
from yohou.plotting import plot_forecast, plot_score_summary

plot_forecast(
    y_test,
    {"SeasonalNaive": y_pred_baseline, "Ridge (Tuned)": y_pred_tuned},
    y_train=y_train,
    n_history=36,
    title="Model Comparison: Tourism Forecast",
    y_label="Monthly visitors",
)

plot_score_summary(
    {"MAE": mae, "MASE": mase},
    y_test,
    {"SeasonalNaive": y_pred_baseline, "Ridge (Tuned)": y_pred_tuned},
    title="Score Comparison",
)
```

Notice how `plot_forecast` overlays predicted and actual values so you can spot where each model over- or under-shoots. `plot_score_summary` condenses the comparison into a single bar chart.

## What You Built

We completed the full evaluation workflow:

- Scored models with [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.MeanAbsoluteError/) and [`MeanAbsoluteScaledError`](/pages/api/generated/yohou.metrics.MeanAbsoluteScaledError/) on a single train/test split
- Used [`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.ExpandingWindowSplitter/) and [`GridSearchCV`](/pages/api/generated/yohou.model_selection.GridSearchCV/) to evaluate across temporal folds and tune hyperparameters
- Refitted the best forecaster and inspected residuals with [`plot_residuals`](/pages/api/generated/yohou.plotting.plot_residuals/)
- Compared forecasts visually with [`plot_forecast`](/pages/api/generated/yohou.plotting.plot_forecast/) and [`plot_score_summary`](/pages/api/generated/yohou.plotting.plot_score_summary/)

## Next Steps

- [Exogenous Features](exogenous-features.md): Add external regressors to your forecasting pipeline
- [Model Selection](../explanation/model-selection.md): Expanding vs. sliding windows, fold design, and when CV estimates are trustworthy
- [Forecast Accuracy](../explanation/forecast-accuracy.md): When to use MAE, MASE, or percentage metrics
- [Choose a Forecasting Method](../how-to/choose-forecasting-method.md): Try nonlinear regressors and compare estimator families
- [Interval Forecasting](../explanation/interval-forecasting.md): Add prediction intervals with [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.SplitConformalForecaster/)
