# Forecasting Workflow

In this tutorial, we will evaluate two forecasters using temporal cross-validation, search for the best hyperparameters, and inspect residuals to diagnose model weaknesses.

## Prerequisites

- Completed [Your First Forecast](first-forecast.md)

## Setup

We use the monthly tourism dataset from the [Monash Time Series Forecasting Archive](https://forecastingdata.org/): 366 monthly regional tourism series. Here we work with a single region (T1), which covers 187 months of visitor arrivals from January 1979 to July 1994. The column name `T1__tourists` follows Yohou's panel format, where the identifier (`T1`) and the variable name (`tourists`) are joined by a double underscore. We rename it to `tourists` for readability.

```python
from sklearn.linear_model import Ridge
from yohou.compose import FeaturePipeline
from yohou.datasets import fetch_tourism_monthly
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.preprocessing import LagTransformer
from yohou.stationarity import SeasonalDifferencing

bunch = fetch_tourism_monthly()
y = (
    bunch.frame
    .select("time", "T1__tourists")
    .drop_nulls()
    .rename({"T1__tourists": "tourists"})
)

forecasting_horizon = 12
y_train = y[:-forecasting_horizon]
y_test = y[-forecasting_horizon:]

baseline = SeasonalNaive(seasonality=12)
baseline.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_baseline = baseline.predict(forecasting_horizon=forecasting_horizon)

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    target_transformer=SeasonalDifferencing(seasonality=12),
    feature_transformer=FeaturePipeline([
        ("lags", LagTransformer(lag=[1, 2, 3, 12])),
    ]),
)
forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_ridge = forecaster.predict(forecasting_horizon=forecasting_horizon)
```

For how this pipeline was built step by step, see [Your First Forecast](first-forecast.md).

## Score with Multiple Metrics

Start by scoring both models on the single train/test split:

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

A MASE below 1.0 means the model outperforms the seasonal naive baseline. Notice that both values exceed 1.0 here: neither model beats the baseline on this particular holdout period. Ridge is the better of the two, but a single split is a single data point. Cross-validation across multiple folds will tell us whether this pattern holds.

## Evaluate with Cross-Validation and Hyperparameter Search

[`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.ExpandingWindowSplitter/) creates multiple temporal train/test folds by growing the training window while holding out a fixed test period. [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) evaluates each parameter combination across all folds and selects the best. Define a search grid using the nested parameter syntax (`estimator__param` to reach regressor parameters, `feature_transformer__step_name__param` for pipeline steps):

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

Notice that `best_score_` is negative, following scikit-learn's convention of negating scorer values so that higher is always better. Negate it to recover the actual MASE: `-search.best_score_`. The CV MASE of 0.87 is below 1.0, confirming that Ridge consistently outperforms the seasonal naive baseline across all three folds. The single holdout was harder than average.

Notice also that all four alpha values achieve the same CV score here. On a short series with few features, Ridge regularization has little effect on the solution. In practice, with more features or larger datasets, the search will show meaningful variation across parameter values.

## Inspect Residuals

Good residuals resemble white noise: no autocorrelation, no trend, and roughly constant variance. Patterned residuals indicate the model is missing structure in the data. We use the best forecaster from the search, refitted on the training data:

```python
from yohou.plotting import plot_residuals

best = search.best_forecaster_
best.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_tuned = best.predict(forecasting_horizon=forecasting_horizon)

plot_residuals(y_pred_tuned, y_test, title="Residuals: Ridge (Tuned)")
```

If the plot shows spikes at seasonal lags, the model has not fully captured the seasonal pattern. If residuals trend upward or downward, a trend component is missing. See [Residual Diagnostics](/pages/explanation/residual-diagnostics/) for interpretation guidance.

## Compare Models Visually

Plot both forecasts against the actual test values to see where each model succeeds and fails:

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
    mase,
    y_test,
    {"SeasonalNaive": y_pred_baseline, "Ridge (Tuned)": y_pred_tuned},
    title="MASE by Model",
)
```

## What You Built

You have run the full evaluation workflow:

- Scored models with `MeanAbsoluteError` and `MeanAbsoluteScaledError` on a single train/test split
- Used `ExpandingWindowSplitter` and `GridSearchCV` to evaluate across temporal folds and tune hyperparameters at the same time
- Refitted the best forecaster and inspected residuals with `plot_residuals` to diagnose model weaknesses
- Compared forecasts visually with `plot_forecast` and `plot_score_summary`

## Next Steps

- [Exogenous Features](exogenous-features.md): Add external regressors to your forecasting pipeline
- [Model Selection](../explanation/model-selection.md): Understand expanding vs. sliding windows, fold design, and when CV estimates are trustworthy
- [Forecast Accuracy](../explanation/forecast-accuracy.md): Understand when to use MAE, MASE, or percentage metrics
- [Choose a Forecasting Method](../how-to/choose-forecasting-method.md): Try nonlinear regressors and compare estimator families
- [Interval Forecasting](../explanation/interval-forecasting.md): Add prediction intervals with `SplitConformalForecaster`
