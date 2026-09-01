# How to Enable Early Stopping

This guide shows you how to hold out a validation tail with
`validation_size` so gradient boosting estimators (LightGBM, XGBoost,
CatBoost) stop training when their validation performance plateaus.

## Prerequisites

- yohou installed ([Installation](installation.md)) plus a boosting library
  (`pip install lightgbm`)
- Familiarity with reduction forecasters
  ([Build Reduction Forecasters](build-reduction-forecasters.md))

## 1. Configure Stopping on the Estimator

Early stopping belongs to the estimator, not to yohou. Configure the rounds
and any metric on the estimator you construct; yohou's only job is delivering
a correctly built evaluation set to its `fit`:

```python
from lightgbm import LGBMRegressor

estimator = LGBMRegressor(
    n_estimators=500,
    early_stopping_round=20,
    verbose=-1,
)
```

The same applies to XGBoost (`early_stopping_rounds` on the constructor) and
CatBoost (`early_stopping_rounds` on the constructor or fit).

## 2. Hold Out a Validation Tail with `validation_size`

Set `validation_size` on the reduction forecaster to hold out the last N time
steps (per group on panel data) from estimator training. Yohou fits
transformers, encoders, and sample weights on the remaining head only, builds
evaluation rows through the fitted transformer stack (lag warmup, step
columns from `X_future`/`X_forecast`, as-of vintage resolution included), and
passes them to the estimator's `fit` as `eval_set`:

```python
from yohou.datasets import fetch_electricity_demand
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer

data = fetch_electricity_demand()
y = data.frame

forecaster = PointReductionForecaster(
    estimator=estimator,
    reduction_strategy="direct",
    actual_transformer=LagTransformer(lag=[1, 2, 24]),
    validation_size=96,
)
forecaster.fit(y=y, forecasting_horizon=24)
```

After fitting, the forecaster has observed the tail, so `predict()` forecasts
the period after the end of all provided data, exactly as without a holdout.

By default only rows whose entire target window lies inside the tail are
evaluated (`validation_size - forecasting_horizon + 1` rows), which requires
`validation_size >= forecasting_horizon`. An estimator whose `fit` accepts no
`eval_set` (most plain sklearn estimators), a `sklearn.multioutput` wrapper,
or a holdout that leaves too little training data all raise a `ValueError` at
fit; the [`PointReductionForecaster`](/pages/api/generated/yohou.point.PointReductionForecaster/)
parameter documentation lists every rejected configuration.

## 3. Read the Result from `estimator_`

Yohou does not wrap the library's reporting; read it directly from the fitted
estimator. With `reduction_strategy="direct"` or `"dir-rec"`, `estimator_` is
a list with one estimator per horizon step:

```python
for step, est in enumerate(forecaster.estimator_, start=1):
    print(f"step {step}: best iteration {est.best_iteration_}")
    # est.evals_result_ holds the per-iteration validation curve
```

With `"multi-output"`, `estimator_` is the single fitted estimator.

## 4. Short Series: `validation_overlap`

Strict evaluation discards the `forecasting_horizon - 1` boundary rows whose
target windows straddle the split. On short series with long horizons that
can consume most of the holdout. Setting `validation_overlap=True` evaluates
those rows too (yielding `validation_size` rows), at a documented cost: their
early targets are time points the model also trained on, so the stopping
signal is partly in-sample:

```python
forecaster = PointReductionForecaster(
    estimator=estimator,
    validation_size=30,
    validation_overlap=True,
)
```

## 5. Know the Trade-off

Boosting libraries do not refit after early stopping: the model you get was
trained without the tail, and the tail's information is spent on choosing the
iteration count. If you want a final model trained on everything, read the
discovered iteration count and refit without the holdout:

```python
best = max(est.best_iteration_ for est in forecaster.estimator_)
final = PointReductionForecaster(
    estimator=LGBMRegressor(n_estimators=best, verbose=-1),
    reduction_strategy="direct",
    actual_transformer=LagTransformer(lag=[1, 2, 24]),
    validation_size=None,
)
final.fit(y=y, forecasting_horizon=24)
```

Inside [`GridSearchCV`](/pages/api/generated/yohou.model_selection.GridSearchCV/)
or [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.RandomizedSearchCV/),
`validation_size` composes with no extra configuration: each fold's inner fit
holds out the tail of its own training window (and trains on correspondingly
less data).

## See Also

- [Tune Forecaster Hyperparameters](tune-hyperparameters.md): search over
  forecaster and estimator parameters with temporal splitters
- [Forecast with CatBoost](forecast-with-catboost.md): gradient-boosted trees
  for point and interval forecasting
- [Build Reduction Forecasters](build-reduction-forecasters.md): lag
  features, transformers, and reduction strategies
- [`PointReductionForecaster` API reference](/pages/api/generated/yohou.point.PointReductionForecaster/)
