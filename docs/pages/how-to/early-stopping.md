# How to Enable Early Stopping

This guide shows you how to give gradient boosting estimators (LightGBM,
XGBoost, CatBoost) an evaluation set so they stop training when their
validation performance plateaus: a held-out tail with `validation_size`, a
window you supply with `validation_y`, or each fold's test window inside a
hyperparameter search.

## Prerequisites

- yohou installed ([Installation](installation.md)) plus a boosting library
  (`pip install lightgbm`)
- Familiarity with reduction forecasters
  ([Build Reduction Forecasters](build-reduction-forecasters.md))

<!-- COMPANION_NOTEBOOKS -->

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
target windows straddle the split, which on short series with long horizons
can consume most of the holdout. Set `validation_overlap=True` to evaluate
those rows too, yielding `validation_size` rows. See
[Validation Holdout](../explanation/reduction-forecasting.md#validation-holdout)
for what that trades away:

```python
short_series_forecaster = PointReductionForecaster(
    estimator=estimator,
    reduction_strategy="direct",
    actual_transformer=LagTransformer(lag=[1, 2, 24]),
    validation_size=30,
    validation_overlap=True,
)
short_series_forecaster.fit(y=y, forecasting_horizon=24)
```

## 5. Wrap the Estimator in a Pipeline

An estimator that needs preprocessing of its own can be a
`sklearn.pipeline.Pipeline`, as long as its final step accepts `eval_set`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline_forecaster = PointReductionForecaster(
    estimator=Pipeline([
        ("scaler", StandardScaler()),
        ("lgbm", LGBMRegressor(n_estimators=500, early_stopping_round=20, verbose=-1)),
    ]),
    reduction_strategy="direct",
    actual_transformer=LagTransformer(lag=[1, 2, 24]),
    validation_size=96,
)
pipeline_forecaster.fit(y=y, forecasting_horizon=24)
```

Yohou fits the pipeline's transformer steps on the training rows only, pushes
the held-out rows through those fitted transformers, and hands the final step
an evaluation set in the same feature space it trains in, so the stopping
metric is comparable to the training loss. `estimator_` stays a fitted
`Pipeline`, so read the booster through its step name
(`pipeline_forecaster.estimator_[0].named_steps["lgbm"].best_iteration_` for
the `"direct"` strategy). A pipeline whose final step cannot accept an
`eval_set`, or that ends in `"passthrough"`, is rejected at fit.

## 6. Early-Stop Interval Forecasters

[`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.IntervalReductionForecaster/)
takes the same `validation_size` parameter. The holdout splits once, and every
quantile estimator it trains (a lower and an upper model per coverage rate)
receives the same evaluation set, each stopping on its own quantile loss:

```python
from lightgbm import LGBMRegressor
from yohou.interval import IntervalReductionForecaster

interval_forecaster = IntervalReductionForecaster(
    estimator=LGBMRegressor(
        objective="quantile",
        alpha=0.5,  # overwritten per bound by the forecaster
        n_estimators=500,
        early_stopping_round=20,
        verbose=-1,
    ),
    reduction_strategy="direct",
    actual_transformer=LagTransformer(lag=[1, 2, 24]),
    validation_size=96,
)
interval_forecaster.fit(y=y, forecasting_horizon=24, coverage_rates=[0.9])
```

Here `interval_forecaster.estimator_` is a dict keyed by bound (for example
`"coverage_rate_0.9_lower"`); read each bound's stopping result from it:

```python
for bound, ests in interval_forecaster.estimator_.items():
    best = ests.best_iteration_ if not isinstance(ests, list) else max(e.best_iteration_ for e in ests)
    print(f"{bound}: best iteration {best}")
```

The default interval estimator (`MultiOutputRegressor(QuantileRegressor())`)
cannot receive an evaluation set and is rejected with `validation_size` set;
pick an eval-capable quantile estimator as above.

## 7. Refit on the Full Series

The fitted model was trained on the head alone. To get one trained on
everything, read the discovered iteration count and refit without the
holdout. See
[Validation Holdout](../explanation/reduction-forecasting.md#validation-holdout)
for why the library does not do this for you:

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

Inside a hyperparameter search the iteration count can be chosen for you; see
[Early Stop Inside a Search](#9-early-stop-inside-a-search).

## 8. Pass Your Own Evaluation Window

When you already hold the evaluation data separately, for example the next
period of a manual backtest, pass it to `fit` as `validation_y` instead of
setting `validation_size`. The window must start one interval after `y` ends
and have the same columns:

```python
train, window = y[:-96], y[-96:]

window_forecaster = PointReductionForecaster(
    estimator=LGBMRegressor(n_estimators=500, early_stopping_round=20, verbose=-1),
    reduction_strategy="direct",
    actual_transformer=LagTransformer(lag=[1, 2, 24]),
)
window_forecaster.fit(y=train, forecasting_horizon=24, validation_y=window)
```

Transformers are fitted on `train` only and the window's evaluation rows are
built through them, exactly as for the `validation_size` tail. The difference
is the state after fitting: the window is not training data, so `predict()`
forecasts the period right after `train`, and you can score the model on the
window with `observe_predict(window)`. If you fitted with `X_actual`, pass the
window's rows as `validation_X_actual`; forecast vintages published during the
window go in `validation_X_forecast`. `validation_y` and `validation_size`
cannot be combined.

## 9. Early Stop Inside a Search

[`GridSearchCV`](/pages/api/generated/yohou.model_selection.GridSearchCV/) and
[`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.RandomizedSearchCV/)
offer two ways to early-stop the candidates they evaluate.

**Hold out a tail inside each fold.** Set `validation_size` on the forecaster
and leave the search as it is. Each fold's fit holds out the end of its own
training window, trains on correspondingly less data, and is scored on the
fold's test window, which the estimator never saw.

**Stop on each fold's test window.** Set `validation="cv"` on the search and
leave `validation_size=None`. Every fold trains on its whole training window
and is evaluated on its own test window after each iteration, up to
`n_estimators`; the search then picks one iteration count per fitted estimator
from the metric averaged over the folds, scores every fold at that count, and
refits on all data with it:

```python
from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV

search = GridSearchCV(
    forecaster=PointReductionForecaster(
        estimator=LGBMRegressor(n_estimators=1000, verbose=-1),
        reduction_strategy="direct",
        actual_transformer=LagTransformer(lag=[1, 2, 24]),
    ),
    param_grid={"estimator__num_leaves": [15, 31], "estimator__learning_rate": [0.05, 0.1]},
    scoring=MeanAbsoluteError(),
    cv=ExpandingWindowSplitter(n_splits=3, test_size=96),
    validation="cv",
)
search.fit(y=y, forecasting_horizon=24)
```

Read the chosen iteration counts from `best_rounds_`, keyed by fitted estimator
(`"step_1"` to `"step_24"` here), and every candidate's from
`search.cv_results_["rounds"]`:

```python
print(search.best_rounds_)
print(search.cv_results_["rounds_at_boundary"])
```

The refitted `search.best_forecaster_` trained each step for its own count,
with early stopping off.

In this mode `n_estimators` is both the largest count the search can choose and
the cost: fold fits never stop early, so each trains every iteration, and the
estimator's own early-stopping patience plays no part. Set it high enough to
pass the best count. A warning, and `rounds_at_boundary`, report a chosen count
equal to `n_estimators`, where a later count might have been better.

The second route trains on more data per fold but chooses the iteration count
on the rows it scores, so `best_score_` is optimistic; the first route keeps the
score unbiased. [Early Stopping on the Scored Fold](../explanation/reduction-forecasting.md#early-stopping-on-the-scored-fold)
explains the difference.

`validation="cv"` works with LightGBM, XGBoost, and CatBoost estimators, bare
or as a `Pipeline`'s final step. It has three requirements:

- CatBoost estimators need an explicit `learning_rate`, because CatBoost
  derives its default learning rate from `iterations` and the refit trains a
  different number of iterations than the folds did.
- `reduction_strategy="dir-rec"` is rejected: its later steps train on the
  earlier steps' predictions, so the steps cannot be cut to separate counts.
- For any other estimator, subclass
  [`BaseEarlyStoppingAdapter`](/pages/api/generated/yohou.model_selection.BaseEarlyStoppingAdapter/)
  and pass an instance as `early_stopping_adapter`.

## See Also

- [Tune Forecaster Hyperparameters](tune-hyperparameters.md): search over
  forecaster and estimator parameters with temporal splitters
- [Forecast with CatBoost](forecast-with-catboost.md): gradient-boosted trees
  for point and interval forecasting
- [Build Reduction Forecasters](build-reduction-forecasters.md): lag
  features, transformers, and reduction strategies
- [`PointReductionForecaster` API reference](/pages/api/generated/yohou.point.PointReductionForecaster/)
- [`IntervalReductionForecaster` API reference](/pages/api/generated/yohou.interval.IntervalReductionForecaster/)
