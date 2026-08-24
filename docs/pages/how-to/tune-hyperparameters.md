# How to Tune Forecaster Hyperparameters

This guide shows you how to find optimal hyperparameters for any yohou
forecaster using cross-validated search with temporal splitters.

## Prerequisites

- yohou installed ([Installation](installation.md))
- Familiarity with fitting and predicting ([Getting Started](../tutorials/getting-started.md))

<!-- COMPANION_NOTEBOOKS -->

## 1. Define a Forecaster and Parameter Grid

Use double underscore (`__`) to refer to nested parameters inside a
forecaster (following scikit-learn convention):

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster
from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import GridSearchCV, ExpandingWindowSplitter, train_test_split
from yohou.datasets import fetch_electricity_demand

data = fetch_electricity_demand()
y = data.frame

y_train, y_test = train_test_split(y, test_size=48)

forecaster = PointReductionForecaster(estimator=Ridge())

param_grid = {"estimator__alpha": [0.01, 0.1, 1.0, 10.0]}
```

## 2. Choose a Splitter

Use [`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.ExpandingWindowSplitter/) to simulate accumulating historical data.
Use [`SlidingWindowSplitter`](/pages/api/generated/yohou.model_selection.SlidingWindowSplitter/) if you want a fixed training window instead:

```python
from yohou.model_selection import ExpandingWindowSplitter

splitter = ExpandingWindowSplitter(n_splits=5, test_size=24)
```

Set `test_size` to match your forecasting horizon. `n_splits` controls
how many train/test windows are evaluated per parameter combination.

## 3. Run Grid Search

Pass the forecaster, parameter grid, scorer, and splitter to
[`GridSearchCV`](/pages/api/generated/yohou.model_selection.GridSearchCV/):

```python
search = GridSearchCV(
    forecaster=forecaster,
    param_grid=param_grid,
    scoring=MeanAbsoluteError(),
    cv=splitter,
    n_jobs=-1,
    refit=True,
)
search.fit(y_train, forecasting_horizon=24)
```

Setting `refit=True` refits the best forecaster on the full training set so
`search.predict()` is ready immediately after `fit`.

## 4. Predict with the Best Model

When `refit=True`, the search object acts as a fitted forecaster:

```python
y_pred = search.predict()
```

Access the winning configuration through `best_params_` and `best_score_`:

```python
print("Best params:", search.best_params_)
print("Best score:", search.best_score_)
```

## 5. Inspect and Visualize Results

`cv_results_` contains per-fold scores for every parameter combination:

```python
import polars as pl

results = pl.DataFrame(search.cv_results_)
print(results.select(["params", "mean_test_score", "rank_test_score"]))
```

Use [`plot_cv_results_scatter`](/pages/api/generated/yohou.plotting.plot_cv_results_scatter/)
to visualize how score changes across parameter values:

```python
from yohou.plotting import plot_cv_results_scatter

fig = plot_cv_results_scatter(search.cv_results_, param_name="estimator__alpha")
fig.show()
```

Look for parameter values where the score flattens or reaches a minimum to
identify the best operating region.

## 6. Use RandomizedSearchCV for Large Spaces

When the grid has many dimensions or continuous ranges, switch to
[`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.RandomizedSearchCV/),
which samples `n_iter` random combinations instead of evaluating every one:

```python
from scipy.stats import loguniform
from yohou.model_selection import RandomizedSearchCV

param_distributions = {
    "estimator__alpha": loguniform(1e-3, 1e3),
}

search = RandomizedSearchCV(
    forecaster=forecaster,
    param_distributions=param_distributions,
    scoring=MeanAbsoluteError(),
    cv=splitter,
    n_iter=20,
    n_jobs=-1,
    refit=True,
    random_state=42,
)
search.fit(y_train, forecasting_horizon=24)
```

`n_iter` controls the number of parameter settings sampled. Use
`GridSearchCV` when total combinations are small (< 50) and
`RandomizedSearchCV` when the space is large or continuous.

## 7. Evaluate with Multiple Metrics

Pass a dict of scorers to `scoring` and set `refit` to the scorer name used
for selecting the best model. For example, combine [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.MeanAbsoluteError/) and [`RootMeanSquaredError`](/pages/api/generated/yohou.metrics.RootMeanSquaredError/):

```python
from yohou.metrics import RootMeanSquaredError

search = GridSearchCV(
    forecaster=forecaster,
    param_grid=param_grid,
    scoring={"mae": MeanAbsoluteError(), "rmse": RootMeanSquaredError()},
    cv=splitter,
    refit="mae",
)
search.fit(y_train, forecasting_horizon=24)

results = pl.DataFrame(search.cv_results_)
print(results.select(["params", "mean_test_mae", "mean_test_rmse"]))
```

All scorers are evaluated on every fold, but only the one named in `refit`
determines which parameters are selected as best.

## 8. Route Extra Metadata into the Inner Walk-Forward

The search's inner loop scores each candidate with a rolling walk-forward.
Extra keyword arguments to `fit` reach it through sklearn metadata routing:
the value lands in the bucket named by your scorers' response method
(`predict` for point scorers, `predict_interval` for interval scorers,
`predict_class_proba` for class-probability scorers).

One discipline applies: a routed key is validated against every predict-family
method that carries it, so set the request `True` on the response method and
explicitly `False` on any other carrier. For example, to score candidates at a
daily walk-forward stride with an interval scorer:

```python
forecaster.set_predict_interval_request(stride=True)   # the response method
forecaster.set_predict_request(stride=False)           # sibling carrier

search = GridSearchCV(
    forecaster=forecaster,
    param_grid=param_grid,
    scoring=IntervalScore(coverage_rates=[0.5, 0.9]),
    cv=splitter,
)
search.fit(y_train, forecasting_horizon=48, stride=24)
```

Forgetting the `False` pairing raises a routing error that names the unset
method and the `set_*_request` call that fixes it. The same pattern carries
`strategy`, `predict_transformed`, or `groups` where the family's walk-forward
accepts them. `cross_validate` and `cross_val_score` do not use routing: they
take `predict_stride` and `predict_forecasting_horizon` as explicit
parameters instead.

## See Also

- [Choose a Forecasting Method](choose-forecasting-method.md): select a forecaster before tuning
- [About Model Selection](../explanation/model-selection.md): temporal cross-validation, splitter design, and search strategy trade-offs
- [Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md): understand the metrics used for scoring
- [Extensions](../reference/extensions.md): `yohou-optuna` provides `OptunaSearchCV` for Bayesian hyperparameter search
- [`yohou.model_selection` API reference](/pages/api/model_selection/)
