# How to Forecast with CatBoost

This guide shows you how to use CatBoost as the estimator inside Yohou's
reduction forecasters for point, interval, and categorical predictions.

## Prerequisites

- Familiarity with the fit/predict API ([Getting Started](../tutorials/getting-started.md))
- CatBoost installed (`pip install catboost`)
- Understanding of reduction forecasting ([Reduction Forecasting](../explanation/reduction-forecasting.md))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Fit a Point Forecaster

Pass a [`CatBoostRegressor`](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor) to [`PointReductionForecaster`](/pages/api/generated/yohou.point.PointReductionForecaster/). Any `CatBoostRegressor` parameter (learning rate, depth, regularization) can be set directly:

```python
from catboost import CatBoostRegressor
from yohou.datasets import fetch_electricity_demand
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer

data = fetch_electricity_demand()
y = data.frame

forecaster = PointReductionForecaster(
    estimator=CatBoostRegressor(iterations=500, verbose=0),
    feature_transformer=LagTransformer(lag=[1, 3, 6, 12]),
)
forecaster.fit(y, forecasting_horizon=12)
predictions = forecaster.predict()
```

## Choose a Reduction Strategy

The `reduction_strategy` parameter controls how horizons are modelled. If each horizon benefits from its own model, use `"direct"`. Set `n_jobs=-1` to train the per-step models in parallel:

```python
forecaster = PointReductionForecaster(
    estimator=CatBoostRegressor(iterations=500, verbose=0),
    feature_transformer=LagTransformer(lag=[1, 3, 6, 12]),
    reduction_strategy="direct",
    n_jobs=-1,
)
```

If you want a single model that predicts all horizons at once, keep the default `"multi-output"`. If you need direct models with recursive feature propagation between steps, use `"dir-rec"`. See [Build Reduction Forecasters](build-reduction-forecasters.md) for a full comparison.

## Produce Interval Forecasts

Pass a `CatBoostRegressor` whose `loss_function` already starts with `MultiQuantile` to [`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.IntervalReductionForecaster/) and specify `coverage_rates` at fit time. The framework reads the `MultiQuantile` loss as a signal to fit a single model for all quantiles, overriding the placeholder alpha with the alpha values derived from `coverage_rates`. This path requires `forecasting_horizon=1`:

```python
from yohou.interval import IntervalReductionForecaster

forecaster = IntervalReductionForecaster(
    estimator=CatBoostRegressor(
        iterations=500,
        loss_function="MultiQuantile:alpha=0.5",
        verbose=0,
    ),
    feature_transformer=LagTransformer(lag=[1, 3, 6, 12]),
)
forecaster.fit(y, forecasting_horizon=1, coverage_rates=[0.90])
intervals = forecaster.predict_interval()
```

To request multiple coverage rates at once, pass them as a list:

```python
forecaster.fit(y, forecasting_horizon=1, coverage_rates=[0.80, 0.90, 0.95])
```

For multi-step horizons, use a single-quantile estimator (such as
`QuantileRegressor`) so the forecaster fits one model per bound via the
`direct` strategy.

## Forecast Categorical Data

Pass a [`CatBoostClassifier`](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier) to [`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.ClassProbaReductionForecaster/) for categorical time series (for example, demand tiers or quality levels):

```python
from catboost import CatBoostClassifier
from yohou.class_proba import ClassProbaReductionForecaster
from yohou.datasets import fetch_air_quality_classification
from yohou.preprocessing import LagTransformer

data = fetch_air_quality_classification()
y_categorical = data.y

forecaster = ClassProbaReductionForecaster(
    estimator=CatBoostClassifier(iterations=500, verbose=0),
    feature_transformer=LagTransformer(lag=[1, 3, 6, 12]),
)
forecaster.fit(y_categorical, forecasting_horizon=12)
y_proba = forecaster.predict_class_proba()
y_labels = forecaster.predict()
```

See [Forecast with Class Probabilities](class-probability-forecasting.md) for evaluation and scoring details.

## Tune CatBoost with Randomized Search

Use Yohou's [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.RandomizedSearchCV/) to search over CatBoost parameters:

```python
from yohou.model_selection import RandomizedSearchCV, ExpandingWindowSplitter

param_distributions = {
    "estimator__iterations": [200, 500, 1000],
    "estimator__depth": [4, 6, 8],
    "estimator__l2_leaf_reg": [1, 3, 10],
}

search = RandomizedSearchCV(
    forecaster=forecaster,
    param_distributions=param_distributions,
    cv=ExpandingWindowSplitter(n_splits=3),
    n_iter=10,
)
search.fit(y, forecasting_horizon=12)
best_forecaster = search.best_forecaster_
```

For exhaustive search, use [`GridSearchCV`](/pages/api/generated/yohou.model_selection.GridSearchCV/) instead. See [Tune Hyperparameters](tune-hyperparameters.md) for details.

## See Also

- [Reduction Forecasting](../explanation/reduction-forecasting.md) for the conceptual background on reduction strategies
- [Interval Forecasting](../explanation/interval-forecasting.md) for interval prediction concepts
- [Build Reduction Forecasters](build-reduction-forecasters.md) for combining estimators, transformers, and strategies
