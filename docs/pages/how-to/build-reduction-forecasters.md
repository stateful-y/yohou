# How to Build Reduction Forecasters

This guide shows you how to build forecasters using the reduction pattern: pick a scikit-learn estimator, configure feature engineering, and choose a multi-step prediction strategy.

## Prerequisites

- Familiarity with the fit/predict API ([Getting Started](../tutorials/getting-started.md))
- Understanding of reduction strategies ([Reduction Forecasting](../explanation/reduction-forecasting.md))

<!-- COMPANION_NOTEBOOKS -->

## Build a Basic Reduction Forecaster

A [`PointReductionForecaster`](/pages/api/generated/yohou.point.PointReductionForecaster/) converts a time series problem into a tabular regression problem. The `feature_transformer` generates the feature matrix, and the `estimator` learns the mapping from features to targets:

```python
from yohou.datasets import fetch_tourism_monthly
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer
from sklearn.linear_model import Ridge

bunch = fetch_tourism_monthly(n_series=1)
y = bunch.frame

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    feature_transformer=LagTransformer(lag=[1, 3, 6, 12]),
    reduction_strategy="multi-output",
)
forecaster.fit(y, forecasting_horizon=12)
predictions = forecaster.predict()
```

Any scikit-learn regressor works as the `estimator`. For tree-based models, see [Forecast with CatBoost](forecast-with-catboost.md).

## Choose a Reduction Strategy

The `reduction_strategy` parameter controls how the forecaster maps features to multiple forecast horizons:

- If you want a single model that predicts all horizons at once, use `"multi-output"`. This is the fastest option and works well when horizons share similar patterns.
- If each horizon benefits from its own model, use `"direct"`. This avoids error propagation between horizons at the cost of training one model per step. Use `n_jobs` to parallelize fitting.
- If you want the flexibility of direct models with recursive features feeding into each, use `"dir-rec"`.

```python
forecaster = PointReductionForecaster(
    estimator=Ridge(),
    feature_transformer=LagTransformer(lag=[1, 3, 6, 12]),
    reduction_strategy="direct",
    n_jobs=-1,  # parallelize across horizons
)
```

## Combine Multiple Feature Transformers

Use [`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/) to concatenate [`LagTransformer`](/pages/api/generated/yohou.preprocessing.LagTransformer/), [`RollingStatisticsTransformer`](/pages/api/generated/yohou.preprocessing.RollingStatisticsTransformer/), [`CalendarFeatureTransformer`](/pages/api/generated/yohou.preprocessing.CalendarFeatureTransformer/), and [`FourierFeatureTransformer`](/pages/api/generated/yohou.preprocessing.FourierFeatureTransformer/) into a single feature matrix:

```python
from yohou.compose import FeatureUnion
from yohou.point import PointReductionForecaster
from yohou.preprocessing import (
    CalendarFeatureTransformer,
    FourierFeatureTransformer,
    LagTransformer,
    RollingStatisticsTransformer,
)
from sklearn.ensemble import HistGradientBoostingRegressor

forecaster = PointReductionForecaster(
    estimator=HistGradientBoostingRegressor(),
    feature_transformer=FeatureUnion([
        ("lags", LagTransformer(lag=[1, 3, 6, 12])),
        ("rolling", RollingStatisticsTransformer(window_size=12, statistics=["mean", "std"])),
        ("calendar", CalendarFeatureTransformer(features=["month", "day_of_week"])),
        ("fourier", FourierFeatureTransformer(seasonality=12, harmonics=[1, 2])),
    ]),
    reduction_strategy="direct",
)
```

For sequential preprocessing before feature engineering, wrap transformers in a [`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/). See [Compose Feature Pipelines](compose-feature-pipelines.md) for details.

## Control Step Feature Alignment

When using the `"direct"` strategy with exogenous features (`X_future` or `X_forecast`), the `step_feature_alignment` parameter controls which step-indexed columns each horizon's estimator sees:

- `"all"` (default): every estimator sees all step columns.
- `"matched"`: each estimator sees only the step column for its horizon. Use this when features degrade with horizon (e.g., weather forecasts).
- `"cumulative"`: the estimator for step $h$ sees step columns 1 through $h$.

```python
forecaster = PointReductionForecaster(
    estimator=HistGradientBoostingRegressor(),
    feature_transformer=LagTransformer(lag=[1, 2, 3]),
    reduction_strategy="direct",
    step_feature_alignment="matched",
)
```

## Produce Prediction Intervals

[`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.IntervalReductionForecaster/) uses the same reduction pattern but produces prediction intervals. It accepts the same `feature_transformer`, `reduction_strategy`, and `step_feature_alignment` parameters:

```python
from yohou.interval import IntervalReductionForecaster

interval_forecaster = IntervalReductionForecaster(
    feature_transformer=LagTransformer(lag=[1, 3, 6, 12]),
    reduction_strategy="multi-output",
)
interval_forecaster.fit(y, forecasting_horizon=12, coverage_rates=[0.1, 0.5, 0.9])
y_pred_int = interval_forecaster.predict_interval()
```

For conformal calibration on top of a point reduction forecaster, see [Produce Prediction Intervals](interval-forecasting.md).

## See Also

- [Reduction Forecasting](../explanation/reduction-forecasting.md) for conceptual background on reduction strategies
- [Use Exogenous Features](exogenous-features.md) for passing `X_actual`, `X_future`, and `X_forecast` to reduction forecasters, including chaining with [`ForecastedFeatureForecaster`](/pages/api/generated/yohou.compose.ForecastedFeatureForecaster/)
- [Compose Feature Pipelines](compose-feature-pipelines.md) for building multi-step feature engineering chains
