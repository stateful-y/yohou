# Point Forecasting

### Naive Forecasters ([View](/examples/point/naive_forecasters/) | [Editable](/examples/point/naive_forecasters/edit/))

**Seasonal Naive Baselines**

Build baseline forecasters with `SeasonalNaive`. This example demonstrates the rolling evaluation workflow: fit once, then repeatedly call `observe` and `predict` to generate forecasts as new data arrives. Every forecasting project should start with a naive baseline, and this notebook shows you how.

### Reduction Forecaster ([View](/examples/point/reduction_forecaster/) | [Editable](/examples/point/reduction_forecaster/edit/))

**Tabular ML-Based Forecasting with sklearn Regressors**

Turn any scikit-learn regressor into a time series forecaster with `PointReductionForecaster`. This example shows how the reduction approach tabularizes time series into supervised learning features, fits a regressor, and generates recursive multi-step predictions. Includes hyperparameter tuning with `GridSearchCV`.

### Observe-Predict Workflow ([View](/examples/point/observe_predict_workflow/) | [Editable](/examples/point/observe_predict_workflow/edit/))

**Rolling-Origin Evaluation and Online Forecasting**

Deep dive into Yohou's stateful lifecycle: `observe`, `predict`, `observe_predict`, and `rewind`. This example demonstrates rolling-origin evaluation where a forecaster is fit once and then incrementally updated as new data becomes available. This is the core pattern for production time series systems.

### Feature Forecasting ([View](/examples/point/feature_forecasting/) | [Editable](/examples/point/feature_forecasting/edit/))

**Forecasting with Forecasted Features**

Chain target and feature forecasters using `ForecastedFeatureForecaster`. When exogenous features aren't known in advance, you need to forecast them too. This example covers three training strategies and shows how to build multi-stage forecasting pipelines.

### Multi-Column Forecasting ([View](/examples/point/multi_column_forecasting/) | [Editable](/examples/point/multi_column_forecasting/edit/))

**Multi-Target Forecasting with ColumnForecaster**

Apply different forecasters to different column subsets using `ColumnForecaster`. This example demonstrates remainder handling strategies and multi-target forecasting on the ETTm1 dataset with column-specific scaling.

### CatBoost Forecasting ([View](/examples/point/catboost_forecasting/) | [Editable](/examples/point/catboost_forecasting/edit/))

**Point Forecasting with CatBoost**

Use `PointReductionForecaster` with `CatBoostRegressor` for gradient-boosted time series forecasting. Compares CatBoost against a linear baseline, demonstrating how Yohou's reduction framework makes it trivial to swap in any scikit-learn-compatible regressor.

### Panel Data Forecasting ([View](/examples/point/panel_data/) | [Editable](/examples/point/panel_data/edit/))

**Panel Data Conventions and Strategies**

Understand panel data conventions (the `__` column separator) and the `panel_strategy` parameter: global, multivariate, and local approaches. Essential reading before working with grouped time series.

### Panel Point Forecasting ([View](/examples/point/panel_forecasting/) | [Editable](/examples/point/panel_forecasting/edit/))

**Global vs Local Models for Panel Data**

Compare global and local panel forecasting strategies. This example covers `ColumnForecaster` specialization, per-group scoring, and selective group training. These are the tools you need for production panel forecasting.
