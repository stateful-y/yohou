# Model Selection

### CV Splitters ([View](/examples/model_selection/cv_splitters/) | [Editable](/examples/model_selection/cv_splitters/edit/))

**Cross-Validation Splitters for Time Series**

Split time series for cross-validation with `ExpandingWindowSplitter` and `SlidingWindowSplitter`. Covers gap control between train and test sets, and includes `plot_splits` for visual verification.

### Hyperparameter Search ([View](/examples/model_selection/hyperparameter_search/) | [Editable](/examples/model_selection/hyperparameter_search/edit/))

**GridSearchCV and RandomizedSearchCV**

Tune forecaster hyperparameters with time series-aware cross-validation. This example demonstrates `GridSearchCV` and `RandomizedSearchCV` with temporal splitters. The same API as scikit-learn, adapted for forecasting.

### Interval Search ([View](/examples/model_selection/interval_search/) | [Editable](/examples/model_selection/interval_search/edit/))

**Tuning Interval Forecaster Hyperparameters**

Search over `calibration_size`, `conformity_scorer`, and quantile levels for interval forecasters using `GridSearchCV`. Shows how to optimize prediction interval quality.

### Multi-Metric Search ([View](/examples/model_selection/multi_metric_search/) | [Editable](/examples/model_selection/multi_metric_search/edit/))

**Multiple Scorers and Refit Strategies**

Use a dictionary of scorers with multiple refit strategies in `RandomizedSearchCV`. For when a single metric isn't enough to select the best model.

### Panel Cross-Validation ([View](/examples/model_selection/panel_cross_validation/) | [Editable](/examples/model_selection/panel_cross_validation/edit/))

**Cross-Validation for Panel Data**

Apply time series CV to panel data with expanding and sliding windows. Combines `GridSearchCV` with panel-aware splitters for grouped hyperparameter tuning.

### Optuna Search ([View](/examples/model_selection/optuna_search/) | [Editable](/examples/model_selection/optuna_search/edit/))

**Bayesian Hyperparameter Optimisation with Optuna**

Use `OptunaSearchCV` from the `yohou-optuna` package for Bayesian search with trial inspection. More sample-efficient than grid or random search for large parameter spaces.

### Nixtla Forecasters ([View](/examples/model_selection/nixtla_forecasters/) | [Editable](/examples/model_selection/nixtla_forecasters/edit/))

**Statistical and Neural Forecasters via Nixtla**

Integrate AutoARIMA, AutoETS, HoltWinters, NBEATS, and NHITS from the `yohou-nixtla` package. Brings classical statistical and deep learning forecasters into Yohou's unified API.

### Nixtla Panel ([View](/examples/model_selection/nixtla_panel/) | [Editable](/examples/model_selection/nixtla_panel/edit/))

**Nixtla Forecasters on Panel Data**

Run statistical forecasters (AutoARIMA, AutoETS) on panel data using Yohou's `__` convention. Shows how Nixtla's models work with grouped time series.
