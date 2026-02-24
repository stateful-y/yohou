# Examples

Learn Yohou through focused, interactive examples. Each notebook demonstrates one core concept and is runnable in the browser or editable locally via [Marimo](https://marimo.io). Examples are organized from introductory walkthroughs to advanced topics.

## Quickstart

### Yohou Quickstart ([View](/examples/quickstart/) | [Editable](/examples/quickstart/edit/))

**End-to-End Time Series Forecasting with a Scikit-Learn API**

Start here for a complete tour of Yohou's capabilities. This notebook walks through data loading, building baseline forecasters, preprocessing pipelines, decomposition, cross-validation, scoring, panel data, time-weighted training, incremental learning, and interval forecasting. By the end, you'll understand how Yohou bridges scikit-learn's tabular ML ecosystem with time series forecasting.

## Point Forecasting

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

Apply different forecasters to different column subsets using `ColumnForecaster`. This example demonstrates remainder handling strategies and demonstrates multi-target forecasting on the ETTm1 dataset with column-specific scaling.

### CatBoost Forecasting ([View](/examples/point/catboost_forecasting/) | [Editable](/examples/point/catboost_forecasting/edit/))

**Point Forecasting with CatBoost**

Use `PointReductionForecaster` with `CatBoostRegressor` for gradient-boosted time series forecasting. Compares CatBoost against a linear baseline, demonstrating how Yohou's reduction framework makes it trivial to swap in any scikit-learn-compatible regressor.

### Panel Data Forecasting ([View](/examples/point/panel_data/) | [Editable](/examples/point/panel_data/edit/))

**Panel Data Conventions and Strategies**

Understand panel data conventions (the `__` column separator) and the `panel_strategy` parameter: global, multivariate, and local approaches. Essential reading before working with grouped time series.

### Panel Point Forecasting ([View](/examples/point/panel_forecasting/) | [Editable](/examples/point/panel_forecasting/edit/))

**Global vs Local Models for Panel Data**

Compare global and local panel forecasting strategies. This example covers `ColumnForecaster` specialization, per-group scoring, and selective group training. These are the tools you need for production panel forecasting.

## Interval Forecasting

### Conformal Prediction Intervals ([View](/examples/interval/conformal_forecasting/) | [Editable](/examples/interval/conformal_forecasting/edit/))

**Distribution-Free Prediction Intervals**

Build prediction intervals with coverage guarantees using `SplitConformalForecaster`. This example shows how conformal prediction provides valid intervals without distributional assumptions. You only need a point forecaster and a calibration set.

### Interval Reduction Forecasting ([View](/examples/interval/interval_reduction/) | [Editable](/examples/interval/interval_reduction/edit/))

**Native Quantile-Based Intervals**

Generate prediction intervals directly from quantile regression with `IntervalReductionForecaster`. Compares the quantile regression approach with the conformal approach, showing the trade-offs between parametric flexibility and distribution-free guarantees.

### CatBoost MultiQuantile ([View](/examples/interval/catboost_multiquantile/) | [Editable](/examples/interval/catboost_multiquantile/edit/))

**Interval Forecasting with CatBoost's Native MultiQuantile Loss**

Fit all quantile levels simultaneously in a single CatBoost model using the native `MultiQuantile` loss function. More efficient than fitting separate models per quantile.

### Conformal Conformity Scorers ([View](/examples/interval/conformal_conformity_scorers/) | [Editable](/examples/interval/conformal_conformity_scorers/edit/))

**How Conformity Scorers Shape Prediction Intervals**

Explore how different conformity scorers (`Residual`, `AbsoluteResidual`, `GammaResidual`, and more) affect interval calibration and width. Understanding scorer choice is key to getting useful conformal intervals.

### Distance Similarity ([View](/examples/interval/distance_similarity/) | [Editable](/examples/interval/distance_similarity/edit/))

**Adaptive Conformal Intervals via Similarity Weighting**

Use `DistanceSimilarity` to up-weight calibration residuals from observations similar to the prediction target. Produces locally-adaptive intervals that narrow in familiar regimes and widen in unfamiliar ones.

### Panel Prediction Intervals ([View](/examples/interval/panel_intervals/) | [Editable](/examples/interval/panel_intervals/edit/))

**Conformal and Quantile Intervals on Panel Data**

Apply both conformal and quantile regression intervals to panel data with per-group calibration and coverage analysis. Shows how interval forecasting scales to grouped time series.

## Composition

### Pipeline Composition ([View](/examples/compose/pipeline_composition/) | [Editable](/examples/compose/pipeline_composition/edit/))

**Nested Pipeline Patterns**

Master Yohou's composition building blocks: `FeaturePipeline` (sequential transforms, horizon = sum), `FeatureUnion` (parallel transforms), `DecompositionPipeline` with features, and multi-level nesting. The foundation for building complex forecasting systems.

### Column Transformer ([View](/examples/compose/column_transformer/) | [Editable](/examples/compose/column_transformer/edit/))

**Column-Wise Feature Transformation**

Apply different transformers to different columns with `ColumnTransformer`. Includes panel-aware column selection for mixed-type DataFrames.

### Feature Union ([View](/examples/compose/feature_union/) | [Editable](/examples/compose/feature_union/edit/))

**Parallel Feature Engineering with FeatureUnion**

Combine multiple transformers in parallel using `FeatureUnion`. The resulting `observation_horizon` is the maximum across all children. This example explains why and demonstrates the pattern.

### Decomposition Variations ([View](/examples/compose/decomposition_variations/) | [Editable](/examples/compose/decomposition_variations/edit/))

**DecompositionPipeline Configurations**

Explore `DecompositionPipeline` variants: 2-component and 3-component decomposition, `target_transformer` integration, and panel decomposition. Shows how to customize decomposition to match your data's structure.

### ForecastedFeatureForecaster (Advanced) ([View](/examples/compose/forecasted_feature_advanced/) | [Editable](/examples/compose/forecasted_feature_advanced/edit/))

**Advanced Feature Forecasting Strategies**

Deep dive into `ForecastedFeatureForecaster` strategies (`actual`, `predicted`, `rewind`), `split_ratio` tuning, and multivariate feature chains. For when simple feature forecasting isn't enough.

### LocalPanelForecaster ([View](/examples/compose/local_panel_forecaster/) | [Editable](/examples/compose/local_panel_forecaster/edit/))

**Independent Per-Group Forecasting**

Fit independent forecasters per panel group with `LocalPanelForecaster`. Uses `sklearn.base.clone()` for isolation, supports parallel execution, and allows selective group training.

### Panel Pipelines ([View](/examples/compose/panel_pipelines/) | [Editable](/examples/compose/panel_pipelines/edit/))

**All Composition Patterns with Panel Data**

Comprehensive showcase of every composition meta-estimator on panel data: `ColumnForecaster`, `DecompositionPipeline`, `FeaturePipeline`, `FeatureUnion`, and `ColumnTransformer`.

## Preprocessing

### Data Cleaning ([View](/examples/preprocessing/data_cleaning/) | [Editable](/examples/preprocessing/data_cleaning/edit/))

**Imputation and Outlier Handling**

Handle missing values and outliers with `SimpleTimeImputer`, `OutlierThresholdHandler`, and `OutlierPercentileHandler`. Includes `plot_missing_data` for visual gap analysis.

### Advanced Imputation ([View](/examples/preprocessing/advanced_imputation/) | [Editable](/examples/preprocessing/advanced_imputation/edit/))

**Seasonal Imputation and sklearn Wrappers**

Go beyond simple imputation with `SeasonalImputer` and sklearn transformer wrappers. Tests strategies on synthetic gaps with known ground truth.

### Resampling ([View](/examples/preprocessing/resampling/) | [Editable](/examples/preprocessing/resampling/edit/))

**Changing Time Series Frequency**

Upsample and downsample time series with `Upsampler` and `Downsampler`. Essential for aligning series with different frequencies.

### Advanced Resampling ([View](/examples/preprocessing/resampling_advanced/) | [Editable](/examples/preprocessing/resampling_advanced/edit/))

**Aggregation Strategies and Interpolation Methods**

Explore different aggregation strategies, boundary handling, and interpolation methods for resampling. Covers the edge cases that matter in production.

### Window Transformers ([View](/examples/preprocessing/window_transformers/) | [Editable](/examples/preprocessing/window_transformers/edit/))

**Rolling Statistics, EMA, and Lag Features**

Generate windowed features with `LagTransformer`, `RollingStatisticsTransformer`, `SlidingWindowFunctionTransformer`, and `ExponentialMovingAverage`. The core feature engineering toolkit for time series.

### Function Transformer ([View](/examples/preprocessing/function_transformer/) | [Editable](/examples/preprocessing/function_transformer/edit/))

**Custom Function-Based Transforms**

Wrap arbitrary polars operations as sklearn-compatible transformers using `FunctionTransformer`. Supports both stateful and stateless transforms. The escape hatch when built-in transformers don't fit.

### Signal Processing ([View](/examples/preprocessing/signal_processing/) | [Editable](/examples/preprocessing/signal_processing/edit/))

**Butterworth, Chebyshev, and Bessel Filters**

Apply signal processing transformers: `NumericalFilter` (Butterworth, Chebyshev, Bessel), `NumericalDifferentiator`, and `NumericalIntegrator`. For when your time series needs frequency-domain treatment.

### Sklearn Wrappers ([View](/examples/preprocessing/sklearn_wrappers/) | [Editable](/examples/preprocessing/sklearn_wrappers/edit/))

**Sklearn-Compatible Scalers for Polars DataFrames**

Use familiar sklearn transformers (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `PowerTransformer`, and more) directly on polars DataFrames. Bridges sklearn's preprocessing ecosystem with Yohou's data model.

### Panel Preprocessing ([View](/examples/preprocessing/panel_preprocessing/) | [Editable](/examples/preprocessing/panel_preprocessing/edit/))

**Panel-Aware Transformations**

Apply transformations that automatically respect panel structure. Includes manual per-group extraction with `get_group_df` and `dict_to_panel`.

## Stationarity

### Decomposition ([View](/examples/stationarity/decomposition/) | [Editable](/examples/stationarity/decomposition/edit/))

**Trend and Seasonality Decomposition**

Decompose time series into trend, seasonality, and residual components using `PolynomialTrendForecaster`, `PatternSeasonalityForecaster`, `FourierSeasonalityForecaster`, and `DecompositionPipeline`. The foundation for stationary modeling.

### Fourier Tuning ([View](/examples/stationarity/fourier_tuning/) | [Editable](/examples/stationarity/fourier_tuning/edit/))

**Fourier Seasonality Harmonic Selection**

Tune the number of harmonics and estimator choice (ElasticNet, Ridge) for Fourier seasonality. Compares Fourier against Pattern seasonality to help choose the right approach for your data.

### Stationarity Transforms ([View](/examples/stationarity/stationarity_transforms/) | [Editable](/examples/stationarity/stationarity_transforms/edit/))

**Differencing, Log, Box-Cox, and Beyond**

Apply stationarity transformers: `LogTransformer`, `BoxCoxTransformer`, `SeasonalDifferencing`, `SeasonalLogDifferencing`, `SeasonalReturn`, and `ASinhTransformer`. Each with invertible transforms for back-transforming predictions.

### Panel Stationarity ([View](/examples/stationarity/panel_stationarity/) | [Editable](/examples/stationarity/panel_stationarity/edit/))

**Per-Group Decomposition and Stationarity**

Apply decomposition (trend + seasonality) and stationarity transformations per panel group. Shows how stationarity techniques scale to grouped time series.

## Metrics

### Point Metrics ([View](/examples/metrics/point_metrics/) | [Editable](/examples/metrics/point_metrics/edit/))

**MAE, RMSE, MAPE, and All Point Scorers**

Evaluate point forecasts with all 8 scorers: MAE, MSE, RMSE, MedianAE, MAPE, sMAPE, RMSSE, and MASE. Covers aggregation modes and time-weighted scoring for emphasizing recent performance.

### Interval Metrics ([View](/examples/metrics/interval_metrics/) | [Editable](/examples/metrics/interval_metrics/edit/))

**Coverage, Winkler Score, and Interval Evaluation**

Evaluate prediction intervals with `EmpiricalCoverage`, `IntervalScore`, `MeanIntervalWidth`, `PinballLoss`, and `CalibrationError`. Essential for assessing whether your intervals are well-calibrated.

### Conformity Scorers ([View](/examples/metrics/conformity_scorers/) | [Editable](/examples/metrics/conformity_scorers/edit/))

**Conformity Scorers for Conformal Prediction**

Understand `Residual`, `AbsoluteResidual`, `GammaResidual`, and `AbsoluteGammaResidual`. These are the components that determine how conformal intervals adapt to your data's error structure.

### Aggregation Modes ([View](/examples/metrics/aggregation_modes/) | [Editable](/examples/metrics/aggregation_modes/edit/))

**Timewise, Componentwise, Groupwise, and More**

Explore all aggregation modes for scorers on panel data: timewise, componentwise, groupwise, coveragewise, and all. Understanding aggregation is critical for interpreting forecast quality across dimensions.

### Time-Weighted Scoring ([View](/examples/metrics/time_weighted_scoring/) | [Editable](/examples/metrics/time_weighted_scoring/edit/))

**Weighting Functions for Temporal Emphasis**

Apply `exponential_decay_weight`, `linear_decay_weight`, `seasonal_emphasis_weight`, and `compose_weights` to emphasize recent or seasonal performance in your evaluation metrics.

## Model Selection

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

## Plotting

### Exploration ([View](/examples/plotting/exploration/) | [Editable](/examples/plotting/exploration/edit/))

**Exploratory Time Series Visualization**

`plot_time_series`, `plot_rolling_statistics`, `plot_boxplot`, and `plot_missing_data` across univariate, multivariate, and panel datasets.

### Correlation ([View](/examples/plotting/correlation/) | [Editable](/examples/plotting/correlation/edit/))

**Correlation and Scatter Diagnostics**

`plot_correlation_heatmap`, `plot_scatter_matrix`, `plot_cross_correlation`, and `plot_lag_scatter` for correlation analysis.

### Seasonal ([View](/examples/plotting/seasonal/) | [Editable](/examples/plotting/seasonal/edit/))

**Seasonal and Frequency Domain Analysis**

`plot_seasonality`, `plot_subseasonality`, `plot_autocorrelation`, `plot_partial_autocorrelation`, and `plot_periodogram` for seasonal diagnostics.

### Decomposition ([View](/examples/plotting/decomposition/) | [Editable](/examples/plotting/decomposition/edit/))

**STL Decomposition Visualization**

`plot_stl_components` with different component selections, robustness settings, and window parameters.

### Forecasting Visualization ([View](/examples/plotting/forecasting_visualization/) | [Editable](/examples/plotting/forecasting_visualization/edit/))

**Forecast and Prediction Interval Plots**

`plot_forecast`, `plot_components`, and `plot_time_weight` with varied parameters and configurations.

### Evaluation ([View](/examples/plotting/evaluation/) | [Editable](/examples/plotting/evaluation/edit/))

**Model Evaluation Plots**

Residual diagnostics, score time series, per-horizon analysis, model comparison bars, and calibration plots.

### Model Selection ([View](/examples/plotting/model_selection/) | [Editable](/examples/plotting/model_selection/edit/))

**CV and Hyperparameter Search Visualization**

`plot_splits` and `plot_cv_results_scatter` for cross-validation and search result visualization.

### Signal Processing ([View](/examples/plotting/signal_processing/) | [Editable](/examples/plotting/signal_processing/edit/))

**Spectrum and Phase Analysis**

`plot_spectrum` and `plot_phase` before and after Butterworth filtering. For frequency-domain diagnostics on filtered signals.

## Next Steps

- **[User Guide](../user-guide/index.md)**: Deep dive into core concepts and architecture
- **[API Reference](../api/index.md)**: Complete class and function documentation
- **[Development](../development/index.md)**: Contributing and developing new estimators
