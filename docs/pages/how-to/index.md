# How-to Guides

Task-oriented recipes for common Yohou workflows. Each guide assumes you have completed the [tutorials](../tutorials/getting-started.md) and are familiar with the basics.

## Setup

- [Installation](installation.md): Install Yohou with pip, uv, or conda, including optional extras and development setup.

## Forecasting

- [Choose a Forecasting Method](choose-forecasting-method.md): Pick the right forecaster and reduction strategy for your data.
- [Build Reduction Forecasters](build-reduction-forecasters.md): Create lag-feature forecasters, combine feature transformers, and choose a reduction strategy.
- [Produce Prediction Intervals](interval-forecasting.md): Wrap a point forecaster with [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/) to generate calibrated prediction intervals.
- [Combine Forecasters with Ensembles](ensemble-forecasting.md): Use voting ensembles to combine multiple forecasters.
- [Forecast with CatBoost](forecast-with-catboost.md): Use gradient-boosted trees for point and interval time series forecasting.
- [Forecast with Class Probabilities](class-probability-forecasting.md): Predict discrete class labels with calibrated probabilities.

## Evaluation & Tuning

- [Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md): Score predictions with point, interval, and classification metrics.
- [Evaluate with Multi-vintage Scoring](multi-vintage-scoring.md): Generate forecasts from successive observation points and break down errors by vintage and horizon step.
- [Tune Forecaster Hyperparameters](tune-hyperparameters.md): Search and tune forecaster hyperparameters with [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) and [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/).

## Visualization

- [Visualize Forecasts](visualize-forecasts.md): Plot forecast output against actuals using [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/), [`plot_residuals`](/pages/api/generated/yohou.plotting.evaluation.plot_residuals/), and [`plot_calibration`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/).
- [Visualize and Compare Model Scores](visualize-scores.md): Plot per-step accuracy, per-vintage trends, score distributions, and model comparisons.

## Data & Features

- [Work with Panel Data](panel-data.md): Handle multiple related time series with the `__` column naming convention.
- [Use Exogenous Features](exogenous-features.md): Incorporate external predictors using X_actual, X_future, and X_forecast.
- [Work with Forecast Vintages](forecast-vintages.md): Prepare, align, and predict with `X_forecast` features from upstream models stamped with an issuance time.
- [Add Calendar and Time Features](time-features.md): Engineer temporal features like day of week, month, and holidays.
- [Use Time Weighting](time-weighting.md): Apply non-uniform weights to emphasize recent or seasonal observations.

## Preprocessing & Pipelines

- [Use Preprocessing Transformers](use-preprocessing-transformers.md): Apply [`ColumnTransformer`](/pages/api/generated/yohou.compose.column_transformer.ColumnTransformer/), [`FunctionTransformer`](/pages/api/generated/yohou.preprocessing.function.FunctionTransformer/), sklearn scalers, and window transformers.
- [Clean and Resample Time Series](clean-and-resample.md): Detect type mismatches, validate value ranges, and change frequency with [`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/) and [`Upsampler`](/pages/api/generated/yohou.preprocessing.resampling.Upsampler/).
- [Compose Feature Pipelines](compose-feature-pipelines.md): Build [`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/) chains, combine branches with [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/), and compose panel-aware pipelines.
- [Apply Stationarity Transforms](apply-stationarity-transforms.md): Remove trends and seasonality with [`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/) and [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/).
- [Apply Signal Processing](apply-signal-processing.md): Smooth noise with digital filters and inspect the frequency spectrum.

## Production Use

- [Save and Load Forecasters](save-load-forecasters.md): Serialize fitted forecasters to disk and reload them for batch scoring or deployment.
- [Handle Missing Data](handle-missing-data.md): Detect and fill gaps in a time series before they propagate through the pipeline.
- [Handle Outliers](handle-outliers.md): Clip or flag outliers and limit their effect on conformal prediction intervals.
- [Handle Complex Seasonality](handle-complex-seasonality.md): Choose between Fourier terms, nested decomposition, and feature engineering for multi-period seasonality.
- [Handle Short Series](handle-short-series.md): Fallback strategies for series too short for seasonal estimation, cross-validation, or scaled metrics.
- [Handle Long Series](handle-long-series.md): Limit history with `observation_horizon`, weight recent errors, and resample to the right frequency.

## Extending

- [Create a Point Forecaster](create-a-point-forecaster.md): Implement a custom point forecaster by subclassing [`BasePointForecaster`](/pages/api/generated/yohou.point.base.BasePointForecaster/).
- [Create an Interval Forecaster](create-an-interval-forecaster.md): Implement a custom interval forecaster by subclassing [`BaseIntervalForecaster`](/pages/api/generated/yohou.interval.base.BaseIntervalForecaster/).
- [Create a Class-Probability Forecaster](create-a-class-proba-forecaster.md): Implement a custom categorical forecaster by subclassing [`BaseClassProbaForecaster`](/pages/api/generated/yohou.class_proba.base.BaseClassProbaForecaster/).
- [Create a Transformer](create-a-transformer.md): Implement a custom time series transformer by subclassing [`BaseTransformer`](/pages/api/generated/yohou.base.transformer.BaseTransformer/).
- [Create Custom Scorers](create-a-scorer.md): Implement a custom evaluation metric by subclassing [`BasePointScorer`](/pages/api/generated/yohou.metrics.base.BasePointScorer/) or [`BaseIntervalScorer`](/pages/api/generated/yohou.metrics.base.BaseIntervalScorer/).
- [Contribute to Yohou](contributing.md): Set up a development environment, run tests, and submit changes.
