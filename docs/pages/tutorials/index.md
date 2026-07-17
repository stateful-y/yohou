# Tutorials

Step-by-step guides that teach you the fundamentals of time series forecasting with Yohou. Start with the first tutorial below, then pick a path based on your interest.

## Start here

- [Getting Started](getting-started.md): Install Yohou, load a dataset, build a full forecasting pipeline, and evaluate multiple models (continuous target). This tutorial gives you the foundation for everything else.

## More output types

- [Class-Probability Forecasting](class-proba-forecasting.md): Forecast categorical outcomes as a probability distribution over classes (categorical target, probability output).
- [Interval Forecasting](interval-forecasting.md): Produce prediction intervals with statistical coverage guarantees (continuous target, prediction intervals).

## Production workflows

- [Forecasting Workflow](forecasting-workflow.md): Cross-validation, hyperparameter search, and residual diagnostics (continuous target).
- [Observe/Predict Workflow](observe-predict.md): Step through a test set in batches using the observe/predict loop (continuous target). Read [Forecasting Workflow](forecasting-workflow.md) first.
- [Exogenous Features](exogenous-features.md): Incorporate external data (X_actual, X_future, X_forecast) into your models (continuous target).
- [Cross-Validation Splitters](cross-validation-splitters.md): Create temporal train/test folds with expanding and sliding window strategies. Assumes familiarity with the core pipeline.

## Advanced modeling

- [Reduction Strategies](reduction-strategies.md): Compare multi-output, direct, and dir-rec reduction strategies.
- [Decomposition](decomposition.md): Build a [`DecompositionPipeline`](/pages/api/generated/yohou.compose.DecompositionPipeline/) with trend, seasonality, and residual forecasters.
- [Panel Data](panel-data.md): Forecast multiple related time series simultaneously using the `__` naming convention and [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.LocalPanelForecaster/). Assumes familiarity with the core pipeline.

## Visualization

- [Exploratory Visualization](exploratory-visualization.md): Plot rolling statistics, boxplots, missing data, outliers, and resampling comparisons.
- [Forecast Visualization](forecast-visualization.md): Visualize single and multi-model forecasts, intervals, decomposition, and time weights.
- [Seasonal Analysis](seasonal-analysis.md): Analyze seasonality with overlays, ACF/PACF, STL decomposition, and heatmaps.
