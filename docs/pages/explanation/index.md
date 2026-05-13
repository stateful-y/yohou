# Explanation

Conceptual background for understanding how Yohou works and why it is designed the way it is. These pages complement the tutorials and how-to guides by giving you the mental models needed to use Yohou confidently on unfamiliar problems.

## Foundations

- [Core Concepts](core-concepts.md): The fit/observe/predict lifecycle, data formats, and the reduction approach to forecasting.
- [Time Series Patterns](time-series-patterns.md): Trend, seasonality, cycles, and noise, and how to recognise them before choosing a method.
- [Forecasting](forecasting.md): How Yohou converts time series into supervised learning problems, and what that means for feature construction and prediction.

## Data Shaping

- [Exogenous Features](exogenous-features.md): The three exogenous types (X_actual, X_future, X_forecast), step-indexed columns, and vintage alignment.
- [Preprocessing](preprocessing.md): Stateful vs. stateless transformers, the `BaseTransformer` contract, and incremental observation in pipelines.
- [Stationarity](stationarity.md): Why non-stationary series are problematic for regression models, and how differencing and decomposition help.
- [Feature Pipelines](feature-pipelines.md): `FeaturePipeline`, `FeatureUnion`, and `ColumnTransformer`: how to compose transformers and how `observation_horizon` propagates.

## Forecasting

- [Forecaster Composition](forecaster-composition.md): `DecompositionPipeline`, `ColumnForecaster`, `ForecastedFeatureForecaster`, `LocalPanelForecaster`, and state propagation through composite forecasters.
- [Interval Forecasting](interval-forecasting.md): Prediction intervals, conformal coverage, and when to use `SplitConformalForecaster` vs. quantile regression.
- [Class-Probability Forecasting](class-probability-forecasting.md): Categorical time series, calibration, and the class-probability forecaster API.
- [Ensemble Forecasting](ensemble-forecasting.md): Voting strategies, error diversity, and when ensembles outperform single models.

## Evaluation

- [Forecast Accuracy](forecast-accuracy.md): MAE, MASE, CRPS, and other metrics: which rewards what behavior, and common pitfalls.
- [Model Selection](model-selection.md): Why standard cross-validation fails for time series, and how expanding and sliding windows preserve temporal order.
- [Weighting](weighting.md): Time weighting, vintage weighting, step weighting, and how they apply at both fit time and score time.
- [Residual Diagnostics](residual-diagnostics.md): How to interpret residual plots and what patterns signal unmodelled structure.
- [Visualization](visualization.md): The plotting module, interactive Plotly figures, and choosing the right plot for your task.

## Production and Practical Use

- [Practical Issues](practical-issues.md): Complex seasonality, short series, missing data, structural breaks, and other production realities.
- [Advanced Topics](advanced.md): Metadata routing, the reduction architecture, and the discovery API.
- [Extending Yohou](extending-yohou.md): Abstract base classes, parameter constraints, integration packages, and the systematic test suites for custom components.

## Reference

- [Glossary](glossary.md): Definitions of key terms used across Yohou documentation.
