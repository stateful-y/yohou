# Examples

Learn Yohou through focused, interactive examples. Each notebook demonstrates one core concept and is runnable in the browser or editable online via the [marimo playground](https://marimo.io).

## Getting Started

Start here. These notebooks cover the core Yohou workflow from a first forecast to walk-forward evaluation, reduction strategies, and working with exogenous features. Each notebook is self-contained.

<!-- GALLERY:section:getting-started -->

## Forecasting Models

Practical how-tos for point forecasters, interval forecasters, class-probability forecasters, and ensembles. Covers gradient boosting integration, quantile regression, conformal methods, and combining multiple models.

<!-- GALLERY:section:forecasting-models -->

## Data & Features

Transformers, feature engineering, stationarity, and pipeline composition. These notebooks show how to clean and resample time series, build lag and rolling features, apply differencing and decomposition, and wire transformers together with FeaturePipeline.

<!-- GALLERY:section:data-features -->

## Panel Data

Yohou handles multi-series panel datasets natively. These notebooks cover every aspect of working with panel data: per-column-subset models with [`ColumnForecaster`](/pages/api/generated/yohou.compose.ColumnForecaster/), per-series models with [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.LocalPanelForecaster/), panel preprocessing and stationarity, prediction intervals across groups, and panel cross-validation.

<!-- GALLERY:section:panel-data -->

## Evaluation & Search

Scoring point, interval, and class-probability forecasts; aggregation modes; time-weighted and multi-vintage scoring; custom scorers; and hyperparameter search with [`GridSearchCV`](/pages/api/generated/yohou.model_selection.GridSearchCV/) and [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.RandomizedSearchCV/).

<!-- GALLERY:section:evaluation-search -->

## Visualization

Interactive plots for exploratory analysis, forecast inspection, seasonal decomposition, correlation diagnostics, signal-processing diagnostics, model selection geometry, and evaluation dashboards. All charts are built with Plotly.

For narrative coverage of each area, see [Exploratory Visualization](../tutorials/exploratory-visualization.md), [Forecast Visualization](../tutorials/forecast-visualization.md), [How to Visualize Forecasts](../how-to/visualize-forecasts.md), and [How to Visualize Scores](../how-to/visualize-scores.md).

<!-- GALLERY:section:visualization -->
