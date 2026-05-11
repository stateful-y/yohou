# How-to Guides

Task-oriented recipes for common Yohou workflows. Each guide assumes you have completed the [tutorials](../tutorials/getting-started.md) and are familiar with the basics.

## Setup

- [Installation](installation.md): Install Yohou with pip, uv, or conda, including optional extras and development setup.

## Forecasting

- [Choose a Forecasting Method](choose-forecasting-method.md): Pick the right forecaster and reduction strategy for your data.
- [Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md): Score predictions with point, interval, and classification metrics.
- [Combine Forecasters with Ensembles](ensemble-forecasting.md): Use voting ensembles to combine multiple forecasters.
- [Forecast Categorical Time Series](classification-forecasting.md): Predict discrete class labels with calibrated probabilities.

## Data & Features

- [Work with Panel Data](panel-data.md): Handle multiple related time series with the `__` column naming convention.
- [Add Calendar and Time Features](time-features.md): Engineer temporal features like day of week, month, and holidays.
- [Use Exogenous Features](exogenous-features.md): Incorporate external predictors using X_actual, X_future, and X_forecast.
- [Use Time Weighting](time-weighting.md): Apply non-uniform weights to emphasize recent or seasonal observations.

## Extending

- [Create Custom Estimators](custom-estimators.md): Implement a custom point forecaster by subclassing `BasePointForecaster`.
- [Create Custom Scorers](creating-a-scorer.md): Implement a custom evaluation metric by subclassing `BasePointScorer` or `BaseIntervalScorer`.

## Reference

- [Troubleshooting](troubleshooting.md): Solutions for common errors and unexpected behavior.
- [Contributing](contribute.md): Development workflow, coding standards, and how to submit changes.
