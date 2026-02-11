# Overview

An overview of Yohou's capabilities organized by module.

!!! info "Under Development"
    Detailed descriptions and usage examples for each capability are being written. See the [API Reference](../api/index.md) for complete documentation.

## Forecasting

Point and interval forecasting with sklearn-compatible estimators. Supports naive baselines, reduction-based forecasters, and conformal prediction.

**Modules**: [`yohou.point`](../api/point.md) · [`yohou.interval`](../api/interval.md)

## Preprocessing

Stateful and stateless time series transformers including rolling windows, signal processing, sklearn scaler wrappers, imputation, and outlier handling.

**Module**: [`yohou.preprocessing`](../api/preprocessing.md)

## Stationarity

Trend and seasonality estimation, stationarity transforms (differencing, log, BoxCox), and decomposition pipelines.

**Module**: [`yohou.stationarity`](../api/stationarity.md)

## Composition

Combine forecasters and transformers into pipelines, feature unions, column transformers, and multi-step decomposition workflows.

**Module**: [`yohou.compose`](../api/compose.md)

## Metrics

Point and interval scoring with per-timestep, per-component, and per-panel aggregation. Conformity scorers for conformal prediction calibration.

**Module**: [`yohou.metrics`](../api/metrics.md)

## Model Selection

Time series cross-validation splitters (expanding window, sliding window) and hyperparameter search (grid, randomized).

**Module**: [`yohou.model_selection`](../api/model-selection.md)

## Visualization

30+ Plotly-based plotting functions for exploration, diagnostics, forecasting, evaluation, and model selection, all with panel data support.

**Module**: [`yohou.plotting`](../api/plotting.md)

## Panel Data

First-class support for multiple related time series using the `__` column naming convention. Panel-aware forecasting, scoring, and visualization.

**Utilities**: [`yohou.utils`](../api/utils.md)

## Datasets

7 bundled time series datasets: Air Passengers, Sunspots, Australian Tourism, Victoria Electricity, Store Sales, Walmart Sales, and ETT-m1.

**Module**: [`yohou.datasets`](../api/datasets.md)
