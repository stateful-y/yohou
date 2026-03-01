# User Guide

The User Guide covers Yohou's design, core concepts, and best practices in depth. Each chapter focuses on a specific area of the library.

!!! info "Under Development"
    The User Guide chapters are being written. Section headings show the planned structure. See the [API Reference](../api/index.md) for complete class and function documentation.

## Chapters

1. **[Core Concepts](core-concepts.md)**: Data format, time column conventions, the fit/observe/predict/rewind lifecycle, observation horizons, and tags.

2. **[Forecasting](forecasting.md)**: Point forecasting with naive baselines, reduction-based forecasters, tabularization, column and feature forecasting.

3. **[Interval Forecasting](interval-forecasting.md)**: Conformal prediction, split conformal, similarity-based intervals, and coverage rates.

4. **[Preprocessing](preprocessing.md)**: Transformers, pipelines, feature unions, column transformers, window operations, signal processing, sklearn wrappers, imputation, and outlier handling.

5. **[Stationarity](stationarity.md)**: Trend and seasonality estimation, stationarity transforms, and the decomposition pipeline.

6. **[Model Selection](model-selection.md)**: Cross-validation splitters, grid and randomized search, scoring, and time-weighted evaluation.

7. **[Visualization](visualization.md)**: Exploration, diagnostics, forecasting, evaluation, and model selection plots.

8. **[Advanced Topics](advanced.md)**: Panel data, metadata routing, time weighting, and custom estimators.
