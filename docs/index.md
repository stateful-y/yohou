![Yohou](assets/logo_dark.png#only-dark){width=800}
![Yohou](assets/logo_light.png#only-light){width=800}

Yohou bridges Scikit-Learn's tabular machine learning ecosystem with time series forecasting. It treats forecasting as a supervised learning reduction problem while preserving temporal structure, giving you the full power of sklearn estimators, pipelines, and cross-validation with native polars DataFrames.

<div class="grid cards" markdown>

-   **Get Started in 5 Minutes**

    ---

    Install Yohou, load a dataset, fit a forecaster, and generate predictions.

    [Getting Started](pages/getting-started/index.md)

-   **Learn the Concepts**

    ---

    Understand core data formats, the fit/observe/predict lifecycle, preprocessing, and composition.

    [User Guide](pages/user-guide/index.md)

-   **API Reference**

    ---

    Complete documentation for every class and function across all 12 submodules.

    [API Reference](pages/api/index.md)

-   **See It In Action**

    ---

    Interactive Marimo notebooks demonstrating forecasting, metrics, visualization, and more.

    [Examples](pages/examples/index.md)

</div>

## Key Features

- **Polars-native**: All data flows use `polars.DataFrame` with a mandatory `"time"` column. No pandas required.
- **sklearn-compatible**: Standard `fit`/`predict` API extended with `observe`, `rewind`, and `observe_predict` for time series.
- **Point & interval forecasting**: From naive baselines to conformal prediction intervals with coverage guarantees.
- **Panel data**: First-class support for multiple related time series via the `__` column naming convention.
- **Composition**: Pipelines, feature unions, column transformers, and decomposition pipelines that compose like sklearn.
- **Time-weighted learning**: Metadata routing support for `time_weight` in fitting and scoring.
- **Interactive visualization**: Plotly-based plotting functions for exploration, diagnostics, and evaluation.
- **Remote datasets**: 8 `fetch_*` functions for Monash/Zenodo time series with local Parquet caching (univariate, multivariate, and panel).

## What's New

See the [Changelog](pages/development/changelog.md) for the latest release notes and updates.

Detailed reference for the Yohou API, including classes, functions, and configuration options.

## License

Yohou is open source and licensed under the [Apache-2.0 License](https://opensource.org/licenses/Apache-2.0). You are free to use, modify, and distribute this software under the terms of this license.
