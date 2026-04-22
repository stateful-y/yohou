![Yohou](assets/logo_dark.png#only-dark){width=800}
![Yohou](assets/logo_light.png#only-light){width=800}

Yohou bridges Scikit-Learn's tabular machine learning ecosystem with time series forecasting. It treats forecasting as a supervised learning reduction problem while preserving temporal structure, giving you the full power of sklearn estimators, pipelines, and cross-validation with native polars DataFrames.

<div class="grid cards" markdown>

-   **Get Started in 5 Minutes**

    ---

    Install Yohou, load a dataset, fit a forecaster, and generate predictions.

    [Getting Started](pages/tutorials/getting-started.md)

-   **Learn the Concepts**

    ---

    Understand core data formats, the fit/observe/predict lifecycle, preprocessing, and composition.

    [Explanation](pages/explanation/core-concepts.md)

-   **Reference**

    ---

    Complete documentation for every class and function across all submodules, plus extensions and changelog.

    [Reference](pages/reference/overview.md)

-   **See It In Action**

    ---

    Interactive Marimo notebooks demonstrating forecasting, metrics, visualization, and more.

    [Examples](pages/examples/index.md)

</div>

## Key Features

- **Polars-native**: All data flows use `polars.DataFrame` with a mandatory `"time"` column. No pandas required.
- **sklearn-compatible**: Standard `fit`/`predict` API extended with `observe`, `rewind`, and `observe_predict` for time series.
- **Point, interval & classification forecasting**: From naive baselines to conformal prediction intervals and calibrated class-probability distributions.
- **Ensemble methods**: Combine multiple forecasters with voting (mean, median, envelope, soft/hard voting).
- **Panel data**: First-class support for multiple related time series via the `__` column naming convention.
- **Composition**: Pipelines, feature unions, column transformers, and decomposition pipelines that compose like sklearn.
- **Time-weighted learning**: Metadata routing support for `time_weight` and `vintage_weight` in fitting and scoring.
- **Interactive visualization**: Plotly-based plotting functions for exploration, diagnostics, and evaluation (install with `pip install yohou[plotting]`).
- **Remote datasets**: 10 `fetch_*` functions for Monash/Zenodo time series with local Parquet caching (univariate, multivariate, panel, and classification).

## What's New

See the [Changelog](pages/reference/changelog.md) for the latest release notes and updates.

## License

This project is licensed under the terms of the [Apache-2.0 License](https://github.com/stateful-y/yohou/blob/main/LICENSE).

## Acknowledgements

We would like to thank [Evolta Technologies](https://www.evolta-technologies.com/) for their support to the project.

![Evolta Technologies](assets/evolta_logo.png){width=400}

This project is maintained by [stateful-y](https://stateful-y.io), an ML consultancy specializing in time series data science & engineering. If you're interested in collaborating or learning more about our services, please visit our website.

![Made by stateful-y](assets/made_by_stateful-y.png){width=200}
