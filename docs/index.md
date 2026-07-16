![Yohou](assets/logo_dark.png#only-dark){width=800}
![Yohou](assets/logo_light.png#only-light){width=800}

Yohou is a time series forecasting framework built on top of Scikit-Learn's ecosystem. It provides a unified interface for building, extending, and comparing any forecasting model, from sklearn-native reductions to statistical models or deep learning integrations and hyperparameter optimization workflows. All models share a consistent API with native DataFrame support, Scikit-Learn-based compositions, and first-class cross-validation.

<div class="grid cards" markdown>

-   **Tutorials**

    ---

    Install Yohou, load a dataset, fit a forecaster, and generate your first predictions.

    [Tutorials](pages/tutorials/index.md)

-   **How-to Guides**

    ---

    Step-by-step guides for panel data, exogenous features, ensembles, custom estimators, and more.

    [How-to Guides](pages/how-to/index.md)

-   **Explanation**

    ---

    Understand the fit/observe/predict lifecycle, data formats, preprocessing, and core design decisions.

    [Explanation](pages/explanation/index.md)

-   **API Reference**

    ---

    Complete documentation for every class and function across all submodules.

    [API Reference](pages/api/index.md)

-   **Extensions**

    ---

    Official and community extensions for hyperparameter optimization, deep learning integrations, and more.

    [Extensions](pages/reference/extensions.md)

-   **Examples**

    ---

    Interactive Marimo notebooks demonstrating forecasting, metrics, visualization, and more.

    [Examples](pages/examples/index.md)

</div>

## Key Features

- **Polars-native**: All data flows use `polars.DataFrame` with a mandatory `"time"` column.
- **Sklearn-compatible**: Standard `fit`/`predict` API with a consistent interface across all forecaster types.
- **Reduction forecasting**: Wrap any sklearn regressor or classifier and Yohou handles windowing, tabularization, and recursive prediction automatically.
- **Point, interval, and class-probability forecasting**: From naive baselines to conformal prediction intervals and calibrated class-probability distributions.
- **Panel data**: First-class support for multiple related time series via the `__` column naming convention, with per-group models via [`LocalPanelForecaster`](pages/api/generated/yohou.compose.LocalPanelForecaster.md).
- **Incremental observation**: Call `observe()` to feed new data, `rewind()` to roll back state, and `observe_predict()` to fast-forward and forecast in one step without retraining.
- **Stateful transformers**: All transformers are stateful and fitted separately from the forecaster, supporting incremental observation and rewind in full preprocessing pipelines.
- **Composable pipelines**: Decomposition pipelines, feature pipelines, feature unions, and column transformers that compose like sklearn.
- **Cross-validation and model selection**: Temporal splitters ([`ExpandingWindowSplitter`](pages/api/generated/yohou.model_selection.ExpandingWindowSplitter.md), [`SlidingWindowSplitter`](pages/api/generated/yohou.model_selection.SlidingWindowSplitter.md)) and [`GridSearchCV`](pages/api/generated/yohou.model_selection.GridSearchCV.md)/[`RandomizedSearchCV`](pages/api/generated/yohou.model_selection.RandomizedSearchCV.md) with no data leakage.
- **Metrics**: Point, interval, and class-probability scorers with stepwise, vintagewise, componentwise, and groupwise aggregation modes.

## What's New

See the [Changelog](pages/reference/changelog.md) for the latest release notes and updates.

## License

This project is licensed under the terms of the [Apache-2.0 License](https://github.com/stateful-y/yohou/blob/main/LICENSE).

## Acknowledgements

We would like to thank [Evolta Technologies](https://www.evolta-technologies.com/) for their support to the project.

![Evolta Technologies](assets/evolta_logo.png){width=400}

This project is maintained by [stateful-y](https://stateful-y.io), an ML consultancy specializing in time series data science & engineering. If you're interested in collaborating or learning more about our services, please visit our website.

![Made by stateful-y](assets/made_by_stateful-y.png){width=200}
