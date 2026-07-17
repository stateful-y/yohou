<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/stateful-y/yohou/main/docs/assets/logo_light.png">
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/stateful-y/yohou/main/docs/assets/logo_dark.png">
    <img src="https://raw.githubusercontent.com/stateful-y/yohou/main/docs/assets/logo_light.png" alt="Yohou">
  </picture>
</p>

[![Python Version](https://img.shields.io/pypi/pyversions/yohou)](https://pypi.org/project/yohou/)
[![License](https://img.shields.io/github/license/stateful-y/yohou)](https://github.com/stateful-y/yohou/blob/main/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/yohou)](https://pypi.org/project/yohou/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/yohou)](https://anaconda.org/conda-forge/yohou)
[![codecov](https://codecov.io/gh/stateful-y/yohou/branch/main/graph/badge.svg)](https://codecov.io/gh/stateful-y/yohou)

## What is Yohou?

Yohou is a time series forecasting framework built on top of Scikit-Learn's ecosystem. It provides a unified interface for building, extending, and comparing any forecasting model, from sklearn-native reductions to statistical models or deep learning integrations and hyperparameter optimization workflows. All models share a consistent API with native DataFrame support, Scikit-Learn-based compositions, and first-class cross-validation.

## What are the features of Yohou?

- **Polars-native**: All data flows use `polars.DataFrame` with a mandatory `"time"` column. No pandas required.
- **Sklearn-compatible**: Standard `fit`/`predict` API with a consistent interface across all forecaster types.
- **Reduction forecasting**: Wrap any Scikit-Learn regressor (`Ridge`, `XGBRegressor`, ...) and Yohou handles windowing, tabularization, and recursive prediction via [`PointReductionForecaster`](https://yohou.readthedocs.io/en/latest/pages/api/generated/yohou.point.PointReductionForecaster/), [`IntervalReductionForecaster`](https://yohou.readthedocs.io/en/latest/pages/api/generated/yohou.interval.IntervalReductionForecaster/), and [`ClassProbaReductionForecaster`](https://yohou.readthedocs.io/en/latest/pages/api/generated/yohou.class_proba.ClassProbaReductionForecaster/).
- **Point, interval, and class-probability forecasting**: From naive baselines to conformal prediction intervals ([`SplitConformalForecaster`](https://yohou.readthedocs.io/en/latest/pages/api/generated/yohou.interval.SplitConformalForecaster/)) and calibrated class-probability distributions.
- **Panel data**: Prefix columns with `group__` and all forecasters, transformers, and metrics operate across groups automatically. Use [`ColumnForecaster`](https://yohou.readthedocs.io/en/latest/pages/api/generated/yohou.compose.ColumnForecaster/) or [`LocalPanelForecaster`](https://yohou.readthedocs.io/en/latest/pages/api/generated/yohou.compose.LocalPanelForecaster/) for per-group models.
- **Incremental observation**: Call `observe()` to feed new data, `rewind()` to roll back state, and `observe_predict()` to fast-forward and forecast in one step without retraining.
- **Stateful transformers**: All transformers implement fit/observe/rewind and participate fully in incremental forecasting pipelines, enabling correct state management across training and deployment.
- **Composable pipelines**: Chain trend, seasonality, and residual forecasters with [`DecompositionPipeline`](https://yohou.readthedocs.io/en/latest/pages/api/generated/yohou.compose.DecompositionPipeline/), or build feature pipelines with [`FeaturePipeline`](https://yohou.readthedocs.io/en/latest/pages/api/generated/yohou.compose.FeaturePipeline/), [`FeatureUnion`](https://yohou.readthedocs.io/en/latest/pages/api/generated/yohou.compose.FeatureUnion/), and [`ColumnTransformer`](https://yohou.readthedocs.io/en/latest/pages/api/generated/yohou.compose.ColumnTransformer/).
- **Cross-validation and model selection**: Temporal splitters ([`ExpandingWindowSplitter`](https://yohou.readthedocs.io/en/latest/pages/api/generated/yohou.model_selection.ExpandingWindowSplitter/), [`SlidingWindowSplitter`](https://yohou.readthedocs.io/en/latest/pages/api/generated/yohou.model_selection.SlidingWindowSplitter/)) and [`GridSearchCV`](https://yohou.readthedocs.io/en/latest/pages/api/generated/yohou.model_selection.GridSearchCV/)/[`RandomizedSearchCV`](https://yohou.readthedocs.io/en/latest/pages/api/generated/yohou.model_selection.RandomizedSearchCV/) designed for time series with no data leakage.
- **Metrics**: Point, interval, and class-probability scorers with stepwise, vintagewise, componentwise, and groupwise aggregation.

## How to install Yohou?

Install the Yohou package using `pip`:

```bash
pip install yohou
```

or using `uv`:

```bash
uv pip install yohou
```

or using `conda`:

```bash
conda install -c conda-forge yohou
```

or using `mamba`:

```bash
mamba install -c conda-forge yohou
```

or alternatively, add `yohou` to your `requirements.txt` or `pyproject.toml` file.

## How to get started with Yohou?

### 1. Load data and split

Yohou datasets are fetched from [Monash/Zenodo](https://forecastingdata.org) and return a `Bunch` with a `.frame` attribute (a Polars DataFrame with a `"time"` column).

```python
from yohou.datasets import fetch_sunspot

bunch = fetch_sunspot()
y = bunch.frame
y_train, y_test = y[:-30], y[-30:]
```

### 2. Fit a forecaster

Wrap an sklearn regressor in a `PointReductionForecaster` with preprocessing pipelines.

```python
from sklearn.linear_model import Ridge

from yohou.compose import FeaturePipeline
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer
from yohou.stationarity import SeasonalDifferencing

forecaster = PointReductionForecaster(
    estimator=Ridge(alpha=10),
    target_transformer=FeaturePipeline([
        ("diff", SeasonalDifferencing(seasonality=27)),
    ]),
    feature_transformer=FeaturePipeline([
        ("lag", LagTransformer(lag=[1, 2, 3, 27])),
    ]),
)
forecaster.fit(y_train, forecasting_horizon=len(y_test))
```

### 3. Predict and evaluate

After fitting, call `predict` and score against the held-out data.

```python
from yohou.metrics import MeanAbsoluteError
from yohou.plotting import plot_forecast

y_pred = forecaster.predict(forecasting_horizon=len(y_test))
scorer = MeanAbsoluteError()
scorer.fit(y_train)
scorer.score(y_test, y_pred)
plot_forecast(y_test, y_pred, y_train=y_train)
```

## How do I use Yohou?

Full documentation is available at [https://yohou.readthedocs.io/](https://yohou.readthedocs.io/).

Interactive examples are available in the `examples/` directory:

- **Online**: [https://yohou.readthedocs.io/en/latest/pages/examples/](https://yohou.readthedocs.io/en/latest/pages/examples/)
- **Locally**: Run `marimo edit examples/quickstart.py` to open an interactive notebook

## Can I contribute?

We welcome contributions, feedback, and questions:

- **Report issues or request features**: [GitHub Issues](https://github.com/stateful-y/yohou/issues)
- **Join the discussion**: [GitHub Discussions](https://github.com/stateful-y/yohou/discussions)
- **Contributing Guide**: [CONTRIBUTING.md](https://github.com/stateful-y/yohou/blob/main/CONTRIBUTING.md)

If you are interested in becoming a maintainer or taking a more active role, please reach out to the maintainers at <contact at stateful-y.io>.

## Where can I learn more?

Here are the main Yohou resources:

- Full documentation: [https://yohou.readthedocs.io/](https://yohou.readthedocs.io/)
- Interactive Examples: [https://yohou.readthedocs.io/en/latest/pages/examples/](https://yohou.readthedocs.io/en/latest/pages/examples/)

For questions and discussions, you can also open a [discussion](https://github.com/stateful-y/yohou/discussions).

## License

This project is licensed under the terms of the [Apache-2.0 License](https://github.com/stateful-y/yohou/blob/main/LICENSE).

## Acknowledgements

We would like to thank [Evolta Technologies](https://www.evolta-technologies.com/) for their support to the project.

<br>

<p align="center">
  <a href="https://www.evolta-technologies.com/">
    <img src="docs/assets/evolta_logo.png" alt="Evolta Technologies" width="400">
  </a>
</p>

<br>

This project is maintained by [stateful-y](https://stateful-y.io), an ML consultancy specializing in time series data science & engineering. If you're interested in collaborating or learning more about our services, please visit our website.

<p align="center">
  <a href="https://stateful-y.io">
    <img src="docs/assets/made_by_stateful-y.png" alt="Made by stateful-y" width="200">
  </a>
</p>
