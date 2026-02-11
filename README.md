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

Yohou is a scikit-learn-compatible time series forecasting framework built on [Polars](https://pola.rs/). It treats forecasting as a supervised learning reduction problem: wrap any sklearn regressor and Yohou handles windowing, tabularization, and recursive prediction while preserving temporal structure. It supports both point and interval forecasting with native panel data capabilities.

Yohou extends sklearn's API with time series-specific operations (`observe`, `rewind`, `observe_predict`) so fitted forecasters can ingest new data incrementally without retraining. After fitting, every forecaster exposes the same `predict` / `predict_interval` / `observe_predict` interface whether it wraps a simple baseline or a full decomposition pipeline.

Currently, Yohou supports Python 3.11+.

## What are the features of Yohou?

- **Reduction forecasting**: Wrap any scikit-learn regressor (`Ridge`, `XGBRegressor`, ...) and Yohou tabularizes, fits, and predicts recursively via `PointReductionForecaster` and `IntervalReductionForecaster`.
- **Incremental observation**: Call `observe()` to feed new data and `observe_predict()` to update and forecast in one step, no refitting required.
- **Composable pipelines**: Chain trend, seasonality, and residual forecasters with `DecompositionPipeline`, or build feature pipelines with `FeaturePipeline`, `FeatureUnion`, and `ColumnTransformer`.
- **Preprocessing & stationarity**: Lag, rolling, and EMA window transforms, signal filters, sklearn scaler wrappers, imputation, outlier handling, and stationarity transforms like `SeasonalDifferencing`, `BoxCoxTransformer`, and Fourier seasonality estimation.
- **Panel data support**: Prefix columns with `group__` and forecasters, transformers, and metrics operate across all groups automatically. Use `ColumnForecaster` or `LocalPanelForecaster` for per-group models.
- **Interval forecasting**: Get calibrated prediction intervals via `SplitConformalForecaster`, `IntervalReductionForecaster` with `DistanceSimilarity`, and conformity scorers.
- **Cross-validation & tuning**: Temporal splitters (`ExpandingWindowSplitter`, `SlidingWindowSplitter`) and `GridSearchCV` / `RandomizedSearchCV` designed for time series with no data leakage across time.
- **Metrics & visualization**: Point and interval scorers with timewise, componentwise, and groupwise aggregation. Over 25 Plotly-based plotting functions for exploration, diagnostics, forecasting, and evaluation.
- **Bundled datasets**: Seven ready-to-use Polars datasets including `air_passengers`, `sunspots`, `walmart_sales`, and `australian_tourism` for quick experimentation.
- **(Experimental) Time-weighted training**: Weight recent or seasonal observations with `exponential_decay_weight`, `linear_decay_weight`, `seasonal_emphasis_weight`, and `compose_weights`, propagated via sklearn metadata routing.

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

Yohou datasets return Polars DataFrames with a `"time"` column.

```python
from yohou.datasets import load_air_passengers

y = load_air_passengers()
y_train, y_test = y[:115], y[115:]
```

### 2. Fit a forecaster

Wrap an sklearn regressor in a `PointReductionForecaster` with a feature pipeline.

```python
from sklearn.linear_model import Ridge

from yohou.compose import FeaturePipeline
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    y_transformers=FeaturePipeline(steps=[("lags", LagTransformer(lags=[1, 12]))]),
    observation_horizon=12,
)
forecaster.fit(y_train, X=None, forecasting_horizon=len(y_test))
```

### 3. Predict and evaluate

After fitting, call `predict` and score against the held-out data.

```python
from yohou.metrics import MeanAbsoluteError
from yohou.plotting import plot_forecast

y_pred = forecaster.predict(forecasting_horizon=len(y_test))
MeanAbsoluteError().score(y_test, y_pred)
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

If you are interested in becoming a maintainer or taking a more active role, please reach out to Guillaume Tauzin on [GitHub Discussions](https://github.com/stateful-y/yohou/discussions).

## Where can I learn more?

Here are the main Yohou resources:

- Full documentation: [https://yohou.readthedocs.io/](https://yohou.readthedocs.io/)
- GitHub Discussions: [https://github.com/stateful-y/yohou/discussions](https://github.com/stateful-y/yohou/discussions)
- Interactive Examples: [https://yohou.readthedocs.io/en/latest/pages/examples/](https://yohou.readthedocs.io/en/latest/pages/examples/)

For questions and discussions, you can also open a [discussion](https://github.com/stateful-y/yohou/discussions).

## License

This project is licensed under the terms of the [Apache-2.0 License](https://github.com/stateful-y/yohou/blob/main/LICENSE).

<p align="center">
  <a href="https://stateful-y.io">
    <img src="docs/assets/made_by_stateful-y.png" alt="Made by stateful-y" width="200">
  </a>
</p>
