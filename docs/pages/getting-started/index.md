# Getting Started

Get up and running with Yohou in minutes. This guide walks you through the complete workflow: install, load data, fit a forecaster, predict, evaluate, and plot.

## Install

=== "uv"

    ```bash
    uv add yohou
    ```

=== "pip"

    ```bash
    pip install yohou
    ```

=== "conda"

    ```bash
    conda install -c conda-forge yohou
    ```

See [Installation](installation.md) for mamba, development setup, and optional packages.

## Load a Dataset

Yohou datasets are fetched from [Monash/Zenodo](https://forecastingdata.org) and cached locally as Polars DataFrames with a mandatory `"time"` column.

```python
from yohou.datasets import fetch_tourism_monthly

bunch = fetch_tourism_monthly()
y = bunch.frame.select("time", "T1__tourists").rename({"T1__tourists": "tourists"})
print(y.head())
```

```text
shape: (5, 2)
┌─────────────────────┬──────────┐
│ time                ┆ tourists │
│ ---                 ┆ ---      │
│ datetime[μs]        ┆ f64      │
╞═════════════════════╪══════════╡
│ 1979-01-01 00:00:00 ┆ 1149.87  │
│ 1979-02-01 00:00:00 ┆ 1053.80  │
│ 1979-03-01 00:00:00 ┆ 1388.88  │
│ 1979-04-01 00:00:00 ┆ 873.24   │
│ 1979-05-01 00:00:00 ┆ 927.98   │
└─────────────────────┴──────────┘
```

Split into train and test sets (last 24 months held out):

```python
y_train, y_test = y[:280], y[280:]
forecasting_horizon = len(y_test)
```

## Start Simple: A Seasonal Baseline

The simplest seasonal model repeats values from one year ago. Every more complex model should beat this baseline.

```python
from yohou.point import SeasonalNaive

baseline = SeasonalNaive(seasonality=12)
baseline.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_baseline = baseline.predict(forecasting_horizon=forecasting_horizon)
```

## Upgrade: Reduction Forecaster with Pipelines

A `PointReductionForecaster` wraps any sklearn regressor and converts time series forecasting into supervised learning. Use a `target_transformer` to stabilize the series and a `feature_transformer` to create lag features:

```python
from sklearn.linear_model import Ridge
from yohou.compose import FeaturePipeline
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer
from yohou.stationarity import LogTransformer, SeasonalDifferencing

forecaster = PointReductionForecaster(
    estimator=Ridge(alpha=10),
    target_transformer=FeaturePipeline([
        ("log", LogTransformer(offset=1.0)),
        ("diff", SeasonalDifferencing(seasonality=12)),
    ]),
    feature_transformer=FeaturePipeline([
        ("lag", LagTransformer(lag=[1, 2, 3])),
    ]),
)
forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred = forecaster.predict(forecasting_horizon=forecasting_horizon)
```

The `target_transformer` applies `LogTransformer` and `SeasonalDifferencing` before fitting, and automatically inverts them at prediction time. The `feature_transformer` creates autoregressive lag features from past values.

## Evaluate

Score predictions against held-out data using `MeanAbsoluteError`:

```python
from yohou.metrics import MeanAbsoluteError

scorer = MeanAbsoluteError()
scorer.fit(y_train)
print(f"Baseline MAE: {scorer.score(y_test, y_pred_baseline):.2f}")
print(f"Reduction MAE: {scorer.score(y_test, y_pred):.2f}")
```

## Plot

```python
from yohou.plotting import plot_forecast

plot_forecast(
    y_test,
    {"Baseline": y_pred_baseline, "Reduction": y_pred},
    y_train=y_train,
    title="Tourism Forecast Comparison",
    y_label="Monthly tourists",
)
```

## Go Further: Interactive Quickstart

The Quickstart notebook extends this guide with decomposition pipelines, cross-validation, hyperparameter search, interval forecasting, time-weighted training, and panel data.

=== "View online"

    [:material-book-open-variant: Open Quickstart notebook](../examples/quickstart.md){ .md-button .md-button--primary }

=== "Run locally"

    Launch the interactive Marimo notebook in your browser:

    ```bash
    uv run marimo edit examples/quickstart.py
    ```

## Next Steps

- **[Installation](installation.md)**: conda/mamba, development setup, and optional packages
- **[Overview](overview.md)**: Tour of all Yohou capabilities with code snippets per module
- **[Quickstart notebook](../examples/quickstart.md)**: Full interactive tour (decomposition, CV, intervals, panel data) — or run locally with `uv run marimo edit examples/quickstart.py`
- **[Point forecasting examples](../examples/index.md#point-forecasting)**: Naive, reduction, feature, and multi-column forecasters
- **[API Reference](../api/index.md)**: Complete documentation for every class and function
- **[User Guide](../user-guide/index.md)**: Core concepts and best practices
