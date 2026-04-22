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

See [Installation](../how-to/installation.md) for mamba, development setup, and optional packages.

## Load a Dataset

Yohou datasets are fetched from [Monash/Zenodo](https://forecastingdata.org) and cached locally as Polars DataFrames with a mandatory `"time"` column.

```python
from yohou.datasets import fetch_tourism_monthly

bunch = fetch_tourism_monthly()
y = bunch.frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
print(y.head())
```

```text
shape: (5, 2)
┌─────────────────────┬───────────┐
│ time                ┆ tourists  │
│ ---                 ┆ ---       │
│ datetime[μs]        ┆ f64       │
╞═════════════════════╪═══════════╡
│ 1979-01-01 00:00:00 ┆ 1149.87   │
│ 1979-02-01 00:00:00 ┆ 1053.8002 │
│ 1979-03-01 00:00:00 ┆ 1388.8798 │
│ 1979-04-01 00:00:00 ┆ 1783.3702 │
│ 1979-05-01 00:00:00 ┆ 1921.0252 │
└─────────────────────┴───────────┘
```

Split into train and test sets (last 24 months held out):

```python
y_train, y_test = y[:-24], y[-24:]
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

## Evaluate

Score the baseline against held-out data using [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.point.MeanAbsoluteError/):

```python
from yohou.metrics import MeanAbsoluteError

scorer = MeanAbsoluteError()
scorer.fit(y_train)
print(f"Baseline MAE: {scorer.score(y_test, y_pred_baseline):.2f}")
```

```text
Baseline MAE: 221.68
```

## Plot

[`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) overlays predictions on held-out actuals:

```python
from yohou.plotting import plot_forecast

plot_forecast(
    y_test,
    {"Baseline": y_pred_baseline},
    y_train=y_train,
    title="Tourism Forecast: Seasonal Naive Baseline",
    y_label="Monthly tourists",
)
```

## What We Built

You installed Yohou, loaded a real dataset, fit a seasonal baseline, scored it, and plotted the forecast. Every model you build will follow this same pattern: load data, split, fit, predict, score, plot.

The baseline is a useful sanity check but not competitive. In **[Your First Forecast](first-forecast.md)**, you will build a reduction pipeline with stationarity transforms and lag features that substantially outperforms the baseline.

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

- **[Your First Forecast](first-forecast.md)**: Build a reduction pipeline with stationarity transforms and lag features
- **[Forecasting Workflow](forecasting-workflow.md)**: Evaluate with cross-validation, hyperparameter search, and residual diagnostics
- **[Installation](../how-to/installation.md)**: conda/mamba, development setup, and optional packages
- **[Core Concepts](../explanation/core-concepts.md)**: Observe/rewind, panel data, and metadata routing

For visualization, install the plotting extra: `pip install yohou[plotting]`
