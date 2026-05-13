# Getting Started

In this tutorial, we will load a real time series dataset, fit a seasonal baseline forecaster, score it, and plot the predictions. Along the way, we will encounter the core Yohou workflow: load, split, fit, predict, score, and plot.

## Install

=== "uv"

    ```bash
    uv add yohou
    ```

=== "pip"

    ```bash
    pip install yohou
    ```

See [Installation](../how-to/installation.md) for conda/mamba, development setup, and optional packages.

The plotting functions used in this guide require the plotting extra:

```bash
pip install yohou[plotting]
```

## Load a Dataset

Yohou datasets are fetched from [Monash/Zenodo](https://forecastingdata.org) and cached locally as Polars DataFrames with a mandatory `"time"` column.

```python
from yohou.datasets import fetch_sunspot

bunch = fetch_sunspot()
y = bunch.frame
print(y.head())
```

```text
shape: (5, 2)
┌─────────────────────┬────────────────┐
│ time                ┆ sunspot_number │
│ ---                 ┆ ---            │
│ datetime[μs]        ┆ f64            │
╞═════════════════════╪════════════════╡
│ 1818-01-08 00:00:00 ┆ 64.0           │
│ 1818-01-09 00:00:00 ┆ 65.0           │
│ 1818-01-10 00:00:00 ┆ 63.0           │
│ 1818-01-11 00:00:00 ┆ 57.0           │
│ 1818-01-12 00:00:00 ┆ 61.0           │
└─────────────────────┴────────────────┘
```

Split into train and test sets (last 30 days held out):

```python
y_train, y_test = y[:-30], y[-30:]
forecasting_horizon = len(y_test)
```

## Fit a Seasonal Baseline

The simplest seasonal model repeats values from one cycle ago. At daily resolution, the solar rotation period of 27 days is a natural choice. We pass `forecasting_horizon` at fit time so Yohou knows how many steps ahead it needs to predict.

```python
from yohou.point import SeasonalNaive

baseline = SeasonalNaive(seasonality=27)
baseline.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_baseline = baseline.predict(forecasting_horizon=forecasting_horizon)
```

Notice that `y_pred_baseline` is a Polars DataFrame with the same `"time"` column as the input, aligned to the test period.

## Evaluate

Score the baseline against held-out data using [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.point.MeanAbsoluteError/). Scorers in Yohou are stateful: `scorer.fit(y_train)` stores the training data so that scale-dependent metrics can normalise correctly.

```python
from yohou.metrics import MeanAbsoluteError

scorer = MeanAbsoluteError()
scorer.fit(y_train)
print(f"Baseline MAE: {scorer.score(y_test, y_pred_baseline):.2f}")
```

The output should look something like:

```text
Baseline MAE: 22.38
```

## Plot

[`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) overlays predictions on held-out actuals:

```python
from yohou.plotting import plot_forecast

plot_forecast(
    y_test,
    {"Baseline": y_pred_baseline},
    y_train=y_train.tail(90),
    title="Sunspot Forecast: Seasonal Naive Baseline",
    y_label="Sunspot number",
)
```

## What You Built

You loaded a real dataset, split it into train and test sets, fit a seasonal baseline, scored it with MAE, and plotted the forecast. This is the complete Yohou workflow. Every model you build will follow this same pattern.

## Next Steps

- **[Your First Forecast](first-forecast.md)**: Build a reduction pipeline with lag and rolling features that substantially outperforms the baseline
- **[Forecasting Workflow](forecasting-workflow.md)**: Evaluate with cross-validation, hyperparameter search, and residual diagnostics
- **[Installation](../how-to/installation.md)**: conda/mamba, development setup, and optional packages
- **[Core Concepts](../explanation/core-concepts.md)**: Observe/rewind, panel data, and metadata routing

The Quickstart notebook extends this guide with decomposition pipelines, cross-validation, hyperparameter search, interval forecasting, time-weighted training, and panel data.

[View](../examples/quickstart.md) · [Open in marimo](/examples/quickstart/edit/)
