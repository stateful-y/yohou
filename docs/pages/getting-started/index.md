# Getting Started

Get up and running with Yohou in minutes. This guide walks you through the complete workflow: install → load data → fit a forecaster → predict → plot.

!!! info "Under Development"
    This quick start guide is being expanded with more detailed explanations of each step. The code examples below are fully functional.

## Install

```bash
uv add yohou
```

Or with pip:

```bash
pip install yohou
```

## Load a Dataset

Datasets are downloaded from [Monash/Zenodo](https://forecastingdata.org) and cached locally.

```python
from yohou.datasets import fetch_tourism_monthly

bunch = fetch_tourism_monthly()
y = bunch.frame.select("time", "T1__tourists").rename({"T1__tourists": "tourists"})
print(y.head())
```

```
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

## Fit a Forecaster

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster

forecaster = PointReductionForecaster(estimator=Ridge(), window_length=12)
forecaster.fit(y[:280], forecasting_horizon=12)
```

## Predict

```python
y_pred = forecaster.predict(forecasting_horizon=12)
print(y_pred.head())
```

## Plot

```python
from yohou.plotting import plot_forecast

fig = plot_forecast(y, y_pred, title="Tourism Forecast")
fig.show()
```

## Next Steps

- **[Installation](installation.md)**: Detailed installation options including dev setup and optional packages
- **[Overview](overview.md)**: What Yohou can do: a tour of all capabilities
- **[User Guide](../user-guide/index.md)**: Deep dive into core concepts and best practices
- **[API Reference](../api/index.md)**: Complete documentation for every class and function
- **[Examples](../examples/index.md)**: Interactive notebooks for hands-on learning
