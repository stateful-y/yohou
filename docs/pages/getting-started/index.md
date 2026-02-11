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

```python
from yohou.datasets import load_air_passengers

y = load_air_passengers()
print(y.head())
```

```
shape: (5, 2)
┌────────────┬─────┐
│ time       ┆ y   │
│ ---        ┆ --- │
│ date       ┆ i64 │
╞════════════╪═════╡
│ 1949-01-01 ┆ 112 │
│ 1949-02-01 ┆ 118 │
│ 1949-03-01 ┆ 132 │
│ 1949-04-01 ┆ 129 │
│ 1949-05-01 ┆ 121 │
└────────────┴─────┘
```

## Fit a Forecaster

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster

forecaster = PointReductionForecaster(estimator=Ridge(), window_length=12)
forecaster.fit(y[:120], forecasting_horizon=12)
```

## Predict

```python
y_pred = forecaster.predict(forecasting_horizon=12)
print(y_pred.head())
```

## Plot

```python
from yohou.plotting import plot_forecast

fig = plot_forecast(y, y_pred, title="Air Passengers Forecast")
fig.show()
```

## Next Steps

- **[Installation](installation.md)**: Detailed installation options including dev setup and optional packages
- **[Overview](overview.md)**: What Yohou can do: a tour of all capabilities
- **[User Guide](../user-guide/index.md)**: Deep dive into core concepts and best practices
- **[API Reference](../api/index.md)**: Complete documentation for every class and function
- **[Examples](../examples/index.md)**: Interactive notebooks for hands-on learning
