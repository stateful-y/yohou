# How to Visualize Forecasts

This guide shows you how to plot predictions against actuals, diagnose
residual patterns, and check interval calibration. All plotting functions
support panel data with automatic faceting.

## Prerequisites

- `yohou[plotting]` installed (`pip install "yohou[plotting]"`)
- A fitted forecaster with predictions ([Getting Started](../tutorials/getting-started.md))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Setup

Load a dataset, fit a forecaster, and generate predictions:

```python
from yohou.datasets import fetch_tourism_monthly
from yohou.point import PointReductionForecaster
from yohou.model_selection import train_test_split
from sklearn.linear_model import Ridge

bunch = fetch_tourism_monthly(n_series=1)
y = bunch.frame
y_train, y_test = train_test_split(y, test_size=12)

forecaster = PointReductionForecaster(estimator=Ridge())
forecaster.fit(y_train, forecasting_horizon=12)
y_pred = forecaster.predict()
```

## 1. Plot a Point Forecast

Pass actuals and predictions to
[`plot_forecast`](/pages/api/generated/yohou.plotting.plot_forecast/).
Include `y_train` to show historical context before the forecast window:

```python
from yohou.plotting import plot_forecast

fig = plot_forecast(y_test, y_pred, y_train=y_train)
fig.show()
```

To limit how much training history is visible, pass `n_history`:

```python
fig = plot_forecast(y_test, y_pred, y_train=y_train, n_history=24)
```

## 2. Compare Multiple Models

Pass a dict of prediction DataFrames to overlay several models. Each model
gets a distinct color:

```python
fig = plot_forecast(
    y_test,
    {"Ridge": y_pred_ridge, "RandomForest": y_pred_rf},
    y_train=y_train,
)
fig.show()
```

To supply your own palette, pass `color_palette=["#e63946", "#457b9d"]`.

## 3. Plot Prediction Intervals

If `y_pred` contains interval columns (named `{target}_lower_{rate}` and
`{target}_upper_{rate}`), `plot_forecast` renders shaded bands automatically.
Pass `coverage_rates` so the legend labels the bands correctly:

```python
fig = plot_forecast(
    y_test, y_pred_interval, y_train=y_train, coverage_rates=[0.90]
)
fig.show()
```

Adjust `band_opacity` (default `0.25`) to make the intervals more or less
prominent.

## 4. Facet Panel Data

For panel datasets, `plot_forecast` detects the panel columns and creates
faceted subplots automatically. Control the layout with `facet_n_cols`:

```python
fig = plot_forecast(
    y_test,
    y_pred,
    y_train=y_train,
    facet_n_cols=3,
)
```

To show only specific groups, pass their prefixes with `groups`:

```python
fig = plot_forecast(
    y_test, y_pred, y_train=y_train, groups=["nsw", "vic"]
)
```

By default, faceting is by individual series (`facet_by="member"`). Set
`facet_by="group"` to facet by panel group instead. The `facet_by`,
`facet_n_cols`, and `groups` parameters work identically on
[`plot_residuals`](/pages/api/generated/yohou.plotting.plot_residuals/)
and
[`plot_calibration`](/pages/api/generated/yohou.plotting.plot_calibration/).

## 5. Inspect Residuals

`plot_residuals`
produces a 4-panel diagnostic (residuals over time, residuals vs fitted,
histogram, Q-Q plot) for a single series, or a faceted layout for panel
data:

```python
from yohou.plotting import plot_residuals

fig = plot_residuals(y_pred, y_test)
fig.show()
```

Pass a single column name via `columns` to get the full 4-panel diagnostic
for that series. When multiple columns are present (the default for panel
data), the function produces a faceted residuals layout:

```python
fig = plot_residuals(
    y_pred, y_test, columns="nsw__demand", marker_size=5, marker_opacity=0.5
)
```

## 6. Check Calibration

`plot_calibration`
compares nominal coverage rates against empirical coverage for interval
forecasts:

```python
from yohou.plotting import plot_calibration

fig = plot_calibration(y_pred_interval, y_test, coverage_rates=[0.90])
fig.show()
```

For class-probability forecasts, omit `coverage_rates` to produce a
reliability diagram instead:

```python
fig = plot_calibration(y_pred_proba, y_test)
```

## 7. Adjust Layout and Style

All plotting functions accept `title`, `x_label`, `y_label`, `width`,
`height`, `line_width`, `show_legend`, and `color_palette`:

```python
fig = plot_forecast(
    y_test, y_pred, y_train=y_train,
    show_transition=False,
    title="Q1 Forecast",
)
```

Set `show_transition=False` to hide the dashed connector between training
history and the forecast window. Refer to the
[`plot_forecast` API reference](/pages/api/generated/yohou.plotting.plot_forecast/)
for the full parameter listing.

## 8. Handle Large Plots

For series with tens of thousands of points, pass `resampler=True` to
enable [plotly-resampler](https://github.com/predict-idlab/plotly-resampler)
for responsive zooming:

```python
fig = plot_forecast(y_test, y_pred, y_train=y_train, resampler=True)
```

In Jupyter, use `resampler="widget"` for an interactive widget instead of
a static figure.

## See Also

- [Produce Prediction Intervals](interval-forecasting.md) for generating interval forecasts to visualize
- [Visualize and Compare Model Scores](visualize-scores.md) for plotting per-step and per-vintage accuracy metrics
- [Forecast with Class Probabilities](class-probability-forecasting.md) for class-probability chart types
- [Working with Panel Data](panel-data.md) for panel data preparation and forecasting
- [About Visualization](../explanation/visualization.md): design decisions behind the plotting API
- [`yohou.plotting` API reference](/pages/api/plotting/)
