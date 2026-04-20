# Visualization

Yohou ships 29 Plotly-based plotting functions for exploring, diagnosing, and evaluating time series and forecasts. Every function returns a `go.Figure` (or `dict[str, go.Figure]` for panel-aware plots with `separate=True`), and panel data is auto-detected via the `group__member` column convention.

**API Reference**: [`yohou.plotting`](../api/plotting.md)
**Examples**: [Plotting](../examples/plotting.md)

## Overview

### Common Parameters

All plot functions share a consistent set of keyword-only layout parameters:

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `title` | `str \| None` | `None` | Plot title (auto-generated if omitted) |
| `x_label` | `str \| None` | `None` | X-axis label |
| `y_label` | `str \| None` | `None` | Y-axis label |
| `width` | `int \| None` | `None` | Plot width in pixels |
| `height` | `int \| None` | `None` | Plot height in pixels |
| `show_legend` | `bool` | `True` | Toggle legend visibility |
| `color_palette` | `list[str] \| None` | `None` | Custom hex color palette (defaults to `palette_yohou`) |

### Panel Data Support

22 of 29 functions auto-detect panel data when columns follow the `group__member` naming convention (e.g. `sales__a`, `sales__b`). Panel-aware functions accept:

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `groups` | `list[str] \| None` | `None` | Filter to specific groups |
| `columns` | `str \| list[str] \| None` | `None` | Filter to specific members within groups |
| `facet_by` | `"group" \| "member" \| None` | `"member"` | Faceting dimension for subplots |
| `facet_n_cols` | `int` | `2` | Number of columns in the facet grid |

When `facet_by="member"`, each panel member gets its own subplot with groups overlaid. When `facet_by="group"`, each group gets a subplot with members overlaid.

Three functions (`plot_correlation_heatmap`, `plot_subseasonality`, `plot_scatter_matrix`) also accept `separate=True` to return a `dict[str, go.Figure]` keyed by group name instead of a single faceted figure.

### Customization and Theming

The default 12-color palette is accessible via [`palette_yohou()`][yohou.plotting.palette_yohou]. Override it per-call with the `color_palette` parameter, or use [`get_color_sequence(n)`][yohou.plotting.get_color_sequence] to retrieve `n` colors (cycling if more than 12 are needed).

For large datasets, enable plotly-resampler globally or per-call:

```python
from yohou.plotting import set_config, config_context

# Global
set_config(resampler=True)

# Temporary
with config_context(resampler=True):
    fig = plot_time_series(df)
```

Functions that render along a time axis support `resampler=True` (static resampler) or `resampler="widget"` (interactive widget resampler).

## Exploration

### plot_time_series

Line plot for one or more time series columns. The first function to reach for when inspecting raw data. Supports `connect_gaps`, `line_width`, and `line_dash` for styling.

```python
from yohou.plotting import plot_time_series
fig = plot_time_series(df, columns=["temperature", "humidity"])
```

### plot_rolling_statistics

Overlay rolling window statistics (mean, std, min, max, median, or quantiles) on the original series. The `window_size` parameter accepts an integer or a dict mapping statistic names to different window sizes.

```python
from yohou.plotting import plot_rolling_statistics
fig = plot_rolling_statistics(df, window_size=7, statistics=["mean", "std"])
```

### plot_boxplot

Boxplots grouped by time periods (e.g. `"1mo"`, `"1w"`). Useful for spotting distributional shifts over time.

```python
from yohou.plotting import plot_boxplot
fig = plot_boxplot(df, period="1mo")
```

### plot_missing_data

Visualize missing data patterns. Supports `kind="bars"` (percentage bars), `kind="matrix"` (binary presence heatmap), and `kind="heatmap"` (aggregated heatmap).

```python
from yohou.plotting import plot_missing_data
fig = plot_missing_data(df, kind="matrix")
```

### plot_distribution

Histogram with optional KDE overlay for value distributions.

```python
from yohou.plotting import plot_distribution
fig = plot_distribution(df, columns="y", show_kde=True, n_bins=50)
```

### plot_outliers

Time series with outlier points highlighted. Supports `method="zscore"`, `"iqr"`, or `"percentile"`.

```python
from yohou.plotting import plot_outliers
fig = plot_outliers(df, method="iqr", threshold=1.5)
```

### plot_resampling_comparison

Side-by-side comparison of original vs. resampled time series.

```python
from yohou.plotting import plot_resampling_comparison
fig = plot_resampling_comparison(df_original, df_resampled, columns="y")
```

## Diagnostics

### Autocorrelation (ACF and PACF)

[`plot_autocorrelation`][yohou.plotting.plot_autocorrelation] and [`plot_partial_autocorrelation`][yohou.plotting.plot_partial_autocorrelation] show the (partial) autocorrelation function as a stem/bar plot with optional confidence bands.

```python
from yohou.plotting import plot_autocorrelation, plot_partial_autocorrelation
fig_acf = plot_autocorrelation(df, max_lags=40)
fig_pacf = plot_partial_autocorrelation(df, max_lags=40)
```

### Correlation Diagnostics

[`plot_correlation_heatmap`][yohou.plotting.plot_correlation_heatmap] displays the Pearson correlation matrix. For panel data, defaults to `facet_by="group"` so each group gets its own heatmap. Use `separate=True` to get individual figures per group.

```python
from yohou.plotting import plot_correlation_heatmap
fig = plot_correlation_heatmap(df, show_values=True)
```

### Seasonality Analysis

[`plot_seasonality`][yohou.plotting.plot_seasonality] overlays one line per cycle (e.g. year) on the same axes, with the seasonal position (e.g. month) on the x-axis. Older cycles fade via an opacity ramp controlled by `opacity_power`.

[`plot_subseasonality`][yohou.plotting.plot_subseasonality] shows seasonal subseries - one mini subplot per season with all cycles plotted and an optional mean line.

```python
from yohou.plotting import plot_seasonality, plot_subseasonality
fig_season = plot_seasonality(df, seasonality="month")
fig_sub = plot_subseasonality(df, seasonality="month", show_mean=True)
```

### Lag Scatter

[`plot_lag_scatter`][yohou.plotting.plot_lag_scatter] plots y(t) vs y(t-lag) for analyzing temporal dependencies. Supports multiple lags and seasonal coloring via the `seasonality` parameter.

```python
from yohou.plotting import plot_lag_scatter
fig = plot_lag_scatter(df, lags=[1, 7, 14])
```

### Spectrum and Frequency Analysis

[`plot_spectrum`][yohou.plotting.plot_spectrum] shows the power spectral density via periodogram. Use `log_scale=True` for log-transformed amplitude and `n_peaks` to annotate dominant frequencies.

[`plot_phase`][yohou.plotting.plot_phase] shows the phase angle of each frequency component. Use `unwrap=True` to avoid discontinuities and `angle_unit="degree"` for degrees.

```python
from yohou.plotting import plot_spectrum, plot_phase
fig_spec = plot_spectrum(df, columns="y", log_scale=True, n_peaks=5)
fig_phase = plot_phase(df, columns="y", unwrap=True)
```

### Cross-Correlation

[`plot_cross_correlation`][yohou.plotting.plot_cross_correlation] plots the cross-correlation function (CCF) between pairs of time series. With 2 columns, a single stem plot is shown; with 3+ columns, all unique pairs are rendered in subplots.

```python
from yohou.plotting import plot_cross_correlation
fig = plot_cross_correlation(df, columns=["x", "y", "z"], max_lags=20)
```

### Calendar Heatmap

[`plot_seasonal_heatmap`][yohou.plotting.plot_seasonal_heatmap] renders a 2-D heatmap of values aggregated across two time dimensions (e.g. hour-of-day by month, day-of-week by hour).

```python
from yohou.plotting import plot_seasonal_heatmap
fig = plot_seasonal_heatmap(df, period="month_by_year", agg="mean")
```

### Scatter Matrix

[`plot_scatter_matrix`][yohou.plotting.plot_scatter_matrix] produces an NxN scatter-plot matrix. For panel data, defaults to `facet_by="group"`.

```python
from yohou.plotting import plot_scatter_matrix
fig = plot_scatter_matrix(df, columns=["a", "b", "c"])
```

### Time Series Decomposition

[`plot_decomposition`][yohou.plotting.plot_decomposition] renders decomposition components (trend, seasonal, residual) as vertically stacked subplots. Supports three decomposition backends via the `method` parameter: STL, MSTL, and classical (`seasonal_decompose`). Also accepts pre-computed component DataFrames.

```python
from yohou.plotting import plot_decomposition

# STL decomposition
fig = plot_decomposition(y, ["trend", "seasonal", "residual"], method="stl")

# Classical decomposition (multiplicative)
fig = plot_decomposition(y, ["trend", "seasonal", "residual"], method="classical", model="multiplicative")

# MSTL (multi-seasonal)
fig = plot_decomposition(y, ["trend", "seasonal", "residual"], method="mstl", periods=[24, 168])

# From pre-computed DataFrames
fig = plot_decomposition(y, {"trend": trend_df, "seasonal": seasonal_df})
```

### Subseasonality

See [Seasonality Analysis](#seasonality-analysis) above for `plot_subseasonality`.

## Forecasting Plots

### plot_forecast

The primary forecast visualization. Supports single or multi-model forecasts, optional historical data (`y_train`), and prediction intervals with configurable `coverage_rates`.

```python
from yohou.plotting import plot_forecast
fig = plot_forecast(
    y_test,
    y_pred={"Model A": pred_a, "Model B": pred_b},
    y_train=train_df,
    coverage_rates=[0.5, 0.9],
)
```

### plot_time_weight

Visualize time-based sample weights as a time series, useful for understanding how recent observations are weighted during training.

```python
from yohou.plotting import plot_time_weight
fig = plot_time_weight(y, time_weight=weight_fn)
```

### plot_decomposition

See [Time Series Decomposition](#time-series-decomposition) above.

## Evaluation Plots

### plot_residuals

Four-panel residual diagnostics: residual time series, histogram, Q-Q plot, and residuals vs. fitted. Produces one 4-panel subplot per column, or facets by panel group.

```python
from yohou.plotting import plot_residuals
fig = plot_residuals(y_pred, y_truth)
```

### plot_calibration

Prediction interval calibration plot comparing nominal coverage rates to empirical coverage. Points above the diagonal indicate under-coverage.

```python
from yohou.plotting import plot_calibration
fig = plot_calibration(y_pred_int, y_truth, coverage_rates=[0.5, 0.8, 0.9, 0.95])
```

### plot_score_time_series

Score values over time for one or more forecasts. Auto-clones the scorer with `aggregation_method="componentwise"` to compute per-timestep scores.

```python
from yohou.metrics import MeanAbsoluteError
from yohou.plotting import plot_score_time_series
fig = plot_score_time_series(MeanAbsoluteError(), y_truth, y_pred)
```

### plot_score_summary

Grouped bar chart comparing multiple models across multiple scorers.

```python
from yohou.plotting import plot_score_summary
fig = plot_score_summary(
    {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()},
    y_truth,
    {"Model A": y_pred_a, "Model B": y_pred_b},
)
```

### plot_score_distribution

Histogram and/or KDE of per-timestep scorer values. Use `kind="histogram"`, `"kde"`, or `"both"`.

```python
from yohou.plotting import plot_score_distribution
fig = plot_score_distribution(scorer, y_truth, y_pred, kind="both")
```

### plot_score_per_step

Score degradation by forecast horizon step - shows how accuracy changes as predictions extend further into the future.

```python
from yohou.plotting import plot_score_per_step
fig = plot_score_per_step(scorer, y_truth, y_pred)
```

## Model Selection Plots

### plot_splits

Timeline visualization of cross-validation splits. Each row shows one fold with train/test regions.

```python
from yohou.plotting import plot_splits
fig = plot_splits(y, splitter)
```

### plot_cv_results_scatter

Scatter plot of hyperparameter search results. X-axis is the swept parameter, y-axis is the mean score. Use `show_error_bars=True` to add standard deviation bars.

```python
from yohou.plotting import plot_cv_results_scatter
fig = plot_cv_results_scatter(cv_results, param_name="alpha", highlight_best=True)
```

## Signal Plots

### plot_phase

See [Spectrum and Frequency Analysis](#spectrum-and-frequency-analysis) above.

### plot_spectrum

See [Spectrum and Frequency Analysis](#spectrum-and-frequency-analysis) above.
