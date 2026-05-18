# Visualization

Yohou includes a Plotly-based plotting module organized around the stages of a
forecasting project. Rather than a bag of unrelated chart types, the functions
map to what you are trying to understand at each phase: exploration, diagnostics,
forecasting, evaluation, and model selection. All functions accept polars
DataFrames, return interactive `plotly.graph_objects.Figure` objects, and handle
panel data automatically through faceted subplots.

## The Forecasting Visualization Workflow

A time series project moves through distinct analytical phases, and different
visualizations serve each one. Exploring raw data asks "what patterns exist?",
diagnostics ask "what structure drives them?", forecast plots ask "did the model
capture them?", and evaluation asks "where does it fail?" Understanding this
progression helps you pick the right plot at the right moment.

### Exploring the Data

The exploration functions reveal the shape, trend, volatility, gaps, and value
distribution of a raw time series.

[`plot_time_series`](/pages/api/generated/yohou.plotting.exploration.plot_time_series/) is the starting point: a line chart of raw observations
over time with support for multiple columns and categorical series rendered as
step lines with markers.
[`plot_rolling_statistics`](/pages/api/generated/yohou.plotting.exploration.plot_rolling_statistics/) exposes two critical properties at once:
a rising or falling rolling mean signals a trend that stationarity transforms
should address, while a widening rolling standard deviation signals
heteroscedasticity (variance that changes over time), which often calls for a
log or Box-Cox transform before modeling. The `agg_method` parameter lets you
switch between mean, median, and standard deviation.
[`plot_boxplot`](/pages/api/generated/yohou.plotting.exploration.plot_boxplot/) grouped by period reveals whether seasonal variation is
additive (constant box height across years) or multiplicative (box height grows
with the level).
[`plot_missing_data`](/pages/api/generated/yohou.plotting.exploration.plot_missing_data/) renders a heatmap that distinguishes between scattered
gaps (typically safe to interpolate) and contiguous blocks (which may require
truncating the series).
[`plot_outliers`](/pages/api/generated/yohou.plotting.exploration.plot_outliers/) flags observations worth investigating via IQR or
z-score detection: clustered outliers at specific calendar positions often
represent genuine events (holiday spikes, supply disruptions) rather than data
errors.
[`plot_distribution`](/pages/api/generated/yohou.plotting.exploration.plot_distribution/) shows histograms with KDE overlays for examining
value distributions, and
[`plot_resampling_comparison`](/pages/api/generated/yohou.plotting.exploration.plot_resampling_comparison/) overlays the same series at different
temporal granularities so you can assess information loss when changing frequency.

### Understanding the Structure

Diagnostic plots expose the lag dependencies, seasonal periodicity, and
frequency-domain structure that determine how past values relate to future ones.

[`plot_autocorrelation`](/pages/api/generated/yohou.plotting.diagnostics.plot_autocorrelation/) confirms periodicity: a significant spike at lag
12 in monthly data confirms yearly seasonality, while slow decay across many
lags suggests the series needs differencing.
[`plot_partial_autocorrelation`](/pages/api/generated/yohou.plotting.diagnostics.plot_partial_autocorrelation/) isolates direct dependence at each
lag, helping determine the number of autoregressive terms or lag features. Both
accept `max_lags` and `confidence_level` parameters and render significance
bounds as dashed lines.

[`plot_seasonality`](/pages/api/generated/yohou.plotting.diagnostics.plot_seasonality/) overlays values by seasonal period, and
[`plot_subseasonality`](/pages/api/generated/yohou.plotting.diagnostics.plot_subseasonality/) reveals within-period patterns (e.g., day-of-week
within each month).
[`plot_seasonal_heatmap`](/pages/api/generated/yohou.plotting.diagnostics.plot_seasonal_heatmap/) renders values on a two-dimensional
period-by-cycle grid.
Together, these reveal whether the seasonal pattern is stable across years
(pointing to [`PatternSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.PatternSeasonalityForecaster/))
or evolving (pointing to Fourier features or adaptive approaches).

[`plot_correlation_heatmap`](/pages/api/generated/yohou.plotting.diagnostics.plot_correlation_heatmap/) shows pairwise correlation across columns
using Pearson, Spearman, or Kendall methods.
[`plot_lag_scatter`](/pages/api/generated/yohou.plotting.diagnostics.plot_lag_scatter/) plots `y(t)` against `y(t-k)` to visually check lag
relationships, while [`plot_scatter_matrix`](/pages/api/generated/yohou.plotting.diagnostics.plot_scatter_matrix/) shows all pairwise scatter
plots in a grid.
[`plot_cross_correlation`](/pages/api/generated/yohou.plotting.diagnostics.plot_cross_correlation/) measures the lagged relationship between two
different series, which is useful for identifying leading indicators or exogenous
feature lags.

### Spectral Analysis

[`plot_spectrum`](/pages/api/generated/yohou.plotting.signal.plot_spectrum/) shows the power spectral density of a series via the
periodogram. Peaks at specific frequencies identify dominant periodicities that
might not be obvious in the time domain. [`plot_phase`](/pages/api/generated/yohou.plotting.signal.plot_phase/) displays the phase angle
from the FFT, which can reveal phase shifts between series or confirm alignment
of periodic components. Both support degree and radian units and optional phase
unwrapping.

### Visualizing Forecasts

Forecast visualization captures how predicted values compare to actuals and what
internal structure a fitted decomposition assigned to trend, seasonality, and
residual.

[`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) overlays predicted values against actuals with optional
historical context controlled by `n_history`. When you pass predictions as a
`dict[str, pl.DataFrame]`, multiple models appear side by side for visual
comparison. Prediction intervals render automatically when the forecast includes
interval columns, with band opacity controlled by `band_opacity` and supported
`coverage_rates` drawn as nested shaded regions. For class-probability forecasts
(columns matching the `{target}_proba_{class}` pattern), the function renders
stacked area charts; categorical forecasts display as step charts.

[`plot_decomposition`](/pages/api/generated/yohou.plotting.forecasting.plot_decomposition/) displays the individual components of a decomposed
series (trend, seasonality, residual) as separate subplots, helping you assess
whether the decomposition captured the right structure.
[`plot_time_weight`](/pages/api/generated/yohou.plotting.forecasting.plot_time_weight/) visualizes how the weighting function emphasized
different time periods during training.

### Evaluating Model Quality

Residual diagnostics, calibration plots, and scoring visualizations reveal where
and when the model struggles.

[`plot_residuals`](/pages/api/generated/yohou.plotting.evaluation.plot_residuals/) renders a four-panel diagnostic layout when given a
single column: residuals over time, histogram, ACF, and Q-Q plot. With multiple
columns it produces faceted time-series residual subplots. A horizontal band
above or below zero signals systematic bias; a fan-out pattern signals
heteroscedasticity.
[`plot_calibration`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/) checks whether prediction intervals achieve their
nominal coverage: points below the diagonal mean the model is overconfident
(intervals too narrow), while points above mean intervals are conservative.

The score visualization functions operate on scorer output DataFrames and slice
performance along different dimensions:
[`plot_score_time_series`](/pages/api/generated/yohou.plotting.evaluation.plot_score_time_series/) shows how a metric evolves over time,
[`plot_score_distribution`](/pages/api/generated/yohou.plotting.evaluation.plot_score_distribution/) shows its histogram,
[`plot_score_summary`](/pages/api/generated/yohou.plotting.evaluation.plot_score_summary/) renders box or violin plots by aggregation
dimension,
[`plot_score_per_step`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_step/) reveals how accuracy degrades across the forecast
horizon (important for choosing between multi-output, direct, and dir-rec
reduction strategies),
[`plot_score_per_vintage`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_vintage/) tracks performance across cross-validation
vintages for hindsight analysis,
[`plot_score_heatmap`](/pages/api/generated/yohou.plotting.evaluation.plot_score_heatmap/) produces a two-dimensional aggregation grid, and
[`plot_group_scores`](/pages/api/generated/yohou.plotting.evaluation.plot_group_scores/) compares panel groups or members directly.

### Cross-Validation and Search

[`plot_splits`](/pages/api/generated/yohou.plotting.model_selection.plot_splits/) visualizes the train/test segments of a
[`BaseSplitter`](/pages/api/generated/yohou.model_selection.split.BaseSplitter/) as a
timeline, confirming that temporal ordering is preserved across folds.
[`plot_cv_results_scatter`](/pages/api/generated/yohou.plotting.model_selection.plot_cv_results_scatter/) connects hyperparameter values to
cross-validation scores, helping you see whether the search explored good
regions of the parameter space.

## Panel Data

Most plotting functions accept a `groups` parameter for working with panel
(grouped) time series. Yohou uses the `group__member` naming convention (double
underscore separator), and the plotting module detects this pattern
automatically: when `groups` is omitted and column names contain `__`, panel
mode activates without any additional configuration.

The `facet_by` parameter controls which axis the subplots facet on:

- `facet_by="member"` (default): one subplot per unique member, with groups
  overlaid. This is the cross-entity comparison view (e.g., "how do sales,
  inventory, and returns compare for each store?").
- `facet_by="group"`: one subplot per group, with members overlaid. This is the
  within-entity view (e.g., "how do all metrics for store A look together?").

The `facet_n_cols` parameter controls how many columns the subplot grid uses.
Passing a specific list of group names to `groups` filters which panels appear,
which is useful for large panel datasets where plotting everything at once would
be overwhelming.

Color consistency across facets is managed by `PanelColorManager`, which assigns
stable colors to member or group names so the same entity always appears in the
same color regardless of subplot position. `LegendTracker` deduplicates legend
entries across subplots so each entity appears in the legend only once.

## Plotly, Styling, and Configuration

Every function returns a `plotly.graph_objects.Figure`, so plots are interactive
by default (zoom, pan, hover, legend toggling). A shared
[`apply_default_layout`](/pages/api/generated/yohou.plotting._utils.apply_default_layout/) utility standardizes appearance: the
`plotly_white` template, Arial font at 12pt, centered titles, and consistent
gridline and border styling. Default width is 1000 pixels; height scales
automatically with the number of subplot rows (300px per row, minimum 400px).

Yohou ships with a 12-color palette accessible via
[`palette_yohou`](/pages/api/generated/yohou.plotting._utils.palette_yohou/): blue, red, green, purple, orange, pink, yellow,
indigo, teal, cyan, gray, and slate. When more series exist than colors, the
palette cycles. All functions accept a `color_palette` parameter to override the
defaults with custom hex codes.

### Parameter Conventions

Functions share consistent parameter naming across modules, making the interface
predictable as you move between phases:

- Exploration and diagnostic functions use `df` as input, `columns` to select
  which value columns to plot, and `groups` for panel data.
- Forecast functions use `y_train`, `y_test`, `y_pred`, and `n_history`.
- All functions accept `title`, `x_label`, `y_label`, `width`, `height`,
  `show_legend`, `color_palette`, and `resampler`.

### Plotly-Resampler for Large Series

When working with high-frequency data (millions of observations), rendering
every point slows the browser. The
[`set_config`](/pages/api/generated/yohou.plotting._utils.set_config/) function enables
[plotly-resampler](https://github.com/predict-idlab/plotly-resampler)
integration, which downsamples the visible trace and loads full-resolution data
on zoom. Three modes are available: `False` (disabled, default), `True`
(Dash-based callback server), and `"widget"` (notebook-native widget). Each
function also accepts a `resampler` parameter to override the global setting
per call. The [`config_context`](/pages/api/generated/yohou.plotting._utils.config_context/) context manager temporarily changes the
configuration and restores it on exit.

## Connections

[Core Concepts](core-concepts.md) covers the `"time"` column contract and panel
data conventions that all plotting functions assume. [Panel Data](panel-data.md)
explains the `group__member` naming convention in depth.
[Residual Diagnostics](residual-diagnostics.md) discusses interpreting
`plot_residuals` output and its connection to conformal prediction.
[Reduction Forecasting](reduction-forecasting.md) describes the prediction
workflow that produces the data these functions visualize. For full function
signatures and parameter reference, see the
[API Reference: yohou.plotting](/pages/api/plotting/).

For practical recipes, see [How to Visualize Forecasts](../how-to/visualize-forecasts.md) and [How to Visualize Scores](../how-to/visualize-scores.md).
