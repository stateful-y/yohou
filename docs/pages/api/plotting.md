---
template: api-submodule.html
---

# yohou.plotting

Interactive time series visualization functions using Plotly. All plotting functions support panel data via the `panel_group_names` parameter.

**User guide**: See the [Visualization](../user-guide/visualization.md) section for further details.

## Exploration

| Name | Description |
| --- | --- |
| [`plot_time_series`](generated/yohou.plotting.exploration.plot_time_series.md) | Plot basic line plots for one or more time series. |
| [`plot_rolling_statistics`](generated/yohou.plotting.exploration.plot_rolling_statistics.md) | Plot rolling window statistics (mean, std, min, max, median, quantiles). |
| [`plot_missing_data`](generated/yohou.plotting.exploration.plot_missing_data.md) | Visualize missing data patterns over time. |
| [`plot_seasonality`](generated/yohou.plotting.diagnostics.plot_seasonality.md) | Plot seasonal overlay. |
| [`plot_subseasonality`](generated/yohou.plotting.diagnostics.plot_subseasonality.md) | Plot seasonal subseries. |
| [`plot_lag_scatter`](generated/yohou.plotting.diagnostics.plot_lag_scatter.md) | Plot scatter plots of y(t) vs y(t-lag) for analysing temporal dependencies. |
| [`plot_autocorrelation`](generated/yohou.plotting.diagnostics.plot_autocorrelation.md) | Plot autocorrelation function (ACF) for time series. |
| [`plot_partial_autocorrelation`](generated/yohou.plotting.diagnostics.plot_partial_autocorrelation.md) | Plot partial autocorrelation function (PACF) for time series. |
| [`plot_cross_correlation`](generated/yohou.plotting.diagnostics.plot_cross_correlation.md) | Plot cross-correlation function (CCF) between two time series. |
| [`plot_correlation_heatmap`](generated/yohou.plotting.diagnostics.plot_correlation_heatmap.md) | Plot correlation matrix heatmap for multiple time series. |
| [`plot_scatter_matrix`](generated/yohou.plotting.diagnostics.plot_scatter_matrix.md) | Plot an N×N scatter-plot matrix. |

### Modelling

| Name | Description |
| --- | --- |
| [`plot_decomposition`](generated/yohou.plotting.forecasting.plot_decomposition.md) | Plot time series decomposition as vertically stacked subplots. |
| [`plot_residuals`](generated/yohou.plotting.evaluation.plot_residuals.md) | Plot diagnostic plots for model residuals. |
| [`plot_boxplot`](generated/yohou.plotting.exploration.plot_boxplot.md) | Plot boxplots grouped by time periods. |
| [`plot_calibration`](generated/yohou.plotting.evaluation.plot_calibration.md) | Plot prediction interval calibration. |
| [`plot_time_weight`](generated/yohou.plotting.forecasting.plot_time_weight.md) | Plot time-based weights as a time series visualization. |

### Evaluation

| Name | Description |
| --- | --- |
| [`plot_splits`](generated/yohou.plotting.model_selection.plot_splits.md) | Plot cross-validation splits as a timeline visualization. |
| [`plot_cv_results_scatter`](generated/yohou.plotting.model_selection.plot_cv_results_scatter.md) | Plot hyperparameter search results as a scatter plot. |
| [`plot_forecast`](generated/yohou.plotting.forecasting.plot_forecast.md) | Plot forecasts with historical data and optional prediction intervals. |
| [`plot_score_time_series`](generated/yohou.plotting.evaluation.plot_score_time_series.md) | Plot scorer values over time for one or more forecasts. |
| [`plot_score_distribution`](generated/yohou.plotting.evaluation.plot_score_distribution.md) | Plot the distribution of per-timestep scorer values. |
| [`plot_score_per_horizon`](generated/yohou.plotting.evaluation.plot_score_per_horizon.md) | Plot scorer value by forecast horizon step. |
| [`plot_model_comparison_bar`](generated/yohou.plotting.evaluation.plot_model_comparison_bar.md) | Plot grouped bar chart comparing multiple models across scorers. |

### Signal processing

| Name | Description |
| --- | --- |
| [`plot_spectrum`](generated/yohou.plotting.signal.plot_spectrum.md) | Plot periodogram (power spectral density) for frequency domain analysis. |
| [`plot_phase`](generated/yohou.plotting.signal.plot_phase.md) | Plot the phase of a time series. |

### Utilities

| Name | Description |
| --- | --- |
| [`apply_default_layout`](generated/yohou.plotting._utils.apply_default_layout.md) | Apply default layout configuration to a figure. |
| [`get_color_sequence`](generated/yohou.plotting._utils.get_color_sequence.md) | Get color sequence for plotting multiple series. |
| [`palette_yohou`](generated/yohou.plotting._utils.palette_yohou.md) | Return the yohou color palette. |
| [`resolve_color_palette`](generated/yohou.plotting._utils.resolve_color_palette.md) | Resolve a user-provided color palette or fall back to the default. |
| [`panel_facet_figure`](generated/yohou.plotting._utils.panel_facet_figure.md) | Create a faceted subplot figure for panel data. |
| [`resolve_panel_columns`](generated/yohou.plotting._utils.resolve_panel_columns.md) | Resolve which panel columns to plot. |
