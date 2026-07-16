---
template: api-submodule.html
---

# yohou.plotting

Time series plotting module for yohou.

### Classes

| Name | Description |
|------|-------------|
| [`LegendTracker`](generated/yohou.plotting.LegendTracker.md) | Track which legend entries have been shown to avoid duplicates. |
| [`PanelColorManager`](generated/yohou.plotting.PanelColorManager.md) | Assign consistent colours to panel members by name. |
| [`RenderContext`](generated/yohou.plotting.RenderContext.md) | Typed context passed to facet render callbacks. |

### Functions

| Name | Description |
|------|-------------|
| [`config_context`](generated/yohou.plotting.config_context.md) | Context manager to temporarily override plotting configuration. |
| [`get_color_sequence`](generated/yohou.plotting.get_color_sequence.md) | Get color sequence for plotting multiple series. |
| [`get_config`](generated/yohou.plotting.get_config.md) | Return a copy of the current global plotting configuration. |
| [`linked_legendgroup_kwargs`](generated/yohou.plotting.linked_legendgroup_kwargs.md) | Build ``legendgroup`` / ``showlegend`` kwargs for linked traces. |
| [`palette_yohou`](generated/yohou.plotting.palette_yohou.md) | Return the yohou color palette. |
| [`resolve_color_palette`](generated/yohou.plotting.resolve_color_palette.md) | Resolve a user-provided color palette or fall back to the default. |
| [`resolve_panel_columns`](generated/yohou.plotting.resolve_panel_columns.md) | Resolve which panel columns to plot. |
| [`set_config`](generated/yohou.plotting.set_config.md) | Set global plotting configuration. |
| [`plot_autocorrelation`](generated/yohou.plotting.plot_autocorrelation.md) | Plot autocorrelation function (ACF) for time series. |
| [`plot_correlation_heatmap`](generated/yohou.plotting.plot_correlation_heatmap.md) | Plot correlation matrix heatmap for multiple time series. |
| [`plot_cross_correlation`](generated/yohou.plotting.plot_cross_correlation.md) | Plot cross-correlation function (CCF) between time series pairs. |
| [`plot_lag_scatter`](generated/yohou.plotting.plot_lag_scatter.md) | Plot scatter plots of y(t) vs y(t-lag) for analysing temporal dependencies. |
| [`plot_partial_autocorrelation`](generated/yohou.plotting.plot_partial_autocorrelation.md) | Plot partial autocorrelation function (PACF) for time series. |
| [`plot_scatter_matrix`](generated/yohou.plotting.plot_scatter_matrix.md) | Plot an N×N scatter-plot matrix. |
| [`plot_seasonal_heatmap`](generated/yohou.plotting.plot_seasonal_heatmap.md) | Plot a 2-D heatmap of aggregated values across two time dimensions. |
| [`plot_seasonality`](generated/yohou.plotting.plot_seasonality.md) | Plot seasonal overlay. |
| [`plot_subseasonality`](generated/yohou.plotting.plot_subseasonality.md) | Plot seasonal subseries. |
| [`plot_calibration`](generated/yohou.plotting.plot_calibration.md) | Plot calibration for interval or class-probability forecasts. |
| [`plot_group_scores`](generated/yohou.plotting.plot_group_scores.md) | Plot scores broken down by panel group. |
| [`plot_residuals`](generated/yohou.plotting.plot_residuals.md) | Plot diagnostic plots for model residuals. |
| [`plot_score_distribution`](generated/yohou.plotting.plot_score_distribution.md) | Plot the distribution of per-timestep scorer values. |
| [`plot_score_heatmap`](generated/yohou.plotting.plot_score_heatmap.md) | Plot a 2D heatmap of scores across two forecast dimensions. |
| [`plot_score_per_step`](generated/yohou.plotting.plot_score_per_step.md) | Plot scorer value by forecast horizon step. |
| [`plot_score_per_vintage`](generated/yohou.plotting.plot_score_per_vintage.md) | Plot scorer value by forecast vintage (observed time). |
| [`plot_score_summary`](generated/yohou.plotting.plot_score_summary.md) | Plot a grouped bar chart comparing aggregate scores across models and scorers. |
| [`plot_score_time_series`](generated/yohou.plotting.plot_score_time_series.md) | Plot scorer values over time for one or more forecasts. |
| [`plot_boxplot`](generated/yohou.plotting.plot_boxplot.md) | Plot boxplots grouped by time periods. |
| [`plot_distribution`](generated/yohou.plotting.plot_distribution.md) | Plot histogram with optional KDE overlay for one or more columns. |
| [`plot_missing_data`](generated/yohou.plotting.plot_missing_data.md) | Visualize missing data patterns over time. |
| [`plot_outliers`](generated/yohou.plotting.plot_outliers.md) | Plot time series with outlier points highlighted. |
| [`plot_resampling_comparison`](generated/yohou.plotting.plot_resampling_comparison.md) | Plot original vs resampled time series for comparison. |
| [`plot_rolling_statistics`](generated/yohou.plotting.plot_rolling_statistics.md) | Plot rolling window statistics (mean, std, min, max, median, quantiles). |
| [`plot_time_series`](generated/yohou.plotting.plot_time_series.md) | Plot basic line plots for one or more time series. |
| [`plot_decomposition`](generated/yohou.plotting.plot_decomposition.md) | Plot time series decomposition as vertically stacked subplots. |
| [`plot_forecast`](generated/yohou.plotting.plot_forecast.md) | Plot forecasts with historical data and optional prediction intervals. |
| [`plot_time_weight`](generated/yohou.plotting.plot_time_weight.md) | Plot time-based weights as a time series visualization. |
| [`plot_cv_results_scatter`](generated/yohou.plotting.plot_cv_results_scatter.md) | Plot hyperparameter search results as a scatter plot. |
| [`plot_nested_splits`](generated/yohou.plotting.plot_nested_splits.md) | Plot a nested cross-validation as a timeline visualization. |
| [`plot_splits`](generated/yohou.plotting.plot_splits.md) | Plot cross-validation splits as a timeline visualization. |
| [`plot_phase`](generated/yohou.plotting.plot_phase.md) | Plot the phase of a time series. |
| [`plot_spectrum`](generated/yohou.plotting.plot_spectrum.md) | Plot periodogram (power spectral density) for frequency domain analysis. |
