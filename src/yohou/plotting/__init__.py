"""Time series plotting module for yohou."""

from yohou.plotting.colors import get_color_sequence, palette_yohou
from yohou.plotting.comparison import plot_comparison, plot_forecast, plot_residuals
from yohou.plotting.config import FacetConfig, LineStyle, MarkerStyle, PlotLayout
from yohou.plotting.diagnostics import (
    plot_autocorrelation,
    plot_correlation_diagnostics,
    plot_partial_autocorrelation,
    plot_seasonality,
)
from yohou.plotting.frequency import plot_lag_scatter, plot_periodogram
from yohou.plotting.quality import plot_missing_data
from yohou.plotting.specialized import plot_calendar_heatmap, plot_cross_correlation
from yohou.plotting.timeseries import (
    plot_boxplot,
    plot_exponential_moving_average,
    plot_prediction_interval,
    plot_rolling_statistics,
    plot_timeseries,
)

__all__ = [
    "palette_yohou",
    "get_color_sequence",
    "LineStyle",
    "MarkerStyle",
    "FacetConfig",
    "PlotLayout",
    "plot_timeseries",
    "plot_rolling_statistics",
    "plot_exponential_moving_average",
    "plot_boxplot",
    "plot_missing_data",
    "plot_autocorrelation",
    "plot_partial_autocorrelation",
    "plot_correlation_diagnostics",
    "plot_seasonality",
    "plot_prediction_interval",
    "plot_lag_scatter",
    "plot_periodogram",
    "plot_cross_correlation",
    "plot_residuals",
    "plot_forecast",
    "plot_comparison",
    "plot_calendar_heatmap",
]
