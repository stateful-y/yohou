"""Time series plotting module for yohou."""

from yohou.plotting._utils import palette_yohou
from yohou.plotting.diagnostics import (
    plot_autocorrelation,
    plot_correlation_heatmap,
    plot_cross_correlation,
    plot_lag_scatter,
    plot_partial_autocorrelation,
    plot_scatter_matrix,
    plot_seasonality,
    plot_subseasonality,
)
from yohou.plotting.evaluation import (
    plot_calibration,
    plot_model_comparison_bar,
    plot_residuals,
    plot_score_distribution,
    plot_score_per_horizon,
    plot_score_time_series,
)
from yohou.plotting.exploration import (
    plot_boxplot,
    plot_missing_data,
    plot_rolling_statistics,
    plot_time_series,
)
from yohou.plotting.forecasting import (
    plot_components,
    plot_forecast,
    plot_time_weight,
)
from yohou.plotting.model_selection import plot_cv_results_scatter, plot_splits
from yohou.plotting.signal import plot_phase, plot_spectrum

__all__ = [
    "palette_yohou",
    "plot_autocorrelation",
    "plot_boxplot",
    "plot_calibration",
    "plot_correlation_heatmap",
    "plot_cross_correlation",
    "plot_cv_results_scatter",
    "plot_components",
    "plot_forecast",
    "plot_lag_scatter",
    "plot_missing_data",
    "plot_model_comparison_bar",
    "plot_partial_autocorrelation",
    "plot_phase",
    "plot_residuals",
    "plot_rolling_statistics",
    "plot_scatter_matrix",
    "plot_score_distribution",
    "plot_score_per_horizon",
    "plot_score_time_series",
    "plot_seasonality",
    "plot_spectrum",
    "plot_splits",
    "plot_subseasonality",
    "plot_time_series",
    "plot_time_weight",
]
