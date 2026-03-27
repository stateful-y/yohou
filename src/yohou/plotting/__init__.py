"""Time series plotting module for yohou."""

from plotly_resampler.aggregation.aggregators import (
    LTTB,
    EveryNthPoint,
    MinMaxAggregator,
    MinMaxLTTB,
)
from plotly_resampler.aggregation.gap_handlers import MedDiffGapHandler, NoGapHandler

from yohou.plotting._utils import (
    LINE_DASH_SEQUENCE,
    PanelColorManager,
    config_context,
    get_color_sequence,
    get_config,
    linked_legendgroup_kwargs,
    palette_yohou,
    resolve_color_palette,
    resolve_panel_columns,
    set_config,
)
from yohou.plotting.diagnostics import (
    plot_autocorrelation,
    plot_correlation_heatmap,
    plot_cross_correlation,
    plot_lag_scatter,
    plot_partial_autocorrelation,
    plot_scatter_matrix,
    plot_seasonal_heatmap,
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
    plot_distribution,
    plot_missing_data,
    plot_outliers,
    plot_resampling_comparison,
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
    "EveryNthPoint",
    "LTTB",
    "LINE_DASH_SEQUENCE",
    "MedDiffGapHandler",
    "MinMaxAggregator",
    "MinMaxLTTB",
    "NoGapHandler",
    "PanelColorManager",
    "config_context",
    "get_color_sequence",
    "get_config",
    "linked_legendgroup_kwargs",
    "palette_yohou",
    "resolve_color_palette",
    "resolve_panel_columns",
    "plot_autocorrelation",
    "plot_boxplot",
    "plot_calibration",
    "plot_correlation_heatmap",
    "plot_cross_correlation",
    "plot_cv_results_scatter",
    "plot_components",
    "plot_distribution",
    "plot_forecast",
    "plot_lag_scatter",
    "plot_missing_data",
    "plot_model_comparison_bar",
    "plot_outliers",
    "plot_partial_autocorrelation",
    "plot_phase",
    "plot_resampling_comparison",
    "plot_residuals",
    "plot_rolling_statistics",
    "plot_scatter_matrix",
    "plot_score_distribution",
    "plot_score_per_horizon",
    "plot_score_time_series",
    "plot_seasonal_heatmap",
    "plot_seasonality",
    "plot_spectrum",
    "plot_splits",
    "plot_subseasonality",
    "plot_time_series",
    "plot_time_weight",
    "set_config",
]
