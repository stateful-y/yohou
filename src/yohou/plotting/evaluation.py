"""Model evaluation and diagnostic visualization functions."""

import copy
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from pydantic import StrictFloat
from scipy import stats

from yohou.metrics import BaseIntervalScorer
from yohou.metrics.base import BaseScorer
from yohou.plotting._utils import (
    _normalize_y_pred,
    _panel_col_display,
    apply_default_layout,
    panel_facet_figure,
    resolve_color_palette,
    resolve_panel_columns,
    validate_plotting_params,
)
from yohou.utils import validate_plotting_data

__all__ = [
    "plot_calibration",
    "plot_model_comparison_bar",
    "plot_residual_time_series",
    "plot_score_distribution",
    "plot_score_per_horizon",
    "plot_score_time_series",
]


def _render_residual_diagnostics(
    residuals_df: pl.DataFrame,
    y_pred: pl.DataFrame,
    col_name: str,
    *,
    color_palette: list[str] | None = None,
    title: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """Create 4-panel residual diagnostics for a single column.

    Parameters
    ----------
    residuals_df : pl.DataFrame
        Residuals with ``"time"`` column and *col_name*.
    y_pred : pl.DataFrame
        Predicted values (for fitted-values scatter).
    col_name : str
        Column to diagnose.
    color_palette : list[str] | None, default=None
        Custom colour palette with (up to) 4 entries.
    title : str | None, default=None
        Plot title.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    **kwargs : dict
        ``marker_size``, ``marker_opacity``, ``n_bins``.

    Returns
    -------
    go.Figure
        Plotly figure with 4 diagnostic subplots.
    """
    residuals = residuals_df[col_name].to_numpy()
    fitted = y_pred[col_name].to_numpy()

    marker_size = kwargs.get("marker_size", 4)
    marker_opacity = kwargs.get("marker_opacity", 0.6)
    n_bins = kwargs.get("n_bins", 30)

    colors = resolve_color_palette(color_palette, 4)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Residuals Over Time",
            "Residuals vs Fitted",
            "Histogram of Residuals",
            "Q-Q Plot",
        ),
    )

    # Residuals over time
    fig.add_trace(
        go.Scatter(
            x=residuals_df["time"],
            y=residuals,
            mode="markers",
            marker={"size": marker_size, "color": colors[0], "opacity": marker_opacity},
            name="Residuals",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line={"dash": "dash", "color": "#DC2626", "width": 1}, row=1, col=1)

    # Residuals vs Fitted
    fig.add_trace(
        go.Scatter(
            x=fitted,
            y=residuals,
            mode="markers",
            marker={"size": marker_size, "color": colors[1], "opacity": marker_opacity},
            name="Residuals vs Fitted",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0, line={"dash": "dash", "color": "#DC2626", "width": 1}, row=1, col=2)

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=n_bins,
            marker={"color": colors[2], "opacity": marker_opacity + 0.1},
            name="Histogram",
        ),
        row=2,
        col=1,
    )

    # Q-Q Plot
    sorted_residuals = np.sort(residuals)
    n = len(sorted_residuals)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))

    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_residuals,
            mode="markers",
            marker={
                "size": marker_size,
                "color": colors[3 % len(colors)],
                "opacity": marker_opacity,
            },
            name="Q-Q Plot",
        ),
        row=2,
        col=2,
    )

    # Reference line for Q-Q plot
    min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
    max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line={"dash": "dash", "color": "#DC2626", "width": 1},
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # Axis labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Fitted Values", row=1, col=2)
    fig.update_yaxes(title_text="Residuals", row=1, col=2)
    fig.update_xaxes(title_text="Residuals", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)

    fig = apply_default_layout(
        fig,
        title=title or "Residual Diagnostics",
        x_label=None,
        y_label=None,
        width=width or 900,
        height=height or 600,
    )
    fig.update_layout(showlegend=False)

    return fig


def plot_residual_time_series(
    y_pred: pl.DataFrame,
    y_truth: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    panel_group_names: list[str] | None = None,
    facet_n_cols: int = 2,
    color_palette: list[str] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """Plot diagnostic plots for model residuals.

    When a single column is selected, creates a 4-panel layout with
    residuals over time, residuals vs fitted values, histogram of
    residuals, and Q-Q plot for checking normality assumptions.  When
    multiple columns are resolved (through *columns* or
    *panel_group_names*), produces a faceted layout showing residuals
    over time for each column.

    Residuals are computed internally as ``y_truth - y_pred`` for matching
    non-time columns.

    Parameters
    ----------
    y_pred : pl.DataFrame
        Predicted values with ``"time"`` column.
    y_truth : pl.DataFrame
        Ground-truth values with ``"time"`` column.
    columns : str | list[str] | None, default=None
        Column(s) to compute residuals for.  When *panel_group_names* is
        set, acts as a member postfix filter (e.g. ``["a"]`` selects
        ``y__a``).  When *None*, uses all common non-time columns. A
        single match triggers 4-panel diagnostics, multiple produce facets.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to facet by.
    facet_n_cols : int, default=2
        Number of columns in the faceted grid when multiple target columns
        are resolved.
    color_palette : list[str] | None, default=None
        Custom color palette. If None, uses yohou palette.
    title : str | None, default=None
        Plot title.
    x_label : str | None, default=None
        X-axis label.
    y_label : str | None, default=None
        Y-axis label.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    **kwargs : dict
        Additional keyword arguments forwarded to subplot rendering:

        - ``marker_size`` : float, default=4. Marker size for scatter plots.
        - ``marker_opacity`` : float, default=0.6. Marker opacity.
        - ``n_bins`` : int, default=30. Number of bins for histogram.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_residual_time_series

    >>> dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
    >>> y_truth = pl.DataFrame({"time": dates, "y": [100 + i for i in range(91)]})
    >>> y_pred = pl.DataFrame({"time": dates, "y": [100 + i + (i % 3) for i in range(91)]})

    >>> fig = plot_residual_time_series(y_pred, y_truth)
    >>> len(fig.data) > 0
    True

    See Also
    --------
    plot_forecast : Plot forecasts with historical data.
    """
    validate_plotting_data(y_pred)
    validate_plotting_data(y_truth)

    # Resolve target columns
    if panel_group_names is not None:
        target_cols = validate_plotting_data(y_pred, columns=columns, panel_group_names=panel_group_names)
        missing = [c for c in target_cols if c not in y_truth.columns]
        if missing:
            msg = f"Columns {missing} not found in y_truth"
            raise ValueError(msg)
    elif columns is not None:
        target_cols = [columns] if isinstance(columns, str) else list(columns)
        for col in target_cols:
            if col not in y_pred.columns:
                msg = f"Column '{col}' not found in y_pred"
                raise ValueError(msg)
            if col not in y_truth.columns:
                msg = f"Column '{col}' not found in y_truth"
                raise ValueError(msg)
    else:
        pred_cols = [c for c in y_pred.columns if c != "time"]
        truth_cols = [c for c in y_truth.columns if c != "time"]
        common = [c for c in pred_cols if c in truth_cols]
        if not common:
            msg = "No common non-time columns found between y_pred and y_truth"
            raise ValueError(msg)
        target_cols = common

    # Compute residuals
    residual_exprs = [(pl.col(c) - y_pred[c]).alias(c) for c in target_cols]
    residuals_df = y_truth.select("time", *residual_exprs)

    # Single column: full 4-panel diagnostics
    if len(target_cols) == 1:
        return _render_residual_diagnostics(
            residuals_df,
            y_pred,
            target_cols[0],
            color_palette=color_palette,
            title=title,
            width=width,
            height=height,
            **kwargs,
        )

    # Multiple columns: faceted residuals over time
    marker_size = kwargs.get("marker_size", 4)
    marker_opacity = kwargs.get("marker_opacity", 0.6)

    if panel_group_names is not None:

        def _render_residual_scatter(
            fig: go.Figure,
            sub_df: pl.DataFrame,
            display_name: str,  # noqa: ARG001
            panel_idx: int,  # noqa: ARG001
            row: int,
            col: int,
        ) -> None:
            """Render residuals over time for a single panel column."""
            base = [c for c in sub_df.columns if c != "time"][0]
            fig.add_trace(
                go.Scatter(
                    x=sub_df["time"],
                    y=sub_df[base],
                    mode="markers",
                    marker={
                        "size": marker_size,
                        "color": "#2563EB",
                        "opacity": marker_opacity,
                    },
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_hline(
                y=0,
                line={"dash": "dash", "color": "#DC2626", "width": 1},
                row=row,
                col=col,
            )

        return panel_facet_figure(
            residuals_df,
            _render_residual_scatter,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_n_cols=facet_n_cols,
            title=title or "Residual Diagnostics",
            x_label=x_label or "Time",
            y_label=y_label or "Residuals",
            width=width,
            height=height,
        )

    # Non-panel multi-column facets
    n_cols_grid = min(len(target_cols), facet_n_cols)
    n_rows = (len(target_cols) + n_cols_grid - 1) // n_cols_grid
    colors = resolve_color_palette(color_palette, len(target_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols_grid,
        subplot_titles=target_cols,
        shared_xaxes=True,
        vertical_spacing=max(0.04, 0.3 / n_rows),
        horizontal_spacing=0.08,
    )

    for idx, col_name in enumerate(target_cols):
        row = idx // n_cols_grid + 1
        col_idx = idx % n_cols_grid + 1
        fig.add_trace(
            go.Scatter(
                x=residuals_df["time"],
                y=residuals_df[col_name],
                mode="markers",
                marker={
                    "size": marker_size,
                    "color": colors[idx % len(colors)],
                    "opacity": marker_opacity,
                },
                showlegend=False,
            ),
            row=row,
            col=col_idx,
        )
        fig.add_hline(
            y=0,
            line={"dash": "dash", "color": "#DC2626", "width": 1},
            row=row,
            col=col_idx,
        )

    row_height = 300
    default_height = max(row_height * n_rows, 400)

    fig = apply_default_layout(
        fig,
        title=title or "Residual Diagnostics",
        x_label=x_label or "Time",
        y_label=y_label or "Residuals",
        width=width,
        height=height or default_height,
    )

    return fig


def _compute_empirical_coverages(
    y_truth_col: np.ndarray,
    y_pred_int: pl.DataFrame,
    target_column: str,
    coverage_rates: list[StrictFloat],
) -> list[float]:
    """Compute empirical coverage for a single target column.

    Parameters
    ----------
    y_truth_col : np.ndarray
        Ground-truth values for one target column.
    y_pred_int : pl.DataFrame
        Prediction intervals DataFrame.
    target_column : str
        Column prefix used to find ``{target_column}_lower_{rate}``
        and ``{target_column}_upper_{rate}`` columns.
    coverage_rates : list of float
        Nominal coverage rates.

    Returns
    -------
    list of float
        Empirical coverage for each rate.

    Raises
    ------
    ValueError
        If the expected interval columns are missing.

    """
    empirical: list[float] = []
    for rate in coverage_rates:
        upper_col = f"{target_column}_upper_{rate}"
        lower_col = f"{target_column}_lower_{rate}"
        if upper_col not in y_pred_int.columns or lower_col not in y_pred_int.columns:
            msg = f"Interval columns '{upper_col}' and '{lower_col}' not found in y_pred_int"
            raise ValueError(msg)
        lower_vals = y_pred_int[lower_col].to_numpy().flatten()
        upper_vals = y_pred_int[upper_col].to_numpy().flatten()
        inside = np.logical_and(
            np.greater_equal(y_truth_col, lower_vals),
            np.less_equal(y_truth_col, upper_vals),
        )
        empirical.append(float(np.mean(inside)))
    return empirical


def plot_calibration(
    y_pred_int: pl.DataFrame,
    y_truth: pl.DataFrame,
    coverage_rates: list[StrictFloat],
    *,
    columns: str | list[str] | None = None,
    panel_group_names: list[str] | None = None,
    facet_n_cols: int = 2,
    color_palette: list[str] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """Plot prediction interval calibration.

    Compares empirical coverage against nominal coverage rates to assess
    whether prediction intervals are properly calibrated. A well-calibrated
    model should have points close to the diagonal reference line.

    Parameters
    ----------
    y_pred_int : pl.DataFrame
        Prediction intervals time series with columns named
        ``"{target_column}_upper_{coverage_rate}"`` and
        ``"{target_column}_lower_{coverage_rate}"``.
    y_truth : pl.DataFrame
        Target time series with actual values to compare against intervals.
    coverage_rates : list of float
        List of coverage rates to check calibration for (e.g., [0.9, 0.95]).
    columns : str | list[str] | None, default=None
        Target column name(s).  When *panel_group_names* is set this acts
        as a member postfix filter (e.g. ``"a"`` selects ``group__a``).
        When ``None``, all common non-time columns of *y_truth* are used.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes for faceted subplots.  When provided, each
        resolved panel column gets its own subplot.
    facet_n_cols : int, default=2
        Number of columns in the facet grid when *panel_group_names* is
        used.
    color_palette : list[str] | None, default=None
        Custom color palette as hex codes. If None, uses yohou palette.
    title : str | None, default=None
        Plot title. Defaults to ``"Calibration plot"``.
    x_label : str | None, default=None
        X-axis label. Defaults to ``"Nominal coverage"``.
    y_label : str | None, default=None
        Y-axis label. Defaults to ``"Empirical coverage"``.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    **kwargs : dict
        Additional styling parameters:

        - ``line_width`` : float, default=2.0
        - ``line_opacity`` : float, default=1.0
        - ``reference_color`` : str, default="#1e293b"
        - ``reference_width`` : float, default=3.0
        - ``reference_dash`` : str, default="dash"
        - ``show_legend`` : bool, default=True

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    ValueError
        If the requested column is not found in y_truth or interval columns
        are missing.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> from yohou.plotting import plot_calibration

    >>> # Create sample data
    >>> n = 100
    >>> y_truth = pl.DataFrame({"y": np.random.randn(n)})
    >>> y_pred_int = pl.DataFrame({
    ...     "y_upper_0.9": np.random.randn(n) + 1.65,
    ...     "y_lower_0.9": np.random.randn(n) - 1.65,
    ...     "y_upper_0.95": np.random.randn(n) + 1.96,
    ...     "y_lower_0.95": np.random.randn(n) - 1.96,
    ... })

    >>> # Plot calibration
    >>> fig = plot_calibration(y_pred_int, y_truth, coverage_rates=[0.9, 0.95])
    >>> len(fig.data)
    2

    >>> # Multi-column calibration
    >>> y_truth_mc = pl.DataFrame({"a": np.random.randn(n), "b": np.random.randn(n)})
    >>> y_pred_mc = pl.DataFrame({
    ...     "a_upper_0.9": np.random.randn(n) + 1.65,
    ...     "a_lower_0.9": np.random.randn(n) - 1.65,
    ...     "b_upper_0.9": np.random.randn(n) + 1.65,
    ...     "b_lower_0.9": np.random.randn(n) - 1.65,
    ... })
    >>> fig = plot_calibration(y_pred_mc, y_truth_mc, coverage_rates=[0.9])
    >>> len(fig.data)  # 2 column traces + 1 reference line
    3

    See Also
    --------
    plot_forecast : Plot forecast with optional prediction intervals.
    plot_residual_time_series : Residual diagnostics with panel facets.
    """
    # Styling kwargs
    line_width = kwargs.get("line_width", 2.0)
    line_opacity = kwargs.get("line_opacity", 1.0)
    reference_color = kwargs.get("reference_color", "#1e293b")
    reference_width = kwargs.get("reference_width", 3.0)
    reference_dash = kwargs.get("reference_dash", "dash")
    show_legend = kwargs.get("show_legend", True)

    if panel_group_names is not None:
        panel_cols = resolve_panel_columns(y_truth, panel_group_names, columns)
        n_panels = len(panel_cols)
        n_cols_grid = min(n_panels, facet_n_cols)
        n_rows = (n_panels + n_cols_grid - 1) // n_cols_grid

        subplot_titles = [_panel_col_display(c) for c in panel_cols]
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols_grid,
            subplot_titles=subplot_titles,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=max(0.04, 0.3 / n_rows),
            horizontal_spacing=0.08,
        )

        palette = resolve_color_palette(color_palette, 1)
        for idx, panel_col in enumerate(panel_cols):
            row = idx // n_cols_grid + 1
            col_idx = idx % n_cols_grid + 1

            truth_vals = y_truth[panel_col].to_numpy().flatten()
            emp_cov = _compute_empirical_coverages(truth_vals, y_pred_int, panel_col, coverage_rates)

            fig.add_trace(
                go.Scatter(
                    x=list(coverage_rates),
                    y=emp_cov,
                    mode="lines+markers",
                    name=_panel_col_display(panel_col),
                    line={"color": palette[0], "width": line_width},
                    opacity=line_opacity,
                    showlegend=False,
                    hovertemplate="<b>%{fullData.name}</b><br>Nominal: %{x:.2f}<br>Coverage: %{y:.3f}<extra></extra>",
                ),
                row=row,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=list(coverage_rates),
                    y=list(coverage_rates),
                    mode="lines",
                    name="Perfect",
                    line={"color": reference_color, "width": reference_width, "dash": reference_dash},
                    showlegend=False,
                ),
                row=row,
                col=col_idx,
            )

        row_height = 300
        default_height = max(row_height * n_rows, 400)
        fig = apply_default_layout(
            fig,
            title=title or "Calibration plot",
            x_label=x_label or "Nominal coverage",
            y_label=y_label or "Empirical coverage",
            width=width,
            height=height or default_height,
        )
        fig.update_layout(showlegend=show_legend)
        return fig

    if columns is not None:
        target_columns = [columns] if isinstance(columns, str) else list(columns)
        for col in target_columns:
            if col not in y_truth.columns:
                msg = f"Target column '{col}' not found in y_truth"
                raise ValueError(msg)
    else:
        target_columns = [c for c in y_truth.columns if c not in ("time", "observed_time")]
        if not target_columns:
            msg = "y_truth has no non-time columns"
            raise ValueError(msg)

    palette = resolve_color_palette(color_palette, len(target_columns))

    fig = go.Figure()

    for col_idx, target_column in enumerate(target_columns):
        truth_vals = y_truth[target_column].to_numpy().flatten()
        emp_cov = _compute_empirical_coverages(truth_vals, y_pred_int, target_column, coverage_rates)

        trace_name = target_column if len(target_columns) > 1 else "Empirical coverage"
        fig.add_trace(
            go.Scatter(
                x=list(coverage_rates),
                y=emp_cov,
                mode="lines+markers",
                name=trace_name,
                line={"color": palette[col_idx % len(palette)], "width": line_width},
                opacity=line_opacity,
                hovertemplate=f"<b>{trace_name}</b><br>Nominal: %{{x:.2f}}<br>Coverage: %{{y:.3f}}<extra></extra>",
            )
        )

    # Reference diagonal (always exactly one)
    fig.add_trace(
        go.Scatter(
            x=list(coverage_rates),
            y=list(coverage_rates),
            mode="lines",
            name="Perfect calibration",
            line={"color": reference_color, "width": reference_width, "dash": reference_dash},
            hovertemplate="<b>Perfect</b><br>Coverage: %{x:.2f}<extra></extra>",
        )
    )

    fig = apply_default_layout(
        fig,
        title=title or "Calibration plot",
        x_label=x_label or "Nominal coverage",
        y_label=y_label or "Empirical coverage",
        width=width,
        height=height,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


def plot_score_time_series(
    scorer: BaseScorer,
    y_truth: pl.DataFrame,
    y_pred: pl.DataFrame | dict[str, pl.DataFrame],
    *,
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """
    Plot scorer values over time for one or more forecasts.

    Evaluates forecast quality at each timestep by computing the scorer with
    componentwise aggregation, then plots the resulting score time series.
    Useful for identifying periods where forecast performance varies.

    Parameters
    ----------
    scorer : BaseScorer
        Yohou scorer instance (e.g., MeanAbsoluteError, RootMeanSquaredError).
        Will be cloned and configured with aggregation_method="componentwise".
    y_truth : pl.DataFrame
        Ground truth values with 'time' column.
    y_pred : pl.DataFrame or dict[str, pl.DataFrame]
        Predicted values with 'observed_time' and 'time' columns.
        - If DataFrame: single forecast to plot
        - If dict: multiple forecasts with keys as model names
    color_palette : list[str] | None, default=None
        Custom color palette as hex codes. If None, uses yohou palette.
    show_legend : bool, default=True
        Whether to show legend when plotting multiple forecasts.
    title : str | None, default=None
        Plot title. If None, generates title from scorer name.
    x_label : str | None, default=None
        X-axis label. Defaults to "time".
    y_label : str | None, default=None
        Y-axis label. If None, uses scorer class name.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    **kwargs : dict
        Additional styling parameters:
        - line_width : float, default=2.0
        - line_dash : str, default="solid"
        - line_opacity : float, default=1.0
        - hovermode : str, default="x unified"
        - show_markers : bool, default=False

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    TypeError
        If y_truth or y_pred is not a Polars DataFrame.
    ValueError
        If DataFrames are empty or missing required columns.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MeanAbsoluteError
    >>> from yohou.plotting import plot_score_time_series

    >>> # Create sample data
    >>> y_truth = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [10.0, 20.0, 30.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [12.0, 19.0, 28.0],
    ... })

    >>> # Plot score time series for single forecast
    >>> scorer = MeanAbsoluteError()
    >>> fig = plot_score_time_series(scorer, y_truth, y_pred)
    >>> len(fig.data)
    1

    >>> # Plot multiple forecasts
    >>> y_pred2 = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [11.0, 21.0, 29.0],
    ... })
    >>> fig = plot_score_time_series(scorer, y_truth, {"Model A": y_pred, "Model B": y_pred2})
    >>> len(fig.data)
    2

    See Also
    --------
    plot_residual_time_series : Plot residual diagnostics.
    plot_forecast : Plot forecasts with historical data.

    Notes
    -----
    - Scorer is automatically configured with aggregation_method="componentwise"
    - For interval scorers, use aggregation_method=["componentwise", "coveragewise"]
    - Requires scorer to support componentwise aggregation
    - All scores are computed independently at each timestep

    """
    # Validate ground truth
    validate_plotting_data(y_truth)

    # Normalize y_pred to dict format
    y_pred_dict = _normalize_y_pred(y_pred)

    # Clone scorer and configure for componentwise aggregation
    scorer_componentwise = copy.deepcopy(scorer)

    # Check if scorer is interval scorer (has different aggregation options)
    if isinstance(scorer_componentwise, BaseIntervalScorer):
        # Interval scorer: aggregate across coverage rates too
        scorer_componentwise.set_params(aggregation_method=["componentwise", "coveragewise"])
    else:
        # Point scorer: only componentwise
        scorer_componentwise.set_params(aggregation_method="componentwise")

    # Fit the cloned scorer (required for validation)
    scorer_componentwise.fit(y_truth)

    # Get styling parameters from kwargs
    line_width = kwargs.get("line_width", 2.0)
    line_dash = kwargs.get("line_dash", "solid")
    line_opacity = kwargs.get("line_opacity", 1.0)
    hovermode = kwargs.get("hovermode", "x unified")
    show_markers = kwargs.get("show_markers", False)

    # Get color palette
    if color_palette is None:
        colors = resolve_color_palette(None, len(y_pred_dict))
    else:
        colors = resolve_color_palette(color_palette, len(y_pred_dict))

    # Create figure
    fig = go.Figure()

    # Compute and plot scores for each model
    for idx, (model_name, y_pred_model) in enumerate(y_pred_dict.items()):
        # Validate prediction DataFrame
        validate_plotting_data(y_pred_model)

        # Compute componentwise scores
        scores_df = scorer_componentwise.score(y_truth, y_pred_model)

        # Type narrow: ensure scores_df is a DataFrame
        if not isinstance(scores_df, pl.DataFrame):
            msg = f"Scorer must return DataFrame for componentwise aggregation, got {type(scores_df).__name__}"
            raise TypeError(msg)

        # scores_df should have "time" column and score columns
        if "time" not in scores_df.columns:
            msg = "Scorer must return DataFrame with 'time' column for componentwise aggregation"
            raise ValueError(msg)

        # Get score columns (all except time)
        score_columns = [col for col in scores_df.columns if col != "time"]

        # If multiple score columns, aggregate (mean) for simplicity
        if len(score_columns) == 1:
            score_values = scores_df[score_columns[0]]
        else:
            # Multiple components: compute mean across components
            score_values = scores_df.select(score_columns).mean().transpose().to_series()

        # Determine mode
        mode = "lines+markers" if show_markers else "lines"

        # Add trace
        fig.add_trace(
            go.Scatter(
                x=scores_df["time"],
                y=score_values,
                mode=mode,
                name=model_name,
                line={"color": colors[idx], "width": line_width, "dash": line_dash},
                opacity=line_opacity,
                marker={"size": 6} if show_markers else None,
                hovertemplate=f"<b>{model_name}</b><br>Time: %{{x}}<br>Score: %{{y:.3f}}<extra></extra>",
            )
        )

    # Apply layout
    if title is None:
        scorer_name = scorer.__class__.__name__
        title = f"{scorer_name} Over Time"

    if x_label is None:
        x_label = "time"

    if y_label is None:
        y_label = scorer.__class__.__name__

    fig = apply_default_layout(
        fig,
        title=title,
        x_label=x_label,
        y_label=y_label,
        width=width,
        height=height,
    )

    # Update hovermode and legend
    fig.update_layout(hovermode=hovermode, showlegend=show_legend)

    return fig


def plot_model_comparison_bar(
    results: dict[str, dict[str, float]],
    *,
    group_by: str = "scorer",
    orientation: str = "vertical",
    sort_by: str | None = None,
    ascending: bool = True,
    color_palette: list[str] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """Plot grouped bar chart comparing multiple models across scorers.

    Creates a grouped bar chart with one group per scorer (or per model),
    allowing side-by-side comparison of multiple model performances.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        Mapping of model names to dicts of scorer name to score value.
        Example: ``{"Model A": {"MAE": 0.5, "RMSE": 0.8}, "Model B": {"MAE": 0.3}}``.
    group_by : str, default="scorer"
        Grouping axis. ``"scorer"`` groups bars by scorer with one bar per model.
        ``"model"`` groups bars by model with one bar per scorer.
    orientation : str, default="vertical"
        ``"vertical"`` for vertical bars, ``"horizontal"`` for horizontal bars.
    sort_by : str | None, default=None
        Name of a model or scorer to sort the category axis by. If None,
        categories appear in insertion order.
    ascending : bool, default=True
        Sort direction when ``sort_by`` is set.
    color_palette : list[str] | None, default=None
        Custom color palette. Falls back to ``resolve_color_palette``.
    title : str | None, default=None
        Plot title. Defaults to ``"Model Comparison"``.
    x_label : str | None, default=None
        X-axis label.
    y_label : str | None, default=None
        Y-axis label.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    **kwargs : dict
        Additional styling:
        - bar_width : float, default=0.8
        - text_auto : bool, default=True. Annotate bars with values.

    Returns
    -------
    go.Figure
        Plotly figure with grouped bar chart.

    Raises
    ------
    ValueError
        If *results* is empty or *group_by* / *orientation* is invalid.

    Examples
    --------
    >>> from yohou.plotting import plot_model_comparison_bar

    >>> scores = {
    ...     "Naive": {"MAE": 12.3, "RMSE": 15.1},
    ...     "LinearRegression": {"MAE": 8.7, "RMSE": 10.4},
    ... }
    >>> fig = plot_model_comparison_bar(scores)
    >>> len(fig.data)
    2

    See Also
    --------
    plot_score_time_series : Per-timestep scorer comparison.
    plot_cv_results_scatter : Cross-validation result scatter.
    """
    if not results:
        msg = "results must be a non-empty dict of model → scorer → score"
        raise ValueError(msg)

    valid_group_by = {"scorer", "model"}
    if group_by not in valid_group_by:
        msg = f"group_by must be one of {valid_group_by}, got '{group_by}'"
        raise ValueError(msg)

    valid_orientation = {"vertical", "horizontal"}
    if orientation not in valid_orientation:
        msg = f"orientation must be one of {valid_orientation}, got '{orientation}'"
        raise ValueError(msg)

    bar_width = kwargs.get("bar_width", 0.8)
    text_auto = kwargs.get("text_auto", True)

    # Collect all model and scorer names
    model_names = list(results.keys())
    scorer_names: list[str] = []
    for scores in results.values():
        for s in scores:
            if s not in scorer_names:
                scorer_names.append(s)

    # Determine grouping
    if group_by == "scorer":
        categories = scorer_names
        series_names = model_names
        # Each series = one model, each category = one scorer
        series_values = [[results[model].get(scorer, 0.0) for scorer in categories] for model in series_names]
    else:
        categories = model_names
        series_names = scorer_names
        series_values = [[results[model].get(scorer, 0.0) for model in categories] for scorer in series_names]

    # Optional sorting
    if sort_by is not None:
        # Find the series to sort by
        sort_idx = None
        for i, name in enumerate(series_names):
            if name == sort_by:
                sort_idx = i
                break
        if sort_idx is not None:
            order = sorted(
                range(len(categories)),
                key=lambda k: series_values[sort_idx][k],
                reverse=not ascending,
            )
            categories = [categories[k] for k in order]
            series_values = [[sv[k] for k in order] for sv in series_values]

    colors = resolve_color_palette(color_palette, len(series_names))
    is_horizontal = orientation == "horizontal"

    fig = go.Figure()
    for i, (name, values) in enumerate(zip(series_names, series_values, strict=True)):
        bar_kwargs: dict = {
            "name": name,
            "marker_color": colors[i % len(colors)],
            "width": bar_width / len(series_names),
        }
        if text_auto:
            bar_kwargs["text"] = [f"{v:.3g}" for v in values]
            bar_kwargs["textposition"] = "outside"

        if is_horizontal:
            bar_kwargs["x"] = values
            bar_kwargs["y"] = categories
            bar_kwargs["orientation"] = "h"
        else:
            bar_kwargs["x"] = categories
            bar_kwargs["y"] = values

        fig.add_trace(go.Bar(**bar_kwargs))

    fig.update_layout(barmode="group")

    title_default = title or "Model Comparison"
    if is_horizontal:
        x_label_default = x_label or "Score"
        y_label_default = y_label or ""
    else:
        x_label_default = x_label or ""
        y_label_default = y_label or "Score"

    fig = apply_default_layout(
        fig,
        title=title_default,
        x_label=x_label_default,
        y_label=y_label_default,
        width=width,
        height=height,
    )

    return fig


def plot_score_distribution(
    scorer: BaseScorer,
    y_truth: pl.DataFrame,
    y_pred: pl.DataFrame | dict[str, pl.DataFrame],
    *,
    kind: Literal["histogram", "kde", "both"] = "histogram",
    n_bins: int = 30,
    show_mean: bool = True,
    show_zero: bool = True,
    panel_group_names: list[str] | None = None,
    facet_n_cols: int = 2,
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """Plot the distribution of per-timestep scorer values.

    Evaluates forecast quality at each timestep (using componentwise
    aggregation) and visualises the resulting score distribution as a
    histogram, KDE, or both.  Supports multi-model comparison via
    overlaid distributions.

    Parameters
    ----------
    scorer : BaseScorer
        Yohou scorer instance (e.g., ``MeanAbsoluteError``).  Will be
        cloned and configured with ``aggregation_method="componentwise"``.
    y_truth : pl.DataFrame
        Ground truth values with ``"time"`` column.
    y_pred : pl.DataFrame or dict[str, pl.DataFrame]
        Predicted values with ``"observed_time"`` and ``"time"`` columns.

        - If DataFrame: single forecast.
        - If dict: keys are model names, values are prediction DataFrames.
    kind : str, default="histogram"
        Distribution visualisation style: ``"histogram"``, ``"kde"`` or
        ``"both"``.
    n_bins : int, default=30
        Number of histogram bins (ignored for ``kind="kde"``).
    show_mean : bool, default=True
        Add a vertical line at the mean score.
    show_zero : bool, default=True
        Add a vertical dashed line at zero (useful as a perfect-forecast
        reference for symmetric scorers).
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot (faceted layout).
    facet_n_cols : int, default=2
        Number of columns in the faceted grid.
    color_palette : list[str] | None, default=None
        Custom colour palette.
    show_legend : bool, default=True
        Whether to show the legend.
    title : str | None, default=None
        Plot title.  Defaults to ``"<ScorerName> Distribution"``.
    x_label : str | None, default=None
        X-axis label.  Defaults to the scorer class name.
    y_label : str | None, default=None
        Y-axis label.  Defaults to ``"Count"`` or ``"Density"``
        depending on *kind*.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    **kwargs : dict
        Additional styling parameters:

        - bar_opacity : float, default=0.6
        - line_width : float, default=2.0
        - kde_points : int, default=200

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    TypeError
        If *y_truth* or *y_pred* is not a Polars DataFrame.
    ValueError
        If *kind* is not one of ``"histogram"``, ``"kde"`` or ``"both"``.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MeanAbsoluteError
    >>> from yohou.plotting import plot_score_distribution

    >>> y_truth = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "value": [10.0, 20.0, 30.0, 40.0, 50.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 5,
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "value": [12.0, 19.0, 28.0, 42.0, 48.0],
    ... })

    >>> fig = plot_score_distribution(MeanAbsoluteError(), y_truth, y_pred)
    >>> len(fig.data) >= 1
    True

    See Also
    --------
    plot_score_time_series : Score values over time.
    plot_score_per_horizon : Score by forecast step.
    """
    from scipy.stats import gaussian_kde  # noqa: PLC0415

    validate_plotting_data(y_truth)

    # ── Validate kind ────────────────────────────────────────────────
    validate_plotting_params(kind=kind, valid_kinds={"histogram", "kde", "both"})

    # ── Normalise y_pred to dict ─────────────────────────────────────
    y_pred_dict: dict[str, pl.DataFrame] = _normalize_y_pred(y_pred)

    # ── Clone scorer for componentwise aggregation ───────────────────
    scorer_cw = copy.deepcopy(scorer)
    if isinstance(scorer_cw, BaseIntervalScorer):
        scorer_cw.set_params(aggregation_method=["componentwise", "coveragewise"])
    else:
        scorer_cw.set_params(aggregation_method="componentwise")
    scorer_cw.fit(y_truth)

    # ── Styling kwargs ───────────────────────────────────────────────
    bar_opacity = kwargs.get("bar_opacity", 0.6)
    line_width = kwargs.get("line_width", 2.0)
    kde_points = kwargs.get("kde_points", 200)

    # ── Colours ──────────────────────────────────────────────────────
    n_models = len(y_pred_dict)
    colors = resolve_color_palette(color_palette, n_models)

    # ── Render callback (re-used for panel facets) ───────────────────
    def _render(
        fig: go.Figure,
        y_truth_sub: pl.DataFrame,
        y_pred_dict_sub: dict[str, pl.DataFrame],
        _colors: list[str],
        _show_legend: bool = True,
    ) -> None:
        """Render score distribution traces onto *fig*."""
        for idx, (mname, y_pred_m) in enumerate(y_pred_dict_sub.items()):
            validate_plotting_data(y_pred_m)
            scores_df = scorer_cw.score(y_truth_sub, y_pred_m)
            if not isinstance(scores_df, pl.DataFrame):
                msg_ = f"Scorer must return DataFrame for componentwise aggregation, got {type(scores_df).__name__}"
                raise TypeError(msg_)

            score_cols = [c for c in scores_df.columns if c != "time"]
            if len(score_cols) == 1:
                score_vals = scores_df[score_cols[0]].drop_nulls().to_numpy()
            else:
                score_vals = scores_df.select(score_cols).to_numpy().flatten()
                score_vals = score_vals[~np.isnan(score_vals)]

            c = _colors[idx % len(_colors)]

            if kind in ("histogram", "both"):
                hist_norm = "probability density" if kind == "both" else ""
                fig.add_trace(
                    go.Histogram(
                        x=score_vals,
                        nbinsx=n_bins,
                        marker_color=c,
                        opacity=bar_opacity,
                        name=mname,
                        legendgroup=mname,
                        showlegend=_show_legend if kind != "both" else False,
                        histnorm=hist_norm,
                        hoverinfo="skip",
                    ),
                )

            if kind in ("kde", "both") and len(score_vals) > 1:
                try:
                    kde = gaussian_kde(score_vals)
                except np.linalg.LinAlgError:
                    pass
                else:
                    x_grid = np.linspace(
                        float(score_vals.min()),
                        float(score_vals.max()),
                        kde_points,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_grid,
                            y=kde(x_grid),
                            mode="lines",
                            line={"color": c, "width": line_width},
                            name=mname,
                            legendgroup=mname,
                            showlegend=_show_legend,
                            hoverinfo="skip",
                        ),
                    )

            if show_mean and len(score_vals) > 0:
                mean_val = float(np.mean(score_vals))
                fig.add_shape(
                    type="line",
                    x0=mean_val,
                    x1=mean_val,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line={"color": c, "dash": "dash", "width": 1.5},
                    legendgroup=mname,
                    showlegend=False,
                )
                fig.add_annotation(
                    x=mean_val,
                    y=1.0,
                    yref="paper",
                    text=f"\u03bc={mean_val:.3f}",
                    font={"color": c, "size": 11},
                    showarrow=False,
                    yanchor="bottom",
                )

        if show_zero:
            fig.add_vline(
                x=0.0,
                line_dash="dot",
                line_color="grey",
            )

    # ── Build figure ─────────────────────────────────────────────────
    fig = go.Figure()
    _render(fig, y_truth, y_pred_dict, colors, _show_legend=show_legend)

    # ── Layout ───────────────────────────────────────────────────────
    scorer_name = scorer.__class__.__name__
    default_title = title or f"{scorer_name} Distribution"
    default_x = x_label or scorer_name
    default_y = y_label or ("Density" if kind in ("kde", "both") else "Count")

    fig = apply_default_layout(
        fig,
        title=default_title,
        x_label=default_x,
        y_label=default_y,
        width=width,
        height=height,
    )

    if n_models > 1:
        fig.update_layout(barmode="overlay", showlegend=show_legend)
    else:
        fig.update_layout(showlegend=show_legend)

    return fig


def plot_score_per_horizon(
    scorer: BaseScorer,
    y_truth: pl.DataFrame,
    y_pred: pl.DataFrame | dict[str, pl.DataFrame],
    *,
    kind: Literal["line", "bar"] = "line",
    show_trend: bool = False,
    panel_group_names: list[str] | None = None,
    facet_n_cols: int = 2,
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """Plot scorer value by forecast horizon step.

    For each step *h* in the forecast window, compute the scorer between
    ``y_truth`` and ``y_pred`` at row *h* and plot the result. This
    reveals how forecast accuracy degrades as the horizon increases.

    Parameters
    ----------
    scorer : BaseScorer
        Yohou scorer instance.  Will be cloned with
        ``aggregation_method="componentwise"``.
    y_truth : pl.DataFrame
        Ground truth with ``"time"`` column.
    y_pred : pl.DataFrame or dict[str, pl.DataFrame]
        Predictions with ``"observed_time"`` and ``"time"`` columns.

        - If DataFrame: single forecast.
        - If dict: keys are model names, values are prediction DataFrames.
    kind : str, default="line"
        Plot kind: ``"line"`` or ``"bar"``.
    show_trend : bool, default=False
        Overlay a linear trend line (``np.polyfit`` degree 1).
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot (faceted layout).
    facet_n_cols : int, default=2
        Columns in the faceted grid.
    color_palette : list[str] | None, default=None
        Custom colour palette.
    show_legend : bool, default=True
        Whether to show the legend.
    title : str | None, default=None
        Plot title. Defaults to ``"<ScorerName> by Horizon Step"``.
    x_label : str | None, default=None
        X-axis label. Defaults to ``"Horizon Step"``.
    y_label : str | None, default=None
        Y-axis label. Defaults to the scorer class name.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    **kwargs : dict
        Additional styling parameters:

        - line_width : float, default=2.0
        - marker_size : float, default=8.0
        - bar_opacity : float, default=0.85

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    TypeError
        If *y_truth* or *y_pred* is not a Polars DataFrame.
    ValueError
        If *kind* is not ``"line"`` or ``"bar"``.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MeanAbsoluteError
    >>> from yohou.plotting import plot_score_per_horizon

    >>> y_truth = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "value": [10.0, 20.0, 30.0, 40.0, 50.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 5,
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "value": [12.0, 19.0, 28.0, 42.0, 48.0],
    ... })

    >>> fig = plot_score_per_horizon(MeanAbsoluteError(), y_truth, y_pred)
    >>> len(fig.data) >= 1
    True

    See Also
    --------
    plot_score_time_series : Score values over time.
    plot_score_distribution : Score distribution histogram/KDE.
    """
    validate_plotting_data(y_truth)

    # ── Validate kind ────────────────────────────────────────────────
    validate_plotting_params(kind=kind, valid_kinds={"line", "bar"})

    # ── Normalise y_pred ─────────────────────────────────────────────
    y_pred_dict: dict[str, pl.DataFrame] = _normalize_y_pred(y_pred)

    # ── Clone scorer for componentwise aggregation ───────────────────
    scorer_cw = copy.deepcopy(scorer)
    if isinstance(scorer_cw, BaseIntervalScorer):
        scorer_cw.set_params(aggregation_method=["componentwise", "coveragewise"])
    else:
        scorer_cw.set_params(aggregation_method="componentwise")
    scorer_cw.fit(y_truth)

    # ── Styling kwargs ───────────────────────────────────────────────
    line_width = kwargs.get("line_width", 2.0)
    marker_size = kwargs.get("marker_size", 8.0)
    bar_opacity = kwargs.get("bar_opacity", 0.85)

    # ── Colours ──────────────────────────────────────────────────────
    n_models = len(y_pred_dict)
    colors = resolve_color_palette(color_palette, n_models)

    fig = go.Figure()

    for idx, (mname, y_pred_m) in enumerate(y_pred_dict.items()):
        validate_plotting_data(y_pred_m)
        scores_df = scorer_cw.score(y_truth, y_pred_m)
        if not isinstance(scores_df, pl.DataFrame):
            msg_ = f"Scorer must return DataFrame for componentwise aggregation, got {type(scores_df).__name__}"
            raise TypeError(msg_)

        score_cols = [c for c in scores_df.columns if c != "time"]
        if len(score_cols) == 1:
            score_vals = scores_df[score_cols[0]].drop_nulls().to_numpy()
        else:
            # average across components at each timestep
            score_vals = scores_df.select(score_cols).mean_horizontal().to_numpy()
            score_vals = score_vals[~np.isnan(score_vals)]

        n_steps = len(score_vals)
        steps = np.arange(1, n_steps + 1)
        c = colors[idx % len(colors)]

        if kind == "line":
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=score_vals,
                    mode="lines+markers",
                    name=mname,
                    line={"color": c, "width": line_width},
                    marker={"size": marker_size, "color": c},
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=steps,
                    y=score_vals,
                    name=mname,
                    marker_color=c,
                    opacity=bar_opacity,
                )
            )

        if show_trend and n_steps >= 2:
            coeffs = np.polyfit(steps, score_vals, 1)
            trend_y = np.polyval(coeffs, steps)
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=trend_y,
                    mode="lines",
                    name=f"{mname} trend",
                    line={"color": c, "width": 1.5, "dash": "dash"},
                    showlegend=show_legend,
                )
            )

    # ── Layout ───────────────────────────────────────────────────────
    scorer_name = scorer.__class__.__name__
    default_title = title or f"{scorer_name} by Horizon Step"
    default_x = x_label or "Horizon Step"
    default_y = y_label or scorer_name

    fig = apply_default_layout(
        fig,
        title=default_title,
        x_label=default_x,
        y_label=default_y,
        width=width,
        height=height,
    )

    if kind == "bar" and n_models > 1:
        fig.update_layout(barmode="group")

    fig.update_layout(showlegend=show_legend)

    return fig
