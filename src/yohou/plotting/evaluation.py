"""Model evaluation and diagnostic visualization functions."""

import copy
from collections.abc import Callable
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
    LegendTracker,
    RenderContext,
    _create_figure,
    _create_subplots,
    _group_panel_columns,
    _make_hovertemplate,
    _member_name,
    _normalize_y_pred,
    _subplot_spacing,
    apply_default_layout,
    panel_facet_figure,
    resolve_color_palette,
    resolve_panel_columns,
)
from yohou.utils import validate_plotting_data, validate_plotting_params
from yohou.utils.panel import inspect_panel

__all__ = [
    "plot_calibration",
    "plot_model_comparison_bar",
    "plot_residuals",
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
    marker_size: float = 4,
    marker_opacity: float = 0.6,
    n_bins: int = 30,
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
    marker_size : float, default=4
        Marker size for scatter plots.
    marker_opacity : float, default=0.6
        Marker opacity.
    n_bins : int, default=30
        Number of bins for the histogram.

    Returns
    -------
    go.Figure
        Plotly figure with 4 diagnostic subplots.
    """
    residuals = residuals_df[col_name].to_numpy()
    fitted = y_pred[col_name].to_numpy()

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

    # Q-Q Plot (z-scored residuals so both axes share the same scale)
    res_mean = np.nanmean(residuals)
    res_std = np.nanstd(residuals, ddof=1)
    z_residuals = residuals - res_mean if res_std == 0 or np.isnan(res_std) else (residuals - res_mean) / res_std
    sorted_z = np.sort(z_residuals)
    n = len(sorted_z)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))

    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_z,
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
    min_val = min(theoretical_quantiles.min(), sorted_z.min())
    max_val = max(theoretical_quantiles.max(), sorted_z.max())
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
    fig.update_yaxes(title_text="Standardised Quantiles", row=2, col=2)

    fig = apply_default_layout(
        fig,
        title=title or "Residual Diagnostics",
        x_label=None,
        y_label=None,
        width=width,
        height=height,
    )
    fig.update_layout(showlegend=False)

    return fig


def plot_residuals(
    y_pred: pl.DataFrame,
    y_truth: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    panel_group_names: list[str] | None = None,
    facet_by: Literal["group", "member"] | None = "member",
    facet_n_cols: int = 2,
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    resampler: bool | Literal["widget"] | None = None,
    marker_size: float = 4,
    marker_opacity: float = 0.6,
    n_bins: int = 30,
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
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data. ``"group"`` creates one subplot per
        group, ``"member"`` one per member. ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in the faceted grid when multiple target columns
        are resolved.
    color_palette : list[str] | None, default=None
        Custom color palette. If None, uses yohou palette.
    show_legend : bool, default=True
        Whether to show the legend.
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
    resampler : bool | Literal["widget"] | None, default=None
        Enable plotly-resampler for large datasets. ``"figure"`` creates a
        ``FigureResampler``, ``"widget"`` a ``FigureWidgetResampler``.
    marker_size : float, default=4
        Marker size for scatter plots.
    marker_opacity : float, default=0.6
        Marker opacity.
    n_bins : int, default=30
        Number of bins for histogram (single-column diagnostics).

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_residuals

    >>> dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
    >>> y_truth = pl.DataFrame({"time": dates, "y": [100 + i for i in range(91)]})
    >>> y_pred = pl.DataFrame({"time": dates, "y": [100 + i + (i % 3) for i in range(91)]})

    >>> fig = plot_residuals(y_pred, y_truth)
    >>> len(fig.data) > 0
    True

    See Also
    --------
    [`plot_forecast`][yohou.plotting.plot_forecast] : Plot forecasts with historical data.
    """
    validate_plotting_data(y_pred)
    validate_plotting_data(y_truth)
    validate_plotting_params(width=width, height=height)

    # Auto-detect panel data
    _, _panel_groups = inspect_panel(y_truth)
    if panel_group_names is None and columns is None and _panel_groups:
        panel_group_names = []

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
            marker_size=marker_size,
            marker_opacity=marker_opacity,
            n_bins=n_bins,
        )

    # Multiple columns: faceted residuals over time

    if panel_group_names is not None:
        pn_cols = resolve_panel_columns(residuals_df, panel_group_names, columns)
        _, all_members = _group_panel_columns(pn_cols)
        member_palette = resolve_color_palette(color_palette, len(all_members))
        legend_tracker = LegendTracker()

        def _render_residual_scatter(ctx: RenderContext) -> None:
            """Render residuals over time for a single panel column."""
            base = [c for c in ctx.sub_df.columns if c != "time"][0]
            color = member_palette[ctx.entity_idx % len(member_palette)]
            ctx.fig.add_trace(
                go.Scatter(
                    x=ctx.sub_df["time"],
                    y=ctx.sub_df[base],
                    mode="markers",
                    name=ctx.display_name,
                    legendgroup=ctx.display_name,
                    marker={
                        "size": marker_size,
                        "color": color,
                        "opacity": marker_opacity,
                    },
                    showlegend=legend_tracker.should_show(ctx.display_name),
                    hovertemplate=_make_hovertemplate(ctx.display_name, "Time", "Residual", decimals=3),
                ),
                row=ctx.row,
                col=ctx.col,
            )
            ctx.fig.add_hline(
                y=0,
                line={"dash": "dash", "color": "#DC2626", "width": 1},
                row=ctx.row,
                col=ctx.col,
            )

        effective_facet_by = facet_by or "member"
        fig = panel_facet_figure(
            residuals_df,
            _render_residual_scatter,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_by=effective_facet_by,
            facet_n_cols=facet_n_cols,
            title=title or "Residual Diagnostics",
            x_label=x_label or "Time",
            y_label=y_label or "Residuals",
            width=width,
            height=height,
            resampler=resampler,
        )
        fig.update_layout(showlegend=show_legend)
        return fig

    # Non-panel multi-column facets
    n_cols_grid = min(len(target_cols), facet_n_cols)
    n_rows = (len(target_cols) + n_cols_grid - 1) // n_cols_grid
    colors = resolve_color_palette(color_palette, len(target_cols))

    fig = _create_subplots(
        resampler,
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
    fig.update_layout(showlegend=show_legend)

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
    facet_by: Literal["group", "member"] | None = "member",
    facet_n_cols: int = 2,
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    line_width: float = 2.0,
    line_opacity: float = 1.0,
    reference_color: str = "#1e293b",
    reference_width: float = 3.0,
    reference_dash: str = "dash",
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
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data. ``"group"`` creates one subplot per
        group, ``"member"`` one per member. ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in the facet grid when *panel_group_names* is
        used.
    color_palette : list[str] | None, default=None
        Custom color palette as hex codes. If None, uses yohou palette.
    show_legend : bool, default=True
        Whether to show the legend.
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
    line_width : float, default=2.0
        Width of the calibration line.
    line_opacity : float, default=1.0
        Opacity of the calibration line.
    reference_color : str, default="#1e293b"
        Colour of the perfect-calibration reference line.
    reference_width : float, default=3.0
        Width of the reference line.
    reference_dash : str, default="dash"
        Dash style of the reference line.

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
    [`plot_forecast`][yohou.plotting.plot_forecast] : Plot forecast with optional prediction intervals.
    [`plot_residuals`][yohou.plotting.plot_residuals] : Residual diagnostics with panel facets.
    """
    # Validate inputs
    if not isinstance(y_truth, pl.DataFrame):
        msg = f"Expected pl.DataFrame for y_truth, got {type(y_truth).__name__}"
        raise TypeError(msg)
    if not isinstance(y_pred_int, pl.DataFrame):
        msg = f"Expected pl.DataFrame for y_pred_int, got {type(y_pred_int).__name__}"
        raise TypeError(msg)
    validate_plotting_params(width=width, height=height)

    # Auto-detect panel data
    _, _panel_groups = inspect_panel(y_truth)
    if panel_group_names is None and columns is None and _panel_groups:
        panel_group_names = []

    if panel_group_names is not None:
        panel_cols = resolve_panel_columns(y_truth, panel_group_names, columns)

        # Group columns by panel prefix, collect unique members
        groups, all_members = _group_panel_columns(panel_cols)

        n_groups = len(groups)
        n_cols_grid = min(n_groups, facet_n_cols)
        n_rows = (n_groups + n_cols_grid - 1) // n_cols_grid

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols_grid,
            subplot_titles=list(groups.keys()),
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=_subplot_spacing(n_rows),
            horizontal_spacing=0.08,
        )

        palette = resolve_color_palette(color_palette, len(all_members))
        legend_tracker = LegendTracker(show_legend=show_legend)
        for group_idx, (_, group_cols) in enumerate(groups.items()):
            row = group_idx // n_cols_grid + 1
            col_idx = group_idx % n_cols_grid + 1

            for panel_col in group_cols:
                member_name = _member_name(panel_col)
                member_idx = all_members.index(member_name)

                truth_vals = y_truth[panel_col].to_numpy().flatten()
                emp_cov = _compute_empirical_coverages(truth_vals, y_pred_int, panel_col, coverage_rates)

                fig.add_trace(
                    go.Scatter(
                        x=list(coverage_rates),
                        y=emp_cov,
                        mode="lines+markers",
                        name=member_name,
                        line={"color": palette[member_idx], "width": line_width},
                        opacity=line_opacity,
                        showlegend=legend_tracker.should_show(member_name),
                        legendgroup=member_name,
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
                    showlegend=legend_tracker.should_show("Perfect"),
                    legendgroup="perfect",
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


def _plot_score_time_series_panel(
    *,
    scorer_componentwise: BaseScorer,
    y_truth: pl.DataFrame,
    y_pred_dict: dict[str, pl.DataFrame],
    panel_group_names: list[str],
    facet_n_cols: int,
    colors: list[str],
    scorer: BaseScorer,
    show_legend: bool,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    width: int | None,
    height: int | None,
    connect_gaps: bool = False,
    time_weight: Callable | pl.DataFrame | None = None,
    resampler: bool | Literal["widget"] | None = None,
    columns: str | list[str] | None = None,
    line_width: float = 2.0,
    line_dash: str = "solid",
    line_opacity: float = 1.0,
    show_markers: bool = False,
) -> go.Figure:
    """Render faceted per-group score time series.

    Parameters
    ----------
    scorer_componentwise : BaseScorer
        Scorer clone already configured for componentwise aggregation.
    y_truth : pl.DataFrame
        Ground truth values.
    y_pred_dict : dict[str, pl.DataFrame]
        Model name to prediction DataFrame mapping.
    panel_group_names : list[str]
        Panel group prefixes to facet by.
    facet_n_cols : int
        Number of columns in the facet grid.
    colors : list[str]
        Resolved color palette (one per model).
    scorer : BaseScorer
        Original scorer (used for title generation).
    show_legend : bool
        Whether to show legend.
    title : str | None
        Plot title override.
    x_label : str | None
        X-axis label override.
    y_label : str | None
        Y-axis label override.
    width : int | None
        Plot width in pixels.
    height : int | None
        Plot height in pixels.
    time_weight : callable or pl.DataFrame or None, default=None
        Time weighting function or DataFrame forwarded to
        ``scorer.score()``.
    line_width : float, default=2.0
        Width of score lines.
    line_dash : str, default="solid"
        Dash style of score lines.
    line_opacity : float, default=1.0
        Opacity of score lines.
    show_markers : bool, default=False
        Whether to show markers on the lines.

    Returns
    -------
    go.Figure
        Faceted figure with one subplot per panel group.

    """
    mode = "lines+markers" if show_markers else "lines"

    _, all_groups = inspect_panel(y_truth)
    groups = list(all_groups) if not panel_group_names else [g for g in panel_group_names if g in all_groups]
    if not groups:
        msg = f"No panel groups found matching {panel_group_names}. Available: {list(all_groups)}"
        raise ValueError(msg)

    n_groups = len(groups)
    n_cols_grid = min(n_groups, facet_n_cols)
    n_rows = (n_groups + n_cols_grid - 1) // n_cols_grid

    fig = _create_subplots(
        resampler,
        rows=n_rows,
        cols=n_cols_grid,
        subplot_titles=list(groups),
        shared_xaxes=True,
        vertical_spacing=max(0.04, 0.3 / n_rows),
        horizontal_spacing=0.08,
    )

    score_kwargs: dict = {}
    if time_weight is not None:
        score_kwargs["time_weight"] = time_weight

    legend_tracker = LegendTracker(show_legend=show_legend)

    for model_idx, (model_name, y_pred_model) in enumerate(y_pred_dict.items()):
        validate_plotting_data(y_pred_model)
        scores_df = scorer_componentwise.score(y_truth, y_pred_model, **score_kwargs)

        if not isinstance(scores_df, pl.DataFrame):
            msg = f"Scorer must return DataFrame for componentwise aggregation, got {type(scores_df).__name__}"
            raise TypeError(msg)

        if "time" not in scores_df.columns:
            msg = "Scorer must return DataFrame with 'time' column for componentwise aggregation"
            raise ValueError(msg)

        score_cols = [c for c in scores_df.columns if c != "time"]

        for group_idx, group_name in enumerate(groups):
            row = group_idx // n_cols_grid + 1
            col = group_idx % n_cols_grid + 1

            group_score_cols = [c for c in score_cols if c.startswith(f"{group_name}__")]
            if columns is not None:
                col_filter = [columns] if isinstance(columns, str) else list(columns)
                group_score_cols = [c for c in group_score_cols if _member_name(c) in col_filter]
            if not group_score_cols:
                continue

            if len(group_score_cols) == 1:
                score_values = scores_df[group_score_cols[0]]
            else:
                score_values = scores_df.select(group_score_cols).mean_horizontal()

            fig.add_trace(
                go.Scatter(
                    x=scores_df["time"],
                    y=score_values,
                    mode=mode,
                    name=model_name,
                    legendgroup=model_name,
                    showlegend=legend_tracker.should_show(model_name),
                    line={"color": colors[model_idx], "width": line_width, "dash": line_dash},
                    opacity=line_opacity,
                    marker={"size": 6} if show_markers else None,
                    connectgaps=connect_gaps,
                    hovertemplate=_make_hovertemplate(model_name, "Time", "Score", decimals=3, extra=group_name),
                ),
                row=row,
                col=col,
            )

    scorer_name = scorer.__class__.__name__
    default_title = title or f"{scorer_name} Over Time"

    row_height = 300
    default_height = max(row_height * n_rows, 400)

    fig = apply_default_layout(
        fig,
        title=default_title,
        x_label=x_label or "Time",
        y_label=y_label or scorer_name,
        width=width,
        height=height or default_height,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


def plot_score_time_series(
    scorer: BaseScorer,
    y_truth: pl.DataFrame,
    y_pred: pl.DataFrame | dict[str, pl.DataFrame],
    *,
    time_weight: Callable | pl.DataFrame | None = None,
    columns: str | list[str] | None = None,
    panel_group_names: list[str] | None = None,
    facet_by: Literal["group", "member"] | None = "member",
    facet_n_cols: int = 2,
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    connect_gaps: bool = False,
    resampler: bool | Literal["widget"] | None = None,
    line_width: float = 2.0,
    line_dash: str = "solid",
    line_opacity: float = 1.0,
    show_markers: bool = False,
) -> go.Figure:
    """Plot scorer values over time for one or more forecasts.

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
    time_weight : callable or pl.DataFrame or None, default=None
        Time weighting function or DataFrame forwarded to
        ``scorer.score()``.  When provided, per-timestep scores are
        weighted before being plotted.
    columns : str | list[str] | None, default=None
        Target column name(s) to include in the score.  When
        *panel_group_names* is set, acts as a member postfix filter
        (e.g. ``"a"`` selects ``group__a``).  When ``None``, all score
        columns are used.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes for faceted subplots.  When provided, each
        group gets its own subplot showing the score time series for that
        group.  Groups are resolved via ``inspect_panel`` against
        *y_truth*.
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data. ``"group"`` creates one subplot per
        group, ``"member"`` one per member. ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in the facet grid when *panel_group_names* is
        used.
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
    connect_gaps : bool, default=False
        Whether to connect gaps in the data with lines.
    resampler : bool | Literal["widget"] | None, default=None
        Enable plotly-resampler for large datasets.  ``True`` or
        ``"widget"`` creates a ``FigureWidgetResampler``; ``False`` or
        ``None`` uses a plain ``go.Figure``.
    line_width : float, default=2.0
        Width of score lines.
    line_dash : str, default="solid"
        Dash style of score lines.
    line_opacity : float, default=1.0
        Opacity of score lines.
    show_markers : bool, default=False
        Whether to show markers on the lines.

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
    [`plot_residuals`][yohou.plotting.plot_residuals] : Plot residual diagnostics.
    [`plot_forecast`][yohou.plotting.plot_forecast] : Plot forecasts with historical data.

    Notes
    -----
    - Scorer is automatically configured with aggregation_method="componentwise"
    - For interval scorers, use aggregation_method=["componentwise", "coveragewise"]
    - Requires scorer to support componentwise aggregation
    - All scores are computed independently at each timestep

    """
    # Validate ground truth
    validate_plotting_data(y_truth)
    validate_plotting_params(width=width, height=height)

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

    # Get color palette
    if color_palette is None:
        colors = resolve_color_palette(None, len(y_pred_dict))
    else:
        colors = resolve_color_palette(color_palette, len(y_pred_dict))

    # Auto-detect panel data
    _, _panel_groups = inspect_panel(y_truth)
    if panel_group_names is None and _panel_groups:
        panel_group_names = []

    if panel_group_names is not None:
        return _plot_score_time_series_panel(
            scorer_componentwise=scorer_componentwise,
            y_truth=y_truth,
            y_pred_dict=y_pred_dict,
            panel_group_names=panel_group_names,
            facet_n_cols=facet_n_cols,
            colors=colors,
            scorer=scorer,
            show_legend=show_legend,
            title=title,
            x_label=x_label,
            y_label=y_label,
            width=width,
            height=height,
            connect_gaps=connect_gaps,
            time_weight=time_weight,
            resampler=resampler,
            columns=columns,
            line_width=line_width,
            line_dash=line_dash,
            line_opacity=line_opacity,
            show_markers=show_markers,
        )

    # Create figure
    fig = _create_figure(resampler)

    # Compute and plot scores for each model
    score_kwargs: dict = {}
    if time_weight is not None:
        score_kwargs["time_weight"] = time_weight

    for idx, (model_name, y_pred_model) in enumerate(y_pred_dict.items()):
        # Validate prediction DataFrame
        validate_plotting_data(y_pred_model)

        # Compute componentwise scores
        scores_df = scorer_componentwise.score(y_truth, y_pred_model, **score_kwargs)

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

        # Optionally filter to requested columns
        if columns is not None:
            col_filter = [columns] if isinstance(columns, str) else list(columns)
            score_columns = [c for c in score_columns if c in col_filter]
            if not score_columns:
                msg = f"None of the requested columns {col_filter!r} found in scorer output"
                raise ValueError(msg)

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
                connectgaps=connect_gaps,
                hovertemplate=_make_hovertemplate(model_name, "Time", "Score", decimals=3),
            )
        )

    # Apply layout
    if title is None:
        scorer_name = scorer.__class__.__name__
        title = f"{scorer_name} Over Time"

    if x_label is None:
        x_label = "Time"

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

    # Update legend
    fig.update_layout(showlegend=show_legend)

    return fig


def plot_model_comparison_bar(
    results: dict[str, dict[str, float]],
    *,
    group_by: str = "scorer",
    orientation: str = "vertical",
    sort_by: str | None = None,
    ascending: bool = True,
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    bar_width: float = 0.8,
    text_auto: bool = True,
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
    show_legend : bool, default=True
        Whether to show the legend.
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
    bar_width : float, default=0.8
        Width of each bar as a fraction of the group width.
    text_auto : bool, default=True
        Annotate bars with their values.

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
    [`plot_score_time_series`][yohou.plotting.plot_score_time_series] : Per-timestep scorer comparison.
    [`plot_cv_results_scatter`][yohou.plotting.plot_cv_results_scatter] : Cross-validation result scatter.
    """
    if not results:
        msg = "results must be a non-empty dict of model → scorer → score"
        raise ValueError(msg)

    validate_plotting_params(width=width, height=height)

    valid_group_by = {"scorer", "model"}
    if group_by not in valid_group_by:
        msg = f"group_by must be one of {valid_group_by}, got '{group_by}'"
        raise ValueError(msg)

    valid_orientation = {"vertical", "horizontal"}
    if orientation not in valid_orientation:
        msg = f"orientation must be one of {valid_orientation}, got '{orientation}'"
        raise ValueError(msg)

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
    fig.update_layout(showlegend=show_legend)

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
    columns: str | list[str] | None = None,
    panel_group_names: list[str] | None = None,
    facet_by: Literal["group", "member"] | None = "member",
    facet_n_cols: int = 2,
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    bar_opacity: float = 0.6,
    line_width: float = 2.0,
    kde_points: int = 200,
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
    columns : str | list[str] | None, default=None
        Target column name(s) to score.  When *panel_group_names* is set
        this acts as a member postfix filter.  ``None`` uses all columns.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot (faceted layout).
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data. ``"group"`` creates one subplot per
        group, ``"member"`` one per member. ``None`` disables faceting.
        Ignored for non-panel data.
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
    bar_opacity : float, default=0.6
        Opacity of histogram bars.
    line_width : float, default=2.0
        Width of KDE lines.
    kde_points : int, default=200
        Number of points for KDE evaluation.

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
    [`plot_score_time_series`][yohou.plotting.plot_score_time_series] : Score values over time.
    [`plot_score_per_horizon`][yohou.plotting.plot_score_per_horizon] : Score by forecast step.
    """
    from scipy.stats import gaussian_kde  # noqa: PLC0415

    validate_plotting_data(y_truth)

    validate_plotting_params(kind=kind, valid_kinds={"histogram", "kde", "both"}, width=width, height=height)

    y_pred_dict: dict[str, pl.DataFrame] = _normalize_y_pred(y_pred)

    scorer_cw = copy.deepcopy(scorer)
    if isinstance(scorer_cw, BaseIntervalScorer):
        scorer_cw.set_params(aggregation_method=["componentwise", "coveragewise"])
    else:
        scorer_cw.set_params(aggregation_method="componentwise")
    scorer_cw.fit(y_truth)

    n_models = len(y_pred_dict)
    colors = resolve_color_palette(color_palette, n_models)

    def _render(
        fig: go.Figure,
        y_truth_sub: pl.DataFrame,
        y_pred_dict_sub: dict[str, pl.DataFrame],
        _colors: list[str],
        _show_legend: bool = True,
        *,
        row: int | None = None,
        col: int | None = None,
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
                    row=row,
                    col=col,
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
                        row=row,
                        col=col,
                    )

            if show_mean and len(score_vals) > 0:
                mean_val = float(np.mean(score_vals))
                fig.add_vline(
                    x=mean_val,
                    line_dash="dash",
                    line_color=c,
                    line_width=1.5,
                    row=row,
                    col=col,
                )
                if row is None:
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

    # Panel dispatch
    _col_filter: set[str] | None = None
    if columns is not None:
        _col_filter = set([columns] if isinstance(columns, str) else columns)

    _, _panel_groups = inspect_panel(y_truth)
    _effective_groups: list[str] | None = None
    if panel_group_names is not None:
        _effective_groups = panel_group_names
    elif _panel_groups:
        _effective_groups = list(_panel_groups)
    if _effective_groups:
        n_cols_grid = min(len(_effective_groups), facet_n_cols)
        n_rows_grid = (len(_effective_groups) + n_cols_grid - 1) // n_cols_grid
        pfig = make_subplots(
            rows=n_rows_grid,
            cols=n_cols_grid,
            subplot_titles=_effective_groups,
            vertical_spacing=max(0.04, 0.3 / n_rows_grid),
        )
        for g_idx, gname in enumerate(_effective_groups):
            r = g_idx // n_cols_grid + 1
            c_i = g_idx % n_cols_grid + 1
            g_cols_truth = [cn for cn in y_truth.columns if cn == "time" or (cn.startswith(f"{gname}__") and (_col_filter is None or _member_name(cn) in _col_filter))]
            y_truth_g = y_truth.select(g_cols_truth) if len(g_cols_truth) > 1 else y_truth
            y_pred_dict_g: dict[str, pl.DataFrame] = {}
            for mname, y_pred_m in y_pred_dict.items():
                gp_cols = [cn for cn in y_pred_m.columns if cn in ("time", "observed_time") or (cn.startswith(f"{gname}__") and (_col_filter is None or _member_name(cn) in _col_filter))]
                y_pred_dict_g[mname] = y_pred_m.select(gp_cols) if len(gp_cols) > 2 else y_pred_m
            _render(pfig, y_truth_g, y_pred_dict_g, colors, show_legend and g_idx == 0, row=r, col=c_i)
        scorer_name = scorer.__class__.__name__
        pfig = apply_default_layout(
            pfig,
            title=title or f"{scorer_name} Distribution",
            x_label=x_label or scorer_name,
            y_label=y_label or ("Density" if kind in ("kde", "both") else "Count"),
            width=width,
            height=height,
        )
        pfig.update_layout(barmode="overlay" if n_models > 1 else "relative", showlegend=show_legend)
        return pfig

    fig = go.Figure()
    if _col_filter is not None:
        _keep_truth = ["time"] + [c for c in y_truth.columns if c != "time" and c in _col_filter]
        y_truth_filt = y_truth.select(_keep_truth)
        y_pred_dict_filt = {k: v.select([c for c in v.columns if c in ("time", "observed_time") or c in _col_filter]) for k, v in y_pred_dict.items()}
        _render(fig, y_truth_filt, y_pred_dict_filt, colors, _show_legend=show_legend)
    else:
        _render(fig, y_truth, y_pred_dict, colors, _show_legend=show_legend)

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
    columns: str | list[str] | None = None,
    panel_group_names: list[str] | None = None,
    facet_by: Literal["group", "member"] | None = "member",
    facet_n_cols: int = 2,
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    line_width: float = 2.0,
    marker_size: float = 8.0,
    marker_opacity: float = 0.8,
    bar_opacity: float = 0.85,
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
    columns : str | list[str] | None, default=None
        Target column name(s) to score.  When *panel_group_names* is set
        this acts as a member postfix filter.  ``None`` uses all columns.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot (faceted layout).
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data. ``"group"`` creates one subplot per
        group, ``"member"`` one per member. ``None`` disables faceting.
        Ignored for non-panel data.
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
    line_width : float, default=2.0
        Width of score lines.
    marker_size : float, default=8.0
        Marker size for line+marker traces.
    marker_opacity : float, default=0.8
        Opacity of scatter markers.
    bar_opacity : float, default=0.85
        Opacity of bars when ``kind="bar"``.

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
    [`plot_score_time_series`][yohou.plotting.plot_score_time_series] : Score values over time.
    [`plot_score_distribution`][yohou.plotting.plot_score_distribution] : Score distribution histogram/KDE.
    """
    validate_plotting_data(y_truth)

    validate_plotting_params(kind=kind, valid_kinds={"line", "bar"}, width=width, height=height)

    y_pred_dict: dict[str, pl.DataFrame] = _normalize_y_pred(y_pred)

    scorer_cw = copy.deepcopy(scorer)
    if isinstance(scorer_cw, BaseIntervalScorer):
        scorer_cw.set_params(aggregation_method=["componentwise", "coveragewise"])
    else:
        scorer_cw.set_params(aggregation_method="componentwise")
    scorer_cw.fit(y_truth)

    n_models = len(y_pred_dict)
    colors = resolve_color_palette(color_palette, n_models)

    def _render_horizon(
        fig: go.Figure,
        y_truth_sub: pl.DataFrame,
        y_pred_dict_sub: dict[str, pl.DataFrame],
        _colors: list[str],
        _show_legend: bool = True,
        *,
        row: int | None = None,
        col: int | None = None,
    ) -> None:
        """Render per-horizon score traces onto *fig*."""
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
                # average across components at each timestep
                score_vals = scores_df.select(score_cols).mean_horizontal().to_numpy()
                score_vals = score_vals[~np.isnan(score_vals)]

            n_steps = len(score_vals)
            steps = np.arange(1, n_steps + 1)
            c = _colors[idx % len(_colors)]

            if kind == "line":
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=score_vals,
                        mode="lines+markers",
                        name=mname,
                        line={"color": c, "width": line_width},
                        marker={"size": marker_size, "color": c, "opacity": marker_opacity},
                    ),
                    row=row,
                    col=col,
                )
            else:
                fig.add_trace(
                    go.Bar(
                        x=steps,
                        y=score_vals,
                        name=mname,
                        marker_color=c,
                        opacity=bar_opacity,
                    ),
                    row=row,
                    col=col,
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
                        showlegend=_show_legend,
                    ),
                    row=row,
                    col=col,
                )

    _col_filter: set[str] | None = None
    if columns is not None:
        _col_filter = set([columns] if isinstance(columns, str) else columns)

    _, _panel_groups = inspect_panel(y_truth)
    _effective_groups: list[str] | None = None
    if panel_group_names is not None:
        _effective_groups = panel_group_names
    elif _panel_groups:
        _effective_groups = list(_panel_groups)
    if _effective_groups:
        n_cols_grid = min(len(_effective_groups), facet_n_cols)
        n_rows_grid = (len(_effective_groups) + n_cols_grid - 1) // n_cols_grid
        pfig = make_subplots(
            rows=n_rows_grid,
            cols=n_cols_grid,
            subplot_titles=_effective_groups,
            vertical_spacing=max(0.04, 0.3 / n_rows_grid),
        )
        for g_idx, gname in enumerate(_effective_groups):
            r = g_idx // n_cols_grid + 1
            c_i = g_idx % n_cols_grid + 1
            g_cols_truth = [cn for cn in y_truth.columns if cn == "time" or (cn.startswith(f"{gname}__") and (_col_filter is None or _member_name(cn) in _col_filter))]
            y_truth_g = y_truth.select(g_cols_truth) if len(g_cols_truth) > 1 else y_truth
            y_pred_dict_g: dict[str, pl.DataFrame] = {}
            for mname, y_pred_m in y_pred_dict.items():
                gp_cols = [cn for cn in y_pred_m.columns if cn in ("time", "observed_time") or (cn.startswith(f"{gname}__") and (_col_filter is None or _member_name(cn) in _col_filter))]
                y_pred_dict_g[mname] = y_pred_m.select(gp_cols) if len(gp_cols) > 2 else y_pred_m
            _render_horizon(pfig, y_truth_g, y_pred_dict_g, colors, show_legend and g_idx == 0, row=r, col=c_i)
        scorer_name = scorer.__class__.__name__
        pfig = apply_default_layout(
            pfig,
            title=title or f"{scorer_name} by Horizon Step",
            x_label=x_label or "Horizon Step",
            y_label=y_label or scorer_name,
            width=width,
            height=height,
        )
        if kind == "bar" and n_models > 1:
            pfig.update_layout(barmode="group")
        pfig.update_layout(showlegend=show_legend)
        return pfig

    fig = go.Figure()
    if _col_filter is not None:
        _keep_truth = ["time"] + [c for c in y_truth.columns if c != "time" and c in _col_filter]
        y_truth_filt = y_truth.select(_keep_truth)
        y_pred_dict_filt = {k: v.select([c for c in v.columns if c in ("time", "observed_time") or c in _col_filter]) for k, v in y_pred_dict.items()}
        _render_horizon(fig, y_truth_filt, y_pred_dict_filt, colors, show_legend)
    else:
        _render_horizon(fig, y_truth, y_pred_dict, colors, show_legend)

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
