"""Exploratory time series plotting functions."""

from typing import Literal

import numpy as np
import polars as pl

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    msg = "plotly is required for yohou plotting. Install: pip install yohou[plotting]"
    raise ImportError(msg) from e

from yohou.plotting._utils import (
    LINE_DASH_SEQUENCE,
    LegendTracker,
    PanelColorManager,
    RenderContext,
    _auto_detect_panel,
    _group_panel_columns,
    _make_hovertemplate,
    _member_name,
    _subplot_spacing,
    apply_default_layout,
    build_category_map,
    facet_figure,
    grouped_legend_kwargs,
    linked_legendgroup_kwargs,
    resolve_color_palette,
    resolve_panel_columns,
)
from yohou.preprocessing import RollingStatisticsTransformer
from yohou.utils import validate_plotting_data, validate_plotting_params
from yohou.utils.polars import is_categorical_dtype
from yohou.utils.validation import interval_to_timedelta

__all__ = [
    "plot_boxplot",
    "plot_distribution",
    "plot_missing_data",
    "plot_outliers",
    "plot_resampling_comparison",
    "plot_rolling_statistics",
    "plot_time_series",
]


def plot_time_series(
    df: pl.DataFrame,
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
    connect_gaps: bool = False,
    resampler: bool | Literal["widget"] | None = None,
    line_width: float = 2.0,
    line_dash: str = "solid",
    line_opacity: float = 1.0,
) -> go.Figure:
    """
    Plot basic line plots for one or more time series.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns to plot.
    columns : str | list[str] | None, default=None
        Column(s) to plot. If None, plots all numeric columns except 'time'.
        If str, plots single column. If list, plots multiple columns.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot. Creates separate subplots per group.
        If None and panel data is detected, plots all groups.
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data.  ``"group"`` creates one subplot per
        group, ``"member"`` one per member.  ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in facet grid when using panel groups.
    color_palette : list[str] | None, default=None
        Custom color palette as hex codes. If None, uses yohou palette.
    show_legend : bool, default=True
        Whether to show legend when plotting multiple columns.
    title : str | None, default=None
        Plot title.
    x_label : str | None, default=None
        X-axis label. Defaults to "time".
    y_label : str | None, default=None
        Y-axis label.
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
        Width of the line traces in pixels.
    line_dash : str, default="solid"
        Dash style for lines. One of ``"solid"``, ``"dash"``, ``"dot"``,
        ``"dashdot"``.
    line_opacity : float, default=1.0
        Opacity of the line traces (0.0 to 1.0).

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    TypeError
        If df is not a Polars DataFrame.
    ValueError
        If DataFrame is empty, missing 'time' column, or specified columns don't exist.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_time_series

    >>> # Create sample data
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
    ...     "y": [100, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180, 175],
    ... })

    >>> # Plot single column
    >>> fig = plot_time_series(df, columns="y")
    >>> len(fig.data)
    1

    >>> # Multiple columns
    >>> df = df.with_columns((pl.col("y") * 1.1).alias("y2"))
    >>> fig = plot_time_series(df, columns=["y", "y2"])
    >>> len(fig.data)
    2

    See Also
    --------
    [`plot_rolling_statistics`][yohou.plotting.plot_rolling_statistics] : Plot rolling window statistics.
    """
    # Validate inputs
    validate_plotting_data(df)
    validate_plotting_params(width=width, height=height)

    if panel_group_names is None and columns is None and _auto_detect_panel(df):
        panel_group_names = []

    if panel_group_names is not None and _auto_detect_panel(df, panel_group_names):
        # Pre-compute palette for consistent colouring across the overlaid dimension
        pn_cols = resolve_panel_columns(df, panel_group_names, columns if panel_group_names is not None else None)
        groups, all_members = _group_panel_columns(pn_cols)
        effective_facet_by = facet_by or "member"
        n_overlay = len(list(groups.keys())) if effective_facet_by == "member" else len(all_members)
        color_palette = resolve_color_palette(color_palette, n_overlay)
        legend_tracker = LegendTracker(show_legend=show_legend)

        def _render_ts(ctx: RenderContext) -> None:
            """Render one trace into the faceted figure."""
            color = color_palette[ctx.entity_idx]
            _show = legend_tracker.should_show(ctx.display_name)
            series = ctx.sub_df[ctx.display_name]
            if is_categorical_dtype(series.dtype):
                sorted_cats, cat_to_int = build_category_map(series)
                y_vals = [cat_to_int.get(str(v), -1) for v in series.to_list()]
                ctx.fig.add_trace(
                    go.Scatter(
                        x=ctx.sub_df["time"],
                        y=y_vals,
                        mode="lines+markers",
                        name=ctx.display_name,
                        line={"color": color, "width": line_width, "dash": line_dash, "shape": "hv"},
                        marker={"size": 5},
                        opacity=line_opacity,
                        showlegend=_show,
                        legendgroup=ctx.display_name,
                        text=[str(v) for v in series.to_list()],
                        hovertemplate=f"<b>{ctx.display_name}</b><br>%{{x}}<br>Class: %{{text}}<extra></extra>",
                    ),
                    row=ctx.row,
                    col=ctx.col,
                )
                ctx.fig.update_yaxes(
                    tickvals=list(range(len(sorted_cats))),
                    ticktext=sorted_cats,
                    row=ctx.row,
                    col=ctx.col,
                )
            else:
                ctx.fig.add_trace(
                    go.Scatter(
                        x=ctx.sub_df["time"],
                        y=series,
                        mode="lines",
                        name=ctx.display_name,
                        line={"color": color, "width": line_width, "dash": line_dash},
                        opacity=line_opacity,
                        connectgaps=connect_gaps,
                        showlegend=_show,
                        legendgroup=ctx.display_name,
                        hovertemplate=f"<b>{ctx.display_name}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
                    ),
                    row=ctx.row,
                    col=ctx.col,
                )

        fig = facet_figure(
            df,
            _render_ts,
            panel_group_names=panel_group_names,
            columns=columns if panel_group_names is not None else None,
            facet_by=effective_facet_by,
            facet_n_cols=facet_n_cols,
            title=title,
            x_label=x_label or "Time",
            y_label=y_label,
            width=width,
            height=height,
            resampler=resampler,
        )
        fig.update_layout(showlegend=show_legend)

        return fig

    # Non-panel case: use column-mode facet_figure (one subplot per column)
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"], include_categorical=True)
    _colors = resolve_color_palette(color_palette, len(plot_columns))
    _col_colors = dict(zip(plot_columns, _colors, strict=False))

    # Pre-compute category maps for any categorical columns
    _cat_maps: dict[str, tuple[list[str], dict[str, int]]] = {}
    for _c in plot_columns:
        if is_categorical_dtype(df[_c].dtype):
            _cat_maps[_c] = build_category_map(df[_c])

    def _render_ts(ctx: RenderContext) -> None:
        """Render one trace into a column-faceted subplot."""
        col_name = ctx.display_name
        color = _col_colors[col_name]
        if col_name in _cat_maps:
            sorted_cats, cat_to_int = _cat_maps[col_name]
            series = ctx.sub_df[col_name]
            y_vals = [cat_to_int.get(str(v), -1) for v in series.to_list()]
            ctx.fig.add_trace(
                go.Scatter(
                    x=ctx.sub_df["time"],
                    y=y_vals,
                    mode="lines+markers",
                    name=col_name,
                    line={"color": color, "width": line_width, "dash": line_dash, "shape": "hv"},
                    marker={"size": 5},
                    opacity=line_opacity,
                    text=[str(v) for v in series.to_list()],
                    hovertemplate=f"<b>{col_name}</b><br>%{{x}}<br>Class: %{{text}}<extra></extra>",
                ),
                row=ctx.row,
                col=ctx.col,
            )
            ctx.fig.update_yaxes(
                tickvals=list(range(len(sorted_cats))),
                ticktext=sorted_cats,
                row=ctx.row,
                col=ctx.col,
            )
        else:
            ctx.fig.add_trace(
                go.Scatter(
                    x=ctx.sub_df["time"],
                    y=ctx.sub_df[col_name],
                    mode="lines",
                    name=col_name,
                    line={"color": color, "width": line_width, "dash": line_dash},
                    opacity=line_opacity,
                    connectgaps=connect_gaps,
                    hovertemplate=f"<b>{col_name}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
                ),
                row=ctx.row,
                col=ctx.col,
            )

    fig = facet_figure(
        df,
        _render_ts,
        columns=plot_columns,
        facet_n_cols=facet_n_cols,
        title=title,
        x_label=x_label or "Time",
        y_label=y_label,
        width=width,
        height=height,
        resampler=resampler,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


def plot_rolling_statistics(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    window_size: int | dict[str, int] = 7,
    statistics: str | list[str] = "mean",
    show_original: bool = True,
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
    line_opacity: float = 0.3,
    smooth_width: float = 2.5,
    smooth_opacity: float = 0.8,
) -> go.Figure:
    """
    Plot rolling window statistics (mean, std, min, max, median, quantiles).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns to plot.
    columns : str | list[str] | None, default=None
        Column(s) to compute statistics for. If None, uses all numeric columns except 'time'.
    window_size : int | dict[str, int], default=7
        Size of the rolling window. When a dict is provided, keys are column names
        and values are per-column window sizes.
    statistics : str | list[str], default="mean"
        Statistic(s) to compute. Options: "mean", "std", "min", "max", "median",
        "q25" (25th percentile), "q75" (75th percentile), "sum".
    show_original : bool, default=True
        Whether to show the original series alongside the statistics.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot.
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data.  ``"group"`` creates one subplot per
        group, ``"member"`` one per member.  ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in facet grid.
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
    connect_gaps : bool, default=False
        Whether to connect gaps in the data with lines.
    resampler : bool | Literal["widget"] | None, default=None
        Enable plotly-resampler for large datasets.  ``True`` or
        ``"widget"`` creates a ``FigureWidgetResampler``; ``False`` or
        ``None`` uses a plain ``go.Figure``.
    line_width : float, default=2.0
        Width of the original series line in pixels.
    line_opacity : float, default=0.3
        Opacity of the original series line.
    smooth_width : float, default=2.5
        Width of the rolling statistic lines in pixels.
    smooth_opacity : float, default=0.8
        Opacity of the rolling statistic lines.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_rolling_statistics

    >>> # Create sample data
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
    ...     "y": [100, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180, 175],
    ... })

    >>> # Simple rolling mean
    >>> fig = plot_rolling_statistics(df, window_size=3, statistics="mean")
    >>> len(fig.data)
    2

    >>> # Rolling mean without original
    >>> fig = plot_rolling_statistics(df, window_size=3, statistics="mean", show_original=False)
    >>> len(fig.data)
    1

    See Also
    --------
    [`plot_time_series`][yohou.plotting.plot_time_series] : Plot basic time series.
    """
    # Validate inputs
    validate_plotting_data(df, min_rows=2)
    validate_plotting_params(width=width, height=height)

    if panel_group_names is None and columns is None and _auto_detect_panel(df):
        panel_group_names = []

    if panel_group_names is not None:
        _color_mgr = PanelColorManager(color_palette)
        legend_tracker = LegendTracker(show_legend=show_legend)

        def _render_rolling(ctx: RenderContext) -> None:
            """Render rolling window statistics traces for a single column."""
            base = [c for c in ctx.sub_df.columns if c != "time"][0]
            _stats = [statistics] if isinstance(statistics, str) else list(statistics)
            ws = window_size.get(base, 7) if isinstance(window_size, dict) else window_size
            member_color = _color_mgr.get_color(ctx.display_name)
            if show_original:
                ctx.fig.add_trace(
                    go.Scatter(
                        x=ctx.sub_df["time"],
                        y=ctx.sub_df[base],
                        mode="lines",
                        name=ctx.display_name,
                        legendgroup=ctx.display_name,
                        line={"color": member_color, "width": line_width},
                        opacity=line_opacity,
                        showlegend=False,
                        connectgaps=connect_gaps,
                    ),
                    row=ctx.row,
                    col=ctx.col,
                )
            t = RollingStatisticsTransformer(window_size=ws, statistics=_stats)
            t.fit(ctx.sub_df)
            df_s = t.transform(ctx.sub_df)
            for si, stat in enumerate(_stats):
                scol = f"{base}_{stat}"
                legend_kw = grouped_legend_kwargs(
                    ctx.display_name,
                    stat,
                    legend_tracker,
                    is_first_in_group=si == 0,
                )
                ctx.fig.add_trace(
                    go.Scatter(
                        x=df_s["time"],
                        y=df_s[scol],
                        mode="lines",
                        line={
                            "color": member_color,
                            "width": smooth_width,
                            "dash": LINE_DASH_SEQUENCE[si % len(LINE_DASH_SEQUENCE)],
                        },
                        opacity=smooth_opacity,
                        connectgaps=connect_gaps,
                        **legend_kw,
                    ),
                    row=ctx.row,
                    col=ctx.col,
                )

        effective_facet_by = facet_by or "member"
        fig = facet_figure(
            df,
            _render_rolling,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_by=effective_facet_by,
            facet_n_cols=facet_n_cols,
            title=title or "Rolling Statistics",
            x_label=x_label or "Time",
            y_label=y_label,
            width=width,
            height=height,
            resampler=resampler,
        )
        fig.update_layout(showlegend=show_legend)
        return fig

    # Non-panel case: column-mode facet_figure
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])
    if isinstance(statistics, str):
        statistics = [statistics]
    _colors = resolve_color_palette(color_palette, len(plot_columns))
    _col_colors = dict(zip(plot_columns, _colors, strict=False))

    def _render_rolling(ctx: RenderContext) -> None:
        """Render rolling statistics into a column-faceted subplot."""
        base = ctx.display_name
        col_color = _col_colors[base]
        if show_original:
            ctx.fig.add_trace(
                go.Scatter(
                    x=ctx.sub_df["time"],
                    y=ctx.sub_df[base],
                    mode="lines",
                    name=base,
                    line={"color": col_color, "width": line_width},
                    opacity=0.3,
                    hovertemplate=f"<b>{base}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
                    connectgaps=connect_gaps,
                ),
                row=ctx.row,
                col=ctx.col,
            )
        ws = window_size.get(base, 7) if isinstance(window_size, dict) else window_size
        transformer = RollingStatisticsTransformer(window_size=ws, statistics=statistics)
        transformer.fit(ctx.sub_df)
        df_stats = transformer.transform(ctx.sub_df)
        for s_idx, stat in enumerate(statistics):
            stat_col = f"{base}_{stat}"
            s_dash = LINE_DASH_SEQUENCE[s_idx % len(LINE_DASH_SEQUENCE)] if len(statistics) > 1 else "solid"
            ctx.fig.add_trace(
                go.Scatter(
                    x=df_stats["time"],
                    y=df_stats[stat_col],
                    mode="lines",
                    name=stat,
                    line={"color": col_color, "width": smooth_width, "dash": s_dash},
                    opacity=smooth_opacity,
                    hovertemplate=f"<b>{stat}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
                    connectgaps=connect_gaps,
                ),
                row=ctx.row,
                col=ctx.col,
            )

    fig = facet_figure(
        df,
        _render_rolling,
        columns=plot_columns,
        facet_n_cols=facet_n_cols,
        title=title or "Rolling Statistics",
        x_label=x_label or "Time",
        y_label=y_label,
        width=width,
        height=height,
        resampler=resampler,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


def plot_boxplot(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    period: str = "1mo",
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
    bar_opacity: float = 0.7,
    show_points: bool | str = "outliers",
    marker_size: float = 4.0,
) -> go.Figure:
    """
    Plot boxplots grouped by time periods.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns to plot.
    columns : str | list[str] | None, default=None
        Column(s) to create boxplots for. If None, uses all numeric columns except 'time'.
    period : str, default="1mo"
        Time period for grouping. Polars duration string.
        Options: "1d" (daily), "1w" (weekly), "1mo" (monthly), "1q" (quarterly), "1y" (yearly).
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot.
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data.  ``"group"`` creates one subplot per
        group, ``"member"`` one per member.  ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in facet grid.
    color_palette : list[str] | None, default=None
        Custom color palette for multi-column plots.
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
    bar_opacity : float, default=0.7
        Opacity of the box shapes (0.0 to 1.0).
    show_points : bool | str, default="outliers"
        Which data points to show. One of ``"outliers"``, ``"all"``, or
        ``False`` to hide all points.
    marker_size : float, default=4.0
        Size of the point markers in pixels.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_boxplot

    >>> # Create sample data
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1w", eager=True),
    ...     "y": [100 + i * 2 + (i % 4) * 10 for i in range(53)],
    ... })

    >>> # Monthly boxplots
    >>> fig = plot_boxplot(df, period="1mo")
    >>> len(fig.data) > 0
    True

    See Also
    --------
    [`plot_time_series`][yohou.plotting.plot_time_series] : Plot basic time series.
    """
    # Validate inputs
    validate_plotting_data(df)
    validate_plotting_params(width=width, height=height)

    if panel_group_names is None and columns is None and _auto_detect_panel(df):
        panel_group_names = []

    if panel_group_names is not None:
        _color_mgr = PanelColorManager(color_palette)
        _legend_tracker = LegendTracker(show_legend=show_legend)

        def _render_boxplot(ctx: RenderContext) -> None:
            """Render period-grouped box plots for a single column."""
            base = [c for c in ctx.sub_df.columns if c != "time"][0]
            _c = _color_mgr.get_color(ctx.display_name)
            _ba = bar_opacity
            _sp = show_points
            _ps = marker_size
            df_g = ctx.sub_df.with_columns(pl.col("time").dt.truncate(period).alias("period"))
            periods_list = df_g.select("period").unique().sort("period")["period"].to_list()
            _show = _legend_tracker.should_show(ctx.display_name)
            for p_idx, pv in enumerate(periods_list):
                pd_data = df_g.filter(pl.col("period") == pv)[base]
                bp = "all" if _sp == "all" else ("outliers" if _sp == "outliers" else False)
                ctx.fig.add_trace(
                    go.Box(
                        y=pd_data,
                        x=[str(pv)] * len(pd_data),
                        name=ctx.display_name,
                        marker={"color": _c},
                        opacity=_ba,
                        boxpoints=bp,
                        marker_size=_ps if bp else None,
                        legendgroup=ctx.display_name,
                        showlegend=_show and p_idx == 0,
                    ),
                    row=ctx.row,
                    col=ctx.col,
                )

        effective_facet_by = facet_by or "member"
        fig = facet_figure(
            df,
            _render_boxplot,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_by=effective_facet_by,
            facet_n_cols=facet_n_cols,
            title=title or "Boxplots",
            x_label=x_label or "Period",
            y_label=y_label,
            width=width,
            height=height,
            shared_xaxes=False,
        )
        fig.update_layout(showlegend=show_legend)
        return fig

    # Non-panel case: column-mode facet_figure
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])
    _colors = resolve_color_palette(color_palette, len(plot_columns))
    _col_colors = dict(zip(plot_columns, _colors, strict=False))

    if show_points == "all":
        boxpoints: str | bool = "all"
    elif show_points == "outliers":
        boxpoints = "outliers"
    else:
        boxpoints = False

    df_grouped = df.with_columns([pl.col("time").dt.truncate(period).alias("period")])
    periods = df_grouped.select("period").unique().sort("period")["period"].to_list()
    period_labels = [str(p) for p in periods]

    def _render_boxplot(ctx: RenderContext) -> None:
        """Render boxplots for one column into a subplot."""
        base = ctx.display_name
        col_color = _col_colors[base]
        for p_idx, period_val in enumerate(periods):
            period_data = df_grouped.filter(pl.col("period") == period_val)[base]
            ctx.fig.add_trace(
                go.Box(
                    y=period_data,
                    x=[period_labels[p_idx]] * len(period_data),
                    name=period_labels[p_idx],
                    marker={"color": col_color},
                    opacity=bar_opacity,
                    boxpoints=boxpoints,
                    marker_size=marker_size if boxpoints else None,
                    showlegend=False,
                    hovertemplate=_make_hovertemplate(base, "Period", "Value"),
                ),
                row=ctx.row,
                col=ctx.col,
            )

    fig = facet_figure(
        df,
        _render_boxplot,
        columns=plot_columns,
        facet_n_cols=facet_n_cols,
        title=title or "Boxplots",
        x_label=x_label or "Period",
        y_label=y_label,
        width=width,
        height=height,
        shared_xaxes=False,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


def _panel_heatmap_missing(
    df: pl.DataFrame,
    *,
    kind: Literal["heatmap", "matrix"],
    panel_group_names: list[str],
    columns: str | list[str] | None,
    facet_by: Literal["group", "member"],
    facet_n_cols: int,
    color_missing: str,
    color_present: str,
    time_aggregation: str | None,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    width: int | None,
    height: int | None,
    row_height: int = 300,
) -> go.Figure:
    """Build a faceted heatmap/matrix of missing data for panel columns."""
    panel_cols = resolve_panel_columns(df, panel_group_names, columns)
    groups, all_members = _group_panel_columns(panel_cols)
    all_group_names = list(groups.keys())

    if facet_by == "member":
        facet_keys = all_members
        overlay_keys_per_facet: dict[str, list[str]] = {
            m: [g for g, cols in groups.items() if any(_member_name(c) == m for c in cols)] for m in all_members
        }
    else:
        facet_keys = all_group_names
        overlay_keys_per_facet = {g: [_member_name(c) for c in cols] for g, cols in groups.items()}

    n_facets = len(facet_keys)
    n_cols_grid = min(n_facets, facet_n_cols)
    n_rows = (n_facets + n_cols_grid - 1) // n_cols_grid

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols_grid,
        subplot_titles=facet_keys,
        shared_xaxes=True,
        vertical_spacing=_subplot_spacing(n_rows),
        horizontal_spacing=0.08,
    )

    colorscale = [[0, color_present], [1, color_missing]]

    for facet_idx, facet_key in enumerate(facet_keys):
        row = facet_idx // n_cols_grid + 1
        col_idx = facet_idx % n_cols_grid + 1

        overlay_keys = overlay_keys_per_facet[facet_key]
        y_labels: list[str] = []
        z_data: list[list[int]] = []

        for overlay_key in overlay_keys:
            if facet_by == "member":
                group_name = overlay_key
                member_name = facet_key
                col_name = next(
                    (c for c in groups[group_name] if _member_name(c) == member_name),
                    None,
                )
                y_label_entry = group_name
            else:
                group_name = facet_key
                member_name = overlay_key
                col_name = next(
                    (c for c in groups[group_name] if _member_name(c) == member_name),
                    None,
                )
                y_label_entry = member_name

            if col_name is None:
                continue

            y_labels.append(y_label_entry)

            if time_aggregation:
                df_agg = df.with_columns(
                    pl.col("time").dt.truncate(time_aggregation).alias("period"),
                )
                periods = df_agg.select("period").unique().sort("period")["period"].to_list()
                col_data = []
                for period in periods:
                    period_df = df_agg.filter(pl.col("period") == period)
                    has_missing = period_df[col_name].null_count() > 0
                    col_data.append(1 if has_missing else 0)
                z_data.append(col_data)
                x_vals = [str(p) for p in periods]
            else:
                col_data = df[col_name].is_null().cast(pl.Int8).to_list()
                z_data.append(col_data)
                x_vals = df["time"].to_list()

        text_data = [["Missing" if v == 1 else "Present" for v in row] for row in z_data]

        fig.add_trace(
            go.Heatmap(
                z=z_data,
                x=x_vals,
                y=y_labels,
                colorscale=colorscale,
                showscale=False,
                customdata=text_data,
                hovertemplate="<b>%{y}</b><br>%{x}<br>%{customdata}<extra></extra>",
            ),
            row=row,
            col=col_idx,
        )

    default_height = max(row_height * n_rows, 400)

    fig = apply_default_layout(
        fig,
        title=title or "Missing Data",
        x_label=x_label or "Time",
        y_label=y_label or "Column",
        width=width,
        height=height or default_height,
    )

    return fig


def plot_missing_data(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    kind: Literal["heatmap", "bars", "matrix"] = "heatmap",
    panel_group_names: list[str] | None = None,
    facet_by: Literal["group", "member"] | None = "member",
    facet_n_cols: int = 2,
    color_palette: list[str] | None = None,
    color_missing: str = "#DC2626",
    color_present: str = "#059669",
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    show_percentages: bool = True,
    time_aggregation: str | None = None,
    sampling_interval: str | None = None,
) -> go.Figure:
    """
    Visualize missing data patterns over time.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to check for missing data. If None, checks all columns except 'time'.
    kind : {"heatmap", "bars", "matrix"}, default="heatmap"
        Visualization kind:
        - "heatmap": time x columns grid showing missing/present
        - "bars": bar chart of missing percentage per column
        - "matrix": binary matrix (missingno-style, time on x-axis)
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot.
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data.  ``"group"`` creates one subplot per
        group, ``"member"`` one per member.  ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in facet grid.
    color_palette : list[str] | None, default=None
        Custom colour hex codes used for bars traces. When ``None``
        the default yohou palette is used.  Heatmap/matrix kinds
        always use ``color_missing`` / ``color_present``.
    color_missing : str, default="#DC2626"
        Color for missing values (red).
    color_present : str, default="#059669"
        Color for present values (green).
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
    show_percentages : bool, default=True
        Whether to display percentage labels on bars.
    time_aggregation : str | None, default=None
        Aggregate to time periods before checking. Polars duration string
        (e.g. ``"1d"``, ``"1w"``, ``"1mo"``).
    sampling_interval : str | None, default=None
        Expected sampling frequency of the time series (e.g. ``"1h"``,
        ``"1d"``).  When provided, the DataFrame is reindexed to the full
        expected time range before counting missing values, so that absent
        timestamps (gap rows) are also detected as missing.  Only
        fixed-length intervals (convertible to ``timedelta``) are supported;
        variable-length intervals such as ``"1mo"`` raise ``ValueError``.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_missing_data

    >>> # Create sample data with missing values
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
    ...     "y": [10, None, 30, 40, None, 60, 70, 80, None, 100],
    ... })

    >>> # Bar chart
    >>> fig = plot_missing_data(df, kind="bars")
    >>> len(fig.data) > 0
    True

    See Also
    --------
    [`plot_time_series`][yohou.plotting.plot_time_series] : Plot basic time series.
    """
    # Validate inputs
    validate_plotting_data(df)
    validate_plotting_params(width=width, height=height)

    # Reindex to full time range when sampling_interval is given so that
    # absent timestamps (gap rows) appear as nulls in the analysis.
    if sampling_interval is not None:
        td = interval_to_timedelta(sampling_interval)
        if td is None:
            msg = (
                f"sampling_interval={sampling_interval!r} is a variable-length "
                "interval and cannot be used for reindexing. Use a fixed-length "
                "interval such as '1h' or '1d'."
            )
            raise ValueError(msg)
        full_range = pl.DataFrame({
            "time": pl.datetime_range(
                df["time"].min(),
                df["time"].max(),
                interval=sampling_interval,
                eager=True,
            ),  # ty: ignore[no-matching-overload]
        })
        df = full_range.join(df, on="time", how="left")

    if panel_group_names is None and columns is None and _auto_detect_panel(df):
        panel_group_names = []

    if panel_group_names is not None:
        effective_facet_by = facet_by or "member"

        if kind == "bars":
            tracker = LegendTracker(show_legend)
            color_mgr = PanelColorManager(color_palette)

            def _render_missing(ctx: RenderContext) -> None:
                """Render missing data count bar chart for a single column."""
                base = [c for c in ctx.sub_df.columns if c != "time"][0]
                _sp = show_percentages
                total = len(ctx.sub_df)
                mc = ctx.sub_df[base].null_count()
                pct = (mc / total) * 100 if total > 0 else 0
                text = f"{pct:.1f}%" if _sp else None
                ctx.fig.add_trace(
                    go.Bar(
                        x=[base],
                        y=[pct],
                        name=ctx.display_name,
                        legendgroup=ctx.display_name,
                        showlegend=tracker.should_show(ctx.display_name),
                        marker={"color": color_mgr.get_color(ctx.display_name)},
                        text=[text] if text else None,
                        textposition="auto" if text else None,
                    ),
                    row=ctx.row,
                    col=ctx.col,
                )

            return facet_figure(
                df,
                _render_missing,
                panel_group_names=panel_group_names,
                columns=columns,
                facet_by=effective_facet_by,
                facet_n_cols=facet_n_cols,
                title=title or "Missing Data",
                x_label=x_label or "Column",
                y_label=y_label or "Missing (%)",
                width=width,
                height=height,
                shared_xaxes=False,
            )

        # Panel heatmap / matrix - custom subplot logic
        if kind not in ("heatmap", "matrix"):
            msg = f"Unknown kind: {kind}. Valid options: heatmap, bars, matrix"
            raise ValueError(msg)
        return _panel_heatmap_missing(
            df,
            kind=kind,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_by=effective_facet_by,
            facet_n_cols=facet_n_cols,
            color_missing=color_missing,
            color_present=color_present,
            time_aggregation=time_aggregation,
            title=title,
            x_label=x_label,
            y_label=y_label,
            width=width,
            height=height,
        )

    # Resolve columns
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"], include_categorical=True)

    if kind == "bars":
        # Column-mode facet_figure for bar chart
        _colors = resolve_color_palette(color_palette, len(plot_columns))
        _col_colors = dict(zip(plot_columns, _colors, strict=False))
        total_rows = len(df)

        def _render_missing(ctx: RenderContext) -> None:
            """Render missing-data bar for one column into a subplot."""
            base = ctx.display_name
            col_color = _col_colors[base]
            missing = ctx.sub_df[base].null_count()
            pct = (missing / total_rows) * 100
            text = f"{pct:.1f}%" if show_percentages else None
            ctx.fig.add_trace(
                go.Bar(
                    x=[base],
                    y=[pct],
                    name=base,
                    marker={"color": col_color},
                    text=[text] if text else None,
                    textposition="auto" if text else None,
                    hovertemplate="<b>%{x}</b><br>Missing: %{y:.1f}%<extra></extra>",
                ),
                row=ctx.row,
                col=ctx.col,
            )

        fig = facet_figure(
            df,
            _render_missing,
            columns=plot_columns,
            facet_n_cols=facet_n_cols,
            title=title or "Missing Data",
            x_label=x_label or "Column",
            y_label=y_label or "Missing (%)",
            width=width,
            height=height,
            shared_xaxes=False,
        )
        fig.update_layout(showlegend=show_legend)
        return fig

    elif kind == "heatmap":
        # Heatmap of missing values over time (binary: 1=missing, 0=present)
        if time_aggregation:
            df_agg = df.with_columns([pl.col("time").dt.truncate(time_aggregation).alias("period")])
            # Group by period and check for missing
            periods = df_agg.select("period").unique().sort("period")["period"].to_list()
            z_data = []
            for col in plot_columns:
                col_data = []
                for period in periods:
                    period_df = df_agg.filter(pl.col("period") == period)
                    has_missing = period_df[col].null_count() > 0
                    col_data.append(1 if has_missing else 0)
                z_data.append(col_data)

            x_vals = [str(p) for p in periods]
        else:
            # Use individual time points
            z_data = []
            for col in plot_columns:
                col_data = df[col].is_null().cast(pl.Int8).to_list()
                z_data.append(col_data)
            x_vals = df["time"].to_list()

        fig = go.Figure()
        # Build custom text matrix for hover readability
        text_data = [["Missing" if v == 1 else "Present" for v in row] for row in z_data]
        fig.add_trace(
            go.Heatmap(
                z=z_data,
                x=x_vals,
                y=plot_columns,
                colorscale=[[0, color_present], [1, color_missing]],
                showscale=False,
                customdata=text_data,
                hovertemplate="<b>%{y}</b><br>%{x}<br>%{customdata}<extra></extra>",
            )
        )

        if x_label is None:
            x_label = "Time"
        if y_label is None:
            y_label = "Column"

    elif kind == "matrix":
        # Binary matrix with time on x-axis
        z_data = []
        for col in plot_columns:
            col_data = df[col].is_null().cast(pl.Int8).to_list()
            z_data.append(col_data)

        text_data = [["Missing" if v == 1 else "Present" for v in row] for row in z_data]
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=z_data,
                x=df["time"].to_list(),
                y=plot_columns,
                colorscale=[[0, color_present], [1, color_missing]],
                showscale=False,
                customdata=text_data,
                hovertemplate="<b>%{y}</b><br>%{x}<br>%{customdata}<extra></extra>",
            )
        )

        if x_label is None:
            x_label = "Time"
        if y_label is None:
            y_label = "Column"
    else:
        msg = f"Unknown kind: {kind}. Valid options: heatmap, bars, matrix"
        raise ValueError(msg)

    # Apply layout
    fig = apply_default_layout(
        fig,
        title=title or "Missing Data",
        x_label=x_label,
        y_label=y_label,
        width=width,
        height=height,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


def plot_distribution(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    n_bins: int = 50,
    show_kde: bool = True,
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
    kde_width: float = 2.5,
    kde_points: int = 200,
    histnorm: str = "probability density",
) -> go.Figure:
    """
    Plot histogram with optional KDE overlay for one or more columns.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns to plot.
    columns : str | list[str] | None, default=None
        Column(s) to plot. If None, uses all numeric columns except 'time'.
    n_bins : int, default=50
        Number of histogram bins.
    show_kde : bool, default=True
        Whether to overlay a kernel density estimate curve.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot. Creates separate subplots per group.
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data.  ``"group"`` creates one subplot per
        group, ``"member"`` one per member.  ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in facet grid.
    color_palette : list[str] | None, default=None
        Custom color palette as hex codes.
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
    bar_opacity : float, default=0.6
        Opacity of the histogram bars (0.0 to 1.0).
    kde_width : float, default=2.5
        Width of the KDE line in pixels.
    kde_points : int, default=200
        Number of points used to evaluate the KDE curve.
    histnorm : str, default="probability density"
        Histogram normalization mode (passed to Plotly).

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    TypeError
        If df is not a Polars DataFrame.
    ValueError
        If DataFrame is empty, missing 'time' column, or specified columns don't exist.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_distribution

    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
    ...     "y": [100, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180, 175],
    ... })
    >>> fig = plot_distribution(df, columns="y")
    >>> len(fig.data) > 0
    True

    See Also
    --------
    [`plot_boxplot`][yohou.plotting.plot_boxplot] : Plot boxplots grouped by time periods.
    [`plot_time_series`][yohou.plotting.plot_time_series] : Plot basic time series.
    """
    # Validate inputs
    validate_plotting_data(df, min_rows=2)
    validate_plotting_params(width=width, height=height)

    if panel_group_names is None and columns is None and _auto_detect_panel(df):
        panel_group_names = []

    if panel_group_names is not None:
        _panel_cols = resolve_panel_columns(df, panel_group_names, columns)
        _panel_colors = resolve_color_palette(color_palette, len(_panel_cols))
        _legend_tracker = LegendTracker(show_legend=show_legend)

        def _render_distribution(ctx: RenderContext) -> None:
            """Render histogram + KDE for a single panel column."""
            base = [c for c in ctx.sub_df.columns if c != "time"][0]
            series = ctx.sub_df[base].drop_nulls()
            values = series.to_numpy()
            _c = _panel_colors[ctx.entity_idx]
            ctx.fig.add_trace(
                go.Histogram(
                    x=values,
                    nbinsx=n_bins,
                    marker={"color": _c},
                    opacity=bar_opacity,
                    histnorm=histnorm,
                    name=ctx.display_name,
                    legendgroup=ctx.display_name,
                    showlegend=_legend_tracker.should_show(ctx.display_name),
                    hovertemplate=(
                        f"<b>{ctx.display_name}</b><br>Value: %{{x:.2f}}<br>Density: %{{y:.4f}}<extra></extra>"
                    ),
                ),
                row=ctx.row,
                col=ctx.col,
            )
            if show_kde and len(values) > 1:
                from scipy.stats import gaussian_kde  # noqa: PLC0415

                kde = gaussian_kde(values)
                x_range = np.linspace(float(values.min()), float(values.max()), kde_points)
                ctx.fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=kde(x_range),
                        mode="lines",
                        line={"color": _c, "width": kde_width},
                        name=f"{ctx.display_name} KDE",
                        legendgroup=ctx.display_name,
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{ctx.display_name} KDE</b><br>Value: %{{x:.2f}}<br>Density: %{{y:.4f}}<extra></extra>"
                        ),
                    ),
                    row=ctx.row,
                    col=ctx.col,
                )

        effective_facet_by = facet_by or "member"
        fig = facet_figure(
            df,
            _render_distribution,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_by=effective_facet_by,
            facet_n_cols=facet_n_cols,
            title=title or "Distribution",
            x_label=x_label or "Value",
            y_label=y_label or "Density",
            width=width,
            height=height,
            shared_xaxes=False,
        )
        fig.update_layout(showlegend=show_legend)
        return fig

    # Non-panel case: column-mode facet_figure
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"], include_categorical=True)
    _colors = resolve_color_palette(color_palette, len(plot_columns))
    _col_colors = dict(zip(plot_columns, _colors, strict=False))

    def _render_distribution(ctx: RenderContext) -> None:
        """Render histogram + optional KDE for one column into a subplot."""
        base = ctx.display_name
        col_color = _col_colors[base]
        series = ctx.sub_df[base].drop_nulls()
        if is_categorical_dtype(series.dtype):
            counts = series.value_counts().sort("count", descending=True)
            cat_col = [c for c in counts.columns if c != "count"][0]
            ctx.fig.add_trace(
                go.Bar(
                    x=[str(v) for v in counts[cat_col].to_list()],
                    y=counts["count"].to_list(),
                    marker={"color": col_color},
                    opacity=bar_opacity,
                    name=base,
                    hovertemplate=f"<b>{base}</b><br>%{{x}}<br>Count: %{{y}}<extra></extra>",
                ),
                row=ctx.row,
                col=ctx.col,
            )
        else:
            values = series.to_numpy()
            ctx.fig.add_trace(
                go.Histogram(
                    x=values,
                    nbinsx=n_bins,
                    marker={"color": col_color},
                    opacity=bar_opacity,
                    histnorm=histnorm,
                    name=base,
                    hovertemplate=(f"<b>{base}</b><br>Value: %{{x:.2f}}<br>Density: %{{y:.4f}}<extra></extra>"),
                ),
                row=ctx.row,
                col=ctx.col,
            )
            if show_kde and len(values) > 1:
                from scipy.stats import gaussian_kde  # noqa: PLC0415

                kde = gaussian_kde(values)
                x_range = np.linspace(float(values.min()), float(values.max()), kde_points)
                ctx.fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=kde(x_range),
                        mode="lines",
                        line={"color": col_color, "width": kde_width},
                        name=f"{base} KDE",
                        showlegend=False,
                        hovertemplate=(f"<b>{base} KDE</b><br>Value: %{{x:.2f}}<br>Density: %{{y:.4f}}<extra></extra>"),
                    ),
                    row=ctx.row,
                    col=ctx.col,
                )

    fig = facet_figure(
        df,
        _render_distribution,
        columns=plot_columns,
        facet_n_cols=facet_n_cols,
        title=title or "Distribution",
        x_label=x_label or "Value",
        y_label=y_label or "Density",
        width=width,
        height=height,
        shared_xaxes=False,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


def plot_outliers(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    method: Literal["zscore", "iqr", "percentile"] = "zscore",
    threshold: float = 3.0,
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
    line_opacity: float = 0.7,
    outlier_color: str = "#DC2626",
    outlier_size: float = 8.0,
    outlier_symbol: str = "x",
    show_bounds: bool = True,
) -> go.Figure:
    """
    Plot time series with outlier points highlighted.

    Overlays the original line plot with scatter markers at detected outlier
    positions. The detection method and threshold are configurable.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns to plot.
    columns : str | list[str] | None, default=None
        Column(s) to analyze. If None, uses all numeric columns except 'time'.
    method : {"zscore", "iqr", "percentile"}, default="zscore"
        Outlier detection method:
        - "zscore": points with |z-score| > threshold (default 3.0)
        - "iqr": points outside [Q1 - threshold*IQR, Q3 + threshold*IQR] (default 1.5)
        - "percentile": points above the threshold-th percentile or below
          the (100-threshold)-th percentile (default 95.0, flags outer 5%)
    threshold : float, default=3.0
        Detection threshold. Interpretation depends on *method*.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot.
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data.  ``"group"`` creates one subplot per
        group, ``"member"`` one per member.  ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in facet grid.
    color_palette : list[str] | None, default=None
        Custom color palette for the series lines.
    show_legend : bool, default=True
        Whether to display the legend.
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
    show_legend : bool, default=True
        Whether to display the legend.
    connect_gaps : bool, default=False
        If True, connect lines across missing data gaps.
    resampler : bool | Literal["widget"] | None, default=None
        Enable plotly-resampler for large datasets.  ``True`` returns a
        ``FigureResampler``, ``"widget"`` a ``FigureWidgetResampler``,
        ``None`` reads from `get_config`.
    line_width : float, default=2.0
        Width of the series line in pixels.
    line_opacity : float, default=0.7
        Opacity of the series line.
    outlier_color : str, default="#DC2626"
        Color for outlier markers.
    outlier_size : float, default=8.0
        Size of outlier markers in pixels.
    outlier_symbol : str, default="x"
        Marker symbol for outlier points.
    show_bounds : bool, default=True
        Whether to show threshold boundary lines.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    TypeError
        If df is not a Polars DataFrame.
    ValueError
        If method is unknown or threshold is invalid.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_outliers

    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
    ...     "y": [100, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180, 175],
    ... })
    >>> fig = plot_outliers(df, columns="y", method="zscore")
    >>> len(fig.data) > 0
    True

    See Also
    --------
    [`plot_time_series`][yohou.plotting.plot_time_series] : Plot basic time series.
    [`plot_boxplot`][yohou.plotting.plot_boxplot] : Plot boxplots grouped by time periods.
    """
    # Validate inputs
    validate_plotting_data(df)

    if method not in ("zscore", "iqr", "percentile"):
        msg = f"Unknown method: {method}. Valid options: zscore, iqr, percentile"
        raise ValueError(msg)
    validate_plotting_params(width=width, height=height)

    def _compute_outlier_mask(series: pl.Series) -> tuple[pl.Series, float | None, float | None]:
        """Return (is_outlier_bool_series, lower_bound, upper_bound)."""
        if method == "zscore":
            mean = series.mean()
            std = series.std()
            if std is None or std == 0 or mean is None:
                return pl.Series([False] * len(series)), None, None
            lower = mean - threshold * std  # ty: ignore[unsupported-operator]
            upper = mean + threshold * std  # ty: ignore[unsupported-operator]
            mask = ((series - mean).abs() / std) > threshold
        elif method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            if q1 is None or q3 is None:
                return pl.Series([False] * len(series)), None, None
            iqr = q3 - q1  # ty: ignore[unsupported-operator]
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            mask = (series < lower) | (series > upper)
        else:  # percentile
            lower = series.quantile(1 - threshold / 100)
            upper = series.quantile(threshold / 100)
            if lower is None or upper is None:
                return pl.Series([False] * len(series)), None, None
            mask = (series < lower) | (series > upper)
        return mask.fill_null(False), lower, upper  # ty: ignore[invalid-return-type]

    if panel_group_names is None and columns is None and _auto_detect_panel(df):
        panel_group_names = []

    if panel_group_names is not None:
        _color_mgr = PanelColorManager(color_palette)
        _legend_tracker = LegendTracker(show_legend=show_legend)

        def _render_outlier(ctx: RenderContext) -> None:
            """Render time series with outlier highlights for a single panel."""
            base = [c for c in ctx.sub_df.columns if c != "time"][0]
            _c = _color_mgr.get_color(ctx.display_name)
            _lg = linked_legendgroup_kwargs(ctx.display_name, _legend_tracker, is_primary=True)
            # Line trace
            ctx.fig.add_trace(
                go.Scatter(
                    x=ctx.sub_df["time"],
                    y=ctx.sub_df[base],
                    mode="lines",
                    line={"color": _c, "width": line_width},
                    opacity=line_opacity,
                    connectgaps=connect_gaps,
                    hovertemplate=(f"<b>{ctx.display_name}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>"),
                    **_lg,
                ),
                row=ctx.row,
                col=ctx.col,
            )
            # Outlier markers
            mask, lower, upper = _compute_outlier_mask(ctx.sub_df[base])
            df_out = ctx.sub_df.filter(mask)
            _lg_sec = linked_legendgroup_kwargs(ctx.display_name, _legend_tracker, is_primary=False)
            if len(df_out) > 0:
                ctx.fig.add_trace(
                    go.Scatter(
                        x=df_out["time"],
                        y=df_out[base],
                        mode="markers",
                        marker={"color": _c, "size": outlier_size, "symbol": outlier_symbol},
                        hovertemplate=(f"<b>{base} OUTLIER</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>"),
                        **_lg_sec,
                    ),
                    row=ctx.row,
                    col=ctx.col,
                )
            # Threshold bounds
            if show_bounds and lower is not None and upper is not None:
                for val in (lower, upper):
                    ctx.fig.add_trace(
                        go.Scatter(
                            x=[ctx.sub_df["time"].min(), ctx.sub_df["time"].max()],
                            y=[val, val],
                            mode="lines",
                            line={"dash": "dash", "color": _c, "width": 1},
                            hoverinfo="skip",
                            **_lg_sec,
                        ),
                        row=ctx.row,
                        col=ctx.col,
                    )

        effective_facet_by = facet_by or "member"
        fig = facet_figure(
            df,
            _render_outlier,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_by=effective_facet_by,
            facet_n_cols=facet_n_cols,
            title=title or "Outlier Detection",
            x_label=x_label or "Time",
            y_label=y_label,
            width=width,
            height=height,
            resampler=resampler,
        )
        fig.update_layout(showlegend=show_legend)
        return fig

    # Non-panel case: column-mode facet_figure
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])
    _colors = resolve_color_palette(color_palette, len(plot_columns))
    _col_colors = dict(zip(plot_columns, _colors, strict=False))

    def _render_outlier(ctx: RenderContext) -> None:
        """Render outlier detection for one column into a subplot."""
        base = ctx.display_name
        col_color = _col_colors[base]
        series = ctx.sub_df[base]
        mask, lower, upper = _compute_outlier_mask(series)

        ctx.fig.add_trace(
            go.Scatter(
                x=ctx.sub_df["time"],
                y=series,
                mode="lines",
                name=base,
                line={"color": col_color, "width": line_width},
                opacity=line_opacity,
                connectgaps=connect_gaps,
                hovertemplate=f"<b>{base}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
            ),
            row=ctx.row,
            col=ctx.col,
        )

        df_outliers = df.filter(mask)
        if len(df_outliers) > 0:
            ctx.fig.add_trace(
                go.Scatter(
                    x=df_outliers["time"],
                    y=df_outliers[base],
                    mode="markers",
                    showlegend=False,
                    marker={"color": col_color, "size": outlier_size, "symbol": outlier_symbol},
                    hovertemplate=f"<b>{base} OUTLIER</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
                ),
                row=ctx.row,
                col=ctx.col,
            )

        if show_bounds and lower is not None and upper is not None:
            time_min = ctx.sub_df["time"].min()
            time_max = ctx.sub_df["time"].max()
            for bound_val in (lower, upper):
                ctx.fig.add_trace(
                    go.Scatter(
                        x=[time_min, time_max],
                        y=[bound_val, bound_val],
                        mode="lines",
                        showlegend=False,
                        line={"dash": "dash", "color": col_color, "width": 1},
                        hoverinfo="skip",
                    ),
                    row=ctx.row,
                    col=ctx.col,
                )

    fig = facet_figure(
        df,
        _render_outlier,
        columns=plot_columns,
        facet_n_cols=facet_n_cols,
        title=title or "Outlier Detection",
        x_label=x_label or "Time",
        y_label=y_label,
        width=width,
        height=height,
        resampler=resampler,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


def plot_resampling_comparison(
    df_original: pl.DataFrame,
    df_resampled: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    original_label: str = "Original",
    resampled_label: str = "Resampled",
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
    original_line_width: float = 1.0,
    original_line_opacity: float = 0.4,
    original_line_dash: str = "solid",
    resampled_line_width: float = 2.5,
    resampled_line_opacity: float = 1.0,
    resampled_line_dash: str = "solid",
) -> go.Figure:
    """
    Plot original vs resampled time series for comparison.

    Overlays two versions of the same series at different temporal
    resolutions to visually assess information loss from aggregation.

    Parameters
    ----------
    df_original : pl.DataFrame
        Original (higher-frequency) DataFrame with 'time' column.
    df_resampled : pl.DataFrame
        Resampled (lower-frequency) DataFrame with 'time' column.
    columns : str | list[str] | None, default=None
        Column(s) to compare. If None, uses all numeric columns except 'time'
        from df_resampled.
    original_label : str, default="Original"
        Legend label for the original series.
    resampled_label : str, default="Resampled"
        Legend label for the resampled series.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot.
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data.  ``"group"`` creates one subplot per
        group, ``"member"`` one per member.  ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in facet grid.
    color_palette : list[str] | None, default=None
        Custom color palette.
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
    connect_gaps : bool, default=False
        Whether to connect gaps in the data with lines.
    resampler : bool | Literal["widget"] | None, default=None
        Enable plotly-resampler for large datasets.  ``True`` or
        ``"widget"`` creates a ``FigureWidgetResampler``; ``False`` or
        ``None`` uses a plain ``go.Figure``.
    original_line_width : float, default=1.0
        Width of the original series line in pixels.
    original_line_opacity : float, default=0.4
        Opacity of the original series line.
    original_line_dash : str, default="solid"
        Dash style for the original series line.
    resampled_line_width : float, default=2.5
        Width of the resampled series line in pixels.
    resampled_line_opacity : float, default=1.0
        Opacity of the resampled series line.
    resampled_line_dash : str, default="solid"
        Dash style for the resampled series line.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    TypeError
        If either DataFrame is not a Polars DataFrame.
    ValueError
        If columns don't exist in both DataFrames.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_resampling_comparison

    >>> hourly = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         pl.datetime(2020, 1, 1),
    ...         pl.datetime(2020, 1, 2, 23),
    ...         "1h",
    ...         eager=True,
    ...     ),
    ...     "y": list(range(48)),
    ... })
    >>> daily = hourly.group_by_dynamic("time", every="1d").agg(pl.col("y").mean())
    >>> fig = plot_resampling_comparison(hourly, daily, columns="y")
    >>> len(fig.data)
    2

    See Also
    --------
    [`plot_time_series`][yohou.plotting.plot_time_series] : Plot basic time series.
    [`plot_rolling_statistics`][yohou.plotting.plot_rolling_statistics] : Plot rolling window statistics.
    """
    # Validate both DataFrames
    validate_plotting_data(df_original)
    validate_plotting_data(df_resampled)
    validate_plotting_params(width=width, height=height)

    # Get styling parameters
    original_width = original_line_width
    original_opacity = original_line_opacity
    original_dash = original_line_dash
    resampled_width = resampled_line_width
    resampled_opacity = resampled_line_opacity
    resampled_dash = resampled_line_dash

    if panel_group_names is None and columns is None and _auto_detect_panel(df_resampled):
        panel_group_names = []

    if panel_group_names is not None:

        def _render_resampling(ctx: RenderContext) -> None:
            """Render original and resampled traces for a single panel."""
            base = [c for c in ctx.sub_df.columns if c != "time"][0]
            _colors = resolve_color_palette(color_palette, 1)
            # Original (from df_original, matching the same full column name)
            full_col = [c for c in df_original.columns if c.endswith(f"__{base}") or c == base]
            if full_col:
                orig_col = full_col[0]
                ctx.fig.add_trace(
                    go.Scatter(
                        x=df_original["time"],
                        y=df_original[orig_col],
                        mode="lines",
                        line={"color": _colors[0], "width": original_width, "dash": original_dash},
                        opacity=original_opacity,
                        name=original_label,
                        showlegend=False,
                    ),
                    row=ctx.row,
                    col=ctx.col,
                )
            # Resampled
            ctx.fig.add_trace(
                go.Scatter(
                    x=ctx.sub_df["time"],
                    y=ctx.sub_df[base],
                    mode="lines+markers",
                    line={"color": _colors[0], "width": resampled_width, "dash": resampled_dash},
                    opacity=resampled_opacity,
                    name=resampled_label,
                    showlegend=False,
                ),
                row=ctx.row,
                col=ctx.col,
            )

        effective_facet_by = facet_by or "member"
        fig = facet_figure(
            df_resampled,
            _render_resampling,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_by=effective_facet_by,
            facet_n_cols=facet_n_cols,
            title=title or f"{original_label} vs {resampled_label}",
            x_label=x_label or "Time",
            y_label=y_label,
            width=width,
            height=height,
            resampler=resampler,
        )
        fig.update_layout(showlegend=show_legend)
        return fig

    # Non-panel case: column-mode facet_figure
    plot_columns = validate_plotting_data(df_resampled, columns=columns, exclude=["time"])
    for col in plot_columns:
        if col not in df_original.columns:
            msg = f"Column '{col}' not found in original DataFrame. Available: {df_original.columns}"
            raise ValueError(msg)

    _colors = resolve_color_palette(color_palette, len(plot_columns))
    _col_colors = dict(zip(plot_columns, _colors, strict=False))

    def _render_resampling(ctx: RenderContext) -> None:
        """Render original vs resampled for one column into a subplot."""
        base = ctx.display_name
        col_color = _col_colors[base]
        ctx.fig.add_trace(
            go.Scatter(
                x=df_original["time"],
                y=df_original[base],
                mode="lines",
                name=f"{base} ({original_label})",
                line={"color": col_color, "width": original_width, "dash": original_dash},
                opacity=original_opacity,
                connectgaps=connect_gaps,
                hovertemplate=f"<b>{base} ({original_label})</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
            ),
            row=ctx.row,
            col=ctx.col,
        )
        ctx.fig.add_trace(
            go.Scatter(
                x=ctx.sub_df["time"],
                y=ctx.sub_df[base],
                mode="lines+markers",
                name=f"{base} ({resampled_label})",
                line={"color": col_color, "width": resampled_width, "dash": resampled_dash},
                opacity=resampled_opacity,
                connectgaps=connect_gaps,
                hovertemplate=f"<b>{base} ({resampled_label})</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
            ),
            row=ctx.row,
            col=ctx.col,
        )

    fig = facet_figure(
        df_resampled,
        _render_resampling,
        columns=plot_columns,
        facet_n_cols=facet_n_cols,
        title=title or f"{original_label} vs {resampled_label}",
        x_label=x_label or "Time",
        y_label=y_label,
        width=width,
        height=height,
        resampler=resampler,
    )
    fig.update_layout(showlegend=show_legend)

    return fig
