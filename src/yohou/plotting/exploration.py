"""Exploratory time series plotting functions."""

from typing import Literal

import numpy as np
import plotly.graph_objects as go
import polars as pl

from yohou.plotting._utils import (
    _create_figure,
    _group_panel_columns,
    apply_default_layout,
    panel_facet_figure,
    resolve_color_palette,
    resolve_panel_columns,
)
from yohou.preprocessing import RollingStatisticsTransformer
from yohou.utils import inspect_panel, validate_plotting_data

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
    facet_n_cols: int = 2,
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    resampler: bool | Literal["widget"] | None = None,
    **kwargs,
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
    **kwargs : dict
        Additional styling parameters:
        - line_width : float, default=2.0
        - line_color : str | None, default=None (uses palette if None)
        - line_dash : str, default="solid" ("solid", "dash", "dot", "dashdot")
        - line_opacity : float, default=1.0
        - hovermode : str, default="closest"

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
    `plot_rolling_statistics` : Plot rolling window statistics.
    """
    # Validate inputs
    validate_plotting_data(df)

    # Get styling parameters from kwargs
    line_width = kwargs.get("line_width", 2.0)
    line_color = kwargs.get("line_color")
    line_dash = kwargs.get("line_dash", "solid")
    line_opacity = kwargs.get("line_opacity", 1.0)
    hovermode = kwargs.get("hovermode", "closest")

    # Detect panel data
    _, panel_groups = inspect_panel(df)

    # Auto-detect panel mode when data contains panel groups
    if panel_group_names is None and columns is None and panel_groups:
        panel_group_names = []

    if panel_group_names is not None and panel_groups:
        # Pre-compute palette for consistent member colouring
        pn_cols = resolve_panel_columns(df, panel_group_names, columns if panel_group_names is not None else None)
        _, all_members = _group_panel_columns(pn_cols)
        color_palette = resolve_color_palette(color_palette, len(all_members))
        seen: set[str] = set()

        def _render_ts(
            fig: go.Figure,
            sub_df: pl.DataFrame,
            member_name: str,
            member_idx: int,
            row: int,
            col_idx: int,
        ) -> None:
            """Render one member trace into the faceted figure."""
            color = line_color if line_color is not None else color_palette[member_idx]
            first_seen = member_name not in seen
            seen.add(member_name)
            fig.add_trace(
                go.Scatter(
                    x=sub_df["time"],
                    y=sub_df[member_name],
                    mode="lines",
                    name=member_name,
                    line={"color": color, "width": line_width, "dash": line_dash},
                    opacity=line_opacity,
                    showlegend=first_seen and show_legend,
                    legendgroup=member_name,
                    hovertemplate=f"<b>{member_name}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
                ),
                row=row,
                col=col_idx,
            )

        fig = panel_facet_figure(
            df,
            _render_ts,
            panel_group_names=panel_group_names,
            columns=columns if panel_group_names is not None else None,
            facet_n_cols=facet_n_cols,
            title=title,
            x_label=x_label or "time",
            y_label=y_label,
            width=width,
            height=height,
            resampler=resampler,
        )
        fig.update_layout(hovermode=hovermode, showlegend=show_legend)

        return fig

    # Non-panel case: columns refer to actual column names
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])
    # Get color palette
    color_palette = resolve_color_palette(color_palette, len(plot_columns))

    # Create figure
    fig = _create_figure(resampler)

    # Plot each column
    for idx, col in enumerate(plot_columns):
        color = line_color if line_color is not None else color_palette[idx]

        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[col],
                mode="lines",
                name=col,
                line={
                    "color": color,
                    "width": line_width,
                    "dash": line_dash,
                },
                opacity=line_opacity,
                hovertemplate=f"<b>{col}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
            )
        )

    # Apply layout
    if x_label is None:
        x_label = "time"

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


def plot_rolling_statistics(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    window_size: int | dict[str, int] = 7,
    statistics: str | list[str] = "mean",
    show_original: bool = True,
    panel_group_names: list[str] | None = None,
    facet_n_cols: int = 2,
    color_palette: list[str] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    resampler: bool | Literal["widget"] | None = None,
    **kwargs,
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
    facet_n_cols : int, default=2
        Number of columns in facet grid.
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
        Additional styling parameters:
        - smooth_color : str, default="#3366FF"
        - smooth_width : float, default=2.5
        - smooth_opacity : float, default=0.8
        - line_width : float, default=1.5
        - line_opacity : float, default=0.5

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
    `plot_time_series` : Plot basic time series.
    """
    # Validate inputs
    validate_plotting_data(df)

    # Auto-detect panel data
    _, _panel_groups = inspect_panel(df)
    if panel_group_names is None and columns is None and _panel_groups:
        panel_group_names = []

    if panel_group_names is not None:

        def _render_rolling(
            fig: go.Figure,
            sub_df: pl.DataFrame,
            display_name: str,
            panel_idx: int,  # noqa: ARG001
            row: int,
            col: int,
        ) -> None:
            """Render rolling window statistics traces for a single column."""
            base = [c for c in sub_df.columns if c != "time"][0]
            _stats = [statistics] if isinstance(statistics, str) else list(statistics)
            ws = window_size.get(base, 7) if isinstance(window_size, dict) else window_size
            if show_original:
                fig.add_trace(
                    go.Scatter(
                        x=sub_df["time"],
                        y=sub_df[base],
                        mode="lines",
                        name=display_name,
                        line={"color": "#94a3b8", "width": kwargs.get("line_width", 1.5)},
                        opacity=kwargs.get("line_opacity", 0.5),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
            t = RollingStatisticsTransformer(window_size=ws, statistics=_stats)
            t.fit(sub_df)
            df_s = t.transform(sub_df)
            colors = resolve_color_palette(None, len(_stats))
            for si, stat in enumerate(_stats):
                scol = f"{base}_{stat}"
                fig.add_trace(
                    go.Scatter(
                        x=df_s["time"],
                        y=df_s[scol],
                        mode="lines",
                        name=stat,
                        line={"color": colors[si % len(colors)], "width": kwargs.get("smooth_width", 2.5)},
                        opacity=kwargs.get("smooth_opacity", 0.8),
                        showlegend=(row == 1 and col == 1),
                    ),
                    row=row,
                    col=col,
                )

        return panel_facet_figure(
            df,
            _render_rolling,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_n_cols=facet_n_cols,
            title=title or "Rolling Statistics",
            x_label=x_label,
            y_label=y_label,
            width=width,
            height=height,
            resampler=resampler,
        )

    # Resolve columns
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])

    # Get styling parameters
    smooth_width = kwargs.get("smooth_width", 2.5)
    smooth_opacity = kwargs.get("smooth_opacity", 0.8)
    line_width = kwargs.get("line_width", 1.5)
    line_opacity = kwargs.get("line_opacity", 0.5)

    # Convert statistics to list
    if isinstance(statistics, str):
        statistics = [statistics]

    multi_col = len(plot_columns) > 1
    colors = resolve_color_palette(color_palette, max(len(plot_columns), len(statistics)))

    fig = _create_figure(resampler)

    for col_idx, col in enumerate(plot_columns):
        col_color = colors[col_idx % len(colors)] if multi_col else "#94a3b8"

        # Show original series if requested
        if show_original:
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df[col],
                    mode="lines",
                    name=f"{col} (Original)" if multi_col else col,
                    line={"color": col_color if multi_col else "#94a3b8", "width": line_width},
                    opacity=line_opacity,
                    legendgroup=col if multi_col else None,
                    hovertemplate=f"<b>{col}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
                )
            )

        # Compute rolling statistics
        df_col = df.select(["time", col])
        ws = window_size.get(col, 7) if isinstance(window_size, dict) else window_size
        transformer = RollingStatisticsTransformer(window_size=ws, statistics=statistics)
        transformer.fit(df_col)
        df_stats = transformer.transform(df_col)

        # Build stat_data dict
        stat_data: dict[str, pl.DataFrame] = {}
        for stat in statistics:
            stat_col = f"{col}_{stat}"
            stat_data[stat] = df_stats.select(["time", stat_col])

        stat_colors = colors if not multi_col else [col_color] * len(statistics)
        for s_idx, (stat, data) in enumerate(stat_data.items()):
            col_name = [c for c in data.columns if c != "time"][0]
            s_color = stat_colors[s_idx % len(stat_colors)]
            fig.add_trace(
                go.Scatter(
                    x=data["time"],
                    y=data[col_name],
                    mode="lines",
                    name=f"{col} ({stat})" if multi_col else stat,
                    line={"color": s_color, "width": smooth_width},
                    opacity=smooth_opacity,
                    legendgroup=col if multi_col else None,
                    hovertemplate=f"<b>{stat}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
                )
            )

    # Apply layout
    if x_label is None:
        x_label = "time"

    fig = apply_default_layout(
        fig,
        title=title,
        x_label=x_label,
        y_label=y_label,
        width=width,
        height=height,
    )

    return fig


def plot_boxplot(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    period: str = "1mo",
    panel_group_names: list[str] | None = None,
    color_palette: list[str] | None = None,
    facet_n_cols: int = 2,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
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
    color_palette : list[str] | None, default=None
        Custom color palette for multi-column plots.
    facet_n_cols : int, default=2
        Number of columns in facet grid.
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
        Additional styling parameters:
        - box_color : str, default="#2563EB"
        - box_opacity : float, default=0.7
        - show_points : bool | str, default="outliers" ("outliers", "all", False)
        - point_size : float, default=4.0

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
    `plot_time_series` : Plot basic time series.
    """
    # Validate inputs
    validate_plotting_data(df)

    # Auto-detect panel data
    _, _panel_groups = inspect_panel(df)
    if panel_group_names is None and columns is None and _panel_groups:
        panel_group_names = []

    if panel_group_names is not None:

        def _render_boxplot(
            fig: go.Figure,
            sub_df: pl.DataFrame,
            display_name: str,  # noqa: ARG001
            panel_idx: int,  # noqa: ARG001
            row: int,
            col: int,
        ) -> None:
            """Render period-grouped box plots for a single column."""
            base = [c for c in sub_df.columns if c != "time"][0]
            _bc = kwargs.get("box_color", "#2563EB")
            _ba = kwargs.get("box_opacity", 0.7)
            _sp = kwargs.get("show_points", "outliers")
            _ps = kwargs.get("point_size", 4.0)
            df_g = sub_df.with_columns(pl.col("time").dt.truncate(period).alias("period"))
            periods_list = df_g.select("period").unique().sort("period")["period"].to_list()
            for pv in periods_list:
                pd_data = df_g.filter(pl.col("period") == pv)[base]
                bp = "all" if _sp == "all" else ("outliers" if _sp == "outliers" else False)
                fig.add_trace(
                    go.Box(
                        y=pd_data,
                        name=str(pv),
                        marker={"color": _bc},
                        opacity=_ba,
                        boxpoints=bp,
                        marker_size=_ps if bp else None,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

        return panel_facet_figure(
            df,
            _render_boxplot,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_n_cols=facet_n_cols,
            title=title or "Boxplots",
            x_label=x_label or "Period",
            y_label=y_label,
            width=width,
            height=height,
            shared_xaxes=False,
        )

    # Resolve columns
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])

    # Get styling parameters
    box_color = kwargs.get("box_color", "#2563EB")
    box_opacity = kwargs.get("box_opacity", 0.7)
    show_points = kwargs.get("show_points", "outliers")
    point_size = kwargs.get("point_size", 4.0)

    multi_col = len(plot_columns) > 1
    col_colors = resolve_color_palette(color_palette, len(plot_columns)) if multi_col else [box_color]

    # Determine point display
    if show_points == "all":
        boxpoints: str | bool = "all"
    elif show_points == "outliers":
        boxpoints = "outliers"
    else:
        boxpoints = False

    # Group by period
    df_grouped = df.with_columns([pl.col("time").dt.truncate(period).alias("period")])
    periods = df_grouped.select("period").unique().sort("period")["period"].to_list()
    period_labels = [str(p) for p in periods]

    fig = go.Figure()

    for col_idx, col in enumerate(plot_columns):
        c_color = col_colors[col_idx]
        for p_idx, period_val in enumerate(periods):
            period_data = df_grouped.filter(pl.col("period") == period_val)[col]
            fig.add_trace(
                go.Box(
                    y=period_data,
                    x=[period_labels[p_idx]] * len(period_data),
                    name=col if multi_col else period_labels[p_idx],
                    marker={"color": c_color},
                    opacity=box_opacity,
                    boxpoints=boxpoints,
                    marker_size=point_size if boxpoints else None,
                    legendgroup=col if multi_col else None,
                    showlegend=multi_col and p_idx == 0,
                    offsetgroup=col if multi_col else None,
                    hovertemplate=f"<b>{col}</b><br>Period: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>",
                )
            )

    if multi_col:
        fig.update_layout(boxmode="group")

    fig = apply_default_layout(
        fig,
        title=title,
        x_label=x_label or "Period",
        y_label=y_label or (plot_columns[0] if not multi_col else "Value"),
        width=width,
        height=height,
    )

    return fig


def plot_missing_data(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    kind: Literal["heatmap", "bars", "matrix"] = "heatmap",
    panel_group_names: list[str] | None = None,
    facet_n_cols: int = 2,
    color_missing: str = "#DC2626",
    color_present: str = "#059669",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
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
    facet_n_cols : int, default=2
        Number of columns in facet grid.
    color_missing : str, default="#DC2626"
        Color for missing values (red).
    color_present : str, default="#059669"
        Color for present values (green).
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
        Additional styling parameters:
        - show_percentages : bool, default=True
        - time_aggregation : str | None, default=None ("1d", "1w", "1mo")

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
    `plot_time_series` : Plot basic time series.
    """
    # Validate inputs
    validate_plotting_data(df)

    # Auto-detect panel data
    _, _panel_groups = inspect_panel(df)
    if panel_group_names is None and columns is None and _panel_groups:
        panel_group_names = []

    if panel_group_names is not None:

        def _render_missing(
            fig: go.Figure,
            sub_df: pl.DataFrame,
            display_name: str,  # noqa: ARG001
            panel_idx: int,  # noqa: ARG001
            row: int,
            col: int,
        ) -> None:
            """Render missing data count bar chart for a single column."""
            base = [c for c in sub_df.columns if c != "time"][0]
            _sp = kwargs.get("show_percentages", True)
            total = len(sub_df)
            mc = sub_df[base].null_count()
            pct = (mc / total) * 100 if total > 0 else 0
            text = f"{pct:.1f}%" if _sp else None
            fig.add_trace(
                go.Bar(
                    x=[base],
                    y=[pct],
                    marker={"color": color_missing},
                    text=[text] if text else None,
                    textposition="auto" if text else None,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        return panel_facet_figure(
            df,
            _render_missing,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_n_cols=facet_n_cols,
            title=title or "Missing Data",
            x_label=x_label or "Column",
            y_label=y_label or "Missing (%)",
            width=width,
            height=height,
            shared_xaxes=False,
        )

    # Resolve columns
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])
    show_percentages = kwargs.get("show_percentages", True)
    time_aggregation = kwargs.get("time_aggregation")

    if kind == "bars":
        # Bar chart showing percentage missing per column
        missing_counts = []
        total_rows = len(df)

        for col in plot_columns:
            missing = df[col].null_count()
            pct = (missing / total_rows) * 100
            missing_counts.append({"column": col, "missing_pct": pct})

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[item["column"] for item in missing_counts],
                y=[item["missing_pct"] for item in missing_counts],
                marker={"color": color_missing},
                text=[f"{item['missing_pct']:.1f}%" for item in missing_counts] if show_percentages else None,
                textposition="auto" if show_percentages else None,
                hovertemplate="<b>%{x}</b><br>Missing: %{y:.1f}%<extra></extra>",
            )
        )

        if x_label is None:
            x_label = "Column"
        if y_label is None:
            y_label = "Missing (%)"

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
        title=title or "Missing Data Visualization",
        x_label=x_label,
        y_label=y_label,
        width=width,
        height=height,
    )

    return fig


def plot_distribution(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    n_bins: int = 50,
    show_kde: bool = True,
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
    facet_n_cols : int, default=2
        Number of columns in facet grid.
    color_palette : list[str] | None, default=None
        Custom color palette as hex codes.
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
        Additional styling parameters:
        - bar_opacity : float, default=0.6
        - kde_width : float, default=2.5
        - kde_points : int, default=200
        - histnorm : str, default="probability density"

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
    `plot_boxplot` : Plot boxplots grouped by time periods.
    `plot_time_series` : Plot basic time series.
    """
    # Validate inputs
    validate_plotting_data(df)

    # Get styling parameters from kwargs
    bar_opacity = kwargs.get("bar_opacity", 0.6)
    kde_width = kwargs.get("kde_width", 2.5)
    kde_points = kwargs.get("kde_points", 200)
    histnorm = kwargs.get("histnorm", "probability density")

    # Detect panel data
    _, _panel_groups = inspect_panel(df)

    if panel_group_names is None and columns is None and _panel_groups:
        panel_group_names = []

    if panel_group_names is not None and _panel_groups:

        def _render_distribution(
            fig: go.Figure,
            sub_df: pl.DataFrame,
            display_name: str,  # noqa: ARG001
            panel_idx: int,  # noqa: ARG001
            row: int,
            col: int,
        ) -> None:
            """Render histogram + KDE for a single panel column."""
            base = [c for c in sub_df.columns if c != "time"][0]
            series = sub_df[base].drop_nulls()
            values = series.to_numpy()
            _colors = resolve_color_palette(color_palette, 1)
            fig.add_trace(
                go.Histogram(
                    x=values,
                    nbinsx=n_bins,
                    marker={"color": _colors[0]},
                    opacity=bar_opacity,
                    histnorm=histnorm,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{base}</b><br>"
                        "Value: %{x:.2f}<br>"
                        "Density: %{y:.4f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            if show_kde and len(values) > 1:
                from scipy.stats import gaussian_kde  # noqa: PLC0415

                kde = gaussian_kde(values)
                x_range = np.linspace(float(values.min()), float(values.max()), kde_points)
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=kde(x_range),
                        mode="lines",
                        line={"color": _colors[0], "width": kde_width},
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{base} KDE</b><br>"
                            "Value: %{x:.2f}<br>"
                            "Density: %{y:.4f}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

        return panel_facet_figure(
            df,
            _render_distribution,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_n_cols=facet_n_cols,
            title=title or "Distribution",
            x_label=x_label or "Value",
            y_label=y_label or "Density",
            width=width,
            height=height,
            shared_xaxes=False,
        )

    # Non-panel case
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])
    colors = resolve_color_palette(color_palette, len(plot_columns))

    fig = go.Figure()

    for idx, col in enumerate(plot_columns):
        series = df[col].drop_nulls()
        values = series.to_numpy()

        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=n_bins,
                name=col,
                marker={"color": colors[idx]},
                opacity=bar_opacity,
                histnorm=histnorm,
                hovertemplate=(
                    f"<b>{col}</b><br>"
                    "Value: %{x:.2f}<br>"
                    "Density: %{y:.4f}<extra></extra>"
                ),
            )
        )

        if show_kde and len(values) > 1:
            from scipy.stats import gaussian_kde  # noqa: PLC0415

            kde = gaussian_kde(values)
            x_range = np.linspace(float(values.min()), float(values.max()), kde_points)
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=kde(x_range),
                    mode="lines",
                    name=f"{col} (KDE)",
                    line={"color": colors[idx], "width": kde_width},
                    hovertemplate=(
                        f"<b>{col} KDE</b><br>"
                        "Value: %{x:.2f}<br>"
                        "Density: %{y:.4f}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(barmode="overlay")

    fig = apply_default_layout(
        fig,
        title=title,
        x_label=x_label or "Value",
        y_label=y_label or "Density",
        width=width,
        height=height,
    )

    return fig


def plot_outliers(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    method: Literal["zscore", "iqr", "percentile"] = "zscore",
    threshold: float = 3.0,
    panel_group_names: list[str] | None = None,
    facet_n_cols: int = 2,
    color_palette: list[str] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    resampler: bool | Literal["widget"] | None = None,
    **kwargs,
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
        - "zscore": points with \|z-score\| > threshold (default 3.0)
        - "iqr": points outside [Q1 - threshold*IQR, Q3 + threshold*IQR] (default 1.5)
        - "percentile": points below threshold-th or above (100-threshold)-th
          percentile (default 5.0)
    threshold : float, default=3.0
        Detection threshold. Interpretation depends on *method*.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot.
    facet_n_cols : int, default=2
        Number of columns in facet grid.
    color_palette : list[str] | None, default=None
        Custom color palette for the series lines.
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
        Additional styling parameters:
        - line_width : float, default=1.5
        - line_opacity : float, default=0.7
        - outlier_color : str, default="#DC2626"
        - outlier_size : float, default=8.0
        - outlier_symbol : str, default="circle"
        - show_bounds : bool, default=True

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
    `plot_time_series` : Plot basic time series.
    `plot_boxplot` : Plot boxplots grouped by time periods.
    """
    # Validate inputs
    validate_plotting_data(df)

    if method not in ("zscore", "iqr", "percentile"):
        msg = f"Unknown method: {method}. Valid options: zscore, iqr, percentile"
        raise ValueError(msg)

    # Get styling parameters
    line_width = kwargs.get("line_width", 1.5)
    line_opacity = kwargs.get("line_opacity", 0.7)
    outlier_color = kwargs.get("outlier_color", "#DC2626")
    outlier_size = kwargs.get("outlier_size", 8.0)
    outlier_symbol = kwargs.get("outlier_symbol", "circle")
    show_bounds = kwargs.get("show_bounds", True)

    def _compute_outlier_mask(series: pl.Series) -> tuple[pl.Series, float | None, float | None]:
        """Return (is_outlier_bool_series, lower_bound, upper_bound)."""
        if method == "zscore":
            mean = series.mean()
            std = series.std()
            if std is None or std == 0 or mean is None:
                return pl.Series([False] * len(series)), None, None
            lower = mean - threshold * std
            upper = mean + threshold * std
            mask = ((series - mean).abs() / std) > threshold
        elif method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            if q1 is None or q3 is None:
                return pl.Series([False] * len(series)), None, None
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            mask = (series < lower) | (series > upper)
        else:  # percentile
            lower = series.quantile(threshold / 100)
            upper = series.quantile(1 - threshold / 100)
            if lower is None or upper is None:
                return pl.Series([False] * len(series)), None, None
            mask = (series < lower) | (series > upper)
        return mask.fill_null(False), lower, upper

    # Auto-detect panel data
    _, _panel_groups = inspect_panel(df)

    if panel_group_names is None and columns is None and _panel_groups:
        panel_group_names = []

    if panel_group_names is not None and _panel_groups:

        def _render_outlier(
            fig: go.Figure,
            sub_df: pl.DataFrame,
            display_name: str,  # noqa: ARG001
            panel_idx: int,  # noqa: ARG001
            row: int,
            col: int,
        ) -> None:
            """Render time series with outlier highlights for a single panel."""
            base = [c for c in sub_df.columns if c != "time"][0]
            _colors = resolve_color_palette(color_palette, 1)
            # Line trace
            fig.add_trace(
                go.Scatter(
                    x=sub_df["time"],
                    y=sub_df[base],
                    mode="lines",
                    line={"color": _colors[0], "width": line_width},
                    opacity=line_opacity,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{base}</b><br>"
                        "%{x}<br>%{y:.2f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            # Outlier markers
            mask, lower, upper = _compute_outlier_mask(sub_df[base])
            df_out = sub_df.filter(mask)
            if len(df_out) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=df_out["time"],
                        y=df_out[base],
                        mode="markers",
                        marker={"color": outlier_color, "size": outlier_size, "symbol": outlier_symbol},
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{base} OUTLIER</b><br>"
                            "%{x}<br>%{y:.2f}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
            # Threshold bounds
            if show_bounds and lower is not None and upper is not None:
                for val in (lower, upper):
                    fig.add_trace(
                        go.Scatter(
                            x=[sub_df["time"].min(), sub_df["time"].max()],
                            y=[val, val],
                            mode="lines",
                            line={"dash": "dash", "color": outlier_color, "width": 1},
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )

        return panel_facet_figure(
            df,
            _render_outlier,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_n_cols=facet_n_cols,
            title=title or f"Outlier Detection - {method}",
            x_label=x_label or "time",
            y_label=y_label,
            width=width,
            height=height,
            resampler=resampler,
        )

    # Non-panel case
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])
    colors = resolve_color_palette(color_palette, len(plot_columns))

    fig = _create_figure(resampler)

    for idx, col in enumerate(plot_columns):
        series = df[col]
        mask, lower, upper = _compute_outlier_mask(series)

        # Series line
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=series,
                mode="lines",
                name=col,
                line={"color": colors[idx], "width": line_width},
                opacity=line_opacity,
                hovertemplate=(
                    f"<b>{col}</b><br>"
                    "%{x}<br>%{y:.2f}<extra></extra>"
                ),
            )
        )

        # Outlier markers
        df_outliers = df.filter(mask)
        n_outliers = len(df_outliers)
        if n_outliers > 0:
            fig.add_trace(
                go.Scatter(
                    x=df_outliers["time"],
                    y=df_outliers[col],
                    mode="markers",
                    name=f"{col} outliers ({n_outliers})",
                    marker={
                        "color": outlier_color,
                        "size": outlier_size,
                        "symbol": outlier_symbol,
                    },
                    hovertemplate=(
                        f"<b>{col} OUTLIER</b><br>"
                        "%{x}<br>%{y:.2f}<extra></extra>"
                    ),
                )
            )

        # Threshold bounds
        if show_bounds and lower is not None and upper is not None:
            time_min = df["time"].min()
            time_max = df["time"].max()
            for bound_val, bound_label in [(lower, "Lower"), (upper, "Upper")]:
                fig.add_trace(
                    go.Scatter(
                        x=[time_min, time_max],
                        y=[bound_val, bound_val],
                        mode="lines",
                        name=f"{col} {bound_label} ({bound_val:.2f})",
                        line={"dash": "dash", "color": outlier_color, "width": 1},
                        showlegend=len(plot_columns) == 1,
                        hoverinfo="skip",
                    )
                )

    fig = apply_default_layout(
        fig,
        title=title or f"Outlier Detection ({method})",
        x_label=x_label or "time",
        y_label=y_label,
        width=width,
        height=height,
    )

    return fig


def plot_resampling_comparison(
    df_original: pl.DataFrame,
    df_resampled: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    original_label: str = "Original",
    resampled_label: str = "Resampled",
    panel_group_names: list[str] | None = None,
    facet_n_cols: int = 2,
    color_palette: list[str] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    resampler: bool | Literal["widget"] | None = None,
    **kwargs,
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
    facet_n_cols : int, default=2
        Number of columns in facet grid.
    color_palette : list[str] | None, default=None
        Custom color palette.
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
        Additional styling parameters:
        - original_width : float, default=1.0
        - original_opacity : float, default=0.4
        - original_dash : str, default="solid"
        - resampled_width : float, default=2.5
        - resampled_opacity : float, default=1.0
        - resampled_dash : str, default="solid"

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
    ...         pl.datetime(2020, 1, 1), pl.datetime(2020, 1, 2, 23), "1h", eager=True,
    ...     ),
    ...     "y": list(range(48)),
    ... })
    >>> daily = hourly.group_by_dynamic("time", every="1d").agg(pl.col("y").mean())
    >>> fig = plot_resampling_comparison(hourly, daily, columns="y")
    >>> len(fig.data)
    2

    See Also
    --------
    `plot_time_series` : Plot basic time series.
    `plot_rolling_statistics` : Plot rolling window statistics.
    """
    # Validate both DataFrames
    validate_plotting_data(df_original)
    validate_plotting_data(df_resampled)

    # Get styling parameters
    original_width = kwargs.get("original_width", 1.0)
    original_opacity = kwargs.get("original_opacity", 0.4)
    original_dash = kwargs.get("original_dash", "solid")
    resampled_width = kwargs.get("resampled_width", 2.5)
    resampled_opacity = kwargs.get("resampled_opacity", 1.0)
    resampled_dash = kwargs.get("resampled_dash", "solid")

    # Auto-detect panel data
    _, _panel_groups = inspect_panel(df_resampled)

    if panel_group_names is None and columns is None and _panel_groups:
        panel_group_names = []

    if panel_group_names is not None and _panel_groups:

        def _render_resampling(
            fig: go.Figure,
            sub_df: pl.DataFrame,
            display_name: str,  # noqa: ARG001
            panel_idx: int,  # noqa: ARG001
            row: int,
            col: int,
        ) -> None:
            """Render original and resampled traces for a single panel."""
            base = [c for c in sub_df.columns if c != "time"][0]
            _colors = resolve_color_palette(color_palette, 1)
            # Original (from df_original, matching the same full column name)
            full_col = [c for c in df_original.columns if c.endswith(f"__{base}") or c == base]
            if full_col:
                orig_col = full_col[0]
                fig.add_trace(
                    go.Scatter(
                        x=df_original["time"],
                        y=df_original[orig_col],
                        mode="lines",
                        line={"color": _colors[0], "width": original_width, "dash": original_dash},
                        opacity=original_opacity,
                        name=original_label,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
            # Resampled
            fig.add_trace(
                go.Scatter(
                    x=sub_df["time"],
                    y=sub_df[base],
                    mode="lines+markers",
                    line={"color": _colors[0], "width": resampled_width, "dash": resampled_dash},
                    opacity=resampled_opacity,
                    name=resampled_label,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        return panel_facet_figure(
            df_resampled,
            _render_resampling,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_n_cols=facet_n_cols,
            title=title or f"{original_label} vs {resampled_label}",
            x_label=x_label or "time",
            y_label=y_label,
            width=width,
            height=height,
            resampler=resampler,
        )

    # Non-panel case: resolve columns from resampled
    plot_columns = validate_plotting_data(df_resampled, columns=columns, exclude=["time"])

    # Ensure columns exist in original
    for col in plot_columns:
        if col not in df_original.columns:
            msg = f"Column '{col}' not found in original DataFrame. Available: {df_original.columns}"
            raise ValueError(msg)

    colors = resolve_color_palette(color_palette, len(plot_columns))
    fig = _create_figure(resampler)

    for idx, col in enumerate(plot_columns):
        color = colors[idx]

        # Original series (faded)
        fig.add_trace(
            go.Scatter(
                x=df_original["time"],
                y=df_original[col],
                mode="lines",
                name=f"{col} ({original_label})",
                line={"color": color, "width": original_width, "dash": original_dash},
                opacity=original_opacity,
                legendgroup=col,
                hovertemplate=(
                    f"<b>{col} ({original_label})</b><br>"
                    "%{x}<br>%{y:.2f}<extra></extra>"
                ),
            )
        )

        # Resampled series (emphasized)
        fig.add_trace(
            go.Scatter(
                x=df_resampled["time"],
                y=df_resampled[col],
                mode="lines+markers",
                name=f"{col} ({resampled_label})",
                line={"color": color, "width": resampled_width, "dash": resampled_dash},
                opacity=resampled_opacity,
                legendgroup=col,
                hovertemplate=(
                    f"<b>{col} ({resampled_label})</b><br>"
                    "%{x}<br>%{y:.2f}<extra></extra>"
                ),
            )
        )

    fig = apply_default_layout(
        fig,
        title=title or f"{original_label} vs {resampled_label}",
        x_label=x_label or "time",
        y_label=y_label,
        width=width,
        height=height,
    )

    return fig
