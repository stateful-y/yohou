"""Core time series plotting functions."""

from typing import Literal

import plotly.graph_objects as go
import polars as pl

from yohou.plotting.colors import get_color_sequence
from yohou.plotting.plotly_utils import apply_default_layout
from yohou.plotting.prep import resolve_columns, validate_dataframe


def plot_timeseries(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    panel_group_name: str | None = None,
    facet_ncol: int = 2,  # noqa: ARG001 - will be used when faceting is implemented
    facet_scales: Literal["free_y", "free_x", "free", "fixed"] = "free_y",  # noqa: ARG001
    dropdown: bool = False,  # noqa: ARG001
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
    Plot basic line plots for one or more time series.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns to plot.
    columns : str | list[str] | None, default=None
        Column(s) to plot. If None, plots all numeric columns except 'time'.
        If str, plots single column. If list, plots multiple columns.
    panel_group_name : str | None, default=None
        Column name for grouping (panel data). Creates separate plots per group.
    facet_ncol : int, default=2
        Number of columns in facet grid when using panel groups.
    facet_scales : {"free_y", "free_x", "free", "fixed"}, default="free_y"
        Scale sharing between facets. Currently only "free_y" supported.
    dropdown : bool, default=False
        If True, use dropdown menu instead of facets for panel groups.
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
        - line_alpha : float, default=1.0
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
    >>> from yohou.plotting import plot_timeseries

    >>> # Create sample data
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
    ...     "y": [100, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180, 175],
    ... })

    >>> # Plot single column
    >>> fig = plot_timeseries(df, columns="y")
    >>> len(fig.data)
    1

    >>> # Multiple columns
    >>> df = df.with_columns((pl.col("y") * 1.1).alias("y2"))
    >>> fig = plot_timeseries(df, columns=["y", "y2"])
    >>> len(fig.data)
    2

    See Also
    --------
    plot_rolling_statistics : Plot rolling window statistics.
    plot_exponential_moving_average : Plot EWM smoothing.
    """
    # Validate inputs
    validate_dataframe(df)

    # Resolve columns to plot
    plot_columns = resolve_columns(df, columns=columns, exclude=["time"])

    # Get styling parameters from kwargs
    line_width = kwargs.get("line_width", 2.0)
    line_color = kwargs.get("line_color")
    line_dash = kwargs.get("line_dash", "solid")
    line_alpha = kwargs.get("line_alpha", 1.0)
    hovermode = kwargs.get("hovermode", "closest")

    # Get color palette
    if color_palette is None:
        color_palette = get_color_sequence(len(plot_columns))
    elif len(color_palette) < len(plot_columns):
        # Cycle through provided palette if not enough colors
        color_palette = [color_palette[i % len(color_palette)] for i in range(len(plot_columns))]

    # Create figure
    fig = go.Figure()

    # Handle panel groups (not implemented yet, simple case for now)
    if panel_group_name is not None:
        msg = "Panel grouping not yet implemented"
        raise NotImplementedError(msg)

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
                opacity=line_alpha,
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
    window_size: int = 7,
    statistics: str | list[str] = "mean",
    show_original: bool = True,
    panel_group_name: str | None = None,
    facet_ncol: int = 2,  # noqa: ARG001
    facet_scales: Literal["free_y", "free_x", "free", "fixed"] = "free_y",  # noqa: ARG001
    dropdown: bool = False,  # noqa: ARG001
    color_palette: list[str] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
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
    window_size : int, default=7
        Size of the rolling window.
    statistics : str | list[str], default="mean"
        Statistic(s) to compute. Options: "mean", "std", "min", "max", "median",
        "q25" (25th percentile), "q75" (75th percentile), "sum".
    show_original : bool, default=True
        Whether to show the original series alongside the statistics.
    panel_group_name : str | None, default=None
        Column name for grouping (panel data).
    facet_ncol : int, default=2
        Number of columns in facet grid.
    facet_scales : {"free_y", "free_x", "free", "fixed"}, default="free_y"
        Scale sharing between facets.
    dropdown : bool, default=False
        Use dropdown menu instead of facets.
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
        - smooth_alpha : float, default=0.8
        - line_width : float, default=1.5
        - line_alpha : float, default=0.5
        - fill_between : bool, default=False (fill area between statistics)
        - band_alpha : float, default=0.2

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
    plot_timeseries : Plot basic time series.
    plot_exponential_moving_average : Plot EWM smoothing.
    """
    # Validate inputs
    validate_dataframe(df)

    if panel_group_name is not None:
        msg = "Panel grouping not yet implemented"
        raise NotImplementedError(msg)

    # Resolve columns
    plot_columns = resolve_columns(df, columns=columns, exclude=["time"])

    if len(plot_columns) > 1:
        msg = "Multiple columns not yet supported for rolling statistics"
        raise NotImplementedError(msg)

    # Get styling parameters
    smooth_color = kwargs.get("smooth_color", "#3366FF")
    smooth_width = kwargs.get("smooth_width", 2.5)
    smooth_alpha = kwargs.get("smooth_alpha", 0.8)
    line_width = kwargs.get("line_width", 1.5)
    line_alpha = kwargs.get("line_alpha", 0.5)
    fill_between = kwargs.get("fill_between", False)
    band_alpha = kwargs.get("band_alpha", 0.2)

    # Convert statistics to list
    if isinstance(statistics, str):
        statistics = [statistics]

    # Create figure
    fig = go.Figure()

    col = plot_columns[0]

    # Show original series if requested
    if show_original:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[col],
                mode="lines",
                name=col,
                line={"color": "#94a3b8", "width": line_width},
                opacity=line_alpha,
                hovertemplate=f"<b>{col}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
            )
        )

    # Compute and plot statistics
    stat_data = {}
    for stat in statistics:
        if stat == "mean":
            stat_data[stat] = df.select([pl.col("time"), pl.col(col).rolling_mean(window_size).alias(f"{col}_mean")])
        elif stat == "std":
            stat_data[stat] = df.select([pl.col("time"), pl.col(col).rolling_std(window_size).alias(f"{col}_std")])
        elif stat == "min":
            stat_data[stat] = df.select([pl.col("time"), pl.col(col).rolling_min(window_size).alias(f"{col}_min")])
        elif stat == "max":
            stat_data[stat] = df.select([pl.col("time"), pl.col(col).rolling_max(window_size).alias(f"{col}_max")])
        elif stat == "median":
            stat_data[stat] = df.select([
                pl.col("time"),
                pl.col(col).rolling_median(window_size).alias(f"{col}_median"),
            ])
        elif stat == "q25":
            stat_data[stat] = df.select([
                pl.col("time"),
                pl.col(col).rolling_quantile(0.25, window_size=window_size).alias(f"{col}_q25"),
            ])
        elif stat == "q75":
            stat_data[stat] = df.select([
                pl.col("time"),
                pl.col(col).rolling_quantile(0.75, window_size=window_size).alias(f"{col}_q75"),
            ])
        elif stat == "sum":
            stat_data[stat] = df.select([pl.col("time"), pl.col(col).rolling_sum(window_size).alias(f"{col}_sum")])
        else:
            msg = f"Unknown statistic: {stat}. Valid options: mean, std, min, max, median, q25, q75, sum"
            raise ValueError(msg)

    # Handle fill_between for confidence bands
    if fill_between and len(statistics) >= 2:
        # Fill between first two statistics
        stat_names = list(stat_data.keys())
        first_stat = stat_data[stat_names[0]]
        second_stat = stat_data[stat_names[1]]

        # Get column names
        first_col_name = [c for c in first_stat.columns if c != "time"][0]
        second_col_name = [c for c in second_stat.columns if c != "time"][0]

        # Add upper bound
        fig.add_trace(
            go.Scatter(
                x=second_stat["time"],
                y=second_stat[second_col_name],
                mode="lines",
                name=stat_names[1],
                line={"color": smooth_color, "width": 0},
                showlegend=False,
                hovertemplate=f"<b>{stat_names[1]}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
            )
        )

        # Add lower bound with fill
        fig.add_trace(
            go.Scatter(
                x=first_stat["time"],
                y=first_stat[first_col_name],
                mode="lines",
                name=f"{stat_names[0]}-{stat_names[1]} band",
                line={"color": smooth_color, "width": 0},
                fill="tonexty",
                fillcolor=f"rgba({int(smooth_color[1:3], 16)}, {int(smooth_color[3:5], 16)}, {int(smooth_color[5:7], 16)}, {band_alpha})",
                hovertemplate=f"<b>{stat_names[0]}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
            )
        )

        # If there's a third statistic (e.g., mean between q25 and q75), plot it
        if len(statistics) >= 3:
            third_stat = stat_data[stat_names[2]]
            third_col_name = [c for c in third_stat.columns if c != "time"][0]

            fig.add_trace(
                go.Scatter(
                    x=third_stat["time"],
                    y=third_stat[third_col_name],
                    mode="lines",
                    name=stat_names[2],
                    line={"color": smooth_color, "width": smooth_width},
                    opacity=smooth_alpha,
                    hovertemplate=f"<b>{stat_names[2]}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
                )
            )
    else:
        # Plot each statistic as a line
        colors = get_color_sequence(len(statistics)) if color_palette is None else color_palette
        for idx, (stat, data) in enumerate(stat_data.items()):
            col_name = [c for c in data.columns if c != "time"][0]
            color = colors[idx % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=data["time"],
                    y=data[col_name],
                    mode="lines",
                    name=stat,
                    line={"color": color, "width": smooth_width},
                    opacity=smooth_alpha,
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


def plot_exponential_moving_average(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    span: int = 7,
    show_original: bool = True,
    panel_group_name: str | None = None,
    facet_ncol: int = 2,  # noqa: ARG001
    facet_scales: Literal["free_y", "free_x", "free", "fixed"] = "free_y",  # noqa: ARG001
    dropdown: bool = False,  # noqa: ARG001
    color_palette: list[str] | None = None,  # noqa: ARG001
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """
    Plot exponentially weighted moving average smoothing.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns to plot.
    columns : str | list[str] | None, default=None
        Column(s) to compute EWM for. If None, uses all numeric columns except 'time'.
    span : int, default=7
        Span for the exponentially weighted window (corresponds to center of mass).
    show_original : bool, default=True
        Whether to show the original series alongside the EWM.
    panel_group_name : str | None, default=None
        Column name for grouping (panel data).
    facet_ncol : int, default=2
        Number of columns in facet grid.
    facet_scales : {"free_y", "free_x", "free", "fixed"}, default="free_y"
        Scale sharing between facets.
    dropdown : bool, default=False
        Use dropdown menu instead of facets.
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
        - smooth_alpha : float, default=0.8
        - line_width : float, default=1.5
        - line_alpha : float, default=0.5

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_exponential_moving_average

    >>> # Create sample data
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
    ...     "y": [100, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180, 175],
    ... })

    >>> # Plot with EWM
    >>> fig = plot_exponential_moving_average(df, span=3)
    >>> len(fig.data)
    2

    >>> # EWM only (no original)
    >>> fig = plot_exponential_moving_average(df, span=3, show_original=False)
    >>> len(fig.data)
    1

    See Also
    --------
    plot_rolling_statistics : Plot rolling window statistics.
    plot_timeseries : Plot basic time series.
    """
    # Validate inputs
    validate_dataframe(df)

    if panel_group_name is not None:
        msg = "Panel grouping not yet implemented"
        raise NotImplementedError(msg)

    # Resolve columns
    plot_columns = resolve_columns(df, columns=columns, exclude=["time"])

    if len(plot_columns) > 1:
        msg = "Multiple columns not yet supported for exponential moving average"
        raise NotImplementedError(msg)

    # Get styling parameters
    smooth_color = kwargs.get("smooth_color", "#3366FF")
    smooth_width = kwargs.get("smooth_width", 2.5)
    smooth_alpha = kwargs.get("smooth_alpha", 0.8)
    line_width = kwargs.get("line_width", 1.5)
    line_alpha = kwargs.get("line_alpha", 0.5)

    # Create figure
    fig = go.Figure()

    col = plot_columns[0]

    # Show original series if requested
    if show_original:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[col],
                mode="lines",
                name=col,
                line={"color": "#94a3b8", "width": line_width},
                opacity=line_alpha,
                hovertemplate=f"<b>{col}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
            )
        )

    # Compute EWM
    ewm_data = df.select([pl.col("time"), pl.col(col).ewm_mean(span=span, adjust=True).alias(f"{col}_ewm")])

    # Plot EWM
    fig.add_trace(
        go.Scatter(
            x=ewm_data["time"],
            y=ewm_data[f"{col}_ewm"],
            mode="lines",
            name=f"EWM({span})",
            line={"color": smooth_color, "width": smooth_width},
            opacity=smooth_alpha,
            hovertemplate=f"<b>EWM({span})</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
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
    panel_group_name: str | None = None,
    facet_ncol: int = 2,  # noqa: ARG001
    facet_scales: Literal["free_y", "free_x", "free", "fixed"] = "free_y",  # noqa: ARG001
    dropdown: bool = False,  # noqa: ARG001
    color_palette: list[str] | None = None,  # noqa: ARG001
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
    panel_group_name : str | None, default=None
        Column name for grouping (panel data).
    facet_ncol : int, default=2
        Number of columns in facet grid.
    facet_scales : {"free_y", "free_x", "free", "fixed"}, default="free_y"
        Scale sharing between facets.
    dropdown : bool, default=False
        Use dropdown menu instead of facets.
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
        - box_color : str, default="#2563EB"
        - box_alpha : float, default=0.7
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
    plot_timeseries : Plot basic time series.
    """
    # Validate inputs
    validate_dataframe(df)

    if panel_group_name is not None:
        msg = "Panel grouping not yet implemented"
        raise NotImplementedError(msg)

    # Resolve columns
    plot_columns = resolve_columns(df, columns=columns, exclude=["time"])

    if len(plot_columns) > 1:
        msg = "Multiple columns not yet supported for boxplots"
        raise NotImplementedError(msg)

    # Get styling parameters
    box_color = kwargs.get("box_color", "#2563EB")
    box_alpha = kwargs.get("box_alpha", 0.7)
    show_points = kwargs.get("show_points", "outliers")
    point_size = kwargs.get("point_size", 4.0)

    col = plot_columns[0]

    # Group by period
    df_grouped = df.with_columns([pl.col("time").dt.truncate(period).alias("period")])

    # Get unique periods sorted
    periods = df_grouped.select("period").unique().sort("period")["period"].to_list()

    # Create figure
    fig = go.Figure()

    # Add boxplot for each period
    for period_val in periods:
        period_data = df_grouped.filter(pl.col("period") == period_val)[col]

        # Determine point display
        if show_points == "all":
            boxpoints = "all"
        elif show_points == "outliers":
            boxpoints = "outliers"
        else:
            boxpoints = False

        fig.add_trace(
            go.Box(
                y=period_data,
                name=str(period_val),
                marker={"color": box_color},
                opacity=box_alpha,
                boxpoints=boxpoints,
                marker_size=point_size if boxpoints else None,
                hovertemplate="<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>",
            )
        )

    # Apply layout
    if x_label is None:
        x_label = "Period"
    if y_label is None:
        y_label = col

    fig = apply_default_layout(
        fig,
        title=title,
        x_label=x_label,
        y_label=y_label,
        width=width,
        height=height,
    )

    return fig


def plot_prediction_interval(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    lower_bound_column: str = "lower_bound",
    upper_bound_column: str = "upper_bound",
    panel_group_name: str | None = None,
    facet_ncol: int = 2,  # noqa: ARG001
    facet_scales: str = "free_y",  # noqa: ARG001
    dropdown: bool = False,  # noqa: ARG001
    color_palette: list[str] | None = None,  # noqa: ARG001
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """
    Plot time series with prediction/confidence intervals.

    Visualizes point forecasts along with uncertainty bands using ribbon plots.
    Users must compute bounds before calling this function.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column, forecast column(s), and bound columns.
    columns : str | list[str] | None, default=None
        Forecast column(s) to plot. If None, uses all numeric columns except 'time' and bounds.
    lower_bound_column : str, default="lower_bound"
        Name of column containing lower bounds.
    upper_bound_column : str, default="upper_bound"
        Name of column containing upper bounds.
    panel_group_name : str | None, default=None
        Column name for grouping (panel data).
    facet_ncol : int, default=2
        Number of columns in facet grid.
    facet_scales : {"free_y", "free_x", "free", "fixed"}, default="free_y"
        Scale configuration for facets.
    dropdown : bool, default=False
        Use dropdown menu instead of facets.
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
        - line_width : float, default=2.0
        - line_color : str | None, default=None (uses palette)
        - band_color : str | None, default=None (uses line_color)
        - band_alpha : float, default=0.15

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_prediction_interval

    >>> # Create sample data with bounds
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
    ...     "forecast": [100, 105, 110, 108, 112, 115, 118, 120, 122, 125],
    ...     "lower_bound": [95, 98, 103, 100, 105, 108, 110, 112, 114, 117],
    ...     "upper_bound": [105, 112, 117, 116, 119, 122, 126, 128, 130, 133],
    ... })

    >>> # Plot with default bounds
    >>> fig = plot_prediction_interval(df, columns="forecast")
    >>> len(fig.data) > 0
    True

    See Also
    --------
    plot_timeseries : Plot basic time series.
    plot_forecast : Plot forecasts with historical data.
    """
    # Validate inputs
    validate_dataframe(df)

    if panel_group_name is not None:
        msg = "Panel grouping not yet implemented"
        raise NotImplementedError(msg)

    # Check that bound columns exist
    if lower_bound_column not in df.columns:
        msg = f"Lower bound column '{lower_bound_column}' not found in DataFrame"
        raise ValueError(msg)
    if upper_bound_column not in df.columns:
        msg = f"Upper bound column '{upper_bound_column}' not found in DataFrame"
        raise ValueError(msg)

    # Resolve columns (exclude time and bound columns)
    exclude_cols = ["time", lower_bound_column, upper_bound_column]
    plot_columns = resolve_columns(df, columns=columns, exclude=exclude_cols)

    # Get kwargs
    line_width = kwargs.get("line_width", 2.0)
    line_color = kwargs.get("line_color")
    band_color = kwargs.get("band_color")
    band_alpha = kwargs.get("band_alpha", 0.15)

    # Get color sequence
    colors = get_color_sequence(len(plot_columns))

    # Create figure
    fig = go.Figure()

    # Plot each column with its interval
    for idx, col in enumerate(plot_columns):
        color = line_color if line_color else colors[idx]
        fill_color = band_color if band_color else color

        # Add upper bound (invisible line for fill)
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[upper_bound_column],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
                name=f"{col}_upper",
            )
        )

        # Add lower bound with fill to upper
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[lower_bound_column],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=f"rgba({int(fill_color[1:3], 16)}, {int(fill_color[3:5], 16)}, {int(fill_color[5:7], 16)}, {band_alpha})",
                showlegend=True,
                name=f"{col} (interval)",
                hovertemplate=f"<b>{col} Interval</b><br>Time: %{{x}}<br>Lower: %{{y:.2f}}<extra></extra>",
            )
        )

        # Add point forecast line
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[col],
                mode="lines",
                line={"color": color, "width": line_width},
                name=col,
                hovertemplate=f"<b>{col}</b><br>Time: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>",
            )
        )

    # Set default labels
    if x_label is None:
        x_label = "Time"
    if y_label is None:
        y_label = "Value"

    # Apply default layout
    fig = apply_default_layout(
        fig,
        title=title,
        x_label=x_label,
        y_label=y_label,
        width=width,
        height=height,
    )

    return fig
