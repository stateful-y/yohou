"""Diagnostic plotting functions for time series analysis."""

import math

import plotly.graph_objects as go
import polars as pl

from yohou.plotting.colors import get_color_sequence
from yohou.plotting.plotly_utils import apply_default_layout
from yohou.plotting.prep import resolve_columns, validate_dataframe


def plot_autocorrelation(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    max_lags: int | None = None,
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
    Plot autocorrelation function (ACF) for time series.

    Shows correlation between the series and its lagged values at different time lags.
    Useful for identifying periodic patterns and determining appropriate MA order.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to analyze. If None, uses all numeric columns except 'time'.
    max_lags : int | None, default=None
        Maximum number of lags to compute. If None, uses min(len(df)//2, 40).
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
        - bar_color : str, default="#2563EB"
        - confidence_level : float, default=0.95
        - show_confidence : bool, default=True

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_autocorrelation

    >>> # Create sample time series
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
    ...     "y": [100 + i % 30 for i in range(366)],
    ... })

    >>> # Plot ACF
    >>> fig = plot_autocorrelation(df, columns="y", max_lags=20)
    >>> len(fig.data) > 0
    True

    See Also
    --------
    plot_partial_autocorrelation : Plot partial autocorrelation function.
    plot_correlation_diagnostics : Plot correlation matrix.
    """
    # Validate inputs
    validate_dataframe(df)

    if panel_group_name is not None:
        msg = "Panel grouping not yet implemented"
        raise NotImplementedError(msg)

    # Resolve columns
    plot_columns = resolve_columns(df, columns=columns, exclude=["time"])

    # Get kwargs
    bar_color = kwargs.get("bar_color", "#2563EB")
    confidence_level = kwargs.get("confidence_level", 0.95)
    show_confidence = kwargs.get("show_confidence", True)

    # Determine max_lags
    n = len(df)
    if max_lags is None:
        max_lags = min(n // 2, 40)

    # Create figure
    fig = go.Figure()

    # Compute ACF for each column
    for col in plot_columns:
        series = df[col].drop_nulls()
        n_series = len(series)

        # Compute autocorrelation
        acf_values = []
        mean_val = series.mean()

        for lag in range(max_lags + 1):
            if lag == 0:
                acf_values.append(1.0)
            else:
                # Compute correlation at lag
                series1 = series[:-lag] - mean_val
                series2 = series[lag:] - mean_val
                numerator = (series1 * series2).sum()
                denominator = ((series - mean_val) ** 2).sum()
                acf = numerator / denominator if denominator != 0 else 0
                acf_values.append(acf)

        # Add bar trace
        fig.add_trace(
            go.Bar(
                x=list(range(max_lags + 1)),
                y=acf_values,
                name=col,
                marker={"color": bar_color},
                hovertemplate=f"<b>{col}</b><br>Lag: %{{x}}<br>ACF: %{{y:.3f}}<extra></extra>",
            )
        )

        # Add confidence bands
        if show_confidence:
            # Approximate confidence interval using 1.96/sqrt(n)
            ci = 1.96 / math.sqrt(n_series) if confidence_level == 0.95 else 2.576 / math.sqrt(n_series)

            fig.add_trace(
                go.Scatter(
                    x=list(range(max_lags + 1)),
                    y=[ci] * (max_lags + 1),
                    mode="lines",
                    line={"dash": "dash", "color": "#DC2626", "width": 1},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(max_lags + 1)),
                    y=[-ci] * (max_lags + 1),
                    mode="lines",
                    line={"dash": "dash", "color": "#DC2626", "width": 1},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Set default labels
    if x_label is None:
        x_label = "Lag"
    if y_label is None:
        y_label = "Autocorrelation"

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


def plot_partial_autocorrelation(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    max_lags: int | None = None,
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
    Plot partial autocorrelation function (PACF) for time series.

    Shows correlation between the series and its lagged values after removing
    the effect of intermediate lags. Useful for determining appropriate AR order.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to analyze. If None, uses all numeric columns except 'time'.
    max_lags : int | None, default=None
        Maximum number of lags to compute. If None, uses min(len(df)//2, 40).
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
        - bar_color : str, default="#059669"
        - confidence_level : float, default=0.95
        - show_confidence : bool, default=True

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_partial_autocorrelation

    >>> # Create sample time series
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
    ...     "y": [100 + i % 30 for i in range(366)],
    ... })

    >>> # Plot PACF
    >>> fig = plot_partial_autocorrelation(df, columns="y", max_lags=20)
    >>> len(fig.data) > 0
    True

    See Also
    --------
    plot_autocorrelation : Plot autocorrelation function.
    """
    # Validate inputs
    validate_dataframe(df)

    if panel_group_name is not None:
        msg = "Panel grouping not yet implemented"
        raise NotImplementedError(msg)

    # Resolve columns
    plot_columns = resolve_columns(df, columns=columns, exclude=["time"])

    # Get kwargs
    bar_color = kwargs.get("bar_color", "#059669")
    confidence_level = kwargs.get("confidence_level", 0.95)
    show_confidence = kwargs.get("show_confidence", True)

    # Determine max_lags
    n = len(df)
    if max_lags is None:
        max_lags = min(n // 2, 40)

    # Create figure
    fig = go.Figure()

    # Compute PACF for each column using Durbin-Levinson recursion
    for col in plot_columns:
        series = df[col].drop_nulls()
        n_series = len(series)
        mean_val = series.mean()

        # Compute ACF first (needed for PACF)
        acf = []
        for lag in range(max_lags + 1):
            if lag == 0:
                acf.append(1.0)
            else:
                series1 = series[:-lag] - mean_val
                series2 = series[lag:] - mean_val
                numerator = (series1 * series2).sum()
                denominator = ((series - mean_val) ** 2).sum()
                acf.append(numerator / denominator if denominator != 0 else 0)

        # Compute PACF using Durbin-Levinson algorithm
        pacf_values = [1.0]  # PACF at lag 0 is always 1

        if max_lags > 0:
            phi = [[acf[1]]]  # Coefficients matrix
            pacf_values.append(acf[1])

            for k in range(2, max_lags + 1):
                # Compute PACF at lag k
                numerator = acf[k] - sum(phi[k - 2][j] * acf[k - j - 1] for j in range(k - 1))
                denominator = 1 - sum(phi[k - 2][j] * acf[j + 1] for j in range(k - 1))
                pacf_k = numerator / denominator if abs(denominator) > 1e-10 else 0
                pacf_values.append(pacf_k)

                # Update phi matrix
                new_phi = [phi[k - 2][j] - pacf_k * phi[k - 2][k - j - 2] for j in range(k - 1)]
                new_phi.append(pacf_k)
                phi.append(new_phi)

        # Add bar trace
        fig.add_trace(
            go.Bar(
                x=list(range(max_lags + 1)),
                y=pacf_values,
                name=col,
                marker={"color": bar_color},
                hovertemplate=f"<b>{col}</b><br>Lag: %{{x}}<br>PACF: %{{y:.3f}}<extra></extra>",
            )
        )

        # Add confidence bands
        if show_confidence:
            ci = 1.96 / math.sqrt(n_series) if confidence_level == 0.95 else 2.576 / math.sqrt(n_series)

            fig.add_trace(
                go.Scatter(
                    x=list(range(max_lags + 1)),
                    y=[ci] * (max_lags + 1),
                    mode="lines",
                    line={"dash": "dash", "color": "#DC2626", "width": 1},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(max_lags + 1)),
                    y=[-ci] * (max_lags + 1),
                    mode="lines",
                    line={"dash": "dash", "color": "#DC2626", "width": 1},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Set default labels
    if x_label is None:
        x_label = "Lag"
    if y_label is None:
        y_label = "Partial Autocorrelation"

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


def plot_correlation_diagnostics(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    panel_group_name: str | None = None,
    facet_ncol: int = 2,  # noqa: ARG001
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
    Plot correlation matrix heatmap for multiple time series.

    Shows pairwise correlations between different time series columns,
    useful for understanding relationships and multicollinearity.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to include. If None, uses all numeric columns except 'time'.
    panel_group_name : str | None, default=None
        Column name for grouping (panel data).
    facet_ncol : int, default=2
        Number of columns in facet grid.
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
        - colorscale : str, default="RdBu_r"
        - show_values : bool, default=True

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_correlation_diagnostics

    >>> # Create sample data with multiple series
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
    ...     "y1": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    ...     "y2": [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
    ... })

    >>> # Plot correlation matrix
    >>> fig = plot_correlation_diagnostics(df)
    >>> len(fig.data) > 0
    True

    See Also
    --------
    plot_autocorrelation : Plot autocorrelation function.
    """
    # Validate inputs
    validate_dataframe(df)

    if panel_group_name is not None:
        msg = "Panel grouping not yet implemented"
        raise NotImplementedError(msg)

    # Resolve columns
    plot_columns = resolve_columns(df, columns=columns, exclude=["time"])

    # Get kwargs
    colorscale = kwargs.get("colorscale", "RdBu_r")
    show_values = kwargs.get("show_values", True)

    # Compute correlation matrix
    corr_matrix = df.select(plot_columns).corr()

    # Create heatmap
    fig = go.Figure()

    # Prepare text annotations
    text_annotations = None
    if show_values:
        text_annotations = [[f"{val:.2f}" if val is not None else "" for val in row] for row in corr_matrix.rows()]

    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.to_numpy(),
            x=plot_columns,
            y=plot_columns,
            colorscale=colorscale,
            zmid=0,
            text=text_annotations,
            texttemplate="%{text}" if show_values else None,
            hovertemplate="<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
        )
    )

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


def plot_seasonality(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    frequency: str = "month",
    panel_group_name: str | None = None,
    facet_ncol: int = 2,  # noqa: ARG001
    facet_scales: str = "free_y",  # noqa: ARG001
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
    Plot seasonal subseries to visualize seasonal patterns.

    Creates subseries plots grouped by seasonal period (month, quarter, weekday, etc.),
    useful for identifying and understanding seasonal patterns.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to plot. If None, uses all numeric columns except 'time'.
    frequency : str, default="month"
        Seasonal frequency to group by: "month", "quarter", "weekday", "week", "hour".
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
    y_label : str | None = None
        Y-axis label.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    **kwargs : dict
        Additional styling parameters:
        - show_mean : bool, default=True
        - line_width : float, default=1.5
        - mean_color : str, default="#DC2626"

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_seasonality

    >>> # Create sample time series
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
    ...     "y": [100 + (i % 30) + (i // 30) * 5 for i in range(366)],
    ... })

    >>> # Plot monthly seasonality
    >>> fig = plot_seasonality(df, columns="y", frequency="month")
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

    # Get kwargs
    show_mean = kwargs.get("show_mean", True)
    line_width = kwargs.get("line_width", 1.5)
    mean_color = kwargs.get("mean_color", "#DC2626")

    # Get color sequence
    colors = color_palette if color_palette else get_color_sequence(len(plot_columns))

    # Create figure
    fig = go.Figure()

    # Extract seasonal component based on frequency
    if frequency == "month":
        df_seasonal = df.with_columns([pl.col("time").dt.month().alias("season")])
        season_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    elif frequency == "quarter":
        df_seasonal = df.with_columns([pl.col("time").dt.quarter().alias("season")])
        season_labels = ["Q1", "Q2", "Q3", "Q4"]
    elif frequency == "weekday":
        df_seasonal = df.with_columns([pl.col("time").dt.weekday().alias("season")])
        season_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    elif frequency == "week":
        df_seasonal = df.with_columns([pl.col("time").dt.week().alias("season")])
        season_labels = None  # Use numeric weeks
    elif frequency == "hour":
        df_seasonal = df.with_columns([pl.col("time").dt.hour().alias("season")])
        season_labels = None  # Use numeric hours
    else:
        msg = f"Unknown frequency: {frequency}. Valid options: month, quarter, weekday, week, hour"
        raise ValueError(msg)

    # Plot each column
    for idx, col in enumerate(plot_columns):
        # Group by season and compute statistics
        seasonal_stats = (
            df_seasonal.group_by("season")
            .agg([pl.col(col).mean().alias("mean"), pl.col(col).alias("values")])
            .sort("season")
        )

        seasons = seasonal_stats["season"].to_list()
        means = seasonal_stats["mean"].to_list()

        # Plot all observations for each season
        for i, season in enumerate(seasons):
            values = seasonal_stats["values"][i]
            x_vals = [season] * len(values)

            # Use season label if available
            season_label = season_labels[season - 1] if season_labels and season <= len(season_labels) else str(season)

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=values,
                    mode="markers",
                    marker={"size": 4, "color": colors[idx], "opacity": 0.3},
                    name=col if i == 0 else None,
                    legendgroup=col,
                    showlegend=(i == 0),
                    hovertemplate=f"<b>{col}</b><br>{season_label}: %{{y:.2f}}<extra></extra>",
                )
            )

        # Add mean line
        if show_mean:
            x_vals = seasons if not season_labels else [season_labels[s - 1] for s in seasons]
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=means,
                    mode="lines+markers",
                    line={"color": mean_color, "width": line_width},
                    marker={"size": 8, "color": mean_color},
                    name=f"{col} (mean)" if len(plot_columns) > 1 else "Mean",
                    legendgroup=col,
                    hovertemplate=f"<b>{col} Mean</b><br>%{{x}}: %{{y:.2f}}<extra></extra>",
                )
            )

    # Set default labels
    if x_label is None:
        x_label = frequency.capitalize()
    if y_label is None:
        y_label = "Value"

    # Update x-axis with categorical labels
    if season_labels:
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(1, len(season_labels) + 1)),
            ticktext=season_labels,
        )

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
