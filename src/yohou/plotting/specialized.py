"""Specialized plotting functions."""

import math

import numpy as np
import polars as pl
from plotly import graph_objects as go
from scipy.stats import norm

from yohou.plotting.plotly_utils import apply_default_layout
from yohou.plotting.prep import validate_dataframe


def plot_cross_correlation(
    df: pl.DataFrame,
    *,
    x_column: str,
    y_column: str,
    lags: int = 40,
    alpha: float = 0.05,
    title: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """
    Plot cross-correlation function (CCF) between two time series.

    Computes and visualizes the cross-correlation between two series at various lags,
    useful for identifying lead-lag relationships and temporal dependencies.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    x_column : str
        Name of the first column (predictor/independent variable).
    y_column : str
        Name of the second column (response/dependent variable).
    lags : int, default=40
        Number of lags to compute (both positive and negative).
    alpha : float, default=0.05
        Significance level for confidence bands.
    title : str | None, default=None
        Plot title.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    **kwargs : dict
        Additional styling parameters:
        - show_markers : bool, default=True
        - marker_size : float, default=6.0
        - marker_color : str, default="#2563EB"
        - line_color : str, default="#94a3b8"

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_cross_correlation

    >>> # Create two time series with lag relationship
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
    ...     "x": [100 + i for i in range(91)],
    ...     "y": [105 + i for i in range(91)],  # y leads x by 5 units
    ... })

    >>> # Plot cross-correlation
    >>> fig = plot_cross_correlation(df, x_column="x", y_column="y", lags=20)
    >>> len(fig.data) > 0
    True

    See Also
    --------
    plot_autocorrelation : Plot autocorrelation function.
    """
    # Validate inputs
    validate_dataframe(df)

    if x_column not in df.columns:
        msg = f"Column '{x_column}' not found in DataFrame"
        raise ValueError(msg)
    if y_column not in df.columns:
        msg = f"Column '{y_column}' not found in DataFrame"
        raise ValueError(msg)

    # Get kwargs
    show_markers = kwargs.get("show_markers", True)
    marker_size = kwargs.get("marker_size", 6.0)
    marker_color = kwargs.get("marker_color", "#2563EB")
    line_color = kwargs.get("line_color", "#94a3b8")

    # Get series
    x = df[x_column].to_numpy()
    y = df[y_column].to_numpy()
    n = len(x)

    # Compute cross-correlation for each lag
    lag_values = list(range(-lags, lags + 1))
    ccf_values = []

    # Normalize series
    x_mean = x.mean()
    y_mean = y.mean()
    x_std = x.std()
    y_std = y.std()

    for lag in lag_values:
        if lag < 0:
            # Negative lag: x leads y
            x_shifted = x[: n + lag]
            y_shifted = y[-lag:]
        elif lag > 0:
            # Positive lag: y leads x
            x_shifted = x[lag:]
            y_shifted = y[: n - lag]
        else:
            # Zero lag
            x_shifted = x
            y_shifted = y

        # Compute correlation
        if len(x_shifted) > 0 and x_std != 0 and y_std != 0:
            correlation = ((x_shifted - x_mean) * (y_shifted - y_mean)).mean() / (x_std * y_std)
            ccf_values.append(correlation)
        else:
            ccf_values.append(0.0)

    # Create figure
    fig = go.Figure()

    # Add stem plot
    if show_markers:
        # Add vertical lines
        for lag, ccf in zip(lag_values, ccf_values, strict=True):
            fig.add_trace(
                go.Scatter(
                    x=[lag, lag],
                    y=[0, ccf],
                    mode="lines",
                    line={"color": line_color, "width": 2},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Add markers
        fig.add_trace(
            go.Scatter(
                x=lag_values,
                y=ccf_values,
                mode="markers",
                marker={"size": marker_size, "color": marker_color},
                name="CCF",
                hovertemplate="<b>Lag %{x}</b><br>CCF: %{y:.3f}<extra></extra>",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=lag_values,
                y=ccf_values,
                mode="lines+markers",
                line={"color": marker_color, "width": 2},
                marker={"size": marker_size, "color": marker_color},
                name="CCF",
                hovertemplate="<b>Lag %{x}</b><br>CCF: %{y:.3f}<extra></extra>",
            )
        )

    # Add confidence bands
    z_value = 1.96  # For 95% confidence (alpha=0.05)
    if alpha != 0.05:
        z_value = norm.ppf(1 - alpha / 2)

    confidence_band = z_value / math.sqrt(n)

    fig.add_hline(
        y=confidence_band,
        line={"dash": "dash", "color": "#DC2626", "width": 1},
        annotation_text="95% CI",
        annotation_position="right",
    )
    fig.add_hline(
        y=-confidence_band,
        line={"dash": "dash", "color": "#DC2626", "width": 1},
    )
    fig.add_hline(y=0, line={"color": "#64748B", "width": 1})

    # Set default labels
    title_default = f"Cross-Correlation: {x_column} vs {y_column}" if title is None else title

    fig = apply_default_layout(
        fig,
        title=title_default,
        x_label="Lag",
        y_label="Cross-Correlation",
        width=width,
        height=height,
    )

    return fig


def plot_calendar_heatmap(
    df: pl.DataFrame,
    *,
    column: str,
    aggregation: str = "sum",
    year: int | None = None,
    colorscale: str = "Blues",
    panel_group_name: str | None = None,
    facet_ncol: int = 1,  # noqa: ARG001
    title: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """
    Plot calendar-style heatmap for daily time series data.

    Creates a calendar heatmap showing values for each day of the year,
    organized by weeks and months, useful for identifying daily patterns and seasonality.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column (daily frequency) and value column.
    column : str
        Name of the column to visualize.
    aggregation : str, default="sum"
        Aggregation method: "sum", "mean", "median", "max", or "min".
    year : int | None, default=None
        Specific year to visualize. If None, uses all available data.
    colorscale : str, default="Blues"
        Plotly colorscale name.
    panel_group_name : str | None, default=None
        Column name for grouping (panel data).
    facet_ncol : int, default=1
        Number of columns in facet grid.
    title : str | None, default=None
        Plot title.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    **kwargs : dict
        Additional styling parameters:
        - show_values : bool, default=False
        - missing_color : str, default="#f0f0f0"

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_calendar_heatmap

    >>> # Create daily data for a year
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
    ...     "sales": [100 + (i % 50) for i in range(366)],
    ... })

    >>> # Plot calendar heatmap
    >>> fig = plot_calendar_heatmap(df, column="sales", year=2020)
    >>> len(fig.data) > 0
    True

    See Also
    --------
    plot_seasonality : Plot seasonal patterns.
    plot_missing_data : Visualize missing data patterns.
    """
    # Validate inputs
    validate_dataframe(df)

    if panel_group_name is not None:
        msg = "Panel grouping not yet implemented"
        raise NotImplementedError(msg)

    if column not in df.columns:
        msg = f"Column '{column}' not found in DataFrame"
        raise ValueError(msg)

    # Get kwargs
    show_values = kwargs.get("show_values", False)

    # Filter by year if specified
    if year:
        df = df.filter(pl.col("time").dt.year() == year)

    if len(df) == 0:
        msg = f"No data available for year {year}"
        raise ValueError(msg)

    # Extract date components
    df = df.with_columns([
        pl.col("time").dt.year().alias("year"),
        pl.col("time").dt.month().alias("month"),
        pl.col("time").dt.day().alias("day"),
        pl.col("time").dt.weekday().alias("weekday"),
        pl.col("time").dt.week().alias("week"),
    ])

    # Aggregate if needed
    if aggregation == "sum":
        df_agg = df.group_by(["year", "month", "day", "weekday", "week"]).agg(pl.col(column).sum())
    elif aggregation == "mean":
        df_agg = df.group_by(["year", "month", "day", "weekday", "week"]).agg(pl.col(column).mean())
    elif aggregation == "median":
        df_agg = df.group_by(["year", "month", "day", "weekday", "week"]).agg(pl.col(column).median())
    elif aggregation == "max":
        df_agg = df.group_by(["year", "month", "day", "weekday", "week"]).agg(pl.col(column).max())
    elif aggregation == "min":
        df_agg = df.group_by(["year", "month", "day", "weekday", "week"]).agg(pl.col(column).min())
    else:
        msg = f"Unknown aggregation: {aggregation}"
        raise ValueError(msg)

    # Pivot data for heatmap: weeks × weekdays
    weeks = df_agg["week"].to_list()
    weekdays = df_agg["weekday"].to_list()
    values = df_agg[column].to_list()

    # Create matrix
    n_weeks = max(weeks) if weeks else 1
    matrix = np.full((7, n_weeks), np.nan)

    for week, weekday, value in zip(weeks, weekdays, values, strict=True):
        # weekday: Polars returns 1=Monday, 7=Sunday; convert to 0-based index
        matrix[weekday - 1, week - 1] = value

    # Create heatmap
    fig = go.Figure()

    # Add heatmap trace
    fig.add_trace(
        go.Heatmap(
            z=matrix,
            x=list(range(1, n_weeks + 1)),
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            colorscale=colorscale,
            text=matrix if show_values else None,
            texttemplate="%{text:.0f}" if show_values else None,
            hovertemplate="Week %{x}<br>%{y}<br>Value: %{z:.2f}<extra></extra>",
        )
    )

    # Set default labels
    year_str = str(year) if year else "All Years"
    title_default = f"Calendar Heatmap - {year_str}" if title is None else title

    fig.update_layout(
        title=title_default,
        xaxis_title="Week",
        yaxis_title="Day of Week",
        width=width or 1000,
        height=height or 300,
    )

    return fig
