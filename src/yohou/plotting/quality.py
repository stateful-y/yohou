"""Data quality visualization functions."""

from typing import Literal

import plotly.graph_objects as go
import polars as pl

from yohou.plotting.plotly_utils import apply_default_layout
from yohou.plotting.prep import resolve_columns, validate_dataframe


def plot_missing_data(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    method: Literal["heatmap", "bars", "matrix"] = "heatmap",
    panel_group_name: str | None = None,
    facet_ncol: int = 2,  # noqa: ARG001
    dropdown: bool = False,  # noqa: ARG001
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
    method : {"heatmap", "bars", "matrix"}, default="heatmap"
        Visualization method:
        - "heatmap": time × columns grid showing missing/present
        - "bars": bar chart of missing percentage per column
        - "matrix": binary matrix (missingno-style)
    panel_group_name : str | None, default=None
        Column name for grouping (panel data).
    facet_ncol : int, default=2
        Number of columns in facet grid.
    dropdown : bool, default=False
        Use dropdown menu instead of facets.
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
    >>> fig = plot_missing_data(df, method="bars")
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
    kwargs.get("show_percentages", True)
    time_aggregation = kwargs.get("time_aggregation")

    if method == "bars":
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
                hovertemplate="<b>%{x}</b><br>Missing: %{y:.1f}%<extra></extra>",
            )
        )

        if x_label is None:
            x_label = "Column"
        if y_label is None:
            y_label = "Missing (%)"

    elif method == "heatmap":
        # Heatmap of missing values over time
        # Create binary matrix: 1 = missing, 0 = present
        df.select(plot_columns).null_count().to_numpy()[0]

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
        fig.add_trace(
            go.Heatmap(
                z=z_data,
                x=x_vals,
                y=plot_columns,
                colorscale=[[0, color_present], [1, color_missing]],
                showscale=False,
                hovertemplate="<b>%{y}</b><br>%{x}<br>Missing: %{z}<extra></extra>",
            )
        )

        if x_label is None:
            x_label = "Time"
        if y_label is None:
            y_label = "Column"

    elif method == "matrix":
        # Binary matrix visualization
        z_data = []
        for col in plot_columns:
            col_data = df[col].is_null().cast(pl.Int8).to_list()
            z_data.append(col_data)

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=z_data,
                x=list(range(len(df))),
                y=plot_columns,
                colorscale=[[0, color_present], [1, color_missing]],
                showscale=False,
                hovertemplate="<b>%{y}</b><br>Index: %{x}<br>Missing: %{z}<extra></extra>",
            )
        )

        if x_label is None:
            x_label = "Index"
        if y_label is None:
            y_label = "Column"
    else:
        msg = f"Unknown method: {method}. Valid options: heatmap, bars, matrix"
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
