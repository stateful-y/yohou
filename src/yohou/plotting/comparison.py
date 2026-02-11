"""Comparison and forecast plotting functions."""

import numpy as np
import polars as pl
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from yohou.plotting.colors import get_color_sequence
from yohou.plotting.plotly_utils import apply_default_layout
from yohou.plotting.prep import resolve_columns, validate_dataframe


def plot_residuals(
    df: pl.DataFrame,
    *,
    residuals_column: str = "residuals",
    fitted_column: str | None = "fitted",
    panel_group_name: str | None = None,
    facet_ncol: int = 2,  # noqa: ARG001
    dropdown: bool = False,  # noqa: ARG001
    color_palette: list[str] | None = None,  # noqa: ARG001
    title: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,  # noqa: ARG001
) -> go.Figure:
    """
    Plot diagnostic plots for model residuals.

    Creates a 4-panel layout with residuals over time, residuals vs fitted values,
    histogram of residuals, and Q-Q plot for checking normality assumptions.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and residuals column.
    residuals_column : str, default="residuals"
        Name of the column containing residuals.
    fitted_column : str | None, default="fitted"
        Name of the column containing fitted values. If None, residuals vs fitted plot is skipped.
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
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    **kwargs : dict
        Additional styling parameters (reserved for future use).

    Returns
    -------
    go.Figure
        Plotly figure object with 4 subplots.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_residuals

    >>> # Create sample data with residuals
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
    ...     "y": [100 + i for i in range(91)],
    ...     "fitted": [100 + i + (i % 3) for i in range(91)],
    ... }).with_columns((pl.col("y") - pl.col("fitted")).alias("residuals"))

    >>> # Plot residual diagnostics
    >>> fig = plot_residuals(df)
    >>> len(fig.data) > 0
    True

    See Also
    --------
    plot_forecast : Plot forecasts with historical data.
    """
    # Validate inputs
    validate_dataframe(df)

    if panel_group_name is not None:
        msg = "Panel grouping not yet implemented"
        raise NotImplementedError(msg)

    if residuals_column not in df.columns:
        msg = f"Residuals column '{residuals_column}' not found in DataFrame"
        raise ValueError(msg)

    # Create subplots
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

    residuals = df[residuals_column].to_numpy()

    # Panel 1: Residuals over time
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=residuals,
            mode="markers",
            marker={"size": 4, "color": "#2563EB", "opacity": 0.6},
            name="Residuals",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line={"dash": "dash", "color": "#DC2626", "width": 1}, row=1, col=1)

    # Panel 2: Residuals vs Fitted
    if fitted_column and fitted_column in df.columns:
        fitted = df[fitted_column].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=fitted,
                y=residuals,
                mode="markers",
                marker={"size": 4, "color": "#7C3AED", "opacity": 0.6},
                name="Residuals vs Fitted",
            ),
            row=1,
            col=2,
        )
        fig.add_hline(y=0, line={"dash": "dash", "color": "#DC2626", "width": 1}, row=1, col=2)

    # Panel 3: Histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            marker={"color": "#059669", "opacity": 0.7},
            name="Histogram",
        ),
        row=2,
        col=1,
    )

    # Panel 4: Q-Q Plot
    # Sort residuals
    sorted_residuals = np.sort(residuals)
    n = len(sorted_residuals)

    # Theoretical quantiles (normal distribution)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))

    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_residuals,
            mode="markers",
            marker={"size": 4, "color": "#EA580C", "opacity": 0.6},
            name="Q-Q Plot",
        ),
        row=2,
        col=2,
    )

    # Add reference line for Q-Q plot
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

    # Update axis labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)

    if fitted_column and fitted_column in df.columns:
        fig.update_xaxes(title_text="Fitted Values", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)

    fig.update_xaxes(title_text="Residuals", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)

    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)

    # Update layout
    title_default = "Residual Diagnostics" if title is None else title
    fig.update_layout(
        title=title_default,
        showlegend=False,
        height=height or 600,
        width=width or 900,
    )

    return fig


def plot_forecast(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    forecast_column: str | None = "forecast",
    is_forecast_column: str = "is_forecast",
    lower_bound_column: str | None = "lower_bound",
    upper_bound_column: str | None = "upper_bound",
    n_history: int | None = None,
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
    Plot forecasts with historical data and prediction intervals.

    Visualizes historical observations and forecasts, optionally showing prediction
    intervals and limiting the number of historical observations displayed.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column, containing both historical and forecast data.
    columns : str | list[str] | None, default=None
        Column(s) to plot. If None, uses all numeric columns except 'time'.
    forecast_column : str | None, default="forecast"
        Name of the column containing forecast values. If None, uses main columns.
    is_forecast_column : str, default="is_forecast"
        Boolean column indicating forecast rows (True) vs historical (False).
    lower_bound_column : str | None, default="lower_bound"
        Name of the lower prediction interval column. If None, no intervals shown.
    upper_bound_column : str | None, default="upper_bound"
        Name of the upper prediction interval column. If None, no intervals shown.
    n_history : int | None, default=None
        Number of historical observations to show. If None, shows all.
    panel_group_name : str | None, default=None
        Column name for grouping (panel data).
    facet_ncol : int, default=2
        Number of columns in facet grid.
    facet_scales : str, default="free_y"
        Scale type for facets.
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
        - history_color : str, default="#2563EB"
        - forecast_color : str, default="#DC2626"
        - line_width : float, default=2.0
        - band_alpha : float, default=0.15
        - show_transition : bool, default=True

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_forecast

    >>> # Create combined historical + forecast data
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 30), "1d", eager=True),
    ...     "y": [100 + i for i in range(121)],
    ...     "is_forecast": [False] * 91 + [True] * 30,
    ... })

    >>> # Plot forecast
    >>> fig = plot_forecast(df, n_history=30)
    >>> len(fig.data) > 0
    True

    See Also
    --------
    plot_prediction_interval : Plot prediction intervals.
    plot_residuals : Plot residual diagnostics.
    """
    # Validate inputs
    validate_dataframe(df)

    if panel_group_name is not None:
        msg = "Panel grouping not yet implemented"
        raise NotImplementedError(msg)

    if is_forecast_column not in df.columns:
        msg = f"Column '{is_forecast_column}' not found in DataFrame"
        raise ValueError(msg)

    # Get kwargs
    history_color = kwargs.get("history_color", "#2563EB")
    forecast_color = kwargs.get("forecast_color", "#DC2626")
    line_width = kwargs.get("line_width", 2.0)
    band_alpha = kwargs.get("band_alpha", 0.15)
    show_transition = kwargs.get("show_transition", True)

    # Limit history if requested
    if n_history:
        forecast_start_idx = df[is_forecast_column].arg_max()
        history_start_idx = max(0, forecast_start_idx - n_history)
        df = df[history_start_idx:]

    # Resolve columns
    exclude_cols = ["time", is_forecast_column]
    if lower_bound_column:
        exclude_cols.append(lower_bound_column)
    if upper_bound_column:
        exclude_cols.append(upper_bound_column)
    if forecast_column:
        exclude_cols.append(forecast_column)

    plot_columns = resolve_columns(df, columns=columns, exclude=exclude_cols)

    # Create figure
    fig = go.Figure()

    for col in plot_columns:
        # Split into historical and forecast
        df_hist = df.filter(~pl.col(is_forecast_column))
        df_forecast = df.filter(pl.col(is_forecast_column))

        # Plot historical data
        if len(df_hist) > 0:
            fig.add_trace(
                go.Scatter(
                    x=df_hist["time"],
                    y=df_hist[col],
                    mode="lines",
                    line={"color": history_color, "width": line_width},
                    name=f"{col} (Historical)",
                )
            )

        # Plot forecast data
        if len(df_forecast) > 0:
            forecast_y = (
                df_forecast[forecast_column]
                if forecast_column and forecast_column in df_forecast.columns
                else df_forecast[col]
            )

            # Add transition point if requested
            x_forecast = df_forecast["time"]
            if show_transition and len(df_hist) > 0:
                # Include last historical point for smooth transition
                x_forecast = pl.concat([df_hist["time"][-1:], df_forecast["time"]])
                last_hist_val = df_hist[col][-1]
                forecast_y = pl.concat([pl.Series([last_hist_val]), forecast_y])

            fig.add_trace(
                go.Scatter(
                    x=x_forecast,
                    y=forecast_y,
                    mode="lines",
                    line={"color": forecast_color, "width": line_width},
                    name=f"{col} (Forecast)",
                )
            )

            # Add prediction intervals if available
            if (
                lower_bound_column
                and upper_bound_column
                and lower_bound_column in df_forecast.columns
                and upper_bound_column in df_forecast.columns
            ):
                # Convert hex to rgba
                rgb = tuple(int(forecast_color[i : i + 2], 16) for i in (1, 3, 5))
                rgba_color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {band_alpha})"

                x_bounds = x_forecast if show_transition else df_forecast["time"]
                upper_vals = df_forecast[upper_bound_column]
                lower_vals = df_forecast[lower_bound_column]

                if show_transition and len(df_hist) > 0:
                    last_val = df_hist[col][-1]
                    # Cast to match series dtype
                    upper_vals = pl.concat([pl.Series([last_val], dtype=upper_vals.dtype), upper_vals])
                    lower_vals = pl.concat([pl.Series([last_val], dtype=lower_vals.dtype), lower_vals])

                # Add ribbon
                fig.add_trace(
                    go.Scatter(
                        x=x_bounds,
                        y=upper_vals,
                        mode="lines",
                        line={"width": 0},
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_bounds,
                        y=lower_vals,
                        fill="tonexty",
                        fillcolor=rgba_color,
                        mode="lines",
                        line={"width": 0},
                        name=f"{col} (Interval)",
                    )
                )

    # Set default labels
    title_default = "Forecast" if title is None else title
    x_label_default = "Time" if x_label is None else x_label
    y_label_default = "Value" if y_label is None else y_label

    fig = apply_default_layout(
        fig,
        title=title_default,
        x_label=x_label_default,
        y_label=y_label_default,
        width=width,
        height=height,
    )

    return fig


def plot_comparison(
    df: pl.DataFrame,
    *,
    columns: list[str],
    comparison_mode: str = "overlay",
    reference_column: str | None = None,
    panel_group_name: str | None = None,
    facet_ncol: int = 2,  # noqa: ARG001
    color_palette: list[str] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """
    Compare multiple time series or model outputs.

    Supports three comparison modes: overlay (all series on same plot),
    facet (separate subplots), or difference (show differences from reference).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and comparison columns.
    columns : list[str]
        List of column names to compare (required).
    comparison_mode : str, default="overlay"
        Comparison mode: "overlay", "facet", or "difference".
    reference_column : str | None, default=None
        Reference column for "difference" mode. If None, uses first column.
    panel_group_name : str | None, default=None
        Column name for grouping (panel data).
    facet_ncol : int, default=2
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
        - line_width : float, default=2.0

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_comparison

    >>> # Create DataFrame with multiple series
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
    ...     "actual": [100 + i for i in range(91)],
    ...     "model1": [100 + i + (i % 5) for i in range(91)],
    ...     "model2": [100 + i + (i % 3) for i in range(91)],
    ... })

    >>> # Compare with overlay mode
    >>> fig = plot_comparison(df, columns=["actual", "model1", "model2"])
    >>> len(fig.data) >= 3
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

    if not columns or len(columns) == 0:
        msg = "Must provide at least one column to compare"
        raise ValueError(msg)

    # Get kwargs
    line_width = kwargs.get("line_width", 2.0)

    # Get color sequence
    colors = color_palette if color_palette else get_color_sequence(len(columns))

    # Create figure based on mode
    if comparison_mode == "overlay":
        fig = go.Figure()

        for col_idx, col in enumerate(columns):
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df[col],
                    mode="lines",
                    line={"width": line_width, "color": colors[col_idx]},
                    name=col,
                )
            )

        title_default = "Series Comparison" if title is None else title
        y_label_default = "Value" if y_label is None else y_label

    elif comparison_mode == "facet":
        n_cols = min(facet_ncol, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=columns,
        )

        for idx, col in enumerate(columns):
            row = idx // n_cols + 1
            col_pos = idx % n_cols + 1

            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df[col],
                    mode="lines",
                    line={"width": line_width, "color": colors[idx]},
                    name=col,
                    showlegend=False,
                ),
                row=row,
                col=col_pos,
            )

        title_default = "Series Comparison (Faceted)" if title is None else title
        y_label_default = "Value" if y_label is None else y_label

    elif comparison_mode == "difference":
        ref_col = reference_column if reference_column else columns[0]
        if ref_col not in df.columns:
            msg = f"Reference column '{ref_col}' not found in DataFrame"
            raise ValueError(msg)

        fig = go.Figure()

        for col_idx, col in enumerate(columns):
            if col == ref_col:
                continue

            difference = df[col] - df[ref_col]
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=difference,
                    mode="lines",
                    line={"width": line_width, "color": colors[col_idx]},
                    name=f"{col} - {ref_col}",
                )
            )

        fig.add_hline(y=0, line={"dash": "dash", "color": "#64748B", "width": 1})

        title_default = f"Difference from {ref_col}" if title is None else title
        y_label_default = "Difference" if y_label is None else y_label

    else:
        msg = f"Unknown comparison_mode: {comparison_mode}. Use 'overlay', 'facet', or 'difference'."
        raise ValueError(msg)

    # Set default labels
    x_label_default = "Time" if x_label is None else x_label

    fig = apply_default_layout(
        fig,
        title=title_default,
        x_label=x_label_default,
        y_label=y_label_default,
        width=width,
        height=height,
    )

    return fig
