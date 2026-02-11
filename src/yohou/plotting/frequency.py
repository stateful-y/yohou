"""Frequency domain analysis plotting functions."""

import numpy as np
import polars as pl
from plotly import graph_objects as go
from scipy.signal import periodogram as scipy_periodogram

from yohou.plotting.colors import get_color_sequence
from yohou.plotting.plotly_utils import apply_default_layout
from yohou.plotting.prep import resolve_columns, validate_dataframe


def plot_lag_scatter(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    lags: int | list[int] = 1,
    panel_group_name: str | None = None,
    facet_ncol: int = 3,  # noqa: ARG001
    dropdown: bool = False,  # noqa: ARG001
    color_palette: list[str] | None = None,
    title: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """
    Plot scatter plots of y(t) vs y(t-lag) for analyzing temporal dependencies.

    Creates scatter plots showing the relationship between current values and
    lagged values, useful for identifying AR patterns and lag-specific correlations.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to analyze. If None, uses all numeric columns except 'time'.
    lags : int | list[int], default=1
        Lag values to plot. Can be single lag or list of lags.
    panel_group_name : str | None, default=None
        Column name for grouping (panel data).
    facet_ncol : int, default=3
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
        Additional styling parameters:
        - marker_size : float, default=4.0
        - marker_alpha : float, default=0.6
        - show_diagonal : bool, default=True
        - show_regression : bool, default=False

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_lag_scatter

    >>> # Create sample time series
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 31), "1d", eager=True),
    ...     "y": [100 + i % 10 for i in range(31)],
    ... })

    >>> # Plot lag-1 scatter
    >>> fig = plot_lag_scatter(df, columns="y", lags=1)
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

    # Normalize lags to list
    lag_list = [lags] if isinstance(lags, int) else lags

    # Get kwargs
    marker_size = kwargs.get("marker_size", 4.0)
    marker_alpha = kwargs.get("marker_alpha", 0.6)
    show_diagonal = kwargs.get("show_diagonal", True)
    show_regression = kwargs.get("show_regression", False)

    # Get color sequence
    colors = color_palette if color_palette else get_color_sequence(len(plot_columns))

    # Create figure
    fig = go.Figure()

    # Create subplots for each lag
    for lag_idx, lag in enumerate(lag_list):
        for col_idx, col in enumerate(plot_columns):
            # Create lagged column
            df_lagged = df.with_columns([pl.col(col).shift(lag).alias(f"{col}_lag_{lag}")])

            # Drop nulls from lagging
            df_lagged = df_lagged.drop_nulls()

            y_current = df_lagged[col]
            y_lagged = df_lagged[f"{col}_lag_{lag}"]

            # Add scatter trace
            fig.add_trace(
                go.Scatter(
                    x=y_lagged,
                    y=y_current,
                    mode="markers",
                    marker={"size": marker_size, "color": colors[col_idx], "opacity": marker_alpha},
                    name=f"{col} (lag={lag})",
                    hovertemplate=f"<b>{col}</b><br>y(t-{lag}): %{{x:.2f}}<br>y(t): %{{y:.2f}}<extra></extra>",
                )
            )

            # Add diagonal line if requested
            if show_diagonal and lag_idx == 0 and col_idx == 0:
                min_val = min(y_lagged.min(), y_current.min())
                max_val = max(y_lagged.max(), y_current.max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        line={"dash": "dash", "color": "#94a3b8", "width": 1},
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

            # Add regression line if requested
            if show_regression:
                # Simple linear regression
                x_mean = y_lagged.mean()
                y_mean = y_current.mean()

                numerator = ((y_lagged - x_mean) * (y_current - y_mean)).sum()
                denominator = ((y_lagged - x_mean) ** 2).sum()

                if denominator != 0:
                    slope = numerator / denominator
                    intercept = y_mean - slope * x_mean

                    x_line = [y_lagged.min(), y_lagged.max()]
                    y_line = [slope * x + intercept for x in x_line]

                    fig.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            line={"color": colors[col_idx], "width": 2},
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

    # Set default labels
    if title is None and len(lag_list) == 1:
        title = f"Lag {lag_list[0]} Scatter Plot"

    x_label_default = f"y(t-{lag_list[0]})" if len(lag_list) == 1 else "y(t-lag)"
    y_label_default = "y(t)"

    # Apply default layout
    fig = apply_default_layout(
        fig,
        title=title,
        x_label=x_label_default,
        y_label=y_label_default,
        width=width,
        height=height,
    )

    return fig


def plot_periodogram(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    detrend: str = "linear",
    log_scale: bool = False,
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
    Plot periodogram for frequency domain analysis of time series.

    Creates a periodogram showing the power spectral density, useful for identifying
    dominant frequencies and periodic patterns in the data.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to analyze. If None, uses all numeric columns except 'time'.
    detrend : str, default="linear"
        Detrending method: "none", "mean", or "linear".
    log_scale : bool, default=False
        Use logarithmic scale for power/amplitude axis.
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
        - line_width : float, default=1.5
        - show_peaks : bool, default=False
        - n_peaks : int, default=3

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> from yohou.plotting import plot_periodogram

    >>> # Create sample time series with periodic component
    >>> t = np.arange(100)
    >>> y = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.25 * t)
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
    ...     "y": y,
    ... })

    >>> # Plot periodogram
    >>> fig = plot_periodogram(df, columns="y")
    >>> len(fig.data) > 0
    True

    See Also
    --------
    plot_lag_scatter : Plot lag scatter plots.
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
    line_width = kwargs.get("line_width", 1.5)
    show_peaks = kwargs.get("show_peaks", False)
    n_peaks = kwargs.get("n_peaks", 3)

    # Get color sequence
    colors = color_palette if color_palette else get_color_sequence(len(plot_columns))

    # Create figure
    fig = go.Figure()

    for col_idx, col in enumerate(plot_columns):
        # Get column values
        y = df[col].to_numpy()

        # Apply detrending
        if detrend == "mean":
            y = y - y.mean()
        elif detrend == "linear":
            # Simple linear detrending
            x = range(len(y))
            x_mean = sum(x) / len(x)
            y_mean = y.mean()
            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(y)))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(len(y)))
            if denominator != 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
                y = y - (slope * range(len(y)) + intercept)

        # Compute periodogram
        freqs, power = scipy_periodogram(y)

        # Skip zero frequency
        freqs = freqs[1:]
        power = power[1:]

        # Add trace
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=power,
                mode="lines",
                line={"width": line_width, "color": colors[col_idx]},
                name=col,
                hovertemplate=f"<b>{col}</b><br>Frequency: %{{x:.4f}}<br>Power: %{{y:.4f}}<extra></extra>",
            )
        )

        # Add peak markers if requested
        if show_peaks and n_peaks > 0:
            # Find n largest peaks
            peak_indices = np.argsort(power)[-n_peaks:][::-1]
            peak_freqs = freqs[peak_indices]
            peak_powers = power[peak_indices]

            fig.add_trace(
                go.Scatter(
                    x=peak_freqs,
                    y=peak_powers,
                    mode="markers",
                    marker={"size": 8, "color": colors[col_idx], "symbol": "diamond"},
                    name=f"{col} (peaks)",
                    hovertemplate=f"<b>{col} Peak</b><br>Frequency: %{{x:.4f}}<br>Power: %{{y:.4f}}<extra></extra>",
                )
            )

    # Set default labels
    title_default = "Periodogram" if title is None else title
    x_label_default = "Frequency" if x_label is None else x_label
    y_label_default = "Power" if y_label is None else y_label

    # Apply default layout
    fig = apply_default_layout(
        fig,
        title=title_default,
        x_label=x_label_default,
        y_label=y_label_default,
        width=width,
        height=height,
    )

    # Apply log scale if requested
    if log_scale:
        fig.update_yaxes(type="log")

    return fig
