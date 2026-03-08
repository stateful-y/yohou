"""Signal processing plotting functions for time series analysis."""

from typing import Literal

import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy.signal import periodogram as scipy_periodogram

from yohou.plotting._utils import (
    apply_default_layout,
    panel_facet_figure,
    resolve_color_palette,
)
from yohou.utils import validate_plotting_data
from yohou.utils.panel import inspect_panel

__all__ = [
    "plot_phase",
    "plot_spectrum",
]


def plot_phase(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    unwrap: bool = True,
    angle_unit: Literal["degree", "radian"] = "radian",
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
    """Plot the phase of a time series.

    Shows the phase angle of each frequency component computed via FFT.
    Useful for understanding temporal alignment of periodic patterns.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to analyze. If None, uses all numeric columns except 'time'.
    unwrap : bool, default=True
        Unwrap phase angles to avoid discontinuities at :math:`\\pm\\pi`.
    angle_unit : Literal["degree", "radian"], default="radian"
        Unit for the phase angle axis.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot.
    facet_n_cols : int, default=2
        Number of columns in facet grid.
    color_palette : list[str] | None, default=None
        Custom color palette.
    title : str | None, default=None
        Plot title.
    x_label : str | None, default=None
        X-axis label.  Defaults to "Frequency (cycles/sample)".
    y_label : str | None, default=None
        Y-axis label.  Defaults to "Phase (radians)" or "Phase (degrees)".
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    **kwargs : dict
        Additional styling parameters:
        - line_width : float, default=1.5

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> from yohou.plotting.signal import plot_phase

    >>> t = np.arange(100)
    >>> y = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.25 * t)
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
    ...     "y": y,
    ... })

    >>> fig = plot_phase(df, columns="y")
    >>> len(fig.data) > 0
    True

    See Also
    --------
    `plot_spectrum` : Plot power spectral density.
    """
    use_degrees = angle_unit == "degree"
    validate_plotting_data(df)

    # Auto-detect panel data
    _, _panel_groups = inspect_panel(df)
    if panel_group_names is None and columns is None and _panel_groups:
        panel_group_names = []

    if panel_group_names is not None:

        def _render_phase(
            fig: go.Figure,
            sub_df: pl.DataFrame,
            display_name: str,  # noqa: ARG001
            panel_idx: int,  # noqa: ARG001
            row: int,
            col: int,
        ) -> None:
            """Render phase spectrum for a single panel group."""
            base = [c for c in sub_df.columns if c != "time"][0]
            _lw = kwargs.get("line_width", 1.5)
            y_arr = sub_df[base].to_numpy().astype(float)
            spectrum = np.fft.rfft(y_arr)
            freqs = np.fft.rfftfreq(len(y_arr))
            phase = np.angle(spectrum)
            if unwrap:
                phase = np.unwrap(phase)
            if use_degrees:
                phase = np.degrees(phase)
            fig.add_trace(
                go.Scatter(
                    x=freqs[1:].tolist(),
                    y=phase[1:].tolist(),
                    mode="lines",
                    line={"width": _lw},
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        unit = "degrees" if use_degrees else "radians"
        return panel_facet_figure(
            df,
            _render_phase,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_n_cols=facet_n_cols,
            title=title or "Phase Spectrum (Panel)",
            x_label=x_label or "Frequency (cycles/sample)",
            y_label=y_label or f"Phase ({unit})",
            width=width,
            height=height,
            shared_xaxes=False,
        )

    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])
    colors = resolve_color_palette(color_palette, len(plot_columns))
    line_width = kwargs.get("line_width", 1.5)

    fig = go.Figure()

    for col_idx, col_name in enumerate(plot_columns):
        y = df[col_name].to_numpy().astype(float)

        spectrum = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(y))
        phase = np.angle(spectrum)
        if unwrap:
            phase = np.unwrap(phase)
        if use_degrees:
            phase = np.degrees(phase)

        # Skip DC component
        freqs = freqs[1:]
        phase = phase[1:]

        unit_label = "°" if use_degrees else "rad"
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=phase,
                mode="lines",
                line={"width": line_width, "color": colors[col_idx]},
                name=col_name,
                hovertemplate=(
                    f"<b>{col_name}</b><br>Frequency: %{{x:.4f}}<br>Phase: %{{y:.2f}} {unit_label}<extra></extra>"
                ),
            )
        )

    unit = "degrees" if use_degrees else "radians"
    fig = apply_default_layout(
        fig,
        title=title or "Phase Spectrum",
        x_label=x_label or "Frequency (cycles/sample)",
        y_label=y_label or f"Phase ({unit})",
        width=width,
        height=height,
    )

    return fig


def plot_spectrum(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    log_scale: bool = False,
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
    """Plot periodogram (power spectral density) for frequency domain analysis.

    Creates a periodogram showing the power spectral density via FFT, useful
    for identifying dominant frequencies and periodic patterns in the data.
    Hover text always includes the corresponding period (1/frequency) and
    detected peaks are annotated with their period in sample units.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to analyze. If None, uses all numeric columns except 'time'.
    log_scale : bool, default=False
        Use logarithmic scale for PSD axis.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot.
    facet_n_cols : int, default=2
        Number of columns in facet grid.
    color_palette : list[str] | None, default=None
        Custom color palette.
    title : str | None, default=None
        Plot title.
    x_label : str | None, default=None
        X-axis label.  Defaults to "Frequency (cycles/sample)".
    y_label : str | None, default=None
        Y-axis label.  Defaults to "Power Spectral Density".
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
    >>> from yohou.plotting import plot_spectrum

    >>> # Create sample time series with periodic component
    >>> t = np.arange(100)
    >>> y = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.25 * t)
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
    ...     "y": y,
    ... })

    >>> # Plot spectrum
    >>> fig = plot_spectrum(df, columns="y")
    >>> len(fig.data) > 0
    True

    See Also
    --------
    `plot_phase` : Plot phase spectrum.
    """
    # Validate inputs
    validate_plotting_data(df)

    # Auto-detect panel data
    _, _panel_groups = inspect_panel(df)
    if panel_group_names is None and columns is None and _panel_groups:
        panel_group_names = []

    if panel_group_names is not None:

        def _render_periodogram(
            fig: go.Figure,
            sub_df: pl.DataFrame,
            display_name: str,  # noqa: ARG001
            panel_idx: int,  # noqa: ARG001
            row: int,
            col: int,
        ) -> None:
            """Render spectral periodogram with optional log scaling for a single column."""
            base = [c for c in sub_df.columns if c != "time"][0]
            _lw = kwargs.get("line_width", 1.5)
            y_arr = sub_df[base].to_numpy()
            freqs, psd = scipy_periodogram(y_arr)
            fig.add_trace(
                go.Scatter(
                    x=freqs[1:].tolist(),
                    y=psd[1:].tolist(),
                    mode="lines",
                    line={"width": _lw},
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            if log_scale:
                fig.update_yaxes(type="log", row=row, col=col)

        return panel_facet_figure(
            df,
            _render_periodogram,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_n_cols=facet_n_cols,
            title=title or "Periodogram (Panel)",
            x_label=x_label or "Frequency (cycles/sample)",
            y_label=y_label or "Power Spectral Density",
            width=width,
            height=height,
            shared_xaxes=False,
        )

    # Resolve columns
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])

    # Get kwargs
    line_width = kwargs.get("line_width", 1.5)
    show_peaks = kwargs.get("show_peaks", False)
    n_peaks = kwargs.get("n_peaks", 3)

    # Get color sequence
    colors = resolve_color_palette(color_palette, len(plot_columns))

    # Create figure
    fig = go.Figure()

    for col_idx, col in enumerate(plot_columns):
        # Get column values
        y = df[col].to_numpy()

        # Compute periodogram
        freqs, power = scipy_periodogram(y)

        # Skip zero frequency
        freqs = freqs[1:]
        power = power[1:]

        # Always include period in hover text
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
        hover = (
            f"<b>{col}</b><br>"
            f"Frequency: %{{x:.4f}}<br>"
            f"Period: %{{customdata:.1f}} samples<br>"
            f"PSD: %{{y:.4f}}<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=power,
                mode="lines",
                line={"width": line_width, "color": colors[col_idx]},
                name=col,
                customdata=periods,
                hovertemplate=hover,
            )
        )

        # Add peak markers if requested
        if show_peaks and n_peaks > 0:
            # Find n largest peaks
            peak_indices = np.argsort(power)[-n_peaks:][::-1]
            peak_freqs = freqs[peak_indices]
            peak_powers = power[peak_indices]
            peak_periods = np.where(peak_freqs > 0, 1.0 / peak_freqs, np.inf)

            hover_peak = (
                f"<b>{col} Peak</b><br>"
                f"Frequency: %{{x:.4f}}<br>"
                f"Period: %{{customdata:.1f}} samples<br>"
                f"PSD: %{{y:.4f}}<extra></extra>"
            )

            fig.add_trace(
                go.Scatter(
                    x=peak_freqs,
                    y=peak_powers,
                    mode="markers+text",
                    marker={"size": 8, "color": colors[col_idx], "symbol": "diamond"},
                    name=f"{col} (peaks)",
                    customdata=peak_periods,
                    text=[f"T={p:.0f}" for p in peak_periods],
                    textposition="top center",
                    hovertemplate=hover_peak,
                )
            )

    title_default = "Periodogram" if title is None else title
    x_label_default = "Frequency (cycles/sample)" if x_label is None else x_label
    y_label_default = "Power Spectral Density" if y_label is None else y_label

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
