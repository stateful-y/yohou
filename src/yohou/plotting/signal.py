"""Signal processing plotting functions for time series analysis."""

from typing import Literal

import numpy as np
import polars as pl
from scipy.signal import periodogram as scipy_periodogram

try:
    import plotly.graph_objects as go
except ImportError as e:
    msg = "plotly is required for yohou plotting. Install: pip install yohou[plotting]"
    raise ImportError(msg) from e

from yohou.plotting._utils import (
    LegendTracker,
    RenderContext,
    _auto_detect_panel,
    _group_panel_columns,
    facet_figure,
    resolve_color_palette,
    resolve_panel_columns,
)
from yohou.utils import validate_plotting_data, validate_plotting_params

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
    groups: list[str] | None = None,
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
    line_width: float = 2.0,
) -> go.Figure:
    r"""Plot the phase of a time series.

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
    groups : list[str] | None, default=None
        Panel group prefixes to plot.
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data.  ``"group"`` creates one subplot per
        group, ``"member"`` one per member.  ``None`` falls back to
        ``"member"``.  Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in facet grid.
    color_palette : list[str] | None, default=None
        Custom color palette.
    show_legend : bool, default=True
        Whether to show the legend.
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
    connect_gaps : bool, default=False
        Whether to connect segments across any NaN values in the computed
        spectrum (rarely needed; the FFT output is uniformly dense over a
        frequency axis, so there are no temporal gaps).
    line_width : float, default=2.0
        Width of the line traces.

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
    [`plot_spectrum`][yohou.plotting.plot_spectrum] : Plot power spectral density.
    """
    use_degrees = angle_unit == "degree"
    unit = "degrees" if use_degrees else "radians"
    validate_plotting_params(width=width, height=height)

    # Auto-detect panel data
    if groups is None and columns is None and _auto_detect_panel(df):
        groups = []

    if groups is not None:
        validate_plotting_data(df)
        _panel_cols = resolve_panel_columns(df, groups, columns)
        effective_facet_by = facet_by or "member"
        _group_map, _members = _group_panel_columns(_panel_cols)
        _n_overlays = len(_group_map) if effective_facet_by == "member" else len(_members)
        _panel_colors = resolve_color_palette(color_palette, _n_overlays)
        _legend_tracker = LegendTracker()

        def _render_phase(ctx: RenderContext) -> None:
            """Render phase spectrum for a single panel group."""
            base = ctx.display_name
            y_arr = ctx.sub_df[base].to_numpy().astype(float)
            spectrum = np.fft.rfft(y_arr)
            freqs = np.fft.rfftfreq(len(y_arr))
            phase = np.angle(spectrum)
            if unwrap:
                phase = np.unwrap(phase)
            if use_degrees:
                phase = np.degrees(phase)
            ctx.fig.add_trace(
                go.Scatter(
                    x=freqs[1:].tolist(),
                    y=phase[1:].tolist(),
                    mode="lines",
                    line={"width": line_width, "color": _panel_colors[ctx.entity_idx]},
                    name=ctx.display_name,
                    legendgroup=ctx.display_name,
                    showlegend=_legend_tracker.should_show(ctx.display_name),
                    connectgaps=connect_gaps,
                ),
                row=ctx.row,
                col=ctx.col,
            )

        fig = facet_figure(
            df,
            _render_phase,
            groups=groups,
            columns=columns,
            facet_n_cols=facet_n_cols,
            facet_by=effective_facet_by,
            title=title or "Phase Spectrum",
            x_label=x_label or "Frequency (cycles/sample)",
            y_label=y_label or f"Phase ({unit})",
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

    def _render_phase(ctx: RenderContext) -> None:
        """Render phase spectrum for one column into a subplot."""
        base = ctx.display_name
        col_color = _col_colors[base]
        y = ctx.sub_df[base].to_numpy().astype(float)
        spectrum = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(y))
        phase = np.angle(spectrum)
        if unwrap:
            phase = np.unwrap(phase)
        if use_degrees:
            phase = np.degrees(phase)
        freqs = freqs[1:]
        phase = phase[1:]
        unit_label = "\u00b0" if use_degrees else "rad"
        ctx.fig.add_trace(
            go.Scatter(
                x=freqs,
                y=phase,
                mode="lines",
                line={"width": line_width, "color": col_color},
                name=base,
                connectgaps=connect_gaps,
                hovertemplate=(
                    f"<b>{base}</b><br>Frequency: %{{x:.4f}}<br>Phase: %{{y:.2f}} {unit_label}<extra></extra>"
                ),
            ),
            row=ctx.row,
            col=ctx.col,
        )

    fig = facet_figure(
        df,
        _render_phase,
        columns=plot_columns,
        facet_n_cols=facet_n_cols,
        title=title or "Phase Spectrum",
        x_label=x_label or "Frequency (cycles/sample)",
        y_label=y_label or f"Phase ({unit})",
        width=width,
        height=height,
        shared_xaxes=False,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


def plot_spectrum(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    log_scale: bool = False,
    groups: list[str] | None = None,
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
    line_width: float = 2.0,
    show_peaks: bool = False,
    n_peaks: int = 3,
) -> go.Figure:
    """Plot periodogram (power spectral density) for frequency domain analysis.

    Creates a periodogram showing the power spectral density via
    ``scipy.signal.periodogram``, useful for identifying dominant frequencies
    and periodic patterns in the data. The signal mean is removed before the
    FFT (``scipy``'s default ``detrend='constant'``), so the DC component is
    suppressed; this differs from ``plot_phase``, which applies no detrending.
    In non-panel mode, hover text includes the corresponding period
    (1/frequency) and detected peaks are annotated with their period in
    sample units; these enrichments are not applied to panel facets.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to analyze. If None, uses all numeric columns except 'time'.
    log_scale : bool, default=False
        Use logarithmic scale for PSD axis.
    groups : list[str] | None, default=None
        Panel group prefixes to plot.
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data.  ``"group"`` creates one subplot per
        group, ``"member"`` one per member.  ``None`` falls back to
        ``"member"``.  Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in facet grid.
    color_palette : list[str] | None, default=None
        Custom color palette.
    show_legend : bool, default=True
        Whether to show the legend.
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
    connect_gaps : bool, default=False
        Whether to connect segments across any NaN values in the computed
        spectrum (rarely needed; the FFT output is uniformly dense over a
        frequency axis, so there are no temporal gaps).
    line_width : float, default=2.0
        Width of the line traces.
    show_peaks : bool, default=False
        Whether to annotate dominant frequency peaks. Only has an effect in
        non-panel mode; ignored when faceting panel data.
    n_peaks : int, default=3
        Number of peaks to annotate when ``show_peaks`` is True. Only has an
        effect in non-panel mode.

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
    [`plot_phase`][yohou.plotting.plot_phase] : Plot phase spectrum.
    """
    # Validate inputs
    validate_plotting_params(width=width, height=height)

    # Auto-detect panel data
    if groups is None and columns is None and _auto_detect_panel(df):
        groups = []

    if groups is not None:
        validate_plotting_data(df)
        _panel_cols = resolve_panel_columns(df, groups, columns)
        _panel_colors = resolve_color_palette(color_palette, len(_panel_cols))
        _legend_tracker = LegendTracker()

        def _render_periodogram(ctx: RenderContext) -> None:
            """Render spectral periodogram with optional log scaling for a single column."""
            base = ctx.display_name
            y_arr = ctx.sub_df[base].to_numpy()
            freqs, psd = scipy_periodogram(y_arr)
            ctx.fig.add_trace(
                go.Scatter(
                    x=freqs[1:].tolist(),
                    y=psd[1:].tolist(),
                    mode="lines",
                    line={"width": line_width, "color": _panel_colors[ctx.entity_idx]},
                    name=ctx.display_name,
                    legendgroup=ctx.display_name,
                    showlegend=_legend_tracker.should_show(ctx.display_name),
                    connectgaps=connect_gaps,
                ),
                row=ctx.row,
                col=ctx.col,
            )
            if log_scale:
                ctx.fig.update_yaxes(type="log", row=ctx.row, col=ctx.col)

        effective_facet_by = facet_by or "member"
        fig = facet_figure(
            df,
            _render_periodogram,
            groups=groups,
            columns=columns,
            facet_n_cols=facet_n_cols,
            facet_by=effective_facet_by,
            title=title or "Periodogram",
            x_label=x_label or "Frequency (cycles/sample)",
            y_label=y_label or "Power Spectral Density",
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

    def _render_spectrum(ctx: RenderContext) -> None:
        """Render periodogram for one column into a subplot."""
        base = ctx.display_name
        col_color = _col_colors[base]
        y = ctx.sub_df[base].to_numpy()
        freqs, power = scipy_periodogram(y)
        freqs = freqs[1:]
        power = power[1:]
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
        hover = (
            f"<b>{base}</b><br>"
            f"Frequency: %{{x:.4f}}<br>"
            f"Period: %{{customdata:.1f}} samples<br>"
            f"PSD: %{{y:.4f}}<extra></extra>"
        )
        ctx.fig.add_trace(
            go.Scatter(
                x=freqs,
                y=power,
                mode="lines",
                line={"width": line_width, "color": col_color},
                name=base,
                customdata=periods,
                connectgaps=connect_gaps,
                hovertemplate=hover,
            ),
            row=ctx.row,
            col=ctx.col,
        )
        if show_peaks and n_peaks > 0:
            peak_indices = np.argsort(power)[-n_peaks:][::-1]
            peak_freqs = freqs[peak_indices]
            peak_powers = power[peak_indices]
            peak_periods = np.where(peak_freqs > 0, 1.0 / peak_freqs, np.inf)
            hover_peak = (
                f"<b>{base} Peak</b><br>"
                f"Frequency: %{{x:.4f}}<br>"
                f"Period: %{{customdata:.1f}} samples<br>"
                f"PSD: %{{y:.4f}}<extra></extra>"
            )
            ctx.fig.add_trace(
                go.Scatter(
                    x=peak_freqs,
                    y=peak_powers,
                    mode="markers+text",
                    marker={"size": 8, "color": col_color, "symbol": "diamond"},
                    name=f"{base} (peaks)",
                    customdata=peak_periods,
                    text=[f"T={p:.0f}" for p in peak_periods],
                    textposition="top center",
                    hovertemplate=hover_peak,
                ),
                row=ctx.row,
                col=ctx.col,
            )
        if log_scale:
            ctx.fig.update_yaxes(type="log", row=ctx.row, col=ctx.col)

    fig = facet_figure(
        df,
        _render_spectrum,
        columns=plot_columns,
        facet_n_cols=facet_n_cols,
        title=title or "Periodogram",
        x_label=x_label or "Frequency (cycles/sample)",
        y_label=y_label or "Power Spectral Density",
        width=width,
        height=height,
        shared_xaxes=False,
    )
    fig.update_layout(showlegend=show_legend)

    return fig
