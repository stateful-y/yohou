"""Diagnostic and frequency-domain plotting functions for time series analysis."""

import math
from typing import Literal, cast

import numpy as np
import polars as pl
from scipy.stats import norm

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    msg = "plotly is required for yohou plotting. Install: pip install yohou[plotting]"
    raise ImportError(msg) from e

from yohou.plotting._utils import (
    DEFAULT_WIDTH,
    LegendTracker,
    PanelColorManager,
    RenderContext,
    _add_confidence_bands,
    _auto_detect_panel,
    _create_subplots,
    _group_panel_columns,
    _make_hovertemplate,
    _member_name,
    _subplot_spacing,
    apply_default_layout,
    facet_figure,
    resolve_color_palette,
    resolve_panel_columns,
)
from yohou.utils import validate_plotting_data, validate_plotting_params
from yohou.utils.panel import inspect_panel

__all__ = [
    "plot_autocorrelation",
    "plot_correlation_heatmap",
    "plot_cross_correlation",
    "plot_lag_scatter",
    "plot_partial_autocorrelation",
    "plot_scatter_matrix",
    "plot_seasonal_heatmap",
    "plot_seasonality",
    "plot_subseasonality",
]


def plot_autocorrelation(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    max_lags: int | None = None,
    confidence_level: float = 0.95,
    show_confidence: bool = True,
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
    color: str = "#2563EB",
) -> go.Figure:
    """Plot autocorrelation function (ACF) for time series.

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
    confidence_level : float, default=0.95
        Confidence level for confidence bands (e.g. ``0.95`` for 95%).
    show_confidence : bool, default=True
        Whether to show confidence bands.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot.
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data.  ``"group"`` creates one subplot per
        group, ``"member"`` one per member.  ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in facet grid.
    color_palette : list[str] | None, default=None
        Custom color palette (one color per column).
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
    color : str, default="#2563EB"
        Bar color for single-column plots (ignored when color_palette is set).

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
    [`plot_partial_autocorrelation`][yohou.plotting.plot_partial_autocorrelation] : Plot partial autocorrelation function.
    [`plot_correlation_heatmap`][yohou.plotting.plot_correlation_heatmap] : Plot correlation matrix.
    """
    # Validate inputs
    validate_plotting_data(df, min_rows=2)
    validate_plotting_params(width=width, height=height)

    if _auto_detect_panel(df) and panel_group_names is None and columns is None:
        panel_group_names = []

    if panel_group_names is not None:
        _panel_cols = resolve_panel_columns(df, panel_group_names, columns)
        _color_mgr = PanelColorManager(color_palette)
        _acf_legend = LegendTracker()

        def _render_acf(ctx: RenderContext) -> None:
            """Render autocorrelation bar chart with confidence bands for a single column."""
            base = [c for c in ctx.sub_df.columns if c != "time"][0]
            member = _member_name(base)
            _bc = _color_mgr.get_color(member)
            series = ctx.sub_df[base].drop_nulls()
            n_s = len(series)
            _ml = max_lags if max_lags is not None else min(n_s // 2, 40)
            acf_vals = _compute_acf_values(series, _ml)
            ctx.fig.add_trace(
                go.Bar(
                    x=list(range(_ml + 1)),
                    y=acf_vals,
                    marker={"color": _bc},
                    legendgroup=ctx.display_name,
                    showlegend=_acf_legend.should_show(ctx.display_name),
                    name=ctx.display_name,
                    hovertemplate=_make_hovertemplate(ctx.display_name, "Lag", "ACF", decimals=3),
                ),
                row=ctx.row,
                col=ctx.col,
            )
            if show_confidence:
                ci = norm.ppf(1 - (1 - confidence_level) / 2) / math.sqrt(n_s)
                _add_confidence_bands(
                    ctx.fig, list(range(_ml + 1)), [ci] * (_ml + 1), [-ci] * (_ml + 1), row=ctx.row, col=ctx.col
                )

        effective_facet_by = facet_by or "member"
        fig = facet_figure(
            df,
            _render_acf,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_by=effective_facet_by,
            facet_n_cols=facet_n_cols,
            title=title or "Autocorrelation (ACF)",
            x_label=x_label or "Lag",
            y_label=y_label or "ACF",
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
    _default_bar_color = color  # fallback single-column color

    n = len(df)
    if max_lags is None:
        max_lags = min(n // 2, 40)

    def _render_acf(ctx: RenderContext) -> None:
        """Render ACF bars for one column into a subplot."""
        base = ctx.display_name
        _bar_color = _col_colors[base] if color_palette is not None else _default_bar_color
        series = ctx.sub_df[base].drop_nulls()
        n_series = len(series)
        acf_values = _compute_acf_values(series, max_lags)
        ctx.fig.add_trace(
            go.Bar(
                x=list(range(max_lags + 1)),
                y=acf_values,
                name=base,
                marker={"color": _bar_color},
                hovertemplate=_make_hovertemplate(base, "Lag", "ACF", decimals=3),
            ),
            row=ctx.row,
            col=ctx.col,
        )
        if show_confidence:
            ci = norm.ppf(1 - (1 - confidence_level) / 2) / math.sqrt(n_series)
            _add_confidence_bands(
                ctx.fig,
                list(range(max_lags + 1)),
                [ci] * (max_lags + 1),
                [-ci] * (max_lags + 1),
                row=ctx.row,
                col=ctx.col,
            )

    fig = facet_figure(
        df,
        _render_acf,
        columns=plot_columns,
        facet_n_cols=facet_n_cols,
        title=title or "Autocorrelation (ACF)",
        x_label=x_label or "Lag",
        y_label=y_label or "Autocorrelation",
        width=width,
        height=height,
        shared_xaxes=False,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


def _compute_acf_values(series: pl.Series, max_lags: int) -> list[float]:
    """Compute autocorrelation values for a single series.

    Parameters
    ----------
    series : pl.Series
        Numeric series (NaN-free) to compute ACF for.
    max_lags : int
        Maximum number of lags to compute.

    Returns
    -------
    list[float]
        ACF values from lag 0 to *max_lags* inclusive.

    """
    mean_val = series.mean()
    acf_values: list[float] = []
    for lag in range(max_lags + 1):
        if lag == 0:
            acf_values.append(1.0)
        else:
            s1 = series[:-lag] - mean_val
            s2 = series[lag:] - mean_val
            numerator = float((s1 * s2).sum())
            denominator = float(((series - mean_val) ** 2).sum())
            acf_values.append(numerator / denominator if denominator != 0 else 0.0)
    return acf_values


def _compute_pacf(
    values: np.ndarray,
    nlags: int,
    *,
    method: str = "yw",
    alpha: float | None = None,
) -> tuple[list[float], list[float] | None, list[float] | None]:
    """Compute PACF using ``statsmodels.tsa.stattools.pacf``.

    Parameters
    ----------
    values : np.ndarray
        1-D numeric array.
    nlags : int
        Number of lags.
    method : str
        PACF method (see ``statsmodels.tsa.stattools.pacf``).
    alpha : float | None
        Significance level for confidence intervals.

    Returns
    -------
    tuple[list[float], list[float] | None, list[float] | None]
        ``(pacf_values, ci_lower, ci_upper)``.

    Raises
    ------
    ImportError
        If ``statsmodels`` is not installed.

    """
    try:
        from statsmodels.tsa.stattools import pacf as sm_pacf  # noqa: PLC0415
    except ImportError:
        msg = "statsmodels is required for PACF computation. Install it with: pip install yohou[plotting]"
        raise ImportError(msg) from None

    if alpha is not None:
        pacf_result, confint = sm_pacf(
            values,
            nlags=nlags,
            method=method,  # ty: ignore[invalid-argument-type]
            alpha=alpha,
        )
        # statsmodels returns CIs centered on the PACF values (pacf +/- z/sqrt(n)).
        # For significance testing we need horizontal bands at +/-z/sqrt(n) (centered
        # on zero), so subtract the PACF values from both bounds.
        ci_lo = (confint[:, 0] - pacf_result).tolist()
        ci_hi = (confint[:, 1] - pacf_result).tolist()
        return pacf_result.tolist(), ci_lo, ci_hi

    pacf_result = sm_pacf(values, nlags=nlags, method=method)  # ty: ignore[invalid-argument-type]
    return pacf_result.tolist(), None, None  # ty: ignore[unresolved-attribute]


def plot_partial_autocorrelation(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    max_lags: int | None = None,
    method: str = "yw",
    confidence_level: float = 0.95,
    show_confidence: bool = True,
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
    color: str = "#059669",
) -> go.Figure:
    """Plot partial autocorrelation function (PACF) for time series.

    Shows correlation between the series and its lagged values after removing
    the effect of intermediate lags. Useful for determining appropriate AR order.

    The computation is delegated to ``statsmodels.tsa.stattools.pacf`` which
    supports several estimation methods and proper confidence intervals.
    Requires ``yohou[plotting]``.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to analyze. If None, uses all numeric columns except 'time'.
    max_lags : int | None, default=None
        Maximum number of lags to compute. If None, uses min(len(df)//2, 40).
    method : str, default="yw"
        PACF estimation method passed to ``statsmodels.tsa.stattools.pacf``.
        Options: ``"yw"`` (Yule-Walker), ``"ols"`` (OLS), ``"ywm"``
        (Yule-Walker mean-adjusted), ``"burg"``, ``"ldb"``
        (Levinson-Durbin biased), ``"ldbiased"``.
    confidence_level : float, default=0.95
        Confidence level for confidence bands (e.g. ``0.95`` for 95%).
    show_confidence : bool, default=True
        Whether to show confidence bands.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot.
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data.  ``"group"`` creates one subplot per
        group, ``"member"`` one per member.  ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in facet grid.
    color_palette : list[str] | None, default=None
        Custom color palette (one color per column).
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
    color : str, default="#059669"
        Bar color for single-column plots (ignored when color_palette is set).

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
    [`plot_autocorrelation`][yohou.plotting.plot_autocorrelation] : Plot autocorrelation function.
    """
    # Validate inputs
    validate_plotting_data(df, min_rows=2)
    validate_plotting_params(width=width, height=height)

    if _auto_detect_panel(df) and panel_group_names is None and columns is None:
        panel_group_names = []

    if panel_group_names is not None:
        _panel_cols = resolve_panel_columns(df, panel_group_names, columns)
        _pacf_color_mgr = PanelColorManager(color_palette)
        _pacf_legend = LegendTracker()

        def _render_pacf(ctx: RenderContext) -> None:
            """Render PACF bar chart with confidence bands for a single panel."""
            base = [c for c in ctx.sub_df.columns if c != "time"][0]
            member = _member_name(base)
            _bc = _pacf_color_mgr.get_color(member)
            series = ctx.sub_df[base].drop_nulls()
            n_s = len(series)
            _ml = max_lags if max_lags is not None else min(n_s // 2, 40)

            alpha_val = 1 - confidence_level if show_confidence else None
            pacf_vals, ci_lo, ci_hi = _compute_pacf(
                series.to_numpy(),
                _ml,
                method=method,
                alpha=alpha_val,
            )

            ctx.fig.add_trace(
                go.Bar(
                    x=list(range(_ml + 1)),
                    y=pacf_vals,
                    marker={"color": _bc},
                    legendgroup=ctx.display_name,
                    showlegend=_pacf_legend.should_show(ctx.display_name),
                    name=ctx.display_name,
                    hovertemplate=_make_hovertemplate(ctx.display_name, "Lag", "PACF", decimals=3),
                ),
                row=ctx.row,
                col=ctx.col,
            )
            if show_confidence and ci_lo is not None and ci_hi is not None:
                _add_confidence_bands(ctx.fig, list(range(_ml + 1)), ci_hi, ci_lo, row=ctx.row, col=ctx.col)

        effective_facet_by = facet_by or "member"
        fig = facet_figure(
            df,
            _render_pacf,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_by=effective_facet_by,
            facet_n_cols=facet_n_cols,
            title=title or "Partial Autocorrelation (PACF)",
            x_label=x_label or "Lag",
            y_label=y_label or "PACF",
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
    _default_bar_color = color  # fallback single-column color

    n = len(df)
    if max_lags is None:
        max_lags = min(n // 2, 40)
    alpha_val = 1 - confidence_level if show_confidence else None

    def _render_pacf(ctx: RenderContext) -> None:
        """Render PACF bars for one column into a subplot."""
        base = ctx.display_name
        _bar_color = _col_colors[base] if color_palette is not None else _default_bar_color
        series = ctx.sub_df[base].drop_nulls()
        values = series.to_numpy()
        pacf_values, ci_lo, ci_hi = _compute_pacf(values, max_lags, method=method, alpha=alpha_val)
        ctx.fig.add_trace(
            go.Bar(
                x=list(range(max_lags + 1)),
                y=pacf_values,
                name=base,
                marker={"color": _bar_color},
                hovertemplate=_make_hovertemplate(base, "Lag", "PACF", decimals=3),
            ),
            row=ctx.row,
            col=ctx.col,
        )
        if show_confidence and ci_lo is not None and ci_hi is not None:
            _add_confidence_bands(
                ctx.fig,
                list(range(max_lags + 1)),
                ci_hi,
                ci_lo,
                row=ctx.row,
                col=ctx.col,
            )

    fig = facet_figure(
        df,
        _render_pacf,
        columns=plot_columns,
        facet_n_cols=facet_n_cols,
        title=title or "Partial Autocorrelation (PACF)",
        x_label=x_label or "Lag",
        y_label=y_label or "Partial Autocorrelation",
        width=width,
        height=height,
        shared_xaxes=False,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


def _correlation_heatmap_separate(
    df: pl.DataFrame,
    groups: dict[str, list[str]],
    *,
    colorscale: str,
    show_values: bool,
    show_legend: bool,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    width: int | None,
    height: int | None,
) -> dict[str, go.Figure]:
    """Build one standalone heatmap figure per panel group."""
    figures: dict[str, go.Figure] = {}
    for gname, gcols in groups.items():
        base_names = [c.split("__", 1)[1] if "__" in c else c for c in gcols]
        sub = df.select(gcols).rename(dict(zip(gcols, base_names, strict=False)))
        corr = sub.drop_nulls().corr()
        text_ann = None
        if show_values:
            text_ann = [[f"{v:.2f}" if v is not None else "" for v in row] for row in corr.rows()]
        gfig = go.Figure()
        gfig.add_trace(
            go.Heatmap(
                z=corr.to_numpy(),
                x=base_names,
                y=base_names,
                colorscale=colorscale,
                zmid=0,
                text=text_ann,
                texttemplate="%{text}" if show_values else None,
                hovertemplate=(f"<b>{gname}</b><br>%{{x}} vs %{{y}}<br>Correlation: %{{z:.3f}}<extra></extra>"),
            )
        )
        gfig = apply_default_layout(
            gfig,
            title=title or "Correlation Heatmap",
            x_label=x_label,
            y_label=y_label,
            width=width,
            height=height,
        )
        gfig.update_layout(showlegend=show_legend)
        figures[gname] = gfig
    return figures


def _correlation_heatmap_by_member(
    df: pl.DataFrame,
    member_groups: dict[str, list[str]],
    *,
    colorscale: str,
    show_values: bool,
    show_legend: bool,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    width: int | None,
    height: int | None,
) -> dict[str, go.Figure]:
    """Build one standalone heatmap figure per member (groups as axis labels)."""
    figures: dict[str, go.Figure] = {}
    for mname, mcols in member_groups.items():
        base_names = [c.split("__", 1)[0] if "__" in c else c for c in mcols]
        sub = df.select(mcols).rename(dict(zip(mcols, base_names, strict=False)))
        corr = sub.drop_nulls().corr()
        text_ann = None
        if show_values:
            text_ann = [[f"{v:.2f}" if v is not None else "" for v in row] for row in corr.rows()]
        mfig = go.Figure()
        mfig.add_trace(
            go.Heatmap(
                z=corr.to_numpy(),
                x=base_names,
                y=base_names,
                colorscale=colorscale,
                zmid=0,
                text=text_ann,
                texttemplate="%{text}" if show_values else None,
                hovertemplate=(f"<b>{mname}</b><br>%{{x}} vs %{{y}}<br>Correlation: %{{z:.3f}}<extra></extra>"),
            )
        )
        mfig = apply_default_layout(
            mfig,
            title=title or f"Correlation Heatmap - {mname}",
            x_label=x_label,
            y_label=y_label,
            width=width,
            height=height,
        )
        mfig.update_layout(showlegend=show_legend)
        figures[mname] = mfig
    return figures


def _correlation_heatmap_single(
    df: pl.DataFrame,
    plot_columns: list[str],
    *,
    colorscale: str,
    show_values: bool,
    show_legend: bool,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    width: int | None,
    height: int | None,
) -> go.Figure:
    """Build a single correlation heatmap for non-panel data."""
    corr_matrix = df.select(plot_columns).drop_nulls().corr()

    fig = go.Figure()

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

    fig = apply_default_layout(
        fig,
        title=title or "Correlation Heatmap",
        x_label=x_label,
        y_label=y_label,
        width=width,
        height=height,
    )
    fig.update_layout(showlegend=show_legend)
    return fig


def plot_correlation_heatmap(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    panel_group_names: list[str] | None = None,
    facet_by: Literal["group", "member"] | None = "group",
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    colorscale: str = "RdBu_r",
    show_values: bool = True,
) -> go.Figure | dict[str, go.Figure]:
    """Plot correlation matrix heatmap for multiple time series.

    Shows pairwise correlations between different time series columns,
    useful for understanding relationships and multicollinearity.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to include. If None, uses all numeric columns except 'time'.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot.  When panel data is detected
        and this is ``None``, all groups are included.
    facet_by : Literal["group", "member"] | None, default="group"
        Faceting axis for panel data.  ``"group"`` returns one figure
        per group (members as axis labels), ``"member"`` returns one
        figure per member (groups as axis labels).  ``None`` returns a
        single figure correlating all panel columns.
        Ignored for non-panel data.
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
    colorscale : str, default="RdBu_r"
        Plotly colorscale for the heatmap.
    show_values : bool, default=True
        Whether to display correlation values on the heatmap cells.

    Returns
    -------
    go.Figure | dict[str, go.Figure]
        Single figure when non-panel or ``facet_by=None``.
        Dict of figures keyed by group or member name otherwise.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_correlation_heatmap

    >>> # Create sample data with multiple series
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
    ...     "y1": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    ...     "y2": [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
    ... })

    >>> # Plot correlation matrix
    >>> fig = plot_correlation_heatmap(df)
    >>> len(fig.data) > 0
    True

    See Also
    --------
    [`plot_autocorrelation`][yohou.plotting.plot_autocorrelation] : Plot autocorrelation function.
    """
    # Validate inputs
    validate_plotting_data(df)
    validate_plotting_params(width=width, height=height)

    # Auto-detect panel data
    _, panel_groups = inspect_panel(df)
    if panel_group_names is None and columns is None and panel_groups:
        panel_group_names = []

    if panel_group_names is not None:
        from yohou.plotting._utils import _group_panel_columns  # noqa: PLC0415

        # Normalize columns to list for member filtering
        col_filter = [columns] if isinstance(columns, str) else columns

        # Filter to requested groups and optionally by member postfix
        groups: dict[str, list[str]] = {}
        for g, gcols in panel_groups.items():
            if not panel_group_names or g in panel_group_names:
                filtered = [c for c in gcols if c.split("__", 1)[1] in col_filter] if col_filter is not None else gcols
                if filtered:
                    groups[g] = filtered
        if not groups:
            msg = f"No panel groups found for {panel_group_names}. Available groups: {list(panel_groups.keys())}"
            raise ValueError(msg)

        all_panel_cols = [c for gcols in groups.values() for c in gcols]
        _, all_members = _group_panel_columns(all_panel_cols)

        _layout_kwargs = {
            "colorscale": colorscale,
            "show_values": show_values,
            "show_legend": show_legend,
            "title": title,
            "x_label": x_label,
            "y_label": y_label,
            "width": width,
            "height": height,
        }

        if facet_by is None:
            return _correlation_heatmap_single(df, all_panel_cols, **_layout_kwargs)

        if facet_by == "group":
            return _correlation_heatmap_separate(df, groups, **_layout_kwargs)

        # facet_by == "member"
        member_groups: dict[str, list[str]] = {}
        for member in all_members:
            member_cols = []
            for _gname, gcols in groups.items():
                col = next((c for c in gcols if _member_name(c) == member), None)
                if col:
                    member_cols.append(col)
            if member_cols:
                member_groups[member] = member_cols
        return _correlation_heatmap_by_member(df, member_groups, **_layout_kwargs)

    # Non-panel path
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])
    return _correlation_heatmap_single(
        df,
        plot_columns,
        colorscale=colorscale,
        show_values=show_values,
        show_legend=show_legend,
        title=title,
        x_label=x_label,
        y_label=y_label,
        width=width,
        height=height,
    )


_SEASON_LABELS_MAP: dict[str, list[str]] = {
    "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "quarter": ["Q1", "Q2", "Q3", "Q4"],
    "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
}


def _add_season_and_cycle(frame: pl.DataFrame, frequency: str) -> pl.DataFrame:
    """Add ``season`` and ``cycle`` columns to *frame* based on *frequency*."""
    if frequency == "month":
        return frame.with_columns(
            pl.col("time").dt.month().alias("season"),
            pl.col("time").dt.year().alias("cycle"),
        )
    if frequency == "quarter":
        return frame.with_columns(
            pl.col("time").dt.quarter().alias("season"),
            pl.col("time").dt.year().alias("cycle"),
        )
    if frequency == "weekday":
        return frame.with_columns(
            pl.col("time").dt.weekday().alias("season"),
            pl.col("time").dt.iso_year().alias("cycle"),
        )
    if frequency == "week":
        return frame.with_columns(
            pl.col("time").dt.week().alias("season"),
            pl.col("time").dt.iso_year().alias("cycle"),
        )
    if frequency == "hour":
        return frame.with_columns(
            pl.col("time").dt.hour().alias("season"),
            pl.col("time").dt.ordinal_day().alias("cycle"),
        )
    msg = f"Unknown frequency: {frequency}. Valid options: month, quarter, weekday, week, hour"
    raise ValueError(msg)


def plot_seasonality(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    seasonality: str = "month",
    highlight: int | list[int] | None = None,
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
    highlight_width: float = 3.0,
    fade_opacity: float = 0.25,
    opacity_power: float = 2.0,
) -> go.Figure:
    """Plot seasonal overlay.

    Shows one line per cycle (e.g. year) overlaid on the same axes, with the
    position within each season (e.g. month 1-12) on the x-axis.  This makes
    it easy to compare how the same season changes across cycles.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to plot. If None, uses all numeric columns except 'time'.
    seasonality : str, default="month"
        Seasonal frequency: "month", "quarter", "weekday", "week", "hour".
    highlight : int | list[int] | None, default=None
        Cycle(s) to emphasise (e.g. specific years). Highlighted cycles get a
        thicker line; others are faded. If None, all cycles are equally visible.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot.
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data.  ``"group"`` creates one subplot per
        group, ``"member"`` one per member.  ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in facet grid.
    color_palette : list[str] | None, default=None
        Custom color palette (one color per cycle).
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
    connect_gaps : bool, default=False
        If True, connect lines across missing data gaps.
    show_legend : bool, default=True
        Whether to display the legend.
    resampler : bool | Literal["widget"] | None, default=None
        Enable plotly-resampler for large datasets.  ``True`` returns a
        ``FigureResampler``, ``"widget"`` a ``FigureWidgetResampler``,
        ``None`` reads from `get_config`.
    line_width : float, default=2.0
        Line width for cycle traces.
    highlight_width : float, default=3.0
        Line width for highlighted cycle traces.
    fade_opacity : float, default=0.25
        Opacity for non-highlighted cycles when ``highlight`` is set.
    opacity_power : float, default=2.0
        Power curve exponent for time-ordered opacity fade.

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
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2022, 12, 31), "1d", eager=True),
    ...     "y": [100 + (i % 30) + (i // 30) * 5 for i in range(1096)],
    ... })

    >>> # Plot monthly seasonality
    >>> fig = plot_seasonality(df, columns="y", seasonality="month")
    >>> len(fig.data) > 0
    True

    See Also
    --------
    [`plot_subseasonality`][yohou.plotting.plot_subseasonality] : Plot seasonal subseries.
    [`plot_time_series`][yohou.plotting.plot_time_series] : Plot basic time series.
    """
    # Resolve highlight to a set
    if highlight is None:
        highlight_set: set[int] = set()
    elif isinstance(highlight, int):
        highlight_set = {highlight}
    else:
        highlight_set = set(highlight)

    _oldest_alpha = 0.1
    _newest_alpha = 0.9

    validate_plotting_data(df, min_rows=2)
    validate_plotting_params(width=width, height=height)

    # Auto-detect panel data
    if panel_group_names is None and columns is None and _auto_detect_panel(df):
        panel_group_names = []

    if panel_group_names is not None:
        _color_mgr = PanelColorManager(color_palette)
        _legend_tracker = LegendTracker(show_legend=show_legend)

        def _render_season(ctx: RenderContext) -> None:
            """Render seasonal overlay for a single panel group."""
            base = [c for c in ctx.sub_df.columns if c != "time"][0]
            dfs = _add_season_and_cycle(ctx.sub_df, seasonality)
            cycles = sorted(dfs["cycle"].unique().to_list())
            member_color = _color_mgr.get_color(ctx.display_name)
            slabels = _SEASON_LABELS_MAP.get(seasonality)
            n_cycles = len(cycles)

            # Collect per-season values for aggregated mean
            season_vals: dict[int | str, list[float]] = {}

            is_legend_member = _legend_tracker.should_show(ctx.display_name)
            r, g, b = int(member_color[1:3], 16), int(member_color[3:5], 16), int(member_color[5:7], 16)

            # Invisible trace for an opaque legend swatch
            if is_legend_member:
                ctx.fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line={"color": member_color, "width": line_width},
                        name=ctx.display_name,
                        legendgroup=ctx.display_name,
                        showlegend=True,
                    ),
                    row=ctx.row,
                    col=ctx.col,
                )

            for ci, cyc in enumerate(cycles):
                cyc_df = dfs.filter(pl.col("cycle") == cyc).sort("season")
                x_vals = (
                    [slabels[int(s) - 1] for s in cyc_df["season"].to_list()]
                    if slabels and all(s <= len(slabels) for s in cyc_df["season"].to_list())
                    else cyc_df["season"].to_list()
                )
                y_vals = cyc_df[base].to_list()

                # Accumulate for mean
                for x, y in zip(x_vals, y_vals, strict=True):
                    season_vals.setdefault(x, []).append(y)

                is_hl = bool(highlight_set) and cyc in highlight_set
                lw = highlight_width if is_hl else line_width
                # Time-ordered opacity via RGBA so legend swatch stays opaque
                if is_hl:
                    alpha = 1.0
                elif highlight_set:
                    alpha = fade_opacity
                else:
                    t = ci / max(n_cycles - 1, 1)
                    alpha = _oldest_alpha + (_newest_alpha - _oldest_alpha) * t ** (1.0 / opacity_power)

                line_rgba = f"rgba({r},{g},{b},{alpha:.2f})"

                ctx.fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines",
                        line={"color": line_rgba, "width": lw},
                        connectgaps=connect_gaps,
                        hovertemplate=f"<b>{ctx.display_name} - {cyc}</b><br>%{{x}}: %{{y:.2f}}<extra></extra>",
                        name=ctx.display_name,
                        legendgroup=ctx.display_name,
                        showlegend=False,
                    ),
                    row=ctx.row,
                    col=ctx.col,
                )

            # Add aggregated mean line (bold, on top)
            if season_vals:
                mean_x = list(season_vals.keys())
                mean_y = [
                    float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else float("nan")
                    for v in season_vals.values()
                ]
                ctx.fig.add_trace(
                    go.Scatter(
                        x=mean_x,
                        y=mean_y,
                        mode="lines",
                        line={"color": member_color, "width": 3},
                        connectgaps=connect_gaps,
                        hovertemplate=f"<b>{ctx.display_name} Mean</b><br>%{{x}}: %{{y:.2f}}<extra></extra>",
                        name=ctx.display_name,
                        legendgroup=ctx.display_name,
                        showlegend=False,
                    ),
                    row=ctx.row,
                    col=ctx.col,
                )

        effective_facet_by = facet_by or "member"
        fig = facet_figure(
            df,
            _render_season,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_by=effective_facet_by,
            facet_n_cols=facet_n_cols,
            title=title or "Seasonal Pattern",
            x_label=x_label or seasonality.capitalize(),
            y_label=y_label,
            width=width,
            height=height,
            shared_xaxes=False,
            resampler=resampler,
        )
        fig.update_layout(showlegend=show_legend)
        return fig

    # Non-panel case: column-mode facet_figure
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])
    season_labels = _SEASON_LABELS_MAP.get(seasonality)
    df_aug = _add_season_and_cycle(df, seasonality)
    cycles = sorted(df_aug["cycle"].unique().to_list())
    _colors = resolve_color_palette(color_palette, len(plot_columns))
    _col_colors = dict(zip(plot_columns, _colors, strict=False))
    n_cycles = len(cycles)

    def _render_season(ctx: RenderContext) -> None:
        """Render seasonal pattern for one column into a subplot."""
        base = ctx.display_name
        col_color = _col_colors[base]
        r, g, b = int(col_color[1:3], 16), int(col_color[3:5], 16), int(col_color[5:7], 16)
        for ci, cyc in enumerate(cycles):
            cyc_df = df_aug.filter(pl.col("cycle") == cyc).sort("season")
            x_vals = (
                [season_labels[int(s) - 1] for s in cyc_df["season"].to_list()]
                if season_labels and all(s <= len(season_labels) for s in cyc_df["season"].to_list())
                else cyc_df["season"].to_list()
            )
            is_hl = bool(highlight_set) and cyc in highlight_set
            lw = highlight_width if is_hl else line_width
            if is_hl:
                alpha = 1.0
            elif highlight_set:
                alpha = fade_opacity
            else:
                t = ci / max(n_cycles - 1, 1)
                alpha = _oldest_alpha + (_newest_alpha - _oldest_alpha) * t ** (1.0 / opacity_power)
            line_rgba = f"rgba({r},{g},{b},{alpha:.2f})"
            ctx.fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=cyc_df[base].to_list(),
                    mode="lines",
                    line={"color": line_rgba, "width": lw},
                    name=base,
                    legendgroup=base,
                    showlegend=False,
                    connectgaps=connect_gaps,
                    hovertemplate=f"<b>{base} - {cyc}</b><br>%{{x}}: %{{y:.2f}}<extra></extra>",
                ),
                row=ctx.row,
                col=ctx.col,
            )

    fig = facet_figure(
        df,
        _render_season,
        columns=plot_columns,
        facet_n_cols=facet_n_cols,
        title=title or "Seasonal Pattern",
        x_label=x_label or seasonality.capitalize(),
        y_label=y_label or "Value",
        width=width,
        height=height,
        shared_xaxes=False,
        resampler=resampler,
    )
    if season_labels:
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(season_labels))),
            ticktext=season_labels,
        )
    fig.update_layout(showlegend=show_legend)

    return fig


def _hex_to_rgba(hex_color: str, opacity: float) -> str:
    """Convert ``#RRGGBB`` to ``rgba(r, g, b, opacity)``."""
    _hex = hex_color.lstrip("#")
    r, g, b = (int(_hex[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {opacity})"


def _add_kind_traces(
    fig: go.Figure,
    *,
    kind: str,
    season_df: pl.DataFrame,
    col_name: str,
    display_name: str,
    col_color: str,
    overlay_opacity: float,
    line_width: float,
    row: int,
    col: int,
    legend_tracker: LegendTracker | None = None,
) -> dict[str, list[float]] | None:
    """Add background traces for non-mean ``kind`` values.

    Returns a mapping with ``"midpoints"`` and ``"total"`` when
    *kind* is ``"lines"`` so callers can place the aggregated mean
    trace on the same numeric x-axis.  Returns ``None`` otherwise.
    """
    if kind == "lines":
        rgba = _hex_to_rgba(col_color, overlay_opacity)
        offset = 0
        midpoints: list[float] = []
        for _cyc, _grp in season_df.group_by("cycle", maintain_order=True):
            _vals = _grp[col_name].to_list()
            n = len(_vals)
            _xs = list(range(offset, offset + n))
            midpoints.append(offset + (n - 1) / 2)
            offset += n
            fig.add_trace(
                go.Scattergl(
                    x=_xs,
                    y=_vals,
                    mode="lines",
                    line={"color": rgba, "width": line_width * 0.5},
                    name=display_name,
                    legendgroup=display_name,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
        return {"midpoints": midpoints, "total": [float(offset)]}
    elif kind in ("box", "violin"):
        trace_cls = go.Box if kind == "box" else go.Violin
        color_key = "marker_color" if kind == "box" else "line_color"
        first_legend = True
        for _cyc, _grp in season_df.group_by("cycle", maintain_order=True):
            vals = _grp[col_name].drop_nulls().to_list()
            cycle_val = _cyc[0] if isinstance(_cyc, tuple) else _cyc
            if first_legend and legend_tracker is not None:
                show_leg = legend_tracker.should_show(display_name)
                first_legend = False
            else:
                show_leg = False
            fig.add_trace(
                trace_cls(
                    y=vals,
                    x=[cycle_val] * len(vals),
                    name=display_name,
                    legendgroup=display_name,
                    showlegend=show_leg,
                    hoverinfo="skip",
                    **{color_key: col_color},
                ),
                row=row,
                col=col,
            )
    return None


def plot_subseasonality(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    seasonality: str = "month",
    kind: Literal["mean", "lines", "box", "violin"] = "mean",
    show_mean: bool = True,
    panel_group_names: list[str] | None = None,
    facet_n_cols: int = 4,
    color_palette: list[str] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    connect_gaps: bool = False,
    show_legend: bool = True,
    resampler: bool | Literal["widget"] | None = None,
    line_width: float = 2.0,
    marker_size: float = 4,
    marker_opacity: float = 0.8,
    overlay_opacity: float = 0.15,
    mean_width: float = 2.0,
    mean_dash: str = "dash",
) -> go.Figure | dict[str, go.Figure]:
    """Plot seasonal subseries.

    Creates one mini subplot per season (e.g. 12 for months, 4 for quarters).
    Within each subplot the values for that season across all cycles are
    connected chronologically, with an optional horizontal mean line.  This
    view highlights both the *level* differences between seasons and any
    *trends within* each season over time.

    For panel data, returns one figure per member with groups overlaid by
    colour.  When only a single member exists, a plain ``go.Figure`` is
    returned instead of a dict.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to plot. If None, uses all numeric columns except 'time'.
    seasonality : str, default="month"
        Seasonal frequency: "month", "quarter", "weekday", "week", "hour".
    kind : {"mean", "lines", "box", "violin"}, default="mean"
        Visualization style within each season subplot:

        - ``"mean"`` - aggregated mean-per-cycle line with markers (default).
        - ``"lines"`` - one thin line per cycle at *overlay_opacity*, with
          the mean line overlaid on top.
        - ``"box"`` - one box per cycle showing value distribution.
        - ``"violin"`` - one violin per cycle showing value distribution.
    show_mean : bool, default=True
        Show a horizontal mean line within each season subplot.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot.
    facet_n_cols : int, default=4
        Number of columns in the subplot grid.
    color_palette : list[str] | None, default=None
        Custom color palette (one color per group).
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
        If True, connect lines across missing data gaps.
    show_legend : bool, default=True
        Whether to display the legend.
    resampler : bool | Literal["widget"] | None, default=None
        Enable plotly-resampler for large datasets.  ``True`` returns a
        ``FigureResampler``, ``"widget"`` a ``FigureWidgetResampler``,
        ``None`` reads from `get_config`.
    line_width : float, default=2.0
        Line width for cycle traces.
    marker_size : float, default=4
        Marker size for data points.
    marker_opacity : float, default=0.8
        Opacity of scatter markers.
    overlay_opacity : float, default=0.15
        Opacity of per-cycle lines when ``kind="lines"``.
    mean_width : float, default=2.0
        Line width for the horizontal mean line.
    mean_dash : str, default="dash"
        Dash style for the mean line.

    Returns
    -------
    go.Figure | dict[str, go.Figure]
        Plotly figure for non-panel or single-member data.  For panel
        data with multiple members, a dict keyed by member name.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_subseasonality

    >>> # Create sample time series
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2022, 12, 31), "1d", eager=True),
    ...     "y": [100 + (i % 30) + (i // 30) * 5 for i in range(1096)],
    ... })

    >>> # Plot monthly subseries
    >>> fig = plot_subseasonality(df, columns="y", seasonality="month")
    >>> len(fig.data) > 0
    True

    See Also
    --------
    [`plot_seasonality`][yohou.plotting.plot_seasonality] : Plot seasonal overlay.
    [`plot_time_series`][yohou.plotting.plot_time_series] : Plot basic time series.
    """
    validate_plotting_data(df, min_rows=2)
    validate_plotting_params(
        kind=kind,
        valid_kinds={"mean", "lines", "box", "violin"},
        width=width,
        height=height,
    )

    if _auto_detect_panel(df) and panel_group_names is None and columns is None:
        panel_group_names = []

    if panel_group_names is not None:
        plot_columns = resolve_panel_columns(df, panel_group_names, columns)
    else:
        plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])
    df_aug = _add_season_and_cycle(df, seasonality)
    seasons = sorted(df_aug["season"].unique().to_list())
    season_labels = _SEASON_LABELS_MAP.get(seasonality)

    n_seasons = len(seasons)
    nrow = math.ceil(n_seasons / facet_n_cols)

    # Build per-subplot titles
    subplot_titles: list[str] = []
    for s in seasons:
        if season_labels and 1 <= s <= len(season_labels):
            subplot_titles.append(season_labels[s - 1])
        else:
            subplot_titles.append(str(s))

    if panel_group_names is not None:
        from yohou.plotting._utils import _group_panel_columns  # noqa: PLC0415

        groups, all_members = _group_panel_columns(plot_columns)

        # Build one figure per member, groups overlaid by colour.
        figures: dict[str, go.Figure] = {}

        for member in all_members:
            color_mgr = PanelColorManager(color_palette)
            legend_tracker = LegendTracker(show_legend=show_legend)

            mfig = _create_subplots(
                resampler,
                rows=nrow,
                cols=facet_n_cols,
                subplot_titles=subplot_titles,
                shared_yaxes=True,
                vertical_spacing=max(0.04, 0.3 / nrow),
            )

            for si, season_val in enumerate(seasons):
                r = si // facet_n_cols + 1
                c_pos = si % facet_n_cols + 1
                season_df = df_aug.filter(pl.col("season") == season_val)

                for gname, group_cols in groups.items():
                    # Find this member's column within the group.
                    col_name = next(
                        (c for c in group_cols if _member_name(c) == member),
                        None,
                    )
                    if col_name is None or col_name not in season_df.columns:
                        continue

                    display_name = gname
                    col_color = color_mgr.get_color(gname)

                    lines_info = None
                    if kind != "mean":
                        lines_info = _add_kind_traces(
                            mfig,
                            kind=kind,
                            season_df=season_df,
                            col_name=col_name,
                            display_name=display_name,
                            col_color=col_color,
                            overlay_opacity=overlay_opacity,
                            line_width=line_width,
                            row=r,
                            col=c_pos,
                            legend_tracker=legend_tracker,
                        )

                    if kind in ("mean", "lines"):
                        agg_df = season_df.group_by("cycle").agg(pl.col(col_name).mean()).sort("cycle")
                        cycles = agg_df["cycle"].to_list()
                        values = agg_df[col_name].to_list()
                        if not cycles:
                            continue

                        if lines_info is not None:
                            x_vals = lines_info["midpoints"]
                            x_mean_range = [0.0, lines_info["total"][0]]
                        else:
                            x_vals = cycles
                            x_mean_range = [cycles[0], cycles[-1]]

                        mfig.add_trace(
                            go.Scatter(
                                x=x_vals,
                                y=values,
                                mode="lines+markers",
                                line={"color": col_color, "width": line_width},
                                marker={"size": marker_size, "opacity": marker_opacity},
                                connectgaps=connect_gaps,
                                hovertemplate=_make_hovertemplate(
                                    display_name,
                                    "Cycle",
                                    "Value",
                                ),
                                name=display_name,
                                legendgroup=display_name,
                                showlegend=legend_tracker.should_show(display_name),
                            ),
                            row=r,
                            col=c_pos,
                        )

                        if show_mean:
                            mean_val = agg_df[col_name].mean()
                            if mean_val is not None and len(cycles) >= 2:
                                mfig.add_trace(
                                    go.Scatter(
                                        x=x_mean_range,
                                        y=[mean_val, mean_val],
                                        mode="lines",
                                        line={
                                            "color": col_color,
                                            "width": mean_width,
                                            "dash": mean_dash,
                                        },
                                        name=f"Mean ({display_name})",
                                        legendgroup=display_name,
                                        showlegend=False,
                                        hovertemplate=f"Mean: {mean_val:.2f}<extra></extra>",
                                    ),
                                    row=r,
                                    col=c_pos,
                                )

            mfig = apply_default_layout(
                mfig,
                title=title or f"Seasonal Subseries - {member}",
                x_label=x_label,
                y_label=y_label,
                width=width,
                height=height if height is not None else max(450, nrow * 280),
            )
            mfig.update_layout(showlegend=show_legend)
            figures[member] = mfig

        if len(figures) == 1:
            return next(iter(figures.values()))
        return figures

    # Non-panel path
    fig = _create_subplots(
        resampler,
        rows=nrow,
        cols=facet_n_cols,
        subplot_titles=subplot_titles,
        shared_yaxes=True,
        vertical_spacing=max(0.04, 0.3 / nrow),
    )

    colors = resolve_color_palette(color_palette, len(plot_columns))
    legend_tracker = LegendTracker(show_legend=show_legend)

    for si, season_val in enumerate(seasons):
        r = si // facet_n_cols + 1
        c = si % facet_n_cols + 1
        season_df = df_aug.filter(pl.col("season") == season_val)

        for ci, col_name in enumerate(plot_columns):
            col_color = colors[ci % len(colors)]

            lines_info = None
            if kind != "mean":
                lines_info = _add_kind_traces(
                    fig,
                    kind=kind,
                    season_df=season_df,
                    col_name=col_name,
                    display_name=col_name,
                    col_color=col_color,
                    overlay_opacity=overlay_opacity,
                    line_width=line_width,
                    row=r,
                    col=c,
                    legend_tracker=legend_tracker,
                )

            if kind in ("mean", "lines"):
                agg_df = season_df.group_by("cycle").agg(pl.col(col_name).mean()).sort("cycle")
                cycles = agg_df["cycle"].to_list()
                values = agg_df[col_name].to_list()
                if not cycles:
                    continue

                # When kind="lines", remap x-values to the numeric axis.
                if lines_info is not None:
                    x_vals = lines_info["midpoints"]
                    x_mean_range = [0.0, lines_info["total"][0]]
                else:
                    x_vals = cycles
                    x_mean_range = [cycles[0], cycles[-1]]

                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=values,
                        mode="lines+markers",
                        line={"color": col_color, "width": line_width},
                        marker={"size": marker_size, "opacity": marker_opacity},
                        connectgaps=connect_gaps,
                        hovertemplate=_make_hovertemplate(col_name, "Cycle", "Value"),
                        name=col_name,
                        legendgroup=col_name,
                        showlegend=legend_tracker.should_show(col_name),
                    ),
                    row=r,
                    col=c,
                )

                if show_mean:
                    mean_val = agg_df[col_name].mean()
                    if mean_val is not None and len(cycles) >= 2:
                        fig.add_trace(
                            go.Scatter(
                                x=x_mean_range,
                                y=[mean_val, mean_val],
                                mode="lines",
                                line={
                                    "color": col_color,
                                    "width": mean_width,
                                    "dash": mean_dash,
                                },
                                name=f"Mean ({col_name})",
                                legendgroup=col_name,
                                showlegend=False,
                                hovertemplate=f"Mean: {mean_val:.2f}<extra></extra>",
                            ),
                            row=r,
                            col=c,
                        )

    fig = apply_default_layout(
        fig,
        title=title or "Seasonal Subseries",
        x_label=x_label,
        y_label=y_label,
        width=width,
        height=height if height is not None else max(450, nrow * 280),
    )
    fig.update_layout(showlegend=show_legend)

    return fig


def plot_lag_scatter(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    lags: int | list[int] | None = None,
    seasonality: str | None = None,
    show_diagonal: bool = True,
    show_regression: bool = False,
    panel_group_names: list[str] | None = None,
    facet_n_cols: int = 3,
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    marker_size: float = 4.0,
    marker_opacity: float = 0.6,
) -> go.Figure | dict[str, go.Figure]:
    """Plot scatter plots of y(t) vs y(t-lag) for analysing temporal dependencies.

    Creates scatter plots showing the relationship between current values and
    lagged values, useful for identifying AR patterns and lag-specific
    correlations.  When ``lags`` has more than one entry, each lag
    gets its own subplot arranged in a grid.  Optionally colour points by
    season using the ``seasonality`` parameter.

    For panel data, each member gets its own figure with one subplot per lag
    and all groups overlaid by colour.  When there are multiple members, a
    ``dict[str, go.Figure]`` keyed by member name is returned.

    For non-panel multivariate data, each column gets its own figure with lag
    subplots, and a ``dict[str, go.Figure]`` keyed by column name is returned.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to analyse. If None, uses all numeric columns except 'time'.
    lags : list[int] | None, default=None
        Lag values to plot. Each lag gets its own subplot when there are
        multiple entries.  Defaults to ``[1]`` when ``None``.
    seasonality : str | None, default=None
        When set, colour markers by season.  Accepts ``"month"``,
        ``"quarter"``, ``"weekday"``, ``"week"`` or ``"hour"``.
    show_diagonal : bool, default=True
        Show a diagonal reference line (y = x).
    show_regression : bool, default=False
        Show a linear regression line fitted to the data.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot.
    facet_n_cols : int, default=3
        Number of columns in the subplot grid.
    color_palette : list[str] | None, default=None
        Custom color palette.
    show_legend : bool, default=True
        Whether to show the legend.
    title : str | None, default=None
        Plot title.
    x_label : str | None, default=None
        X-axis label. Defaults to ``"y(t-{lag})"`` per subplot.
    y_label : str | None, default=None
        Y-axis label. Defaults to ``"y(t)"`` per subplot.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    marker_size : float, default=4.0
        Size of scatter markers.
    marker_opacity : float, default=0.6
        Opacity of scatter markers.

    Returns
    -------
    go.Figure | dict[str, go.Figure]
        Single figure for univariate data (one column or one panel member).
        Dictionary of figures keyed by column/member name for multivariate
        data.

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
    >>> fig = plot_lag_scatter(df, columns="y", lags=[1])
    >>> len(fig.data) > 0
    True

    See Also
    --------
    [`plot_autocorrelation`][yohou.plotting.plot_autocorrelation] : Plot autocorrelation function.
    """
    # Validate inputs
    validate_plotting_data(df)
    validate_plotting_params(width=width, height=height)

    if lags is None:
        lags = [1]
    elif isinstance(lags, int):
        lags = list(range(1, lags + 1))

    # Auto-detect panel data
    _, _panel_groups = inspect_panel(df)
    if panel_group_names is None and columns is None and _panel_groups:
        panel_group_names = []

    # Panel path
    if panel_group_names is not None:
        panel_cols = resolve_panel_columns(df, panel_group_names, columns)
        groups, all_members = _group_panel_columns(panel_cols)

        n_lags = len(lags)
        figures: dict[str, go.Figure] = {}

        for member in all_members:
            color_mgr = PanelColorManager(color_palette)
            legend_tracker = LegendTracker(show_legend=show_legend)

            ncols_grid = min(facet_n_cols, n_lags)
            nrows_grid = math.ceil(n_lags / ncols_grid)
            subtitles = [f"lag {k}" for k in lags]

            fig = make_subplots(
                rows=nrows_grid,
                cols=ncols_grid,
                subplot_titles=subtitles,
                horizontal_spacing=0.08,
                vertical_spacing=max(0.04, 0.3 / nrows_grid),
            )

            for li, lag in enumerate(lags):
                r = li // ncols_grid + 1
                c = li % ncols_grid + 1
                all_vals: list[float] = []

                for gname, group_cols in groups.items():
                    col_name = next(
                        (col for col in group_cols if _member_name(col) == member),
                        None,
                    )
                    if col_name is None or col_name not in df.columns:
                        continue

                    dl = (
                        df
                        .select(col_name)
                        .with_columns(
                            pl.col(col_name).shift(lag).alias("lagged"),
                        )
                        .drop_nulls()
                    )
                    if len(dl) == 0:
                        continue

                    group_color = color_mgr.get_color(gname)
                    fig.add_trace(
                        go.Scattergl(
                            x=dl["lagged"],
                            y=dl[col_name],
                            mode="markers",
                            marker={"size": marker_size, "color": group_color, "opacity": marker_opacity},
                            name=gname,
                            legendgroup=gname,
                            showlegend=legend_tracker.should_show(gname),
                            hovertemplate=(
                                f"<b>{gname}: {member}</b><br>y(t-{lag}): %{{x:.2f}}<br>y(t): %{{y:.2f}}<extra></extra>"
                            ),
                        ),
                        row=r,
                        col=c,
                    )

                    all_vals.extend(dl[col_name].to_list())
                    all_vals.extend(dl["lagged"].to_list())

                if show_diagonal and all_vals:
                    vmin = min(all_vals)
                    vmax = max(all_vals)
                    fig.add_trace(
                        go.Scatter(
                            x=[vmin, vmax],
                            y=[vmin, vmax],
                            mode="lines",
                            line={"dash": "dash", "color": "#94a3b8", "width": 1},
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=r,
                        col=c,
                    )

                fig.update_xaxes(title_text=x_label or f"y(t-{lag})", row=r, col=c)
                fig.update_yaxes(title_text=y_label or "y(t)", row=r, col=c)

            cell_size = 260
            default_width = max(DEFAULT_WIDTH, ncols_grid * cell_size + 100)
            default_height = max(400, nrows_grid * cell_size + 100)

            fig = apply_default_layout(
                fig,
                title=title or f"Lag Scatter - {member}",
                x_label=None,
                y_label=None,
                width=width or default_width,
                height=height or default_height,
            )
            fig.update_layout(showlegend=show_legend)
            figures[member] = fig

        if len(figures) == 1:
            return next(iter(figures.values()))
        return figures

    # Non-panel path
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])

    season_labels: list[str] | None = None
    season_colors: list[str] | None = None
    df_aug: pl.DataFrame | None = None

    if seasonality is not None:
        df_aug = _add_season_and_cycle(df, seasonality)
        label_map = _SEASON_LABELS_MAP.get(seasonality)
        seasons_raw = sorted(df_aug["season"].unique().to_list())
        n_seasons = len(seasons_raw)
        season_labels = (
            [label_map[s - 1] for s in seasons_raw] if label_map is not None else [str(s) for s in seasons_raw]
        )
        season_colors = resolve_color_palette(color_palette, n_seasons)

    figures = {}

    for col in plot_columns:
        use_subplots = len(lags) > 1

        if use_subplots:
            n_lags = len(lags)
            ncols = min(facet_n_cols, n_lags)
            nrows = math.ceil(n_lags / ncols)

            subtitles = [f"lag {k}" for k in lags]
            fig = make_subplots(
                rows=nrows,
                cols=ncols,
                subplot_titles=subtitles,
                horizontal_spacing=0.08,
                vertical_spacing=max(0.04, 0.3 / nrows),
            )

            for lag_idx, lag in enumerate(lags):
                r = lag_idx // ncols + 1
                c = lag_idx % ncols + 1
                is_first_cell = lag_idx == 0

                if df_aug is not None and season_labels is not None and season_colors is not None:
                    seasons_raw = sorted(df_aug["season"].unique().to_list())
                    for si, (sval, slabel) in enumerate(zip(seasons_raw, season_labels, strict=False)):
                        sub = df_aug.filter(pl.col("season") == sval)
                        dl = sub.with_columns(pl.col(col).shift(lag).alias("lagged")).drop_nulls()
                        if len(dl) == 0:
                            continue
                        fig.add_trace(
                            go.Scattergl(
                                x=dl["lagged"],
                                y=dl[col],
                                mode="markers",
                                marker={
                                    "size": marker_size,
                                    "color": season_colors[si % len(season_colors)],
                                    "opacity": marker_opacity,
                                },
                                name=slabel,
                                legendgroup=slabel,
                                legendgrouptitle=(
                                    {"text": seasonality.title()}  # ty: ignore[unresolved-attribute]
                                    if is_first_cell and si == 0
                                    else None
                                ),
                                showlegend=is_first_cell,
                                hovertemplate=(
                                    f"<b>{col}</b> ({slabel})<br>"
                                    f"y(t-{lag}): %{{x:.2f}}<br>"
                                    f"y(t): %{{y:.2f}}<extra></extra>"
                                ),
                            ),
                            row=r,
                            col=c,
                        )
                else:
                    colors = resolve_color_palette(color_palette, 1)
                    dl = df.with_columns(pl.col(col).shift(lag).alias("lagged")).drop_nulls()
                    fig.add_trace(
                        go.Scattergl(
                            x=dl["lagged"],
                            y=dl[col],
                            mode="markers",
                            marker={
                                "size": marker_size,
                                "color": colors[0],
                                "opacity": marker_opacity,
                            },
                            name=col,
                            legendgroup=col,
                            showlegend=is_first_cell,
                            hovertemplate=(
                                f"<b>{col}</b><br>y(t-{lag}): %{{x:.2f}}<br>y(t): %{{y:.2f}}<extra></extra>"
                            ),
                        ),
                        row=r,
                        col=c,
                    )

                if show_diagonal:
                    source = df_aug if df_aug is not None else df
                    dl_all = source.with_columns(pl.col(col).shift(lag).alias("lagged")).drop_nulls()
                    vmin = min(
                        float(dl_all[col].min()),  # ty: ignore[invalid-argument-type]
                        float(dl_all["lagged"].min()),  # ty: ignore[invalid-argument-type]
                    )
                    vmax = max(
                        float(dl_all[col].max()),  # ty: ignore[invalid-argument-type]
                        float(dl_all["lagged"].max()),  # ty: ignore[invalid-argument-type]
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[vmin, vmax],
                            y=[vmin, vmax],
                            mode="lines",
                            line={"dash": "dash", "color": "#94a3b8", "width": 1},
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=r,
                        col=c,
                    )

                fig.update_xaxes(title_text=x_label or f"y(t-{lag})", row=r, col=c)
                fig.update_yaxes(title_text=y_label or "y(t)", row=r, col=c)

            cell_size = 260
            default_width = max(DEFAULT_WIDTH, ncols * cell_size + 100)
            default_height = max(400, nrows * cell_size + 100)

            fig = apply_default_layout(
                fig,
                title=title or "Lag Scatter",
                x_label=None,
                y_label=None,
                width=width or default_width,
                height=height or default_height,
            )
        else:
            fig = go.Figure()
            lag = lags[0]

            if df_aug is not None and season_labels is not None and season_colors is not None:
                seasons_raw = sorted(df_aug["season"].unique().to_list())
                for si, (sval, slabel) in enumerate(zip(seasons_raw, season_labels, strict=False)):
                    sub = df_aug.filter(pl.col("season") == sval)
                    dl = sub.with_columns(pl.col(col).shift(lag).alias("lagged")).drop_nulls()
                    if len(dl) == 0:
                        continue
                    fig.add_trace(
                        go.Scattergl(
                            x=dl["lagged"],
                            y=dl[col],
                            mode="markers",
                            marker={
                                "size": marker_size,
                                "color": season_colors[si % len(season_colors)],
                                "opacity": marker_opacity,
                            },
                            name=slabel,
                            legendgroup=slabel,
                            legendgrouptitle=(
                                {"text": seasonality.title()} if si == 0 else None  # ty: ignore[unresolved-attribute]
                            ),
                            showlegend=True,
                            hovertemplate=(
                                f"<b>{col}</b> ({slabel})<br>y(t-{lag}): %{{x:.2f}}<br>y(t): %{{y:.2f}}<extra></extra>"
                            ),
                        )
                    )
            else:
                colors = resolve_color_palette(color_palette, 1)
                dl = df.with_columns(pl.col(col).shift(lag).alias("lagged")).drop_nulls()
                fig.add_trace(
                    go.Scattergl(
                        x=dl["lagged"],
                        y=dl[col],
                        mode="markers",
                        marker={
                            "size": marker_size,
                            "color": colors[0],
                            "opacity": marker_opacity,
                        },
                        name=f"{col} (lag={lag})",
                        hovertemplate=(f"<b>{col}</b><br>y(t-{lag}): %{{x:.2f}}<br>y(t): %{{y:.2f}}<extra></extra>"),
                    )
                )

            if show_diagonal:
                source = df_aug if df_aug is not None else df
                dl_all = source.with_columns(pl.col(col).shift(lag).alias("lagged")).drop_nulls()
                y_lagged_min = dl_all["lagged"].min()
                y_lagged_max = dl_all["lagged"].max()
                y_current_min = dl_all[col].min()
                y_current_max = dl_all[col].max()
                if y_lagged_min is not None and y_current_min is not None:
                    min_val = min(cast(float, y_lagged_min), cast(float, y_current_min))
                    max_val = max(cast(float, y_lagged_max), cast(float, y_current_max))
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

            if show_regression:
                source = df_aug if df_aug is not None else df
                dl_all = source.with_columns(pl.col(col).shift(lag).alias("lagged")).drop_nulls()
                y_lagged = dl_all["lagged"]
                y_current = dl_all[col]
                x_mean = y_lagged.mean()
                y_mean = y_current.mean()

                numerator = float(((y_lagged - x_mean) * (y_current - y_mean)).sum())
                denominator = float(((y_lagged - x_mean) ** 2).sum())

                if denominator != 0 and x_mean is not None and y_mean is not None:
                    slope = numerator / denominator
                    intercept = cast(float, y_mean) - slope * cast(float, x_mean)

                    y_lagged_min = y_lagged.min()
                    y_lagged_max = y_lagged.max()
                    if y_lagged_min is not None and y_lagged_max is not None:
                        x_line = [cast(float, y_lagged_min), cast(float, y_lagged_max)]
                        y_line = [slope * x + intercept for x in x_line]
                        reg_colors = resolve_color_palette(color_palette, 1)
                        fig.add_trace(
                            go.Scatter(
                                x=x_line,
                                y=y_line,
                                mode="lines",
                                line={"color": reg_colors[0], "width": 2},
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )

            fig = apply_default_layout(
                fig,
                title=title or f"Lag {lag} Scatter Plot",
                x_label=x_label or f"y(t-{lag})",
                y_label=y_label or "y(t)",
                width=width,
                height=height,
            )

        fig.update_layout(showlegend=show_legend)
        figures[col] = fig

    if len(figures) == 1:
        return next(iter(figures.values()))
    return figures


def _compute_ccf(x: np.ndarray, y: np.ndarray, max_lags: int) -> list[float]:
    """Compute cross-correlation between *x* and *y* for lags in ``[-max_lags, max_lags]``."""
    n = len(x)
    x_mean, y_mean = x.mean(), y.mean()
    x_std, y_std = x.std(), y.std()
    lag_values = range(-max_lags, max_lags + 1)
    ccf: list[float] = []
    for lag in lag_values:
        if lag < 0:
            xa, ya = x[: n + lag], y[-lag:]
        elif lag > 0:
            xa, ya = x[lag:], y[: n - lag]
        else:
            xa, ya = x, y
        if len(xa) > 0 and x_std != 0 and y_std != 0:
            ccf.append(float(((xa - x_mean) * (ya - y_mean)).mean() / (x_std * y_std)))
        else:
            ccf.append(0.0)
    return ccf


def _upper_triangle_pairs(names: list[str]) -> list[tuple[str, str]]:
    """Return unique upper-triangle pairs from *names*."""
    return [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]


def plot_cross_correlation(
    df: pl.DataFrame,
    *,
    columns: list[str] | None = None,
    max_lags: int | None = None,
    confidence_level: float = 0.95,
    panel_group_names: list[str] | None = None,
    facet_n_cols: int = 2,
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot cross-correlation function (CCF) between time series pairs.

    Computes and visualizes the cross-correlation between pairs of series at
    various lags, useful for identifying lead-lag relationships and temporal
    dependencies.

    When ``columns`` is ``None`` (or has more than 2 entries), all unique
    upper-triangle pairs are computed and arranged in a subplot grid.  When
    exactly two columns are given, a single CCF plot is produced (matching
    legacy behaviour).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : list[str] | None, default=None
        Column names to cross-correlate.  When ``None``, all numeric
        columns are used and every unique upper-triangle pair is plotted.
        When exactly two columns are given, a single CCF plot is produced.
    max_lags : int | None, default=None
        Number of lags to compute (both positive and negative).
        If None, uses ``min(len(df) // 2, 40)``.
    confidence_level : float, default=0.95
        Confidence level for confidence bands (e.g. ``0.95`` for 95%).
    panel_group_names : list[str] | None, default=None
        Panel group prefixes.  When set (or auto-detected), CCF is
        computed between members within each group using an
        upper-triangle matrix layout.
    facet_n_cols : int, default=2
        Number of columns in the subplot grid.
    color_palette : list[str] | None, default=None
        Custom color palette for pair traces.
    show_legend : bool, default=True
        Whether to show the legend.
    title : str | None, default=None
        Plot title.
    x_label : str | None, default=None
        X-axis label. Defaults to ``"Lag"``.
    y_label : str | None, default=None
        Y-axis label. Defaults to ``"Cross-Correlation"``.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.

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
    >>> fig = plot_cross_correlation(df, columns=["x", "y"], max_lags=20)
    >>> len(fig.data) > 0
    True

    See Also
    --------
    [`plot_autocorrelation`][yohou.plotting.plot_autocorrelation] : Plot autocorrelation function.
    """
    # Validate inputs
    validate_plotting_data(df)
    validate_plotting_params(width=width, height=height)

    if max_lags is None:
        max_lags = min(len(df) // 2, 40)

    _lag_vals = list(range(-max_lags, max_lags + 1))

    def _add_ccf_bar(
        fig: go.Figure,
        lag_vals: list[int],
        ccf: list[float],
        pair_color: str,
        pair_label: str,
        show_in_legend: bool,
        *,
        row: int | None = None,
        col: int | None = None,
    ) -> None:
        """Add a single ACF-style bar trace."""
        trace_kwargs: dict = {}
        if row is not None and col is not None:
            trace_kwargs["row"] = row
            trace_kwargs["col"] = col
        fig.add_trace(
            go.Bar(
                x=lag_vals,
                y=ccf,
                marker={"color": pair_color},
                name=pair_label,
                showlegend=show_in_legend,
                legendgroup=pair_label,
                hovertemplate=_make_hovertemplate(pair_label, "Lag", "CCF", decimals=3),
            ),
            **trace_kwargs,
        )

    def _add_ccf_bands(
        fig: go.Figure,
        n_pts: int,
        *,
        row: int | None = None,
        col: int | None = None,
    ) -> None:
        """Add confidence bands and zero line."""
        hline_kw: dict = {}
        if row is not None and col is not None:
            hline_kw["row"] = row
            hline_kw["col"] = col
        if confidence_level > 0:
            cb = norm.ppf(1 - (1 - confidence_level) / 2) / math.sqrt(n_pts)
            fig.add_hline(y=cb, line={"dash": "dash", "color": "#DC2626", "width": 1}, **hline_kw)
            fig.add_hline(y=-cb, line={"dash": "dash", "color": "#DC2626", "width": 1}, **hline_kw)
        fig.add_hline(y=0, line={"color": "#64748B", "width": 1}, **hline_kw)

    # Auto-detect panel data
    if panel_group_names is None and _auto_detect_panel(df) and columns is None:
        panel_group_names = []

    if panel_group_names is not None:
        _panel_cols = resolve_panel_columns(df, panel_group_names, columns)
        groups, _all_members = _group_panel_columns(_panel_cols)
        color_mgr = PanelColorManager(color_palette)

        # Build per-group pair lists
        group_pairs: list[tuple[str, list[tuple[str, str]]]] = []
        for gname, gcols in groups.items():
            member_names = [_member_name(c) for c in gcols]
            pairs = _upper_triangle_pairs(member_names)
            if pairs:
                group_pairs.append((gname, pairs))

        if not group_pairs:
            msg = f"Need at least 2 members per group for cross-correlation. Groups: {list(groups.keys())}"
            raise ValueError(msg)

        # All groups + pairs in one figure
        all_cells: list[tuple[str, str, str]] = []
        for gname, pairs in group_pairs:
            for a, b in pairs:
                all_cells.append((gname, a, b))

        n = len(all_cells)
        n_cols_grid = min(n, facet_n_cols)
        n_rows_grid = math.ceil(n / n_cols_grid)
        legend_tracker = LegendTracker(show_legend=show_legend)

        fig = make_subplots(
            rows=n_rows_grid,
            cols=n_cols_grid,
            subplot_titles=[f"{gn}: {a} vs {b}" for gn, a, b in all_cells],
            vertical_spacing=_subplot_spacing(n_rows_grid),
            horizontal_spacing=_subplot_spacing(n_cols_grid) if n_cols_grid > 1 else 0.08,
        )

        for idx, (gname, a_name, b_name) in enumerate(all_cells):
            r = idx // n_cols_grid + 1
            c = idx % n_cols_grid + 1
            col_a = f"{gname}__{a_name}"
            col_b = f"{gname}__{b_name}"
            clean = df.select(col_a, col_b).drop_nulls()
            x_arr = clean[col_a].to_numpy()
            y_arr = clean[col_b].to_numpy()
            ccf = _compute_ccf(x_arr, y_arr, max_lags)
            pair_label = f"{a_name} vs {b_name}"
            pair_color = color_mgr.get_color(pair_label)
            _add_ccf_bar(
                fig,
                _lag_vals,
                ccf,
                pair_color,
                pair_label,
                legend_tracker.should_show(pair_label),
                row=r,
                col=c,
            )
            _add_ccf_bands(fig, len(x_arr), row=r, col=c)

        fig = apply_default_layout(
            fig,
            title=title or "Cross-Correlation",
            x_label=x_label or "Lag",
            y_label=y_label or "Cross-Correlation",
            width=width,
            height=height,
        )
        fig.update_layout(showlegend=show_legend)
        return fig

    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])

    if len(plot_columns) < 2:  # noqa: PLR2004
        msg = f"Cross-correlation requires at least 2 columns, got {len(plot_columns)}: {plot_columns}"
        raise ValueError(msg)

    pairs = _upper_triangle_pairs(plot_columns)
    color_mgr = PanelColorManager(color_palette)
    legend_tracker = LegendTracker()

    if len(pairs) == 1:
        # Single pair - flat figure
        x_column, y_column = pairs[0]
        clean = df.select(x_column, y_column).drop_nulls()
        x_arr = clean[x_column].to_numpy()
        y_arr = clean[y_column].to_numpy()
        n_pts = len(x_arr)
        ccf = _compute_ccf(x_arr, y_arr, max_lags)
        pair_label = f"{x_column} vs {y_column}"
        pair_color = color_mgr.get_color(pair_label)

        fig = go.Figure()
        _add_ccf_bar(fig, _lag_vals, ccf, pair_color, pair_label, True)

        if confidence_level > 0:
            cb = norm.ppf(1 - (1 - confidence_level) / 2) / math.sqrt(n_pts)
            ci_pct = f"{confidence_level:.0%}"
            fig.add_hline(
                y=cb,
                line={"dash": "dash", "color": "#DC2626", "width": 1},
                annotation_text=f"{ci_pct} CI",
                annotation_position="right",
            )
            fig.add_hline(y=-cb, line={"dash": "dash", "color": "#DC2626", "width": 1})
        fig.add_hline(y=0, line={"color": "#64748B", "width": 1})

        fig = apply_default_layout(
            fig,
            title=title or "Cross-Correlation",
            x_label=x_label or "Lag",
            y_label=y_label or "Cross-Correlation",
            width=width,
            height=height,
        )
        fig.update_layout(showlegend=show_legend)
        return fig

    # Multiple pairs - upper-triangle subplot grid
    n = len(pairs)
    n_cols_grid = min(n, facet_n_cols)
    n_rows_grid = math.ceil(n / n_cols_grid)

    fig = make_subplots(
        rows=n_rows_grid,
        cols=n_cols_grid,
        subplot_titles=[f"{a} vs {b}" for a, b in pairs],
        vertical_spacing=_subplot_spacing(n_rows_grid),
        horizontal_spacing=_subplot_spacing(n_cols_grid) if n_cols_grid > 1 else 0.08,
    )

    for idx, (col_a, col_b) in enumerate(pairs):
        r = idx // n_cols_grid + 1
        c = idx % n_cols_grid + 1
        clean = df.select(col_a, col_b).drop_nulls()
        x_arr = clean[col_a].to_numpy()
        y_arr = clean[col_b].to_numpy()
        ccf = _compute_ccf(x_arr, y_arr, max_lags)
        pair_label = f"{col_a} vs {col_b}"
        pair_color = color_mgr.get_color(pair_label)
        _add_ccf_bar(
            fig,
            _lag_vals,
            ccf,
            pair_color,
            pair_label,
            legend_tracker.should_show(pair_label),
            row=r,
            col=c,
        )
        _add_ccf_bands(fig, len(x_arr), row=r, col=c)

    fig = apply_default_layout(
        fig,
        title=title or "Cross-Correlation",
        x_label=x_label or "Lag",
        y_label=y_label or "Cross-Correlation",
        width=width,
        height=height,
    )
    fig.update_layout(showlegend=show_legend)
    return fig


def _subsample_df(
    df_work: pl.DataFrame,
    max_points: int | None,
    seasons_raw: list[int] | None,
) -> pl.DataFrame:
    """Randomly subsample *df_work*, optionally stratified by season."""
    if max_points is None or len(df_work) <= max_points:
        return df_work
    if seasons_raw is not None:
        n_per = max(1, max_points // len(seasons_raw))
        parts = [
            df_work.filter(pl.col("season") == s).sample(
                n=min(n_per, df_work.filter(pl.col("season") == s).height),
                seed=42,
            )
            for s in seasons_raw
        ]
        return pl.concat(parts)
    return df_work.sample(n=max_points, seed=42)


def _render_scatter_block(  # noqa: PLR0913
    fig: go.Figure,
    df_work: pl.DataFrame,
    plot_columns: list[str],
    *,
    row_offset: int,
    col_offset: int,
    n: int,
    seasonality: str | None,
    season_labels: list[str] | None,
    season_colors: list[str] | None,
    seasons_raw: list[int] | None,
    diagonal: str | None,
    show_correlation: bool,
    uniform_color: str,
    marker_size: float,
    marker_opacity: float,
    corr_font_size: int,
    show_legend: bool,
    total_cols: int,
    gaussian_kde,  # noqa: ANN001
    pearsonr,  # noqa: ANN001
) -> None:
    """Render an M×M scatter matrix block into *fig* at the given offset."""
    for ri in range(n):
        for ci in range(n):
            row = row_offset + ri + 1
            col = col_offset + ci + 1
            col_y = plot_columns[ri]
            col_x = plot_columns[ci]

            if ri == ci:
                if diagonal is None:
                    pass
                elif diagonal == "kde":
                    vals = df_work[col_x].drop_nulls().to_numpy()
                    if len(vals) > 1:
                        try:
                            kde = gaussian_kde(vals)
                        except np.linalg.LinAlgError:
                            pass
                        else:
                            data_range = float(vals.max()) - float(vals.min())
                            if data_range > 0 and kde.factor > 1e-10:
                                x_grid = np.linspace(float(vals.min()), float(vals.max()), 200)
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_grid,
                                        y=kde(x_grid),
                                        mode="lines",
                                        line={"color": uniform_color, "width": 1.5},
                                        showlegend=False,
                                        hoverinfo="skip",
                                    ),
                                    row=row,
                                    col=col,
                                )
                elif diagonal == "histogram":
                    vals = df_work[col_x].drop_nulls().to_numpy()
                    fig.add_trace(
                        go.Histogram(
                            x=vals,
                            marker_color=uniform_color,
                            opacity=0.7,
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )

            elif ri > ci:
                if season_labels is not None and season_colors is not None and seasons_raw is not None:
                    for si, (sval, slabel) in enumerate(zip(seasons_raw, season_labels, strict=False)):
                        sub = df_work.filter(pl.col("season") == sval).drop_nulls(subset=[col_x, col_y])
                        if len(sub) == 0:
                            continue
                        is_first = ri == 1 and ci == 0
                        fig.add_trace(
                            go.Scattergl(
                                x=sub[col_x],
                                y=sub[col_y],
                                mode="markers",
                                marker={
                                    "size": marker_size,
                                    "color": season_colors[si % len(season_colors)],
                                    "opacity": marker_opacity,
                                },
                                name=slabel,
                                legendgroup=slabel,
                                showlegend=show_legend and is_first,
                                hoverinfo="skip",
                            ),
                            row=row,
                            col=col,
                        )
                else:
                    sub = df_work.drop_nulls(subset=[col_x, col_y])
                    fig.add_trace(
                        go.Scattergl(
                            x=sub[col_x],
                            y=sub[col_y],
                            mode="markers",
                            marker={
                                "size": marker_size,
                                "color": uniform_color,
                                "opacity": marker_opacity,
                            },
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )

            elif show_correlation:
                sub = df_work.drop_nulls(subset=[col_x, col_y])
                if len(sub) > 2:
                    r_val, _ = pearsonr(sub[col_x].to_numpy(), sub[col_y].to_numpy())
                    font_sz = max(10, int(corr_font_size * abs(r_val)))
                    axis_idx = (row - 1) * total_cols + col
                    x_key = f"xaxis{axis_idx}" if axis_idx > 1 else "xaxis"
                    y_key = f"yaxis{axis_idx}" if axis_idx > 1 else "yaxis"
                    x_domain = fig.layout[x_key].domain
                    y_domain = fig.layout[y_key].domain
                    fig.add_annotation(
                        text=f"{r_val:.2f}",
                        xref="paper",
                        yref="paper",
                        x=(x_domain[0] + x_domain[1]) / 2,
                        y=(y_domain[0] + y_domain[1]) / 2,
                        showarrow=False,
                        font={"size": font_sz},
                    )


def _prepare_season_info(
    df: pl.DataFrame,
    seasonality: str | None,
    color_palette: list[str] | None,
) -> tuple[pl.DataFrame, list[str] | None, list[str] | None, list[int] | None]:
    """Return (df_work, season_labels, season_colors, seasons_raw)."""
    if seasonality is None:
        return df, None, None, None
    df_work = _add_season_and_cycle(df, seasonality)
    label_map = _SEASON_LABELS_MAP.get(seasonality)
    seasons_raw = sorted(df_work["season"].unique().to_list())
    n_seasons = len(seasons_raw)
    season_labels = [label_map[s - 1] for s in seasons_raw] if label_map is not None else [str(s) for s in seasons_raw]
    season_colors = resolve_color_palette(color_palette, n_seasons)
    return df_work, season_labels, season_colors, seasons_raw


def _scatter_matrix_facet(  # noqa: PLR0913
    *,
    df: pl.DataFrame,
    groups: dict[str, list[str]],
    seasonality: str | None,
    diagonal: str | None,
    show_correlation: bool,
    max_points: int | None,
    color_palette: list[str] | None,
    title: str | None,
    width: int | None,
    height: int | None,
    gaussian_kde,  # noqa: ANN001
    pearsonr,  # noqa: ANN001
    marker_size: float = 3.0,
    marker_opacity: float = 0.5,
) -> dict[str, go.Figure]:
    """One figure per panel group, each containing its own M×M scatter matrix."""
    corr_font_size = 28

    figures: dict[str, go.Figure] = {}

    for gname, gcols in groups.items():
        base_names = [c.split("__", 1)[1] if "__" in c else c for c in gcols]
        sub_df = df.select(["time", *gcols]).rename(dict(zip(gcols, base_names, strict=False)))
        m = len(base_names)
        if m < 2:
            continue

        df_work, season_labels, season_colors, seasons_raw = _prepare_season_info(sub_df, seasonality, color_palette)
        df_work = _subsample_df(df_work, max_points, seasons_raw)
        uniform_color = resolve_color_palette(color_palette, 1)[0]

        fig = make_subplots(
            rows=m,
            cols=m,
            horizontal_spacing=_subplot_spacing(m),
            vertical_spacing=_subplot_spacing(m),
        )

        _render_scatter_block(
            fig,
            df_work,
            base_names,
            row_offset=0,
            col_offset=0,
            n=m,
            seasonality=seasonality,
            season_labels=season_labels,
            season_colors=season_colors,
            seasons_raw=seasons_raw,
            diagonal=diagonal,
            show_correlation=show_correlation,
            uniform_color=uniform_color,
            marker_size=marker_size,
            marker_opacity=marker_opacity,
            corr_font_size=corr_font_size,
            show_legend=True,
            total_cols=m,
            gaussian_kde=gaussian_kde,
            pearsonr=pearsonr,
        )

        # Axis labels on outermost edges
        for ri in range(m):
            for ci in range(m):
                axis_idx = ri * m + ci + 1
                x_key = f"xaxis{axis_idx}" if axis_idx > 1 else "xaxis"
                y_key = f"yaxis{axis_idx}" if axis_idx > 1 else "yaxis"
                if ri == m - 1:
                    fig.layout[x_key].title = {"text": base_names[ci], "font": {"size": 11}}
                else:
                    fig.layout[x_key].showticklabels = False
                if ci == 0:
                    fig.layout[y_key].title = {"text": base_names[ri], "font": {"size": 11}}
                else:
                    fig.layout[y_key].showticklabels = False

        cell_size = 180
        default_size = max(600, m * cell_size + 100)
        fig = apply_default_layout(
            fig,
            title=title or "Scatter Matrix",
            x_label=None,
            y_label=None,
            width=width or default_size,
            height=height or default_size,
        )
        figures[gname] = fig

    return figures


def _scatter_matrix_by_member(  # noqa: PLR0913
    *,
    df: pl.DataFrame,
    member_groups: dict[str, list[str]],
    seasonality: str | None,
    diagonal: str | None,
    show_correlation: bool,
    max_points: int | None,
    color_palette: list[str] | None,
    title: str | None,
    width: int | None,
    height: int | None,
    gaussian_kde,  # noqa: ANN001
    pearsonr,  # noqa: ANN001
    marker_size: float = 3.0,
    marker_opacity: float = 0.5,
) -> dict[str, go.Figure]:
    """One figure per member, each containing its own G×G scatter matrix (groups as axes)."""
    corr_font_size = 28

    figures: dict[str, go.Figure] = {}

    for mname, mcols in member_groups.items():
        base_names = [c.split("__", 1)[0] if "__" in c else c for c in mcols]
        sub_df = df.select(["time", *mcols]).rename(dict(zip(mcols, base_names, strict=False)))
        m = len(base_names)
        if m < 2:
            continue

        df_work, season_labels, season_colors, seasons_raw = _prepare_season_info(sub_df, seasonality, color_palette)
        df_work = _subsample_df(df_work, max_points, seasons_raw)
        uniform_color = resolve_color_palette(color_palette, 1)[0]

        fig = make_subplots(
            rows=m,
            cols=m,
            horizontal_spacing=_subplot_spacing(m),
            vertical_spacing=_subplot_spacing(m),
        )

        _render_scatter_block(
            fig,
            df_work,
            base_names,
            row_offset=0,
            col_offset=0,
            n=m,
            seasonality=seasonality,
            season_labels=season_labels,
            season_colors=season_colors,
            seasons_raw=seasons_raw,
            diagonal=diagonal,
            show_correlation=show_correlation,
            uniform_color=uniform_color,
            marker_size=marker_size,
            marker_opacity=marker_opacity,
            corr_font_size=corr_font_size,
            show_legend=True,
            total_cols=m,
            gaussian_kde=gaussian_kde,
            pearsonr=pearsonr,
        )

        # Axis labels on outermost edges
        for ri in range(m):
            for ci in range(m):
                axis_idx = ri * m + ci + 1
                x_key = f"xaxis{axis_idx}" if axis_idx > 1 else "xaxis"
                y_key = f"yaxis{axis_idx}" if axis_idx > 1 else "yaxis"
                if ri == m - 1:
                    fig.layout[x_key].title = {"text": base_names[ci], "font": {"size": 11}}
                else:
                    fig.layout[x_key].showticklabels = False
                if ci == 0:
                    fig.layout[y_key].title = {"text": base_names[ri], "font": {"size": 11}}
                else:
                    fig.layout[y_key].showticklabels = False

        cell_size = 180
        default_size = max(600, m * cell_size + 100)
        fig = apply_default_layout(
            fig,
            title=title or f"Scatter Matrix - {mname}",
            x_label=None,
            y_label=None,
            width=width or default_size,
            height=height or default_size,
        )
        figures[mname] = fig

    return figures


def plot_scatter_matrix(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    seasonality: str | None = None,
    diagonal: str | None = "kde",
    show_correlation: bool = True,
    max_points: int | None = 10_000,
    panel_group_names: list[str] | None = None,
    facet_by: Literal["group", "member"] | None = "group",
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    width: int | None = None,
    height: int | None = None,
    marker_size: float = 3.0,
    marker_opacity: float = 0.5,
) -> go.Figure | dict[str, go.Figure]:
    """Plot an N×N scatter-plot matrix.

    Creates an N×N grid for selected numeric columns:

    * **Lower triangle** - pairwise scatter plots.
    * **Diagonal** - KDE curves (default), histograms, or nothing.
    * **Upper triangle** - Pearson *r* annotations with font size
      scaled by ``|r|``.

    Points can optionally be coloured by season (``seasonality`` parameter).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with ``"time"`` column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to include.  If ``None``, uses all numeric columns
        except ``"time"``.
    seasonality : str | None, default=None
        Colour markers by season.  Accepts ``"month"``, ``"quarter"``,
        ``"weekday"``, ``"week"`` or ``"hour"``.
    diagonal : str | None, default="kde"
        What to show on the diagonal: ``"kde"``, ``"histogram"`` or
        ``None`` (leave blank).
    show_correlation : bool, default=True
        Show Pearson *r* annotations in the upper triangle.
    max_points : int | None, default=10_000
        Maximum number of points to plot per scatter subplot.  When the
        DataFrame exceeds this, a random sample is taken **once** so all
        subplots share the same consistent subset.  When ``seasonality``
        is set, sampling is stratified per season to preserve relative
        proportions.  Set to ``None`` to disable subsampling.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot.  When ``None`` and the DataFrame
        contains panel data, all groups are plotted automatically.
    facet_by : Literal["group", "member"] | None, default="group"
        Faceting axis for panel data.  ``"group"`` returns one figure
        per group (members as axis labels), ``"member"`` returns one
        figure per member (groups as axis labels).  ``None`` returns
        a single figure with all panel columns.
        Ignored for non-panel data.
    color_palette : list[str] | None, default=None
        Custom colour palette.  If ``None``, uses yohou palette.
    show_legend : bool, default=True
        Whether to show the legend.
    title : str | None, default=None
        Plot title.
    width : int | None, default=None
        Plot width in pixels.  Defaults to ``max(600, n*180+100)``.
    height : int | None, default=None
        Plot height in pixels.  Defaults to same as *width*.
    marker_size : float, default=3.0
        Size of scatter markers.
    marker_opacity : float, default=0.5
        Opacity of scatter markers.

    Returns
    -------
    go.Figure | dict[str, go.Figure]
        Single figure when non-panel or ``facet_by=None``.
        Dict of figures keyed by group or member name otherwise.

    Raises
    ------
    ValueError
        If fewer than two numeric columns are selected.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> from yohou.plotting import plot_scatter_matrix

    >>> rng = np.random.default_rng(0)
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
    ...     "a": rng.standard_normal(100),
    ...     "b": rng.standard_normal(100),
    ...     "c": rng.standard_normal(100),
    ... })

    >>> fig = plot_scatter_matrix(df, columns=["a", "b", "c"])
    >>> len(fig.data) > 0
    True

    See Also
    --------
    [`plot_correlation_heatmap`][yohou.plotting.plot_correlation_heatmap] : Plot correlation heatmap.
    [`plot_lag_scatter`][yohou.plotting.plot_lag_scatter] : Lag scatter plots.
    """
    from scipy.stats import gaussian_kde, pearsonr  # noqa: PLC0415

    validate_plotting_data(df)
    validate_plotting_params(width=width, height=height)

    _, _panel_groups = inspect_panel(df)
    if panel_group_names is None and columns is None and _panel_groups:
        panel_group_names = []

    if panel_group_names is not None:
        from yohou.plotting._utils import _group_panel_columns  # noqa: PLC0415

        # Normalize columns to list for member filtering
        col_filter = [columns] if isinstance(columns, str) else columns

        # Filter to requested groups and optionally by member postfix
        groups: dict[str, list[str]] = {}
        for g, gcols in _panel_groups.items():
            if not panel_group_names or g in panel_group_names:
                filtered = [c for c in gcols if c.split("__", 1)[1] in col_filter] if col_filter is not None else gcols
                if filtered:
                    groups[g] = filtered
        if not groups:
            msg = f"No panel groups found for {panel_group_names}. Available groups: {list(_panel_groups.keys())}"
            raise ValueError(msg)

        _scatter_kwargs = {
            "seasonality": seasonality,
            "diagonal": diagonal,
            "show_correlation": show_correlation,
            "max_points": max_points,
            "color_palette": color_palette,
            "title": title,
            "width": width,
            "height": height,
            "gaussian_kde": gaussian_kde,
            "pearsonr": pearsonr,
            "marker_size": marker_size,
            "marker_opacity": marker_opacity,
        }

        if facet_by is None:
            # Flatten all panel columns into a single scatter matrix
            all_panel_cols = [c for gcols in groups.values() for c in gcols]
            plot_columns = all_panel_cols
            # Fall through to the non-panel path below
        elif facet_by == "member":
            all_panel_cols = [c for gcols in groups.values() for c in gcols]
            _, all_members = _group_panel_columns(all_panel_cols)
            member_groups: dict[str, list[str]] = {}
            for member in all_members:
                member_cols = []
                for _gname, gcols in groups.items():
                    col = next((c for c in gcols if _member_name(c) == member), None)
                    if col:
                        member_cols.append(col)
                if member_cols:
                    member_groups[member] = member_cols
            return _scatter_matrix_by_member(df=df, member_groups=member_groups, **_scatter_kwargs)
        else:
            # facet_by == "group"
            return _scatter_matrix_facet(df=df, groups=groups, **_scatter_kwargs)
    else:
        plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])
    n = len(plot_columns)
    if n < 2:
        msg = f"plot_scatter_matrix requires at least 2 numeric columns, got {n}: {plot_columns}"
        raise ValueError(msg)

    corr_font_size = 28

    # Season setup
    season_labels: list[str] | None = None
    season_colors: list[str] | None = None
    seasons_raw: list[int] | None = None
    df_work = df

    if seasonality is not None:
        df_work = _add_season_and_cycle(df, seasonality)
        label_map = _SEASON_LABELS_MAP.get(seasonality)
        seasons_raw = sorted(df_work["season"].unique().to_list())
        n_seasons = len(seasons_raw)
        season_labels = (
            [label_map[s - 1] for s in seasons_raw] if label_map is not None else [str(s) for s in seasons_raw]
        )
        season_colors = resolve_color_palette(color_palette, n_seasons)

    if max_points is not None and len(df_work) > max_points:
        if seasons_raw is not None:
            # Stratified sampling: preserve season proportions
            n_per_season = max(1, max_points // len(seasons_raw))
            parts = [
                df_work.filter(pl.col("season") == s).sample(
                    n=min(n_per_season, df_work.filter(pl.col("season") == s).height),
                    seed=42,
                )
                for s in seasons_raw
            ]
            df_work = pl.concat(parts)
        else:
            df_work = df_work.sample(n=max_points, seed=42)

    fig = make_subplots(
        rows=n,
        cols=n,
        horizontal_spacing=0.03,
        vertical_spacing=0.03,
    )

    uniform_color = resolve_color_palette(color_palette, 1)[0]

    for ri in range(n):
        for ci in range(n):
            row = ri + 1
            col = ci + 1
            col_y = plot_columns[ri]
            col_x = plot_columns[ci]

            if ri == ci:
                if diagonal is None:
                    pass  # leave blank
                elif diagonal == "kde":
                    vals = df_work[col_x].drop_nulls().to_numpy()
                    if len(vals) > 1:
                        try:
                            kde = gaussian_kde(vals)
                        except np.linalg.LinAlgError:
                            pass
                        else:
                            data_range = float(vals.max()) - float(vals.min())
                            if data_range > 0 and kde.factor > 1e-10:  # ty: ignore[unsupported-operator]
                                x_grid = np.linspace(float(vals.min()), float(vals.max()), 200)
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_grid,
                                        y=kde(x_grid),
                                        mode="lines",
                                        line={"color": uniform_color, "width": 1.5},
                                        showlegend=False,
                                        hoverinfo="skip",
                                    ),
                                    row=row,
                                    col=col,
                                )
                elif diagonal == "histogram":
                    vals = df_work[col_x].drop_nulls().to_numpy()
                    fig.add_trace(
                        go.Histogram(
                            x=vals,
                            marker_color=uniform_color,
                            opacity=0.7,
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )

            elif ri > ci:
                if season_labels is not None and season_colors is not None and seasons_raw is not None:
                    for si, (sval, slabel) in enumerate(zip(seasons_raw, season_labels, strict=False)):
                        sub = df_work.filter(pl.col("season") == sval).drop_nulls(subset=[col_x, col_y])
                        if len(sub) == 0:
                            continue
                        is_first_scatter = ri == 1 and ci == 0
                        fig.add_trace(
                            go.Scattergl(
                                x=sub[col_x],
                                y=sub[col_y],
                                mode="markers",
                                marker={
                                    "size": marker_size,
                                    "color": season_colors[si % len(season_colors)],
                                    "opacity": marker_opacity,
                                },
                                name=slabel,
                                legendgroup=slabel,
                                showlegend=is_first_scatter,
                                hoverinfo="skip",
                            ),
                            row=row,
                            col=col,
                        )
                else:
                    sub = df_work.drop_nulls(subset=[col_x, col_y])
                    fig.add_trace(
                        go.Scattergl(
                            x=sub[col_x],
                            y=sub[col_y],
                            mode="markers",
                            marker={
                                "size": marker_size,
                                "color": uniform_color,
                                "opacity": marker_opacity,
                            },
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )

            elif show_correlation:
                sub = df_work.drop_nulls(subset=[col_x, col_y])
                if len(sub) > 2:
                    r_val, _ = pearsonr(sub[col_x].to_numpy(), sub[col_y].to_numpy())
                    font_sz = max(10, int(corr_font_size * abs(r_val)))
                    # Use paper coordinates derived from subplot axis domains
                    # so each annotation lands in its own cell.
                    axis_idx = ri * n + ci + 1
                    x_key = f"xaxis{axis_idx}" if axis_idx > 1 else "xaxis"
                    y_key = f"yaxis{axis_idx}" if axis_idx > 1 else "yaxis"
                    x_domain = fig.layout[x_key].domain
                    y_domain = fig.layout[y_key].domain
                    fig.add_annotation(
                        text=f"{r_val:.2f}",
                        xref="paper",
                        yref="paper",
                        x=(x_domain[0] + x_domain[1]) / 2,
                        y=(y_domain[0] + y_domain[1]) / 2,
                        showarrow=False,
                        font={"size": font_sz},
                    )

    # Axis labels on outermost edges only
    for ri in range(n):
        for ci in range(n):
            axis_idx = ri * n + ci + 1
            x_key = f"xaxis{axis_idx}" if axis_idx > 1 else "xaxis"
            y_key = f"yaxis{axis_idx}" if axis_idx > 1 else "yaxis"
            # Bottom row gets x-axis labels
            if ri == n - 1:
                fig.layout[x_key].title = {"text": plot_columns[ci], "font": {"size": 11}}
            else:
                fig.layout[x_key].showticklabels = False
            # Left column gets y-axis labels
            if ci == 0:
                fig.layout[y_key].title = {"text": plot_columns[ri], "font": {"size": 11}}
            else:
                fig.layout[y_key].showticklabels = False

    cell_size = 180
    default_size = max(600, n * cell_size + 100)

    fig = apply_default_layout(
        fig,
        title=title or "Scatter Matrix",
        x_label=None,
        y_label=None,
        width=width or default_size,
        height=height or default_size,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


_PERIOD_EXTRACTORS: dict[str, str] = {
    "hour": "hour",
    "day_of_week": "weekday",
    "day_of_month": "day",
    "week": "week",
    "month": "month",
    "quarter": "quarter",
    "year": "year",
}

_PERIOD_LABELS: dict[str, list[str] | None] = {
    "hour": [str(h) for h in range(24)],
    "day_of_week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "day_of_month": None,
    "week": None,
    "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "quarter": ["Q1", "Q2", "Q3", "Q4"],
    "year": None,
}


def _extract_period(col: pl.Expr, period: str) -> pl.Expr:
    """Extract a temporal period from a datetime expression.

    Parameters
    ----------
    col : pl.Expr
        A Polars expression referencing a datetime column.
    period : str
        One of the keys in ``_PERIOD_EXTRACTORS``.

    Returns
    -------
    pl.Expr
        Integer expression with the extracted period value.

    Raises
    ------
    ValueError
        If *period* is not a recognised period name.
    """
    method_name = _PERIOD_EXTRACTORS.get(period)
    if method_name is None:
        msg = f"Unknown period: {period!r}. Valid options: {', '.join(sorted(_PERIOD_EXTRACTORS))}"
        raise ValueError(msg)
    return getattr(col.dt, method_name)()


def _resolve_axis_labels(
    values: list[int],
    period: str,
) -> list[str]:
    """Map numeric period values to human-readable labels when available."""
    labels = _PERIOD_LABELS.get(period)
    if labels is None:
        return [str(v) for v in values]
    # Weekday: Polars weekday() returns 1=Mon..7=Sun; labels are 0-indexed
    if period == "day_of_week":
        return [labels[int(v) - 1] if 1 <= int(v) <= 7 else str(v) for v in values]
    # Month, quarter: 1-indexed → 0-indexed labels
    if period in ("month", "quarter"):
        return [labels[int(v) - 1] if 1 <= int(v) <= len(labels) else str(v) for v in values]
    # Hour (already 0-indexed in the label list)
    return [labels[int(v)] if 0 <= int(v) < len(labels) else str(v) for v in values]


def plot_seasonal_heatmap(
    df: pl.DataFrame,
    columns: str | list[str] | None = None,
    *,
    x_period: str = "hour",
    y_period: str = "month",
    agg: str = "mean",
    panel_group_names: list[str] | None = None,
    facet_by: Literal["group", "member"] | None = "group",
    facet_n_cols: int = 2,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    colorscale: str = "Viridis",
    show_values: bool = True,
    value_format: str = ".1f",
    reverse_y: bool = False,
) -> go.Figure:
    """Plot a 2-D heatmap of aggregated values across two time dimensions.

    Reveals seasonal patterns by showing aggregated values in a grid
    of two temporal periods (e.g. hour-of-day x month-of-year).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and the target *columns*.
    columns : str, list of str, or None, default=None
        Numeric column(s) to aggregate and visualise.  When ``None``,
        all numeric columns (excluding ``"time"``) are used.  For panel
        data this selects member postfixes within each group.
    x_period : str, default="hour"
        Temporal period for the x-axis.  Options: "hour", "day_of_week",
        "day_of_month", "week", "month", "quarter", "year".
    y_period : str, default="month"
        Temporal period for the y-axis.  Same options as *x_period*.
    agg : str, default="mean"
        Aggregation function: "mean", "median", "sum", "count",
        "std", "min", or "max".
    panel_group_names : list[str] | None, default=None
        Panel group prefixes.  When panel data is detected the *columns*
        are resolved as member postfixes within each group.  When
        ``None`` and panel columns are present, auto-detects all groups.
    facet_by : Literal["group", "member"] | None, default="group"
        Faceting axis for panel data.  ``"group"`` creates one subplot per
        group, ``"member"`` one per member.  ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Columns in the facet grid.
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
    colorscale : str, default="Viridis"
        Plotly colorscale for the heatmap.
    show_values : bool, default=True
        Whether to display values on the heatmap cells.
    value_format : str, default=".1f"
        Format string for heatmap cell values.
    reverse_y : bool, default=False
        Whether to reverse the y-axis.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    TypeError
        If *df* is not a Polars DataFrame.
    ValueError
        If requested *columns* do not exist, or *x_period*/*y_period*
        are invalid.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_seasonal_heatmap

    >>> df = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         pl.datetime(2020, 1, 1),
    ...         pl.datetime(2020, 12, 31, 23),
    ...         "1h",
    ...         eager=True,
    ...     ),
    ...     "temp": [20.0 + (i % 24) * 0.5 for i in range(8784)],
    ... })
    >>> fig = plot_seasonal_heatmap(df, "temp", x_period="hour", y_period="month")
    >>> len(fig.data) > 0
    True

    See Also
    --------
    [`plot_seasonality`][yohou.plotting.plot_seasonality] : Plot seasonal overlay.
    [`plot_subseasonality`][yohou.plotting.plot_subseasonality] : Plot seasonal subseries.
    """
    # Validate
    validate_plotting_data(df)
    validate_plotting_params(width=width, height=height)

    # Map aggregation name to a Polars expression builder
    _AGG_MAP = {
        "mean": lambda c: pl.col(c).mean(),
        "median": lambda c: pl.col(c).median(),
        "sum": lambda c: pl.col(c).sum(),
        "count": lambda c: pl.col(c).count(),
        "std": lambda c: pl.col(c).std(),
        "min": lambda c: pl.col(c).min(),
        "max": lambda c: pl.col(c).max(),
    }
    if agg not in _AGG_MAP:
        msg = f"Unknown agg: {agg!r}. Valid options: {', '.join(sorted(_AGG_MAP))}"
        raise ValueError(msg)

    def _build_heatmap(
        frame: pl.DataFrame,
        target_col: str,
        display_name: str | None = None,
    ) -> go.Heatmap:
        """Build a single Heatmap trace from *frame*."""
        # Extract periods
        df_aug = frame.with_columns(
            _extract_period(pl.col("time"), x_period).alias("_x"),
            _extract_period(pl.col("time"), y_period).alias("_y"),
        )

        # Aggregate
        agg_expr = _AGG_MAP[agg](target_col).alias("_val")
        df_agg = df_aug.group_by("_y", "_x").agg(agg_expr).sort("_y", "_x")

        # Pivot to matrix
        pivot = df_agg.pivot(on="_x", index="_y", values="_val").sort("_y")
        x_cols = sorted([c for c in pivot.columns if c != "_y"], key=int)
        y_vals = pivot["_y"].to_list()
        z = pivot.select(x_cols).to_numpy()

        x_display = _resolve_axis_labels([int(c) for c in x_cols], x_period)
        y_display = _resolve_axis_labels([int(v) for v in y_vals], y_period)

        text_ann = None
        if show_values:
            text_ann = [[f"{v:{value_format}}" if v is not None and not np.isnan(v) else "" for v in row] for row in z]

        label = display_name or target_col
        return go.Heatmap(
            z=z,
            x=x_display,
            y=y_display,
            colorscale=colorscale,
            text=text_ann,
            texttemplate="%{text}" if show_values else None,
            hovertemplate=(
                f"<b>{label}</b><br>{x_period}: %{{x}}<br>{y_period}: %{{y}}<br>{agg}: %{{z:.2f}}<extra></extra>"
            ),
        )

    # Auto-detect panel data
    if panel_group_names is None and _auto_detect_panel(df):
        panel_group_names = []

    if panel_group_names is not None:
        from yohou.plotting._utils import _group_panel_columns  # noqa: PLC0415

        panel_cols = resolve_panel_columns(df, panel_group_names, columns)
        groups, _all_members = _group_panel_columns(panel_cols)

        # Build flat list of (group_name, member_col, display_name) tuples
        all_cells: list[tuple[str, str, str]] = []
        for gname, gcols in groups.items():
            for col in gcols:
                mname = _member_name(col)
                all_cells.append((gname, col, f"{gname}: {mname}"))

        n = len(all_cells)
        n_cols_grid = min(n, facet_n_cols)
        n_rows_grid = math.ceil(n / n_cols_grid)

        fig = make_subplots(
            rows=n_rows_grid,
            cols=n_cols_grid,
            subplot_titles=[cell[2] for cell in all_cells],
            vertical_spacing=_subplot_spacing(n_rows_grid),
            horizontal_spacing=_subplot_spacing(n_cols_grid) if n_cols_grid > 1 else 0.08,
        )

        for idx, (_gname, member_col, display_label) in enumerate(all_cells):
            r = idx // n_cols_grid + 1
            c = idx % n_cols_grid + 1
            mname = _member_name(member_col)
            sub_df = df.select("time", pl.col(member_col).alias(mname))
            trace = _build_heatmap(sub_df, mname, display_label)
            trace.showscale = idx == n - 1
            fig.add_trace(trace, row=r, col=c)

        _col_label = ", ".join(columns) if isinstance(columns, list) else (columns or "all")
        fig = apply_default_layout(
            fig,
            title=title or "Seasonal Heatmap",
            x_label=x_label or x_period.replace("_", " ").title(),
            y_label=y_label or y_period.replace("_", " ").title(),
            width=width,
            height=height,
        )
        if reverse_y:
            fig.update_yaxes(autorange="reversed")
        fig.update_layout(showlegend=show_legend)
        return fig

    # Non-panel case: column-mode facet_figure
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])
    _last_col = plot_columns[-1]

    def _render_heatmap(ctx: RenderContext) -> None:
        """Render a single heatmap trace into a subplot."""
        col_name = ctx.display_name
        trace = _build_heatmap(ctx.sub_df, col_name, display_name=col_name)
        trace.showscale = col_name == _last_col
        ctx.fig.add_trace(trace, row=ctx.row, col=ctx.col)

    fig = facet_figure(
        df,
        _render_heatmap,
        columns=plot_columns,
        facet_n_cols=facet_n_cols,
        title=title or "Seasonal Heatmap",
        x_label=x_label or x_period.replace("_", " ").title(),
        y_label=y_label or y_period.replace("_", " ").title(),
        width=width,
        height=height,
        shared_xaxes=False,
    )
    if reverse_y:
        fig.update_yaxes(autorange="reversed")
    fig.update_layout(showlegend=show_legend)
    return fig
