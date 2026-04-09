"""Diagnostic and frequency-domain plotting functions for time series analysis."""

import math
from typing import cast

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from scipy.stats import norm

from yohou.plotting._utils import (
    apply_default_layout,
    panel_facet_figure,
    resolve_color_palette,
)
from yohou.utils import validate_plotting_data
from yohou.utils.panel import inspect_panel

__all__ = [
    "plot_autocorrelation",
    "plot_correlation_heatmap",
    "plot_cross_correlation",
    "plot_lag_scatter",
    "plot_partial_autocorrelation",
    "plot_scatter_matrix",
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
    facet_n_cols: int = 2,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
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
        - bar_color : str, default="#2563EB"

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
    `plot_partial_autocorrelation` : Plot partial autocorrelation function.
    `plot_correlation_heatmap` : Plot correlation matrix.
    """
    # Validate inputs
    validate_plotting_data(df)

    # Auto-detect panel data
    _, _panel_groups = inspect_panel(df)
    if panel_group_names is None and columns is None and _panel_groups:
        panel_group_names = []

    if panel_group_names is not None:

        def _render_acf(
            fig: go.Figure,
            sub_df: pl.DataFrame,
            display_name: str,  # noqa: ARG001
            panel_idx: int,  # noqa: ARG001
            row: int,
            col: int,
        ) -> None:
            """Render autocorrelation bar chart with confidence bands for a single column."""
            base = [c for c in sub_df.columns if c != "time"][0]
            _bc = kwargs.get("bar_color", "#2563EB")
            series = sub_df[base].drop_nulls()
            n_s = len(series)
            _ml = max_lags if max_lags is not None else min(n_s // 2, 40)
            mean_v = series.mean()
            acf_vals = []
            for lag in range(_ml + 1):
                if lag == 0:
                    acf_vals.append(1.0)
                else:
                    s1 = series[:-lag] - mean_v
                    s2 = series[lag:] - mean_v
                    num = float((s1 * s2).sum())
                    den = float(((series - mean_v) ** 2).sum())
                    acf_vals.append(num / den if den != 0 else 0)
            fig.add_trace(
                go.Bar(x=list(range(_ml + 1)), y=acf_vals, marker={"color": _bc}, showlegend=False),
                row=row,
                col=col,
            )
            if show_confidence:
                ci = norm.ppf(1 - (1 - confidence_level) / 2) / math.sqrt(n_s)
                for val in (ci, -ci):
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(_ml + 1)),
                            y=[val] * (_ml + 1),
                            mode="lines",
                            line={"dash": "dash", "color": "#DC2626", "width": 1},
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )

        return panel_facet_figure(
            df,
            _render_acf,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_n_cols=facet_n_cols,
            title=title or "Autocorrelation (Panel)",
            x_label=x_label or "Lag",
            y_label=y_label or "ACF",
            width=width,
            height=height,
            shared_xaxes=False,
        )

    # Resolve columns
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])

    # Get kwargs
    bar_color = kwargs.get("bar_color", "#2563EB")

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
                numerator = float((series1 * series2).sum())
                denominator = float(((series - mean_val) ** 2).sum())
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
            ci = norm.ppf(1 - (1 - confidence_level) / 2) / math.sqrt(n_series)

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


def _compute_pacf_durbin_levinson(
    values: np.ndarray,
    nlags: int,
    alpha: float | None,
) -> tuple[list[float], list[float] | None, list[float] | None]:
    """Compute PACF using Durbin-Levinson recursion (built-in fallback).

    Parameters
    ----------
    values : np.ndarray
        1-D numeric array with no NaNs.
    nlags : int
        Number of lags.
    alpha : float | None
        Significance level for confidence intervals.  If *None*, no
        intervals are returned.

    Returns
    -------
    tuple[list[float], list[float] | None, list[float] | None]
        ``(pacf_values, ci_lower, ci_upper)``.  ``ci_lower`` and
        ``ci_upper`` are *None* when *alpha* is *None*.

    """
    n = len(values)
    mean_val = float(np.nanmean(values))

    # ACF
    acf: list[float] = [1.0]
    centered = values - mean_val
    denom = float(np.sum(centered**2))
    for lag in range(1, nlags + 1):
        num = float(np.sum(centered[:-lag] * centered[lag:]))
        acf.append(num / denom if denom != 0 else 0.0)

    # Durbin-Levinson
    pacf_vals: list[float] = [1.0]
    if nlags > 0:
        phi: list[list[float]] = [[acf[1]]]
        pacf_vals.append(acf[1])
        for k in range(2, nlags + 1):
            num = acf[k] - sum(phi[k - 2][j] * acf[k - j - 1] for j in range(k - 1))
            den = 1.0 - sum(phi[k - 2][j] * acf[j + 1] for j in range(k - 1))
            pk = num / den if abs(den) > 1e-10 else 0.0
            pacf_vals.append(pk)
            new_phi = [phi[k - 2][j] - pk * phi[k - 2][k - j - 2] for j in range(k - 1)]
            new_phi.append(pk)
            phi.append(new_phi)

    ci_lo: list[float] | None = None
    ci_hi: list[float] | None = None
    if alpha is not None:
        z = norm.ppf(1 - alpha / 2)
        ci = z / math.sqrt(n)
        ci_lo = [-ci] * (nlags + 1)
        ci_hi = [ci] * (nlags + 1)

    return pacf_vals, ci_lo, ci_hi


def _compute_pacf(
    values: np.ndarray,
    nlags: int,
    *,
    method: str = "yw",
    alpha: float | None = None,
) -> tuple[list[float], list[float] | None, list[float] | None]:
    """Compute PACF, preferring statsmodels when available.

    Parameters
    ----------
    values : np.ndarray
        1-D numeric array.
    nlags : int
        Number of lags.
    method : str
        PACF method (see ``statsmodels.tsa.stattools.pacf``).  Ignored
        when falling back to Durbin-Levinson.
    alpha : float | None
        Significance level for confidence intervals.

    Returns
    -------
    tuple[list[float], list[float] | None, list[float] | None]
        ``(pacf_values, ci_lower, ci_upper)``.

    """
    import warnings  # noqa: PLC0415

    try:
        from statsmodels.tsa.stattools import pacf as sm_pacf  # noqa: PLC0415
    except ImportError:
        if method != "yw":
            warnings.warn(
                f"statsmodels is not installed; ignoring method='{method}' "
                "and falling back to built-in Durbin-Levinson (equivalent to "
                "'yw'). Install statsmodels for additional PACF methods: "
                "pip install yohou[plotting]",
                UserWarning,
                stacklevel=3,
            )
        return _compute_pacf_durbin_levinson(values, nlags, alpha)

    if alpha is not None:
        pacf_result, confint = sm_pacf(values, nlags=nlags, method=method, alpha=alpha)  # fmt: skip  # ty: ignore[invalid-argument-type]
        # statsmodels returns CIs centered on the PACF values (pacf ± z/√n).
        # For significance testing we need horizontal bands at ±z/√n (centered
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
    facet_n_cols: int = 2,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """Plot partial autocorrelation function (PACF) for time series.

    Shows correlation between the series and its lagged values after removing
    the effect of intermediate lags. Useful for determining appropriate AR order.

    When ``statsmodels`` is installed the computation is delegated to
    ``statsmodels.tsa.stattools.pacf`` which supports several estimation
    methods and proper confidence intervals.  Without ``statsmodels`` a
    built-in Durbin-Levinson recursion is used (equivalent to ``method="yw"``).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to analyze. If None, uses all numeric columns except 'time'.
    max_lags : int | None, default=None
        Maximum number of lags to compute. If None, uses min(len(df)//2, 40).
    method : str, default="yw"
        PACF estimation method (requires ``statsmodels``).  Options:
        ``"yw"`` (Yule-Walker), ``"ols"`` (OLS), ``"ywm"`` (Yule-Walker
        mean-adjusted), ``"burg"``, ``"ldb"`` (Levinson-Durbin biased),
        ``"ldbiased"``.  Ignored when ``statsmodels`` is not installed.
    confidence_level : float, default=0.95
        Confidence level for confidence bands (e.g. ``0.95`` for 95%).
    show_confidence : bool, default=True
        Whether to show confidence bands.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot.
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
        - bar_color : str, default="#059669"

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
    `plot_autocorrelation` : Plot autocorrelation function.
    """
    # Validate inputs
    validate_plotting_data(df)

    # Auto-detect panel data
    _, _panel_groups = inspect_panel(df)
    if panel_group_names is None and columns is None and _panel_groups:
        panel_group_names = []

    if panel_group_names is not None:

        def _render_pacf(
            fig: go.Figure,
            sub_df: pl.DataFrame,
            display_name: str,  # noqa: ARG001
            panel_idx: int,  # noqa: ARG001
            row: int,
            col: int,
        ) -> None:
            """Render PACF bar chart with confidence bands for a single panel."""
            base = [c for c in sub_df.columns if c != "time"][0]
            _bc = kwargs.get("bar_color", "#059669")
            series = sub_df[base].drop_nulls()
            n_s = len(series)
            _ml = max_lags if max_lags is not None else min(n_s // 2, 40)

            alpha_val = 1 - confidence_level if show_confidence else None
            pacf_vals, ci_lo, ci_hi = _compute_pacf(
                series.to_numpy(),
                _ml,
                method=method,
                alpha=alpha_val,
            )

            fig.add_trace(
                go.Bar(x=list(range(_ml + 1)), y=pacf_vals, marker={"color": _bc}, showlegend=False),
                row=row,
                col=col,
            )
            if show_confidence and ci_lo is not None and ci_hi is not None:
                for vals in (ci_hi, ci_lo):
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(_ml + 1)),
                            y=vals,
                            mode="lines",
                            line={"dash": "dash", "color": "#DC2626", "width": 1},
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )

        return panel_facet_figure(
            df,
            _render_pacf,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_n_cols=facet_n_cols,
            title=title or "Partial Autocorrelation (Panel)",
            x_label=x_label or "Lag",
            y_label=y_label or "PACF",
            width=width,
            height=height,
            shared_xaxes=False,
        )

    # Resolve columns
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])

    # Get kwargs
    bar_color = kwargs.get("bar_color", "#059669")

    # Determine max_lags
    n = len(df)
    if max_lags is None:
        max_lags = min(n // 2, 40)

    alpha_val = 1 - confidence_level if show_confidence else None

    # Create figure
    fig = go.Figure()

    for col_name in plot_columns:
        series = df[col_name].drop_nulls()
        values = series.to_numpy()

        pacf_values, ci_lo, ci_hi = _compute_pacf(
            values,
            max_lags,
            method=method,
            alpha=alpha_val,
        )

        # Add bar trace
        fig.add_trace(
            go.Bar(
                x=list(range(max_lags + 1)),
                y=pacf_values,
                name=col_name,
                marker={"color": bar_color},
                hovertemplate=f"<b>{col_name}</b><br>Lag: %{{x}}<br>PACF: %{{y:.3f}}<extra></extra>",
            )
        )

        # Add confidence bands
        if show_confidence and ci_lo is not None and ci_hi is not None:
            fig.add_trace(
                go.Scatter(
                    x=list(range(max_lags + 1)),
                    y=ci_hi,
                    mode="lines",
                    line={"dash": "dash", "color": "#DC2626", "width": 1},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(max_lags + 1)),
                    y=ci_lo,
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


def plot_correlation_heatmap(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    panel_group_names: list[str] | None = None,
    facet_n_cols: int = 3,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
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
        Panel group prefixes to plot.
    facet_n_cols : int, default=3
        Number of columns in the faceted grid when multiple panel groups
        are present.
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
    `plot_autocorrelation` : Plot autocorrelation function.
    """
    # Validate inputs
    validate_plotting_data(df)

    # Auto-detect panel data
    _, panel_groups = inspect_panel(df)
    if panel_group_names is None and columns is None and panel_groups:
        panel_group_names = []

    if panel_group_names is not None:
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

        group_names = list(groups.keys())
        n_groups = len(group_names)

        grid_cols = min(n_groups, facet_n_cols)
        grid_rows = math.ceil(n_groups / grid_cols)

        fig = make_subplots(
            rows=grid_rows,
            cols=grid_cols,
            subplot_titles=group_names,
            horizontal_spacing=max(0.05, 0.3 / grid_cols),
        )

        colorscale = kwargs.get("colorscale", "RdBu_r")
        show_values = kwargs.get("show_values", True)

        for gi, gname in enumerate(group_names):
            gcols = groups[gname]
            # Strip prefix for display
            base_names = [c.split("__", 1)[1] if "__" in c else c for c in gcols]
            sub = df.select(gcols).rename(dict(zip(gcols, base_names, strict=False)))
            corr = sub.drop_nulls().corr()
            text_ann = None
            if show_values:
                text_ann = [[f"{v:.2f}" if v is not None else "" for v in row] for row in corr.rows()]
            fig.add_trace(
                go.Heatmap(
                    z=corr.to_numpy(),
                    x=base_names,
                    y=base_names,
                    colorscale=colorscale,
                    zmid=0,
                    text=text_ann,
                    texttemplate="%{text}" if show_values else None,
                    hovertemplate=(f"<b>{gname}</b><br>%{{x}} vs %{{y}}<br>Correlation: %{{z:.3f}}<extra></extra>"),
                    showscale=(gi == n_groups - 1),
                ),
                row=gi // grid_cols + 1,
                col=gi % grid_cols + 1,
            )

        fig = apply_default_layout(
            fig,
            title=title or "Correlation Heatmap (Panel)",
            x_label=x_label,
            y_label=y_label,
            width=width or max(400 * grid_cols, 600),
            height=height or max(400 * grid_rows, 400),
        )
        return fig

    # Resolve columns
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])

    # Get kwargs
    colorscale = kwargs.get("colorscale", "RdBu_r")
    show_values = kwargs.get("show_values", True)

    # Compute correlation matrix
    corr_matrix = df.select(plot_columns).drop_nulls().corr()

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


# ── Seasonal helpers (shared by plot_seasonality / plot_subseasonality) ──

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
    facet_n_cols: int = 2,
    color_palette: list[str] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
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
    facet_n_cols : int, default=2
        Number of columns in facet grid.
    color_palette : list[str] | None, default=None
        Custom color palette (one color per cycle).
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
        - highlight_width : float, default=3.0
        - fade_opacity : float, default=0.25

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
    `plot_subseasonality` : Plot seasonal subseries.
    `plot_time_series` : Plot basic time series.
    """

    # Resolve highlight to a set
    if highlight is None:
        highlight_set: set[int] = set()
    elif isinstance(highlight, int):
        highlight_set = {highlight}
    else:
        highlight_set = set(highlight)

    line_width = kwargs.get("line_width", 1.5)
    highlight_width = kwargs.get("highlight_width", 3.0)
    fade_opacity = kwargs.get("fade_opacity", 0.25)

    # ── Validate ─────────────────────────────────────────────────────────
    validate_plotting_data(df)

    # Auto-detect panel data
    _, _panel_groups = inspect_panel(df)
    if panel_group_names is None and columns is None and _panel_groups:
        panel_group_names = []

    if panel_group_names is not None:

        def _render_season(
            fig: go.Figure,
            sub_df: pl.DataFrame,
            display_name: str,  # noqa: ARG001
            panel_idx: int,  # noqa: ARG001
            row: int,
            col: int,
        ) -> None:
            """Render seasonal overlay for a single panel group."""
            base = [c for c in sub_df.columns if c != "time"][0]
            dfs = _add_season_and_cycle(sub_df, seasonality)
            cycles = sorted(dfs["cycle"].unique().to_list())
            _colors = resolve_color_palette(color_palette, len(cycles))
            slabels = _SEASON_LABELS_MAP.get(seasonality)
            for ci, cyc in enumerate(cycles):
                cyc_df = dfs.filter(pl.col("cycle") == cyc).sort("season")
                x_vals = (
                    [slabels[int(s) - 1] for s in cyc_df["season"].to_list()]
                    if slabels and all(s <= len(slabels) for s in cyc_df["season"].to_list())
                    else cyc_df["season"].to_list()
                )
                is_hl = bool(highlight_set) and cyc in highlight_set
                lw = highlight_width if is_hl else line_width
                opacity = 1.0 if (not highlight_set or is_hl) else fade_opacity
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=cyc_df[base].to_list(),
                        mode="lines",
                        line={"color": _colors[ci % len(_colors)], "width": lw},
                        opacity=opacity,
                        name=str(cyc),
                        legendgroup=str(cyc),
                        showlegend=False,
                        hovertemplate=f"<b>{base} - {cyc}</b><br>%{{x}}: %{{y:.2f}}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

        return panel_facet_figure(
            df,
            _render_season,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_n_cols=facet_n_cols,
            title=title or f"Seasonal Plot - {seasonality} (Panel)",
            x_label=x_label or seasonality.capitalize(),
            y_label=y_label,
            width=width,
            height=height,
            shared_xaxes=False,
        )

    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])
    season_labels = _SEASON_LABELS_MAP.get(seasonality)

    df_aug = _add_season_and_cycle(df, seasonality)
    cycles = sorted(df_aug["cycle"].unique().to_list())
    colors = resolve_color_palette(color_palette, len(cycles))

    fig = go.Figure()

    for col_name in plot_columns:
        for ci, cyc in enumerate(cycles):
            cyc_df = df_aug.filter(pl.col("cycle") == cyc).sort("season")
            x_vals = (
                [season_labels[int(s) - 1] for s in cyc_df["season"].to_list()]
                if season_labels and all(s <= len(season_labels) for s in cyc_df["season"].to_list())
                else cyc_df["season"].to_list()
            )
            is_hl = bool(highlight_set) and cyc in highlight_set
            lw = highlight_width if is_hl else line_width
            opacity = 1.0 if (not highlight_set or is_hl) else fade_opacity

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=cyc_df[col_name].to_list(),
                    mode="lines",
                    line={"color": colors[ci % len(colors)], "width": lw},
                    opacity=opacity,
                    name=str(cyc),
                    legendgroup=str(cyc),
                    showlegend=(col_name == plot_columns[0]),
                    hovertemplate=f"<b>{col_name} - {cyc}</b><br>%{{x}}: %{{y:.2f}}<extra></extra>",
                )
            )

    # Set default labels
    if x_label is None:
        x_label = seasonality.capitalize()
    if y_label is None:
        y_label = "Value"

    # Update x-axis with categorical labels
    if season_labels:
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(season_labels))),
            ticktext=season_labels,
        )

    fig = apply_default_layout(
        fig,
        title=title,
        x_label=x_label,
        y_label=y_label,
        width=width,
        height=height,
    )

    return fig


def plot_subseasonality(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    seasonality: str = "month",
    show_mean: bool = True,
    facet_n_cols: int = 4,
    color_palette: list[str] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """Plot seasonal subseries.

    Creates one mini subplot per season (e.g. 12 for months, 4 for quarters).
    Within each subplot the values for that season across all cycles are
    connected chronologically, with an optional horizontal mean line.  This
    view highlights both the *level* differences between seasons and any
    *trends within* each season over time.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to plot. If None, uses all numeric columns except 'time'.
    seasonality : str, default="month"
        Seasonal frequency: "month", "quarter", "weekday", "week", "hour".
    show_mean : bool, default=True
        Show a horizontal mean line within each season subplot.
    facet_n_cols : int, default=4
        Number of columns in the subplot grid.
    color_palette : list[str] | None, default=None
        Custom color palette (one color per column).
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
        - marker_size : float, default=4
        - mean_color : str, default="#DC2626"
        - mean_width : float, default=2.0
        - mean_dash : str, default="dash"

    Returns
    -------
    go.Figure
        Plotly figure object.

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
    `plot_seasonality` : Plot seasonal overlay.
    `plot_time_series` : Plot basic time series.
    """
    validate_plotting_data(df)

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

    fig = make_subplots(
        rows=nrow,
        cols=facet_n_cols,
        subplot_titles=subplot_titles,
        shared_yaxes=True,
    )

    colors = resolve_color_palette(color_palette, len(plot_columns))
    line_width = kwargs.get("line_width", 1.5)
    marker_size = kwargs.get("marker_size", 4)
    mean_color = kwargs.get("mean_color", "#DC2626")
    mean_width = kwargs.get("mean_width", 2.0)
    mean_dash = kwargs.get("mean_dash", "dash")

    for si, season_val in enumerate(seasons):
        r = si // facet_n_cols + 1
        c = si % facet_n_cols + 1

        season_df = df_aug.filter(pl.col("season") == season_val)

        for ci, col_name in enumerate(plot_columns):
            # Aggregate by cycle (mean) when multiple observations exist
            agg_df = season_df.group_by("cycle").agg(pl.col(col_name).mean()).sort("cycle")
            cycles = agg_df["cycle"].to_list()
            values = agg_df[col_name].to_list()

            fig.add_trace(
                go.Scatter(
                    x=cycles,
                    y=values,
                    mode="lines+markers",
                    line={"color": colors[ci % len(colors)], "width": line_width},
                    marker={"size": marker_size},
                    name=col_name,
                    legendgroup=col_name,
                    showlegend=(si == 0),
                    hovertemplate=(f"<b>{col_name}</b><br>Cycle: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>"),
                ),
                row=r,
                col=c,
            )

            if show_mean:
                mean_val = agg_df[col_name].mean()
                if mean_val is not None and len(cycles) >= 2:
                    fig.add_trace(
                        go.Scatter(
                            x=[cycles[0], cycles[-1]],
                            y=[mean_val, mean_val],
                            mode="lines",
                            line={
                                "color": mean_color,
                                "width": mean_width,
                                "dash": mean_dash,
                            },
                            name="Mean",
                            legendgroup="__mean__",
                            showlegend=(si == 0 and ci == 0),
                            hovertemplate=f"Mean: {mean_val:.2f}<extra></extra>",
                        ),
                        row=r,
                        col=c,
                    )

    fig = apply_default_layout(
        fig,
        title=title or f"Seasonal Subseries - {seasonality}",
        x_label=x_label,
        y_label=y_label,
        width=width,
        height=height if height is not None else max(450, nrow * 280),
    )

    return fig


def plot_lag_scatter(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    lags: int | list[int] = 1,
    seasonality: str | None = None,
    show_diagonal: bool = True,
    show_regression: bool = False,
    panel_group_names: list[str] | None = None,
    facet_n_cols: int = 3,
    color_palette: list[str] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """Plot scatter plots of y(t) vs y(t-lag) for analysing temporal dependencies.

    Creates scatter plots showing the relationship between current values and
    lagged values, useful for identifying AR patterns and lag-specific
    correlations.  When ``lags`` is a list with more than one entry, each lag
    gets its own subplot arranged in a grid.  Optionally colour points by
    season using the ``seasonality`` parameter.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : str | list[str] | None, default=None
        Column(s) to analyse. If None, uses all numeric columns except 'time'.
    lags : int | list[int], default=1
        Lag values to plot. Can be single lag or list of lags.
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
        Number of columns in facet / subplot grid.
    color_palette : list[str] | None, default=None
        Custom color palette.
    title : str | None, default=None
        Plot title.
    x_label : str | None, default=None
        X-axis label. Defaults to ``"y(t-{lag})"`` for single-lag plots.
    y_label : str | None, default=None
        Y-axis label. Defaults to ``"y(t)"``.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    **kwargs : dict
        Additional styling parameters:
        - marker_size : float, default=4.0
        - marker_opacity : float, default=0.6

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
    `plot_autocorrelation` : Plot autocorrelation function.
    """
    # Validate inputs
    validate_plotting_data(df)

    # Auto-detect panel data
    _, _panel_groups = inspect_panel(df)
    if panel_group_names is None and columns is None and _panel_groups:
        panel_group_names = []

    if panel_group_names is not None:
        _lag_list = [lags] if isinstance(lags, int) else lags

        def _render_lag(
            fig: go.Figure,
            sub_df: pl.DataFrame,
            display_name: str,  # noqa: ARG001
            panel_idx: int,  # noqa: ARG001
            row: int,
            col: int,
        ) -> None:
            """Render lag scatter plots for each requested lag of a single column."""
            base = [c for c in sub_df.columns if c != "time"][0]
            _ms = kwargs.get("marker_size", 4.0)
            _ma = kwargs.get("marker_opacity", 0.6)
            _sd = show_diagonal
            colors = resolve_color_palette(color_palette, len(_lag_list))
            for li, lag in enumerate(_lag_list):
                dl = sub_df.with_columns(pl.col(base).shift(lag).alias("lagged")).drop_nulls()
                fig.add_trace(
                    go.Scatter(
                        x=dl[base],
                        y=dl["lagged"],
                        mode="markers",
                        marker={"size": _ms, "color": colors[li % len(colors)], "opacity": _ma},
                        name=f"lag={lag}",
                        showlegend=(row == 1 and col == 1),
                    ),
                    row=row,
                    col=col,
                )
            if _sd and len(sub_df) > 0:
                vmin = sub_df[base].min()
                vmax = sub_df[base].max()
                fig.add_trace(
                    go.Scatter(
                        x=[vmin, vmax],
                        y=[vmin, vmax],
                        mode="lines",
                        line={"dash": "dash", "color": "#94a3b8", "width": 1},
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

        return panel_facet_figure(
            df,
            _render_lag,
            panel_group_names=panel_group_names,
            columns=columns,
            facet_n_cols=facet_n_cols,
            title=title or "Lag Scatter (Panel)",
            width=width,
            height=height,
            shared_xaxes=False,
        )

    # Resolve columns
    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])
    lag_list = [lags] if isinstance(lags, int) else lags

    # Get kwargs
    marker_size = kwargs.get("marker_size", 4.0)
    marker_opacity = kwargs.get("marker_opacity", 0.6)

    # ── Season coloring setup ────────────────────────────────────────────
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

    # ── Multi-lag subplot grid vs single-lag flat figure ─────────────────
    use_subplots = len(lag_list) > 1

    if use_subplots:
        n_lags = len(lag_list)
        ncols = min(facet_n_cols, n_lags)
        nrows = math.ceil(n_lags / ncols)

        subtitles = [f"lag {k}" for k in lag_list]
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=subtitles,
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )

        for lag_idx, lag in enumerate(lag_list):
            r = lag_idx // ncols + 1
            c = lag_idx % ncols + 1
            is_first_cell = lag_idx == 0

            for col in plot_columns:
                if df_aug is not None and season_labels is not None and season_colors is not None:
                    # Season-colored traces
                    seasons_raw = sorted(df_aug["season"].unique().to_list())
                    for si, (sval, slabel) in enumerate(zip(seasons_raw, season_labels, strict=False)):
                        sub = df_aug.filter(pl.col("season") == sval)
                        dl = sub.with_columns(pl.col(col).shift(lag).alias("lagged")).drop_nulls()
                        if len(dl) == 0:
                            continue
                        fig.add_trace(
                            go.Scatter(
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
                    # Uniform coloring
                    colors = resolve_color_palette(color_palette, len(plot_columns))
                    dl = df.with_columns(pl.col(col).shift(lag).alias("lagged")).drop_nulls()
                    fig.add_trace(
                        go.Scatter(
                            x=dl["lagged"],
                            y=dl[col],
                            mode="markers",
                            marker={
                                "size": marker_size,
                                "color": colors[plot_columns.index(col) % len(colors)],
                                "opacity": marker_opacity,
                            },
                            name=col,
                            showlegend=is_first_cell,
                            hovertemplate=(
                                f"<b>{col}</b><br>y(t-{lag}): %{{x:.2f}}<br>y(t): %{{y:.2f}}<extra></extra>"
                            ),
                        ),
                        row=r,
                        col=c,
                    )

                # Diagonal reference line
                if show_diagonal:
                    source = df_aug if df_aug is not None else df
                    dl_all = source.with_columns(pl.col(col).shift(lag).alias("lagged")).drop_nulls()
                    vmin = min(float(dl_all[col].min()), float(dl_all["lagged"].min()))  # fmt: skip  # ty: ignore[invalid-argument-type]
                    vmax = max(float(dl_all[col].max()), float(dl_all["lagged"].max()))  # fmt: skip  # ty: ignore[invalid-argument-type]
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

        cell_size = 260
        default_width = max(600, ncols * cell_size + 100)
        default_height = max(400, nrows * cell_size + 100)

        fig = apply_default_layout(
            fig,
            title=title or "Lag Scatter",
            x_label=None,
            y_label=None,
            width=width or default_width,
            height=height or default_height,
        )

        return fig

    # ── Single-lag flat figure ───────────────────────────────────────────
    fig = go.Figure()

    lag = lag_list[0]

    for col in plot_columns:
        if df_aug is not None and season_labels is not None and season_colors is not None:
            # Season-colored traces
            seasons_raw = sorted(df_aug["season"].unique().to_list())
            for si, (sval, slabel) in enumerate(zip(seasons_raw, season_labels, strict=False)):
                sub = df_aug.filter(pl.col("season") == sval)
                dl = sub.with_columns(pl.col(col).shift(lag).alias("lagged")).drop_nulls()
                if len(dl) == 0:
                    continue
                fig.add_trace(
                    go.Scatter(
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
                        showlegend=True,
                        hovertemplate=(
                            f"<b>{col}</b> ({slabel})<br>y(t-{lag}): %{{x:.2f}}<br>y(t): %{{y:.2f}}<extra></extra>"
                        ),
                    )
                )
        else:
            # Uniform coloring (original behaviour)
            colors = resolve_color_palette(color_palette, len(plot_columns))
            dl = df.with_columns(pl.col(col).shift(lag).alias("lagged")).drop_nulls()
            fig.add_trace(
                go.Scatter(
                    x=dl["lagged"],
                    y=dl[col],
                    mode="markers",
                    marker={
                        "size": marker_size,
                        "color": colors[plot_columns.index(col) % len(colors)],
                        "opacity": marker_opacity,
                    },
                    name=f"{col} (lag={lag})",
                    hovertemplate=(f"<b>{col}</b><br>y(t-{lag}): %{{x:.2f}}<br>y(t): %{{y:.2f}}<extra></extra>"),
                )
            )

        # Add diagonal line if requested
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

        # Add regression line if requested
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
                    reg_colors = resolve_color_palette(color_palette, len(plot_columns))
                    fig.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            line={
                                "color": reg_colors[plot_columns.index(col) % len(reg_colors)],
                                "width": 2,
                            },
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

    # Set default labels
    if title is None:
        title = f"Lag {lag} Scatter Plot"

    x_label_default = x_label or f"y(t-{lag})"
    y_label_default = y_label or "y(t)"

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


def plot_cross_correlation(
    df: pl.DataFrame,
    *,
    columns: list[str],
    max_lags: int = 40,
    confidence_level: float = 0.95,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """Plot cross-correlation function (CCF) between two time series.

    Computes and visualizes the cross-correlation between two series at various lags,
    useful for identifying lead-lag relationships and temporal dependencies.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns.
    columns : list[str]
        Exactly two column names: ``[predictor, response]``.
    max_lags : int, default=40
        Number of lags to compute (both positive and negative).
    confidence_level : float, default=0.95
        Confidence level for confidence bands (e.g. ``0.95`` for 95%).
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
    >>> fig = plot_cross_correlation(df, columns=["x", "y"], max_lags=20)
    >>> len(fig.data) > 0
    True

    See Also
    --------
    `plot_autocorrelation` : Plot autocorrelation function.
    """
    # Validate inputs
    validate_plotting_data(df)

    if len(columns) != 2:  # noqa: PLR2004
        msg = f"columns must contain exactly 2 column names, got {len(columns)}"
        raise ValueError(msg)

    x_column, y_column = columns

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

    # Get series - drop rows where either column is null so NaN does not
    # poison the mean / std / correlation calculations.
    clean = df.select(x_column, y_column).drop_nulls()
    x = clean[x_column].to_numpy()
    y = clean[y_column].to_numpy()
    n = len(x)

    # Compute cross-correlation for each lag
    lag_values = list(range(-max_lags, max_lags + 1))
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
    alpha = 1 - confidence_level
    z_value = norm.ppf(1 - alpha / 2)

    confidence_band = z_value / math.sqrt(n)

    ci_pct = (
        f"{confidence_level:.0%}" if confidence_level == int(confidence_level * 100) / 100 else f"{confidence_level:%}"
    )
    fig.add_hline(
        y=confidence_band,
        line={"dash": "dash", "color": "#DC2626", "width": 1},
        annotation_text=f"{ci_pct} CI",
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
        x_label=x_label or "Lag",
        y_label=y_label or "Cross-Correlation",
        width=width,
        height=height,
    )

    return fig


def plot_scatter_matrix(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    seasonality: str | None = None,
    diagonal: str | None = "kde",
    show_correlation: bool = True,
    color_palette: list[str] | None = None,
    title: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
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
    color_palette : list[str] | None, default=None
        Custom colour palette.  If ``None``, uses yohou palette.
    title : str | None, default=None
        Plot title.
    width : int | None, default=None
        Plot width in pixels.  Defaults to ``max(600, n*180+100)``.
    height : int | None, default=None
        Plot height in pixels.  Defaults to same as *width*.
    **kwargs : dict
        Additional styling parameters:

        - marker_size : float, default=3.0
        - marker_opacity : float, default=0.5
        - corr_font_size : int, default=28

    Returns
    -------
    go.Figure
        Plotly figure object.

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
    `plot_correlation_heatmap` : Plot correlation heatmap.
    `plot_lag_scatter` : Lag scatter plots.
    """
    from scipy.stats import gaussian_kde, pearsonr  # noqa: PLC0415

    validate_plotting_data(df)

    plot_columns = validate_plotting_data(df, columns=columns, exclude=["time"])
    n = len(plot_columns)
    if n < 2:
        msg = f"plot_scatter_matrix requires at least 2 numeric columns, got {n}: {plot_columns}"
        raise ValueError(msg)

    # Styling kwargs
    marker_size = kwargs.get("marker_size", 3.0)
    marker_opacity = kwargs.get("marker_opacity", 0.5)
    corr_font_size = kwargs.get("corr_font_size", 28)

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
                # ── Diagonal: KDE / histogram / nothing ──────────────
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
                # ── Lower triangle: scatter ──────────────────────────
                if season_labels is not None and season_colors is not None and seasons_raw is not None:
                    for si, (sval, slabel) in enumerate(zip(seasons_raw, season_labels, strict=False)):
                        sub = df_work.filter(pl.col("season") == sval).drop_nulls(subset=[col_x, col_y])
                        if len(sub) == 0:
                            continue
                        is_first_scatter = ri == 1 and ci == 0
                        fig.add_trace(
                            go.Scatter(
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
                        go.Scatter(
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

            # ── Upper triangle: Pearson r annotation ─────────────
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

    return fig
