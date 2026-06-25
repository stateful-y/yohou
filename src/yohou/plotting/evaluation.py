"""Model evaluation and diagnostic visualization functions."""

import copy
import re
import warnings
from typing import Literal

import numpy as np
import polars as pl
from pydantic import StrictFloat
from scipy import stats

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    msg = "plotly is required for yohou plotting. Install: pip install yohou[plotting]"
    raise ImportError(msg) from e

from yohou.metrics import BaseIntervalScorer
from yohou.metrics.base import BaseScorer
from yohou.plotting._utils import (
    LegendTracker,
    RenderContext,
    _create_figure,
    _create_subplots,
    _group_panel_columns,
    _make_hovertemplate,
    _member_name,
    _normalize_y_pred,
    _subplot_spacing,
    apply_default_layout,
    facet_figure,
    resolve_color_palette,
    resolve_panel_columns,
)
from yohou.utils import validate_plotting_data, validate_plotting_params
from yohou.utils.panel import inspect_panel

__all__ = [
    "plot_calibration",
    "plot_group_scores",
    "plot_residuals",
    "plot_score_distribution",
    "plot_score_heatmap",
    "plot_score_per_step",
    "plot_score_per_vintage",
    "plot_score_summary",
    "plot_score_time_series",
]

_SCORER_META_COLS = frozenset({"time", "vintage_time", "forecasting_step", "coverage_rate"})


def _normalize_scorers(
    scorer: BaseScorer | dict[str, BaseScorer],
) -> dict[str, BaseScorer]:
    """Normalize scorer input to a dict mapping name to scorer.

    Parameters
    ----------
    scorer : BaseScorer or dict[str, BaseScorer]
        Single scorer or dict of named scorers.

    Returns
    -------
    dict[str, BaseScorer]
        Mapping of scorer name to scorer instance.

    """
    if isinstance(scorer, dict):
        return scorer  # type: ignore
    return {scorer.__class__.__name__: scorer}


def _prepare_scorer_for_componentwise(scorer: BaseScorer) -> BaseScorer:
    """Clone and configure a scorer for componentwise aggregation.

    Parameters
    ----------
    scorer : BaseScorer
        Scorer to prepare.

    Returns
    -------
    BaseScorer
        Cloned scorer with componentwise aggregation set.

    """
    scorer_cw = copy.deepcopy(scorer)
    if isinstance(scorer_cw, BaseIntervalScorer):
        scorer_cw.set_params(aggregation_method=["componentwise", "coveragewise"])
    else:
        scorer_cw.set_params(aggregation_method="componentwise")
    return scorer_cw


def _warn_large_grid(n_scorers: int, n_models: int, threshold: int = 12) -> None:
    """Warn when the combination of scorers and models exceeds a threshold."""
    total = n_scorers * n_models
    if total > threshold:
        warnings.warn(
            f"Large facet grid: {n_scorers} scorer(s) x {n_models} model(s) = "
            f"{total} panels. Consider reducing the number of scorers or models.",
            UserWarning,
            stacklevel=3,
        )


def _render_residual_diagnostics(
    residuals_df: pl.DataFrame,
    y_pred: pl.DataFrame,
    col_name: str,
    *,
    color_palette: list[str] | None = None,
    title: str | None = None,
    width: int | None = None,
    height: int | None = None,
    marker_size: float = 4,
    marker_opacity: float = 0.6,
    n_bins: int = 30,
) -> go.Figure:
    """Create 4-panel residual diagnostics for a single column.

    Parameters
    ----------
    residuals_df : pl.DataFrame
        Residuals with ``"time"`` column and *col_name*.
    y_pred : pl.DataFrame
        Predicted values (for fitted-values scatter).
    col_name : str
        Column to diagnose.
    color_palette : list[str] | None, default=None
        Custom colour palette with (up to) 4 entries.
    title : str | None, default=None
        Plot title.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    marker_size : float, default=4
        Marker size for scatter plots.
    marker_opacity : float, default=0.6
        Marker opacity.
    n_bins : int, default=30
        Number of bins for the histogram.

    Returns
    -------
    go.Figure
        Plotly figure with 4 diagnostic subplots.
    """
    residuals = residuals_df[col_name].to_numpy()
    fitted = y_pred[col_name].to_numpy()

    colors = resolve_color_palette(color_palette, 4)

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

    # Residuals over time
    fig.add_trace(
        go.Scatter(
            x=residuals_df["time"],
            y=residuals,
            mode="markers",
            marker={"size": marker_size, "color": colors[0], "opacity": marker_opacity},
            name="Residuals",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line={"dash": "dash", "color": "#DC2626", "width": 1}, row=1, col=1)

    # Residuals vs Fitted
    fig.add_trace(
        go.Scatter(
            x=fitted,
            y=residuals,
            mode="markers",
            marker={"size": marker_size, "color": colors[1], "opacity": marker_opacity},
            name="Residuals vs Fitted",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0, line={"dash": "dash", "color": "#DC2626", "width": 1}, row=1, col=2)

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=n_bins,
            marker={"color": colors[2], "opacity": marker_opacity + 0.1},
            name="Histogram",
        ),
        row=2,
        col=1,
    )

    # Q-Q Plot (z-scored residuals so both axes share the same scale)
    res_mean = np.nanmean(residuals)
    res_std = np.nanstd(residuals, ddof=1)
    z_residuals = residuals - res_mean if res_std == 0 or np.isnan(res_std) else (residuals - res_mean) / res_std
    sorted_z = np.sort(z_residuals)
    n = len(sorted_z)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))

    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_z,
            mode="markers",
            marker={
                "size": marker_size,
                "color": colors[3 % len(colors)],
                "opacity": marker_opacity,
            },
            name="Q-Q Plot",
        ),
        row=2,
        col=2,
    )

    # Reference line for Q-Q plot
    min_val = min(theoretical_quantiles.min(), sorted_z.min())
    max_val = max(theoretical_quantiles.max(), sorted_z.max())
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

    # Force equal x/y range on Q-Q plot so the reference line is at 45 degrees
    pad = (max_val - min_val) * 0.05
    fig.update_xaxes(range=[min_val - pad, max_val + pad], row=2, col=2)
    fig.update_yaxes(range=[min_val - pad, max_val + pad], row=2, col=2)

    # Axis labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Fitted Values", row=1, col=2)
    fig.update_yaxes(title_text="Residuals", row=1, col=2)
    fig.update_xaxes(title_text="Residuals", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
    fig.update_yaxes(title_text="Standardised Quantiles", row=2, col=2)

    fig = apply_default_layout(
        fig,
        title=title or "Residual Diagnostics",
        x_label=None,
        y_label=None,
        width=width,
        height=height,
    )
    fig.update_layout(showlegend=False)

    return fig


def plot_residuals(
    y_pred: pl.DataFrame,
    y_truth: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
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
    resampler: bool | Literal["widget"] | None = None,
    marker_size: float = 4,
    marker_opacity: float = 0.6,
    n_bins: int = 30,
) -> go.Figure:
    """Plot diagnostic plots for model residuals.

    When a single column is selected, creates a 4-panel layout with
    residuals over time, residuals vs fitted values, histogram of
    residuals, and Q-Q plot for checking normality assumptions.  When
    multiple columns are resolved (through *columns* or
    *groups*), produces a faceted layout showing residuals
    over time for each column.

    Residuals are computed internally as ``y_truth - y_pred`` for matching
    non-time columns.

    Parameters
    ----------
    y_pred : pl.DataFrame
        Predicted values with ``"time"`` column.
    y_truth : pl.DataFrame
        Ground-truth values with ``"time"`` column.
    columns : str | list[str] | None, default=None
        Column(s) to compute residuals for.  When *groups* is
        set, acts as a member postfix filter (e.g. ``["a"]`` selects
        ``y__a``).  When *None*, uses all common non-time columns. A
        single match triggers 4-panel diagnostics, multiple produce facets.
    groups : list[str] | None, default=None
        Panel group prefixes to facet by.
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data. ``"group"`` creates one subplot per
        group, ``"member"`` one per member. ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in the faceted grid when multiple target columns
        are resolved.
    color_palette : list[str] | None, default=None
        Custom color palette. If None, uses yohou palette.
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
    resampler : bool | Literal["widget"] | None, default=None
        Enable plotly-resampler for large datasets. ``True`` creates a
        ``FigureResampler``, ``"widget"`` a ``FigureWidgetResampler``;
        ``False`` or ``None`` uses a plain ``go.Figure``.
    marker_size : float, default=4
        Marker size for scatter plots.
    marker_opacity : float, default=0.6
        Marker opacity.
    n_bins : int, default=30
        Number of bins for histogram (single-column diagnostics).

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_residuals

    >>> dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
    >>> y_truth = pl.DataFrame({"time": dates, "y": [100 + i for i in range(91)]})
    >>> y_pred = pl.DataFrame({"time": dates, "y": [100 + i + (i % 3) for i in range(91)]})

    >>> fig = plot_residuals(y_pred, y_truth)
    >>> len(fig.data) > 0
    True

    See Also
    --------
    [`plot_forecast`][yohou.plotting.plot_forecast] : Plot forecasts with historical data.
    """
    validate_plotting_data(y_pred)
    validate_plotting_data(y_truth)
    validate_plotting_params(width=width, height=height)

    # Auto-detect panel data
    _, _panel_groups = inspect_panel(y_truth)
    if groups is None and columns is None and _panel_groups:
        groups = []

    # Resolve target columns
    if groups is not None:
        target_cols = validate_plotting_data(y_pred, columns=columns, groups=groups)
        missing = [c for c in target_cols if c not in y_truth.columns]
        if missing:
            msg = f"Columns {missing} not found in y_truth"
            raise ValueError(msg)
    elif columns is not None:
        target_cols = [columns] if isinstance(columns, str) else list(columns)
        for col in target_cols:
            if col not in y_pred.columns:
                msg = f"Column '{col}' not found in y_pred"
                raise ValueError(msg)
            if col not in y_truth.columns:
                msg = f"Column '{col}' not found in y_truth"
                raise ValueError(msg)
    else:
        pred_cols = [c for c in y_pred.columns if c != "time"]
        truth_cols = [c for c in y_truth.columns if c != "time"]
        common = [c for c in pred_cols if c in truth_cols]
        if not common:
            msg = "No common non-time columns found between y_pred and y_truth"
            raise ValueError(msg)
        target_cols = common

    # Compute residuals
    residual_exprs = [(pl.col(c) - y_pred[c]).alias(c) for c in target_cols]
    residuals_df = y_truth.select("time", *residual_exprs)

    # Single column: full 4-panel diagnostics
    if len(target_cols) == 1:
        return _render_residual_diagnostics(
            residuals_df,
            y_pred,
            target_cols[0],
            color_palette=color_palette,
            title=title,
            width=width,
            height=height,
            marker_size=marker_size,
            marker_opacity=marker_opacity,
            n_bins=n_bins,
        )

    # Multiple columns: faceted residuals over time

    if groups is not None:
        pn_cols = resolve_panel_columns(residuals_df, groups, columns)
        _, all_members = _group_panel_columns(pn_cols)
        member_palette = resolve_color_palette(color_palette, len(all_members))
        legend_tracker = LegendTracker()

        def _render_residual_scatter(ctx: RenderContext) -> None:
            """Render residuals over time for a single panel column."""
            base = [c for c in ctx.sub_df.columns if c != "time"][0]
            color = member_palette[ctx.entity_idx % len(member_palette)]
            ctx.fig.add_trace(
                go.Scatter(
                    x=ctx.sub_df["time"],
                    y=ctx.sub_df[base],
                    mode="markers",
                    name=ctx.display_name,
                    legendgroup=ctx.display_name,
                    marker={
                        "size": marker_size,
                        "color": color,
                        "opacity": marker_opacity,
                    },
                    showlegend=legend_tracker.should_show(ctx.display_name),
                    hovertemplate=_make_hovertemplate(ctx.display_name, "Time", "Residual", decimals=3),
                ),
                row=ctx.row,
                col=ctx.col,
            )
            ctx.fig.add_hline(
                y=0,
                line={"dash": "dash", "color": "#DC2626", "width": 1},
                row=ctx.row,
                col=ctx.col,
            )

        effective_facet_by = facet_by or "member"
        fig = facet_figure(
            residuals_df,
            _render_residual_scatter,
            groups=groups,
            columns=columns,
            facet_by=effective_facet_by,
            facet_n_cols=facet_n_cols,
            title=title or "Residual Diagnostics",
            x_label=x_label or "Time",
            y_label=y_label or "Residuals",
            width=width,
            height=height,
            resampler=resampler,
        )
        fig.update_layout(showlegend=show_legend)
        return fig

    # Non-panel multi-column: column-mode facet_figure
    _colors = resolve_color_palette(color_palette, len(target_cols))
    _col_colors = dict(zip(target_cols, _colors, strict=False))

    def _render_residual(ctx: RenderContext) -> None:
        """Render residual scatter for one column into a subplot."""
        base = ctx.display_name
        col_color = _col_colors[base]
        ctx.fig.add_trace(
            go.Scatter(
                x=residuals_df["time"],
                y=residuals_df[base],
                mode="markers",
                marker={"size": marker_size, "color": col_color, "opacity": marker_opacity},
                showlegend=False,
            ),
            row=ctx.row,
            col=ctx.col,
        )
        ctx.fig.add_hline(
            y=0,
            line={"dash": "dash", "color": "#DC2626", "width": 1},
            row=ctx.row,
            col=ctx.col,
        )

    fig = facet_figure(
        residuals_df,
        _render_residual,
        columns=target_cols,
        facet_n_cols=facet_n_cols,
        title=title or "Residual Diagnostics",
        x_label=x_label or "Time",
        y_label=y_label or "Residuals",
        width=width,
        height=height,
        shared_xaxes=True,
        resampler=resampler,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


_COVERAGE_RATE_RE = re.compile(r"_(?:lower|upper)_(\d*\.?\d+)$")


def _detect_coverage_rates(y_pred: pl.DataFrame) -> list[float]:
    """Discover coverage rates from a prediction frame's interval columns.

    Scans for the ``{base}_lower_{rate}`` / ``{base}_upper_{rate}`` naming and
    returns the sorted rates that have **both** a lower and an upper bound.

    Parameters
    ----------
    y_pred : pl.DataFrame
        Prediction frame with interval bound columns.

    Returns
    -------
    list of float
        Sorted coverage rates present as matched lower/upper pairs.
    """
    lowers: set[str] = set()
    uppers: set[str] = set()
    for col in y_pred.columns:
        match = _COVERAGE_RATE_RE.search(col)
        if match is None:
            continue
        (uppers if "_upper_" in col else lowers).add(match.group(1))
    return sorted(float(rate) for rate in (lowers & uppers))


def _compute_empirical_coverages(
    y_truth_col: np.ndarray,
    y_pred_int: pl.DataFrame,
    target_column: str,
    coverage_rates: list[StrictFloat],
) -> list[float]:
    """Compute empirical coverage for a single target column.

    Parameters
    ----------
    y_truth_col : np.ndarray
        Ground-truth values for one target column.
    y_pred_int : pl.DataFrame
        Prediction intervals DataFrame.
    target_column : str
        Column prefix used to find ``{target_column}_lower_{rate}``
        and ``{target_column}_upper_{rate}`` columns.
    coverage_rates : list of float
        Nominal coverage rates.

    Returns
    -------
    list of float
        Empirical coverage for each rate.

    Raises
    ------
    ValueError
        If the expected interval columns are missing.

    """
    empirical: list[float] = []
    for rate in coverage_rates:
        upper_col = f"{target_column}_upper_{rate}"
        lower_col = f"{target_column}_lower_{rate}"
        if upper_col not in y_pred_int.columns or lower_col not in y_pred_int.columns:
            msg = f"Interval columns '{upper_col}' and '{lower_col}' not found in y_pred_int"
            raise ValueError(msg)
        lower_vals = y_pred_int[lower_col].to_numpy().flatten()
        upper_vals = y_pred_int[upper_col].to_numpy().flatten()
        inside = np.logical_and(
            np.greater_equal(y_truth_col, lower_vals),
            np.less_equal(y_truth_col, upper_vals),
        )
        empirical.append(float(np.mean(inside)))
    return empirical


def _compute_quantile_coverages(
    y_truth_col: np.ndarray,
    y_pred_int: pl.DataFrame,
    target_column: str,
    coverage_rates: list[StrictFloat],
) -> tuple[list[float], list[float]]:
    """Empirical CDF coverage ``P(y <= q)`` at each quantile level for one column.

    The two-sided interval bounds are reinterpreted as one-sided quantiles: a
    band at coverage ``c`` contributes the lower quantile ``(1 - c) / 2`` and
    the upper quantile ``(1 + c) / 2``.  The point/median column, when present,
    contributes the ``0.5`` quantile.

    Parameters
    ----------
    y_truth_col : np.ndarray
        Ground-truth values for one target column.
    y_pred_int : pl.DataFrame
        Prediction intervals DataFrame.
    target_column : str
        Column prefix used to find ``{target_column}_lower_{rate}`` /
        ``{target_column}_upper_{rate}`` bounds and the median column
        ``{target_column}``.
    coverage_rates : list of float
        Nominal coverage rates whose bounds are reinterpreted as quantiles.

    Returns
    -------
    tuple[list of float, list of float]
        Sorted nominal quantile levels and the matching empirical CDF coverage.

    Raises
    ------
    ValueError
        If the expected interval columns are missing.
    """
    quantiles: dict[float, np.ndarray] = {}
    for rate in coverage_rates:
        upper_col = f"{target_column}_upper_{rate}"
        lower_col = f"{target_column}_lower_{rate}"
        if upper_col not in y_pred_int.columns or lower_col not in y_pred_int.columns:
            msg = f"Interval columns '{upper_col}' and '{lower_col}' not found in y_pred_int"
            raise ValueError(msg)
        quantiles[round((1.0 - rate) / 2.0, 6)] = y_pred_int[lower_col].to_numpy().flatten()
        quantiles[round((1.0 + rate) / 2.0, 6)] = y_pred_int[upper_col].to_numpy().flatten()
    # The median/point column (same name as the truth column) is the 0.5 quantile.
    if target_column in y_pred_int.columns:
        quantiles[0.5] = y_pred_int[target_column].to_numpy().flatten()

    levels = sorted(quantiles)
    empirical = [float(np.mean(np.less_equal(y_truth_col, quantiles[level]))) for level in levels]
    return levels, empirical


def _plot_calibration_class_proba(
    y_pred: pl.DataFrame,
    y_truth: pl.DataFrame,
    *,
    target: str | None,
    n_bins: int,
    color_palette: list[str] | None,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    width: int | None,
    height: int | None,
    show_legend: bool = True,
    line_width: float = 2.0,
    reference_color: str = "#1e293b",
    reference_dash: str = "dash",
) -> go.Figure:
    """Reliability diagram for class-probability predictions.

    Parameters
    ----------
    y_pred : pl.DataFrame
        Class-probability predictions with ``{target}_proba_{class}``
        columns.
    y_truth : pl.DataFrame
        Ground truth with categorical target columns.
    target : str or None
        Target column name.
    n_bins : int
        Number of probability bins.
    color_palette : list of str or None
        Custom color palette.
    title : str or None
        Figure title.
    x_label : str or None
        X-axis label.
    y_label : str or None
        Y-axis label.
    width : int or None
        Figure width.
    height : int or None
        Figure height.
    show_legend : bool, default=True
        Whether to show the legend.
    line_width : float, default=2.0
        Width of the model series line.
    reference_color : str, default="#1e293b"
        Color of the diagonal reference line.
    reference_dash : str, default="dash"
        Dash style of the diagonal reference line.

    Returns
    -------
    go.Figure
        Reliability diagram figure.

    """
    validate_plotting_data(y_pred, min_rows=1)
    validate_plotting_data(y_truth, min_rows=1)

    proba_cols = [c for c in y_pred.columns if "_proba_" in c]
    targets = _discover_proba_targets(proba_cols)

    if target is None:
        if len(targets) > 1:
            msg = f"Multiple targets found: {sorted(targets)}. Please specify the 'target' parameter."
            raise ValueError(msg)
        target = next(iter(targets))

    if target not in targets:
        msg = f"Target '{target}' not found. Available targets: {sorted(targets)}"
        raise ValueError(msg)

    if target not in y_truth.columns:
        msg = f"Target '{target}' not found in y_truth columns: {y_truth.columns}"
        raise ValueError(msg)

    target_proba_cols = targets[target]
    class_labels = [c.split("_proba_", 1)[1] for c in target_proba_cols]
    colors = resolve_color_palette(color_palette, n=len(class_labels))

    # Align on time
    common = y_pred.select("time").join(y_truth.select("time"), on="time", how="inner")
    pred_aligned = y_pred.join(common, on="time", how="inner").sort("time")
    truth_aligned = y_truth.join(common, on="time", how="inner").sort("time")
    true_labels = truth_aligned[target].cast(pl.String)

    fig = go.Figure()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for i, (pcol, label) in enumerate(zip(target_proba_cols, class_labels, strict=True)):
        predicted_probs = pred_aligned[pcol].to_numpy()
        is_class = (true_labels == label).to_numpy()

        mean_pred: list[float] = []
        empirical_freq: list[float] = []

        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if b < n_bins - 1:
                mask = (predicted_probs >= lo) & (predicted_probs < hi)
            else:
                mask = (predicted_probs >= lo) & (predicted_probs <= hi)
            count = int(mask.sum())
            if count == 0:
                continue
            mean_pred.append(float(predicted_probs[mask].mean()))
            empirical_freq.append(float(is_class[mask].mean()))

        if mean_pred:
            fig.add_trace(
                go.Scatter(
                    x=mean_pred,
                    y=empirical_freq,
                    mode="lines+markers",
                    name=label,
                    line={"color": colors[i], "width": line_width},
                    hovertemplate=(
                        f"<b>{label}</b><br>Predicted: %{{x:.3f}}<br>Observed: %{{y:.3f}}<br><extra></extra>"
                    ),
                )
            )

    # Perfect calibration diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line={"color": reference_color, "width": 2.0, "dash": reference_dash},
            hovertemplate="<b>Perfect</b><br>%{x:.2f}<extra></extra>",
        )
    )

    fig = apply_default_layout(
        fig,
        title=title or "Reliability Diagram",
        x_label=x_label or "Mean predicted probability",
        y_label=y_label or "Empirical frequency",
        width=width,
        height=height,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


def plot_calibration(
    y_pred: pl.DataFrame,
    y_truth: pl.DataFrame,
    coverage_rates: list[StrictFloat] | None = None,
    *,
    type: Literal["interval", "quantile"] = "quantile",
    columns: str | list[str] | None = None,
    target: str | None = None,
    n_bins: int = 10,
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
    line_width: float = 2.0,
    line_opacity: float = 1.0,
    reference_color: str = "#1e293b",
    reference_width: float = 3.0,
    reference_dash: str = "dash",
) -> go.Figure:
    """Plot calibration for interval or class-probability forecasts.

    Automatically detects the prediction type from column names:

    - **Interval / quantile predictions**: columns named
      ``"{target}_upper_{rate}"`` / ``"{target}_lower_{rate}"``.  With
      ``type="interval"`` the two-sided band coverage
      ``P(lower <= y <= upper)`` is compared against the nominal
      *coverage_rates*.  With ``type="quantile"`` each bound is treated as
      a single quantile (the band at coverage ``c`` yields quantile levels
      ``(1 - c) / 2`` and ``(1 + c) / 2``, and the median/point column adds
      the ``0.5`` level), and the empirical CDF coverage ``P(y <= q)`` is
      compared against the nominal quantile level.
    - **Class-probability predictions**: columns containing
      ``"_proba_"`` are binned and compared against empirical
      frequencies (reliability diagram).  *type* is ignored.

    A well-calibrated model has points close to the diagonal.

    Parameters
    ----------
    y_pred : pl.DataFrame
        Predicted values.  Either prediction intervals with
        ``"{target}_upper_{rate}"`` / ``"{target}_lower_{rate}"``
        columns, or class-probability predictions with
        ``"{target}_proba_{class}"`` columns.
    y_truth : pl.DataFrame
        Ground truth values.
    coverage_rates : list of float or None, default=None
        Nominal coverage rates for interval calibration (e.g.,
        ``[0.9, 0.95]``).  When ``None``, every coverage rate present as a
        matched ``_lower_{rate}`` / ``_upper_{rate}`` pair in *y_pred* is
        detected and plotted.  Ignored for class-probability predictions.
    type : {"interval", "quantile"}, default="quantile"
        Calibration mode for bound-based predictions.  ``"quantile"``
        reinterprets each bound (and the median column) as a single quantile
        and plots the empirical CDF coverage ``P(y <= q)`` against the nominal
        quantile level; ``"interval"`` plots two-sided band coverage against
        the nominal coverage rate.  Ignored for class-probability predictions.
    columns : str | list[str] | None, default=None
        Target column name(s) for interval calibration.  When
        *groups* is set this acts as a member postfix
        filter.  Ignored for class-probability predictions (use
        *target* instead).
    target : str or None, default=None
        Target column name for class-probability calibration.  Required
        when multiple targets are present in the probability columns.
        Ignored for interval calibration.
    n_bins : int, default=10
        Number of bins for class-probability calibration.  Ignored for
        interval calibration.
    groups : list[str] | None, default=None
        Panel group prefixes for faceted subplots.  When provided, each
        resolved panel column gets its own subplot.
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data. ``"group"`` creates one subplot per
        group, ``"member"`` one per member. ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in the facet grid when *groups* is
        used.
    color_palette : list[str] | None, default=None
        Custom color palette as hex codes. If None, uses yohou palette.
    show_legend : bool, default=True
        Whether to show the legend.
    title : str | None, default=None
        Plot title. Defaults to ``"Calibration plot"`` for intervals or
        ``"Reliability Diagram"`` for class probabilities.
    x_label : str | None, default=None
        X-axis label.
    y_label : str | None, default=None
        Y-axis label.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    line_width : float, default=2.0
        Width of the calibration line.
    line_opacity : float, default=1.0
        Opacity of the calibration line.
    reference_color : str, default="#1e293b"
        Colour of the perfect-calibration reference line.
    reference_width : float, default=3.0
        Width of the reference line.
    reference_dash : str, default="dash"
        Dash style of the reference line.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    ValueError
        If interval columns are missing, *coverage_rates* is not
        provided for interval predictions, or class-probability columns
        are ambiguous.

    Examples
    --------
    Interval calibration:

    >>> import polars as pl
    >>> import numpy as np
    >>> from datetime import datetime, timedelta
    >>> from yohou.plotting import plot_calibration

    >>> # Create sample data
    >>> n = 100
    >>> rng = np.random.default_rng(0)
    >>> times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    >>> y_truth = pl.DataFrame({"time": times, "y": rng.standard_normal(n)})
    >>> y_pred_int = pl.DataFrame({
    ...     "time": times,
    ...     "y_upper_0.9": rng.standard_normal(n) + 1.65,
    ...     "y_lower_0.9": rng.standard_normal(n) - 1.65,
    ...     "y_upper_0.95": rng.standard_normal(n) + 1.96,
    ...     "y_lower_0.95": rng.standard_normal(n) - 1.96,
    ... })

    >>> # Plot calibration
    >>> fig = plot_calibration(y_pred_int, y_truth, coverage_rates=[0.9, 0.95])
    >>> len(fig.data)
    2

    Class-probability calibration (reliability diagram):

    >>> from datetime import datetime
    >>> y_pred_proba = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 11)],
    ...     "w_proba_sunny": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.85],
    ...     "w_proba_rainy": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.15],
    ... })
    >>> y_truth_cat = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 11)],
    ...     "w": ["sunny", "sunny", "sunny", "rainy", "rainy", "rainy", "rainy", "rainy", "sunny", "sunny"],
    ... })
    >>> fig = plot_calibration(y_pred_proba, y_truth_cat)
    >>> isinstance(fig, go.Figure)
    True

    See Also
    --------
    [`plot_forecast`][yohou.plotting.plot_forecast] : Plot forecast with optional prediction intervals.
    [`plot_residuals`][yohou.plotting.plot_residuals] : Residual diagnostics with panel facets.
    """
    # Validate inputs
    if not isinstance(y_truth, pl.DataFrame):
        msg = f"Expected pl.DataFrame for y_truth, got {y_truth.__class__.__name__}"
        raise TypeError(msg)
    if not isinstance(y_pred, pl.DataFrame):
        msg = f"Expected pl.DataFrame for y_pred, got {y_pred.__class__.__name__}"
        raise TypeError(msg)
    validate_plotting_params(width=width, height=height)

    # Detect class-probability columns
    proba_cols = [c for c in y_pred.columns if "_proba_" in c]
    if proba_cols:
        return _plot_calibration_class_proba(
            y_pred=y_pred,
            y_truth=y_truth,
            target=target,
            n_bins=n_bins,
            color_palette=color_palette,
            title=title,
            x_label=x_label,
            y_label=y_label,
            width=width,
            height=height,
            show_legend=show_legend,
            line_width=line_width,
            reference_color=reference_color,
            reference_dash=reference_dash,
        )

    # Interval / quantile calibration path. When the caller does not pin the
    # rates, plot every coverage rate present as a matched lower/upper pair.
    if coverage_rates is None:
        coverage_rates = _detect_coverage_rates(y_pred)
        if not coverage_rates:
            msg = (
                "No interval columns found in y_pred for calibration. Expected "
                "'{base}_lower_{rate}' / '{base}_upper_{rate}' columns, or pass "
                "coverage_rates explicitly (e.g., [0.9, 0.95])."
            )
            raise ValueError(msg)

    # Mode-dependent labels: quantile mode plots one-sided CDF coverage against
    # the nominal quantile level, interval mode two-sided coverage vs the rate.
    _is_quantile = type == "quantile"
    _default_title = "Quantile calibration" if _is_quantile else "Calibration plot"
    _default_x_label = "Nominal quantile level" if _is_quantile else "Nominal coverage"
    _hover_x = "Quantile" if _is_quantile else "Nominal"

    def _calibration_xy(panel_col: str) -> tuple[list[float], list[float]]:
        """Return ``(x, empirical)`` for one column under the active mode."""
        truth_vals = y_truth[panel_col].to_numpy().flatten()
        if _is_quantile:
            return _compute_quantile_coverages(truth_vals, y_pred, panel_col, coverage_rates)
        return list(coverage_rates), _compute_empirical_coverages(truth_vals, y_pred, panel_col, coverage_rates)

    # Auto-detect panel data
    _, _panel_groups = inspect_panel(y_truth)
    if groups is None and columns is None and _panel_groups:
        groups = []

    if groups is not None:
        panel_cols = resolve_panel_columns(y_truth, groups, columns)

        # Group columns by panel prefix, collect unique members
        grouped, all_members = _group_panel_columns(panel_cols)

        n_groups = len(grouped)
        n_cols_grid = min(n_groups, facet_n_cols)
        n_rows = (n_groups + n_cols_grid - 1) // n_cols_grid

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols_grid,
            subplot_titles=list(grouped.keys()),
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=_subplot_spacing(n_rows),
            horizontal_spacing=0.08,
        )

        palette = resolve_color_palette(color_palette, len(all_members))
        legend_tracker = LegendTracker(show_legend=show_legend)
        for group_idx, (_, group_cols) in enumerate(grouped.items()):
            row = group_idx // n_cols_grid + 1
            col_idx = group_idx % n_cols_grid + 1

            for panel_col in group_cols:
                member_name = _member_name(panel_col)
                member_idx = all_members.index(member_name)

                x_vals, emp_cov = _calibration_xy(panel_col)

                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=emp_cov,
                        mode="lines+markers",
                        name=member_name,
                        line={"color": palette[member_idx], "width": line_width},
                        opacity=line_opacity,
                        showlegend=legend_tracker.should_show(member_name),
                        legendgroup=member_name,
                        hovertemplate=(
                            f"<b>%{{fullData.name}}</b><br>{_hover_x}: %{{x:.2f}}<br>Coverage: %{{y:.3f}}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col_idx,
                )

            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Perfect",
                    line={"color": reference_color, "width": reference_width, "dash": reference_dash},
                    showlegend=legend_tracker.should_show("Perfect"),
                    legendgroup="perfect",
                ),
                row=row,
                col=col_idx,
            )

        row_height = 300
        default_height = max(row_height * n_rows, 400)
        fig = apply_default_layout(
            fig,
            title=title or _default_title,
            x_label=x_label or _default_x_label,
            y_label=y_label or "Empirical coverage",
            width=width,
            height=height or default_height,
        )
        fig.update_layout(showlegend=show_legend)
        # Calibration axes are coverage probabilities: pin both to the full
        # [0, 1] range and force a square aspect (1:1 scale, box constrained to
        # the domain) so every subplot reads on the same scale and the diagonal
        # sits at a true 45 degrees.  ``make_subplots`` shares axes within each
        # row/column via ``matches``; a matched axis cannot also honour its own
        # ``scaleanchor``, which leaves only the first column square.  Clearing
        # ``matches`` lets each cell apply its own square constraint.
        fig.update_xaxes(range=[0, 1], constrain="domain")
        fig.update_yaxes(range=[0, 1], constrain="domain")
        for _xaxis in [_k for _k in fig.layout if _k.startswith("xaxis")]:
            fig.layout[_xaxis].matches = None
        for _yaxis in [_k for _k in fig.layout if _k.startswith("yaxis")]:
            fig.layout[_yaxis].matches = None
            fig.layout[_yaxis].scaleanchor = "x" + _yaxis.removeprefix("yaxis")
            fig.layout[_yaxis].scaleratio = 1
        return fig

    if columns is not None:
        target_columns = [columns] if isinstance(columns, str) else list(columns)
        for col in target_columns:
            if col not in y_truth.columns:
                msg = f"Target column '{col}' not found in y_truth"
                raise ValueError(msg)
    else:
        target_columns = [c for c in y_truth.columns if c not in ("time", "vintage_time")]
        if not target_columns:
            msg = "y_truth has no non-time columns"
            raise ValueError(msg)

    palette = resolve_color_palette(color_palette, len(target_columns))

    fig = go.Figure()

    for col_idx, target_column in enumerate(target_columns):
        x_vals, emp_cov = _calibration_xy(target_column)

        trace_name = target_column if len(target_columns) > 1 else "Empirical coverage"
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=emp_cov,
                mode="lines+markers",
                name=trace_name,
                line={"color": palette[col_idx % len(palette)], "width": line_width},
                opacity=line_opacity,
                hovertemplate=f"<b>{trace_name}</b><br>{_hover_x}: %{{x:.2f}}<br>Coverage: %{{y:.3f}}<extra></extra>",
            )
        )

    # Reference diagonal (always exactly one), spanning the full [0, 1] range.
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line={"color": reference_color, "width": reference_width, "dash": reference_dash},
            hovertemplate="<b>Perfect</b><br>Coverage: %{x:.2f}<extra></extra>",
        )
    )

    fig = apply_default_layout(
        fig,
        title=title or _default_title,
        x_label=x_label or _default_x_label,
        y_label=y_label or "Empirical coverage",
        width=width,
        height=height,
    )
    fig.update_layout(showlegend=show_legend)
    # Calibration axes are coverage probabilities: pin both to the full [0, 1]
    # range and force a square aspect (1:1 scale, box constrained to the domain)
    # so the diagonal sits at a true 45 degrees.
    fig.update_xaxes(range=[0, 1], constrain="domain")
    fig.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1, constrain="domain")

    return fig


def _plot_score_time_series_panel(
    *,
    scorer_componentwise: BaseScorer,
    y_truth: pl.DataFrame,
    y_pred_dict: dict[str, pl.DataFrame],
    groups: list[str],
    facet_n_cols: int,
    colors: list[str],
    scorer: BaseScorer,
    show_legend: bool,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    width: int | None,
    height: int | None,
    connect_gaps: bool = False,
    resampler: bool | Literal["widget"] | None = None,
    columns: str | list[str] | None = None,
    line_width: float = 2.0,
    line_dash: str = "solid",
    line_opacity: float = 1.0,
    show_markers: bool = False,
) -> go.Figure:
    """Render faceted per-group score time series.

    Parameters
    ----------
    scorer_componentwise : BaseScorer
        Scorer clone already configured for componentwise aggregation.
    y_truth : pl.DataFrame
        Ground truth values.
    y_pred_dict : dict[str, pl.DataFrame]
        Model name to prediction DataFrame mapping.
    groups : list[str]
        Panel group prefixes to facet by.
    facet_n_cols : int
        Number of columns in the facet grid.
    colors : list[str]
        Resolved color palette (one per model).
    scorer : BaseScorer
        Original scorer (used for title generation).
    show_legend : bool
        Whether to show legend.
    title : str | None
        Plot title override.
    x_label : str | None
        X-axis label override.
    y_label : str | None
        Y-axis label override.
    width : int | None
        Plot width in pixels.
    height : int | None
        Plot height in pixels.
    line_width : float, default=2.0
        Width of score lines.
    line_dash : str, default="solid"
        Dash style of score lines.
    line_opacity : float, default=1.0
        Opacity of score lines.
    show_markers : bool, default=False
        Whether to show markers on the lines.

    Returns
    -------
    go.Figure
        Faceted figure with one subplot per panel group.

    """
    mode = "lines+markers" if show_markers else "lines"

    _, all_groups = inspect_panel(y_truth)
    requested_groups = groups
    groups = list(all_groups) if not groups else [g for g in groups if g in all_groups]
    if not groups:
        msg = f"No panel groups found matching {requested_groups}. Available: {list(all_groups)}"
        raise ValueError(msg)

    n_groups = len(groups)
    n_cols_grid = min(n_groups, facet_n_cols)
    n_rows = (n_groups + n_cols_grid - 1) // n_cols_grid

    fig = _create_subplots(
        resampler,
        rows=n_rows,
        cols=n_cols_grid,
        subplot_titles=list(groups),
        shared_xaxes=True,
        vertical_spacing=max(0.04, 0.3 / n_rows),
        horizontal_spacing=0.08,
    )

    legend_tracker = LegendTracker(show_legend=show_legend)

    for model_idx, (model_name, y_pred_model) in enumerate(y_pred_dict.items()):
        validate_plotting_data(y_pred_model)
        scores_df = scorer_componentwise.score(y_truth, y_pred_model)

        if not isinstance(scores_df, pl.DataFrame):
            msg = f"Scorer must return DataFrame for componentwise aggregation, got {type(scores_df).__name__}"
            raise TypeError(msg)

        if "time" not in scores_df.columns:
            msg = "Scorer must return DataFrame with 'time' column for componentwise aggregation"
            raise ValueError(msg)

        score_cols = [c for c in scores_df.columns if c not in _SCORER_META_COLS]

        for group_idx, group_name in enumerate(groups):
            row = group_idx // n_cols_grid + 1
            col = group_idx % n_cols_grid + 1

            group_score_cols = [c for c in score_cols if c.startswith(f"{group_name}__")]
            if columns is not None:
                col_filter = [columns] if isinstance(columns, str) else list(columns)
                group_score_cols = [c for c in group_score_cols if _member_name(c) in col_filter]
            if not group_score_cols:
                continue

            if len(group_score_cols) == 1:
                score_values = scores_df[group_score_cols[0]]
            else:
                score_values = scores_df.select(group_score_cols).mean_horizontal()

            trace_df = scores_df.select([c for c in scores_df.columns if c in {"time", "vintage_time"}]).with_columns(
                score_values.alias("_score")
            )

            _add_vintage_traces(
                fig,
                trace_df,
                x_col="time",
                y_col="_score",
                name=model_name,
                color=colors[model_idx],
                line_width=line_width,
                line_dash=line_dash,
                base_opacity=line_opacity,
                mode=mode,
                show_markers=show_markers,
                connect_gaps=connect_gaps,
                showlegend=legend_tracker.should_show(model_name),
                legendgroup=model_name,
                row=row,
                col=col,
                hovertemplate_extra=group_name,
            )

    scorer_name = scorer.__class__.__name__
    default_title = title or f"{scorer_name} Over Time"

    row_height = 300
    default_height = max(row_height * n_rows, 400)

    fig = apply_default_layout(
        fig,
        title=default_title,
        x_label=x_label or "Time",
        y_label=y_label or scorer_name,
        width=width,
        height=height or default_height,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


def _add_vintage_traces(
    fig: go.Figure,
    df: pl.DataFrame,
    *,
    x_col: str,
    y_col: str,
    name: str,
    color: str,
    line_width: float,
    line_dash: str,
    base_opacity: float,
    mode: str,
    show_markers: bool,
    connect_gaps: bool,
    showlegend: bool,
    decimals: int = 3,
    legendgroup: str | None = None,
    row: int | None = None,
    col: int | None = None,
    hovertemplate_extra: str = "",
) -> None:
    """Add one or more traces, splitting by vintage when ``vintage_time`` is present.

    When the DataFrame has an ``vintage_time`` column with multiple unique
    values, one ``go.Scatter`` trace is created per vintage.  Older vintages
    are drawn with lower opacity and hidden from the legend so that only the
    newest vintage trace represents the series.
    """
    has_vintages = "vintage_time" in df.columns and df["vintage_time"].n_unique() > 1

    if not has_vintages:
        base_ht = _make_hovertemplate(name, "Time", "Score", decimals=decimals, extra=hovertemplate_extra)
        trace = go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode=mode,
            name=name,
            legendgroup=legendgroup,
            showlegend=showlegend,
            line={"color": color, "width": line_width, "dash": line_dash},
            opacity=base_opacity,
            marker={"size": 6} if show_markers else None,
            connectgaps=connect_gaps,
            hovertemplate=base_ht,
        )
        kwargs: dict = {}
        if row is not None:
            kwargs["row"] = row
            kwargs["col"] = col
        fig.add_trace(trace, **kwargs)
        return

    vintages = df["vintage_time"].unique().sort().to_list()
    n = len(vintages)
    base_ht = _make_hovertemplate(name, "Time", "Score", decimals=decimals, extra=hovertemplate_extra)

    for rank, v in enumerate(vintages):
        opacity = 0.3 + 0.7 * (rank / (n - 1))
        is_newest = rank == n - 1
        vintage_ht = base_ht.replace("<extra>", f"<br>Vintage: {v}<extra>")

        trace = go.Scatter(
            x=df.filter(pl.col("vintage_time") == v)[x_col],
            y=df.filter(pl.col("vintage_time") == v)[y_col],
            mode=mode,
            name=name,
            legendgroup=legendgroup,
            showlegend=showlegend and is_newest,
            line={"color": color, "width": line_width, "dash": line_dash},
            opacity=opacity,
            marker={"size": 6} if show_markers else None,
            connectgaps=connect_gaps,
            hovertemplate=vintage_ht,
        )
        kwargs = {}
        if row is not None:
            kwargs["row"] = row
            kwargs["col"] = col
        fig.add_trace(trace, **kwargs)


def _compute_componentwise_scores(
    scorer_cw: BaseScorer,
    y_truth: pl.DataFrame,
    y_pred_model: pl.DataFrame,
    columns: str | list[str] | None,
) -> pl.DataFrame:
    """Compute componentwise scores for one scorer-model pair.

    Returns
    -------
    pl.DataFrame
        DataFrame with ``time``, optionally ``vintage_time``, and a single
        ``score`` column containing the (possibly averaged) metric values.
    """
    validate_plotting_data(y_pred_model)

    scores_df = scorer_cw.score(y_truth, y_pred_model)

    if not isinstance(scores_df, pl.DataFrame):
        msg = f"Scorer must return DataFrame for componentwise aggregation, got {type(scores_df).__name__}"
        raise TypeError(msg)

    if "time" not in scores_df.columns:
        msg = "Scorer must return DataFrame with 'time' column for componentwise aggregation"
        raise ValueError(msg)

    score_columns = [col for col in scores_df.columns if col not in _SCORER_META_COLS]

    if columns is not None:
        col_filter = [columns] if isinstance(columns, str) else list(columns)
        score_columns = [c for c in score_columns if c in col_filter]
        if not score_columns:
            msg = f"None of the requested columns {col_filter!r} found in scorer output"
            raise ValueError(msg)

    if len(score_columns) == 1:
        score_values = scores_df[score_columns[0]]
    else:
        score_values = scores_df.select(score_columns).mean_horizontal()

    keep_cols = ["time"]
    if "vintage_time" in scores_df.columns:
        keep_cols.append("vintage_time")
    return scores_df.select(keep_cols).with_columns(score_values.alias("score"))


def plot_score_time_series(
    scorer: BaseScorer | dict[str, BaseScorer],
    y_truth: pl.DataFrame,
    y_pred: pl.DataFrame | dict[str, pl.DataFrame],
    *,
    compare_by: Literal["scorer", "model"] = "scorer",
    columns: str | list[str] | None = None,
    groups: list[str] | None = None,
    facet_by: Literal["group", "member", "vintage"] | None = "member",
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
    line_dash: str = "solid",
    line_opacity: float = 1.0,
    show_markers: bool = False,
) -> go.Figure:
    """Plot scorer values over time for one or more forecasts.

    Evaluates forecast quality at each timestep by computing the scorer with
    componentwise aggregation, then plots the resulting score time series.
    Useful for identifying periods where forecast performance varies.

    Parameters
    ----------
    scorer : BaseScorer or dict[str, BaseScorer]
        Yohou scorer instance (e.g., MeanAbsoluteError, RootMeanSquaredError).
        Will be cloned and configured with aggregation_method="componentwise".

        - If BaseScorer: single scorer to evaluate.
        - If dict: keys are scorer names, values are scorer instances.
          When combined with dict ``y_pred``, the ``compare_by`` parameter
          controls which dimension is faceted vs overlaid.
    y_truth : pl.DataFrame
        Ground truth values with 'time' column.
    y_pred : pl.DataFrame or dict[str, pl.DataFrame]
        Predicted values with 'vintage_time' and 'time' columns.
        - If DataFrame: single forecast to plot
        - If dict: multiple forecasts with keys as model names
    compare_by : str, default="scorer"
        When both ``scorer`` and ``y_pred`` are dicts, controls which
        dimension is overlaid (colored lines) vs faceted (subplots):

        - ``"scorer"``: overlay scorers, facet by model.
        - ``"model"``: overlay models, facet by scorer.

        Ignored when either ``scorer`` or ``y_pred`` is not a dict.
    columns : str | list[str] | None, default=None
        Target column name(s) to include in the score.  When
        *groups* is set, acts as a member postfix filter
        (e.g. ``"a"`` selects ``group__a``).  When ``None``, all score
        columns are used.
    groups : list[str] | None, default=None
        Panel group prefixes for faceted subplots.  When provided, each
        group gets its own subplot showing the score time series for that
        group.  Groups are resolved via ``inspect_panel`` against
        *y_truth*.
    facet_by : Literal["group", "member", "vintage"] | None, default="member"
        Faceting axis for panel data or vintage data.  ``"group"`` creates
        one subplot per group, ``"member"`` one per member. ``"vintage"``
        creates one subplot per ``vintage_time`` value found in *y_pred*,
        showing how score evolves across forecast origins.
        ``None`` disables faceting. ``"group"`` and ``"member"`` are
        ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in the facet grid when *groups* is
        used.
    color_palette : list[str] | None, default=None
        Custom color palette as hex codes. If None, uses yohou palette.
    show_legend : bool, default=True
        Whether to show legend when plotting multiple forecasts.
    title : str | None, default=None
        Plot title. If None, generates title from scorer name.
    x_label : str | None, default=None
        X-axis label. Defaults to "time".
    y_label : str | None, default=None
        Y-axis label. If None, uses scorer class name.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    connect_gaps : bool, default=False
        Whether to connect gaps in the data with lines.
    resampler : bool | Literal["widget"] | None, default=None
        Enable plotly-resampler for large datasets.  ``True`` creates a
        ``FigureResampler``, ``"widget"`` a ``FigureWidgetResampler``;
        ``False`` or ``None`` uses a plain ``go.Figure``.
    line_width : float, default=2.0
        Width of score lines.
    line_dash : str, default="solid"
        Dash style of score lines.
    line_opacity : float, default=1.0
        Opacity of score lines.
    show_markers : bool, default=False
        Whether to show markers on the lines.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    TypeError
        If y_truth or y_pred is not a Polars DataFrame.
    ValueError
        If DataFrames are empty or missing required columns, if *scorer*
        is a dict (multi-scorer) and panel data is detected in *y_truth*,
        or if *scorer* is a dict and ``facet_by="vintage"``.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MeanAbsoluteError
    >>> from yohou.plotting import plot_score_time_series

    >>> # Create sample data
    >>> y_truth = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [10.0, 20.0, 30.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "vintage_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [12.0, 19.0, 28.0],
    ... })

    >>> # Plot score time series for single forecast
    >>> scorer = MeanAbsoluteError()
    >>> fig = plot_score_time_series(scorer, y_truth, y_pred)
    >>> len(fig.data)
    1

    >>> # Plot multiple forecasts
    >>> y_pred2 = pl.DataFrame({
    ...     "vintage_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [11.0, 21.0, 29.0],
    ... })
    >>> fig = plot_score_time_series(scorer, y_truth, {"Model A": y_pred, "Model B": y_pred2})
    >>> len(fig.data)
    2

    See Also
    --------
    [`plot_residuals`][yohou.plotting.plot_residuals] : Plot residual diagnostics.
    [`plot_forecast`][yohou.plotting.plot_forecast] : Plot forecasts with historical data.

    Notes
    -----
    - Scorer is automatically configured with aggregation_method="componentwise"
    - For interval scorers, use aggregation_method=["componentwise", "coveragewise"]
    - Requires scorer to support componentwise aggregation
    - All scores are computed independently at each timestep
    - Use ``facet_by="vintage"`` to compare score curves across forecast
      origins (requires ``y_pred`` with multiple ``vintage_time`` values)

    """
    # Validate ground truth
    validate_plotting_data(y_truth)
    validate_plotting_params(width=width, height=height)

    # Normalize inputs
    y_pred_dict = _normalize_y_pred(y_pred)
    scorer_dict = _normalize_scorers(scorer)

    # Prepare and fit each scorer for componentwise aggregation
    scorer_cw_dict: dict[str, BaseScorer] = {}
    for s_name, s in scorer_dict.items():
        s_cw = _prepare_scorer_for_componentwise(s)
        s_cw.fit(y_truth)
        scorer_cw_dict[s_name] = s_cw

    n_scorers = len(scorer_cw_dict)
    n_models = len(y_pred_dict)
    multi_scorer = n_scorers > 1

    # Auto-detect panel data
    _, _panel_groups = inspect_panel(y_truth)
    if groups is None and _panel_groups:
        groups = []

    if groups is not None:
        if facet_by == "vintage":
            msg = "facet_by='vintage' cannot be combined with panel groups. Use facet_by='group' or 'member' for panel data."
            raise ValueError(msg)
        if multi_scorer:
            msg = (
                "Multi-scorer is not supported with panel data in plot_score_time_series. Pass a single scorer instead."
            )
            raise ValueError(msg)
        first_key = next(iter(scorer_dict))
        first_scorer = scorer_dict[first_key]
        first_scorer_cw = scorer_cw_dict[first_key]
        colors = resolve_color_palette(color_palette, n_models)
        return _plot_score_time_series_panel(
            scorer_componentwise=first_scorer_cw,
            y_truth=y_truth,
            y_pred_dict=y_pred_dict,
            groups=groups,
            facet_n_cols=facet_n_cols,
            colors=colors,
            scorer=first_scorer,
            show_legend=show_legend,
            title=title,
            x_label=x_label,
            y_label=y_label,
            width=width,
            height=height,
            connect_gaps=connect_gaps,
            resampler=resampler,
            columns=columns,
            line_width=line_width,
            line_dash=line_dash,
            line_opacity=line_opacity,
            show_markers=show_markers,
        )

    # Case: facet_by="vintage" -> one subplot per vintage_time value
    if facet_by == "vintage":
        if multi_scorer:
            msg = "Multi-scorer is not supported with facet_by='vintage'. Pass a single scorer instead."
            raise ValueError(msg)

        first_key = next(iter(scorer_dict))
        first_scorer = scorer_dict[first_key]
        first_scorer_cw = scorer_cw_dict[first_key]

        # Collect all vintage labels across models
        all_vintages: list = []
        cw_results: dict[str, pl.DataFrame] = {}
        for model_name, y_pred_model in y_pred_dict.items():
            cw_df = _compute_componentwise_scores(first_scorer_cw, y_truth, y_pred_model, columns)
            cw_results[model_name] = cw_df
            if "vintage_time" in cw_df.columns:
                for v in cw_df["vintage_time"].unique().sort().to_list():
                    if v not in all_vintages:
                        all_vintages.append(v)

        if len(all_vintages) < 2:
            msg = (
                "facet_by='vintage' requires predictions with multiple vintages "
                "(multiple vintage_time values). The provided y_pred has a single vintage."
            )
            raise ValueError(msg)

        n_vintages = len(all_vintages)
        n_cols_grid = min(n_vintages, facet_n_cols)
        n_rows = (n_vintages + n_cols_grid - 1) // n_cols_grid
        colors = resolve_color_palette(color_palette, n_models)

        vintage_labels = [str(v) for v in reversed(all_vintages)]
        fig = _create_subplots(
            resampler,
            rows=n_rows,
            cols=n_cols_grid,
            subplot_titles=vintage_labels,
            shared_xaxes=True,
            vertical_spacing=min(0.04, 0.3 / max(n_rows - 1, 1)),
            horizontal_spacing=0.08,
        )

        mode = "lines+markers" if show_markers else "lines"
        legend_tracker = LegendTracker()

        for vintage_idx, vintage_val in enumerate(all_vintages):
            rev_idx = n_vintages - 1 - vintage_idx
            row = rev_idx // n_cols_grid + 1
            col = rev_idx % n_cols_grid + 1

            for model_idx, (model_name, cw_df) in enumerate(cw_results.items()):
                if "vintage_time" not in cw_df.columns:
                    continue
                vintage_df = cw_df.filter(pl.col("vintage_time") == vintage_val)
                if vintage_df.is_empty():
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=vintage_df["time"],
                        y=vintage_df["score"],
                        mode=mode,
                        name=model_name,
                        legendgroup=model_name,
                        showlegend=legend_tracker.should_show(model_name),
                        line={"color": colors[model_idx], "width": line_width, "dash": line_dash},
                        opacity=line_opacity,
                        marker={"size": 6} if show_markers else None,
                        connectgaps=connect_gaps,
                        hovertemplate=_make_hovertemplate(model_name, "Time", "Score", decimals=3),
                    ),
                    row=row,
                    col=col,
                )

        scorer_name = first_scorer.__class__.__name__
        default_title = title or f"{scorer_name} Over Time"
        row_height = 300
        default_height = max(row_height * n_rows, 400)

        fig = apply_default_layout(
            fig,
            title=default_title,
            x_label=x_label or "Time",
            y_label=y_label or scorer_name,
            width=width,
            height=height or default_height,
        )
        fig.update_layout(showlegend=show_legend)
        return fig

    mode = "lines+markers" if show_markers else "lines"

    # Case: multi-scorer AND multi-model -> faceted subplots
    if multi_scorer and n_models > 1:
        _warn_large_grid(n_scorers, n_models)

        if compare_by == "model":
            # Facet by scorer, overlay models
            facet_labels = list(scorer_cw_dict.keys())
            overlay_labels = list(y_pred_dict.keys())
        else:
            # Facet by model, overlay scorers
            facet_labels = list(y_pred_dict.keys())
            overlay_labels = list(scorer_cw_dict.keys())

        n_facets = len(facet_labels)
        n_cols = min(facet_n_cols, n_facets)
        n_rows = (n_facets + n_cols - 1) // n_cols
        colors = resolve_color_palette(color_palette, len(overlay_labels))

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=facet_labels,
            shared_xaxes=True,
            vertical_spacing=_subplot_spacing(n_rows),
        )

        legend_tracker = LegendTracker()

        for facet_idx, facet_label in enumerate(facet_labels):
            row = facet_idx // n_cols + 1
            col = facet_idx % n_cols + 1

            for overlay_idx, overlay_label in enumerate(overlay_labels):
                if compare_by == "model":
                    s_cw = scorer_cw_dict[facet_label]
                    y_pm = y_pred_dict[overlay_label]
                else:
                    s_cw = scorer_cw_dict[overlay_label]
                    y_pm = y_pred_dict[facet_label]

                cw_df = _compute_componentwise_scores(s_cw, y_truth, y_pm, columns)

                _add_vintage_traces(
                    fig,
                    cw_df,
                    x_col="time",
                    y_col="score",
                    name=overlay_label,
                    color=colors[overlay_idx],
                    line_width=line_width,
                    line_dash=line_dash,
                    base_opacity=line_opacity,
                    mode=mode,
                    show_markers=show_markers,
                    connect_gaps=connect_gaps,
                    showlegend=legend_tracker.should_show(overlay_label),
                    row=row,
                    col=col,
                )

        default_title = title or "Score Over Time"
        row_height = 300
        default_height = max(row_height * n_rows, 400)

        fig = apply_default_layout(
            fig,
            title=default_title,
            x_label=x_label or "Time",
            y_label=y_label or "Score",
            width=width,
            height=height or default_height,
        )
        fig.update_layout(showlegend=show_legend)
        return fig

    # Case: single figure (single scorer or single model)
    fig = _create_figure(resampler)

    if multi_scorer:
        # Overlay scorers (single model case)
        colors = resolve_color_palette(color_palette, n_scorers)
        y_pred_single = next(iter(y_pred_dict.values()))

        for idx, s_name in enumerate(scorer_cw_dict):
            cw_df = _compute_componentwise_scores(scorer_cw_dict[s_name], y_truth, y_pred_single, columns)
            _add_vintage_traces(
                fig,
                cw_df,
                x_col="time",
                y_col="score",
                name=s_name,
                color=colors[idx],
                line_width=line_width,
                line_dash=line_dash,
                base_opacity=line_opacity,
                mode=mode,
                show_markers=show_markers,
                connect_gaps=connect_gaps,
                showlegend=True,
            )

        default_title = title or "Score Over Time"
        default_y_label = y_label or "Score"
    else:
        # Overlay models (single scorer case - original behavior)
        colors = resolve_color_palette(color_palette, n_models)
        first_scorer_cw = next(iter(scorer_cw_dict.values()))

        for idx, (model_name, y_pred_model) in enumerate(y_pred_dict.items()):
            cw_df = _compute_componentwise_scores(first_scorer_cw, y_truth, y_pred_model, columns)
            _add_vintage_traces(
                fig,
                cw_df,
                x_col="time",
                y_col="score",
                name=model_name,
                color=colors[idx],
                line_width=line_width,
                line_dash=line_dash,
                base_opacity=line_opacity,
                mode=mode,
                show_markers=show_markers,
                connect_gaps=connect_gaps,
                showlegend=True,
            )

        first_scorer = next(iter(scorer_dict.values()))
        scorer_name = first_scorer.__class__.__name__
        default_title = title or f"{scorer_name} Over Time"
        default_y_label = y_label or scorer_name

    fig = apply_default_layout(
        fig,
        title=default_title,
        x_label=x_label or "Time",
        y_label=default_y_label,
        width=width,
        height=height,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


def plot_score_distribution(
    scorer: BaseScorer | dict[str, BaseScorer],
    y_truth: pl.DataFrame,
    y_pred: pl.DataFrame | dict[str, pl.DataFrame],
    *,
    kind: Literal["histogram", "kde", "both"] = "histogram",
    compare_by: Literal["scorer", "model"] = "scorer",
    n_bins: int = 30,
    show_mean: bool = True,
    show_zero: bool = True,
    columns: str | list[str] | None = None,
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
    bar_opacity: float = 0.6,
    line_width: float = 2.0,
    kde_points: int = 200,
) -> go.Figure:
    """Plot the distribution of per-timestep scorer values.

    Evaluates forecast quality at each timestep (using componentwise
    aggregation) and visualises the resulting score distribution as a
    histogram, KDE, or both.  Supports multi-model comparison via
    overlaid distributions.

    Parameters
    ----------
    scorer : BaseScorer or dict[str, BaseScorer]
        Yohou scorer instance (e.g., ``MeanAbsoluteError``).  Will be
        cloned and configured with ``aggregation_method="componentwise"``.

        - If BaseScorer: single scorer to evaluate.
        - If dict: keys are scorer names, values are scorer instances.
          When combined with dict ``y_pred``, the ``compare_by`` parameter
          controls which dimension is faceted vs overlaid.
    y_truth : pl.DataFrame
        Ground truth values with ``"time"`` column.
    y_pred : pl.DataFrame or dict[str, pl.DataFrame]
        Predicted values with ``"vintage_time"`` and ``"time"`` columns.

        - If DataFrame: single forecast.
        - If dict: keys are model names, values are prediction DataFrames.
    kind : str, default="histogram"
        Distribution visualisation style: ``"histogram"``, ``"kde"`` or
        ``"both"``.
    compare_by : str, default="scorer"
        When both ``scorer`` and ``y_pred`` are dicts, controls which
        dimension is overlaid (colored traces) vs faceted (subplots):

        - ``"scorer"``: overlay scorers, facet by model.
        - ``"model"``: overlay models, facet by scorer.

        Ignored when either ``scorer`` or ``y_pred`` is not a dict.
    n_bins : int, default=30
        Number of histogram bins (ignored for ``kind="kde"``).
    show_mean : bool, default=True
        Add a vertical line at the mean score.
    show_zero : bool, default=True
        Add a vertical dashed line at zero (useful as a perfect-forecast
        reference for symmetric scorers).
    columns : str | list[str] | None, default=None
        Target column name(s) to score.  When *groups* is set
        this acts as a member postfix filter.  ``None`` uses all columns.
    groups : list[str] | None, default=None
        Panel group prefixes to plot (faceted layout).
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data. ``"group"`` creates one subplot per
        group, ``"member"`` one per member. ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in the faceted grid.
    color_palette : list[str] | None, default=None
        Custom colour palette.
    show_legend : bool, default=True
        Whether to show the legend.
    title : str | None, default=None
        Plot title.  Defaults to ``"<ScorerName> Distribution"``.
    x_label : str | None, default=None
        X-axis label.  Defaults to the scorer class name.
    y_label : str | None, default=None
        Y-axis label.  Defaults to ``"Count"`` or ``"Density"``
        depending on *kind*.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    bar_opacity : float, default=0.6
        Opacity of histogram bars.
    line_width : float, default=2.0
        Width of KDE lines.
    kde_points : int, default=200
        Number of points for KDE evaluation.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    TypeError
        If *y_truth* or *y_pred* is not a Polars DataFrame.
    ValueError
        If *kind* is not one of ``"histogram"``, ``"kde"`` or ``"both"``.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MeanAbsoluteError
    >>> from yohou.plotting import plot_score_distribution

    >>> y_truth = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "value": [10.0, 20.0, 30.0, 40.0, 50.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "vintage_time": [datetime(2019, 12, 31)] * 5,
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "value": [12.0, 19.0, 28.0, 42.0, 48.0],
    ... })

    >>> fig = plot_score_distribution(MeanAbsoluteError(), y_truth, y_pred)
    >>> len(fig.data) >= 1
    True

    See Also
    --------
    [`plot_score_time_series`][yohou.plotting.plot_score_time_series] : Score values over time.
    [`plot_score_per_step`][yohou.plotting.plot_score_per_step] : Score by forecast step.
    """
    from scipy.stats import gaussian_kde  # noqa: PLC0415

    validate_plotting_data(y_truth)

    validate_plotting_params(kind=kind, valid_kinds={"histogram", "kde", "both"}, width=width, height=height)

    y_pred_dict: dict[str, pl.DataFrame] = _normalize_y_pred(y_pred)
    scorer_dict = _normalize_scorers(scorer)

    # Prepare and fit each scorer for componentwise aggregation
    scorer_cw_dict: dict[str, BaseScorer] = {}
    for s_name, s in scorer_dict.items():
        s_cw = _prepare_scorer_for_componentwise(s)
        s_cw.fit(y_truth)
        scorer_cw_dict[s_name] = s_cw

    n_scorers = len(scorer_cw_dict)
    n_models = len(y_pred_dict)
    multi_scorer = n_scorers > 1

    def _render(
        fig: go.Figure,
        y_truth_sub: pl.DataFrame,
        y_pred_dict_sub: dict[str, pl.DataFrame],
        _colors: list[str],
        scorer_cw_r: BaseScorer,
        _show_legend: bool = True,
        *,
        row: int | None = None,
        col: int | None = None,
    ) -> None:
        """Render score distribution traces onto *fig*."""
        for idx, (mname, y_pred_m) in enumerate(y_pred_dict_sub.items()):
            validate_plotting_data(y_pred_m)
            scores_df = scorer_cw_r.score(y_truth_sub, y_pred_m)
            if not isinstance(scores_df, pl.DataFrame):
                msg_ = f"Scorer must return DataFrame for componentwise aggregation, got {type(scores_df).__name__}"
                raise TypeError(msg_)

            score_cols = [c for c in scores_df.columns if c not in _SCORER_META_COLS]
            if len(score_cols) == 1:
                score_vals = scores_df[score_cols[0]].drop_nulls().to_numpy()
            else:
                score_vals = scores_df.select(score_cols).to_numpy().flatten()
                score_vals = score_vals[~np.isnan(score_vals)]

            c = _colors[idx % len(_colors)]

            if kind in ("histogram", "both"):
                hist_norm = "probability density" if kind == "both" else ""
                fig.add_trace(
                    go.Histogram(
                        x=score_vals,
                        nbinsx=n_bins,
                        marker_color=c,
                        opacity=bar_opacity,
                        name=mname,
                        legendgroup=mname,
                        showlegend=_show_legend if kind != "both" else False,
                        histnorm=hist_norm,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

            if kind in ("kde", "both") and len(score_vals) > 1:
                try:
                    kde = gaussian_kde(score_vals)
                except np.linalg.LinAlgError:
                    warnings.warn(
                        f"Could not compute KDE for model {mname!r} "
                        "(scores have zero variance); skipping its density trace.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    x_grid = np.linspace(
                        float(score_vals.min()),
                        float(score_vals.max()),
                        kde_points,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_grid,
                            y=kde(x_grid),
                            mode="lines",
                            line={"color": c, "width": line_width},
                            name=mname,
                            legendgroup=mname,
                            showlegend=_show_legend,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )

            if show_mean and len(score_vals) > 0:
                mean_val = float(np.mean(score_vals))
                fig.add_vline(
                    x=mean_val,
                    line_dash="dash",
                    line_color=c,
                    line_width=1.5,
                    row=row,
                    col=col,
                )
                if row is None:
                    fig.add_annotation(
                        x=mean_val,
                        y=1.0,
                        yref="paper",
                        text=f"\u03bc={mean_val:.3f}",
                        font={"color": c, "size": 11},
                        showarrow=False,
                        yanchor="bottom",
                    )

        if show_zero:
            fig.add_vline(
                x=0.0,
                line_dash="dot",
                line_color="grey",
            )

    # Column filter
    _col_filter: set[str] | None = None
    if columns is not None:
        _col_filter = set([columns] if isinstance(columns, str) else columns)

    # Panel dispatch
    _, _panel_groups = inspect_panel(y_truth)
    _effective_groups: list[str] | None = None
    if groups is not None:
        _effective_groups = groups
    elif _panel_groups:
        _effective_groups = list(_panel_groups)
    if _effective_groups:
        if multi_scorer:
            msg = (
                "Multi-scorer is not supported with panel data in "
                "plot_score_distribution. Pass a single scorer instead."
            )
            raise ValueError(msg)

        first_cw = next(iter(scorer_cw_dict.values()))
        colors = resolve_color_palette(color_palette, n_models)

        n_cols_grid = min(len(_effective_groups), facet_n_cols)
        n_rows_grid = (len(_effective_groups) + n_cols_grid - 1) // n_cols_grid
        pfig = make_subplots(
            rows=n_rows_grid,
            cols=n_cols_grid,
            subplot_titles=_effective_groups,
            vertical_spacing=max(0.04, 0.3 / n_rows_grid),
        )
        for g_idx, gname in enumerate(_effective_groups):
            r = g_idx // n_cols_grid + 1
            c_i = g_idx % n_cols_grid + 1
            g_cols_truth = [
                cn
                for cn in y_truth.columns
                if cn == "time"
                or (cn.startswith(f"{gname}__") and (_col_filter is None or _member_name(cn) in _col_filter))
            ]
            y_truth_g = y_truth.select(g_cols_truth) if len(g_cols_truth) > 1 else y_truth
            y_pred_dict_g: dict[str, pl.DataFrame] = {}
            for mname, y_pred_m in y_pred_dict.items():
                gp_cols = [
                    cn
                    for cn in y_pred_m.columns
                    if cn in ("time", "vintage_time")
                    or (cn.startswith(f"{gname}__") and (_col_filter is None or _member_name(cn) in _col_filter))
                ]
                has_group_col = any(cn not in ("time", "vintage_time") for cn in gp_cols)
                y_pred_dict_g[mname] = y_pred_m.select(gp_cols) if has_group_col else y_pred_m
            _render(pfig, y_truth_g, y_pred_dict_g, colors, first_cw, show_legend and g_idx == 0, row=r, col=c_i)

        first_scorer = next(iter(scorer_dict.values()))
        scorer_name = first_scorer.__class__.__name__
        pfig = apply_default_layout(
            pfig,
            title=title or f"{scorer_name} Distribution",
            x_label=x_label or scorer_name,
            y_label=y_label or ("Density" if kind in ("kde", "both") else "Count"),
            width=width,
            height=height,
        )
        pfig.update_layout(barmode="overlay" if n_models > 1 else "relative", showlegend=show_legend)
        return pfig

    # Multi-scorer + multi-model -> faceted subplots
    if multi_scorer and n_models > 1:
        _warn_large_grid(n_scorers, n_models)

        if compare_by == "model":
            facet_labels = list(scorer_cw_dict.keys())
            overlay_labels = list(y_pred_dict.keys())
        else:
            facet_labels = list(y_pred_dict.keys())
            overlay_labels = list(scorer_cw_dict.keys())

        n_facets = len(facet_labels)
        n_cols_f = min(facet_n_cols, n_facets)
        n_rows_f = (n_facets + n_cols_f - 1) // n_cols_f
        colors = resolve_color_palette(color_palette, len(overlay_labels))

        pfig = make_subplots(
            rows=n_rows_f,
            cols=n_cols_f,
            subplot_titles=facet_labels,
            vertical_spacing=_subplot_spacing(n_rows_f),
        )

        legend_tracker = LegendTracker()
        for facet_idx, facet_label in enumerate(facet_labels):
            r = facet_idx // n_cols_f + 1
            c_i = facet_idx % n_cols_f + 1
            for overlay_idx, overlay_label in enumerate(overlay_labels):
                if compare_by == "model":
                    s_cw = scorer_cw_dict[facet_label]
                    ypd_sub = {overlay_label: y_pred_dict[overlay_label]}
                else:
                    s_cw = scorer_cw_dict[overlay_label]
                    ypd_sub = {overlay_label: y_pred_dict[facet_label]}
                _render(
                    pfig,
                    y_truth,
                    ypd_sub,
                    [colors[overlay_idx]],
                    s_cw,
                    legend_tracker.should_show(overlay_label),
                    row=r,
                    col=c_i,
                )

        pfig = apply_default_layout(
            pfig,
            title=title or "Score Distribution",
            x_label=x_label or "Score",
            y_label=y_label or ("Density" if kind in ("kde", "both") else "Count"),
            width=width,
            height=height or max(300 * n_rows_f, 400),
        )
        pfig.update_layout(barmode="overlay", showlegend=show_legend)
        return pfig

    # Non-panel single figure
    fig = go.Figure()

    if multi_scorer:
        # Overlay scorers (single model)
        colors = resolve_color_palette(color_palette, n_scorers)
        y_pred_single = next(iter(y_pred_dict.values()))
        for idx, (s_name, s_cw) in enumerate(scorer_cw_dict.items()):
            _render(fig, y_truth, {s_name: y_pred_single}, [colors[idx]], s_cw, _show_legend=show_legend)
    else:
        # Overlay models (single scorer - original behavior)
        first_cw = next(iter(scorer_cw_dict.values()))
        colors = resolve_color_palette(color_palette, n_models)
        if _col_filter is not None:
            _keep_truth = ["time"] + [c for c in y_truth.columns if c != "time" and c in _col_filter]
            y_truth_filt = y_truth.select(_keep_truth)
            y_pred_dict_filt = {
                k: v.select([c for c in v.columns if c in ("time", "vintage_time") or c in _col_filter])
                for k, v in y_pred_dict.items()
            }
            _render(fig, y_truth_filt, y_pred_dict_filt, colors, first_cw, _show_legend=show_legend)
        else:
            _render(fig, y_truth, y_pred_dict, colors, first_cw, _show_legend=show_legend)

    if multi_scorer:
        default_title = title or "Score Distribution"
        default_x = x_label or "Score"
    else:
        first_scorer = next(iter(scorer_dict.values()))
        scorer_name = first_scorer.__class__.__name__
        default_title = title or f"{scorer_name} Distribution"
        default_x = x_label or scorer_name
    default_y = y_label or ("Density" if kind in ("kde", "both") else "Count")

    fig = apply_default_layout(
        fig,
        title=default_title,
        x_label=default_x,
        y_label=default_y,
        width=width,
        height=height,
    )

    if n_models > 1 or multi_scorer:
        fig.update_layout(barmode="overlay", showlegend=show_legend)
    else:
        fig.update_layout(showlegend=show_legend)

    return fig


def plot_score_summary(
    scorer: BaseScorer | dict[str, BaseScorer],
    y_truth: pl.DataFrame,
    y_pred: pl.DataFrame | dict[str, pl.DataFrame],
    *,
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    bar_opacity: float = 0.85,
    sort_ascending: bool | None = None,
    text_auto: bool = True,
) -> go.Figure:
    """Plot a grouped bar chart comparing aggregate scores across models and scorers.

    For each combination of scorer and model, compute a single aggregate
    score and display the results as a grouped bar chart. This is useful
    for quick model comparison without the per-step detail.

    Parameters
    ----------
    scorer : BaseScorer or dict[str, BaseScorer]
        Yohou scorer instance.

        - If BaseScorer: single scorer to evaluate.
        - If dict: keys are scorer names, values are scorer instances.
    y_truth : pl.DataFrame
        Ground truth with ``"time"`` column.
    y_pred : pl.DataFrame or dict[str, pl.DataFrame]
        Predictions with ``"vintage_time"`` and ``"time"`` columns.

        - If DataFrame: single forecast.
        - If dict: keys are model names, values are prediction DataFrames.
    color_palette : list[str] | None, default=None
        Custom colour palette.
    show_legend : bool, default=True
        Whether to show the legend.
    title : str | None, default=None
        Plot title. Defaults to ``"Model Comparison"``.
    x_label : str | None, default=None
        X-axis label. Defaults to ``""``.
    y_label : str | None, default=None
        Y-axis label. Defaults to ``"Score"``.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    bar_opacity : float, default=0.85
        Opacity of bars.
    sort_ascending : bool or None, default=None
        Sort bars by score value. ``True`` for ascending, ``False`` for
        descending, ``None`` to keep insertion order.
    text_auto : bool, default=True
        Annotate bars with their values.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    TypeError
        If *y_truth* or *y_pred* is not a Polars DataFrame.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MeanAbsoluteError
    >>> from yohou.plotting import plot_score_summary

    >>> y_truth = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "value": [10.0, 20.0, 30.0, 40.0, 50.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "vintage_time": [datetime(2019, 12, 31)] * 5,
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "value": [12.0, 19.0, 28.0, 42.0, 48.0],
    ... })

    >>> fig = plot_score_summary(MeanAbsoluteError(), y_truth, y_pred)
    >>> len(fig.data) >= 1
    True

    See Also
    --------
    [`plot_score_per_step`][yohou.plotting.plot_score_per_step] : Per-step score line/bar chart.
    [`plot_score_time_series`][yohou.plotting.plot_score_time_series] : Score values over time.
    """
    validate_plotting_data(y_truth)
    validate_plotting_params(width=width, height=height)

    y_pred_dict: dict[str, pl.DataFrame] = _normalize_y_pred(y_pred)
    scorer_dict = _normalize_scorers(scorer)

    scorer_names = list(scorer_dict.keys())
    model_names = list(y_pred_dict.keys())

    results: dict[str, dict[str, float]] = {}
    for m_name, y_pred_m in y_pred_dict.items():
        validate_plotting_data(y_pred_m)
        model_scores: dict[str, float] = {}
        for s_name, s_orig in scorer_dict.items():
            s_agg = copy.deepcopy(s_orig)
            s_agg.fit(y_truth)
            score_val = s_agg.score(y_truth, y_pred_m)
            if isinstance(score_val, pl.DataFrame):
                score_cols = [c for c in score_val.columns if c not in _SCORER_META_COLS]
                score_val = float(score_val.select(score_cols).mean_horizontal().mean())  # type: ignore
            model_scores[s_name] = float(score_val)  # type: ignore
        results[m_name] = model_scores

    categories = scorer_names
    series_names = model_names
    series_values = [[results[m].get(s, 0.0) for s in categories] for m in series_names]

    if sort_ascending is not None and series_values:
        order = sorted(
            range(len(categories)),
            key=lambda k: series_values[0][k],
            reverse=not sort_ascending,
        )
        categories = [categories[k] for k in order]
        series_values = [[sv[k] for k in order] for sv in series_values]

    colors = resolve_color_palette(color_palette, len(series_names))
    fig = go.Figure()

    for i, (name, values) in enumerate(zip(series_names, series_values, strict=True)):
        bar_kwargs: dict = {
            "name": name,
            "marker_color": colors[i % len(colors)],
            "opacity": bar_opacity,
            "x": categories,
            "y": values,
        }
        if text_auto:
            bar_kwargs["text"] = [f"{v:.3g}" for v in values]
            bar_kwargs["textposition"] = "outside"
        fig.add_trace(go.Bar(**bar_kwargs))

    fig.update_layout(barmode="group")
    fig = apply_default_layout(
        fig,
        title=title or "Model Comparison",
        x_label=x_label or "",
        y_label=y_label or "Score",
        width=width,
        height=height,
    )
    fig.update_layout(showlegend=show_legend)
    return fig


def plot_score_per_step(
    scorer: BaseScorer | dict[str, BaseScorer],
    y_truth: pl.DataFrame,
    y_pred: pl.DataFrame | dict[str, pl.DataFrame],
    *,
    kind: Literal["line", "bar"] = "line",
    compare_by: Literal["scorer", "model"] = "scorer",
    show_trend: bool = False,
    columns: str | list[str] | None = None,
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
    line_width: float = 2.0,
    marker_size: float = 8.0,
    marker_opacity: float = 0.8,
    bar_opacity: float = 0.85,
) -> go.Figure:
    """Plot scorer value by forecast horizon step.

    For each step *h* in the forecast window, compute the scorer between
    ``y_truth`` and ``y_pred`` at row *h* and plot the result. This
    reveals how forecast accuracy degrades as the horizon increases.

    Parameters
    ----------
    scorer : BaseScorer or dict[str, BaseScorer]
        Yohou scorer instance.  Will be cloned with
        ``aggregation_method="componentwise"``.

        - If BaseScorer: single scorer to evaluate.
        - If dict: keys are scorer names, values are scorer instances.
          When combined with dict ``y_pred``, the ``compare_by``
          parameter controls which dimension is faceted vs overlaid.
    y_truth : pl.DataFrame
        Ground truth with ``"time"`` column.
    y_pred : pl.DataFrame or dict[str, pl.DataFrame]
        Predictions with ``"vintage_time"`` and ``"time"`` columns.

        - If DataFrame: single forecast.
        - If dict: keys are model names, values are prediction DataFrames.
    kind : str, default="line"
        Plot kind: ``"line"`` or ``"bar"``.

        - ``"line"``: per-step line chart with markers.
        - ``"bar"``: per-step bar chart.
    compare_by : str, default="scorer"
        When both ``scorer`` and ``y_pred`` are dicts, controls which
        dimension is overlaid (colored traces) vs faceted (subplots):

        - ``"scorer"``: overlay scorers, facet by model.
        - ``"model"``: overlay models, facet by scorer.

        Ignored when either ``scorer`` or ``y_pred`` is not a dict.
    show_trend : bool, default=False
        Overlay a linear trend line (``np.polyfit`` degree 1).
    columns : str | list[str] | None, default=None
        Target column name(s) to score.  When *groups* is set
        this acts as a member postfix filter.  ``None`` uses all columns.
    groups : list[str] | None, default=None
        Panel group prefixes to plot (faceted layout).
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data. ``"group"`` creates one subplot per
        group, ``"member"`` one per member. ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Columns in the faceted grid.
    color_palette : list[str] | None, default=None
        Custom colour palette.
    show_legend : bool, default=True
        Whether to show the legend.
    title : str | None, default=None
        Plot title. Defaults to ``"<ScorerName> by Horizon Step"``.
    x_label : str | None, default=None
        X-axis label. Defaults to ``"Horizon Step"``.
    y_label : str | None, default=None
        Y-axis label. Defaults to the scorer class name.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    line_width : float, default=2.0
        Width of score lines.
    marker_size : float, default=8.0
        Marker size for line+marker traces.
    marker_opacity : float, default=0.8
        Opacity of scatter markers.
    bar_opacity : float, default=0.85
        Opacity of bars when ``kind="bar"``.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    TypeError
        If *y_truth* or *y_pred* is not a Polars DataFrame.
    ValueError
        If *kind* is not ``"line"`` or ``"bar"``.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MeanAbsoluteError
    >>> from yohou.plotting import plot_score_per_step

    >>> y_truth = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "value": [10.0, 20.0, 30.0, 40.0, 50.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "vintage_time": [datetime(2019, 12, 31)] * 5,
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "value": [12.0, 19.0, 28.0, 42.0, 48.0],
    ... })

    >>> fig = plot_score_per_step(MeanAbsoluteError(), y_truth, y_pred)
    >>> len(fig.data) >= 1
    True

    See Also
    --------
    [`plot_score_summary`][yohou.plotting.plot_score_summary] : Grouped bar chart of aggregate scores.
    [`plot_score_time_series`][yohou.plotting.plot_score_time_series] : Score values over time.
    [`plot_score_distribution`][yohou.plotting.plot_score_distribution] : Score distribution histogram/KDE.
    """
    validate_plotting_data(y_truth)

    validate_plotting_params(kind=kind, valid_kinds={"line", "bar"}, width=width, height=height)

    y_pred_dict: dict[str, pl.DataFrame] = _normalize_y_pred(y_pred)
    scorer_dict = _normalize_scorers(scorer)

    # Prepare and fit each scorer for componentwise aggregation
    scorer_cw_dict: dict[str, BaseScorer] = {}
    for s_name, s in scorer_dict.items():
        s_cw = _prepare_scorer_for_componentwise(s)
        s_cw.fit(y_truth)
        scorer_cw_dict[s_name] = s_cw

    n_scorers = len(scorer_cw_dict)
    n_models = len(y_pred_dict)
    multi_scorer = n_scorers > 1

    def _render_horizon(
        fig: go.Figure,
        y_truth_sub: pl.DataFrame,
        y_pred_dict_sub: dict[str, pl.DataFrame],
        _colors: list[str],
        scorer_cw_r: BaseScorer,
        _show_legend: bool = True,
        *,
        row: int | None = None,
        col: int | None = None,
        legendgroup: str | None = None,
    ) -> None:
        """Render per-horizon score traces onto *fig*."""
        for idx, (mname, y_pred_m) in enumerate(y_pred_dict_sub.items()):
            validate_plotting_data(y_pred_m)
            scores_df = scorer_cw_r.score(y_truth_sub, y_pred_m)
            if not isinstance(scores_df, pl.DataFrame):
                msg_ = f"Scorer must return DataFrame for componentwise aggregation, got {type(scores_df).__name__}"
                raise TypeError(msg_)

            score_cols = [c for c in scores_df.columns if c not in _SCORER_META_COLS]
            if len(score_cols) == 1:
                score_vals = scores_df[score_cols[0]].drop_nulls().to_numpy()
            else:
                # average across components at each timestep
                score_vals = scores_df.select(score_cols).mean_horizontal().to_numpy()
                score_vals = score_vals[~np.isnan(score_vals)]

            n_steps = len(score_vals)
            steps = np.arange(1, n_steps + 1)
            c = _colors[idx % len(_colors)]

            lg_kwargs: dict = {}
            if legendgroup is not None:
                lg_kwargs["legendgroup"] = legendgroup

            if kind == "line":
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=score_vals,
                        mode="lines+markers",
                        name=mname,
                        showlegend=_show_legend,
                        line={"color": c, "width": line_width},
                        marker={"size": marker_size, "color": c, "opacity": marker_opacity},
                        **lg_kwargs,
                    ),
                    row=row,
                    col=col,
                )
            else:
                fig.add_trace(
                    go.Bar(
                        x=steps,
                        y=score_vals,
                        name=mname,
                        showlegend=_show_legend,
                        marker_color=c,
                        opacity=bar_opacity,
                        **lg_kwargs,
                    ),
                    row=row,
                    col=col,
                )

            if show_trend and n_steps >= 2:
                coeffs = np.polyfit(steps, score_vals, 1)
                trend_y = np.polyval(coeffs, steps)
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=trend_y,
                        mode="lines",
                        name=f"{mname} trend",
                        line={"color": c, "width": 1.5, "dash": "dash"},
                        showlegend=_show_legend,
                        **lg_kwargs,
                    ),
                    row=row,
                    col=col,
                )

    _col_filter: set[str] | None = None
    if columns is not None:
        _col_filter = set([columns] if isinstance(columns, str) else columns)

    # Panel dispatch
    _, _panel_groups = inspect_panel(y_truth)
    _effective_groups: list[str] | None = None
    if groups is not None:
        _effective_groups = groups
    elif _panel_groups:
        _effective_groups = list(_panel_groups)
    if _effective_groups:
        if multi_scorer:
            msg = "Multi-scorer is not supported with panel data in plot_score_per_step. Pass a single scorer instead."
            raise ValueError(msg)

        first_cw = next(iter(scorer_cw_dict.values()))
        colors = resolve_color_palette(color_palette, n_models)

        n_cols_grid = min(len(_effective_groups), facet_n_cols)
        n_rows_grid = (len(_effective_groups) + n_cols_grid - 1) // n_cols_grid
        pfig = make_subplots(
            rows=n_rows_grid,
            cols=n_cols_grid,
            subplot_titles=_effective_groups,
            vertical_spacing=max(0.04, 0.3 / n_rows_grid),
        )
        for g_idx, gname in enumerate(_effective_groups):
            r = g_idx // n_cols_grid + 1
            c_i = g_idx % n_cols_grid + 1
            g_cols_truth = [
                cn
                for cn in y_truth.columns
                if cn == "time"
                or (cn.startswith(f"{gname}__") and (_col_filter is None or _member_name(cn) in _col_filter))
            ]
            y_truth_g = y_truth.select(g_cols_truth) if len(g_cols_truth) > 1 else y_truth
            y_pred_dict_g: dict[str, pl.DataFrame] = {}
            for mname, y_pred_m in y_pred_dict.items():
                gp_cols = [
                    cn
                    for cn in y_pred_m.columns
                    if cn in ("time", "vintage_time")
                    or (cn.startswith(f"{gname}__") and (_col_filter is None or _member_name(cn) in _col_filter))
                ]
                has_group_col = any(cn not in ("time", "vintage_time") for cn in gp_cols)
                y_pred_dict_g[mname] = y_pred_m.select(gp_cols) if has_group_col else y_pred_m
            _render_horizon(
                pfig,
                y_truth_g,
                y_pred_dict_g,
                colors,
                first_cw,
                show_legend and g_idx == 0,
                row=r,
                col=c_i,
            )

        first_scorer = next(iter(scorer_dict.values()))
        scorer_name = first_scorer.__class__.__name__
        pfig = apply_default_layout(
            pfig,
            title=title or f"{scorer_name} by Horizon Step",
            x_label=x_label or "Horizon Step",
            y_label=y_label or scorer_name,
            width=width,
            height=height,
        )
        if kind == "bar" and n_models > 1:
            pfig.update_layout(barmode="group")
        pfig.update_layout(showlegend=show_legend)
        return pfig

    # Multi-scorer + multi-model -> faceted subplots
    if multi_scorer and n_models > 1:
        _warn_large_grid(n_scorers, n_models)

        if compare_by == "model":
            facet_labels = list(scorer_cw_dict.keys())
            overlay_labels = list(y_pred_dict.keys())
        else:
            facet_labels = list(y_pred_dict.keys())
            overlay_labels = list(scorer_cw_dict.keys())

        n_facets = len(facet_labels)
        n_cols_f = min(facet_n_cols, n_facets)
        n_rows_f = (n_facets + n_cols_f - 1) // n_cols_f
        colors = resolve_color_palette(color_palette, len(overlay_labels))

        pfig = make_subplots(
            rows=n_rows_f,
            cols=n_cols_f,
            subplot_titles=facet_labels,
            vertical_spacing=_subplot_spacing(n_rows_f),
        )

        legend_tracker = LegendTracker()
        for facet_idx, facet_label in enumerate(facet_labels):
            r = facet_idx // n_cols_f + 1
            c_i = facet_idx % n_cols_f + 1
            for overlay_idx, overlay_label in enumerate(overlay_labels):
                if compare_by == "model":
                    s_cw = scorer_cw_dict[facet_label]
                    ypd_sub = {overlay_label: y_pred_dict[overlay_label]}
                else:
                    s_cw = scorer_cw_dict[overlay_label]
                    ypd_sub = {overlay_label: y_pred_dict[facet_label]}
                _render_horizon(
                    pfig,
                    y_truth,
                    ypd_sub,
                    [colors[overlay_idx]],
                    s_cw,
                    legend_tracker.should_show(overlay_label),
                    row=r,
                    col=c_i,
                )

        pfig = apply_default_layout(
            pfig,
            title=title or "Score by Horizon Step",
            x_label=x_label or "Horizon Step",
            y_label=y_label or "Score",
            width=width,
            height=height or max(300 * n_rows_f, 400),
        )
        if kind == "bar":
            pfig.update_layout(barmode="group")
        pfig.update_layout(showlegend=show_legend)
        return pfig

    # Non-panel single figure

    # Determine effective truth/pred for column filtering
    if _col_filter is not None:
        _keep_truth = ["time"] + [c for c in y_truth.columns if c != "time" and c in _col_filter]
        _yt_eff = y_truth.select(_keep_truth)
        _ypd_eff = {
            k: v.select([c for c in v.columns if c in ("time", "vintage_time") or c in _col_filter])
            for k, v in y_pred_dict.items()
        }
    else:
        _yt_eff = y_truth
        _ypd_eff = y_pred_dict

    # Detect number of score components
    _score_cols = [c for c in _yt_eff.columns if c != "time"]

    if multi_scorer:
        # Overlay scorers (single model)
        colors = resolve_color_palette(color_palette, n_scorers)
        y_pred_single = next(iter(_ypd_eff.values()))
        fig = go.Figure()
        for idx, (s_name, s_cw) in enumerate(scorer_cw_dict.items()):
            _render_horizon(
                fig,
                _yt_eff,
                {s_name: y_pred_single},
                [colors[idx]],
                s_cw,
                _show_legend=show_legend,
            )
    elif len(_score_cols) > 1:
        # Multi-component: create subplots per component, group legend per model
        first_cw = next(iter(scorer_cw_dict.values()))
        colors = resolve_color_palette(color_palette, n_models)
        n_comps = len(_score_cols)
        n_cols_c = min(facet_n_cols, n_comps)
        n_rows_c = (n_comps + n_cols_c - 1) // n_cols_c

        fig = make_subplots(
            rows=n_rows_c,
            cols=n_cols_c,
            subplot_titles=_score_cols,
            vertical_spacing=_subplot_spacing(n_rows_c),
        )

        legend_tracker = LegendTracker()
        for comp_idx, comp_col in enumerate(_score_cols):
            r = comp_idx // n_cols_c + 1
            c_i = comp_idx % n_cols_c + 1
            yt_comp = _yt_eff.select(["time", comp_col])
            ypd_comp = {
                k: v.select([c for c in v.columns if c in ("time", "vintage_time", comp_col)])
                for k, v in _ypd_eff.items()
            }
            for m_idx, (mname, y_pred_m) in enumerate(ypd_comp.items()):
                _render_horizon(
                    fig,
                    yt_comp,
                    {mname: y_pred_m},
                    [colors[m_idx]],
                    first_cw,
                    legend_tracker.should_show(mname),
                    row=r,
                    col=c_i,
                    legendgroup=mname,
                )
    else:
        # Single component, overlay models (original behavior)
        first_cw = next(iter(scorer_cw_dict.values()))
        colors = resolve_color_palette(color_palette, n_models)
        fig = go.Figure()
        _render_horizon(fig, _yt_eff, _ypd_eff, colors, first_cw, show_legend)

    if multi_scorer:
        default_title = title or "Score by Horizon Step"
        default_y = y_label or "Score"
        default_height = height
    elif len(_score_cols) > 1:
        first_scorer = next(iter(scorer_dict.values()))
        scorer_name = first_scorer.__class__.__name__
        default_title = title or f"{scorer_name} by Horizon Step"
        default_y = y_label or scorer_name
        n_rows_c = (len(_score_cols) + min(facet_n_cols, len(_score_cols)) - 1) // min(facet_n_cols, len(_score_cols))
        default_height = height or max(300 * n_rows_c, 400)
    else:
        first_scorer = next(iter(scorer_dict.values()))
        scorer_name = first_scorer.__class__.__name__
        default_title = title or f"{scorer_name} by Horizon Step"
        default_y = y_label or scorer_name
        default_height = height

    fig = apply_default_layout(
        fig,
        title=default_title,
        x_label=x_label or "Horizon Step",
        y_label=default_y,
        width=width,
        height=default_height,
    )

    if kind == "bar" and (n_models > 1 or multi_scorer):
        fig.update_layout(barmode="group")

    fig.update_layout(showlegend=show_legend)

    return fig


def plot_score_per_vintage(
    scorer: BaseScorer | dict[str, BaseScorer],
    y_truth: pl.DataFrame,
    y_pred: pl.DataFrame | dict[str, pl.DataFrame],
    *,
    kind: Literal["line", "bar"] = "line",
    compare_by: Literal["scorer", "model"] = "scorer",
    show_trend: bool = False,
    facet_n_cols: int = 2,
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    line_width: float = 2.0,
    marker_size: float = 8.0,
    marker_opacity: float = 0.8,
    bar_opacity: float = 0.85,
) -> go.Figure:
    """Plot scorer value by forecast vintage (observed time).

    For each vintage origin in ``y_pred``, compute the scorer across
    the forecast steps that originated from that vintage and plot the
    result. This reveals how forecast quality changes depending on
    when the forecast was made.

    Parameters
    ----------
    scorer : BaseScorer or dict[str, BaseScorer]
        Yohou scorer instance.  Will be cloned with
        ``aggregation_method="stepwise"`` to collapse the horizon
        dimension.

        - If BaseScorer: single scorer to evaluate.
        - If dict: keys are scorer names, values are scorer instances.
    y_truth : pl.DataFrame
        Ground truth with ``"time"`` column.
    y_pred : pl.DataFrame or dict[str, pl.DataFrame]
        Predictions with ``"vintage_time"`` and ``"time"`` columns.

        - If DataFrame: single forecast.
        - If dict: keys are model names, values are prediction DataFrames.
    kind : str, default="line"
        Plot kind: ``"line"`` or ``"bar"``.
    compare_by : str, default="scorer"
        When both ``scorer`` and ``y_pred`` are dicts, controls which
        dimension is overlaid vs faceted.
    show_trend : bool, default=False
        Overlay a linear trend line.
    facet_n_cols : int, default=2
        Columns in the faceted grid.
    color_palette : list[str] | None, default=None
        Custom colour palette.
    show_legend : bool, default=True
        Whether to show the legend.
    title : str | None, default=None
        Plot title.
    x_label : str | None, default=None
        X-axis label. Defaults to ``"Vintage (Observed Time)"``.
    y_label : str | None, default=None
        Y-axis label.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    line_width : float, default=2.0
        Width of score lines.
    marker_size : float, default=8.0
        Marker size for line+marker traces.
    marker_opacity : float, default=0.8
        Opacity of scatter markers.
    bar_opacity : float, default=0.85
        Opacity of bars when ``kind="bar"``.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    ValueError
        If ``y_pred`` has only a single vintage (single vintage_time).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MeanAbsoluteError
    >>> from yohou.plotting import plot_score_per_vintage

    >>> y_truth = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)],
    ...     "value": [10.0, 20.0, 30.0, 40.0, 50.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "vintage_time": ([datetime(2019, 12, 30)] * 5 + [datetime(2019, 12, 31)] * 5),
    ...     "time": [datetime(2020, 1, i) for i in range(1, 6)] * 2,
    ...     "value": [12.0, 19.0, 28.0, 42.0, 48.0, 11.0, 21.0, 29.0, 41.0, 49.0],
    ... }).sort("time")

    >>> fig = plot_score_per_vintage(MeanAbsoluteError(), y_truth, y_pred)
    >>> len(fig.data) >= 1
    True

    See Also
    --------
    [`plot_score_time_series`][yohou.plotting.plot_score_time_series] : Per-timestep score with ``facet_by="vintage"`` for detailed vintage comparison.
    [`plot_score_per_step`][yohou.plotting.plot_score_per_step] : Score by horizon step.
    [`plot_score_heatmap`][yohou.plotting.plot_score_heatmap] : 2D heatmap of scores.
    """
    validate_plotting_data(y_truth)
    validate_plotting_params(kind=kind, valid_kinds={"line", "bar"}, width=width, height=height)

    y_pred_dict = _normalize_y_pred(y_pred)
    scorer_dict = _normalize_scorers(scorer)

    n_scorers = len(scorer_dict)
    n_models = len(y_pred_dict)
    multi_scorer = n_scorers > 1

    def _compute_vintage_scores(
        s: BaseScorer,
        y_t: pl.DataFrame,
        y_p: pl.DataFrame,
    ) -> tuple[pl.Series, pl.Series]:
        """Compute per-vintage aggregate scores."""
        s_cw = copy.deepcopy(s)
        if isinstance(s_cw, BaseIntervalScorer):
            s_cw.set_params(aggregation_method=["stepwise", "coveragewise"])
        else:
            s_cw.set_params(aggregation_method="stepwise")
        s_cw.fit(y_t)
        scores_df = s_cw.score(y_t, y_p)
        if not isinstance(scores_df, pl.DataFrame):
            msg = f"Scorer must return DataFrame for stepwise aggregation, got {type(scores_df).__name__}"
            raise TypeError(msg)
        if "vintage_time" not in scores_df.columns:
            msg = (
                "Scorer did not return 'vintage_time' column. "
                "Ensure y_pred has multiple vintages (vintage_time values)."
            )
            raise ValueError(msg)
        score_cols = [c for c in scores_df.columns if c not in _SCORER_META_COLS]
        if len(score_cols) == 1:
            score_values = scores_df[score_cols[0]]
        else:
            score_values = scores_df.select(score_cols).mean_horizontal()
        return scores_df["vintage_time"], score_values

    # Validate that data has multiple vintages
    first_y_pred = next(iter(y_pred_dict.values()))
    if "vintage_time" in first_y_pred.columns:
        n_vintages = first_y_pred["vintage_time"].n_unique()
        if n_vintages <= 1:
            msg = (
                "y_pred has only a single vintage (vintage_time). "
                "plot_score_per_vintage requires at least 2 vintages. "
                "Use plot_score_per_step for single-vintage data."
            )
            raise ValueError(msg)

    def _render_vintage(
        fig: go.Figure,
        trace_name: str,
        vintages: pl.Series,
        score_values: pl.Series,
        color: str,
        _show_legend: bool = True,
        *,
        row: int | None = None,
        col: int | None = None,
    ) -> None:
        """Render a single vintage trace on the figure."""
        if kind == "line":
            fig.add_trace(
                go.Scatter(
                    x=vintages,
                    y=score_values,
                    mode="lines+markers",
                    name=trace_name,
                    line={"color": color, "width": line_width},
                    marker={"size": marker_size, "color": color, "opacity": marker_opacity},
                    showlegend=_show_legend,
                ),
                row=row,
                col=col,
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=vintages,
                    y=score_values,
                    name=trace_name,
                    marker_color=color,
                    opacity=bar_opacity,
                    showlegend=_show_legend,
                ),
                row=row,
                col=col,
            )

        if show_trend and len(score_values) >= 2:
            sv_np = score_values.to_numpy().astype(float)
            x_num = np.arange(len(sv_np))
            coeffs = np.polyfit(x_num, sv_np, 1)
            trend_y = np.polyval(coeffs, x_num)
            fig.add_trace(
                go.Scatter(
                    x=vintages,
                    y=trend_y,
                    mode="lines",
                    name=f"{trace_name} trend",
                    line={"color": color, "width": 1.5, "dash": "dash"},
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    # Multi-scorer + multi-model -> faceted subplots
    if multi_scorer and n_models > 1:
        _warn_large_grid(n_scorers, n_models)

        if compare_by == "model":
            facet_labels = list(scorer_dict.keys())
            overlay_labels = list(y_pred_dict.keys())
        else:
            facet_labels = list(y_pred_dict.keys())
            overlay_labels = list(scorer_dict.keys())

        n_facets = len(facet_labels)
        n_cols = min(facet_n_cols, n_facets)
        n_rows = (n_facets + n_cols - 1) // n_cols
        colors = resolve_color_palette(color_palette, len(overlay_labels))

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=facet_labels,
            shared_xaxes=True,
            vertical_spacing=_subplot_spacing(n_rows),
        )

        legend_tracker = LegendTracker()
        for facet_idx, facet_label in enumerate(facet_labels):
            r = facet_idx // n_cols + 1
            c = facet_idx % n_cols + 1
            for overlay_idx, overlay_label in enumerate(overlay_labels):
                if compare_by == "model":
                    s_orig = scorer_dict[facet_label]
                    y_pm = y_pred_dict[overlay_label]
                else:
                    s_orig = scorer_dict[overlay_label]
                    y_pm = y_pred_dict[facet_label]

                vintages, sv = _compute_vintage_scores(s_orig, y_truth, y_pm)
                _render_vintage(
                    fig,
                    overlay_label,
                    vintages,
                    sv,
                    colors[overlay_idx],
                    legend_tracker.should_show(overlay_label),
                    row=r,
                    col=c,
                )

        fig = apply_default_layout(
            fig,
            title=title or "Score by Vintage",
            x_label=x_label or "Vintage (Observed Time)",
            y_label=y_label or "Score",
            width=width,
            height=height or max(300 * n_rows, 400),
        )
        if kind == "bar":
            fig.update_layout(barmode="group")
        fig.update_layout(showlegend=show_legend)
        return fig

    # Single figure
    fig = go.Figure()

    if multi_scorer:
        # Overlay scorers (single model)
        colors = resolve_color_palette(color_palette, n_scorers)
        y_pred_single = next(iter(y_pred_dict.values()))
        for idx, (s_name, s_orig) in enumerate(scorer_dict.items()):
            vintages, sv = _compute_vintage_scores(s_orig, y_truth, y_pred_single)
            _render_vintage(fig, s_name, vintages, sv, colors[idx], show_legend)

        default_title = title or "Score by Vintage"
        default_y = y_label or "Score"
    else:
        # Overlay models
        colors = resolve_color_palette(color_palette, n_models)
        first_scorer = next(iter(scorer_dict.values()))
        for idx, (m_name, y_pm) in enumerate(y_pred_dict.items()):
            vintages, sv = _compute_vintage_scores(first_scorer, y_truth, y_pm)
            _render_vintage(fig, m_name, vintages, sv, colors[idx], show_legend)

        scorer_name = first_scorer.__class__.__name__
        default_title = title or f"{scorer_name} by Vintage"
        default_y = y_label or scorer_name

    fig = apply_default_layout(
        fig,
        title=default_title,
        x_label=x_label or "Vintage (Observed Time)",
        y_label=default_y,
        width=width,
        height=height,
    )

    if kind == "bar" and (n_models > 1 or multi_scorer):
        fig.update_layout(barmode="group")

    fig.update_layout(showlegend=show_legend)
    return fig


def plot_score_heatmap(
    scorer: BaseScorer,
    y_truth: pl.DataFrame,
    y_pred: pl.DataFrame,
    *,
    x_dim: Literal["step", "vintage"] = "step",
    y_dim: Literal["step", "vintage"] = "vintage",
    color_palette: str | None = None,
    text_format: str = ".2f",
    show_text: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot a 2D heatmap of scores across two forecast dimensions.

    Creates a heatmap where each cell shows the score for a specific
    combination of forecast dimensions (e.g., horizon step vs vintage).

    Parameters
    ----------
    scorer : BaseScorer
        Yohou scorer instance. Used to compute per-cell scores.
    y_truth : pl.DataFrame
        Ground truth with ``"time"`` column.
    y_pred : pl.DataFrame
        Predictions with ``"vintage_time"`` and ``"time"`` columns.
        Only single-model input is supported.
    x_dim : str, default="step"
        Dimension for the x-axis: ``"step"`` (forecast horizon) or
        ``"vintage"`` (observed time).
    y_dim : str, default="vintage"
        Dimension for the y-axis: ``"step"`` or ``"vintage"``.
        Must differ from ``x_dim``.
    color_palette : str or None, default=None
        Plotly colorscale name. If None, auto-selects based on
        ``scorer._lower_is_better``: ``"Blues"`` when lower is better,
        ``"Blues_r"`` otherwise.
    text_format : str, default=".2f"
        Format string for cell text annotations.
    show_text : bool, default=True
        Whether to display numeric annotations in cells.
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

    Returns
    -------
    go.Figure
        Plotly figure with heatmap.

    Raises
    ------
    ValueError
        If ``x_dim`` and ``y_dim`` are the same, or if ``y_pred`` has
        a single vintage.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MeanAbsoluteError
    >>> from yohou.plotting import plot_score_heatmap

    >>> y_truth = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 4)],
    ...     "value": [10.0, 20.0, 30.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "vintage_time": [datetime(2019, 12, 30)] * 3 + [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, i) for i in range(1, 4)] * 2,
    ...     "value": [12.0, 19.0, 28.0, 11.0, 21.0, 29.0],
    ... }).sort("time")

    >>> fig = plot_score_heatmap(MeanAbsoluteError(), y_truth, y_pred)
    >>> len(fig.data)
    1

    See Also
    --------
    [`plot_score_per_step`][yohou.plotting.plot_score_per_step] : Score by horizon step.
    [`plot_score_per_vintage`][yohou.plotting.plot_score_per_vintage] : Score by vintage.
    """
    validate_plotting_data(y_truth)
    validate_plotting_data(y_pred)
    validate_plotting_params(width=width, height=height)

    if x_dim == y_dim:
        msg = f"x_dim and y_dim must differ, got x_dim={x_dim!r} and y_dim={y_dim!r}"
        raise ValueError(msg)

    valid_dims = {"step", "vintage"}
    for dim_name, dim_val in [("x_dim", x_dim), ("y_dim", y_dim)]:
        if dim_val not in valid_dims:
            msg = f"{dim_name} must be one of {valid_dims}, got {dim_val!r}"
            raise ValueError(msg)

    if "vintage_time" not in y_pred.columns:
        msg = "y_pred must have a 'vintage_time' column for heatmap plotting"
        raise ValueError(msg)

    vintages = y_pred["vintage_time"].unique().sort()
    if len(vintages) <= 1:
        msg = "y_pred has only a single vintage (vintage_time). plot_score_heatmap requires at least 2 vintages."
        raise ValueError(msg)

    # Compute per-vintage, per-step scores
    scorer_cw = _prepare_scorer_for_componentwise(copy.deepcopy(scorer))
    scorer_cw.fit(y_truth)

    score_matrix: list[list[float]] = []
    vintage_labels: list[str] = []
    n_steps: int | None = None

    for vintage_val in vintages:
        y_pred_v = y_pred.filter(pl.col("vintage_time") == vintage_val)
        scores_df = scorer_cw.score(y_truth, y_pred_v)

        if not isinstance(scores_df, pl.DataFrame):
            msg = f"Scorer must return DataFrame for componentwise aggregation, got {type(scores_df).__name__}"
            raise TypeError(msg)

        score_cols = [c for c in scores_df.columns if c not in _SCORER_META_COLS]
        if len(score_cols) == 1:
            row_scores = scores_df[score_cols[0]].to_list()
        else:
            row_scores = scores_df.select(score_cols).mean_horizontal().to_list()

        if n_steps is None:
            n_steps = len(row_scores)

        # Pad or truncate to consistent length
        row_scores = row_scores[:n_steps] + [float("nan")] * max(0, n_steps - len(row_scores))

        score_matrix.append(row_scores)
        vintage_labels.append(str(vintage_val))

    step_labels = [str(i) for i in range(1, (n_steps or 0) + 1)]
    z = np.array(score_matrix)

    # Arrange dimensions
    if x_dim == "step" and y_dim == "vintage":
        x_labels = step_labels
        y_labels = vintage_labels
        z_plot = z
    else:
        # x_dim == "vintage" and y_dim == "step"
        x_labels = vintage_labels
        y_labels = step_labels
        z_plot = z.T

    # Auto-select colorscale
    if color_palette is None:
        lower_is_better = getattr(scorer, "_lower_is_better", True)
        colorscale = "Blues" if lower_is_better else "Blues_r"
    else:
        colorscale = color_palette

    # Build text annotations
    text_vals = [[f"{v:{text_format}}" for v in row] for row in z_plot] if show_text else None

    fig = go.Figure(
        data=go.Heatmap(
            z=z_plot,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            text=text_vals,
            texttemplate="%{text}" if show_text else None,
            hovertemplate="x: %{x}<br>y: %{y}<br>Score: %{z:.3f}<extra></extra>",
        )
    )

    scorer_name = scorer.__class__.__name__
    default_x = x_label or ("Horizon Step" if x_dim == "step" else "Vintage (Observed Time)")
    default_y = y_label or ("Horizon Step" if y_dim == "step" else "Vintage (Observed Time)")
    default_title = title or f"{scorer_name} Heatmap"

    fig = apply_default_layout(
        fig,
        title=default_title,
        x_label=default_x,
        y_label=default_y,
        width=width,
        height=height,
    )

    return fig


def plot_group_scores(
    scorer: BaseScorer | dict[str, BaseScorer],
    y_truth: pl.DataFrame,
    y_pred: pl.DataFrame | dict[str, pl.DataFrame],
    *,
    kind: Literal["bar", "box", "heatmap"] = "bar",
    distribute_by: Literal["time", "vintage", "step"] = "time",
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    bar_opacity: float = 0.85,
    sort_ascending: bool | None = None,
    text_auto: bool = True,
) -> go.Figure:
    """Plot scores broken down by panel group.

    Requires panel data where ``y_truth`` has group columns
    (e.g. ``group__member``). Computes the scorer for each group
    independently and visualises the results.

    Parameters
    ----------
    scorer : BaseScorer or dict[str, BaseScorer]
        Yohou scorer instance.

        - If BaseScorer: single scorer to evaluate per group.
        - If dict: keys are scorer names, values are scorer instances.
    y_truth : pl.DataFrame
        Ground truth panel data with ``"time"`` and group columns.
    y_pred : pl.DataFrame or dict[str, pl.DataFrame]
        Predictions.

        - If DataFrame: single forecast.
        - If dict: keys are model names, values are prediction DataFrames.
    kind : str, default="bar"
        Plot kind: ``"bar"`` for aggregate bar chart, ``"box"`` for
        box-plot distribution of per-record scores, ``"heatmap"`` for
        a 2D heatmap of groups vs models/scorers.
    distribute_by : str, default="time"
        For ``kind="box"``, the dimension whose variability the
        box plot shows:

        - ``"time"``: per-timestep score variability.
        - ``"vintage"``: per-vintage score variability.
        - ``"step"``: per-step score variability.

        Ignored for ``kind="bar"``.
    color_palette : list[str] | None, default=None
        Custom colour palette.
    show_legend : bool, default=True
        Whether to show the legend.
    title : str | None, default=None
        Plot title.
    x_label : str | None, default=None
        X-axis label. Defaults to ``"Group"``.
    y_label : str | None, default=None
        Y-axis label.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    bar_opacity : float, default=0.85
        Opacity of bars.
    sort_ascending : bool or None, default=None
        Sort groups by score. ``True`` for ascending, ``False`` for
        descending, ``None`` for insertion order.
    text_auto : bool, default=True
        Annotate bars with their values (only for ``kind="bar"``).

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    ValueError
        If ``y_truth`` does not contain panel group columns.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.metrics import MeanAbsoluteError
    >>> from yohou.plotting import plot_group_scores

    >>> y_truth = pl.DataFrame({
    ...     "time": [datetime(2020, 1, i) for i in range(1, 4)],
    ...     "region__east": [10.0, 20.0, 30.0],
    ...     "region__west": [15.0, 25.0, 35.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "vintage_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, i) for i in range(1, 4)],
    ...     "region__east": [12.0, 19.0, 28.0],
    ...     "region__west": [14.0, 26.0, 33.0],
    ... })

    >>> fig = plot_group_scores(MeanAbsoluteError(), y_truth, y_pred)
    >>> len(fig.data) >= 1
    True

    See Also
    --------
    [`plot_score_time_series`][yohou.plotting.plot_score_time_series] : Score over time.
    [`plot_score_per_step`][yohou.plotting.plot_score_per_step] : Score by horizon step.
    """
    validate_plotting_data(y_truth)
    validate_plotting_params(kind=kind, valid_kinds={"bar", "box", "heatmap"}, width=width, height=height)

    y_pred_dict = _normalize_y_pred(y_pred)
    scorer_dict = _normalize_scorers(scorer)

    # Detect panel groups
    _, panel_groups = inspect_panel(y_truth)
    if not panel_groups:
        msg = (
            "y_truth does not contain panel group columns (e.g. 'group__member'). "
            "plot_group_scores requires panel data."
        )
        raise ValueError(msg)

    group_names = sorted(panel_groups)
    n_models = len(y_pred_dict)

    if kind == "bar":
        # Compute aggregate score per group per model per scorer
        # For bar chart: x-axis = groups, one bar per model (for single scorer)
        #                or grouped by scorer x model

        all_entries: list[tuple[str, str, str, float]] = []  # (scorer, model, group, score)

        for s_name, s_orig in scorer_dict.items():
            for m_name, y_pm in y_pred_dict.items():
                validate_plotting_data(y_pm)
                for gname in group_names:
                    g_truth_cols = ["time"] + [c for c in y_truth.columns if c.startswith(f"{gname}__")]
                    g_pred_cols = [
                        c for c in y_pm.columns if c in ("time", "vintage_time") or c.startswith(f"{gname}__")
                    ]
                    y_truth_g = y_truth.select(g_truth_cols)
                    y_pm_g = y_pm.select(g_pred_cols)

                    s_agg = copy.deepcopy(s_orig)
                    s_agg.fit(y_truth_g)
                    score_val = s_agg.score(y_truth_g, y_pm_g)

                    if isinstance(score_val, pl.DataFrame):
                        score_cols = [c for c in score_val.columns if c not in _SCORER_META_COLS]
                        group_score = float(score_val.select(score_cols).mean_horizontal().mean())  # type: ignore
                    else:
                        group_score = float(score_val)  # type: ignore
                    all_entries.append((s_name, m_name, gname, group_score))

        # Build grouped bar chart
        n_scorers = len(scorer_dict)
        multi_scorer = n_scorers > 1

        if multi_scorer and n_models > 1:
            # Series = model x scorer combination
            series_labels = [f"{m} / {s}" for s in scorer_dict for m in y_pred_dict]
        elif multi_scorer:
            series_labels = list(scorer_dict.keys())
        else:
            series_labels = list(y_pred_dict.keys())

        colors = resolve_color_palette(color_palette, len(series_labels))

        # Optional sorting by first series
        if sort_ascending is not None and all_entries:
            first_series_scores = {
                e[2]: e[3]
                for e in all_entries
                if (e[0] == list(scorer_dict.keys())[0] and e[1] == list(y_pred_dict.keys())[0])
            }
            group_names = sorted(
                group_names,
                key=lambda g: first_series_scores.get(g, 0.0),
                reverse=not sort_ascending,
            )

        fig = go.Figure()
        series_idx = 0

        for s_name in scorer_dict:
            for m_name in y_pred_dict:
                values = []
                for gn in group_names:
                    match = [e[3] for e in all_entries if e[0] == s_name and e[1] == m_name and e[2] == gn]
                    values.append(match[0] if match else 0.0)

                if multi_scorer and n_models > 1:
                    label = f"{m_name} / {s_name}"
                elif multi_scorer:
                    label = s_name
                else:
                    label = m_name

                bar_kwargs: dict = {
                    "x": group_names,
                    "y": values,
                    "name": label,
                    "marker_color": colors[series_idx % len(colors)],
                    "opacity": bar_opacity,
                }
                if text_auto:
                    bar_kwargs["text"] = [f"{v:.3g}" for v in values]
                    bar_kwargs["textposition"] = "outside"

                fig.add_trace(go.Bar(**bar_kwargs))
                series_idx += 1

        fig.update_layout(barmode="group")

        if multi_scorer:
            default_title = title or "Group Scores"
            default_y = y_label or "Score"
        else:
            first_scorer = next(iter(scorer_dict.values()))
            scorer_name = first_scorer.__class__.__name__
            default_title = title or f"{scorer_name} by Group"
            default_y = y_label or scorer_name

        fig = apply_default_layout(
            fig,
            title=default_title,
            x_label=x_label or "Group",
            y_label=default_y,
            width=width,
            height=height,
        )
        fig.update_layout(showlegend=show_legend)
        return fig

    if kind == "heatmap":
        # Heatmap: groups on y-axis, models/scorers on x-axis, score as color
        all_entries: list[tuple[str, str, str, float]] = []

        for s_name, s_orig in scorer_dict.items():
            for m_name, y_pm in y_pred_dict.items():
                validate_plotting_data(y_pm)
                for gname in group_names:
                    g_truth_cols = ["time"] + [c for c in y_truth.columns if c.startswith(f"{gname}__")]
                    g_pred_cols = [
                        c for c in y_pm.columns if c in ("time", "vintage_time") or c.startswith(f"{gname}__")
                    ]
                    y_truth_g = y_truth.select(g_truth_cols)
                    y_pm_g = y_pm.select(g_pred_cols)

                    s_agg = copy.deepcopy(s_orig)
                    s_agg.fit(y_truth_g)
                    score_val = s_agg.score(y_truth_g, y_pm_g)

                    if isinstance(score_val, pl.DataFrame):
                        score_cols = [c for c in score_val.columns if c not in _SCORER_META_COLS]
                        group_score = float(score_val.select(score_cols).mean_horizontal().mean())  # type: ignore
                    else:
                        group_score = float(score_val)  # type: ignore
                    all_entries.append((s_name, m_name, gname, group_score))

        n_scorers = len(scorer_dict)
        multi_scorer = n_scorers > 1

        if multi_scorer and n_models > 1:
            series_labels = [f"{m} / {s}" for s in scorer_dict for m in y_pred_dict]
        elif multi_scorer:
            series_labels = list(scorer_dict.keys())
        else:
            series_labels = list(y_pred_dict.keys())

        # Build z-matrix: rows = groups, cols = series
        z_matrix: list[list[float]] = []
        for gname in group_names:
            row: list[float] = []
            for s_name in scorer_dict:
                for m_name in y_pred_dict:
                    match = [e[3] for e in all_entries if e[0] == s_name and e[1] == m_name and e[2] == gname]
                    row.append(match[0] if match else 0.0)
            z_matrix.append(row)

        # Auto-select colorscale direction
        first_scorer = next(iter(scorer_dict.values()))
        lower_better = getattr(first_scorer, "_lower_is_better", True)
        colorscale = "Blues" if lower_better else "Blues_r"

        text_vals = [[f"{v:.3g}" for v in row] for row in z_matrix] if text_auto else None

        fig = go.Figure(
            data=go.Heatmap(
                z=z_matrix,
                x=series_labels,
                y=group_names,
                colorscale=colorscale,
                text=text_vals,
                texttemplate="%{text}" if text_auto else "",
                hovertemplate="Group: %{y}<br>%{x}: %{z:.3g}<extra></extra>",
            )
        )

        scorer_name = first_scorer.__class__.__name__
        default_title = title or ("Group Scores" if multi_scorer else f"{scorer_name} by Group")

        fig = apply_default_layout(
            fig,
            title=default_title,
            x_label=x_label
            or (
                "Model / Scorer"
                if multi_scorer and n_models > 1
                else "Model"
                if n_models > 1
                else "Scorer"
                if multi_scorer
                else "Model"
            ),
            y_label=y_label or "Group",
            width=width,
            height=height,
        )
        fig.update_layout(showlegend=False)
        return fig

    # kind="box": distribution of per-record scores across groups
    # Use componentwise or stepwise/vintagewise aggregation depending on distribute_by
    agg_map = {
        "time": "componentwise",
        "vintage": "stepwise",
        "step": "vintagewise",
    }
    agg_method = agg_map[distribute_by]

    first_scorer = next(iter(scorer_dict.values()))
    first_y_pred = next(iter(y_pred_dict.values()))

    s_cw = copy.deepcopy(first_scorer)
    if isinstance(s_cw, BaseIntervalScorer):
        s_cw.set_params(aggregation_method=[agg_method, "coveragewise"])
    else:
        s_cw.set_params(aggregation_method=agg_method)
    s_cw.fit(y_truth)

    scores_df = s_cw.score(y_truth, first_y_pred)
    if not isinstance(scores_df, pl.DataFrame):
        msg = f"Scorer must return DataFrame for {agg_method} aggregation, got {type(scores_df).__name__}"
        raise TypeError(msg)

    colors = resolve_color_palette(color_palette, len(group_names))
    fig = go.Figure()

    for g_idx, gname in enumerate(group_names):
        g_cols = [c for c in scores_df.columns if c.startswith(f"{gname}__")]
        if not g_cols:
            continue
        if len(g_cols) == 1:
            g_scores = scores_df[g_cols[0]].drop_nulls().to_numpy()
        else:
            g_scores = scores_df.select(g_cols).to_numpy().flatten()
            g_scores = g_scores[~np.isnan(g_scores)]

        fig.add_trace(
            go.Box(
                y=g_scores,
                name=gname,
                marker_color=colors[g_idx % len(colors)],
                boxmean=True,
            )
        )

    scorer_name = first_scorer.__class__.__name__
    fig = apply_default_layout(
        fig,
        title=title or f"{scorer_name} Distribution by Group",
        x_label=x_label or "Group",
        y_label=y_label or scorer_name,
        width=width,
        height=height,
    )
    fig.update_layout(showlegend=show_legend)
    return fig


def _discover_proba_targets(
    proba_cols: list[str],
) -> dict[str, list[str]]:
    """Group probability columns by target name.

    Parameters
    ----------
    proba_cols : list of str
        Column names containing ``_proba_``.

    Returns
    -------
    dict
        Mapping from target name to list of probability column names.

    """
    targets: dict[str, list[str]] = {}
    for col in proba_cols:
        target_name = col.split("_proba_", 1)[0]
        targets.setdefault(target_name, []).append(col)
    return targets
