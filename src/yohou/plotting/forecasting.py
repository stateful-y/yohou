"""Forecast visualization functions."""

import re
from typing import Literal

import numpy as np
import polars as pl
from plotly.subplots import make_subplots

try:
    import plotly.graph_objects as go
except ImportError as e:
    msg = "plotly is required for yohou plotting. Install: pip install yohou[plotting]"
    raise ImportError(msg) from e

from yohou.plotting._utils import (
    LegendTracker,
    PanelColorManager,
    RenderContext,
    _create_figure,
    _create_subplots,
    _fill_trace_kwargs,
    _group_panel_columns,
    _make_hovertemplate,
    _member_name,
    _subplot_spacing,
    apply_default_layout,
    build_category_map,
    facet_figure,
    palette_yohou,
    resolve_color_palette,
    resolve_panel_columns,
)
from yohou.plotting.evaluation import _discover_proba_targets, _hex_to_rgba
from yohou.utils import inspect_panel, validate_plotting_data, validate_plotting_params

# The full palette list is used as the default effective palette in every
# plot_forecast code path. Slots 0/1/2 are reserved for the three semantic
# traces (history, single-model forecast, actual); slot 3+ are model colors.
_PALETTE = list(palette_yohou().values())

__all__ = [
    "plot_decomposition",
    "plot_forecast",
    "plot_time_weight",
]


def _detect_prediction_mode(df: pl.DataFrame) -> str:
    """Detect the prediction type from a single prediction DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        A prediction DataFrame to inspect.

    Returns
    -------
    str
        ``"class_proba"`` if ``_proba_`` columns are present,
        ``"categorical"`` if any value column is String/Categorical,
        ``"numeric"`` otherwise.

    """
    if any("_proba_" in c for c in df.columns):
        return "class_proba"
    value_cols = [c for c in df.columns if c not in ("time", "observed_time")]
    if value_cols and any(df[c].dtype in (pl.String, pl.Categorical) for c in value_cols):
        return "categorical"
    return "numeric"


def plot_forecast(
    y_test: pl.DataFrame,
    y_pred: pl.DataFrame | dict[str, pl.DataFrame],
    *,
    y_train: pl.DataFrame | None = None,
    columns: str | list[str] | None = None,
    coverage_rates: list[float] | None = None,
    n_history: int | None = None,
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
    band_opacity: float = 0.25,
    show_transition: bool = True,
) -> go.Figure:
    """Plot forecasts with historical data and optional prediction intervals.

    Accepts separate DataFrames for actuals and predictions following an
    sklearn-like API. Automatically detects interval columns from y_pred
    when coverage_rates is provided.

    When *y_pred* is a ``dict[str, pl.DataFrame]``, each entry is treated as
    a separate model and plotted with a distinct color for side-by-side
    comparison.

    **Categorical support**: When *y_pred* contains class-probability
    columns (``{target}_proba_{class}`` pattern), renders a stacked area
    chart of predicted probabilities with ground-truth class markers from
    *y_test*. When *y_pred* contains categorical (string) columns, renders
    a step chart comparing predicted and actual class labels over time.

    Parameters
    ----------
    y_test : pl.DataFrame
        Actual test values with 'time' column.
    y_pred : pl.DataFrame | dict[str, pl.DataFrame]
        Forecast values with 'time' column. May also contain interval columns
        named ``{col}_lower_{rate}`` and ``{col}_upper_{rate}``.
        If a dict, keys are model names and values are prediction DataFrames.
    y_train : pl.DataFrame | None, default=None
        Historical training data with 'time' column. If provided, shown
        before the forecast period.
    columns : str | list[str] | None, default=None
        Target column(s) to plot from *y_test*.  When ``None``, all
        non-time columns are used.  Associated interval columns
        (``{col}_lower_{rate}`` / ``{col}_upper_{rate}``) are kept
        automatically.
    coverage_rates : list[float] | None, default=None
        Coverage rates to display intervals for (e.g., [0.9, 0.95]).
        Looks for ``{col}_lower_{rate}`` / ``{col}_upper_{rate}`` in y_pred.
    n_history : int | None, default=None
        Number of historical observations to show from y_train. If None, shows all.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot. If None and panel data is detected,
        plots all groups. Creates faceted subplots.
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data.  ``"group"`` creates one subplot per
        group, ``"member"`` one per member.  ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in facet grid for panel data.
    color_palette : list[str] | None, default=None
        Custom color palette.
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
    connect_gaps : bool, default=False
        Whether to connect gaps in the data with lines.
    resampler : bool | Literal["widget"] | None, default=None
        Enable plotly-resampler for large datasets.  ``True`` or
        ``"widget"`` creates a ``FigureWidgetResampler``; ``False`` or
        ``None`` uses a plain ``go.Figure``.
    line_width : float, default=2.0
        Width of line traces.
    band_opacity : float, default=0.25
        Opacity of prediction interval bands.
    show_transition : bool, default=True
        Whether to show a dashed connector between the last training
        point and the first forecast point.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    TypeError
        If inputs are not Polars DataFrames.
    ValueError
        If DataFrames are empty or missing 'time' column.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_forecast

    >>> # Create sample data
    >>> y_train = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
    ...     "y": [100 + i for i in range(91)],
    ... })
    >>> y_test = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
    ...     "y": [191 + i for i in range(30)],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
    ...     "y": [190 + i + (i % 3) for i in range(30)],
    ... })

    >>> fig = plot_forecast(y_test, y_pred, y_train=y_train)
    >>> len(fig.data) >= 2
    True

    Multi-model comparison:

    >>> y_pred_b = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
    ...     "y": [192 + i for i in range(30)],
    ... })
    >>> fig = plot_forecast(y_test, {"Model A": y_pred, "Model B": y_pred_b})
    >>> len(fig.data) >= 3
    True

    See Also
    --------
    [`plot_residuals`][yohou.plotting.plot_residuals] : Plot residual diagnostics.
    [`plot_model_comparison_bar`][yohou.plotting.plot_model_comparison_bar] : Grouped bar chart for scorer comparison.
    """
    # Validate inputs
    validate_plotting_data(y_test)
    if isinstance(y_pred, dict):
        for _name, pred_df in y_pred.items():
            validate_plotting_data(pred_df)  # ty: ignore[invalid-argument-type]
    else:
        validate_plotting_data(y_pred)
    if y_train is not None:
        validate_plotting_data(y_train)
    validate_plotting_params(width=width, height=height)

    # Semantic colors always come from the effective palette: slot 0 = history,
    # slot 1 = single-model forecast, slot 2 = actual, slot 3+ = model comparison.
    eff_palette = color_palette if color_palette is not None else _PALETTE
    forecast_color = eff_palette[1 % len(eff_palette)]
    actual_color = eff_palette[2 % len(eff_palette)]

    # Detect panel data
    _, y_test_panels = inspect_panel(y_test)
    is_panel = bool(y_test_panels)

    # Auto-detect prediction type from the first prediction DataFrame
    _first_pred = next(iter(y_pred.values())) if isinstance(y_pred, dict) else y_pred
    prediction_mode = _detect_prediction_mode(_first_pred)  # ty: ignore[invalid-argument-type]

    # For panel data, delegate to faceted handler (single or multi-model)
    if is_panel:
        return _plot_forecast_panel(
            y_test=y_test,
            y_pred=y_pred,
            y_train=y_train,
            coverage_rates=coverage_rates,
            n_history=n_history,
            panel_group_names=panel_group_names,
            facet_by=facet_by,
            facet_n_cols=facet_n_cols,
            color_palette=color_palette,
            title=title,
            x_label=x_label,
            y_label=y_label,
            width=width,
            height=height,
            line_width=line_width,
            band_opacity=band_opacity,
            show_transition=show_transition,
            connect_gaps=connect_gaps,
            resampler=resampler,
            show_legend=show_legend,
            prediction_mode=prediction_mode,
        )

    # Non-panel class-probability predictions
    if prediction_mode == "class_proba":
        return _plot_forecast_class_proba(
            y_test=y_test,
            y_pred=y_pred,
            columns=columns,
            color_palette=color_palette,
            title=title,
            x_label=x_label,
            y_label=y_label,
            width=width,
            height=height,
            facet_n_cols=facet_n_cols,
        )

    # Non-panel categorical predictions
    if prediction_mode == "categorical":
        return _plot_forecast_categorical(
            y_test=y_test,
            y_pred=y_pred,
            y_train=y_train,
            n_history=n_history,
            columns=columns,
            color_palette=color_palette,
            title=title,
            x_label=x_label,
            y_label=y_label,
            width=width,
            height=height,
            facet_n_cols=facet_n_cols,
        )

    # Multi-model dict: delegate to dedicated helper
    if isinstance(y_pred, dict):
        return _plot_forecast_multi_model(
            y_test=y_test,
            y_preds=y_pred,  # ty: ignore[invalid-argument-type]
            y_train=y_train,
            coverage_rates=coverage_rates,
            n_history=n_history,
            columns=columns,
            color_palette=color_palette,
            title=title,
            x_label=x_label,
            y_label=y_label,
            width=width,
            height=height,
            facet_n_cols=facet_n_cols,
            line_width=line_width,
            band_opacity=band_opacity,
            show_transition=show_transition,
            connect_gaps=connect_gaps,
            resampler=resampler,
            show_legend=show_legend,
        )

    # Non-panel, single-model case
    interval_pattern = re.compile(r"^.+_(lower|upper)_[\d.]+$")
    pred_value_cols = [
        c for c in y_pred.columns if c not in ("time", "observed_time") and not interval_pattern.match(c)
    ]
    test_value_cols = [c for c in y_test.columns if c != "time"]

    # Apply columns filter (Decision #8: columns param on plot_forecast)
    if columns is not None:
        col_list = [columns] if isinstance(columns, str) else list(columns)
        test_value_cols = [c for c in col_list if c in test_value_cols]
    plot_columns = test_value_cols

    def _render_forecast(ctx: RenderContext) -> None:
        """Render train/interval/actual/forecast traces for one target column."""
        col = ctx.display_name
        pred_col = col if col in pred_value_cols else (pred_value_cols[0] if pred_value_cols else None)

        # Training data
        if y_train is not None and col in y_train.columns:
            train_df = y_train.tail(n_history) if n_history is not None else y_train
            _hex = actual_color.lstrip("#")
            _rgb = tuple(int(_hex[i : i + 2], 16) for i in (0, 2, 4))
            _train_color = f"rgba({_rgb[0]}, {_rgb[1]}, {_rgb[2]}, 0.4)"
            ctx.fig.add_trace(
                go.Scatter(
                    x=train_df["time"],
                    y=train_df[col],
                    mode="lines",
                    line={"color": _train_color, "width": line_width},
                    connectgaps=connect_gaps,
                    name=f"{col} (Train)",
                    legendrank=0,
                    hovertemplate=_make_hovertemplate(f"{col} Train", "Time", "Value"),
                ),
                row=ctx.row,
                col=ctx.col,
            )

        # Prediction intervals (rendered before Actual so actual sits on top)
        if coverage_rates:
            interval_base = pred_col if pred_col is not None else col
            _hex = forecast_color.lstrip("#")
            rgb = tuple(int(_hex[i : i + 2], 16) for i in (0, 2, 4))
            sorted_rates = sorted(coverage_rates)
            n_rates = len(sorted_rates)
            for sort_idx, rate in enumerate(sorted_rates):
                lower_col = f"{interval_base}_lower_{rate}"
                upper_col = f"{interval_base}_upper_{rate}"
                if lower_col in y_pred.columns and upper_col in y_pred.columns:
                    rate_opacity = band_opacity * (1.0 - 0.45 * sort_idx / max(1, n_rates - 1))
                    rgba = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {rate_opacity:.3f})"
                    t = y_pred["time"].to_list()
                    y_upper = y_pred[upper_col].to_list()
                    y_lower = y_pred[lower_col].to_list()
                    x_band = t + t[::-1]
                    y_band = y_upper + y_lower[::-1]
                    ctx.fig.add_trace(
                        go.Scatter(
                            x=x_band,
                            y=y_band,
                            fill="toself",
                            fillcolor=rgba,
                            mode="lines",
                            line={"width": 0, "color": rgba},
                            name=f"{col} ({rate:.0%} PI)",
                            legendrank=11 + sort_idx,
                            hoverinfo="skip",
                        ),
                        row=ctx.row,
                        col=ctx.col,
                    )

        # Actual test data (prepend last train point to close the gap)
        _x_actual = y_test["time"]
        _y_actual = y_test[col]
        if y_train is not None and col in y_train.columns:
            _x_actual = pl.concat([pl.Series("time", [y_train["time"][-1]]), _x_actual])
            _y_actual = pl.concat([pl.Series([y_train[col][-1]], dtype=_y_actual.dtype), _y_actual])
        ctx.fig.add_trace(
            go.Scatter(
                x=_x_actual,
                y=_y_actual,
                mode="lines",
                line={"color": actual_color, "width": line_width},
                connectgaps=connect_gaps,
                name=f"{col} (Actual)",
                legendrank=1,
                hovertemplate=_make_hovertemplate(f"{col} Actual", "Time", "Value"),
            ),
            row=ctx.row,
            col=ctx.col,
        )

        # Forecast
        if pred_col is not None and pred_col in y_pred.columns:
            x_forecast = y_pred["time"]
            forecast_y = y_pred[pred_col]
            if show_transition and y_train is not None and col in y_train.columns:
                last_train_time = y_train["time"][-1]
                last_train_val = y_train[col][-1]
                x_forecast = pl.concat([pl.Series("time", [last_train_time]), y_pred["time"]])
                forecast_y = pl.concat([pl.Series([last_train_val], dtype=forecast_y.dtype), forecast_y])
            ctx.fig.add_trace(
                go.Scatter(
                    x=x_forecast,
                    y=forecast_y,
                    mode="lines",
                    line={"color": forecast_color, "width": line_width},
                    connectgaps=connect_gaps,
                    name=f"{col} (Forecast)",
                    legendrank=10,
                    hovertemplate=_make_hovertemplate(f"{col} Forecast", "Time", "Value"),
                ),
                row=ctx.row,
                col=ctx.col,
            )

    fig = facet_figure(
        y_test,
        _render_forecast,
        columns=plot_columns,
        facet_n_cols=facet_n_cols,
        title=title or "Forecast",
        x_label=x_label or "Time",
        y_label=y_label or "Value",
        width=width,
        height=height,
        resampler=resampler,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


def _render_class_proba_traces(
    fig: go.Figure,
    pred_df: pl.DataFrame,
    y_test: pl.DataFrame,
    *,
    colors: list[str],
    target_proba_cols: list[str],
    class_labels: list[str],
    target: str,
    stack_group: str = "proba",
    row: int | None = None,
    col: int | None = None,
    show_legend: bool = True,
    line_width: float = 1.5,
    band_opacity: float = 0.6,
    marker_size: int = 10,
    seen_legend: set[str] | None = None,
) -> None:
    """Add class-probability traces (stacked area + truth markers) to a figure.

    Parameters
    ----------
    fig : go.Figure
        Figure to add traces to (may have subplots).
    pred_df : pl.DataFrame
        Single prediction DataFrame with ``_proba_`` columns.
    y_test : pl.DataFrame
        Ground-truth DataFrame.
    colors : list of str
        One hex color per class label.
    target_proba_cols : list of str
        Probability columns for the target.
    class_labels : list of str
        Class label names (same order as ``target_proba_cols``).
    target : str
        Target column name in ``y_test``.
    stack_group : str
        Plotly stackgroup identifier.
    row : int or None
        Subplot row (None for single-figure).
    col : int or None
        Subplot column (None for single-figure).
    show_legend : bool
        Whether to show legend for these traces.
    line_width : float
        Line width.
    band_opacity : float
        Fill opacity.
    marker_size : int
        Truth marker size.
    seen_legend : set of str or None
        Legend dedup tracker for panel mode.

    """
    _add = {"row": row, "col": col} if row is not None else {}

    for i, (proba_col, label) in enumerate(zip(target_proba_cols, class_labels, strict=True)):
        _show = show_legend if seen_legend is None else (label not in seen_legend and show_legend)
        if seen_legend is not None:
            seen_legend.add(label)
        fig.add_trace(
            go.Scatter(
                x=pred_df["time"],
                y=pred_df[proba_col],
                name=label,
                mode="lines",
                line={"width": line_width, "color": colors[i]},
                stackgroup=stack_group,
                fillcolor=_hex_to_rgba(colors[i], band_opacity),
                showlegend=_show,
                legendgroup=label,
                hovertemplate=f"<b>{label}</b><br>Time: %{{x}}<br>P: %{{y:.3f}}<extra></extra>",
            ),
            **_add,
        )

    if target in y_test.columns:
        common_times = set(y_test["time"].to_list()) & set(pred_df["time"].to_list())
        if common_times:
            truth_f = y_test.filter(pl.col("time").is_in(list(common_times))).sort("time")
            pred_f = pred_df.filter(pl.col("time").is_in(list(common_times))).sort("time")
            label_to_idx = {lb: idx for idx, lb in enumerate(class_labels)}
            marker_y: list[float] = []
            m_colors: list[str] = []
            for r_idx in range(len(truth_f)):
                tl = str(truth_f[target][r_idx])
                if tl in label_to_idx:
                    cn = target_proba_cols[label_to_idx[tl]]
                    marker_y.append(float(pred_f[cn][r_idx]))
                    m_colors.append(colors[label_to_idx[tl]])
                else:
                    marker_y.append(0.0)
                    m_colors.append(colors[0])

            _show_truth = show_legend if seen_legend is None else ("True class" not in seen_legend and show_legend)
            if seen_legend is not None:
                seen_legend.add("True class")
            fig.add_trace(
                go.Scatter(
                    x=truth_f["time"],
                    y=marker_y,
                    name="True class",
                    mode="markers",
                    marker={"size": marker_size, "color": m_colors, "symbol": "diamond"},
                    showlegend=_show_truth,
                    legendgroup="truth",
                    hovertemplate="<b>True: %{text}</b><br>Time: %{x}<br>P(true): %{y:.3f}<extra></extra>",
                    text=truth_f[target].cast(pl.String).to_list(),
                ),
                **_add,
            )


def _render_categorical_traces(
    fig: go.Figure,
    pred_df: pl.DataFrame,
    y_test: pl.DataFrame,
    y_train: pl.DataFrame | None,
    *,
    cat_cols: list[str],
    cat_to_int: dict[str, int],
    model_name: str,
    model_color: str,
    actual_color: str,
    is_multi_model: bool,
    n_history: int | None = None,
    row: int | None = None,
    col: int | None = None,
    show_legend: bool = True,
    line_width: float = 2.0,
    seen_legend: set[str] | None = None,
) -> None:
    """Add categorical traces (step chart) to a figure.

    Parameters
    ----------
    fig : go.Figure
        Figure to add traces to (may have subplots).
    pred_df : pl.DataFrame
        Prediction DataFrame with categorical columns.
    y_test : pl.DataFrame
        Ground-truth DataFrame.
    y_train : pl.DataFrame or None
        Training data.
    cat_cols : list of str
        Categorical column names.
    cat_to_int : dict
        Category-name to integer mapping.
    model_name : str
        Model display name.
    model_color : str
        Hex color for this model.
    actual_color : str
        Hex color for actual traces.
    is_multi_model : bool
        Whether multiple models are being compared.
    n_history : int or None
        Training rows to show.
    row : int or None
        Subplot row (None for single-figure).
    col : int or None
        Subplot column (None for single-figure).
    show_legend : bool
        Whether to show legend.
    line_width : float
        Line width.
    seen_legend : set of str or None
        Legend dedup tracker for panel mode.

    """
    _add = {"row": row, "col": col} if row is not None else {}

    for cat_col in cat_cols:
        # Training data
        if y_train is not None and cat_col in y_train.columns:
            train_df = y_train.tail(n_history) if n_history is not None else y_train
            y_vals = [cat_to_int.get(str(v), -1) for v in train_df[cat_col].to_list()]
            _hex = actual_color.lstrip("#")
            _rgb = tuple(int(_hex[i : i + 2], 16) for i in (0, 2, 4))
            _train_c = f"rgba({_rgb[0]}, {_rgb[1]}, {_rgb[2]}, 0.4)"
            train_label = f"{cat_col} (Train)"
            _show = show_legend if seen_legend is None else (train_label not in seen_legend and show_legend)
            if seen_legend is not None:
                seen_legend.add(train_label)
            fig.add_trace(
                go.Scatter(
                    x=train_df["time"],
                    y=y_vals,
                    mode="lines+markers",
                    line={"color": _train_c, "width": line_width, "shape": "hv"},
                    marker={"size": 4},
                    name=train_label,
                    showlegend=_show,
                    legendrank=0,
                    text=[str(v) for v in train_df[cat_col].to_list()],
                    hovertemplate=f"<b>{cat_col} Train</b><br>Time: %{{x}}<br>Class: %{{text}}<extra></extra>",
                ),
                **_add,
            )

        # Actual test data
        if cat_col in y_test.columns:
            y_vals = [cat_to_int.get(str(v), -1) for v in y_test[cat_col].to_list()]
            actual_label = f"{cat_col} (Actual)"
            _show = show_legend if seen_legend is None else (actual_label not in seen_legend and show_legend)
            if seen_legend is not None:
                seen_legend.add(actual_label)
            fig.add_trace(
                go.Scatter(
                    x=y_test["time"],
                    y=y_vals,
                    mode="lines+markers",
                    line={"color": actual_color, "width": line_width, "shape": "hv"},
                    marker={"size": 6},
                    name=actual_label,
                    showlegend=_show,
                    legendrank=1,
                    text=[str(v) for v in y_test[cat_col].to_list()],
                    hovertemplate=f"<b>{cat_col} Actual</b><br>Time: %{{x}}<br>Class: %{{text}}<extra></extra>",
                ),
                **_add,
            )

        # Prediction
        if cat_col not in pred_df.columns:
            continue
        y_vals = [cat_to_int.get(str(v), -1) for v in pred_df[cat_col].to_list()]
        fc_name = f"{cat_col} ({model_name})" if is_multi_model else f"{cat_col} (Forecast)"
        _show = show_legend if seen_legend is None else (fc_name not in seen_legend and show_legend)
        if seen_legend is not None:
            seen_legend.add(fc_name)
        fig.add_trace(
            go.Scatter(
                x=pred_df["time"],
                y=y_vals,
                mode="lines+markers",
                line={"color": model_color, "width": line_width, "shape": "hv", "dash": "dash"},
                marker={"size": 5, "symbol": "square"},
                name=fc_name,
                showlegend=_show,
                legendrank=10,
                text=[str(v) for v in pred_df[cat_col].to_list()],
                hovertemplate=f"<b>{fc_name}</b><br>Time: %{{x}}<br>Class: %{{text}}<extra></extra>",
            ),
            **_add,
        )


def _collect_categories(
    cat_cols: list[str],
    y_test: pl.DataFrame,
    preds: dict[str, pl.DataFrame],
    y_train: pl.DataFrame | None,
) -> tuple[list[str], dict[str, int]]:
    """Collect and sort all unique categories across data sources.

    Parameters
    ----------
    cat_cols : list of str
        Categorical column names.
    y_test : pl.DataFrame
        Test data.
    preds : dict of str to pl.DataFrame
        Model predictions.
    y_train : pl.DataFrame or None
        Training data.

    Returns
    -------
    sorted_cats : list of str
        Sorted unique category names.
    cat_to_int : dict of str to int
        Category to integer mapping.

    """
    series_list: list[pl.Series] = []
    for cat_col in cat_cols:
        if cat_col in y_test.columns:
            series_list.append(y_test[cat_col].cast(pl.String))
        for pred_df in preds.values():
            if cat_col in pred_df.columns:
                series_list.append(pred_df[cat_col].cast(pl.String))
        if y_train is not None and cat_col in y_train.columns:
            series_list.append(y_train[cat_col].cast(pl.String))
    return build_category_map(*series_list)


def _extract_member_df(df: pl.DataFrame, suffix: str) -> pl.DataFrame:
    """Extract columns for a panel member and strip the panel group prefix.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with panel columns (``prefix__suffix`` naming).
    suffix : str
        Panel member suffix to extract.

    Returns
    -------
    pl.DataFrame
        DataFrame with ``"time"`` and unprefixed columns for the member.

    """
    cols: dict[str, pl.Series] = {"time": df["time"]}
    if "observed_time" in df.columns:
        cols["observed_time"] = df["observed_time"]
    for c in df.columns:
        if c in ("time", "observed_time"):
            continue
        prefix, sep, member = c.partition("__")
        if sep and member == suffix:
            cols[prefix] = df[c]
    return pl.DataFrame(cols)


def _plot_forecast_class_proba(
    y_test: pl.DataFrame,
    y_pred: pl.DataFrame | dict[str, pl.DataFrame],
    *,
    columns: str | list[str] | None,
    color_palette: list[str] | None,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    width: int | None,
    height: int | None,
    facet_n_cols: int,
    **kwargs: object,
) -> go.Figure:
    """Plot class-probability forecasts as stacked area charts with truth markers.

    Parameters
    ----------
    y_test : pl.DataFrame
        Ground-truth DataFrame with ``"time"`` and categorical target columns.
    y_pred : pl.DataFrame or dict of str to pl.DataFrame
        Predictions with ``{target}_proba_{class}`` columns.
    columns : str or list of str or None
        Target name filter.
    color_palette : list of str or None
        Custom color palette.
    title : str or None
        Plot title.
    x_label : str or None
        X-axis label.
    y_label : str or None
        Y-axis label.
    width : int or None
        Plot width in pixels.
    height : int or None
        Plot height in pixels.
    facet_n_cols : int
        Number of facet columns.
    **kwargs : object
        ``line_width``, ``band_opacity``, ``marker_size``.

    Returns
    -------
    go.Figure
        Plotly figure with stacked area chart.

    """
    line_width = kwargs.get("line_width", 1.5)
    band_opacity = kwargs.get("band_opacity", 0.6)
    marker_size = kwargs.get("marker_size", 10)

    preds = y_pred if isinstance(y_pred, dict) else {"Forecast": y_pred}
    first_pred = next(iter(preds.values()))

    proba_cols = [c for c in first_pred.columns if "_proba_" in c]  # ty: ignore[unresolved-attribute]
    all_targets = _discover_proba_targets(proba_cols)

    # Filter targets by columns parameter (Decision #10)
    if columns is not None:
        col_list = [columns] if isinstance(columns, str) else list(columns)
        all_targets = {t: cols for t, cols in all_targets.items() if t in col_list}

    target_names = list(all_targets.keys())

    def _render_class_proba(ctx: RenderContext) -> None:
        """Render stacked area + truth markers for one target across models."""
        target = ctx.display_name
        target_proba_cols = all_targets[target]
        class_labels = [c.split("_proba_", 1)[1] for c in target_proba_cols]
        colors = resolve_color_palette(color_palette, n=len(class_labels))

        for model_idx, (model_name, pred_df) in enumerate(preds.items()):
            _render_class_proba_traces(
                ctx.fig,
                pred_df,  # ty: ignore[invalid-argument-type]
                y_test,
                colors=colors,
                target_proba_cols=target_proba_cols,
                class_labels=class_labels,
                target=target,
                stack_group=f"proba_{model_name}_{target}",
                row=ctx.row,
                col=ctx.col,
                show_legend=model_idx == 0,
                line_width=line_width,  # ty: ignore[invalid-argument-type]
                band_opacity=band_opacity,  # ty: ignore[invalid-argument-type]
                marker_size=marker_size,  # ty: ignore[invalid-argument-type]
            )

    fig = facet_figure(
        y_test,
        _render_class_proba,
        columns=target_names,
        facet_n_cols=facet_n_cols,
        title=title or "Class Probability Forecast",
        x_label=x_label or "Time",
        y_label=y_label or "Probability",
        width=width,
        height=height,
    )
    return fig


def _plot_forecast_categorical(
    y_test: pl.DataFrame,
    y_pred: pl.DataFrame | dict[str, pl.DataFrame],
    *,
    y_train: pl.DataFrame | None,
    n_history: int | None,
    columns: str | list[str] | None,
    color_palette: list[str] | None,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    width: int | None,
    height: int | None,
    facet_n_cols: int,
    **kwargs: object,
) -> go.Figure:
    """Plot categorical forecasts as step charts with class labels on the y-axis.

    Parameters
    ----------
    y_test : pl.DataFrame
        Actual test values with ``"time"`` and categorical target columns.
    y_pred : pl.DataFrame or dict of str to pl.DataFrame
        Predicted class labels with the same target column names.
    y_train : pl.DataFrame or None
        Historical training data.
    n_history : int or None
        Number of historical observations to show.
    columns : str or list of str or None
        Columns to include.
    color_palette : list of str or None
        Custom color palette.
    title : str or None
        Plot title.
    x_label : str or None
        X-axis label.
    y_label : str or None
        Y-axis label.
    width : int or None
        Plot width in pixels.
    height : int or None
        Plot height in pixels.
    facet_n_cols : int
        Number of facet columns.
    **kwargs : object
        ``line_width``.

    Returns
    -------
    go.Figure
        Plotly figure with categorical step chart.

    """
    line_width = kwargs.get("line_width", 2.0)

    eff_palette = color_palette if color_palette is not None else _PALETTE
    actual_color = eff_palette[2 % len(eff_palette)]
    _model_pal = eff_palette[3:] or eff_palette

    preds = y_pred if isinstance(y_pred, dict) else {"Forecast": y_pred}
    model_names = list(preds.keys())
    model_colors = resolve_color_palette(_model_pal, len(model_names))
    first_pred = next(iter(preds.values()))

    cat_cols = [
        c
        for c in first_pred.columns  # ty: ignore[unresolved-attribute]
        if c not in ("time", "observed_time") and first_pred[c].dtype in (pl.String, pl.Categorical)  # ty: ignore[not-subscriptable]
    ]

    if columns is not None:
        col_list = [columns] if isinstance(columns, str) else list(columns)
        cat_cols = [c for c in col_list if c in cat_cols]

    sorted_cats, cat_to_int = _collect_categories(cat_cols, y_test, preds, y_train)  # ty: ignore[invalid-argument-type]

    def _render_categorical(ctx: RenderContext) -> None:
        """Render categorical step chart traces for one column across models."""
        for model_idx, (model_name, pred_df) in enumerate(preds.items()):
            _render_categorical_traces(
                ctx.fig,
                pred_df,  # ty: ignore[invalid-argument-type]
                y_test,
                y_train,
                cat_cols=[ctx.display_name],
                cat_to_int=cat_to_int,
                model_name=model_name,  # ty: ignore[invalid-argument-type]
                model_color=model_colors[model_idx],
                actual_color=actual_color,
                is_multi_model=len(model_names) > 1,
                n_history=n_history,
                row=ctx.row,
                col=ctx.col,
                show_legend=model_idx == 0 or len(model_names) > 1,
                line_width=line_width,  # ty: ignore[invalid-argument-type]
            )

    fig = facet_figure(
        y_test,
        _render_categorical,
        columns=cat_cols,
        facet_n_cols=facet_n_cols,
        title=title or "Categorical Forecast",
        x_label=x_label or "Time",
        y_label=y_label or "Class",
        width=width,
        height=height,
    )
    fig.update_yaxes(
        tickvals=list(range(len(sorted_cats))),
        ticktext=sorted_cats,
    )
    return fig


def _plot_forecast_multi_model(
    y_test: pl.DataFrame,
    y_preds: dict[str, pl.DataFrame],
    *,
    y_train: pl.DataFrame | None,
    coverage_rates: list[float] | None,
    n_history: int | None,
    columns: str | list[str] | None,
    color_palette: list[str] | None,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    width: int | None,
    height: int | None,
    facet_n_cols: int,
    line_width: float,
    band_opacity: float,
    show_transition: bool,
    connect_gaps: bool = False,
    resampler: bool | Literal["widget"] | None = None,
    show_legend: bool = True,
) -> go.Figure:
    """Plot multiple model forecasts overlaid on the same axes.

    Parameters
    ----------
    y_test : pl.DataFrame
        Actual test values.
    y_preds : dict[str, pl.DataFrame]
        Model name to prediction DataFrame mapping.
    y_train : pl.DataFrame | None
        Historical data.
    coverage_rates : list[float] | None
        Coverage rates for intervals.
    n_history : int | None
        Number of history points to show.
    columns : str | list[str] | None
        Columns to include.
    color_palette : list[str] | None
        Color palette. Slots 0/1/2 = history/forecast/actual; 3+ = models.
    title : str | None
        Plot title.
    x_label : str | None
        X-axis label.
    y_label : str | None
        Y-axis label.
    width : int | None
        Plot width.
    height : int | None
        Plot height.
    facet_n_cols : int
        Number of facet columns.
    line_width : float
        Line width.
    band_opacity : float
        Interval fill opacity.
    show_transition : bool
        Whether to connect train to forecast.

    Returns
    -------
    go.Figure
        Plotly figure with overlaid model forecasts.

    """
    eff_palette = color_palette if color_palette is not None else _PALETTE
    actual_color = eff_palette[2 % len(eff_palette)]
    _model_pal = eff_palette[3:] or eff_palette

    interval_pattern = re.compile(r"^.+_(lower|upper)_[\d.]+$")
    test_value_cols = [c for c in y_test.columns if c != "time"]

    if columns is not None:
        col_list = [columns] if isinstance(columns, str) else list(columns)
        test_value_cols = [c for c in col_list if c in test_value_cols]
    plot_columns = test_value_cols

    model_names = list(y_preds.keys())
    colors = resolve_color_palette(_model_pal, len(model_names))

    def _render_multi_model(ctx: RenderContext) -> None:
        """Render train/intervals/actual/forecast for one target column across models."""
        col = ctx.display_name

        # Training data
        if y_train is not None and col in y_train.columns:
            train_df = y_train.tail(n_history) if n_history is not None else y_train
            _hex = actual_color.lstrip("#")
            _rgb = tuple(int(_hex[i : i + 2], 16) for i in (0, 2, 4))
            _train_color = f"rgba({_rgb[0]}, {_rgb[1]}, {_rgb[2]}, 0.4)"
            ctx.fig.add_trace(
                go.Scatter(
                    x=train_df["time"],
                    y=train_df[col],
                    mode="lines",
                    line={"color": _train_color, "width": line_width},
                    connectgaps=connect_gaps,
                    name=f"{col} (Train)",
                    legendrank=0,
                    hovertemplate=_make_hovertemplate("Train", "Time", "Value"),
                ),
                row=ctx.row,
                col=ctx.col,
            )

        # Interval bands per model (behind everything else)
        for model_idx, (model_name, y_pred) in enumerate(y_preds.items()):
            model_color = colors[model_idx % len(colors)]
            pred_value_cols = [
                c for c in y_pred.columns if c not in ("time", "observed_time") and not interval_pattern.match(c)
            ]
            pred_col = col if col in pred_value_cols else (pred_value_cols[0] if pred_value_cols else None)
            interval_base = pred_col if (pred_col is not None and pred_col in y_pred.columns) else col
            if not coverage_rates:
                continue
            for sort_idx, rate in enumerate(sorted(coverage_rates)):
                lower_col = f"{interval_base}_lower_{rate}"
                upper_col = f"{interval_base}_upper_{rate}"
                if lower_col in y_pred.columns and upper_col in y_pred.columns:
                    rgb = tuple(int(model_color[i : i + 2], 16) for i in (1, 3, 5))
                    rgba = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {band_opacity})"
                    t = y_pred["time"].to_list()
                    y_upper = y_pred[upper_col].to_list()
                    y_lower = y_pred[lower_col].to_list()
                    x_band = t + t[::-1]
                    y_band = y_upper + y_lower[::-1]
                    ctx.fig.add_trace(
                        go.Scatter(
                            x=x_band,
                            y=y_band,
                            fill="toself",
                            fillcolor=rgba,
                            mode="lines",
                            line={"width": 0, "color": rgba},
                            name=f"{model_name} ({rate:.0%} PI)",
                            legendgroup=model_name,
                            legendrank=10 + model_idx * 100 + sort_idx + 1,
                            hoverinfo="skip",
                        ),
                        row=ctx.row,
                        col=ctx.col,
                    )

        # Actual test data (prepend last train point to close the gap)
        _x_actual = y_test["time"]
        _y_actual = y_test[col]
        if y_train is not None and col in y_train.columns:
            _x_actual = pl.concat([pl.Series("time", [y_train["time"][-1]]), _x_actual])
            _y_actual = pl.concat([pl.Series([y_train[col][-1]], dtype=_y_actual.dtype), _y_actual])
        ctx.fig.add_trace(
            go.Scatter(
                x=_x_actual,
                y=_y_actual,
                mode="lines",
                line={"color": actual_color, "width": line_width},
                connectgaps=connect_gaps,
                name=f"{col} (Actual)",
                legendrank=1,
                hovertemplate=_make_hovertemplate("Actual", "Time", "Value"),
            ),
            row=ctx.row,
            col=ctx.col,
        )

        # Forecast lines per model (on top)
        for model_idx, (model_name, y_pred) in enumerate(y_preds.items()):
            model_color = colors[model_idx % len(colors)]
            pred_value_cols = [
                c for c in y_pred.columns if c not in ("time", "observed_time") and not interval_pattern.match(c)
            ]
            pred_col = col if col in pred_value_cols else (pred_value_cols[0] if pred_value_cols else None)
            if pred_col is None or pred_col not in y_pred.columns:
                continue
            x_fc = y_pred["time"]
            y_fc = y_pred[pred_col]
            if show_transition and y_train is not None and col in y_train.columns:
                x_fc = pl.concat([pl.Series("time", [y_train["time"][-1]]), y_pred["time"]])
                y_fc = pl.concat([pl.Series([y_train[col][-1]], dtype=y_fc.dtype), y_fc])
            _fc_name = f"{col} ({model_name})"
            ctx.fig.add_trace(
                go.Scatter(
                    x=x_fc,
                    y=y_fc,
                    mode="lines",
                    line={"color": model_color, "width": line_width},
                    connectgaps=connect_gaps,
                    name=_fc_name,
                    legendgroup=model_name,
                    legendrank=10 + model_idx * 100,
                    hovertemplate=_make_hovertemplate(_fc_name, "Time", "Value"),
                ),
                row=ctx.row,
                col=ctx.col,
            )

    fig = facet_figure(
        y_test,
        _render_multi_model,
        columns=plot_columns,
        facet_n_cols=facet_n_cols,
        title=title or "Forecast Comparison",
        x_label=x_label or "Time",
        y_label=y_label or "Value",
        width=width,
        height=height,
        resampler=resampler,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


def _render_interval_bands(
    fig,
    model_preds,
    model_colors,
    col,
    interval_pattern,
    coverage_rates,
    multi_member,
    is_multi_model,
    base_color,
    member,
    band_opacity,
    legend_tracker,
    lg_key,
    row,
    col_grid,
):
    """Add prediction-interval bands to the forecast figure."""
    for m_idx, (m_name, m_pred) in enumerate(model_preds.items()):
        m_color = model_colors[m_idx % len(model_colors)]
        pred_value_cols = [
            c for c in m_pred.columns if c not in ("time", "observed_time") and not interval_pattern.match(c)
        ]
        pred_col = col if col in pred_value_cols else None
        interval_base = pred_col if pred_col is not None else col
        if not coverage_rates:
            continue
        band_c = base_color if (multi_member and not is_multi_model) else m_color
        _hex_b = band_c.lstrip("#")
        rgb = tuple(int(_hex_b[i : i + 2], 16) for i in (0, 2, 4))
        for sort_idx, rate in enumerate(sorted(coverage_rates)):
            lower_c = f"{interval_base}_lower_{rate}"
            upper_c = f"{interval_base}_upper_{rate}"
            if lower_c in m_pred.columns and upper_c in m_pred.columns:
                rgba = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {band_opacity})"
                t = m_pred["time"].to_list()
                y_upper = m_pred[upper_c].to_list()
                y_lower = m_pred[lower_c].to_list()
                x_band = t + t[::-1]
                y_band = y_upper + y_lower[::-1]
                if multi_member:
                    pi_label = f"{member} {m_name} ({rate:.0%} PI)" if is_multi_model else f"{member} ({rate:.0%} PI)"
                else:
                    pi_label = f"{rate:.0%} PI" if not is_multi_model else f"{m_name} ({rate:.0%} PI)"
                _show_pi = legend_tracker.should_show(pi_label)
                fig.add_trace(
                    go.Scatter(
                        x=x_band,
                        y=y_band,
                        fill="toself",
                        fillcolor=rgba,
                        mode="lines",
                        line={"width": 0, "color": rgba},
                        name=pi_label,
                        showlegend=_show_pi,
                        legendgroup=lg_key or m_name,
                        legendrank=10 + m_idx * 100 + sort_idx + 1,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col_grid,
                    **_fill_trace_kwargs(fig),
                )


def _render_forecast_trace(
    fig,
    model_preds,
    model_colors,
    col,
    interval_pattern,
    coverage_rates,
    multi_member,
    is_multi_model,
    base_color,
    member,
    line_width,
    connect_gaps,
    show_transition,
    y_train,
    legend_tracker,
    lg_key,
    row,
    col_grid,
):
    """Add forecast line traces to the figure."""
    for m_idx, (m_name, m_pred) in enumerate(model_preds.items()):
        m_color = model_colors[m_idx % len(model_colors)]
        pred_value_cols = [
            c for c in m_pred.columns if c not in ("time", "observed_time") and not interval_pattern.match(c)
        ]
        pred_col = col if col in pred_value_cols else None

        has_point = pred_col is not None
        interval_base = pred_col if has_point else col
        has_intervals = bool(
            coverage_rates and any(f"{interval_base}_lower_{rate}" in m_pred.columns for rate in coverage_rates)
        )

        if not has_point and not has_intervals:
            continue

        if has_point:
            assert pred_col is not None
            x_fc = m_pred["time"]
            y_fc = m_pred[pred_col]
            if show_transition and y_train is not None and col in y_train.columns:
                x_fc = pl.concat([pl.Series("time", [y_train["time"][-1]]), m_pred["time"]])
                y_fc = pl.concat([pl.Series([y_train[col][-1]], dtype=y_fc.dtype), y_fc])

            if multi_member:
                fc_color = base_color if not is_multi_model else m_color
                fc_label = f"{member} (Forecast)" if not is_multi_model else f"{member} ({m_name})"
                fc_dash: str | None = "dash"
            else:
                fc_color = m_color
                fc_label = m_name
                fc_dash = None

            line_spec: dict = {"color": fc_color, "width": line_width}
            if fc_dash:
                line_spec["dash"] = fc_dash

            _show_fc = legend_tracker.should_show(fc_label)
            fig.add_trace(
                go.Scatter(
                    x=x_fc,
                    y=y_fc,
                    mode="lines",
                    line=line_spec,
                    connectgaps=connect_gaps,
                    name=fc_label,
                    showlegend=_show_fc,
                    legendgroup=lg_key or m_name,
                    legendrank=10 + m_idx * 100,
                ),
                row=row,
                col=col_grid,
            )


def _plot_forecast_panel_typed(
    y_test: pl.DataFrame,
    y_pred: pl.DataFrame | dict[str, pl.DataFrame],
    *,
    prediction_mode: str,
    y_train: pl.DataFrame | None,
    n_history: int | None,
    panel_group_names: list[str] | None,
    facet_n_cols: int,
    color_palette: list[str] | None,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    width: int | None,
    height: int | None,
    line_width: float,
    band_opacity: float,
) -> go.Figure:
    """Panel plot for class-probability or categorical predictions.

    Parameters
    ----------
    y_test : pl.DataFrame
        Actual test values with panel columns.
    y_pred : pl.DataFrame or dict of str to pl.DataFrame
        Predictions with panel columns.
    prediction_mode : str
        ``"class_proba"`` or ``"categorical"``.
    y_train : pl.DataFrame or None
        Historical training data.
    n_history : int or None
        Number of training rows to show.
    panel_group_names : list of str or None
        Panel group prefixes to include.
    facet_n_cols : int
        Columns in facet grid.
    color_palette : list of str or None
        Custom color palette.
    title : str or None
        Plot title.
    x_label : str or None
        X-axis label.
    y_label : str or None
        Y-axis label.
    width : int or None
        Plot width in pixels.
    height : int or None
        Plot height in pixels.
    line_width : float
        Line width.
    band_opacity : float
        Fill opacity.

    Returns
    -------
    go.Figure
        Plotly figure with faceted subplots.

    """
    eff_palette = color_palette if color_palette is not None else _PALETTE
    actual_color = eff_palette[2 % len(eff_palette)]
    _model_pal = eff_palette[3:] or eff_palette

    is_multi_model = isinstance(y_pred, dict)
    if is_multi_model:
        model_preds: dict[str, pl.DataFrame] = y_pred  # ty: ignore[invalid-assignment]
        model_names = list(model_preds.keys())
        model_colors = resolve_color_palette(_model_pal, len(model_names))
    else:
        model_preds = {"Forecast": y_pred}
        model_names = ["Forecast"]
        model_colors = [eff_palette[1 % len(eff_palette)]]

    # Collect unique panel member suffixes from test data
    suffix_by_prefix: dict[str, set[str]] = {}
    for c in y_test.columns:
        if c in ("time", "observed_time"):
            continue
        prefix, sep, member = c.partition("__")
        if sep:
            suffix_by_prefix.setdefault(prefix, set()).add(member)

    # Apply panel_group_names filter (filters by prefix)
    if panel_group_names is not None:
        suffix_by_prefix = {k: v for k, v in suffix_by_prefix.items() if k in panel_group_names}

    all_suffixes: set[str] = set()
    for members in suffix_by_prefix.values():
        all_suffixes |= members
    suffixes_list = sorted(all_suffixes)

    if not suffixes_list:
        msg = f"No panel members found for groups: {panel_group_names}"
        raise ValueError(msg)

    # Create subplots
    n_panels = len(suffixes_list)
    n_rows = (n_panels + facet_n_cols - 1) // facet_n_cols
    n_cols_grid = min(n_panels, facet_n_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols_grid,
        subplot_titles=suffixes_list,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    seen_legend: set[str] = set()

    # Pre-compute category mapping for categorical mode
    sorted_cats: list[str] = []
    cat_to_int: dict[str, int] = {}
    if prediction_mode == "categorical":
        all_cats: set[str] = set()
        for suffix in suffixes_list:
            test_m = _extract_member_df(y_test, suffix)
            train_m = _extract_member_df(y_train, suffix) if y_train is not None else None
            for pred_df in model_preds.values():
                pred_m = _extract_member_df(pred_df, suffix)  # ty: ignore[invalid-argument-type]
                for cc in pred_m.columns:
                    if cc in ("time", "observed_time"):
                        continue
                    if pred_m[cc].dtype in (pl.String, pl.Categorical):
                        all_cats.update(pred_m[cc].cast(pl.String).unique().to_list())
            for cc in test_m.columns:
                if cc in ("time", "observed_time"):
                    continue
                if test_m[cc].dtype in (pl.String, pl.Categorical):
                    all_cats.update(test_m[cc].cast(pl.String).unique().to_list())
            if train_m is not None:
                for cc in train_m.columns:
                    if cc in ("time", "observed_time"):
                        continue
                    if train_m[cc].dtype in (pl.String, pl.Categorical):
                        all_cats.update(train_m[cc].cast(pl.String).unique().to_list())
        sorted_cats = sorted(all_cats)
        cat_to_int = {cat: i for i, cat in enumerate(sorted_cats)}

    for suffix_idx, suffix in enumerate(suffixes_list):
        row = suffix_idx // facet_n_cols + 1
        col_grid = suffix_idx % facet_n_cols + 1

        test_member = _extract_member_df(y_test, suffix)
        train_member = _extract_member_df(y_train, suffix) if y_train is not None else None

        if prediction_mode == "class_proba":
            for model_idx, (model_name, pred_df) in enumerate(model_preds.items()):
                pred_member = _extract_member_df(pred_df, suffix)  # ty: ignore[invalid-argument-type]
                proba_cols = [c for c in pred_member.columns if "_proba_" in c]
                if not proba_cols:
                    continue
                targets = _discover_proba_targets(proba_cols)
                target = next(iter(targets))
                target_proba_cols = targets[target]
                class_labels = [c.split("_proba_", 1)[1] for c in target_proba_cols]
                colors = resolve_color_palette(color_palette, n=len(class_labels))

                _render_class_proba_traces(
                    fig,
                    pred_member,
                    test_member,
                    colors=colors,
                    target_proba_cols=target_proba_cols,
                    class_labels=class_labels,
                    target=target,
                    stack_group=f"proba_{model_name}_{suffix}",
                    row=row,
                    col=col_grid,
                    show_legend=model_idx == 0,
                    line_width=line_width,
                    band_opacity=band_opacity,
                    marker_size=10,
                    seen_legend=seen_legend,
                )

        elif prediction_mode == "categorical":
            first_pred_member = _extract_member_df(
                next(iter(model_preds.values())),  # ty: ignore[invalid-argument-type]
                suffix,
            )
            cat_cols = [
                c
                for c in first_pred_member.columns
                if c not in ("time", "observed_time") and first_pred_member[c].dtype in (pl.String, pl.Categorical)
            ]

            for model_idx, (model_name, pred_df) in enumerate(model_preds.items()):
                pred_member = _extract_member_df(pred_df, suffix)  # ty: ignore[invalid-argument-type]
                _render_categorical_traces(
                    fig,
                    pred_member,
                    test_member,
                    train_member,
                    cat_cols=cat_cols,
                    cat_to_int=cat_to_int,
                    model_name=model_name,
                    model_color=model_colors[model_idx],
                    actual_color=actual_color,
                    is_multi_model=is_multi_model,
                    n_history=n_history,
                    row=row,
                    col=col_grid,
                    show_legend=True,
                    line_width=line_width,
                    seen_legend=seen_legend,
                )

    default_title = "Class Probability Forecast" if prediction_mode == "class_proba" else "Categorical Forecast"
    default_y = "Probability" if prediction_mode == "class_proba" else "Class"

    fig = apply_default_layout(
        fig,
        title=title or (default_title + (" Comparison" if is_multi_model else "")),
        width=width,
        height=height or (300 * n_rows),
    )

    for c in range(1, n_cols_grid + 1):
        fig.update_xaxes(title_text=x_label or "Time", row=n_rows, col=c)
    fig.update_yaxes(title_text=y_label or default_y)

    if prediction_mode == "categorical" and sorted_cats:
        for suffix_idx in range(n_panels):
            fig.update_yaxes(
                tickvals=list(range(len(sorted_cats))),
                ticktext=sorted_cats,
                row=suffix_idx // facet_n_cols + 1,
                col=suffix_idx % facet_n_cols + 1,
            )

    return fig


def _plot_forecast_panel(
    y_test: pl.DataFrame,
    y_pred: pl.DataFrame | dict[str, pl.DataFrame],
    *,
    y_train: pl.DataFrame | None,
    coverage_rates: list[float] | None,
    n_history: int | None,
    panel_group_names: list[str] | None,
    facet_by: Literal["group", "member"] | None,
    facet_n_cols: int,
    color_palette: list[str] | None,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    width: int | None,
    height: int | None,
    line_width: float,
    band_opacity: float,
    show_transition: bool,
    connect_gaps: bool = False,
    resampler: bool | Literal["widget"] | None = None,
    show_legend: bool = True,
    prediction_mode: str = "numeric",
) -> go.Figure:
    """Plot forecast with panel data as faceted subplots.

    Supports both single-model (DataFrame) and multi-model (dict) predictions.
    Each subplot receives its own legend positioned inside the subplot area.
    The ``facet_by`` parameter determines the subplot axis:

    * ``"group"`` - one subplot per panel group, members overlaid by colour.
    * ``"member"`` - one subplot per member, groups overlaid by colour.
    * ``None`` - all series in a single subplot.

    Parameters
    ----------
    y_test : pl.DataFrame
        Actual test values with panel columns.
    y_pred : pl.DataFrame | dict[str, pl.DataFrame]
        Forecast values with panel columns, or dict mapping model names to
        prediction DataFrames for multi-model comparison.
    y_train : pl.DataFrame | None
        Historical data.
    coverage_rates : list[float] | None
        Coverage rates for intervals.
    n_history : int | None
        Number of history points to show.
    panel_group_names : list[str] | None
        Groups to plot.
    facet_by : Literal["group", "member"] | None
        Controls subplot organisation.
    facet_n_cols : int
        Columns in facet grid.
    color_palette : list[str] | None
        Color palette. Slots 0/1/2 = history/forecast/actual; 3+ = models.
    title : str | None
        Plot title.
    x_label : str | None
        X-axis label.
    y_label : str | None
        Y-axis label.
    width : int | None
        Plot width.
    height : int | None
        Plot height.
    line_width : float
        Line width.
    band_opacity : float
        Interval fill opacity.
    show_transition : bool
        Whether to connect train to forecast.
    prediction_mode : str
        One of ``"numeric"``, ``"class_proba"``, ``"categorical"``.

    Returns
    -------
    go.Figure
        Plotly figure with faceted panel subplots.

    """
    if prediction_mode != "numeric":
        return _plot_forecast_panel_typed(
            y_test,
            y_pred,
            prediction_mode=prediction_mode,
            y_train=y_train,
            n_history=n_history,
            panel_group_names=panel_group_names,
            facet_n_cols=facet_n_cols,
            color_palette=color_palette,
            title=title,
            x_label=x_label,
            y_label=y_label,
            width=width,
            height=height,
            line_width=line_width,
            band_opacity=band_opacity,
        )
    eff_palette = color_palette if color_palette is not None else _PALETTE
    history_color = eff_palette[0]
    actual_color = eff_palette[2 % len(eff_palette)]
    forecast_color = eff_palette[1 % len(eff_palette)]
    _model_pal = eff_palette[3:] or eff_palette

    _, test_panels = inspect_panel(y_test)

    # Group panel columns by group prefix
    groups: dict[str, list[str]] = {}
    for prefix, cols in test_panels.items():
        if panel_group_names is None or prefix in panel_group_names:
            groups.setdefault(prefix, []).extend(cols)

    if not groups:
        msg = f"No panel columns found for groups: {panel_group_names}"
        raise ValueError(msg)

    all_flat_cols = [c for cols in groups.values() for c in cols]
    _, all_members = _group_panel_columns(all_flat_cols)
    all_group_names = list(groups.keys())

    # Build faceting structure
    # facets: dict from subplot label -> list of columns in that subplot
    # sub_names: the distinct sub-identifiers used for coloring within facets
    # get_sub_name(col) -> the sub-identifier for a column
    if facet_by == "member":
        facets: dict[str, list[str]] = {}
        for member in all_members:
            cols_for_member = []
            for gcols in groups.values():
                col = next((c for c in gcols if _member_name(c) == member), None)
                if col:
                    cols_for_member.append(col)
            if cols_for_member:
                facets[member] = cols_for_member
        sub_names = all_group_names
        sub_palette = resolve_color_palette(_model_pal, len(sub_names))

        def _get_sub_name(col: str) -> str:
            """Return group name from panel column."""
            return col.split("__", 1)[0]

        def _get_sub_color(col: str) -> str:
            """Return palette color for the column."""
            return sub_palette[sub_names.index(_get_sub_name(col))]

    elif facet_by is None:
        facets = {"All": all_flat_cols}
        # Each column is its own sub-identifier
        sub_names = [_member_name(c) if "__" in c else c for c in all_flat_cols]
        sub_palette = resolve_color_palette(_model_pal, len(all_flat_cols))
        _sub_col_map = dict(zip(all_flat_cols, range(len(all_flat_cols)), strict=False))

        def _get_sub_name(col: str) -> str:
            """Return composite group/member identifier."""
            gname = col.split("__", 1)[0] if "__" in col else ""
            mname = _member_name(col) if "__" in col else col
            return f"{gname}/{mname}" if gname else mname

        def _get_sub_color(col: str) -> str:
            """Return palette color for the column."""
            return sub_palette[_sub_col_map.get(col, 0)]

    else:
        # facet_by == "group" (default)
        facets = groups
        sub_names = all_members
        sub_palette = resolve_color_palette(_model_pal, len(sub_names))

        def _get_sub_name(col: str) -> str:
            """Return member name from panel column."""
            return _member_name(col)

        def _get_sub_color(col: str) -> str:
            """Return palette color for the column."""
            return sub_palette[sub_names.index(_get_sub_name(col))]

    multi_sub = any(len(cols) > 1 for cols in facets.values())

    # Normalise y_pred into a model-name -> DataFrame mapping
    is_multi_model = isinstance(y_pred, dict)
    if is_multi_model:
        assert isinstance(y_pred, dict)
        model_preds: dict[str, pl.DataFrame] = y_pred  # ty: ignore[invalid-assignment]
        model_names = list(model_preds.keys())
        model_colors = resolve_color_palette(_model_pal, len(model_names))
    else:
        model_preds = {"Forecast": y_pred}
        model_names = ["Forecast"]
        model_colors = [forecast_color]

    n_facets = len(facets)
    n_rows = (n_facets + facet_n_cols - 1) // facet_n_cols
    n_cols_grid = min(n_facets, facet_n_cols)

    fig = _create_subplots(
        resampler,
        rows=n_rows,
        cols=n_cols_grid,
        subplot_titles=list(facets.keys()),
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    interval_pattern = re.compile(r"^.+_(lower|upper)_[\d.]+$")
    legend_tracker = LegendTracker()

    for facet_idx, (_, facet_cols) in enumerate(facets.items()):
        row = facet_idx // facet_n_cols + 1
        col_grid = facet_idx % facet_n_cols + 1

        for col in facet_cols:
            sub_name = _get_sub_name(col)
            base_color = _get_sub_color(col)

            # Resolve per-trace colours and labels
            if multi_sub:
                _hex = base_color.lstrip("#")
                _rgb = tuple(int(_hex[i : i + 2], 16) for i in (0, 2, 4))
                train_c: str = f"rgba({_rgb[0]}, {_rgb[1]}, {_rgb[2]}, 0.4)"
                actual_c = base_color
                train_label = f"{sub_name} (Train)"
                actual_label = f"{sub_name} (Actual)"
                lg_key: str | None = sub_name
            else:
                train_c = history_color
                actual_c = actual_color
                train_label = "Train"
                actual_label = "Actual"
                lg_key = None

            # Train
            if y_train is not None and col in y_train.columns:
                train_df = y_train.tail(n_history) if n_history is not None else y_train
                _show = legend_tracker.should_show(train_label)
                fig.add_trace(
                    go.Scatter(
                        x=train_df["time"],
                        y=train_df[col],
                        mode="lines",
                        line={"color": train_c, "width": line_width},
                        connectgaps=connect_gaps,
                        name=train_label,
                        showlegend=_show,
                        legendgroup=lg_key or "train",
                        legendrank=0,
                    ),
                    row=row,
                    col=col_grid,
                )

            # Interval bands first (behind Actual and forecast lines).
            _render_interval_bands(
                fig=fig,
                model_preds=model_preds,
                model_colors=model_colors,
                col=col,
                interval_pattern=interval_pattern,
                coverage_rates=coverage_rates,
                multi_member=multi_sub,
                is_multi_model=is_multi_model,
                base_color=base_color if multi_sub else None,
                member=sub_name,
                band_opacity=band_opacity,
                legend_tracker=legend_tracker,
                lg_key=lg_key,
                row=row,
                col_grid=col_grid,
            )

            # Actual
            if col in y_test.columns:
                _show_actual = legend_tracker.should_show(actual_label)
                fig.add_trace(
                    go.Scatter(
                        x=y_test["time"],
                        y=y_test[col],
                        mode="lines",
                        line={"color": actual_c, "width": line_width},
                        connectgaps=connect_gaps,
                        name=actual_label,
                        showlegend=_show_actual,
                        legendgroup=lg_key or "actual",
                        legendrank=1,
                    ),
                    row=row,
                    col=col_grid,
                )

            # Forecast lines on top of bands and actual.
            _render_forecast_trace(
                fig=fig,
                model_preds=model_preds,
                model_colors=model_colors,
                col=col,
                interval_pattern=interval_pattern,
                coverage_rates=coverage_rates,
                multi_member=multi_sub,
                is_multi_model=is_multi_model,
                base_color=base_color if multi_sub else None,
                member=sub_name,
                line_width=line_width,
                connect_gaps=connect_gaps,
                show_transition=show_transition,
                y_train=y_train,
                legend_tracker=legend_tracker,
                lg_key=lg_key,
                row=row,
                col_grid=col_grid,
            )

    fig = apply_default_layout(
        fig,
        title=title or ("Forecast Comparison" if is_multi_model else "Forecast"),
        width=width,
        height=height or (300 * n_rows),
    )

    # X-axis title only on the bottom row (axes are shared above it).
    for c in range(1, n_cols_grid + 1):
        fig.update_xaxes(title_text=x_label or "Time", row=n_rows, col=c)
    fig.update_yaxes(title_text=y_label or "Value")
    fig.update_layout(showlegend=show_legend)

    return fig


def plot_time_weight(
    df: pl.DataFrame,
    *,
    weight_column: str = "time_weight",
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
    fill: bool = True,
    fill_opacity: float = 0.3,
) -> go.Figure:
    """
    Plot time-based weights as a time series visualization.

    Creates a plot showing how time weights vary over the time axis,
    useful for visualizing weighting schemes used in time-weighted forecasting
    or scoring (e.g., linear decay, exponential decay, recent-weighted).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and weight column.
    weight_column : str, default="time_weight"
        Name of the column containing weight values. For panel data,
        this is the base name (without the group prefix).
    panel_group_names : list[str] | None, default=None
        List of panel group prefixes to plot. If None, plots all groups.
        Creates subplots for each group.
    facet_by : Literal["group", "member"] | None, default="member"
        Faceting axis for panel data.  ``"group"`` creates one subplot per
        group, ``"member"`` one per member.  ``None`` disables faceting.
        Ignored for non-panel data.
    facet_n_cols : int, default=2
        Number of columns in facet grid when using panel groups.
    color_palette : list[str] | None, default=None
        Custom color palette as hex codes. If None, uses yohou palette.
    show_legend : bool, default=True
        Whether to show the legend.
    title : str | None, default=None
        Plot title. Defaults to "Time Weights".
    x_label : str | None, default=None
        X-axis label. Defaults to "Time".
    y_label : str | None, default=None
        Y-axis label. Defaults to "Weight".
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    connect_gaps : bool, default=False
        Whether to connect gaps in the data with lines.
    resampler : bool | Literal["widget"] | None, default=None
        Enable plotly-resampler for large datasets.  ``True`` or
        ``"widget"`` creates a ``FigureWidgetResampler``; ``False`` or
        ``None`` uses a plain ``go.Figure``.
    line_width : float, default=2.0
        Width of the weight line trace.
    fill : bool, default=True
        Whether to fill the area under the weight curve.
    fill_opacity : float, default=0.3
        Opacity of the filled area when ``fill`` is True.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    TypeError
        If df is not a Polars DataFrame.
    ValueError
        If DataFrame is empty, missing 'time' column, or weight_column not found.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_time_weight

    >>> # Create sample data with linear decay weights
    >>> n = 100
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
    ...     "time_weight": [(i + 1) / n for i in range(n)],
    ... })

    >>> # Plot time weights
    >>> fig = plot_time_weight(df)
    >>> len(fig.data) > 0
    True

    >>> # Panel data example
    >>> df_panel = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
    ...     "weight__store_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    ...     "weight__store_2": [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0],
    ... })
    >>> fig = plot_time_weight(df_panel, weight_column="weight")
    >>> len(fig.data) >= 2
    True

    See Also
    --------
    [`plot_time_series`][yohou.plotting.plot_time_series] : Plot basic time series.
    """
    # Validate inputs
    validate_plotting_data(df)
    validate_plotting_params(width=width, height=height)

    # Detect panel data structure
    _, panel_groups = inspect_panel(df)

    # Check if this is panel data with prefixed weight columns
    weight_panel_cols: dict[str, list[str]] = {}
    for group_prefix, cols in panel_groups.items():
        for col in cols:
            base_name = col.split("__", 1)[1] if "__" in col else col
            if weight_column in (base_name, col, group_prefix):
                if group_prefix not in weight_panel_cols:
                    weight_panel_cols[group_prefix] = []
                weight_panel_cols[group_prefix].append(col)

    # If we found panel weight columns
    if weight_panel_cols:
        # Filter to requested groups if specified
        if panel_group_names is not None:
            weight_panel_cols = {k: v for k, v in weight_panel_cols.items() if k in panel_group_names}

        if not weight_panel_cols:
            msg = f"No weight columns found for panel groups: {panel_group_names}"
            raise ValueError(msg)

        # Collect unique member names for consistent colouring
        flat_cols = [c for cols in weight_panel_cols.values() for c in cols]
        _, all_members = _group_panel_columns(flat_cols)
        all_group_names = list(weight_panel_cols.keys())

        # Build faceting structure
        if facet_by == "member":
            facets: dict[str, list[str]] = {}
            for member in all_members:
                cols_for_member = []
                for gcols in weight_panel_cols.values():
                    col = next((c for c in gcols if _member_name(c) == member), None)
                    if col:
                        cols_for_member.append(col)
                if cols_for_member:
                    facets[member] = cols_for_member
            sub_names = all_group_names

            def _get_sub_name(col: str) -> str:
                """Return group name from panel column."""
                return col.split("__", 1)[0]

        elif facet_by is None:
            facets = {"All": flat_cols}
            sub_names = [f"{c.split('__', 1)[0]}/{_member_name(c)}" if "__" in c else c for c in flat_cols]

            def _get_sub_name(col: str) -> str:
                """Return composite group/member identifier."""
                gname = col.split("__", 1)[0] if "__" in col else ""
                mname = _member_name(col) if "__" in col else col
                return f"{gname}/{mname}" if gname else mname

        else:
            # facet_by == "group" (default)
            facets = weight_panel_cols
            sub_names = all_members

            def _get_sub_name(col: str) -> str:
                """Return member name from panel column."""
                return _member_name(col)

        # Get colors (one per unique sub-identifier)
        sub_palette = resolve_color_palette(color_palette, len(sub_names))

        # Create subplots
        n_facets = len(facets)
        n_rows = (n_facets + facet_n_cols - 1) // facet_n_cols
        n_cols_grid = min(n_facets, facet_n_cols)
        fig = _create_subplots(
            resampler,
            rows=n_rows,
            cols=n_cols_grid,
            subplot_titles=list(facets.keys()),
            shared_xaxes=True,
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
        )

        seen_subs: set[str] = set()
        for facet_idx, (_, facet_cols) in enumerate(facets.items()):
            row = facet_idx // facet_n_cols + 1
            col_idx = facet_idx % facet_n_cols + 1

            for col in facet_cols:
                sub_name = _get_sub_name(col)
                s_idx = sub_names.index(sub_name)
                color = sub_palette[s_idx % len(sub_palette)]
                first_seen = sub_name not in seen_subs
                seen_subs.add(sub_name)

                # Convert hex to rgba for fill
                rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
                rgba_fill = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {fill_opacity})"

                fig.add_trace(
                    go.Scatter(
                        x=df["time"],
                        y=df[col],
                        mode="lines",
                        line={"color": color, "width": line_width},
                        connectgaps=connect_gaps,
                        fill="tozeroy" if fill else None,
                        fillcolor=rgba_fill if fill else None,
                        name=sub_name,
                        showlegend=first_seen,
                        legendgroup=sub_name,
                        hovertemplate=(f"{sub_name}<br>Time: %{{x}}<br>Weight: %{{y:.3f}}<extra></extra>"),
                    ),
                    row=row,
                    col=col_idx,
                    **((_fill_trace_kwargs(fig)) if fill else {}),
                )

        # Update layout
        title_default = title or "Time Weights by Panel"
        fig.update_layout(
            title=title_default,
            height=height or (300 * n_rows),
            width=width,
            showlegend=show_legend,
        )

        # Update axes labels
        for c in range(1, n_cols_grid + 1):
            fig.update_xaxes(title_text=x_label or "Time", row=n_rows, col=c)
        fig.update_yaxes(title_text=y_label or "Weight")

        return fig

    # Non-panel case: single weight column
    if weight_column not in df.columns:
        msg = f"Weight column '{weight_column}' not found in DataFrame. Available columns: {df.columns}"
        raise ValueError(msg)

    # Get colors
    if color_palette is None:
        color_palette = resolve_color_palette(None, 1)

    color = color_palette[0]

    # Convert hex to rgba for fill
    rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
    rgba_fill = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {fill_opacity})"

    # Create figure
    fig = _create_figure(resampler)

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df[weight_column],
            mode="lines",
            line={"color": color, "width": line_width},
            connectgaps=connect_gaps,
            fill="tozeroy" if fill else None,
            fillcolor=rgba_fill if fill else None,
            name="Time Weight",
            hovertemplate="Time: %{x}<br>Weight: %{y:.3f}<extra></extra>",
        ),
        **((_fill_trace_kwargs(fig)) if fill else {}),
    )

    # Set default labels
    title_default = title or "Time Weights"
    x_label_default = x_label or "Time"
    y_label_default = y_label or "Weight"

    fig = apply_default_layout(
        fig,
        title=title_default,
        x_label=x_label_default,
        y_label=y_label_default,
        width=width,
        height=height,
    )
    fig.update_layout(showlegend=show_legend)

    return fig


_INTERVAL_TO_STL_PERIOD: dict[str, int] = {
    "1m": 60,
    "5m": 12 * 24,
    "10m": 6 * 24,
    "15m": 4 * 24,
    "30m": 2 * 24,
    "1h": 24,
    "1d": 7,
    "7d": 52,
    "1mo": 12,
    "2mo": 6,
    "3mo": 4,
    "6mo": 2,
}

_INTERVAL_TO_MSTL_PERIODS: dict[str, list[int]] = {
    "1m": [60, 60 * 24],
    "5m": [12, 12 * 24],
    "10m": [6, 6 * 24],
    "15m": [4 * 24, 4 * 24 * 7],
    "30m": [2 * 24, 2 * 24 * 7],
    "1h": [24, 24 * 365],
    "1d": [7, 365],
    "7d": [4, 52],
    "1mo": [12],
}

_INTERVAL_HOURS: dict[str, float] = {
    "1m": 1 / 60,
    "1min": 1 / 60,
    "5m": 5 / 60,
    "5min": 5 / 60,
    "10m": 10 / 60,
    "10min": 10 / 60,
    "15m": 0.25,
    "15min": 0.25,
    "30m": 0.5,
    "30min": 0.5,
    "1h": 1,
    "2h": 2,
    "3h": 3,
    "4h": 4,
    "6h": 6,
    "12h": 12,
    "1d": 24,
    "7d": 168,
    "1mo": 730.5,
    "2mo": 1461,
    "3mo": 2191.5,
    "6mo": 4383,
}


def _period_to_label(period: int, interval: str | None) -> str:
    """Derive a human-readable seasonality label from period and sampling interval.

    Computes ``period * interval_hours`` to get the cycle length in hours,
    then bins into a descriptive label ("daily", "weekly", …).

    Parameters
    ----------
    period : int
        Number of observations per seasonal cycle.
    interval : str | None
        Sampling interval string (e.g. ``"1h"``, ``"15min"``, ``"1d"``).
        If ``None`` or unrecognised, the raw *period* is returned as a string.

    Returns
    -------
    str
        Human-readable label such as ``"daily"``, ``"weekly"``, ``"annual"``,
        or a fallback like ``"6h"`` / ``"24"``.

    """
    if interval is None:
        return str(period)
    h = _INTERVAL_HOURS.get(interval)
    if h is None:
        return str(period)
    days = period * h / 24
    if days < 0.5:
        hours = period * h
        return f"{hours:g}h"
    if days < 1.5:
        return "daily"
    if days < 10:
        return "weekly"
    if days < 45:
        return "monthly"
    if days < 120:
        return "quarterly"
    if days < 250:
        return "semi-annual"
    return "annual"


def _format_component_label(name: str) -> str:
    """Format a decomposition component name for subplot display.

    Parameters
    ----------
    name : str
        Internal component key, e.g. ``"trend"``, ``"seasonal_daily"``,
        ``"seasonal_adjusted"``.

    Returns
    -------
    str
        Title-cased display label.  ``"seasonal_<suffix>"`` becomes
        ``"Seasonal (<suffix>)"``, except ``"seasonal_adjusted"`` which
        becomes ``"Seasonal Adjusted"``.

    """
    if name == "seasonal_adjusted":
        return "Seasonal Adjusted"
    m = re.match(r"^seasonal_(.+)$", name)
    if m:
        return f"Seasonal ({m.group(1)})"
    return name.replace("_", " ").title()


def _clean_series(series: pl.Series) -> np.ndarray:
    """Interpolate NaNs in *series* and return a clean float numpy array.

    Uses linear interpolation followed by forward-fill and backward-fill
    to eliminate all missing values.  Emits a ``UserWarning`` if any
    values were interpolated.

    Parameters
    ----------
    series : pl.Series
        Numeric series, possibly containing ``None`` / ``NaN`` values.

    Returns
    -------
    np.ndarray
        1-D float64 array with no missing values.

    """
    import warnings  # noqa: PLC0415

    values = series.to_list()
    clean = pl.Series(values).interpolate().forward_fill().backward_fill()
    n_interpolated = sum(v is None or (isinstance(v, float) and np.isnan(v)) for v in values) - sum(
        v is None or (isinstance(v, float) and np.isnan(v)) for v in clean.to_list()
    )
    if n_interpolated > 0:
        warnings.warn(
            f"Interpolated {n_interpolated} NaN value(s) before decomposition.",
            UserWarning,
            stacklevel=4,
        )
    return clean.to_numpy().astype(float)


def _compute_stl(
    series: pl.Series,
    *,
    period: int,
    trend_window: int | None = None,
    seasonal_window: int | None = None,
    low_pass_window: int | None = None,
    robust: bool = True,
    model: Literal["additive", "multiplicative"] = "additive",
) -> dict[str, list[float]]:
    """Run STL decomposition on a single numeric series.

    Parameters
    ----------
    series : pl.Series
        Numeric values (NaN-interpolated before decomposition).
    period : int
        Seasonal period in observations.
    trend_window : int | None
        Trend smoother window.  Passed to ``STL(trend=...)``.
    seasonal_window : int | None
        Seasonal smoother window.  Passed to ``STL(seasonal=...)``.
    low_pass_window : int | None
        Low-pass filter window.  Passed to ``STL(low_pass=...)``.
    robust : bool
        Whether to use robust fitting (down-weights outliers).
    model : {"additive", "multiplicative"}
        Decomposition model.  ``"multiplicative"`` applies a log-transform
        before STL and exponentiates the results back.

    Returns
    -------
    dict[str, list[float]]
        Keys: ``observed``, ``trend``, ``seasonal``, ``residual``,
        ``seasonal_adjusted``.

    """
    import warnings  # noqa: PLC0415

    try:
        from statsmodels.tsa.seasonal import STL  # noqa: PLC0415
    except ImportError:
        msg = (
            "statsmodels is required for STL decomposition. "
            "Install it with:  pip install yohou[plotting]  "
            "or  pip install statsmodels"
        )
        raise ImportError(msg) from None

    clean_np = _clean_series(series)

    # Multiplicative: log-transform with auto-offset
    _offset = 0.0
    if model == "multiplicative":
        warnings.warn(
            "Multiplicative STL uses a log-transform approximation "
            "(log -> additive STL -> exp).  Results may differ from "
            "a true multiplicative decomposition.",
            UserWarning,
            stacklevel=3,
        )
        min_val = clean_np.min()
        if min_val <= 0:
            _offset = abs(min_val) + 1e-8
            clean_np = clean_np + _offset
        clean_np = np.log(clean_np)

    stl_kwargs: dict = {"period": period, "robust": robust}
    if trend_window is not None:
        stl_kwargs["trend"] = trend_window if trend_window % 2 == 1 else trend_window + 1
    if seasonal_window is not None:
        stl_kwargs["seasonal"] = seasonal_window if seasonal_window % 2 == 1 else seasonal_window + 1
    if low_pass_window is not None:
        stl_kwargs["low_pass"] = low_pass_window if low_pass_window % 2 == 1 else low_pass_window + 1

    result = STL(clean_np, **stl_kwargs).fit()

    if model == "multiplicative":
        observed = np.exp(clean_np) - _offset
        trend = np.exp(result.trend)
        seasonal = np.exp(result.seasonal)
        residual = np.exp(result.resid)
        seasonal_adjusted = observed / seasonal
    else:
        observed = clean_np
        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid
        seasonal_adjusted = observed - seasonal

    return {
        "observed": np.asarray(observed).tolist(),
        "trend": np.asarray(trend).tolist(),
        "seasonal": np.asarray(seasonal).tolist(),
        "residual": np.asarray(residual).tolist(),
        "seasonal_adjusted": np.asarray(seasonal_adjusted).tolist(),
    }


def _compute_mstl(
    series: pl.Series,
    *,
    periods: list[int],
    robust: bool = True,
    model: Literal["additive", "multiplicative"] = "additive",
) -> dict[str, list[float]]:
    """Run MSTL (multi-seasonal) decomposition on a single numeric series.

    Parameters
    ----------
    series : pl.Series
        Numeric values (NaN-interpolated before decomposition).
    periods : list[int]
        Seasonal periods in observations (e.g. ``[24, 8760]`` for hourly
        data with daily and annual cycles).
    robust : bool
        Whether to use robust fitting (down-weights outliers).
    model : {"additive", "multiplicative"}
        Decomposition model.  ``"multiplicative"`` applies a log-transform
        before MSTL and exponentiates the results back.

    Returns
    -------
    dict[str, list[float]]
        Keys: ``observed``, ``trend``, ``seasonal_<period>`` for each
        period, ``residual``, ``seasonal_adjusted``.

    """
    import warnings  # noqa: PLC0415

    try:
        from statsmodels.tsa.seasonal import MSTL  # noqa: PLC0415
    except ImportError:
        msg = (
            "statsmodels>=0.14 is required for MSTL decomposition. "
            "Install it with:  pip install yohou[plotting]  "
            "or  pip install 'statsmodels>=0.14'"
        )
        raise ImportError(msg) from None

    clean_np = _clean_series(series)

    # Multiplicative: log-transform with auto-offset
    _offset = 0.0
    if model == "multiplicative":
        warnings.warn(
            "Multiplicative MSTL uses a log-transform approximation "
            "(log -> additive MSTL -> exp).  Results may differ from "
            "a true multiplicative decomposition.",
            UserWarning,
            stacklevel=3,
        )
        min_val = clean_np.min()
        if min_val <= 0:
            _offset = abs(min_val) + 1e-8
            clean_np = clean_np + _offset
        clean_np = np.log(clean_np)

    sorted_periods = sorted(periods)

    result = MSTL(clean_np, periods=sorted_periods, stl_kwargs={"robust": robust}).fit()

    seasonal = result.seasonal
    if seasonal.ndim == 1:
        seasonal = seasonal.reshape(-1, 1)
    n_seasonal = seasonal.shape[1]
    if n_seasonal < len(sorted_periods):
        dropped = sorted_periods[n_seasonal:]
        msg = (
            f"MSTL returned {n_seasonal} seasonal component(s) but "
            f"{len(sorted_periods)} period(s) were requested. "
            f"Period(s) {dropped} were dropped by statsmodels - the "
            f"series is too short for these periods (need at least "
            f"2x the period length). Either remove the large period(s) "
            f"or use a longer time series."
        )
        raise ValueError(msg)

    if model == "multiplicative":
        observed = np.exp(clean_np) - _offset
        trend = np.exp(result.trend)
        residual = np.exp(result.resid)
        out: dict[str, list[float]] = {
            "observed": np.asarray(observed).tolist(),
            "trend": np.asarray(trend).tolist(),
            "residual": np.asarray(residual).tolist(),
        }
        seasonal_product = np.ones_like(clean_np)
        for i, p in enumerate(sorted_periods):
            s = np.exp(seasonal[:, i])
            out[f"seasonal_{p}"] = np.asarray(s).tolist()
            seasonal_product *= s
        out["seasonal_adjusted"] = np.asarray(observed / seasonal_product).tolist()
    else:
        out = {
            "observed": clean_np.tolist(),
            "trend": result.trend.tolist(),
            "residual": result.resid.tolist(),
        }
        seasonal_sum = np.zeros_like(clean_np)
        for i, p in enumerate(sorted_periods):
            s = seasonal[:, i]
            out[f"seasonal_{p}"] = s.tolist()
            seasonal_sum += s
        out["seasonal_adjusted"] = (clean_np - seasonal_sum).tolist()

    return out


def _compute_classical(
    series: pl.Series,
    *,
    period: int,
    model: Literal["additive", "multiplicative"] = "additive",
    two_sided: bool = True,
    extrapolate_trend: int | str = 0,
) -> dict[str, list[float]]:
    """Run classical seasonal decomposition on a single numeric series.

    Parameters
    ----------
    series : pl.Series
        Numeric values (NaN-interpolated before decomposition).
    period : int
        Seasonal period in observations.
    model : {"additive", "multiplicative"}
        Decomposition model.
    two_sided : bool
        Use a two-sided (centered) moving average for the trend.
    extrapolate_trend : int | str
        If set to > 0, the trend resulting from the convolution is
        linear least-squares extrapolated on both ends.  Use ``"freq"``
        to extrapolate by ``period`` observations.

    Returns
    -------
    dict[str, list[float]]
        Keys: ``observed``, ``trend``, ``seasonal``, ``residual``,
        ``seasonal_adjusted``.

    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose  # noqa: PLC0415
    except ImportError:
        msg = (
            "statsmodels is required for classical decomposition. "
            "Install it with:  pip install yohou[plotting]  "
            "or  pip install statsmodels"
        )
        raise ImportError(msg) from None

    clean_np = _clean_series(series)

    # Auto-offset for multiplicative model with non-positive values
    _offset = 0.0
    if model == "multiplicative":
        min_val = clean_np.min()
        if min_val <= 0:
            _offset = abs(min_val) + 1e-8
            clean_np = clean_np + _offset

    result = seasonal_decompose(
        clean_np,
        model=model,
        period=period,
        two_sided=two_sided,
        extrapolate_trend=extrapolate_trend,
    )

    observed = clean_np - _offset
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

    seasonal_adjusted = observed / seasonal if model == "multiplicative" else observed - seasonal

    return {
        "observed": np.asarray(observed).tolist(),
        "trend": np.asarray(trend).tolist(),
        "seasonal": np.asarray(seasonal).tolist(),
        "residual": np.asarray(residual).tolist(),
        "seasonal_adjusted": np.asarray(seasonal_adjusted).tolist(),
    }


def plot_decomposition(
    y: pl.DataFrame,
    components: dict[str, pl.DataFrame] | list[str] | tuple[str, ...],
    *,
    method: Literal["stl", "mstl", "classical"] | None = None,
    columns: str | list[str] | None = None,
    panel_group_names: list[str] | None = None,
    show_original: bool = True,
    period: int | str = "auto",
    periods: list[int] | str | None = None,
    model: Literal["additive", "multiplicative"] = "additive",
    robust: bool = True,
    trend_window: int | None = None,
    seasonal_window: int | None = None,
    low_pass_window: int | None = None,
    two_sided: bool = True,
    extrapolate_trend: int | str = 0,
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
) -> go.Figure | dict[str, go.Figure]:
    """Plot time series decomposition as vertically stacked subplots.

    Displays the original series and its decomposed components (e.g. trend,
    seasonal, residual) in separate panels sharing the same time axis.

    There are two modes of operation:

    **Pre-computed mode** (default) - pass *components* as a ``dict`` mapping
    component names to DataFrames produced by a Yohou decomposition pipeline.

    **Decomposition mode** - pass *components* as a ``list`` or ``tuple`` of
    component names and set *method* to ``"stl"``, ``"mstl"``, or
    ``"classical"``.  The function runs the decomposition internally and
    renders the requested components.

    Parameters
    ----------
    y : pl.DataFrame
        Original time series with ``"time"`` column.
    components : dict[str, pl.DataFrame] | list[str] | tuple[str, ...]
        **dict** - mapping of component names to DataFrames (pre-computed
        mode).  Each DataFrame must have a ``"time"`` column plus value
        columns matching *y*.

        **list/tuple of str** - component names to compute and display.
        Requires *method* to be set.
        Valid names: ``"observed"``, ``"trend"``, ``"seasonal"``,
        ``"residual"``, ``"seasonal_adjusted"``.
    method : {"stl", "mstl", "classical"} or None
        Decomposition backend.  Required when *components* is a list.
        ``None`` means pre-computed dict mode.
    columns : str | list[str] | None
        Value columns to plot.  ``None`` uses all numeric non-time columns.
    panel_group_names : list[str] | None
        Panel group prefixes to include.  For panel data, returns one
        figure per member with groups overlaid by colour.
    show_original : bool
        Include the original series as the first subplot.
    period : int | str
        Seasonal period (STL, MSTL, classical).  ``"auto"`` infers from
        the sampling interval.
    periods : list[int] | str | None
        Seasonal periods for MSTL.  **Required** when ``method="mstl"``.
    model : {"additive", "multiplicative"}
        Decomposition model.  STL/MSTL use a log-transform approximation
        for multiplicative; classical uses native statsmodels support.
    robust : bool
        Use robust fitting (STL/MSTL only, down-weights outliers).
    trend_window : int | None
        Trend smoother window (STL only).
    seasonal_window : int | None
        Seasonal smoother window (STL only).
    low_pass_window : int | None
        Low-pass filter window (STL only).
    two_sided : bool
        Two-sided (centered) moving average for trend (classical only).
    extrapolate_trend : int | str
        Extrapolate trend at edges (classical only).  ``0`` leaves NaN.
    color_palette : list[str] | None
        Custom color palette.
    show_legend : bool
        Whether to show the legend.
    title : str | None
        Plot title.
    x_label : str | None
        X-axis label on bottom subplot.  Defaults to ``"Time"``.
    y_label : str | None
        Y-axis label.
    width : int | None
        Plot width in pixels.
    height : int | None
        Plot height in pixels.
    connect_gaps : bool
        Whether to connect gaps with lines.
    resampler : bool | Literal["widget"] | None
        Enable plotly-resampler for large datasets.
    line_width : float
        Width of component line traces.
    line_dash : str
        Dash style for component lines.

    Returns
    -------
    go.Figure | dict[str, go.Figure]
        Plotly figure (or dict of figures for panel data).

    Raises
    ------
    TypeError
        If *y* is not a Polars DataFrame.
    ValueError
        If *components* is a list without *method*, DataFrames are empty,
        unknown component names, or ``method="mstl"`` without *periods*.
    ImportError
        When ``statsmodels`` is not installed.

    Examples
    --------
    Pre-computed mode:

    >>> import polars as pl
    >>> from yohou.plotting import plot_decomposition

    >>> dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
    >>> y = pl.DataFrame({"time": dates, "y": list(range(len(dates)))})
    >>> comps = {
    ...     "trend": pl.DataFrame({"time": dates, "y": [i * 0.5 for i in range(len(dates))]}),
    ...     "residual": pl.DataFrame({"time": dates, "y": [i * 0.5 for i in range(len(dates))]}),
    ... }
    >>> fig = plot_decomposition(y, comps)
    >>> len(fig.data) >= 3
    True

    STL mode:

    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2018, 1, 1), pl.date(2022, 12, 31), "1mo", eager=True),
    ...     "y": [100 + 10 * (i % 12) + i * 0.5 for i in range(60)],
    ... })
    >>> fig = plot_decomposition(df, ["trend", "seasonal"], method="stl")  # doctest: +SKIP

    Classical mode:

    >>> fig = plot_decomposition(df, ["trend", "seasonal"], method="classical")  # doctest: +SKIP

    See Also
    --------
    [`plot_forecast`][yohou.plotting.plot_forecast] : Forecast visualization.
    [`plot_seasonality`][yohou.plotting.plot_seasonality] : Seasonal pattern analysis.
    """
    import warnings  # noqa: PLC0415

    validate_plotting_data(y)
    validate_plotting_params(width=width, height=height)

    decomp_mode = isinstance(components, list | tuple) and (not components or isinstance(components[0], str))

    # Decomposition mode
    if decomp_mode:
        if method is None:
            msg = (
                "method is required when components is a list/tuple. "
                "Use method='stl', method='mstl', or method='classical'."
            )
            raise ValueError(msg)

        if method == "mstl" and periods is None:
            msg = "periods is required when method='mstl'."
            raise ValueError(msg)

        # Warn on mismatched params
        _stl_only = {
            "trend_window": trend_window,
            "seasonal_window": seasonal_window,
            "low_pass_window": low_pass_window,
        }
        _classical_only = {"two_sided": two_sided, "extrapolate_trend": extrapolate_trend}

        if method != "stl":
            for pname, pval in _stl_only.items():
                if pval is not None:
                    warnings.warn(
                        f"'{pname}' is only used with method='stl' (got method='{method}'); ignored.",
                        UserWarning,
                        stacklevel=2,
                    )
        if method != "classical":
            if two_sided is not True:
                warnings.warn(
                    f"'two_sided' is only used with method='classical' (got method='{method}'); ignored.",
                    UserWarning,
                    stacklevel=2,
                )
            if extrapolate_trend != 0:
                warnings.warn(
                    f"'extrapolate_trend' is only used with method='classical' (got method='{method}'); ignored.",
                    UserWarning,
                    stacklevel=2,
                )

        components_list = list(components)

        # "observed" in the list maps to show_original
        if "observed" in components_list:
            show_original = True
            components_list.remove("observed")

        # Validate component names early
        valid_names = {"trend", "seasonal", "residual", "seasonal_adjusted"}
        unknown = {c for c in components_list if c not in valid_names and not re.match(r"^seasonal_\w+$", c)}
        if unknown:
            all_valid = sorted(valid_names | {"observed"})
            msg = f"Unknown components: {unknown}. Valid: {all_valid} (also seasonal_<period>)"
            raise ValueError(msg)

        if not components_list and not show_original:
            msg = "components must contain at least one displayable component"
            raise ValueError(msg)

        value_cols = validate_plotting_data(y, columns=columns, exclude=["time"])

        if method == "mstl":
            components = _mstl_to_component_dict(
                y,
                components_list,
                value_cols,
                periods,  # ty: ignore[invalid-argument-type]
                robust,
                model=model,
            )
            title = title or "MSTL Decomposition"
        elif method == "classical":
            components = _classical_to_component_dict(
                y,
                components_list,
                value_cols,
                period=period,
                model=model,
                two_sided=two_sided,
                extrapolate_trend=extrapolate_trend,
            )
            title = title or "Classical Decomposition"
        else:
            # method == "stl"
            components = _stl_to_component_dict(
                y,
                components_list,
                value_cols,
                period=period,
                model=model,
                robust=robust,
                trend_window=trend_window,
                seasonal_window=seasonal_window,
                low_pass_window=low_pass_window,
            )
            title = title or "STL Decomposition"

        # Fall through to the shared dict plotting below

    # Dict validation
    if not isinstance(components, dict):
        msg = (
            "components must be a dict[str, pl.DataFrame] (pre-computed mode) "
            "or a list[str]/tuple[str, ...] with method set (decomposition mode)"
        )
        raise TypeError(msg)

    if not components:
        msg = "components dict must be non-empty"
        raise ValueError(msg)
    for comp_df in components.values():
        validate_plotting_data(comp_df)

    value_cols = validate_plotting_data(y, columns=columns)

    # Detect panel data
    _, panel_groups = inspect_panel(y)
    is_panel = bool(panel_groups)

    # Auto-enter panel mode when panel columns are detected
    if is_panel and panel_group_names is None and columns is None:
        panel_group_names = []

    if is_panel and panel_group_names is not None:
        return _plot_decomposition_panel(
            y=y,
            components=components,
            value_cols=value_cols,
            decomp_mode=decomp_mode,
            show_original=show_original,
            panel_group_names=panel_group_names,
            color_palette=color_palette,
            title=title,
            x_label=x_label,
            y_label=y_label,
            width=width,
            height=height,
            connect_gaps=connect_gaps,
            line_width=line_width,
            line_dash=line_dash,
            resampler=resampler,
            show_legend=show_legend,
        )

    # Build subplot structure (shared by both modes)
    panel_names: list[str] = []
    if show_original:
        panel_names.append("Original")
    panel_names.extend(_format_component_label(name) if decomp_mode else name for name in components)

    n_rows = len(panel_names)
    fig = _create_subplots(
        resampler,
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
    )

    # Set component names as y-axis titles
    for idx, pname in enumerate(panel_names):
        yaxis_key = f"yaxis{idx + 1}" if idx > 0 else "yaxis"
        fig.layout[yaxis_key].title = {"text": pname, "font": {"size": 12}}

    colors = resolve_color_palette(color_palette, len(value_cols))
    row_offset = 0

    # Original series
    if show_original:
        row_offset = 1
        for i, col in enumerate(value_cols):
            if col in y.columns:
                fig.add_trace(
                    go.Scatter(
                        x=y["time"],
                        y=y[col],
                        mode="lines",
                        line={
                            "color": colors[i % len(colors)],
                            "width": line_width,
                            "dash": line_dash,
                        },
                        name=col,
                        legendgroup=col,
                        showlegend=True,
                        connectgaps=connect_gaps,
                    ),
                    row=1,
                    col=1,
                )

    # Component panels
    _meta_cols = {"time", "observed_time"}
    for comp_idx, (comp_name, comp_df) in enumerate(components.items()):
        row = comp_idx + 1 + row_offset
        # Resolve value columns for this component: prefer names matching y,
        # otherwise fall back to the component's own numeric columns (handles
        # renamed columns from transformers, e.g. "log_off_0.0_tourists").
        comp_value_cols = [c for c in value_cols if c in comp_df.columns]
        if not comp_value_cols:
            comp_value_cols = [c for c in comp_df.columns if c not in _meta_cols]
        for i, col in enumerate(comp_value_cols):
            # Use the original value_cols name for legend when available,
            # otherwise use the component column name.
            legend_col = value_cols[i] if i < len(value_cols) else col
            display_name = _format_component_label(comp_name) if decomp_mode else legend_col
            fig.add_trace(
                go.Scatter(
                    x=comp_df["time"],
                    y=comp_df[col],
                    mode="lines",
                    line={
                        "color": colors[i % len(colors)],
                        "width": line_width,
                        "dash": line_dash,
                    },
                    name=display_name,
                    legendgroup=display_name if decomp_mode else legend_col,
                    showlegend=(comp_idx == 0 and not show_original),
                    connectgaps=connect_gaps,
                ),
                row=row,
                col=1,
            )

    title_default = title or "Time Series Decomposition"
    default_height = max(300 * n_rows, 400)

    fig = apply_default_layout(
        fig,
        title=title_default,
        x_label=None,
        y_label=None,
        width=width,
        height=height or default_height,
    )

    # When the caller supplies a y-label, apply it to every subplot row
    # so it appears on the left.  Otherwise keep the per-row titles that
    # make_subplots already set (e.g. Trend, Seasonal, Residual).
    if y_label is not None:
        fig.update_yaxes(title_text=y_label)

    # Show x-axis label on bottom subplot only
    x_label_text = x_label if x_label is not None else "Time"
    bottom_xaxis = f"xaxis{n_rows}" if n_rows > 1 else "xaxis"
    fig.layout[bottom_xaxis].title = {"text": x_label_text}

    return fig


def _plot_decomposition_panel(
    y: pl.DataFrame,
    components: dict[str, pl.DataFrame],
    value_cols: list[str],
    decomp_mode: bool,
    show_original: bool,
    panel_group_names: list[str],
    color_palette: list[str] | None,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    width: int | None,
    height: int | None,
    connect_gaps: bool,
    line_width: float,
    line_dash: str,
    resampler: bool | Literal["widget"] | None,
    show_legend: bool = True,
) -> go.Figure | dict[str, go.Figure]:
    """Render ``plot_decomposition`` for panel data.

    Returns one figure per member with groups overlaid by colour.
    When there is only one member, returns a single ``go.Figure``.
    """
    panel_cols = resolve_panel_columns(y, panel_group_names or None, None)
    groups, all_members = _group_panel_columns(panel_cols)

    # Component row labels
    comp_labels: list[str] = []
    if show_original:
        comp_labels.append("Original")
    comp_labels.extend(_format_component_label(name) if decomp_mode else name for name in components)
    n_rows = len(comp_labels)

    figures: dict[str, go.Figure] = {}

    for member in all_members:
        color_mgr = PanelColorManager(color_palette)
        legend_tracker = LegendTracker(show_legend=show_legend)

        fig = _create_subplots(
            resampler,
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=_subplot_spacing(n_rows),
        )

        def _add_traces(
            data_df: pl.DataFrame,
            comp_row: int,
            *,
            _member: str = member,
            _fig: go.Figure = fig,
            _color_mgr: PanelColorManager = color_mgr,
            _legend_tracker: LegendTracker = legend_tracker,
        ) -> None:
            """Add component traces for a single member to the figure."""
            for gname, group_cols in groups.items():
                col_name = next(
                    (c for c in group_cols if _member_name(c) == _member),
                    None,
                )
                if col_name is None or col_name not in data_df.columns:
                    continue
                _fig.add_trace(
                    go.Scatter(
                        x=data_df["time"],
                        y=data_df[col_name],
                        mode="lines",
                        line={
                            "color": _color_mgr.get_color(gname),
                            "width": line_width,
                            "dash": line_dash,
                        },
                        name=gname,
                        legendgroup=gname,
                        showlegend=_legend_tracker.should_show(gname),
                        connectgaps=connect_gaps,
                    ),
                    row=comp_row,
                    col=1,
                )

        row_offset = 0
        if show_original:
            row_offset = 1
            _add_traces(y, comp_row=1)

        for comp_idx, (_, comp_df) in enumerate(components.items()):
            _add_traces(comp_df, comp_row=comp_idx + 1 + row_offset)

        # Set component names as y-axis titles on the left
        for idx, label in enumerate(comp_labels):
            yaxis_key = f"yaxis{idx + 1}" if idx > 0 else "yaxis"
            fig.layout[yaxis_key].title = {"text": label, "font": {"size": 12}}

        title_default = title or f"Time Series Decomposition - {member}"
        default_height = max(300 * n_rows, 400)

        fig = apply_default_layout(
            fig,
            title=title_default,
            x_label=None,
            y_label=None,
            width=width,
            height=height or default_height,
        )

        # When the caller supplies a y-label, override the per-row titles.
        if y_label is not None:
            fig.update_yaxes(title_text=y_label)

        # Show x-axis label on bottom subplot only
        x_label_text = x_label if x_label is not None else "Time"
        xaxis_key = f"xaxis{n_rows}" if n_rows > 1 else "xaxis"
        if xaxis_key in fig.layout:
            fig.layout[xaxis_key].title = {"text": x_label_text}

        fig.update_layout(showlegend=show_legend)
        figures[member] = fig

    if len(figures) == 1:
        return next(iter(figures.values()))
    return figures


def _stl_to_component_dict(
    y: pl.DataFrame,
    components: list[str],
    columns: list[str],
    *,
    period: int | str = "auto",
    model: Literal["additive", "multiplicative"] = "additive",
    robust: bool = True,
    trend_window: int | None = None,
    seasonal_window: int | None = None,
    low_pass_window: int | None = None,
) -> dict[str, pl.DataFrame]:
    """Compute STL decomposition and return as component DataFrames.

    Parameters
    ----------
    y : pl.DataFrame
        Input DataFrame with ``"time"`` column and numeric columns.
    components : list[str]
        Component names to compute (e.g. ``"trend"``, ``"seasonal"``).
    columns : list[str]
        Numeric column names from *y* to decompose.
    period : int | str
        Seasonal period or ``"auto"`` to infer.
    model : {"additive", "multiplicative"}
        Decomposition model.
    robust : bool
        Use robust fitting.
    trend_window, seasonal_window, low_pass_window : int | None
        Smoother window parameters forwarded to STL.

    Returns
    -------
    dict[str, pl.DataFrame]
        Mapping from component name to a DataFrame with ``"time"`` plus
        one column per decomposed series.

    """
    # Resolve STL period
    if isinstance(period, str) and period == "auto":
        from yohou.utils.validation import check_interval_consistency  # noqa: PLC0415

        interval = check_interval_consistency(y)
        stl_period = _INTERVAL_TO_STL_PERIOD.get(interval)
        if stl_period is None:
            msg = (
                f"Cannot infer STL period for interval '{interval}'. "
                f"Supported intervals: {sorted(_INTERVAL_TO_STL_PERIOD)}. "
                f"Pass an explicit period."
            )
            raise ValueError(msg)
    else:
        stl_period = int(period)

    # Compute STL for each column and collect per-component values
    result_data: dict[str, dict[str, list[float]]] = {comp: {} for comp in components}

    for col_name in columns:
        stl_result = _compute_stl(
            y[col_name],
            period=stl_period,
            trend_window=trend_window,
            seasonal_window=seasonal_window,
            low_pass_window=low_pass_window,
            robust=robust,
            model=model,
        )
        for comp in components:
            result_data[comp][col_name] = stl_result[comp]

    # Build a DataFrame per component
    time_col = y["time"]
    return {comp: pl.DataFrame({"time": time_col, **col_values}) for comp, col_values in result_data.items()}


def _mstl_to_component_dict(
    y: pl.DataFrame,
    components: list[str],
    columns: list[str],
    periods: list[int] | str,
    robust: bool,
    *,
    model: Literal["additive", "multiplicative"] = "additive",
) -> dict[str, pl.DataFrame]:
    """Compute MSTL decomposition and return as component DataFrames.

    Resolves periods (auto-detect or explicit), runs ``_compute_mstl``
    per column, then remaps the numeric ``seasonal_<N>`` keys to
    human-readable labels (e.g. ``seasonal_daily``) via
    ``_period_to_label``.

    Parameters
    ----------
    y : pl.DataFrame
        Input DataFrame with ``"time"`` column and numeric columns.
    components : list[str]
        Requested component names (e.g. ``"trend"``, ``"seasonal"``).
        ``"seasonal"`` is expanded into one key per period.
    columns : list[str]
        Numeric column names from *y* to decompose.
    periods : list[int] | str
        Seasonal periods, or ``"auto"`` to infer from the sampling
        interval via ``_INTERVAL_TO_MSTL_PERIODS``.
    robust : bool
        Whether to use robust fitting (down-weights outliers).

    Returns
    -------
    dict[str, pl.DataFrame]
        Mapping from component name (with human-readable seasonal
        labels) to a DataFrame with ``"time"`` plus one column per
        decomposed series.

    """
    from yohou.utils.validation import check_interval_consistency  # noqa: PLC0415

    if isinstance(periods, str) and periods == "auto":
        interval = check_interval_consistency(y)
        resolved = _INTERVAL_TO_MSTL_PERIODS.get(interval)
        if resolved is None:
            msg = (
                f"Cannot infer MSTL periods for interval '{interval}'. "
                f"Supported intervals: {sorted(_INTERVAL_TO_MSTL_PERIODS)}. "
                f"Pass an explicit list via periods=[...]."
            )
            raise ValueError(msg)
        sorted_periods = sorted(resolved)
    else:
        sorted_periods = sorted(periods)

    # Infer interval for human-readable labels (fall back to numeric if unknown)
    try:
        interval = check_interval_consistency(y)
    except (ValueError, TypeError):
        interval = None

    # Map numeric seasonal keys → human-readable seasonal keys
    numeric_to_human: dict[str, str] = {}
    for p in sorted_periods:
        label = _period_to_label(p, interval)
        numeric_to_human[f"seasonal_{p}"] = f"seasonal_{label}"

    numeric_seasonal_keys = [f"seasonal_{p}" for p in sorted_periods]

    # Expand "seasonal" into per-period keys (numeric, for _compute_mstl indexing)
    expanded_numeric: list[str] = []
    for comp in components:
        if comp == "seasonal":
            expanded_numeric.extend(numeric_seasonal_keys)
        else:
            expanded_numeric.append(comp)

    # Build output with human-readable keys
    human_keys = [numeric_to_human.get(k, k) for k in expanded_numeric]
    result_data: dict[str, dict[str, list[float]]] = {hk: {} for hk in human_keys}

    for col_name in columns:
        mstl_result = _compute_mstl(y[col_name], periods=sorted_periods, robust=robust, model=model)
        for nk, hk in zip(expanded_numeric, human_keys, strict=True):
            result_data[hk][col_name] = mstl_result[nk]

    time_col = y["time"]
    return {comp: pl.DataFrame({"time": time_col, **col_values}) for comp, col_values in result_data.items()}


def _classical_to_component_dict(
    y: pl.DataFrame,
    components: list[str],
    columns: list[str],
    *,
    period: int | str = "auto",
    model: Literal["additive", "multiplicative"] = "additive",
    two_sided: bool = True,
    extrapolate_trend: int | str = 0,
) -> dict[str, pl.DataFrame]:
    """Compute classical seasonal decomposition and return as component DataFrames.

    Parameters
    ----------
    y : pl.DataFrame
        Input DataFrame with ``"time"`` column and numeric columns.
    components : list[str]
        Component names to compute (e.g. ``"trend"``, ``"seasonal"``).
    columns : list[str]
        Numeric column names from *y* to decompose.
    period : int | str
        Seasonal period or ``"auto"`` to infer.
    model : {"additive", "multiplicative"}
        Decomposition model.
    two_sided : bool
        Use centered moving average for trend.
    extrapolate_trend : int | str
        Extrapolate trend at edges.

    Returns
    -------
    dict[str, pl.DataFrame]
        Mapping from component name to a DataFrame with ``"time"`` plus
        one column per decomposed series.

    """
    # Resolve period
    if isinstance(period, str) and period == "auto":
        from yohou.utils.validation import check_interval_consistency  # noqa: PLC0415

        interval = check_interval_consistency(y)
        classical_period = _INTERVAL_TO_STL_PERIOD.get(interval)
        if classical_period is None:
            msg = (
                f"Cannot infer period for interval '{interval}'. "
                f"Supported intervals: {sorted(_INTERVAL_TO_STL_PERIOD)}. "
                f"Pass an explicit period."
            )
            raise ValueError(msg)
    else:
        classical_period = int(period)

    result_data: dict[str, dict[str, list[float]]] = {comp: {} for comp in components}

    for col_name in columns:
        classical_result = _compute_classical(
            y[col_name],
            period=classical_period,
            model=model,
            two_sided=two_sided,
            extrapolate_trend=extrapolate_trend,
        )
        for comp in components:
            result_data[comp][col_name] = classical_result[comp]

    time_col = y["time"]
    return {comp: pl.DataFrame({"time": time_col, **col_values}) for comp, col_values in result_data.items()}
