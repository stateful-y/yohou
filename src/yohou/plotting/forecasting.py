"""Forecast visualization functions."""

import re

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from yohou.plotting._utils import (
    _group_panel_columns,
    _member_name,
    apply_default_layout,
    palette_yohou,
    resolve_color_palette,
)
from yohou.plotting.evaluation import _discover_proba_targets, _hex_to_rgba
from yohou.utils import inspect_panel, validate_plotting_data

# The full palette list is used as the default effective palette in every
# plot_forecast code path. Slots 0/1/2 are reserved for the three semantic
# traces (history, single-model forecast, actual); slot 3+ are model colors.
_PALETTE = list(palette_yohou().values())

__all__ = [
    "plot_components",
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
    coverage_rates: list[float] | None = None,
    n_history: int | None = None,
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
    coverage_rates : list[float] | None, default=None
        Coverage rates to display intervals for (e.g., [0.9, 0.95]).
        Looks for ``{col}_lower_{rate}`` / ``{col}_upper_{rate}`` in y_pred.
    n_history : int | None, default=None
        Number of historical observations to show from y_train. If None, shows all.
    panel_group_names : list[str] | None, default=None
        Panel group prefixes to plot. If None and panel data is detected,
        plots all groups. Creates faceted subplots.
    facet_n_cols : int, default=2
        Number of columns in facet grid for panel data.
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
        - band_opacity : float, default=0.15
        - show_transition : bool, default=True

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
    `plot_residuals` : Plot residual diagnostics.
    `plot_model_comparison_bar` : Grouped bar chart for scorer comparison.
    """
    # Validate inputs
    validate_plotting_data(y_test)
    if isinstance(y_pred, dict):
        for _name, pred_df in y_pred.items():
            validate_plotting_data(pred_df)  # type: ignore[invalid-argument-type]
    else:
        validate_plotting_data(y_pred)
    if y_train is not None:
        validate_plotting_data(y_train)

    # Semantic colors always come from the effective palette: slot 0 = history,
    # slot 1 = single-model forecast, slot 2 = actual, slot 3+ = model comparison.
    eff_palette = color_palette if color_palette is not None else _PALETTE
    forecast_color = eff_palette[1 % len(eff_palette)]
    actual_color = eff_palette[2 % len(eff_palette)]
    line_width = kwargs.get("line_width", 2.0)
    band_opacity = kwargs.get("band_opacity", 0.25)
    show_transition = kwargs.get("show_transition", True)

    # Detect panel data
    _, y_test_panels = inspect_panel(y_test)
    is_panel = bool(y_test_panels)

    # Auto-detect prediction type from the first prediction DataFrame
    _first_pred = next(iter(y_pred.values())) if isinstance(y_pred, dict) else y_pred
    prediction_mode = _detect_prediction_mode(_first_pred)

    # For panel data, delegate to faceted handler (single or multi-model)
    if is_panel:
        return _plot_forecast_panel(
            y_test=y_test,
            y_pred=y_pred,
            y_train=y_train,
            coverage_rates=coverage_rates,
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
            show_transition=show_transition,
            prediction_mode=prediction_mode,
        )

    # Non-panel class-probability predictions
    if prediction_mode == "class_proba":
        return _plot_forecast_class_proba(
            y_test=y_test,
            y_pred=y_pred,
            color_palette=color_palette,
            title=title,
            x_label=x_label,
            y_label=y_label,
            width=width,
            height=height,
            **kwargs,
        )

    # Non-panel categorical predictions
    if prediction_mode == "categorical":
        return _plot_forecast_categorical(
            y_test=y_test,
            y_pred=y_pred,
            y_train=y_train,
            n_history=n_history,
            color_palette=color_palette,
            title=title,
            x_label=x_label,
            y_label=y_label,
            width=width,
            height=height,
            **kwargs,
        )

    # Multi-model dict: delegate to dedicated helper
    if isinstance(y_pred, dict):
        return _plot_forecast_multi_model(
            y_test=y_test,
            y_preds=y_pred,  # type: ignore[invalid-argument-type]
            y_train=y_train,
            coverage_rates=coverage_rates,
            n_history=n_history,
            color_palette=color_palette,
            title=title,
            x_label=x_label,
            y_label=y_label,
            width=width,
            height=height,
            line_width=line_width,
            band_opacity=band_opacity,
            show_transition=show_transition,
        )

    # Non-panel, single-model case
    interval_pattern = re.compile(r"^.+_(lower|upper)_[\d.]+$")
    pred_value_cols = [
        c for c in y_pred.columns if c not in ("time", "observed_time") and not interval_pattern.match(c)
    ]
    test_value_cols = [c for c in y_test.columns if c != "time"]
    plot_columns = test_value_cols

    multi_col = len(plot_columns) > 1
    _model_pal = eff_palette[3:] or eff_palette
    col_colors = resolve_color_palette(_model_pal, len(plot_columns)) if multi_col else []

    fig = go.Figure()

    for col_idx, col in enumerate(plot_columns):
        # Colour & legend setup: per-column colours for multi-column plots
        if multi_col:
            col_color = col_colors[col_idx]
            actual_c = col_color
            forecast_c = col_color
            f_dash: str | None = "dash"
        else:
            actual_c, forecast_c = actual_color, forecast_color
            f_dash = None

        actual_name = f"{col} (Actual)"
        forecast_name = f"{col} (Forecast)"

        # Compute pred_col early so interval rendering can hoist before Actual.
        pred_col = col if col in pred_value_cols else (pred_value_cols[0] if pred_value_cols else None)
        fc_group = f"forecast_{col}" if multi_col else ""

        # Training data
        if y_train is not None and col in y_train.columns:
            train_df = y_train.tail(n_history) if n_history is not None else y_train
            _hex = actual_c.lstrip("#")
            _rgb = tuple(int(_hex[i : i + 2], 16) for i in (0, 2, 4))
            _train_color = f"rgba({_rgb[0]}, {_rgb[1]}, {_rgb[2]}, 0.4)"
            _train_name = f"{col} (Train)"
            fig.add_trace(
                go.Scatter(
                    x=train_df["time"],
                    y=train_df[col],
                    mode="lines",
                    line={"color": _train_color, "width": line_width},
                    name=_train_name,
                    legendgroup=fc_group,
                    legendrank=0,
                    hovertemplate=f"<b>{col} Train</b><br>Time: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>",
                )
            )

        # Prediction intervals - rendered before Actual so the actual line sits on top.
        # Works even when there is no point forecast (predict_interval-only DataFrames).
        # legendrank >= 11 places them after Train (0), Actual (1), and Forecast (10).
        if coverage_rates:
            interval_base = pred_col if pred_col is not None else col
            _hex = forecast_c.lstrip("#")
            rgb = tuple(int(_hex[i : i + 2], 16) for i in (0, 2, 4))
            sorted_rates = sorted(coverage_rates)
            n_rates = len(sorted_rates)
            for sort_idx, rate in enumerate(sorted_rates):
                lower_col = f"{interval_base}_lower_{rate}"
                upper_col = f"{interval_base}_upper_{rate}"

                if lower_col in y_pred.columns and upper_col in y_pred.columns:
                    # Narrower bands (lower rate, lower sort_idx) rendered most opaque;
                    # wider outer bands fade out gracefully.
                    rate_opacity = band_opacity * (1.0 - 0.45 * sort_idx / max(1, n_rates - 1))
                    rgba = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {rate_opacity:.3f})"
                    t = y_pred["time"].to_list()
                    y_upper = y_pred[upper_col].to_list()
                    y_lower = y_pred[lower_col].to_list()
                    x_band = t + t[::-1]
                    y_band = y_upper + y_lower[::-1]
                    pi_name = f"{col} ({rate:.0%} PI)"
                    fig.add_trace(
                        go.Scatter(
                            x=x_band,
                            y=y_band,
                            fill="toself",
                            fillcolor=rgba,
                            mode="lines",
                            line={"width": 0, "color": rgba},
                            name=pi_name,
                            legendgroup=fc_group,
                            legendrank=11 + sort_idx,
                            hoverinfo="skip",
                        )
                    )

        # Actual test data (prepend last train point to close the gap)
        _x_actual = y_test["time"]
        _y_actual = y_test[col]
        if y_train is not None and col in y_train.columns:
            _x_actual = pl.concat([pl.Series("time", [y_train["time"][-1]]), _x_actual])
            _y_actual = pl.concat([pl.Series([y_train[col][-1]], dtype=_y_actual.dtype), _y_actual])
        fig.add_trace(
            go.Scatter(
                x=_x_actual,
                y=_y_actual,
                mode="lines",
                line={"color": actual_c, "width": line_width},
                name=actual_name,
                legendgroup=fc_group,
                legendrank=1,
                hovertemplate=f"<b>{col} Actual</b><br>Time: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>",
            )
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

            line_spec: dict = {"color": forecast_c, "width": line_width}
            if f_dash:
                line_spec["dash"] = f_dash

            fig.add_trace(
                go.Scatter(
                    x=x_forecast,
                    y=forecast_y,
                    mode="lines",
                    line=line_spec,
                    name=forecast_name,
                    legendgroup=fc_group,
                    legendrank=10,
                    hovertemplate=f"<b>{col} Forecast</b><br>Time: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>",
                )
            )

    fig = apply_default_layout(
        fig,
        title=title or "Forecast",
        x_label=x_label or "Time",
        y_label=y_label or "Value",
        width=width,
        height=height,
    )

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
    all_categories: set[str] = set()
    for cat_col in cat_cols:
        if cat_col in y_test.columns:
            all_categories.update(y_test[cat_col].cast(pl.String).unique().to_list())
        for pred_df in preds.values():
            if cat_col in pred_df.columns:
                all_categories.update(pred_df[cat_col].cast(pl.String).unique().to_list())
        if y_train is not None and cat_col in y_train.columns:
            all_categories.update(y_train[cat_col].cast(pl.String).unique().to_list())
    sorted_cats = sorted(all_categories)
    cat_to_int = {cat: i for i, cat in enumerate(sorted_cats)}
    return sorted_cats, cat_to_int


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
    color_palette: list[str] | None,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    width: int | None,
    height: int | None,
    **kwargs: object,
) -> go.Figure:
    """Plot class-probability forecasts as stacked area charts with truth markers.

    Parameters
    ----------
    y_test : pl.DataFrame
        Ground-truth DataFrame with ``"time"`` and categorical target columns.
    y_pred : pl.DataFrame or dict of str to pl.DataFrame
        Predictions with ``{target}_proba_{class}`` columns.
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
    model_names = list(preds.keys())
    n_models = len(model_names)

    if n_models > 1:
        fig = make_subplots(
            rows=n_models,
            cols=1,
            subplot_titles=model_names,
            shared_xaxes=True,
            vertical_spacing=0.08,
        )
    else:
        fig = go.Figure()

    for model_idx, (model_name, pred_df) in enumerate(preds.items()):
        proba_cols = [c for c in pred_df.columns if "_proba_" in c]
        targets = _discover_proba_targets(proba_cols)
        target = next(iter(targets))
        target_proba_cols = targets[target]
        class_labels = [c.split("_proba_", 1)[1] for c in target_proba_cols]
        colors = resolve_color_palette(color_palette, n=len(class_labels))

        _render_class_proba_traces(
            fig,
            pred_df,
            y_test,
            colors=colors,
            target_proba_cols=target_proba_cols,
            class_labels=class_labels,
            target=target,
            stack_group=f"proba_{model_name}",
            row=model_idx + 1 if n_models > 1 else None,
            col=1 if n_models > 1 else None,
            show_legend=model_idx == 0,
            line_width=line_width,
            band_opacity=band_opacity,
            marker_size=marker_size,
        )

    fig = apply_default_layout(
        fig,
        title=title or "Class Probability Forecast",
        x_label=x_label or "Time",
        y_label=y_label or "Probability",
        width=width,
        height=height or (300 * n_models if n_models > 1 else None),
    )
    return fig


def _plot_forecast_categorical(
    y_test: pl.DataFrame,
    y_pred: pl.DataFrame | dict[str, pl.DataFrame],
    *,
    y_train: pl.DataFrame | None,
    n_history: int | None,
    color_palette: list[str] | None,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    width: int | None,
    height: int | None,
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
        for c in first_pred.columns
        if c not in ("time", "observed_time") and first_pred[c].dtype in (pl.String, pl.Categorical)
    ]

    sorted_cats, cat_to_int = _collect_categories(cat_cols, y_test, preds, y_train)

    fig = go.Figure()

    for model_idx, (model_name, pred_df) in enumerate(preds.items()):
        _render_categorical_traces(
            fig,
            pred_df,
            y_test,
            y_train,
            cat_cols=cat_cols,
            cat_to_int=cat_to_int,
            model_name=model_name,
            model_color=model_colors[model_idx],
            actual_color=actual_color,
            is_multi_model=len(model_names) > 1,
            n_history=n_history,
            show_legend=model_idx == 0 or len(model_names) > 1,
            line_width=line_width,
        )

    fig = apply_default_layout(
        fig,
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
    return fig


def _plot_forecast_multi_model(
    y_test: pl.DataFrame,
    y_preds: dict[str, pl.DataFrame],
    *,
    y_train: pl.DataFrame | None,
    coverage_rates: list[float] | None,
    n_history: int | None,
    color_palette: list[str] | None,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    width: int | None,
    height: int | None,
    line_width: float,
    band_opacity: float,
    show_transition: bool,
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

    Returns
    -------
    go.Figure
        Plotly figure with overlaid model forecasts.

    """
    eff_palette = color_palette if color_palette is not None else _PALETTE
    actual_color = eff_palette[2 % len(eff_palette)]
    _model_pal = eff_palette[3:] or eff_palette
    # Per-column colors for multivariate actual/train traces
    _actual_pal = eff_palette[3:] or eff_palette

    interval_pattern = re.compile(r"^.+_(lower|upper)_[\d.]+$")
    test_value_cols = [c for c in y_test.columns if c != "time"]
    plot_columns = test_value_cols

    model_names = list(y_preds.keys())
    colors = resolve_color_palette(_model_pal, len(model_names))

    fig = go.Figure()
    multi_col = len(plot_columns) > 1
    _col_colors = resolve_color_palette(_actual_pal, len(plot_columns)) if multi_col else []

    for _, col in enumerate(plot_columns):
        # Train data once
        if y_train is not None and col in y_train.columns:
            train_df = y_train
            if n_history is not None:
                train_df = train_df.tail(n_history)
            _ac = _col_colors[list(plot_columns).index(col)] if multi_col else actual_color
            _hex = _ac.lstrip("#")
            _rgb = tuple(int(_hex[i : i + 2], 16) for i in (0, 2, 4))
            _train_color = f"rgba({_rgb[0]}, {_rgb[1]}, {_rgb[2]}, 0.4)"
            _train_name = f"{col} (Train)"
            fig.add_trace(
                go.Scatter(
                    x=train_df["time"],
                    y=train_df[col],
                    mode="lines",
                    line={"color": _train_color, "width": line_width},
                    name=_train_name,
                    legendgroup=f"col_{col}" if multi_col else "actual",
                    legendrank=0,
                    hovertemplate="<b>Train</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>",
                )
            )

        # Interval bands first (behind everything else).
        for model_idx, (model_name, y_pred) in enumerate(y_preds.items()):
            model_color = colors[model_idx % len(colors)]
            pred_value_cols = [
                c for c in y_pred.columns if c not in ("time", "observed_time") and not interval_pattern.match(c)
            ]
            pred_col = col if col in pred_value_cols else pred_value_cols[0] if pred_value_cols else None
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
                    fig.add_trace(
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
                        )
                    )

        # Actual data
        _actual_group = f"col_{col}" if multi_col else "actual"
        _ac = _col_colors[list(plot_columns).index(col)] if multi_col else actual_color
        # Prepend last train point to close the gap between train and actual
        _x_actual = y_test["time"]
        _y_actual = y_test[col]
        if y_train is not None and col in y_train.columns:
            _x_actual = pl.concat([pl.Series("time", [y_train["time"][-1]]), _x_actual])
            _y_actual = pl.concat([pl.Series([y_train[col][-1]], dtype=_y_actual.dtype), _y_actual])
        fig.add_trace(
            go.Scatter(
                x=_x_actual,
                y=_y_actual,
                mode="lines",
                line={"color": _ac, "width": line_width},
                name=f"{col} (Actual)",
                legendgroup=_actual_group,
                legendrank=1,
                hovertemplate="<b>Actual</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>",
            )
        )

        # Forecast lines on top
        for model_idx, (model_name, y_pred) in enumerate(y_preds.items()):
            model_color = colors[model_idx % len(colors)]
            pred_value_cols = [
                c for c in y_pred.columns if c not in ("time", "observed_time") and not interval_pattern.match(c)
            ]
            pred_col = col if col in pred_value_cols else pred_value_cols[0] if pred_value_cols else None
            interval_base = pred_col if (pred_col is not None and pred_col in y_pred.columns) else col
            has_point = pred_col is not None and pred_col in y_pred.columns
            has_intervals = bool(
                coverage_rates and any(f"{interval_base}_lower_{rate}" in y_pred.columns for rate in coverage_rates)
            )

            if not has_point and not has_intervals:
                continue

            if has_point:
                assert pred_col is not None
                x_fc = y_pred["time"]
                y_fc = y_pred[pred_col]

                if show_transition and y_train is not None and col in y_train.columns:
                    x_fc = pl.concat([pl.Series("time", [y_train["time"][-1]]), y_pred["time"]])
                    y_fc = pl.concat([pl.Series([y_train[col][-1]], dtype=y_fc.dtype), y_fc])

                _fc_name = f"{col} ({model_name})"
                fig.add_trace(
                    go.Scatter(
                        x=x_fc,
                        y=y_fc,
                        mode="lines",
                        line={"color": model_color, "width": line_width},
                        name=_fc_name,
                        legendgroup=model_name,
                        legendrank=10 + model_idx * 100,
                        hovertemplate=(f"<b>{_fc_name}</b><br>Time: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>"),
                    )
                )

    title_default = title or "Forecast Comparison"
    x_label_default = x_label or "Time"
    y_label_default = y_label or "Value"

    fig = apply_default_layout(
        fig,
        title=title_default,
        x_label=x_label_default,
        y_label=y_label_default,
        width=width,
        height=height,
    )

    return fig


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
        model_preds: dict[str, pl.DataFrame] = y_pred  # type: ignore[assignment]
        model_names = list(model_preds.keys())
        model_colors = resolve_color_palette(_model_pal, len(model_names))
    else:
        model_preds = {"Forecast": y_pred}  # type: ignore[dict-item]
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
                pred_m = _extract_member_df(pred_df, suffix)
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
                pred_member = _extract_member_df(pred_df, suffix)
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
            first_pred_member = _extract_member_df(next(iter(model_preds.values())), suffix)
            cat_cols = [
                c
                for c in first_pred_member.columns
                if c not in ("time", "observed_time") and first_pred_member[c].dtype in (pl.String, pl.Categorical)
            ]

            for model_idx, (model_name, pred_df) in enumerate(model_preds.items()):
                pred_member = _extract_member_df(pred_df, suffix)
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
    prediction_mode: str = "numeric",
) -> go.Figure:
    """Plot forecast with panel data as faceted subplots.

    Supports both single-model (DataFrame) and multi-model (dict) predictions.
    Each subplot receives its own legend positioned inside the subplot area.
    When a panel group contains multiple members (multivariate), each member
    is drawn in a distinct colour with dashed lines for forecasts.

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

    # Detect multivariate panels (multiple members in any group)
    multi_member = any(len(cols) > 1 for cols in groups.values())

    if multi_member:
        all_flat_cols = [c for cols in groups.values() for c in cols]
        _, all_members = _group_panel_columns(all_flat_cols)
        member_palette = resolve_color_palette(_model_pal, len(all_members))
    else:
        all_members = []
        member_palette = []

    # Normalise y_pred into a model-name -> DataFrame mapping
    is_multi_model = isinstance(y_pred, dict)
    if is_multi_model:
        assert isinstance(y_pred, dict)
        model_preds: dict[str, pl.DataFrame] = y_pred  # type: ignore[assignment]
        model_names = list(model_preds.keys())
        model_colors = resolve_color_palette(_model_pal, len(model_names))
    else:
        assert isinstance(y_pred, pl.DataFrame)
        model_preds = {"Forecast": y_pred}
        model_names = ["Forecast"]
        model_colors = [forecast_color]

    n_groups = len(groups)
    n_rows = (n_groups + facet_n_cols - 1) // facet_n_cols
    n_cols_grid = min(n_groups, facet_n_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols_grid,
        subplot_titles=list(groups.keys()),
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    interval_pattern = re.compile(r"^.+_(lower|upper)_[\d.]+$")

    # Track which legend entries we have already shown globally
    # to avoid duplicates when multiple groups share the same labels.
    seen_legend: set[str] = set()

    for group_idx, (_, group_cols) in enumerate(groups.items()):
        row = group_idx // facet_n_cols + 1
        col_grid = group_idx % facet_n_cols + 1

        for col in group_cols:
            member = _member_name(col)

            # Resolve per-trace colours and labels
            if multi_member:
                member_idx = all_members.index(member)
                base_color = member_palette[member_idx]
                _hex = base_color.lstrip("#")
                _rgb = tuple(int(_hex[i : i + 2], 16) for i in (0, 2, 4))
                train_c: str = f"rgba({_rgb[0]}, {_rgb[1]}, {_rgb[2]}, 0.4)"
                actual_c = base_color
                train_label = f"{member} (Train)"
                actual_label = f"{member} (Actual)"
                lg_key: str | None = member
            else:
                train_c = history_color
                actual_c = actual_color
                train_label = "Train"
                actual_label = "Actual"
                lg_key = None

            # Train
            if y_train is not None and col in y_train.columns:
                train_df = y_train.tail(n_history) if n_history is not None else y_train
                _show = train_label not in seen_legend
                seen_legend.add(train_label)
                fig.add_trace(
                    go.Scatter(
                        x=train_df["time"],
                        y=train_df[col],
                        mode="lines",
                        line={"color": train_c, "width": line_width},
                        name=train_label,
                        showlegend=_show,
                        legendgroup=lg_key or "train",
                        legendrank=0,
                    ),
                    row=row,
                    col=col_grid,
                )

            # Interval bands first (behind Actual and forecast lines).
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
                            pi_label = (
                                f"{member} {m_name} ({rate:.0%} PI)" if is_multi_model else f"{member} ({rate:.0%} PI)"
                            )
                        else:
                            pi_label = f"{rate:.0%} PI" if not is_multi_model else f"{m_name} ({rate:.0%} PI)"
                        _show_pi = pi_label not in seen_legend
                        seen_legend.add(pi_label)
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
                        )

            # Actual
            if col in y_test.columns:
                _show_actual = actual_label not in seen_legend
                seen_legend.add(actual_label)
                fig.add_trace(
                    go.Scatter(
                        x=y_test["time"],
                        y=y_test[col],
                        mode="lines",
                        line={"color": actual_c, "width": line_width},
                        name=actual_label,
                        showlegend=_show_actual,
                        legendgroup=lg_key or "actual",
                        legendrank=1,
                    ),
                    row=row,
                    col=col_grid,
                )

            # Forecast lines on top of bands and actual.
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

                    _show_fc = fc_label not in seen_legend
                    seen_legend.add(fc_label)
                    fig.add_trace(
                        go.Scatter(
                            x=x_fc,
                            y=y_fc,
                            mode="lines",
                            line=line_spec,
                            name=fc_label,
                            showlegend=_show_fc,
                            legendgroup=lg_key or m_name,
                            legendrank=10 + m_idx * 100,
                        ),
                        row=row,
                        col=col_grid,
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

    return fig


def plot_time_weight(
    df: pl.DataFrame,
    *,
    weight_column: str = "time_weight",
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
    facet_n_cols : int, default=2
        Number of columns in facet grid when using panel groups.
    color_palette : list[str] | None, default=None
        Custom color palette as hex codes. If None, uses yohou palette.
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
    **kwargs : dict
        Additional styling parameters:
        - line_width : float, default=2.0
        - fill : bool, default=True (fills area under curve)
        - fill_opacity : float, default=0.3

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
    `plot_time_series` : Plot basic time series.
    """
    # Validate inputs
    validate_plotting_data(df)

    # Get styling params
    line_width = kwargs.get("line_width", 2.0)
    fill = kwargs.get("fill", True)
    fill_opacity = kwargs.get("fill_opacity", 0.3)

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

        n_groups = len(weight_panel_cols)

        # Get colors (one per unique member)
        if color_palette is None:
            color_palette = resolve_color_palette(None, len(all_members))

        # Create subplots (one per group)
        n_rows = (n_groups + facet_n_cols - 1) // facet_n_cols
        n_cols_grid = min(n_groups, facet_n_cols)
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols_grid,
            subplot_titles=list(weight_panel_cols.keys()),
            shared_xaxes=True,
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
        )

        seen_members: set[str] = set()
        for group_idx, (_, group_cols) in enumerate(weight_panel_cols.items()):
            row = group_idx // facet_n_cols + 1
            col_idx = group_idx % facet_n_cols + 1

            for col in group_cols:
                member_name = _member_name(col)
                member_idx = all_members.index(member_name)
                color = color_palette[member_idx % len(color_palette)]
                first_seen = member_name not in seen_members
                seen_members.add(member_name)

                # Convert hex to rgba for fill
                rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
                rgba_fill = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {fill_opacity})"

                fig.add_trace(
                    go.Scatter(
                        x=df["time"],
                        y=df[col],
                        mode="lines",
                        line={"color": color, "width": line_width},
                        fill="tozeroy" if fill else None,
                        fillcolor=rgba_fill if fill else None,
                        name=member_name,
                        showlegend=first_seen,
                        legendgroup=member_name,
                        hovertemplate=(f"{member_name}<br>Time: %{{x}}<br>Weight: %{{y:.3f}}<extra></extra>"),
                    ),
                    row=row,
                    col=col_idx,
                )

        # Update layout
        title_default = title or "Time Weights by Panel"
        fig.update_layout(
            title=title_default,
            height=height or (300 * n_rows),
            width=width,
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
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df[weight_column],
            mode="lines",
            line={"color": color, "width": line_width},
            fill="tozeroy" if fill else None,
            fillcolor=rgba_fill if fill else None,
            name="Time Weight",
            hovertemplate="Time: %{x}<br>Weight: %{y:.3f}<extra></extra>",
        )
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

    return fig


_INTERVAL_TO_STL_PERIOD: dict[str, int] = {
    "1h": 24,
    "1d": 7,
    "7d": 52,
    "1mo": 12,
    "2mo": 6,
    "3mo": 4,
    "6mo": 2,
}


def _compute_stl(
    series: pl.Series,
    *,
    period: int,
    trend_window: int | None = None,
    seasonal_window: int | None = None,
    low_pass_window: int | None = None,
    robust: bool = True,
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

    values = series.to_list()
    clean = pl.Series(values).interpolate().forward_fill().backward_fill()
    n_interpolated = sum(v is None or (isinstance(v, float) and np.isnan(v)) for v in values) - sum(
        v is None or (isinstance(v, float) and np.isnan(v)) for v in clean.to_list()
    )
    if n_interpolated > 0:
        warnings.warn(
            f"Interpolated {n_interpolated} NaN value(s) before STL decomposition.",
            UserWarning,
            stacklevel=3,
        )

    clean_np = clean.to_numpy().astype(float)

    stl_kwargs: dict = {"period": period, "robust": robust}
    if trend_window is not None:
        stl_kwargs["trend"] = trend_window if trend_window % 2 == 1 else trend_window + 1
    if seasonal_window is not None:
        stl_kwargs["seasonal"] = seasonal_window if seasonal_window % 2 == 1 else seasonal_window + 1
    if low_pass_window is not None:
        stl_kwargs["low_pass"] = low_pass_window if low_pass_window % 2 == 1 else low_pass_window + 1

    result = STL(clean_np, **stl_kwargs).fit()

    return {
        "observed": clean_np.tolist(),
        "trend": result.trend.tolist(),
        "seasonal": result.seasonal.tolist(),
        "residual": result.resid.tolist(),
        "seasonal_adjusted": (clean_np - result.seasonal).tolist(),
    }


def plot_components(
    y: pl.DataFrame,
    components: dict[str, pl.DataFrame] | list[str] | tuple[str, ...],
    *,
    columns: str | list[str] | None = None,
    show_original: bool = True,
    stl_kwargs: dict | None = None,
    color_palette: list[str] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """Plot time series decomposition as vertically stacked subplots.

    Displays the original series and its decomposed components (e.g. trend,
    seasonal, residual) in separate panels sharing the same time axis.

    There are two modes of operation:

    **Pre-computed mode** (default) -- pass *components* as a ``dict`` mapping
    component names to DataFrames produced by a Yohou decomposition pipeline.

    **STL mode** -- pass *components* as a ``list`` or ``tuple`` of component
    names (subset of ``"observed"``, ``"trend"``, ``"seasonal"``,
    ``"residual"``, ``"seasonal_adjusted"``).  The function runs
    ``statsmodels.tsa.seasonal.STL`` internally and renders the requested
    components.  Use *stl_kwargs* to configure the decomposition.

    Parameters
    ----------
    y : pl.DataFrame
        Original time series with ``"time"`` column.
    components : dict[str, pl.DataFrame] | list[str] | tuple[str, ...]
        **dict** -- mapping of component names to DataFrames (pre-computed
        mode).  Each DataFrame must have a ``"time"`` column plus value
        columns matching *y*.  Typical keys are ``"trend"``,
        ``"seasonality"``, ``"residual"``.

        **list/tuple of str** -- STL component names to compute and display.
        Valid names: ``"observed"``, ``"trend"``, ``"seasonal"``,
        ``"residual"``, ``"seasonal_adjusted"``.  When ``"observed"`` is
        included it is equivalent to ``show_original=True``.
    columns : str | list[str] | None, default=None
        Value columns to plot. If None, all numeric non-time columns of *y*
        are used.
    show_original : bool, default=True
        Include the original series as the first subplot.  In STL mode this
        is automatically set to ``True`` when ``"observed"`` appears in
        *components*.
    stl_kwargs : dict | None, default=None
        Extra keyword arguments forwarded to the internal STL computation.
        Only used in STL mode (ignored when *components* is a dict).
        Supported keys:

        - ``period`` (int | str): seasonal period (default ``"auto"``).
        - ``trend_window`` (int | None): trend smoother window length.
        - ``seasonal_window`` (int | None): seasonal smoother window length.
        - ``low_pass_window`` (int | None): low-pass filter window length.
        - ``robust`` (bool): use robust fitting (default ``True``).
    color_palette : list[str] | None, default=None
        Custom color palette. Falls back to the default yohou palette.
    title : str | None, default=None
        Plot title. Defaults to ``"Time Series Decomposition"`` (pre-computed)
        or ``"STL Decomposition"`` (STL mode).
    x_label : str | None, default=None
        X-axis label shown on the bottom subplot. Defaults to ``"Time"``.
    y_label : str | None, default=None
        Y-axis label.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    **kwargs : dict
        Additional styling:

        - line_width : float, default=2.0
        - line_dash : str, default="solid" (STL mode only)

    Returns
    -------
    go.Figure
        Plotly figure with vertically stacked subplots.

    Raises
    ------
    TypeError
        If *y* is not a Polars DataFrame.
    ValueError
        If DataFrames are empty, missing ``"time"`` column, *components*
        is empty, or unknown STL component names are given.
    ImportError
        When *components* is a list/tuple and ``statsmodels`` is not
        installed.

    Examples
    --------
    Pre-computed mode:

    >>> import polars as pl
    >>> from yohou.plotting import plot_components

    >>> dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
    >>> y = pl.DataFrame({"time": dates, "y": list(range(len(dates)))})
    >>> comps = {
    ...     "trend": pl.DataFrame({"time": dates, "y": [i * 0.5 for i in range(len(dates))]}),
    ...     "residual": pl.DataFrame({"time": dates, "y": [i * 0.5 for i in range(len(dates))]}),
    ... }
    >>> fig = plot_components(y, comps)
    >>> len(fig.data) >= 3
    True

    STL mode:

    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2018, 1, 1), pl.date(2022, 12, 31), "1mo", eager=True),
    ...     "y": [100 + 10 * (i % 12) + i * 0.5 for i in range(60)],
    ... })
    >>> fig = plot_components(df, ["trend", "seasonal"])  # doctest: +SKIP
    >>> len(fig.data) > 0  # doctest: +SKIP
    True

    See Also
    --------
    `plot_forecast` : Forecast visualization.
    `plot_seasonality` : Seasonal pattern analysis.
    """
    validate_plotting_data(y)
    line_width = kwargs.get("line_width", 2.0)
    line_dash = kwargs.get("line_dash", "solid")

    stl_mode = isinstance(components, list | tuple) and (not components or isinstance(components[0], str))

    # -- STL mode: compute decomposition, then convert to dict ---------------
    if stl_mode:
        components_list = list(components)

        # "observed" in the list maps to show_original
        if "observed" in components_list:
            show_original = True
            components_list.remove("observed")

        # Validate component names early
        valid_stl = {"trend", "seasonal", "residual", "seasonal_adjusted"}
        unknown = set(components_list) - valid_stl
        if unknown:
            all_valid = sorted(valid_stl | {"observed"})
            msg = f"Unknown components: {unknown}. Valid: {all_valid}"
            raise ValueError(msg)

        if not components_list and not show_original:
            msg = "components must contain at least one displayable component"
            raise ValueError(msg)

        value_cols = validate_plotting_data(y, columns=columns, exclude=["time"])

        components = _stl_to_component_dict(y, components_list, value_cols, stl_kwargs)
        title = title or "STL Decomposition"

        # Fall through to the shared dict plotting below

    # -- Dict validation -----------------------------------------------------
    if not isinstance(components, dict):
        msg = (
            "components must be a dict[str, pl.DataFrame] (pre-computed mode) or a list[str]/tuple[str, ...] (STL mode)"
        )
        raise TypeError(msg)

    if not components:
        msg = "components dict must be non-empty"
        raise ValueError(msg)
    for comp_df in components.values():
        validate_plotting_data(comp_df)

    value_cols = validate_plotting_data(y, columns=columns)

    # -- Build subplot structure (shared by both modes) ----------------------
    panel_names: list[str] = []
    if show_original:
        panel_names.append("Original")
    panel_names.extend(name.replace("_", " ").title() if stl_mode else name for name in components)

    n_rows = len(panel_names)
    fig = make_subplots(
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
            display_name = comp_name.replace("_", " ").title() if stl_mode else legend_col
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
                    legendgroup=display_name if stl_mode else legend_col,
                    showlegend=(comp_idx == 0 and not show_original),
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
        y_label=y_label,
        width=width,
        height=height or default_height,
    )

    # Show x-axis label on bottom subplot only
    x_label_text = x_label if x_label is not None else "Time"
    bottom_xaxis = f"xaxis{n_rows}" if n_rows > 1 else "xaxis"
    fig.layout[bottom_xaxis].title = {"text": x_label_text}

    return fig


def _stl_to_component_dict(
    y: pl.DataFrame,
    components: list[str],
    columns: list[str],
    stl_kwargs: dict | None,
) -> dict[str, pl.DataFrame]:
    """Compute STL decomposition and return as component DataFrames.

    Parameters
    ----------
    y : pl.DataFrame
        Input DataFrame with ``"time"`` column and numeric columns.
    components : list[str]
        STL component names to compute (e.g. ``"trend"``, ``"seasonal"``).
    columns : list[str]
        Numeric column names from *y* to decompose.
    stl_kwargs : dict | None
        Extra keyword arguments forwarded to `_compute_stl`.

    Returns
    -------
    dict[str, pl.DataFrame]
        Mapping from component name to a DataFrame with ``"time"`` plus
        one column per decomposed series.

    """
    stl_opts = stl_kwargs or {}
    period = stl_opts.get("period", "auto")
    trend_window = stl_opts.get("trend_window")
    seasonal_window = stl_opts.get("seasonal_window")
    low_pass_window = stl_opts.get("low_pass_window")
    robust = stl_opts.get("robust", True)

    # Resolve STL period
    if isinstance(period, str) and period == "auto":
        from yohou.utils.validation import check_interval_consistency  # noqa: PLC0415

        interval = check_interval_consistency(y)
        stl_period = _INTERVAL_TO_STL_PERIOD.get(interval)
        if stl_period is None:
            msg = (
                f"Cannot infer STL period for interval '{interval}'. "
                f"Supported intervals: {sorted(_INTERVAL_TO_STL_PERIOD)}. "
                f"Pass an explicit integer via stl_kwargs={{'period': <int>}}."
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
        )
        for comp in components:
            result_data[comp][col_name] = stl_result[comp]

    # Build a DataFrame per component
    time_col = y["time"]
    return {comp: pl.DataFrame({"time": time_col, **col_values}) for comp, col_values in result_data.items()}
