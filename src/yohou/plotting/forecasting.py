"""Forecast visualization functions."""

import re

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from yohou.plotting._utils import (
    apply_default_layout,
    palette_yohou,
    resolve_color_palette,
)
from yohou.utils import inspect_locality, validate_plotting_data

# The full palette list is used as the default effective palette in every
# plot_forecast code path. Slots 0/1/2 are reserved for the three semantic
# traces (history, single-model forecast, actual); slot 3+ are model colors.
_PALETTE = list(palette_yohou().values())

__all__ = [
    "plot_components",
    "plot_forecast",
    "plot_time_weight",
]


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
    plot_residual_time_series : Plot residual diagnostics.
    plot_model_comparison_bar : Grouped bar chart for scorer comparison.
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
    history_color = eff_palette[0]
    forecast_color = eff_palette[1 % len(eff_palette)]
    actual_color = eff_palette[2 % len(eff_palette)]
    line_width = kwargs.get("line_width", 2.0)
    band_opacity = kwargs.get("band_opacity", 0.25)
    show_transition = kwargs.get("show_transition", True)

    # Detect panel data
    _, y_test_panels = inspect_locality(y_test)
    is_panel = bool(y_test_panels)

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
            actual_c, forecast_c = col_color, col_color
            f_dash: str | None = "dash"
            train_name: str | None = "Train" if col_idx == 0 else None
            train_show = col_idx == 0
            actual_name = f"{col} (Actual)"
            forecast_name = f"{col} (Forecast)"
        else:
            actual_c, forecast_c = actual_color, forecast_color
            f_dash = None
            train_name = "Train"
            train_show = True
            actual_name = "Actual"
            forecast_name = "Forecast"

        # Compute pred_col early so interval rendering can hoist before Actual.
        pred_col = col if col in pred_value_cols else (pred_value_cols[0] if pred_value_cols else None)
        fc_group = f"forecast_{col}" if multi_col else "forecast"

        # Training data
        if y_train is not None and col in y_train.columns:
            train_df = y_train.tail(n_history) if n_history is not None else y_train
            fig.add_trace(
                go.Scatter(
                    x=train_df["time"],
                    y=train_df[col],
                    mode="lines",
                    line={"color": history_color, "width": line_width},
                    name=train_name,
                    showlegend=train_show,
                    legendgroup="train",
                    legendrank=1,
                    hovertemplate=f"<b>{col} Train</b><br>Time: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>",
                )
            )

        # Prediction intervals – rendered before Actual so the actual line sits on top.
        # Works even when there is no point forecast (predict_interval-only DataFrames).
        # legendrank >= 3 places them after Train (1) and Actual (2) in the legend.
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
                    pi_name = f"{col} ({rate:.0%} PI)" if multi_col else f"Forecast ({rate:.0%} PI)"
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
                            legendrank=3 + sort_idx,
                            hoverinfo="skip",
                        )
                    )

        # Actual test data
        fig.add_trace(
            go.Scatter(
                x=y_test["time"],
                y=y_test[col],
                mode="lines",
                line={"color": actual_c, "width": line_width},
                name=actual_name,
                legendgroup=f"actual_{col}" if multi_col else "actual",
                legendrank=2,
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
    history_color = eff_palette[0]
    actual_color = eff_palette[2 % len(eff_palette)]
    _model_pal = eff_palette[3:] or eff_palette

    interval_pattern = re.compile(r"^.+_(lower|upper)_[\d.]+$")
    test_value_cols = [c for c in y_test.columns if c != "time"]
    plot_columns = test_value_cols

    model_names = list(y_preds.keys())
    colors = resolve_color_palette(_model_pal, len(model_names))

    fig = go.Figure()

    for col in plot_columns:
        # Train data once
        if y_train is not None and col in y_train.columns:
            train_df = y_train
            if n_history is not None:
                train_df = train_df.tail(n_history)
            fig.add_trace(
                go.Scatter(
                    x=train_df["time"],
                    y=train_df[col],
                    mode="lines",
                    line={"color": history_color, "width": line_width},
                    name=f"{col} (Train)" if len(plot_columns) > 1 else "Train",
                    legendgroup="train",
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
            for rate in sorted(coverage_rates):
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
                            hoverinfo="skip",
                        )
                    )

        # Actual data
        fig.add_trace(
            go.Scatter(
                x=y_test["time"],
                y=y_test[col],
                mode="lines",
                line={"color": actual_color, "width": line_width},
                name=f"{col} (Actual)" if len(plot_columns) > 1 else "Actual",
                legendgroup="actual",
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

                fig.add_trace(
                    go.Scatter(
                        x=x_fc,
                        y=y_fc,
                        mode="lines",
                        line={"color": model_color, "width": line_width},
                        name=model_name,
                        legendgroup=model_name,
                        hovertemplate=(f"<b>{model_name}</b><br>Time: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>"),
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
) -> go.Figure:
    """Plot forecast with panel data as faceted subplots.

    Supports both single-model (DataFrame) and multi-model (dict) predictions.

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

    Returns
    -------
    go.Figure
        Plotly figure with faceted panel subplots.

    """
    eff_palette = color_palette if color_palette is not None else _PALETTE
    history_color = eff_palette[0]
    actual_color = eff_palette[2 % len(eff_palette)]
    forecast_color = eff_palette[1 % len(eff_palette)]
    _model_pal = eff_palette[3:] or eff_palette

    _, test_panels = inspect_locality(y_test)

    # Get all panel columns (one per group member)
    all_panel_cols: list[str] = []
    for prefix, cols in test_panels.items():
        if panel_group_names is None or prefix in panel_group_names:
            all_panel_cols.extend(cols)

    if not all_panel_cols:
        msg = f"No panel columns found for groups: {panel_group_names}"
        raise ValueError(msg)

    # Normalise y_pred into a model-name → DataFrame mapping
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

    n_panels = len(all_panel_cols)
    n_rows = (n_panels + facet_n_cols - 1) // facet_n_cols
    n_cols_grid = min(n_panels, facet_n_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols_grid,
        subplot_titles=[c.replace("__", " \u2013 ") for c in all_panel_cols],
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    interval_pattern = re.compile(r"^.+_(lower|upper)_[\d.]+$")

    for idx, col in enumerate(all_panel_cols):
        row = idx // facet_n_cols + 1
        col_grid = idx % facet_n_cols + 1

        # Train
        if y_train is not None and col in y_train.columns:
            train_df = y_train.tail(n_history) if n_history is not None else y_train
            fig.add_trace(
                go.Scatter(
                    x=train_df["time"],
                    y=train_df[col],
                    mode="lines",
                    line={"color": history_color, "width": line_width},
                    name="Train" if idx == 0 else None,
                    showlegend=(idx == 0),
                    legendgroup="train",
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
            _hex = m_color.lstrip("#")
            rgb = tuple(int(_hex[i : i + 2], 16) for i in (0, 2, 4))
            for rate in sorted(coverage_rates):
                lower_c = f"{interval_base}_lower_{rate}"
                upper_c = f"{interval_base}_upper_{rate}"
                if lower_c in m_pred.columns and upper_c in m_pred.columns:
                    rgba = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {band_opacity})"
                    t = m_pred["time"].to_list()
                    y_upper = m_pred[upper_c].to_list()
                    y_lower = m_pred[lower_c].to_list()
                    x_band = t + t[::-1]
                    y_band = y_upper + y_lower[::-1]
                    pi_label = f"{rate:.0%} PI" if not is_multi_model else f"{m_name} ({rate:.0%} PI)"
                    fig.add_trace(
                        go.Scatter(
                            x=x_band,
                            y=y_band,
                            fill="toself",
                            fillcolor=rgba,
                            mode="lines",
                            line={"width": 0, "color": rgba},
                            name=pi_label if idx == 0 else None,
                            showlegend=(idx == 0),
                            legendgroup=m_name,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col_grid,
                    )

        # Actual
        if col in y_test.columns:
            fig.add_trace(
                go.Scatter(
                    x=y_test["time"],
                    y=y_test[col],
                    mode="lines",
                    line={"color": actual_color, "width": line_width},
                    name="Actual" if idx == 0 else None,
                    showlegend=(idx == 0),
                    legendgroup="actual",
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

                fig.add_trace(
                    go.Scatter(
                        x=x_fc,
                        y=y_fc,
                        mode="lines",
                        line={"color": m_color, "width": line_width},
                        name=m_name if idx == 0 else None,
                        showlegend=(idx == 0),
                        legendgroup=m_name,
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
    plot_time_series : Plot basic time series.
    """
    # Validate inputs
    validate_plotting_data(df)

    # Get styling params
    line_width = kwargs.get("line_width", 2.0)
    fill = kwargs.get("fill", True)
    fill_opacity = kwargs.get("fill_opacity", 0.3)

    # Detect panel data structure
    _, panel_groups = inspect_locality(df)

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

        # Get all weight columns to plot
        all_weight_cols: list[str] = []
        for cols in weight_panel_cols.values():
            all_weight_cols.extend(cols)

        n_groups = len(all_weight_cols)

        # Get colors
        if color_palette is None:
            color_palette = resolve_color_palette(None, n_groups)

        # Create subplots
        n_rows = (n_groups + facet_n_cols - 1) // facet_n_cols
        n_cols_grid = min(n_groups, facet_n_cols)
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols_grid,
            subplot_titles=[c.replace("__", " \u2013 ") for c in all_weight_cols],
            shared_xaxes=True,
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
        )

        for idx, col in enumerate(all_weight_cols):
            row = idx // facet_n_cols + 1
            col_idx = idx % facet_n_cols + 1
            color = color_palette[idx % len(color_palette)]

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
                    name=col.replace("__", " \u2013 "),
                    showlegend=False,
                    hovertemplate=(f"{col}<br>Time: %{{x}}<br>Weight: %{{y:.3f}}<extra></extra>"),
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


def plot_components(
    y: pl.DataFrame,
    components: dict[str, pl.DataFrame],
    *,
    columns: str | list[str] | None = None,
    show_original: bool = True,
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

    Parameters
    ----------
    y : pl.DataFrame
        Original time series with ``"time"`` column.
    components : dict[str, pl.DataFrame]
        Mapping of component names to DataFrames. Each DataFrame must have a
        ``"time"`` column plus value columns matching *y*. Typical keys are
        ``"trend"``, ``"seasonality"``, ``"residual"``.
    columns : str | list[str] | None, default=None
        Value columns to plot. If None, all numeric non-time columns of *y*
        are used.
    show_original : bool, default=True
        Include the original series as the first subplot.
    color_palette : list[str] | None, default=None
        Custom color palette. Falls back to the default yohou palette.
    title : str | None, default=None
        Plot title. Defaults to ``"Time Series Decomposition"``.
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

    Returns
    -------
    go.Figure
        Plotly figure with vertically stacked subplots.

    Raises
    ------
    TypeError
        If *y* is not a Polars DataFrame.
    ValueError
        If DataFrames are empty, missing ``"time"`` column, or *components*
        is empty.

    Examples
    --------
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

    See Also
    --------
    plot_forecast : Forecast visualization.
    plot_seasonality : Seasonal pattern analysis.
    """
    validate_plotting_data(y)
    if not components:
        msg = "components dict must be non-empty"
        raise ValueError(msg)
    for comp_df in components.values():
        validate_plotting_data(comp_df)

    value_cols = validate_plotting_data(y, columns=columns)
    line_width = kwargs.get("line_width", 2.0)

    # Build subplot structure
    panel_names: list[str] = []
    if show_original:
        panel_names.append("Original")
    panel_names.extend(components.keys())

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
                        line={"color": colors[i % len(colors)], "width": line_width},
                        name=col if len(value_cols) > 1 else "Original",
                        legendgroup=col,
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )

    # Component panels
    for comp_idx, (comp_name, comp_df) in enumerate(components.items()):
        row = comp_idx + 1 + row_offset
        for i, col in enumerate(value_cols):
            if col in comp_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=comp_df["time"],
                        y=comp_df[col],
                        mode="lines",
                        line={"color": colors[i % len(colors)], "width": line_width},
                        name=f"{col} ({comp_name})" if len(value_cols) > 1 else comp_name,
                        legendgroup=col,
                        showlegend=(comp_idx == 0 and not show_original) or False,
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
