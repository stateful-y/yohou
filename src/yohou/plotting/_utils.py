"""Internal utilities for yohou plotting: colors, validation, and layout."""

from __future__ import annotations

import copy
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Literal

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler, FigureWidgetResampler
from plotly_resampler.aggregation.aggregation_interface import AbstractAggregator
from plotly_resampler.aggregation.gap_handler_interface import AbstractGapHandler
from plotly_resampler.aggregation.gap_handlers import NoGapHandler
from plotly_resampler.figure_resampler.figure_resampler_interface import (
    AbstractFigureAggregator,
)

from yohou.utils import inspect_panel

__all__ = [
    "DEFAULT_LAYOUT",
    "apply_default_layout",
    "config_context",
    "get_config",
    "palette_yohou",
    "panel_facet_figure",
    "resolve_color_palette",
    "set_config",
]


_global_config: dict = {
    "resampler": False,
    "resampler_n_shown_samples": None,
    "resampler_downsampler": None,
    "resampler_gap_handler": None,
    "resampler_trace_prefix_suffix": None,
    "resampler_show_mean_aggregation_size": None,
}


def set_config(
    *,
    resampler: bool | Literal["widget"] | None = None,
    resampler_n_shown_samples: int | None = None,
    resampler_downsampler: AbstractAggregator | None = None,
    resampler_gap_handler: AbstractGapHandler | None = None,
    resampler_trace_prefix_suffix: tuple[str, str] | None = None,
    resampler_show_mean_aggregation_size: bool | None = None,
) -> None:
    """Set global plotting configuration.

    Parameters
    ----------
    resampler : bool | Literal["widget"] | None, default=None
        Controls plotly-resampler usage for time-axis plots.

        - ``False`` — plain ``go.Figure`` (default).
        - ``True`` — ``FigureResampler`` (Dash-based callback server).
        - ``"widget"`` — ``FigureWidgetResampler`` (notebook-native widget).
        - ``None`` — leave current value unchanged.
    resampler_n_shown_samples : int | None, default=None
        Default number of samples shown per trace when resampling is active.
        ``None`` leaves current value unchanged (library default: 1000).
    resampler_downsampler : AbstractAggregator | None, default=None
        Default downsampling algorithm (e.g. ``MinMaxLTTB()``, ``LTTB()``).
        ``None`` leaves current value unchanged (library default: ``MinMaxLTTB()``).
    resampler_gap_handler : AbstractGapHandler | None, default=None
        Default gap detection strategy (e.g. ``MedDiffGapHandler()``, ``NoGapHandler()``).
        ``None`` leaves current value unchanged (library default: ``MedDiffGapHandler()``).
    resampler_trace_prefix_suffix : tuple[str, str] | None, default=None
        Prefix and suffix added to legend names of resampled traces.
        ``None`` leaves current value unchanged (library default: ``('[R] ', '')``).
    resampler_show_mean_aggregation_size : bool | None, default=None
        Whether to show the mean aggregation bin size as a legend suffix.
        ``None`` leaves current value unchanged (library default: ``True``).

    Examples
    --------
    >>> from yohou.plotting import set_config, get_config
    >>> set_config(resampler="widget")
    >>> get_config()["resampler"]
    'widget'
    """
    if resampler is not None:
        _global_config["resampler"] = resampler
    if resampler_n_shown_samples is not None:
        _global_config["resampler_n_shown_samples"] = resampler_n_shown_samples
    if resampler_downsampler is not None:
        _global_config["resampler_downsampler"] = resampler_downsampler
    if resampler_gap_handler is not None:
        _global_config["resampler_gap_handler"] = resampler_gap_handler
    if resampler_trace_prefix_suffix is not None:
        _global_config["resampler_trace_prefix_suffix"] = resampler_trace_prefix_suffix
    if resampler_show_mean_aggregation_size is not None:
        _global_config["resampler_show_mean_aggregation_size"] = resampler_show_mean_aggregation_size


def get_config() -> dict:
    """Return a copy of the current global plotting configuration.

    Returns
    -------
    dict
        Dictionary with current configuration values.

    Examples
    --------
    >>> from yohou.plotting import get_config
    >>> get_config()
    {'resampler': False}
    """
    return _global_config.copy()


@contextmanager
def config_context(
    *,
    resampler: bool | Literal["widget"] | None = None,
    resampler_n_shown_samples: int | None = None,
    resampler_downsampler: AbstractAggregator | None = None,
    resampler_gap_handler: AbstractGapHandler | None = None,
    resampler_trace_prefix_suffix: tuple[str, str] | None = None,
    resampler_show_mean_aggregation_size: bool | None = None,
) -> Generator[None, None, None]:
    """Context manager to temporarily override plotting configuration.

    Parameters
    ----------
    resampler : bool | Literal["widget"] | None, default=None
        Temporary resampler mode.  ``None`` leaves the current value
        unchanged.
    resampler_n_shown_samples : int | None, default=None
        Temporary number of shown samples.  ``None`` leaves unchanged.
    resampler_downsampler : AbstractAggregator | None, default=None
        Temporary downsampling algorithm.  ``None`` leaves unchanged.
    resampler_gap_handler : AbstractGapHandler | None, default=None
        Temporary gap handler.  ``None`` leaves unchanged.
    resampler_trace_prefix_suffix : tuple[str, str] | None, default=None
        Temporary legend prefix/suffix.  ``None`` leaves unchanged.
    resampler_show_mean_aggregation_size : bool | None, default=None
        Temporary aggregation size display.  ``None`` leaves unchanged.

    Examples
    --------
    >>> from yohou.plotting import config_context, get_config, set_config
    >>> set_config(resampler=False)
    >>> with config_context(resampler="widget"):
    ...     assert get_config()["resampler"] == "widget"
    >>> assert get_config()["resampler"] is False
    """
    old = _global_config.copy()
    try:
        set_config(
            resampler=resampler,
            resampler_n_shown_samples=resampler_n_shown_samples,
            resampler_downsampler=resampler_downsampler,
            resampler_gap_handler=resampler_gap_handler,
            resampler_trace_prefix_suffix=resampler_trace_prefix_suffix,
            resampler_show_mean_aggregation_size=resampler_show_mean_aggregation_size,
        )
        yield
    finally:
        _global_config.clear()
        _global_config.update(old)


def _get_resampler_mode(
    resampler: bool | Literal["widget"] | None,
) -> bool | Literal["widget"]:
    """Resolve the effective resampler mode.

    If *resampler* is ``None``, read from the global config; otherwise
    use the explicit value.
    """
    if resampler is None:
        return _global_config["resampler"]
    return resampler


def _build_resampler_kwargs() -> dict:
    """Build kwargs dict for FigureResampler/FigureWidgetResampler from config.

    Only includes keys that are explicitly set (not None), so the library
    uses its own defaults for anything left unset.
    """
    _config_to_kwarg = {
        "resampler_n_shown_samples": "default_n_shown_samples",
        "resampler_downsampler": "default_downsampler",
        "resampler_gap_handler": "default_gap_handler",
        "resampler_trace_prefix_suffix": "resampled_trace_prefix_suffix",
        "resampler_show_mean_aggregation_size": "show_mean_aggregation_size",
    }
    return {
        kwarg: _global_config[key]
        for key, kwarg in _config_to_kwarg.items()
        if _global_config[key] is not None
    }


def _create_figure(
    resampler: bool | Literal["widget"] | None = None,
    **kwargs,
) -> go.Figure:
    """Create a Plotly figure, optionally wrapped with plotly-resampler.

    Parameters
    ----------
    resampler : bool | Literal["widget"] | None, default=None
        Resampler mode.  ``None`` reads from :func:`get_config`.
    **kwargs
        Forwarded to ``go.Figure()``.

    Returns
    -------
    go.Figure
        A plain figure, ``FigureResampler``, or ``FigureWidgetResampler``.
    """
    mode = _get_resampler_mode(resampler)
    if mode == "widget":
        return FigureWidgetResampler(
            go.Figure(**kwargs),
            **_build_resampler_kwargs(),
        )
    if mode:
        return FigureResampler(
            go.Figure(**kwargs),
            **_build_resampler_kwargs(),
        )
    return go.Figure(**kwargs)


def _create_subplots(
    resampler: bool | Literal["widget"] | None = None,
    **subplots_kwargs,
) -> go.Figure:
    """Create subplots, optionally wrapped with plotly-resampler.

    Parameters
    ----------
    resampler : bool | Literal["widget"] | None, default=None
        Resampler mode.  ``None`` reads from :func:`get_config`.
    **subplots_kwargs
        Forwarded to ``plotly.subplots.make_subplots()``.

    Returns
    -------
    go.Figure
        A plain figure, ``FigureResampler``, or ``FigureWidgetResampler``.
    """
    fig = make_subplots(**subplots_kwargs)
    mode = _get_resampler_mode(resampler)
    if mode == "widget":
        return FigureWidgetResampler(
            fig,
            **_build_resampler_kwargs(),
        )
    if mode:
        return FigureResampler(
            fig,
            **_build_resampler_kwargs(),
        )
    return fig


def _fill_trace_kwargs(fig: go.Figure) -> dict:
    """Return extra ``add_trace`` kwargs for filled traces.

    When *fig* is a plotly-resampler figure, filled traces
    (``fill='tonexty'``, etc.) must use ``NoGapHandler`` to avoid
    visual artifacts from gap interleaving.

    Returns an empty dict for plain figures.
    """
    if isinstance(fig, AbstractFigureAggregator):
        return {"gap_handler": NoGapHandler()}
    return {}


def palette_yohou() -> dict[str, str]:
    """
    Return the yohou color palette.

    Returns
    -------
    dict[str, str]
        Dictionary mapping color names to hex codes.

    Examples
    --------
    >>> from yohou.plotting import palette_yohou
    >>> colors = palette_yohou()
    >>> colors["blue"]
    '#2563EB'
    >>> len(colors)
    12
    """
    return {
        "blue": "#2563EB",  # Primary blue
        "red": "#DC2626",  # Error/alert red
        "green": "#059669",  # Success green
        "purple": "#7C3AED",  # Accent purple
        "orange": "#EA580C",  # Warning orange
        "pink": "#DB2777",  # Pink accent
        "yellow": "#CA8A04",  # Gold/yellow
        "indigo": "#4F46E5",  # Indigo accent
        "teal": "#0D9488",  # Teal accent
        "cyan": "#0891B2",  # Cyan accent
        "gray": "#64748B",  # Neutral gray
        "slate": "#475569",  # Dark slate
    }


def get_color_sequence(n: int | None = None) -> list[str]:
    """
    Get color sequence for plotting multiple series.

    Parameters
    ----------
    n : int | None, default=None
        Number of colors to return. If None, returns all colors.
        If n exceeds available colors, cycles through the palette.

    Returns
    -------
    list[str]
        List of color hex codes.

    Examples
    --------
    >>> from yohou.plotting._utils import get_color_sequence
    >>> colors = get_color_sequence(3)
    >>> len(colors)
    3
    >>> colors[0]
    '#2563EB'

    >>> # Get all colors
    >>> all_colors = get_color_sequence()
    >>> len(all_colors)
    12

    >>> # Cycle through palette for many series
    >>> many_colors = get_color_sequence(25)
    >>> len(many_colors)
    25
    """
    colors = list(palette_yohou().values())
    if n is None:
        return colors
    return [colors[i % len(colors)] for i in range(n)]


def resolve_color_palette(color_palette: list[str] | None, n: int) -> list[str]:
    """Resolve a user-provided color palette or fall back to the default.

    When *color_palette* is ``None`` the default yohou palette is used.
    When the palette has fewer colours than *n*, the colours are cycled.

    Parameters
    ----------
    color_palette : list[str] | None
        User-provided colour hex codes, or ``None`` for the default.
    n : int
        Number of colours needed.

    Returns
    -------
    list[str]
        List of exactly *n* colour hex codes.

    Examples
    --------
    >>> from yohou.plotting._utils import resolve_color_palette
    >>> resolve_color_palette(None, 2)
    ['#2563EB', '#DC2626']

    >>> resolve_color_palette(["red", "blue"], 4)
    ['red', 'blue', 'red', 'blue']
    """
    if color_palette is None:
        return get_color_sequence(n)
    return [color_palette[i % len(color_palette)] for i in range(n)]


def _normalize_y_pred(
    y_pred: pl.DataFrame | dict[str, pl.DataFrame],
    default_name: str = "Forecast",
) -> dict[str, pl.DataFrame]:
    """Normalise y_pred to a ``{name: DataFrame}`` dictionary.

    Parameters
    ----------
    y_pred : pl.DataFrame or dict of str to pl.DataFrame
        Prediction(s) to normalise.
    default_name : str, default="Forecast"
        Key used when *y_pred* is a single DataFrame.

    Returns
    -------
    dict[str, pl.DataFrame]
        Normalised dictionary.

    Raises
    ------
    TypeError
        If *y_pred* is neither a DataFrame nor a dict.
    """
    if isinstance(y_pred, pl.DataFrame):
        return {default_name: y_pred}
    if isinstance(y_pred, dict):
        return y_pred
    msg = f"y_pred must be pl.DataFrame or dict, got {type(y_pred).__name__}"
    raise TypeError(msg)


DEFAULT_LAYOUT = {
    "template": "plotly_white",
    "font": {"family": "Arial, sans-serif", "size": 12, "color": "#1e293b"},
    "title": {"font": {"size": 16, "color": "#0f172a"}, "x": 0.5, "xanchor": "center"},
    "margin": {"l": 60, "r": 40, "t": 60, "b": 60},
    "hovermode": "closest",
    "hoverlabel": {"bgcolor": "white", "font_size": 11, "font_family": "Arial"},
    "xaxis": {
        "showgrid": True,
        "gridcolor": "#e2e8f0",
        "gridwidth": 1,
        "zeroline": False,
        "showline": True,
        "linecolor": "#cbd5e1",
    },
    "yaxis": {
        "showgrid": True,
        "gridcolor": "#e2e8f0",
        "gridwidth": 1,
        "zeroline": False,
        "showline": True,
        "linecolor": "#cbd5e1",
    },
    "legend": {
        "bgcolor": "rgba(255,255,255,0.8)",
        "bordercolor": "#cbd5e1",
        "borderwidth": 1,
        "font": {"size": 11},
    },
}


def apply_default_layout(
    fig: go.Figure,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """
    Apply default layout configuration to a figure.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to apply layout to.
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
        Figure with applied layout.

    Examples
    --------
    >>> import plotly.graph_objects as go
    >>> from yohou.plotting._utils import apply_default_layout
    >>> fig = go.Figure()
    >>> fig = apply_default_layout(fig, title="Test", x_label="Time")
    >>> fig.layout.title.text
    'Test'
    """
    layout_update = copy.deepcopy(DEFAULT_LAYOUT)

    if title is not None:
        layout_update["title"]["text"] = title  # type: ignore[index]
    if x_label is not None:
        layout_update["xaxis"]["title"] = x_label  # type: ignore[index]
    if y_label is not None:
        layout_update["yaxis"]["title"] = y_label  # type: ignore[index]
    if width is not None:
        layout_update["width"] = width  # type: ignore[invalid-assignment]
    if height is not None:
        layout_update["height"] = height  # type: ignore[invalid-assignment]

    fig.update_layout(layout_update)
    return fig


def resolve_panel_columns(
    df: pl.DataFrame,
    panel_group_names: list[str] | None = None,
    columns: str | list[str] | None = None,
) -> list[str]:
    """Resolve which panel columns to plot.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with panel columns (``group__member`` pattern).
    panel_group_names : list[str] | None, default=None
        Group prefixes to include.  If ``None`` or empty, all groups
        are included.
    columns : str | list[str] | None, default=None
        Member names (postfixes after ``__``) to include within the
        selected groups.  If ``None``, all members of each group are
        included.

    Returns
    -------
    list[str]
        Ordered list of full panel column names.

    Raises
    ------
    ValueError
        When no panel columns match the requested groups/members.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting._utils import resolve_panel_columns
    >>> df = pl.DataFrame({
    ...     "time": [1, 2],
    ...     "sales__a": [10, 20],
    ...     "sales__b": [30, 40],
    ... })
    >>> resolve_panel_columns(df)
    ['sales__a', 'sales__b']

    >>> resolve_panel_columns(df, columns="a")
    ['sales__a']

    >>> resolve_panel_columns(df, panel_group_names=["sales"], columns=["b"])
    ['sales__b']
    """
    _, panels = inspect_panel(df)
    if isinstance(columns, str):
        columns = [columns]
    cols: list[str] = []
    for prefix, members in panels.items():
        if not panel_group_names or prefix in panel_group_names:
            if columns is not None:
                for member in members:
                    _, _, postfix = member.partition("__")
                    if postfix in columns:
                        cols.append(member)
            else:
                cols.extend(members)
    if not cols:
        if columns is not None:
            msg = f"No panel columns found for groups={panel_group_names} with members={columns}"
        else:
            msg = f"No panel columns found for groups: {panel_group_names}"
        raise ValueError(msg)
    return cols


def _group_panel_columns(
    panel_cols: list[str],
) -> tuple[dict[str, list[str]], list[str]]:
    """Group panel columns by prefix and collect unique member names.

    Parameters
    ----------
    panel_cols : list[str]
        Flat list of panel column names (e.g. ``["T3__a", "T3__b", "T4__a"]``).

    Returns
    -------
    groups : dict[str, list[str]]
        Mapping from group prefix to its full column names,
        preserving insertion order.
    all_members : list[str]
        Unique member postfixes in first-seen order, usable as a
        stable colour index across groups.

    Examples
    --------
    >>> _group_panel_columns(["T3__a", "T3__b", "T4__a", "T4__b"])
    ({'T3': ['T3__a', 'T3__b'], 'T4': ['T4__a', 'T4__b']}, ['a', 'b'])

    >>> _group_panel_columns(["plain_col"])
    ({'plain_col': ['plain_col']}, ['plain_col'])
    """
    groups: dict[str, list[str]] = {}
    all_members: list[str] = []
    for col in panel_cols:
        prefix, sep, member = col.partition("__")
        key = prefix if sep else col
        m = member if sep else col
        groups.setdefault(key, []).append(col)
        if m not in all_members:
            all_members.append(m)
    return groups, all_members


def _member_name(col: str) -> str:
    """Return the member postfix of a panel column (part after ``__``).

    Parameters
    ----------
    col : str
        Full panel column name.

    Returns
    -------
    str
        Member postfix, or the column name itself when there is no
        ``__`` separator.
    """
    _, sep, member = col.partition("__")
    return member if sep else col


def panel_facet_figure(
    df: pl.DataFrame,
    render_fn: Callable[[go.Figure, pl.DataFrame, str, int, int, int], None],
    *,
    panel_group_names: list[str] | None = None,
    columns: str | list[str] | None = None,
    facet_n_cols: int = 2,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    row_height: int = 300,
    shared_xaxes: bool = True,
    resampler: bool | Literal["widget"] | None = None,
) -> go.Figure:
    """Create a faceted subplot figure for panel data.

    Panel columns are grouped by their group prefix (the part before
    ``__``).  Each group gets one subplot titled with the group name.
    Within each group, the *render_fn* callback is invoked once per
    member column with a sub-DataFrame containing ``"time"`` and the
    member column renamed to its member name (part after ``__``).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with panel columns.
    render_fn : Callable
        ``(fig, sub_df, member_name, member_idx, row, col) -> None``.
        *member_name* is the member postfix (e.g. ``"a"`` for
        column ``"y__a"``).  *member_idx* is the index of that
        member among all unique members, guaranteeing consistent
        colouring across groups.
    panel_group_names : list[str] | None, default=None
        Group prefixes to include (``None`` means all).
    columns : str | list[str] | None, default=None
        Member names (postfixes after ``__``) to include within the
        selected groups.  If ``None``, all members are plotted.
    facet_n_cols : int, default=2
        Number of columns in the facet grid.
    title : str | None, default=None
        Figure title.
    x_label : str | None, default=None
        X-axis label.
    y_label : str | None, default=None
        Y-axis label.
    width : int | None, default=None
        Figure width in pixels.
    height : int | None, default=None
        Figure height in pixels.  Defaults to ``row_height * n_rows``.
    row_height : int, default=300
        Height per facet row when *height* is ``None``.
    shared_xaxes : bool, default=True
        Share x-axes across all subplots.

    Returns
    -------
    go.Figure
        Plotly figure with faceted panel subplots.

    Examples
    --------
    >>> import polars as pl, plotly.graph_objects as go
    >>> from yohou.plotting._utils import panel_facet_figure
    >>> df = pl.DataFrame({
    ...     "time": [1, 2, 3],
    ...     "y__a": [10, 20, 30],
    ...     "y__b": [40, 50, 60],
    ... })
    >>> def render(fig, sub_df, name, idx, row, col):
    ...     base = [c for c in sub_df.columns if c != "time"][0]
    ...     fig.add_trace(
    ...         go.Scatter(x=sub_df["time"], y=sub_df[base], name=name),
    ...         row=row,
    ...         col=col,
    ...     )
    >>> fig = panel_facet_figure(df, render, title="Panel Demo")
    >>> len(fig.data)
    2
    """
    panel_cols = resolve_panel_columns(df, panel_group_names, columns)
    groups, all_members = _group_panel_columns(panel_cols)

    n_groups = len(groups)
    n_cols_grid = min(n_groups, facet_n_cols)
    n_rows = (n_groups + n_cols_grid - 1) // n_cols_grid

    fig = _create_subplots(
        resampler,
        rows=n_rows,
        cols=n_cols_grid,
        subplot_titles=list(groups.keys()),
        shared_xaxes=shared_xaxes,
        vertical_spacing=max(0.04, 0.3 / n_rows),
        horizontal_spacing=0.08,
    )

    for group_idx, (_, group_cols) in enumerate(groups.items()):
        row = group_idx // n_cols_grid + 1
        col_idx = group_idx % n_cols_grid + 1

        for col in group_cols:
            member = _member_name(col)
            member_idx = all_members.index(member)

            sub_df = df.select("time", pl.col(col).alias(member))
            render_fn(fig, sub_df, member, member_idx, row, col_idx)

    default_height = max(row_height * n_rows, 400)

    fig = apply_default_layout(
        fig,
        title=title,
        x_label=x_label,
        y_label=y_label,
        width=width,
        height=height or default_height,
    )

    return fig
