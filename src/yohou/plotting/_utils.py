"""Internal utilities for yohou plotting: colors, validation, and layout."""

import copy
from collections.abc import Callable
from typing import Any

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from yohou.utils import inspect_panel

__all__ = [
    "DEFAULT_LAYOUT",
    "apply_default_layout",
    "palette_yohou",
    "panel_facet_figure",
    "resolve_color_palette",
]


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
    layout_update: dict[str, Any] = copy.deepcopy(DEFAULT_LAYOUT)

    if title is not None:
        layout_update["title"]["text"] = title  # ty: ignore[invalid-assignment]
    if x_label is not None:
        layout_update["xaxis"]["title"] = x_label  # ty: ignore[invalid-assignment]
    if y_label is not None:
        layout_update["yaxis"]["title"] = y_label  # ty: ignore[invalid-assignment]
    if width is not None:
        layout_update["width"] = width  # ty: ignore[invalid-assignment]
    if height is not None:
        layout_update["height"] = height  # ty: ignore[invalid-assignment]

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

    fig = make_subplots(
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
