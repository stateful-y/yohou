"""Internal utilities for yohou plotting: colors, validation, and layout."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

try:
    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots
except ImportError as e:
    msg = "plotly is required for yohou plotting. Install: pip install yohou[plotting]"
    raise ImportError(msg) from e

if TYPE_CHECKING:
    from plotly_resampler.aggregation.aggregation_interface import AbstractAggregator
    from plotly_resampler.aggregation.gap_handler_interface import AbstractGapHandler

from yohou.utils import inspect_panel

__all__ = [
    "DEFAULT_LAYOUT",
    "DEFAULT_WIDTH",
    "LINE_DASH_SEQUENCE",
    "LegendTracker",
    "PanelColorManager",
    "RenderContext",
    "apply_default_layout",
    "config_context",
    "get_color_sequence",
    "get_config",
    "grouped_legend_kwargs",
    "linked_legendgroup_kwargs",
    "palette_yohou",
    "panel_facet_figure",
    "resolve_color_palette",
    "resolve_panel_columns",
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

LINE_DASH_SEQUENCE: list[str] = [
    "solid",
    "dash",
    "dot",
    "dashdot",
    "longdash",
    "longdashdot",
]


@dataclass(frozen=True)
class RenderContext:
    """Typed context passed to panel facet render callbacks.

    Replaces the previous 6-positional-arg callback signature with
    named, self-documenting fields.

    Parameters
    ----------
    fig : go.Figure
        The subplots figure to add traces to.
    sub_df : pl.DataFrame
        Subset DataFrame with ``"time"`` plus one value column.
    display_name : str
        The name of the overlaid entity (member name when
        ``facet_by="group"``, group name when ``facet_by="member"``).
    entity_idx : int
        Index of the overlaid entity in the color palette.
    row : int
        Subplot row (1-indexed).
    col : int
        Subplot column (1-indexed).
    facet_by : str
        ``"group"`` or ``"member"`` - indicates the faceting axis.
    facet_name : str
        The facet axis value (group name when ``facet_by="group"``,
        member name when ``facet_by="member"``).
    group_name : str
        Panel group name, always available for context.
    member_name : str
        Panel member name, always available for context.

    See Also
    --------
    [`panel_facet_figure`][yohou.plotting._utils.panel_facet_figure] : Factory that creates faceted figures and passes ``RenderContext`` to callbacks.
    [`LegendTracker`][yohou.plotting._utils.LegendTracker] : De-duplicates legend entries across subplots.
    """

    fig: go.Figure
    sub_df: pl.DataFrame
    display_name: str
    entity_idx: int
    row: int
    col: int
    facet_by: str
    facet_name: str
    group_name: str
    member_name: str


def _subplot_spacing(n: int, base: float = 0.3, floor: float = 0.04) -> float:
    """Compute adaptive subplot spacing that decreases with row/column count.

    Parameters
    ----------
    n : int
        Number of rows or columns.
    base : float, default=0.3
        Numerator of the spacing formula.
    floor : float, default=0.04
        Minimum spacing value.

    Returns
    -------
    float
        ``max(floor, base / n)``
    """
    return max(floor, base / max(n, 1))


"""Ordered Plotly dash styles for distinguishing multiple line series."""


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

        - ``False`` - plain ``go.Figure`` (default).
        - ``True`` - ``FigureResampler`` (Dash-based callback server).
        - ``"widget"`` - ``FigureWidgetResampler`` (notebook-native widget).
        - ``None`` - leave current value unchanged.
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
    >>> set_config(resampler=False)

    See Also
    --------
    [`get_config`][yohou.plotting.get_config] : Read the current configuration.
    [`config_context`][yohou.plotting.config_context] : Temporarily override configuration.
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
    >>> get_config()["resampler"]
    False

    See Also
    --------
    [`set_config`][yohou.plotting.set_config] : Set global configuration values.
    [`config_context`][yohou.plotting.config_context] : Temporarily override configuration.
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

    See Also
    --------
    [`set_config`][yohou.plotting.set_config] : Set global configuration values.
    [`get_config`][yohou.plotting.get_config] : Read the current configuration.
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
    return {kwarg: _global_config[key] for key, kwarg in _config_to_kwarg.items() if _global_config[key] is not None}


def _create_figure(
    resampler: bool | Literal["widget"] | None = None,
    **kwargs,
) -> go.Figure:
    """Create a Plotly figure, optionally wrapped with plotly-resampler.

    Parameters
    ----------
    resampler : bool | Literal["widget"] | None, default=None
        Resampler mode.  ``None`` reads from `get_config`.
    **kwargs
        Forwarded to ``go.Figure()``.

    Returns
    -------
    go.Figure
        A plain figure, ``FigureResampler``, or ``FigureWidgetResampler``.
    """
    mode = _get_resampler_mode(resampler)
    if mode == "widget":
        try:
            from plotly_resampler import FigureWidgetResampler  # noqa: PLC0415
        except ImportError as e:
            msg = "plotly-resampler is required for resampler mode. Install: pip install plotly-resampler"
            raise ImportError(msg) from e
        return FigureWidgetResampler(
            go.Figure(**kwargs),
            **_build_resampler_kwargs(),
        )
    if mode:
        try:
            from plotly_resampler import FigureResampler  # noqa: PLC0415
        except ImportError as e:
            msg = "plotly-resampler is required for resampler mode. Install: pip install plotly-resampler"
            raise ImportError(msg) from e
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
        Resampler mode.  ``None`` reads from `get_config`.
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
        try:
            from plotly_resampler import FigureWidgetResampler  # noqa: PLC0415
        except ImportError as e:
            msg = "plotly-resampler is required for resampler mode. Install: pip install plotly-resampler"
            raise ImportError(msg) from e
        return FigureWidgetResampler(
            fig,
            **_build_resampler_kwargs(),
        )
    if mode:
        try:
            from plotly_resampler import FigureResampler  # noqa: PLC0415
        except ImportError as e:
            msg = "plotly-resampler is required for resampler mode. Install: pip install plotly-resampler"
            raise ImportError(msg) from e
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
    try:
        from plotly_resampler.aggregation.gap_handlers import NoGapHandler  # noqa: PLC0415
        from plotly_resampler.figure_resampler.figure_resampler_interface import (  # noqa: PLC0415
            AbstractFigureAggregator,
        )
    except ImportError:
        return {}
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
    if not color_palette:
        msg = "color_palette must not be empty"
        raise ValueError(msg)
    return [color_palette[i % len(color_palette)] for i in range(n)]


class LegendTracker:
    """Track which legend entries have been shown to avoid duplicates.

    In panel / faceted figures the same logical trace (e.g. a member
    named ``"sales"``) appears in multiple subplots.  Only the first
    occurrence should carry ``showlegend=True``; subsequent ones must
    set ``showlegend=False`` while sharing the same ``legendgroup``.

    Parameters
    ----------
    show_legend : bool, default=True
        Global override.  When ``False``, `should_show` always
        returns ``False`` regardless of whether the name has been seen.

    Examples
    --------
    >>> tracker = LegendTracker()
    >>> tracker.should_show("sales")
    True
    >>> tracker.should_show("sales")
    False
    >>> tracker.should_show("demand")
    True

    >>> tracker_off = LegendTracker(show_legend=False)
    >>> tracker_off.should_show("sales")
    False
    """

    def __init__(self, show_legend: bool = True) -> None:
        self._seen: set[str] = set()
        self._show_legend = show_legend

    def should_show(self, name: str) -> bool:
        """Return ``True`` the first time *name* is passed (and legend is on)."""
        if not self._show_legend:
            return False
        first_seen = name not in self._seen
        self._seen.add(name)
        return first_seen


class PanelColorManager:
    """Assign consistent colours to panel members by name.

    Unlike positional indexing (where the first column always gets
    colour 0), this class maps *member names* to colours.  The same
    name always receives the same colour within a manager instance,
    regardless of the order it appears in different panel groups.

    Parameters
    ----------
    color_palette : list[str] | None, default=None
        Custom colour hex codes.  When ``None`` the default
        `palette_yohou` palette is used.

    Examples
    --------
    >>> mgr = PanelColorManager()
    >>> mgr.get_color("a")
    '#2563EB'
    >>> mgr.get_color("b")
    '#DC2626'
    >>> mgr.get_color("a")  # same name -> same colour
    '#2563EB'
    """

    def __init__(self, color_palette: list[str] | None = None) -> None:
        self._palette = color_palette or list(palette_yohou().values())
        self._name_to_color: dict[str, str] = {}
        self._counter = 0

    def get_color(self, member_name: str) -> str:
        """Return the colour assigned to *member_name*."""
        if member_name not in self._name_to_color:
            self._name_to_color[member_name] = self._palette[self._counter % len(self._palette)]
            self._counter += 1
        return self._name_to_color[member_name]

    def get_colors(self, names: list[str]) -> list[str]:
        """Return colours for a list of member names."""
        return [self.get_color(n) for n in names]


def linked_legendgroup_kwargs(
    member_name: str,
    legend_tracker: LegendTracker,
    *,
    is_primary: bool,
) -> dict:
    """Build ``legendgroup`` / ``showlegend`` kwargs for linked traces.

    All traces that share the same *member_name* ``legendgroup`` are
    toggled together when the user clicks the legend entry.  Only the
    *primary* trace (typically the line) should carry
    ``showlegend=True`` (once); secondary traces (outlier markers,
    threshold lines, confidence bands, ...) set ``showlegend=False``
    while still belonging to the same group.

    Parameters
    ----------
    member_name : str
        Legend group identifier (e.g. panel member display name).
    legend_tracker : LegendTracker
        Tracker used to de-duplicate legend entries across subplots.
    is_primary : bool
        ``True`` for the main trace that should appear in the legend;
        ``False`` for auxiliary traces (markers, bands, ...).

    Returns
    -------
    dict
        Ready-to-unpack kwargs for ``fig.add_trace(go.Scatter(..., **kw))``.

    Examples
    --------
    >>> from yohou.plotting import linked_legendgroup_kwargs
    >>> tracker = LegendTracker()
    >>> linked_legendgroup_kwargs("sales", tracker, is_primary=True)
    {'legendgroup': 'sales', 'showlegend': True, 'name': 'sales'}
    >>> linked_legendgroup_kwargs("sales", tracker, is_primary=False)
    {'legendgroup': 'sales', 'showlegend': False}

    See Also
    --------
    [`grouped_legend_kwargs`][yohou.plotting._utils.grouped_legend_kwargs] : Build kwargs for titled legend groups.
    [`LegendTracker`][yohou.plotting._utils.LegendTracker] : Track which legend entries have been shown.
    """
    if is_primary:
        return {
            "legendgroup": member_name,
            "showlegend": legend_tracker.should_show(member_name),
            "name": member_name,
        }
    return {
        "legendgroup": member_name,
        "showlegend": False,
    }


def grouped_legend_kwargs(
    group_title: str,
    entry_name: str,
    legend_tracker: LegendTracker,
    *,
    is_first_in_group: bool,
) -> dict:
    """Build kwargs for a trace that belongs to a titled legend group.

    Plotly's ``legendgrouptitle`` displays a header above a group of
    legend entries.  All traces sharing the same ``legendgroup`` toggle
    together; each trace gets its own visible ``name`` in the legend.

    Parameters
    ----------
    group_title : str
        Title shown above the legend group (e.g. member name ``"a"``).
    entry_name : str
        Individual trace label (e.g. ``"mean"``, ``"std"``).
    legend_tracker : LegendTracker
        De-duplicates entries across subplots.
    is_first_in_group : bool
        When ``True``, attaches the ``legendgrouptitle`` dict.

    Returns
    -------
    dict
        Ready-to-unpack kwargs for ``fig.add_trace(go.Scatter(..., **kw))``.

    Examples
    --------
    >>> tracker = LegendTracker()
    >>> grouped_legend_kwargs("sensor_a", "mean", tracker, is_first_in_group=True)
    {'legendgroup': 'sensor_a', 'showlegend': True, 'name': 'mean', 'legendgrouptitle': {'text': 'sensor_a'}}
    >>> grouped_legend_kwargs("sensor_a", "std", tracker, is_first_in_group=False)
    {'legendgroup': 'sensor_a', 'showlegend': True, 'name': 'std'}

    See Also
    --------
    [`linked_legendgroup_kwargs`][yohou.plotting.linked_legendgroup_kwargs] : Build kwargs for linked (ungrouped) legend entries.
    [`LegendTracker`][yohou.plotting._utils.LegendTracker] : Track which legend entries have been shown.
    """
    key = f"{group_title}::{entry_name}"
    kw: dict = {
        "legendgroup": group_title,
        "showlegend": legend_tracker.should_show(key),
        "name": entry_name,
    }
    if is_first_in_group:
        kw["legendgrouptitle"] = {"text": group_title}
    return kw


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


DEFAULT_WIDTH: int = 1000
"""Default figure width in pixels applied when *width* is ``None``."""

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
    hovermode: str | None = None,
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
    hovermode : str | None, default=None
        Plotly hovermode (e.g. ``"x unified"``).  ``None`` keeps the
        default (``"closest"`` from :data:`DEFAULT_LAYOUT`).

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
        layout_update["title"]["text"] = title  # type: ignore[index, invalid-assignment]  # ty:ignore[invalid-assignment]
    if x_label is not None:
        layout_update["xaxis"]["title"] = x_label  # type: ignore[index, invalid-assignment]  # ty:ignore[invalid-assignment]
    if y_label is not None:
        layout_update["yaxis"]["title"] = y_label  # type: ignore[index, invalid-assignment]  # ty:ignore[invalid-assignment]
    layout_update["width"] = width if width is not None else DEFAULT_WIDTH  # type: ignore[invalid-assignment]  # ty:ignore[invalid-assignment]
    if height is not None:
        layout_update["height"] = height  # type: ignore[invalid-assignment]  # ty:ignore[invalid-assignment]
    if hovermode is not None:
        layout_update["hovermode"] = hovermode  # type: ignore[invalid-assignment]

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


def _auto_detect_panel(
    df: pl.DataFrame,
    panel_group_names: list[str] | None = None,
) -> bool:
    """Return ``True`` if *df* contains panel columns for the given groups.

    Panel data always results in a faceted layout - callers should not
    apply a non-panel fallback when this returns ``True``.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to inspect.
    panel_group_names : list[str] | None, default=None
        Group prefixes to check.  ``None`` checks for any panel columns.

    Returns
    -------
    bool
        ``True`` when at least one matching panel column is found.
    """
    try:
        return len(resolve_panel_columns(df, panel_group_names)) > 0
    except ValueError:
        return False


def _add_confidence_bands(
    fig: go.Figure,
    x_values: list,
    ci_upper: list[float],
    ci_lower: list[float],
    *,
    row: int | None = None,
    col: int | None = None,
    color: str = "#DC2626",
) -> None:
    """Add symmetric or asymmetric dashed confidence band traces.

    Parameters
    ----------
    fig : go.Figure
        Target figure.
    x_values : list
        X-axis values shared by both bands.
    ci_upper : list[float]
        Upper confidence boundary values.
    ci_lower : list[float]
        Lower confidence boundary values.
    row : int | None, default=None
        Subplot row (1-indexed). Pass together with *col* for subplots.
    col : int | None, default=None
        Subplot column (1-indexed).
    color : str, default="#DC2626"
        Dashed line color.

    """
    for y_vals in (ci_upper, ci_lower):
        trace = go.Scatter(
            x=x_values,
            y=y_vals,
            mode="lines",
            line={"dash": "dash", "color": color, "width": 1},
            showlegend=False,
            hoverinfo="skip",
        )
        if row is not None and col is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)


def _make_hovertemplate(
    name: str,
    x_label: str,
    y_label: str,
    *,
    decimals: int = 2,
    extra: str = "",
) -> str:
    """Return a standard Plotly hovertemplate string.

    Parameters
    ----------
    name : str
        Bold trace name shown at the top.
    x_label : str
        Label for the x value.
    y_label : str
        Label for the y value.
    decimals : int, default=2
        Decimal places for the y value.
    extra : str, default=""
        Content for the ``<extra>`` tag (shown on the right side of the
        hover box).  Empty string hides the extra section.

    Returns
    -------
    str
        Plotly hovertemplate string.

    Examples
    --------
    >>> _make_hovertemplate("sensor_a", "Time", "Value")
    '<b>sensor_a</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>'
    >>> _make_hovertemplate("MAE", "Lag", "ACF", decimals=3)
    '<b>MAE</b><br>Lag: %{x}<br>ACF: %{y:.3f}<extra></extra>'

    """
    return f"<b>{name}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y:.{decimals}f}}<extra>{extra}</extra>"


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
    render_fn: Callable[[RenderContext], None],
    *,
    panel_group_names: list[str] | None = None,
    columns: str | list[str] | None = None,
    facet_by: Literal["group", "member"] = "member",
    facet_n_cols: int = 2,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    row_height: int = 300,
    shared_xaxes: bool = True,
    subplot_v_spacing: float | None = None,
    subplot_h_spacing: float | None = None,
    resampler: bool | Literal["widget"] | None = None,
) -> go.Figure:
    """Create a faceted subplot figure for panel data.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with panel columns (``group__member`` pattern).
    render_fn : Callable[[RenderContext], None]
        Callback receiving a `RenderContext` for each trace.
    panel_group_names : list[str] | None, default=None
        Group prefixes to include (``None`` means all).
    columns : str | list[str] | None, default=None
        Member names to include within selected groups.
    facet_by : Literal["group", "member"], default="member"
        Faceting axis.

        - ``"member"`` - one subplot per unique member; groups are
          overlaid within each subplot (cross-entity comparison).
        - ``"group"`` - one subplot per group; members are overlaid
          within each subplot (within-entity analysis).
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
    subplot_v_spacing : float | None, default=None
        Vertical spacing between subplots.
    subplot_h_spacing : float | None, default=None
        Horizontal spacing between subplots.
    resampler : bool | Literal["widget"] | None, default=None
        Resampler mode.

    Returns
    -------
    go.Figure
        Plotly figure with faceted panel subplots.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting._utils import panel_facet_figure, RenderContext, LegendTracker
    >>> df = pl.DataFrame({
    ...     "time": [1, 2, 3],
    ...     "sales__a": [10, 20, 30],
    ...     "sales__b": [15, 25, 35],
    ... })
    >>> tracker = LegendTracker()
    >>> def render(ctx: RenderContext) -> None:
    ...     ctx.fig.add_scatter(
    ...         x=ctx.sub_df["time"],
    ...         y=ctx.sub_df[ctx.display_name],
    ...         name=ctx.display_name,
    ...         row=ctx.row,
    ...         col=ctx.col,
    ...     )
    >>> fig = panel_facet_figure(df, render)
    >>> len(fig.data) > 0
    True

    See Also
    --------
    [`RenderContext`][yohou.plotting.RenderContext] : Typed context passed to the render callback.
    [`resolve_panel_columns`][yohou.plotting.resolve_panel_columns] : Resolve which panel columns to plot.
    """
    panel_cols = resolve_panel_columns(df, panel_group_names, columns)
    groups, all_members = _group_panel_columns(panel_cols)
    all_group_names = list(groups.keys())

    if facet_by == "member":
        facet_keys = all_members
        overlay_keys_per_facet = {
            m: [g for g, cols in groups.items() if any(_member_name(c) == m for c in cols)] for m in all_members
        }
        # Warn about asymmetric groups
        full_group_set = set(all_group_names)
        missing = {
            m: full_group_set - set(overlay_keys_per_facet[m])
            for m in all_members
            if set(overlay_keys_per_facet[m]) != full_group_set
        }
        if missing:
            pairs = ", ".join(f"({g}, {m})" for m, gs in missing.items() for g in sorted(gs))
            warnings.warn(
                f"Asymmetric panel: missing (group, member) pairs skipped: {pairs}",
                stacklevel=2,
            )
    else:
        facet_keys = all_group_names
        overlay_keys_per_facet = {g: [_member_name(c) for c in cols] for g, cols in groups.items()}

    n_facets = len(facet_keys)
    n_cols_grid = min(n_facets, facet_n_cols)
    n_rows = (n_facets + n_cols_grid - 1) // n_cols_grid

    fig = _create_subplots(
        resampler,
        rows=n_rows,
        cols=n_cols_grid,
        subplot_titles=facet_keys,
        shared_xaxes=shared_xaxes,
        vertical_spacing=subplot_v_spacing if subplot_v_spacing is not None else _subplot_spacing(n_rows),
        horizontal_spacing=subplot_h_spacing if subplot_h_spacing is not None else 0.08,
    )

    for facet_idx, facet_key in enumerate(facet_keys):
        row = facet_idx // n_cols_grid + 1
        col_idx = facet_idx % n_cols_grid + 1

        overlay_keys = overlay_keys_per_facet[facet_key]
        for entity_idx, overlay_key in enumerate(overlay_keys):
            if facet_by == "member":
                group_name = overlay_key
                member_name = facet_key
                col_name = next(
                    (c for c in groups[group_name] if _member_name(c) == member_name),
                    None,
                )
                if col_name is None:
                    continue
                display_name = group_name
            else:
                group_name = facet_key
                member_name = overlay_key
                col_name = next(
                    (c for c in groups[group_name] if _member_name(c) == member_name),
                    None,
                )
                if col_name is None:
                    continue
                display_name = member_name

            sub_df = df.select("time", pl.col(col_name).alias(display_name))
            ctx = RenderContext(
                fig=fig,
                sub_df=sub_df,
                display_name=display_name,
                entity_idx=entity_idx,
                row=row,
                col=col_idx,
                facet_by=facet_by,
                facet_name=facet_key,
                group_name=group_name,
                member_name=member_name,
            )
            render_fn(ctx)

    default_height = max(row_height * n_rows, 400)

    # Auto-switch hovermode for member faceting with overlaid groups
    hovermode = None
    if facet_by == "member" and len(all_group_names) > 1:
        hovermode = "x unified"

    fig = apply_default_layout(
        fig,
        title=title,
        x_label=x_label,
        y_label=y_label,
        width=width,
        height=height or default_height,
        hovermode=hovermode,
    )

    return fig
