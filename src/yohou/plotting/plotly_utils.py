"""Plotly utility functions and layout configuration."""

import plotly.graph_objects as go

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
    >>> from yohou.plotting.plotly_utils import apply_default_layout
    >>> fig = go.Figure()
    >>> fig = apply_default_layout(fig, title="Test", x_label="Time")
    >>> fig.layout.title.text
    'Test'
    """
    layout_update = DEFAULT_LAYOUT.copy()

    if title is not None:
        layout_update["title"]["text"] = title
    if x_label is not None:
        layout_update["xaxis"]["title"] = x_label
    if y_label is not None:
        layout_update["yaxis"]["title"] = y_label
    if width is not None:
        layout_update["width"] = width
    if height is not None:
        layout_update["height"] = height

    fig.update_layout(layout_update)
    return fig
