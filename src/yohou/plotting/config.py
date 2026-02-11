"""Configuration models for yohou plotting using Pydantic."""

from typing import Literal

from pydantic import BaseModel, Field


class LineStyle(BaseModel):
    """
    Line styling configuration.

    Parameters
    ----------
    width : float, default=2.0
        Line width, must be between 0.1 and 10.0.
    color : str, default="#2563EB"
        Line color as hex code.
    alpha : float, default=1.0
        Line opacity, must be between 0.0 and 1.0.
    dash : {"solid", "dash", "dot", "dashdot"}, default="solid"
        Line dash style.

    Examples
    --------
    >>> from yohou.plotting import LineStyle
    >>> style = LineStyle()
    >>> style.width
    2.0
    >>> style.color
    '#2563EB'

    >>> custom_style = LineStyle(width=3.0, color="#DC2626", dash="dash")
    >>> custom_style.dash
    'dash'
    """

    width: float = Field(default=2.0, ge=0.1, le=10.0)
    color: str = "#2563EB"
    alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    dash: Literal["solid", "dash", "dot", "dashdot"] = "solid"


class MarkerStyle(BaseModel):
    """
    Marker styling configuration.

    Parameters
    ----------
    size : float, default=6.0
        Marker size, must be between 1.0 and 20.0.
    color : str, default="#2563EB"
        Marker color as hex code.
    alpha : float, default=1.0
        Marker opacity, must be between 0.0 and 1.0.
    symbol : str, default="circle"
        Marker symbol name (Plotly symbol).

    Examples
    --------
    >>> from yohou.plotting import MarkerStyle
    >>> marker = MarkerStyle()
    >>> marker.size
    6.0

    >>> custom_marker = MarkerStyle(size=10.0, symbol="square", alpha=0.7)
    >>> custom_marker.symbol
    'square'
    """

    size: float = Field(default=6.0, ge=1.0, le=20.0)
    color: str = "#2563EB"
    alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    symbol: str = "circle"


class FacetConfig(BaseModel):
    """
    Faceting configuration for subplots.

    Parameters
    ----------
    ncol : int, default=2
        Number of columns in facet grid, must be at least 1.
    scales : {"free_y", "free_x", "free", "fixed"}, default="free_y"
        Scale sharing between facets.
    spacing_horizontal : float, default=0.05
        Horizontal spacing between facets, must be between 0.0 and 0.5.
    spacing_vertical : float, default=0.08
        Vertical spacing between facets, must be between 0.0 and 0.5.

    Examples
    --------
    >>> from yohou.plotting import FacetConfig
    >>> config = FacetConfig()
    >>> config.ncol
    2
    >>> config.scales
    'free_y'

    >>> custom = FacetConfig(ncol=3, scales="free", spacing_horizontal=0.1)
    >>> custom.ncol
    3
    """

    ncol: int = Field(default=2, ge=1)
    scales: Literal["free_y", "free_x", "free", "fixed"] = "free_y"
    spacing_horizontal: float = Field(default=0.05, ge=0.0, le=0.5)
    spacing_vertical: float = Field(default=0.08, ge=0.0, le=0.5)


class PlotLayout(BaseModel):
    """
    Plot layout configuration.

    Parameters
    ----------
    title : str | None, default=None
        Plot title.
    title_font_size : int, default=16
        Title font size, must be between 8 and 32.
    x_label : str | None, default=None
        X-axis label.
    y_label : str | None, default=None
        Y-axis label.
    axis_font_size : int, default=12
        Axis label font size, must be between 6 and 24.
    legend_font_size : int, default=11
        Legend font size, must be between 6 and 20.
    width : int | None, default=None
        Plot width in pixels, must be at least 200 if specified.
    height : int | None, default=None
        Plot height in pixels, must be at least 200 if specified.
    template : str, default="plotly_white"
        Plotly template name.

    Examples
    --------
    >>> from yohou.plotting import PlotLayout
    >>> layout = PlotLayout()
    >>> layout.title_font_size
    16
    >>> layout.template
    'plotly_white'

    >>> custom = PlotLayout(title="My Plot", width=800, height=600)
    >>> custom.title
    'My Plot'
    >>> custom.width
    800
    """

    title: str | None = None
    title_font_size: int = Field(default=16, ge=8, le=32)
    x_label: str | None = None
    y_label: str | None = None
    axis_font_size: int = Field(default=12, ge=6, le=24)
    legend_font_size: int = Field(default=11, ge=6, le=20)
    width: int | None = Field(default=None, ge=200)
    height: int | None = Field(default=None, ge=200)
    template: str = "plotly_white"
