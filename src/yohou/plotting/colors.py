"""Color palette system for yohou plotting."""


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
        "teal": "#0D9488",  # Teal accent
        "pink": "#DB2777",  # Pink accent
        "indigo": "#4F46E5",  # Indigo accent
        "yellow": "#CA8A04",  # Gold/yellow
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
    >>> from yohou.plotting import get_color_sequence
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
