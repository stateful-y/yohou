---
description: "Guide for creating new plotting functions in Yohou. Use when adding custom visualizations to the plotting module."
---

# Creating New Plotting Functions

## Overview

Yohou's plotting module uses **Plotly** for interactive visualizations with a consistent API. All plotting functions:
- Accept polars DataFrames as input
- Return `plotly.graph_objects.Figure` objects
- Follow a unified parameter naming convention
- Support panel data via `panel_group_name` parameter
- Use the yohou color palette by default

**Location**: `src/yohou/plotting/`

---

## Module Organization

```
src/yohou/plotting/
├── __init__.py           # Public API exports
├── timeseries.py         # Core time series plots (line, boxplot, rolling stats)
├── comparison.py         # Forecast comparison (residuals, forecast overlay)
├── diagnostics.py        # Statistical diagnostics (ACF, PACF, seasonality)
├── frequency.py          # Frequency domain (periodogram, lag scatter)
├── specialized.py        # Domain-specific (cross-correlation, calendar heatmap)
├── visualization.py      # Interval plots (prediction intervals, calibration)
├── quality.py            # Data quality (missing data visualization)
├── colors.py             # Color palette utilities
├── plotly_utils.py       # Plotly helpers (layout, themes)
└── prep.py               # Input validation and preprocessing
```

---

## Minimal Plotting Function Template

```python
"""Module docstring (e.g., 'Core time series plotting functions')."""

from typing import Literal

import plotly.graph_objects as go
import polars as pl

from yohou.plotting.colors import get_color_sequence
from yohou.plotting.plotly_utils import apply_default_layout
from yohou.plotting.prep import resolve_columns, validate_dataframe


def plot_my_visualization(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    panel_group_name: str | None = None,
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """
    Create custom time series visualization.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with 'time' column and numeric columns to plot.
    columns : str | list[str] | None, default=None
        Column(s) to plot. If None, plots all numeric columns except 'time'.
    panel_group_name : str | None, default=None
        Column name for grouping (panel data). Creates separate plots per group.
    color_palette : list[str] | None, default=None
        Custom color palette as hex codes. If None, uses yohou palette.
    show_legend : bool, default=True
        Whether to show legend.
    title : str | None, default=None
        Plot title.
    x_label : str | None, default=None
        X-axis label. Defaults to "time".
    y_label : str | None, default=None
        Y-axis label.
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    **kwargs : dict
        Additional styling parameters (line_width, marker_size, etc.).

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    TypeError
        If df is not a Polars DataFrame.
    ValueError
        If DataFrame is empty, missing 'time' column, or specified columns don't exist.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_my_visualization

    >>> # Create sample data
    >>> df = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
    ...     "value": [100, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180, 175],
    ... })

    >>> # Create plot
    >>> fig = plot_my_visualization(df, columns="value")
    >>> len(fig.data)  # One trace
    1
    """
    # 1. Validate input DataFrame
    validate_dataframe(df)

    # 2. Resolve columns to plot
    columns_to_plot = resolve_columns(df, columns)

    # 3. Get color palette
    colors = get_color_sequence(
        n_colors=len(columns_to_plot),
        custom_palette=color_palette,
    )

    # 4. Create figure
    fig = go.Figure()

    # 5. Add traces for each column
    for i, col in enumerate(columns_to_plot):
        # Extract default styling from kwargs
        line_width = kwargs.get("line_width", 2.0)
        line_dash = kwargs.get("line_dash", "solid")

        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[col],
                mode="lines",
                name=col,
                line=dict(
                    color=colors[i],
                    width=line_width,
                    dash=line_dash,
                ),
                showlegend=show_legend,
            )
        )

    # 6. Apply default layout
    fig = apply_default_layout(
        fig,
        title=title,
        x_label=x_label or "time",
        y_label=y_label,
        width=width,
        height=height,
        show_legend=show_legend,
    )

    return fig
```

---

## Key Utilities

### Input Validation

```python
from yohou.plotting.prep import validate_dataframe, resolve_columns

# Validate DataFrame structure
validate_dataframe(df)  # Checks: is polars, not empty, has 'time' column

# Resolve columns (None → all numeric, str → [str], list → list)
columns_to_plot = resolve_columns(df, columns)
```

### Color Management

```python
from yohou.plotting.colors import get_color_sequence

# Get yohou color palette
colors = get_color_sequence(n_colors=5)  # Returns 5 colors

# Use custom palette
colors = get_color_sequence(n_colors=5, custom_palette=["#FF0000", "#00FF00", "#0000FF"])
```

### Layout Styling

```python
from yohou.plotting.plotly_utils import apply_default_layout

# Apply consistent yohou styling
fig = apply_default_layout(
    fig,
    title="My Plot",
    x_label="Time",
    y_label="Value",
    width=800,
    height=600,
    show_legend=True,
)
```

---

## Panel Data Support Pattern

For panel data visualizations, use facets or dropdowns:

```python
def plot_panel_aware(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    panel_group_name: str | None = None,  # Key parameter for panel data
    facet_ncol: int = 2,                   # Columns in facet grid
    dropdown: bool = False,                # Use dropdown instead of facets
    **kwargs,
) -> go.Figure:
    """Plot with panel data support."""

    if panel_group_name is not None:
        # Split by panel group
        groups = df[panel_group_name].unique().to_list()

        if dropdown:
            # Create dropdown menu
            fig = go.Figure()
            for group in groups:
                group_df = df.filter(pl.col(panel_group_name) == group)
                # Add traces with visibility control
                # (Implementation details depend on use case)
        else:
            # Create faceted plot (subplots)
            from plotly.subplots import make_subplots

            n_groups = len(groups)
            n_rows = (n_groups + facet_ncol - 1) // facet_ncol

            fig = make_subplots(
                rows=n_rows,
                cols=facet_ncol,
                subplot_titles=[str(g) for g in groups],
            )

            for i, group in enumerate(groups):
                row = i // facet_ncol + 1
                col = i % facet_ncol + 1

                group_df = df.filter(pl.col(panel_group_name) == group)
                # Add traces to subplot
                fig.add_trace(
                    go.Scatter(x=group_df["time"], y=group_df[columns[0]]),
                    row=row,
                    col=col,
                )

    else:
        # Single plot (no panel grouping)
        fig = go.Figure()
        # ... standard plot logic

    return fig
```

---

## Common Plot Types

### Time Series Line Plot

**See**: `src/yohou/plotting/timeseries.py` - `plot_timeseries()`

### Forecast Comparison

**See**: `src/yohou/plotting/comparison.py` - `plot_forecast()`, `plot_residuals()`

### Statistical Diagnostics

**See**: `src/yohou/plotting/diagnostics.py` - `plot_autocorrelation()`, `plot_partial_autocorrelation()`

### Interval Visualization

**See**: `src/yohou/plotting/visualization.py` - `plot_prediction_intervals()`, `plot_calibration()`

---

## Styling Conventions

### Default Parameters

```python
# Line plots
line_width: float = 2.0
line_dash: str = "solid"  # "solid", "dash", "dot", "dashdot"
line_alpha: float = 1.0

# Marker plots
marker_size: float = 8.0
marker_symbol: str = "circle"

# Colors
color_palette: list[str] | None = None  # None = yohou palette

# Layout
width: int | None = None  # None = auto
height: int | None = None  # None = auto
show_legend: bool = True
```

### Hover Templates

```python
fig.update_traces(
    hovertemplate="<b>%{fullData.name}</b><br>"
                  "Time: %{x}<br>"
                  "Value: %{y:.2f}<br>"
                  "<extra></extra>"  # Removes secondary box
)
```

---

## Testing Patterns

All plotting functions should have tests in `tests/plotting/test_<module>.py`:

```python
import polars as pl
import pytest
from yohou.plotting import plot_my_visualization


def test_plot_returns_figure(sample_data):
    """Test that plot returns a Plotly figure."""
    fig = plot_my_visualization(sample_data, columns="value")
    assert hasattr(fig, "data")  # Is a Plotly figure


def test_plot_with_multiple_columns(sample_data):
    """Test plotting multiple columns."""
    fig = plot_my_visualization(sample_data, columns=["value1", "value2"])
    assert len(fig.data) == 2  # Two traces


def test_plot_raises_on_missing_column(sample_data):
    """Test error handling for missing columns."""
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        plot_my_visualization(sample_data, columns="nonexistent")
```

---

## Checklist Before Committing

1. `uvx ruff check --fix src/yohou/plotting/<file>.py`
2. `uvx ruff format src/yohou/plotting/<file>.py`
3. `uvx ty check src/yohou/plotting/<file>.py`
4. `uvx interrogate src/yohou/plotting/<file>.py` (docstring coverage)
5. `uv run pytest tests/plotting/test_<file>.py -v`
6. `uv run pytest --doctest-modules src/yohou/plotting/<file>.py`
7. `uvx nox -s fix` (all quality checks)
8. Add to `__init__.py` exports

---

## Common Pitfalls

- **Missing time column check**: Always validate `"time"` column exists
- **Not using yohou palette**: Use `get_color_sequence()` for consistent colors
- **Inconsistent parameter names**: Follow convention (e.g., `show_legend`, not `display_legend`)
- **No panel data support**: Consider adding `panel_group_name` parameter for versatility
- **Mutating input DataFrame**: Always operate on copies or return new DataFrames
- **Missing doctest examples**: All public functions need runnable examples
- **Not applying default layout**: Use `apply_default_layout()` for consistent styling

---

## Real-World Examples to Study

**Core visualizations**:
- `src/yohou/plotting/timeseries.py` - `plot_timeseries()`, `plot_boxplot()`
- `src/yohou/plotting/comparison.py` - `plot_forecast()`, `plot_residuals()`
- `src/yohou/plotting/diagnostics.py` - `plot_autocorrelation()`

**Panel data examples**:
- `examples/panel_data_tutorial.py` - Faceting and dropdown demos
- `examples/m4_monthly.py` - Panel data visualization patterns

**Testing**:
- `tests/plotting/` - Plotting test suite (not yet implemented, but should follow patterns above)
