"""Shared fixtures for plotting tests."""

import pytest

# Skip the entire plotting test directory when the plotting extra is not installed
# (e.g. on Python 3.14 where tsdownsample cannot be built yet).
plotly = pytest.importorskip("plotly", reason="plotting extra not installed")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from plotly import graph_objects as go  # noqa: E402


@pytest.fixture
def panel_daily_df():
    """Create a daily panel DataFrame with two groups (y__a, y__b).

    366 rows (2020 full year), daily frequency.
    Both panel columns have seasonal-like patterns.
    """
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
    n = len(dates)
    return pl.DataFrame({
        "time": dates,
        "y__a": [100 + i % 30 + (i % 7) * 3 for i in range(n)],
        "y__b": [200 + i % 20 + (i % 5) * 5 for i in range(n)],
    })


@pytest.fixture
def panel_monthly_df():
    """Create a monthly panel DataFrame with two groups (y__a, y__b).

    12 rows, monthly frequency.
    """
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
        "y__a": [100, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180, 175],
        "y__b": [200, 210, 205, 220, 230, 225, 240, 250, 245, 260, 270, 265],
    })


@pytest.fixture
def panel_monthly_3groups():
    """Create a monthly panel with three groups (sales__a, sales__b, sales__c).

    12 rows, monthly frequency. Uses 'sales' as the group prefix.
    """
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
        "sales__a": [10, 12, 11, 13, 14, 13, 15, 16, 15, 17, 18, 17],
        "sales__b": [20, 22, 21, 23, 24, 23, 25, 26, 25, 27, 28, 27],
        "sales__c": [30, 32, 31, 33, 34, 33, 35, 36, 35, 37, 38, 37],
    })


@pytest.fixture
def panel_with_nulls():
    """Create a daily panel DataFrame with missing values.

    366 rows, daily frequency. Both columns have periodic nulls.
    """
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
    n = len(dates)
    return pl.DataFrame({
        "time": dates,
        "y__a": [100 + i if i % 5 != 0 else None for i in range(n)],
        "y__b": [200 + i if i % 7 != 0 else None for i in range(n)],
    })


@pytest.fixture
def panel_two_groups_df():
    """Create a panel DataFrame with two group prefixes (temp and wind).

    Used for testing panel_group_names filtering.
    """
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
    n = len(dates)
    return pl.DataFrame({
        "time": dates,
        "temp__city_a": [20.0 + i * 0.1 for i in range(n)],
        "temp__city_b": [15.0 + i * 0.1 for i in range(n)],
        "wind__city_a": [5.0 + np.sin(i * 0.3) for i in range(n)],
        "wind__city_b": [3.0 + np.cos(i * 0.3) for i in range(n)],
    })


@pytest.fixture
def short_df():
    """Create a short DataFrame for lag/cross-correlation and spectrum tests.

    91 rows (Q1 2020), daily frequency. Two numeric columns x and y.
    Identical definition previously duplicated in test_diagnostics.py and test_signal.py.
    """
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
        "x": [100 + i % 20 for i in range(91)],
        "y": [150 + (i % 15) * 2 for i in range(91)],
    })


@pytest.fixture
def simple_df():
    """Single-column daily DataFrame, 366 rows (full year 2020)."""
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
    return pl.DataFrame({
        "time": dates,
        "y": [100 + i % 30 for i in range(len(dates))],
    })


@pytest.fixture
def multi_column_df():
    """Three-column daily DataFrame, 31 rows (January 2020)."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 31), "1d", eager=True),
        "y1": list(range(31)),
        "y2": [x * 2 + 5 for x in range(31)],
        "y3": [x * 3 + 1 for x in range(31)],
    })


@pytest.fixture
def df_with_nulls():
    """Daily DataFrame with periodic nulls, 366 rows (full year 2020)."""
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
    n = len(dates)
    return pl.DataFrame({
        "time": dates,
        "y": [100 + i if i % 5 != 0 else None for i in range(n)],
        "z": [200 + i if i % 7 != 0 else None for i in range(n)],
    })


def visible_legend_names(fig: go.Figure) -> list[str]:
    """Return legend names from traces where showlegend is True."""
    return [t.name for t in fig.data if getattr(t, "showlegend", None) is True]


def legend_groups(fig: go.Figure) -> set[str]:
    """Return the set of distinct legendgroup values across all traces."""
    return {t.legendgroup for t in fig.data if getattr(t, "legendgroup", None)}


def assert_legend_groups(fig: go.Figure, expected: set[str]) -> None:
    """Assert that the figure contains exactly *expected* legend groups."""
    actual = legend_groups(fig)
    assert actual == expected, f"Expected legendgroups {expected}, got {actual}"


def assert_consistent_colors(fig: go.Figure, group_name: str) -> None:
    """Assert every trace in *group_name* uses the same base colour."""
    colors = set()
    for t in fig.data:
        if getattr(t, "legendgroup", None) == group_name:
            marker = getattr(t, "marker", None)
            if marker and getattr(marker, "color", None):
                colors.add(marker.color)
            line = getattr(t, "line", None)
            if line and getattr(line, "color", None):
                colors.add(line.color)
    assert len(colors) <= 1, f"Inconsistent colours for group '{group_name}': {colors}"


def assert_linked_legend(fig: go.Figure, group_name: str, min_traces: int = 2) -> None:
    """Assert at least *min_traces* traces share legendgroup *group_name*."""
    count = sum(1 for t in fig.data if getattr(t, "legendgroup", None) == group_name)
    assert count >= min_traces, f"Expected >= {min_traces} traces with legendgroup='{group_name}', got {count}"


def assert_figure_valid(fig: go.Figure, *, min_traces: int = 1) -> None:
    """Assert that fig is a non-empty go.Figure."""
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= min_traces


def assert_layout(
    fig: go.Figure,
    *,
    title: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Assert layout properties when provided."""
    if title is not None:
        assert fig.layout.title.text == title
    if width is not None:
        assert fig.layout.width == width
    if height is not None:
        assert fig.layout.height == height


def has_legendgrouptitle(fig: go.Figure) -> bool:
    """Return True if any trace has a legendgrouptitle set."""
    for t in fig.data:
        lgt = getattr(t, "legendgrouptitle", None)
        if lgt and getattr(lgt, "text", None):
            return True
    return False


def trace_opacities(fig: go.Figure, legendgroup: str) -> list[float]:
    """Return opacity values for all traces in the given legendgroup.

    Checks trace-level ``opacity`` first, then falls back to the alpha
    channel of an ``rgba(...)`` line color.
    """
    import re

    opacities = []
    for t in fig.data:
        if getattr(t, "legendgroup", None) == legendgroup:
            op = getattr(t, "opacity", None)
            if op is not None:
                opacities.append(float(op))
                continue
            line = getattr(t, "line", None)
            if line:
                color = getattr(line, "color", None)
                if color and isinstance(color, str):
                    m = re.match(r"rgba\(\d+,\d+,\d+,([\d.]+)\)", color)
                    if m:
                        opacities.append(float(m.group(1)))
    return opacities


@pytest.fixture
def panel_hourly_df():
    """Hourly panel DataFrame, 2 months, 2 members. For seasonal heatmap tests."""
    dates = pl.datetime_range(pl.datetime(2020, 1, 1), pl.datetime(2020, 2, 29, 23), "1h", eager=True)
    n = len(dates)
    return pl.DataFrame({
        "time": dates,
        "y__a": [10.0 + np.sin(i * 0.3) * 5 for i in range(n)],
        "y__b": [20.0 + np.cos(i * 0.3) * 3 for i in range(n)],
    })


@pytest.fixture
def panel_3year_daily_df():
    """3-year daily panel DataFrame, 2 members. For subseasonality tests."""
    dates = pl.date_range(pl.date(2018, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
    n = len(dates)
    return pl.DataFrame({
        "time": dates,
        "y__a": [100 + i % 30 + (i % 7) * 3 for i in range(n)],
        "y__b": [200 + i % 20 + (i % 5) * 5 for i in range(n)],
    })
