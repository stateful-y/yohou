"""Tests for quality plotting functions."""

import polars as pl
import pytest
from plotly import graph_objects as go

from yohou.plotting import plot_missing_data


@pytest.fixture
def df_with_nulls():
    """Create a sample dataframe with missing values."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
        "y": [100 + i if i % 5 != 0 else None for i in range(366)],
        "z": [200 + i if i % 7 != 0 else None for i in range(366)],
    })


def test_plot_missing_data_bars(df_with_nulls):
    """Test missing data bars visualization."""
    fig = plot_missing_data(df_with_nulls, method="bars")
    assert len(fig.data) > 0
    assert isinstance(fig.data[0], go.Bar)


def test_plot_missing_data_heatmap(df_with_nulls):
    """Test missing data heatmap visualization."""
    fig = plot_missing_data(df_with_nulls, method="heatmap")
    assert len(fig.data) > 0
    assert isinstance(fig.data[0], go.Heatmap)


def test_plot_missing_data_matrix(df_with_nulls):
    """Test missing data matrix visualization."""
    fig = plot_missing_data(df_with_nulls, method="matrix")
    assert len(fig.data) > 0
    assert isinstance(fig.data[0], go.Heatmap)


def test_plot_missing_data_time_aggregation(df_with_nulls):
    """Test missing data with time aggregation."""
    fig = plot_missing_data(df_with_nulls, method="heatmap", time_aggregation="1mo")
    assert len(fig.data) > 0


def test_plot_missing_data_custom_colors(df_with_nulls):
    """Test missing data with custom colors."""
    fig = plot_missing_data(
        df_with_nulls,
        method="heatmap",
        color_missing="#FF0000",
        color_present="#00FF00",
    )
    assert len(fig.data) > 0


def test_plot_missing_data_invalid_method(df_with_nulls):
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError, match="Unknown method"):
        plot_missing_data(df_with_nulls, method="invalid")  # type: ignore


def test_plot_missing_data_panel_not_implemented(df_with_nulls):
    """Test that panel grouping raises NotImplementedError."""
    df = df_with_nulls.with_columns(pl.lit("A").alias("group"))
    with pytest.raises(NotImplementedError, match="Panel grouping not yet implemented"):
        plot_missing_data(df, method="heatmap", panel_group_name="group")
