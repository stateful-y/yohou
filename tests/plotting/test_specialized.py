"""Tests for specialized plotting functions."""

import polars as pl
import pytest

from yohou.plotting import plot_calendar_heatmap, plot_cross_correlation


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
        "x": [100 + i % 20 for i in range(91)],
        "y": [150 + (i % 15) * 2 for i in range(91)],
    })


# Tests for plot_cross_correlation


def test_plot_cross_correlation_basic(sample_df):
    """Test basic cross-correlation functionality."""
    fig = plot_cross_correlation(sample_df, x_column="x", y_column="y", lags=20)
    assert len(fig.data) > 0


def test_plot_cross_correlation_different_lags(sample_df):
    """Test with different lag values."""
    fig10 = plot_cross_correlation(sample_df, x_column="x", y_column="y", lags=10)
    fig30 = plot_cross_correlation(sample_df, x_column="x", y_column="y", lags=30)
    assert len(fig10.data) > 0
    assert len(fig30.data) > 0


def test_plot_cross_correlation_alpha(sample_df):
    """Test different confidence levels."""
    fig = plot_cross_correlation(sample_df, x_column="x", y_column="y", alpha=0.01)
    assert len(fig.data) > 0


def test_plot_cross_correlation_styling(sample_df):
    """Test custom styling."""
    fig = plot_cross_correlation(
        sample_df,
        x_column="x",
        y_column="y",
        marker_size=8.0,
        marker_color="#DC2626",
        line_color="#059669",
    )
    assert len(fig.data) > 0


def test_plot_cross_correlation_missing_column(sample_df):
    """Test error handling for missing columns."""
    with pytest.raises(ValueError, match="not found"):
        plot_cross_correlation(sample_df, x_column="missing", y_column="y")

    with pytest.raises(ValueError, match="not found"):
        plot_cross_correlation(sample_df, x_column="x", y_column="missing")


# Tests for plot_calendar_heatmap


def test_plot_calendar_heatmap_basic():
    """Test basic calendar heatmap."""
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
        "sales": [100 + (i % 50) for i in range(366)],
    })
    fig = plot_calendar_heatmap(df, column="sales", year=2020)
    assert len(fig.data) > 0


def test_plot_calendar_heatmap_aggregations():
    """Test different aggregation methods."""
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
        "value": [100 + i for i in range(366)],
    })

    for agg in ["sum", "mean", "median", "max", "min"]:
        fig = plot_calendar_heatmap(df, column="value", aggregation=agg, year=2020)
        assert len(fig.data) > 0


def test_plot_calendar_heatmap_missing_column():
    """Test error when column missing."""
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
        "value": [100 + i for i in range(91)],
    })

    with pytest.raises(ValueError, match="not found"):
        plot_calendar_heatmap(df, column="missing")


def test_plot_calendar_heatmap_no_data_for_year():
    """Test error when no data for specified year."""
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
        "value": [100 + i for i in range(91)],
    })

    with pytest.raises(ValueError, match="No data available"):
        plot_calendar_heatmap(df, column="value", year=2025)


def test_plot_calendar_heatmap_panel_not_implemented():
    """Test panel grouping raises NotImplementedError."""
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
        "value": [100 + i for i in range(91)],
    })

    with pytest.raises(NotImplementedError, match="Panel grouping"):
        plot_calendar_heatmap(df, column="value", panel_group_name="group")
