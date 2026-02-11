"""Tests for diagnostic plotting functions."""

import polars as pl
import pytest
from plotly import graph_objects as go

from yohou.plotting import (
    plot_autocorrelation,
    plot_correlation_diagnostics,
    plot_partial_autocorrelation,
    plot_seasonality,
)


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
        "y": [100 + i % 30 for i in range(366)],
        "y2": [150 + (i % 20) * 2 for i in range(366)],
    })


@pytest.fixture
def multi_column_df():
    """Create DataFrame with multiple columns for correlation testing."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 31), "1d", eager=True),
        "y1": list(range(31)),
        "y2": [x * 2 + 5 for x in range(31)],
        "y3": [x**2 for x in range(31)],
    })


# Tests for plot_autocorrelation


def test_plot_autocorrelation_basic(sample_df):
    """Test basic autocorrelation plot."""
    fig = plot_autocorrelation(sample_df, columns="y", max_lags=10)
    assert len(fig.data) > 0
    assert isinstance(fig.data[0], go.Bar)


def test_plot_autocorrelation_auto_max_lags(sample_df):
    """Test autocorrelation with automatic max_lags."""
    fig = plot_autocorrelation(sample_df, columns="y")
    assert len(fig.data) > 0


def test_plot_autocorrelation_no_confidence(sample_df):
    """Test autocorrelation without confidence bands."""
    fig = plot_autocorrelation(sample_df, columns="y", max_lags=10, show_confidence=False)
    # Should have fewer traces without confidence bands
    assert len(fig.data) > 0


def test_plot_autocorrelation_custom_styling(sample_df):
    """Test autocorrelation with custom styling."""
    fig = plot_autocorrelation(sample_df, columns="y", max_lags=10, bar_color="#FF0000")
    assert len(fig.data) > 0


def test_plot_autocorrelation_panel_not_implemented(sample_df):
    """Test that panel grouping raises NotImplementedError."""
    df = sample_df.with_columns(pl.lit("A").alias("group"))
    with pytest.raises(NotImplementedError, match="Panel grouping not yet implemented"):
        plot_autocorrelation(df, columns="y", panel_group_name="group")


# Tests for plot_partial_autocorrelation


def test_plot_partial_autocorrelation_basic(sample_df):
    """Test basic PACF plot."""
    fig = plot_partial_autocorrelation(sample_df, columns="y", max_lags=10)
    assert len(fig.data) > 0
    assert isinstance(fig.data[0], go.Bar)


def test_plot_partial_autocorrelation_auto_max_lags(sample_df):
    """Test PACF with automatic max_lags."""
    fig = plot_partial_autocorrelation(sample_df, columns="y")
    assert len(fig.data) > 0


def test_plot_partial_autocorrelation_no_confidence(sample_df):
    """Test PACF without confidence bands."""
    fig = plot_partial_autocorrelation(sample_df, columns="y", max_lags=10, show_confidence=False)
    assert len(fig.data) > 0


def test_plot_partial_autocorrelation_custom_styling(sample_df):
    """Test PACF with custom styling."""
    fig = plot_partial_autocorrelation(sample_df, columns="y", max_lags=10, bar_color="#00FF00")
    assert len(fig.data) > 0


def test_plot_partial_autocorrelation_panel_not_implemented(sample_df):
    """Test that panel grouping raises NotImplementedError."""
    df = sample_df.with_columns(pl.lit("A").alias("group"))
    with pytest.raises(NotImplementedError, match="Panel grouping not yet implemented"):
        plot_partial_autocorrelation(df, columns="y", panel_group_name="group")


# Tests for plot_correlation_diagnostics


def test_plot_correlation_diagnostics_basic(multi_column_df):
    """Test basic correlation matrix."""
    fig = plot_correlation_diagnostics(multi_column_df)
    assert len(fig.data) > 0
    assert isinstance(fig.data[0], go.Heatmap)


def test_plot_correlation_diagnostics_no_values(multi_column_df):
    """Test correlation matrix without values displayed."""
    fig = plot_correlation_diagnostics(multi_column_df, show_values=False)
    assert len(fig.data) > 0


def test_plot_correlation_diagnostics_custom_colorscale(multi_column_df):
    """Test correlation matrix with custom colorscale."""
    fig = plot_correlation_diagnostics(multi_column_df, colorscale="Viridis")
    assert len(fig.data) > 0


def test_plot_correlation_diagnostics_subset_columns(multi_column_df):
    """Test correlation matrix with subset of columns."""
    fig = plot_correlation_diagnostics(multi_column_df, columns=["y1", "y2"])
    assert len(fig.data) > 0


def test_plot_correlation_diagnostics_panel_not_implemented(multi_column_df):
    """Test that panel grouping raises NotImplementedError."""
    df = multi_column_df.with_columns(pl.lit("A").alias("group"))
    with pytest.raises(NotImplementedError, match="Panel grouping not yet implemented"):
        plot_correlation_diagnostics(df, panel_group_name="group")


# Tests for plot_seasonality


def test_plot_seasonality_month(sample_df):
    """Test monthly seasonality plot."""
    fig = plot_seasonality(sample_df, columns="y", frequency="month")
    assert len(fig.data) > 0


def test_plot_seasonality_quarter(sample_df):
    """Test quarterly seasonality plot."""
    fig = plot_seasonality(sample_df, columns="y", frequency="quarter")
    assert len(fig.data) > 0


def test_plot_seasonality_weekday(sample_df):
    """Test weekday seasonality plot."""
    fig = plot_seasonality(sample_df, columns="y", frequency="weekday")
    assert len(fig.data) > 0


def test_plot_seasonality_no_mean(sample_df):
    """Test seasonality plot without mean line."""
    fig = plot_seasonality(sample_df, columns="y", frequency="month", show_mean=False)
    assert len(fig.data) > 0


def test_plot_seasonality_custom_styling(sample_df):
    """Test seasonality with custom styling."""
    fig = plot_seasonality(
        sample_df,
        columns="y",
        frequency="month",
        line_width=2.5,
        mean_color="#FF00FF",
    )
    assert len(fig.data) > 0


def test_plot_seasonality_invalid_frequency(sample_df):
    """Test that invalid frequency raises ValueError."""
    with pytest.raises(ValueError, match="Unknown frequency"):
        plot_seasonality(sample_df, columns="y", frequency="invalid")


def test_plot_seasonality_panel_not_implemented(sample_df):
    """Test that panel grouping raises NotImplementedError."""
    df = sample_df.with_columns(pl.lit("A").alias("group"))
    with pytest.raises(NotImplementedError, match="Panel grouping not yet implemented"):
        plot_seasonality(df, columns="y", frequency="month", panel_group_name="group")
