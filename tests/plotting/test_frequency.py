"""Tests for frequency analysis plotting functions."""

import polars as pl
import pytest
from plotly import graph_objects as go

from yohou.plotting import plot_lag_scatter, plot_periodogram


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
        "y": [100 + i % 20 for i in range(91)],
        "y2": [150 + (i % 15) * 2 for i in range(91)],
    })


# Tests for plot_lag_scatter


def test_plot_lag_scatter_single_lag(sample_df):
    """Test lag scatter with single lag."""
    fig = plot_lag_scatter(sample_df, columns="y", lags=1)
    assert len(fig.data) > 0
    assert isinstance(fig.data[0], go.Scatter)


def test_plot_lag_scatter_multiple_lags(sample_df):
    """Test lag scatter with multiple lags."""
    fig = plot_lag_scatter(sample_df, columns="y", lags=[1, 7, 14])
    assert len(fig.data) > 0


def test_plot_lag_scatter_with_diagonal(sample_df):
    """Test lag scatter with diagonal line."""
    fig = plot_lag_scatter(sample_df, columns="y", lags=1, show_diagonal=True)
    assert len(fig.data) > 0


def test_plot_lag_scatter_with_regression(sample_df):
    """Test lag scatter with regression line."""
    fig = plot_lag_scatter(sample_df, columns="y", lags=1, show_regression=True)
    assert len(fig.data) > 0


def test_plot_lag_scatter_custom_styling(sample_df):
    """Test lag scatter with custom styling."""
    fig = plot_lag_scatter(
        sample_df,
        columns="y",
        lags=1,
        marker_size=6.0,
        marker_alpha=0.8,
    )
    assert len(fig.data) > 0


def test_plot_lag_scatter_panel_not_implemented(sample_df):
    """Test that panel grouping raises NotImplementedError."""
    df = sample_df.with_columns(pl.lit("A").alias("group"))
    with pytest.raises(NotImplementedError, match="Panel grouping not yet implemented"):
        plot_lag_scatter(df, columns="y", lags=1, panel_group_name="group")


def test_plot_periodogram_basic():
    """Test basic periodogram functionality."""
    import numpy as np

    # Create periodic signal
    t = np.arange(100)
    y = np.sin(2 * np.pi * 0.1 * t)
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
        "y": y,
    })

    fig = plot_periodogram(df, columns="y")
    assert len(fig.data) >= 1
    assert fig.data[0].type == "scatter"


def test_plot_periodogram_detrending():
    """Test detrending options."""
    import numpy as np

    t = np.arange(100)
    y = np.sin(2 * np.pi * 0.1 * t) + t * 0.1  # Add trend
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
        "y": y,
    })

    # Test different detrending methods
    fig_none = plot_periodogram(df, detrend="none")
    fig_mean = plot_periodogram(df, detrend="mean")
    fig_linear = plot_periodogram(df, detrend="linear")

    assert len(fig_none.data) >= 1
    assert len(fig_mean.data) >= 1
    assert len(fig_linear.data) >= 1


def test_plot_periodogram_log_scale():
    """Test log scale option."""
    import numpy as np

    t = np.arange(100)
    y = np.sin(2 * np.pi * 0.1 * t)
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
        "y": y,
    })

    fig = plot_periodogram(df, log_scale=True)
    assert len(fig.data) >= 1
    assert fig.layout.yaxis.type == "log"


def test_plot_periodogram_peaks():
    """Test peak detection."""
    import numpy as np

    t = np.arange(100)
    y = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.25 * t)
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
        "y": y,
    })

    fig = plot_periodogram(df, show_peaks=True, n_peaks=2)
    assert len(fig.data) >= 2  # Line trace + peak markers


def test_plot_periodogram_multiple_columns():
    """Test plotting multiple columns."""
    import numpy as np

    t = np.arange(100)
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
        "y1": np.sin(2 * np.pi * 0.1 * t),
        "y2": np.sin(2 * np.pi * 0.2 * t),
    })

    fig = plot_periodogram(df, columns=["y1", "y2"])
    assert len(fig.data) >= 2


def test_plot_periodogram_panel_not_implemented(sample_df):
    """Test that panel grouping raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Panel grouping"):
        plot_periodogram(sample_df, panel_group_name="id")
