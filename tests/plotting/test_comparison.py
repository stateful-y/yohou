"""Tests for comparison plotting functions."""

import polars as pl
import pytest

from yohou.plotting import plot_comparison, plot_forecast, plot_residuals


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
        "y": [100 + i for i in range(91)],
        "fitted": [100 + i + (i % 3) for i in range(91)],
        "model1": [100 + i + (i % 5) for i in range(91)],
        "model2": [100 + i + (i % 7) for i in range(91)],
    }).with_columns((pl.col("y") - pl.col("fitted")).alias("residuals"))


# Tests for plot_residuals


def test_plot_residuals_basic(sample_df):
    """Test basic residual plotting."""
    fig = plot_residuals(sample_df)
    assert len(fig.data) > 0


def test_plot_residuals_without_fitted(sample_df):
    """Test residuals without fitted column."""
    df = sample_df.drop("fitted")
    fig = plot_residuals(df, fitted_column=None)
    assert len(fig.data) > 0


def test_plot_residuals_custom_columns(sample_df):
    """Test with custom column names."""
    df = sample_df.rename({"residuals": "resid", "fitted": "fit"})
    fig = plot_residuals(df, residuals_column="resid", fitted_column="fit")
    assert len(fig.data) > 0


def test_plot_residuals_missing_column(sample_df):
    """Test error when residuals column missing."""
    with pytest.raises(ValueError, match="not found"):
        plot_residuals(sample_df, residuals_column="missing")


def test_plot_residuals_panel_not_implemented(sample_df):
    """Test panel grouping raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Panel grouping"):
        plot_residuals(sample_df, panel_group_name="group")


# Tests for plot_forecast


def test_plot_forecast_basic():
    """Test basic forecast plotting."""
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 30), "1d", eager=True),
        "y": [100 + i for i in range(121)],
        "is_forecast": [False] * 91 + [True] * 30,
    })
    fig = plot_forecast(df, n_history=30)
    assert len(fig.data) > 0


def test_plot_forecast_with_intervals():
    """Test forecast with prediction intervals."""
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 30), "1d", eager=True),
        "y": [100 + i for i in range(121)],
        "is_forecast": [False] * 91 + [True] * 30,
        "lower_bound": [95 + i for i in range(121)],
        "upper_bound": [105 + i for i in range(121)],
    })
    fig = plot_forecast(df, n_history=30)
    assert len(fig.data) > 0


def test_plot_forecast_missing_column():
    """Test error when is_forecast column missing."""
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
        "y": [100 + i for i in range(91)],
    })
    with pytest.raises(ValueError, match="not found"):
        plot_forecast(df)


def test_plot_forecast_panel_not_implemented():
    """Test panel grouping raises NotImplementedError."""
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
        "y": [100 + i for i in range(91)],
        "is_forecast": [False] * 91,
    })
    with pytest.raises(NotImplementedError, match="Panel grouping"):
        plot_forecast(df, panel_group_name="group")


# Tests for plot_comparison


def test_plot_comparison_overlay(sample_df):
    """Test comparison with overlay mode."""
    fig = plot_comparison(sample_df, columns=["y", "model1", "model2"])
    assert len(fig.data) >= 3


def test_plot_comparison_facet(sample_df):
    """Test comparison with facet mode."""
    fig = plot_comparison(sample_df, columns=["y", "model1"], comparison_mode="facet")
    assert len(fig.data) >= 2


def test_plot_comparison_difference(sample_df):
    """Test comparison with difference mode."""
    fig = plot_comparison(
        sample_df, columns=["y", "model1", "model2"], comparison_mode="difference", reference_column="y"
    )
    assert len(fig.data) >= 2


def test_plot_comparison_empty_columns(sample_df):
    """Test error when no columns provided."""
    with pytest.raises(ValueError, match="at least one column"):
        plot_comparison(sample_df, columns=[])


def test_plot_comparison_invalid_mode(sample_df):
    """Test error for invalid comparison mode."""
    with pytest.raises(ValueError, match="Unknown comparison_mode"):
        plot_comparison(sample_df, columns=["y", "model1"], comparison_mode="invalid")


def test_plot_comparison_panel_not_implemented(sample_df):
    """Test panel grouping raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Panel grouping"):
        plot_comparison(sample_df, columns=["y"], panel_group_name="group")
