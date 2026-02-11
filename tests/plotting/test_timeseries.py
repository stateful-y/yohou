"""Unit tests for timeseries plotting functions."""

import polars as pl
import pytest

from yohou.plotting import (
    plot_boxplot,
    plot_exponential_moving_average,
    plot_prediction_interval,
    plot_rolling_statistics,
    plot_timeseries,
)


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
        "y": [100, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180, 175],
        "y2": [110, 125, 120, 135, 145, 140, 155, 165, 160, 175, 185, 180],
    })


def test_plot_timeseries_single_column(sample_df):
    """Test plotting a single column."""
    fig = plot_timeseries(sample_df, columns="y")
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].mode == "lines"
    assert fig.data[0].name == "y"


def test_plot_timeseries_multiple_columns(sample_df):
    """Test plotting multiple columns."""
    fig = plot_timeseries(sample_df, columns=["y", "y2"])
    assert len(fig.data) == 2
    assert fig.data[0].name == "y"
    assert fig.data[1].name == "y2"


def test_plot_timeseries_all_columns(sample_df):
    """Test plotting all numeric columns (default)."""
    fig = plot_timeseries(sample_df, columns=None)
    # Should plot y and y2 (not time)
    assert len(fig.data) == 2


def test_plot_timeseries_with_title(sample_df):
    """Test plot with custom title."""
    fig = plot_timeseries(sample_df, columns="y", title="Test Plot")
    assert fig.layout.title.text == "Test Plot"


def test_plot_timeseries_custom_labels(sample_df):
    """Test plot with custom axis labels."""
    fig = plot_timeseries(sample_df, columns="y", x_label="Date", y_label="Value")
    assert fig.layout.xaxis.title.text == "Date"
    assert fig.layout.yaxis.title.text == "Value"


def test_plot_timeseries_custom_size(sample_df):
    """Test plot with custom dimensions."""
    fig = plot_timeseries(sample_df, columns="y", width=800, height=600)
    assert fig.layout.width == 800
    assert fig.layout.height == 600


def test_plot_timeseries_custom_styling(sample_df):
    """Test plot with custom styling via kwargs."""
    fig = plot_timeseries(sample_df, columns="y", line_width=3.0, line_color="#DC2626", line_dash="dash")
    assert fig.data[0].line.width == 3.0
    assert fig.data[0].line.color == "#DC2626"
    assert fig.data[0].line.dash == "dash"


def test_plot_timeseries_no_legend(sample_df):
    """Test plot without legend."""
    fig = plot_timeseries(sample_df, columns=["y", "y2"], show_legend=False)
    assert fig.layout.showlegend is False


def test_plot_timeseries_panel_not_implemented(sample_df):
    """Test that panel grouping raises NotImplementedError."""
    df = sample_df.with_columns(pl.lit("A").alias("group"))
    with pytest.raises(NotImplementedError, match="Panel grouping not yet implemented"):
        plot_timeseries(df, columns="y", panel_group_name="group")


# Tests for plot_rolling_statistics


def test_plot_rolling_statistics_basic(sample_df):
    """Test basic rolling mean."""
    fig = plot_rolling_statistics(sample_df, columns="y", window_size=3, statistics="mean")
    # Should have original + rolling mean
    assert len(fig.data) == 2


def test_plot_rolling_statistics_no_original(sample_df):
    """Test rolling mean without original series."""
    fig = plot_rolling_statistics(sample_df, columns="y", window_size=3, statistics="mean", show_original=False)
    assert len(fig.data) == 1


def test_plot_rolling_statistics_multiple_stats(sample_df):
    """Test multiple statistics."""
    fig = plot_rolling_statistics(sample_df, columns="y", window_size=3, statistics=["mean", "std"])
    # Original + 2 statistics
    assert len(fig.data) == 3


def test_plot_rolling_statistics_all_stat_types(sample_df):
    """Test all available statistics."""
    stats = ["mean", "std", "min", "max", "median", "q25", "q75", "sum"]
    for stat in stats:
        fig = plot_rolling_statistics(sample_df, columns="y", window_size=3, statistics=stat, show_original=False)
        assert len(fig.data) == 1


def test_plot_rolling_statistics_fill_between(sample_df):
    """Test fill_between with two statistics."""
    fig = plot_rolling_statistics(
        sample_df, columns="y", window_size=3, statistics=["min", "max"], fill_between=True, show_original=False
    )
    # Should have 2 traces for the band
    assert len(fig.data) == 2


def test_plot_rolling_statistics_invalid_stat(sample_df):
    """Test that invalid statistic raises error."""
    with pytest.raises(ValueError, match="Unknown statistic"):
        plot_rolling_statistics(sample_df, columns="y", window_size=3, statistics="invalid")


def test_plot_rolling_statistics_panel_not_implemented(sample_df):
    """Test that panel grouping raises NotImplementedError."""
    df = sample_df.with_columns(pl.lit("A").alias("group"))
    with pytest.raises(NotImplementedError, match="Panel grouping not yet implemented"):
        plot_rolling_statistics(df, columns="y", window_size=3, panel_group_name="group")


# Tests for plot_exponential_moving_average


def test_plot_exponential_moving_average_basic(sample_df):
    """Test basic EWM plot."""
    fig = plot_exponential_moving_average(sample_df, columns="y", span=3)
    # Should have original + EWM
    assert len(fig.data) == 2


def test_plot_exponential_moving_average_no_original(sample_df):
    """Test EWM without original series."""
    fig = plot_exponential_moving_average(sample_df, columns="y", span=3, show_original=False)
    assert len(fig.data) == 1
    assert "EWM" in fig.data[0].name


def test_plot_exponential_moving_average_custom_span(sample_df):
    """Test EWM with custom span."""
    fig = plot_exponential_moving_average(sample_df, columns="y", span=6)
    assert len(fig.data) == 2
    assert "EWM(6)" in fig.data[1].name


def test_plot_exponential_moving_average_styling(sample_df):
    """Test EWM with custom styling."""
    fig = plot_exponential_moving_average(
        sample_df, columns="y", span=3, show_original=False, smooth_color="#DC2626", smooth_width=3.0
    )
    assert fig.data[0].line.color == "#DC2626"
    assert fig.data[0].line.width == 3.0


def test_plot_exponential_moving_average_panel_not_implemented(sample_df):
    """Test that panel grouping raises NotImplementedError."""
    df = sample_df.with_columns(pl.lit("A").alias("group"))
    with pytest.raises(NotImplementedError, match="Panel grouping not yet implemented"):
        plot_exponential_moving_average(df, columns="y", span=3, panel_group_name="group")


# Tests for plot_boxplot


def test_plot_boxplot_basic(sample_df):
    """Test basic boxplot."""
    fig = plot_boxplot(sample_df, columns="y", period="1mo")
    assert len(fig.data) > 0


def test_plot_boxplot_different_periods():
    """Test boxplot with different periods."""
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1w", eager=True),
        "y": [100 + i * 2 + (i % 4) * 10 for i in range(53)],
    })

    for period in ["1w", "1mo", "1q"]:
        fig = plot_boxplot(df, columns="y", period=period)
        assert len(fig.data) > 0


def test_plot_boxplot_show_points(sample_df):
    """Test boxplot with point display options."""
    fig = plot_boxplot(sample_df, columns="y", period="1mo", show_points="all")
    assert len(fig.data) > 0

    fig = plot_boxplot(sample_df, columns="y", period="1mo", show_points=False)
    assert len(fig.data) > 0


def test_plot_boxplot_styling(sample_df):
    """Test boxplot with custom styling."""
    fig = plot_boxplot(sample_df, columns="y", period="1mo", box_color="#DC2626", box_alpha=0.9)
    assert len(fig.data) > 0


def test_plot_boxplot_panel_not_implemented(sample_df):
    """Test that panel grouping raises NotImplementedError."""
    df = sample_df.with_columns(pl.lit("A").alias("group"))
    with pytest.raises(NotImplementedError, match="Panel grouping not yet implemented"):
        plot_boxplot(df, columns="y", period="1mo", panel_group_name="group")


# Tests for plot_prediction_interval


def test_plot_prediction_interval_basic():
    """Test basic prediction interval plot."""
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
        "forecast": [100, 105, 110, 108, 112, 115, 118, 120, 122, 125],
        "lower_bound": [95, 98, 103, 100, 105, 108, 110, 112, 114, 117],
        "upper_bound": [105, 112, 117, 116, 119, 122, 126, 128, 130, 133],
    })
    fig = plot_prediction_interval(df, columns="forecast")
    assert len(fig.data) > 0


def test_plot_prediction_interval_custom_bounds():
    """Test prediction interval with custom bound columns."""
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
        "y": list(range(10)),
        "lower": [x - 2 for x in range(10)],
        "upper": [x + 2 for x in range(10)],
    })
    fig = plot_prediction_interval(df, columns="y", lower_bound_column="lower", upper_bound_column="upper")
    assert len(fig.data) > 0


def test_plot_prediction_interval_styling():
    """Test prediction interval with custom styling."""
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
        "forecast": list(range(10)),
        "lower_bound": [x - 1 for x in range(10)],
        "upper_bound": [x + 1 for x in range(10)],
    })
    fig = plot_prediction_interval(
        df,
        columns="forecast",
        line_width=3.0,
        line_color="#DC2626",
        band_alpha=0.3,
    )
    assert len(fig.data) > 0


def test_plot_prediction_interval_missing_column():
    """Test that missing bound column raises ValueError."""
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
        "y": list(range(10)),
    })
    with pytest.raises(ValueError, match="Lower bound column"):
        plot_prediction_interval(df)


def test_plot_prediction_interval_panel_not_implemented():
    """Test that panel grouping raises NotImplementedError."""
    df = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
        "forecast": list(range(10)),
        "lower_bound": [x - 1 for x in range(10)],
        "upper_bound": [x + 1 for x in range(10)],
        "group": ["A"] * 10,
    })
    with pytest.raises(NotImplementedError, match="Panel grouping not yet implemented"):
        plot_prediction_interval(df, columns="forecast", panel_group_name="group")
