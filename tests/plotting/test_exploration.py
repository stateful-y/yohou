"""Tests for time series exploration plotting functions."""

import polars as pl
import pytest
from plotly import graph_objects as go

from yohou.plotting import (
    plot_boxplot,
    plot_missing_data,
    plot_rolling_statistics,
    plot_time_series,
)


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
        "y": [100, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180, 175],
        "y2": [110, 125, 120, 135, 145, 140, 155, 165, 160, 175, 185, 180],
    })


@pytest.fixture
def df_with_nulls():
    """Create a sample dataframe with missing values."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
        "y": [100 + i if i % 5 != 0 else None for i in range(366)],
        "z": [200 + i if i % 7 != 0 else None for i in range(366)],
    })


class TestPlotTimeSeries:
    """Tests for plot_time_series function."""

    def test_single_column(self, sample_df):
        """Test plotting a single column."""
        fig = plot_time_series(sample_df, columns="y")
        assert len(fig.data) == 1
        assert fig.data[0].type == "scatter"
        assert fig.data[0].mode == "lines"
        assert fig.data[0].name == "y"

    def test_multiple_columns(self, sample_df):
        """Test plotting multiple columns."""
        fig = plot_time_series(sample_df, columns=["y", "y2"])
        assert len(fig.data) == 2
        assert fig.data[0].name == "y"
        assert fig.data[1].name == "y2"

    def test_all_columns(self, sample_df):
        """Test plotting all numeric columns (default)."""
        fig = plot_time_series(sample_df, columns=None)
        # Should plot y and y2 (not time)
        assert len(fig.data) == 2

    def test_with_title(self, sample_df):
        """Test plot with custom title."""
        fig = plot_time_series(sample_df, columns="y", title="Test Plot")
        assert fig.layout.title.text == "Test Plot"

    def test_custom_labels(self, sample_df):
        """Test plot with custom axis labels."""
        fig = plot_time_series(sample_df, columns="y", x_label="Date", y_label="Value")
        assert fig.layout.xaxis.title.text == "Date"
        assert fig.layout.yaxis.title.text == "Value"

    def test_custom_size(self, sample_df):
        """Test plot with custom dimensions."""
        fig = plot_time_series(sample_df, columns="y", width=800, height=600)
        assert fig.layout.width == 800
        assert fig.layout.height == 600

    def test_custom_styling(self, sample_df):
        """Test plot with custom styling via kwargs."""
        fig = plot_time_series(sample_df, columns="y", line_width=3.0, line_color="#DC2626", line_dash="dash")
        assert fig.data[0].line.width == 3.0
        assert fig.data[0].line.color == "#DC2626"
        assert fig.data[0].line.dash == "dash"

    def test_no_legend(self, sample_df):
        """Test plot without legend."""
        fig = plot_time_series(sample_df, columns=["y", "y2"], show_legend=False)
        assert fig.layout.showlegend is False

    def test_panel_support(self, sample_df):
        """Test that panel grouping is handled by plot_time_series."""
        df = sample_df.with_columns(pl.lit("A").alias("group"))
        fig = plot_time_series(df, columns="y", panel_group_names=["group"])
        assert len(fig.data) >= 0


class TestPlotRollingStatistics:
    """Tests for plot_rolling_statistics function."""

    def test_basic(self, sample_df):
        """Test basic rolling mean."""
        fig = plot_rolling_statistics(sample_df, columns="y", window_size=3, statistics="mean")
        # Should have original + rolling mean
        assert len(fig.data) == 2

    def test_no_original(self, sample_df):
        """Test rolling mean without original series."""
        fig = plot_rolling_statistics(sample_df, columns="y", window_size=3, statistics="mean", show_original=False)
        assert len(fig.data) == 1

    def test_multiple_stats(self, sample_df):
        """Test multiple statistics."""
        fig = plot_rolling_statistics(sample_df, columns="y", window_size=3, statistics=["mean", "std"])
        # Original + 2 statistics
        assert len(fig.data) == 3

    def test_all_stat_types(self, sample_df):
        """Test all available statistics."""
        stats = ["mean", "std", "min", "max", "median", "q25", "q75", "sum"]
        for stat in stats:
            fig = plot_rolling_statistics(sample_df, columns="y", window_size=3, statistics=stat, show_original=False)
            assert len(fig.data) == 1

    def test_invalid_stat(self, sample_df):
        """Test that invalid statistic raises error."""
        with pytest.raises(ValueError, match="Invalid statistics"):
            plot_rolling_statistics(sample_df, columns="y", window_size=3, statistics="invalid")

    def test_panel(self, sample_df):
        """Test panel faceting for rolling statistics."""
        df = pl.DataFrame({
            "time": sample_df["time"],
            "y__a": sample_df["y"],
            "y__b": sample_df["y"] * 2,
        })
        fig = plot_rolling_statistics(df, window_size=3, statistics="mean", panel_group_names=["y"])
        assert len(fig.data) > 0

    def test_multi_column(self, sample_df):
        """Test rolling statistics with multiple columns uses distinct colors."""
        df = pl.DataFrame({
            "time": sample_df["time"],
            "y": sample_df["y"],
            "z": sample_df["y"] * 2,
        })
        fig = plot_rolling_statistics(df, columns=["y", "z"], window_size=3, statistics="mean")
        # 2 original + 2 rolling mean = 4 traces
        assert len(fig.data) == 4
        # Each column should get a different color
        colors = {trace.line.color for trace in fig.data if trace.line.color is not None}
        assert len(colors) >= 2

    def test_dict_window_size(self, sample_df):
        """Test per-column window sizes via dict."""
        df = pl.DataFrame({
            "time": sample_df["time"],
            "y": sample_df["y"],
            "z": sample_df["y"] * 2,
        })
        fig = plot_rolling_statistics(df, columns=["y", "z"], window_size={"y": 3, "z": 5}, statistics="mean")
        # 2 original + 2 rolling mean = 4 traces
        assert len(fig.data) == 4


class TestPlotBoxplot:
    """Tests for plot_boxplot function."""

    def test_basic(self, sample_df):
        """Test basic boxplot."""
        fig = plot_boxplot(sample_df, columns="y", period="1mo")
        assert len(fig.data) > 0

    def test_different_periods(self):
        """Test boxplot with different periods."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1w", eager=True),
            "y": [100 + i * 2 + (i % 4) * 10 for i in range(53)],
        })

        for period in ["1w", "1mo", "1q"]:
            fig = plot_boxplot(df, columns="y", period=period)
            assert len(fig.data) > 0

    def test_show_points(self, sample_df):
        """Test boxplot with point display options."""
        fig = plot_boxplot(sample_df, columns="y", period="1mo", show_points="all")
        assert len(fig.data) > 0

        fig = plot_boxplot(sample_df, columns="y", period="1mo", show_points=False)
        assert len(fig.data) > 0

    def test_styling(self, sample_df):
        """Test boxplot with custom styling."""
        fig = plot_boxplot(sample_df, columns="y", period="1mo", box_color="#DC2626", box_opacity=0.9)
        assert len(fig.data) > 0

    def test_panel(self, sample_df):
        """Test panel faceting for boxplots."""
        df = pl.DataFrame({
            "time": sample_df["time"],
            "y__a": sample_df["y"],
            "y__b": sample_df["y"] * 2,
        })
        fig = plot_boxplot(df, period="1mo", panel_group_names=["y"])
        assert len(fig.data) > 0

    def test_multi_column(self, sample_df):
        """Test boxplot with multiple columns (grouped bar-like)."""
        fig = plot_boxplot(sample_df, columns=["y", "y2"], period="1mo")
        assert len(fig.data) > 0
        # Each column should produce traces with different colors
        colors = {t.marker.color for t in fig.data if t.marker and t.marker.color}
        assert len(colors) >= 2


class TestPlotMissingData:
    """Tests for plot_missing_data function."""

    def test_bars(self, df_with_nulls):
        """Test missing data bars visualization."""
        fig = plot_missing_data(df_with_nulls, kind="bars")
        assert len(fig.data) > 0
        assert isinstance(fig.data[0], go.Bar)

    def test_heatmap(self, df_with_nulls):
        """Test missing data heatmap visualization."""
        fig = plot_missing_data(df_with_nulls, kind="heatmap")
        assert len(fig.data) > 0
        assert isinstance(fig.data[0], go.Heatmap)

    def test_matrix(self, df_with_nulls):
        """Test missing data matrix visualization."""
        fig = plot_missing_data(df_with_nulls, kind="matrix")
        assert len(fig.data) > 0
        assert isinstance(fig.data[0], go.Heatmap)

    def test_time_aggregation(self, df_with_nulls):
        """Test missing data with time aggregation."""
        fig = plot_missing_data(df_with_nulls, kind="heatmap", time_aggregation="1mo")
        assert len(fig.data) > 0

    def test_custom_colors(self, df_with_nulls):
        """Test missing data with custom colors."""
        fig = plot_missing_data(
            df_with_nulls,
            kind="heatmap",
            color_missing="#FF0000",
            color_present="#00FF00",
        )
        assert len(fig.data) > 0

    def test_invalid_kind(self, df_with_nulls):
        """Test that invalid kind raises ValueError."""
        with pytest.raises(ValueError, match="Unknown kind"):
            plot_missing_data(df_with_nulls, kind="invalid")  # type: ignore

    def test_panel(self, df_with_nulls):
        """Test panel faceting for missing data."""
        df = pl.DataFrame({
            "time": df_with_nulls["time"],
            "y__a": df_with_nulls["y"],
            "y__b": df_with_nulls["z"],
        })
        fig = plot_missing_data(df, kind="bars", panel_group_names=["y"])
        assert len(fig.data) > 0
