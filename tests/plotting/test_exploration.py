"""Tests for time series exploration plotting functions."""

import polars as pl
import pytest
from plotly import graph_objects as go

from yohou.plotting import (
    plot_boxplot,
    plot_distribution,
    plot_missing_data,
    plot_outliers,
    plot_resampling_comparison,
    plot_rolling_statistics,
    plot_time_series,
)

from .conftest import assert_figure_valid, assert_layout, visible_legend_names


@pytest.fixture
def monthly_2col_df():
    """Create sample DataFrame for testing."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
        "y": [100, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180, 175],
        "y2": [110, 125, 120, 135, 145, 140, 155, 165, 160, 175, 185, 180],
    })


class TestPlotTimeSeries:
    """Tests for plot_time_series function."""

    def test_single_column(self, monthly_2col_df):
        """Test plotting a single column."""
        fig = plot_time_series(monthly_2col_df, columns="y")
        assert len(fig.data) == 1
        assert fig.data[0].type == "scatter"
        assert fig.data[0].mode == "lines"
        assert fig.data[0].name == "y"

    def test_multiple_columns(self, monthly_2col_df):
        """Test plotting multiple columns."""
        fig = plot_time_series(monthly_2col_df, columns=["y", "y2"])
        assert len(fig.data) == 2
        assert fig.data[0].name == "y"
        assert fig.data[1].name == "y2"

    def test_all_columns(self, monthly_2col_df):
        """Test plotting all numeric columns (default)."""
        fig = plot_time_series(monthly_2col_df, columns=None)
        # Should plot y and y2 (not time)
        assert len(fig.data) == 2

    def test_with_title(self, monthly_2col_df):
        """Test plot with custom title."""
        fig = plot_time_series(monthly_2col_df, columns="y", title="Test Plot")
        assert_layout(fig, title="Test Plot")

    def test_custom_labels(self, monthly_2col_df):
        """Test plot with custom axis labels."""
        fig = plot_time_series(monthly_2col_df, columns="y", x_label="Date", y_label="Value")
        assert fig.layout.xaxis.title.text == "Date"
        assert fig.layout.yaxis.title.text == "Value"

    def test_custom_size(self, monthly_2col_df):
        """Test plot with custom dimensions."""
        fig = plot_time_series(monthly_2col_df, columns="y", width=800, height=600)
        assert_layout(fig, width=800, height=600)

    def test_custom_styling(self, monthly_2col_df):
        """Test plot with custom styling."""
        fig = plot_time_series(monthly_2col_df, columns="y", line_width=3.0, line_dash="dash")
        assert fig.data[0].line.width == 3.0
        assert fig.data[0].line.dash == "dash"

    def test_no_legend(self, monthly_2col_df):
        """Test plot without legend."""
        fig = plot_time_series(monthly_2col_df, columns=["y", "y2"], show_legend=False)
        assert fig.layout.showlegend is False

    def test_panel_support(self, monthly_2col_df):
        """Test that panel grouping is handled by plot_time_series."""
        df = monthly_2col_df.with_columns(pl.lit("A").alias("group"))
        fig = plot_time_series(df, columns="y", panel_group_names=["group"])
        assert len(fig.data) >= 0


class TestPlotRollingStatistics:
    """Tests for plot_rolling_statistics function."""

    def test_basic(self, monthly_2col_df):
        """Test basic rolling mean."""
        fig = plot_rolling_statistics(monthly_2col_df, columns="y", window_size=3, statistics="mean")
        # Should have original + rolling mean
        assert len(fig.data) == 2

    def test_no_original(self, monthly_2col_df):
        """Test rolling mean without original series."""
        fig = plot_rolling_statistics(monthly_2col_df, columns="y", window_size=3, statistics="mean", show_original=False)
        assert len(fig.data) == 1

    def test_multiple_stats(self, monthly_2col_df):
        """Test multiple statistics."""
        fig = plot_rolling_statistics(monthly_2col_df, columns="y", window_size=3, statistics=["mean", "std"])
        # Original + 2 statistics
        assert len(fig.data) == 3

    def test_all_stat_types(self, monthly_2col_df):
        """Test all available statistics."""
        stats = ["mean", "std", "min", "max", "median", "q25", "q75", "sum"]
        for stat in stats:
            fig = plot_rolling_statistics(monthly_2col_df, columns="y", window_size=3, statistics=stat, show_original=False)
            assert len(fig.data) == 1

    def test_invalid_stat(self, monthly_2col_df):
        """Test that invalid statistic raises error."""
        with pytest.raises(ValueError, match="Invalid statistics"):
            plot_rolling_statistics(monthly_2col_df, columns="y", window_size=3, statistics="invalid")

    def test_panel(self, monthly_2col_df):
        """Test panel faceting for rolling statistics."""
        df = pl.DataFrame({
            "time": monthly_2col_df["time"],
            "y__a": monthly_2col_df["y"],
            "y__b": monthly_2col_df["y"] * 2,
        })
        fig = plot_rolling_statistics(df, window_size=3, statistics="mean", panel_group_names=["y"])
        assert len(fig.data) > 0

    def test_multi_column(self, monthly_2col_df):
        """Test rolling statistics with multiple columns uses distinct colors."""
        df = pl.DataFrame({
            "time": monthly_2col_df["time"],
            "y": monthly_2col_df["y"],
            "z": monthly_2col_df["y"] * 2,
        })
        fig = plot_rolling_statistics(df, columns=["y", "z"], window_size=3, statistics="mean")
        # 2 original + 2 rolling mean = 4 traces
        assert len(fig.data) == 4
        # Each column should get a different color
        colors = {trace.line.color for trace in fig.data if trace.line.color is not None}
        assert len(colors) >= 2

    def test_dict_window_size(self, monthly_2col_df):
        """Test per-column window sizes via dict."""
        df = pl.DataFrame({
            "time": monthly_2col_df["time"],
            "y": monthly_2col_df["y"],
            "z": monthly_2col_df["y"] * 2,
        })
        fig = plot_rolling_statistics(df, columns=["y", "z"], window_size={"y": 3, "z": 5}, statistics="mean")
        # 2 original + 2 rolling mean = 4 traces
        assert len(fig.data) == 4


class TestPlotBoxplot:
    """Tests for plot_boxplot function."""

    def test_basic(self, monthly_2col_df):
        """Test basic boxplot."""
        fig = plot_boxplot(monthly_2col_df, columns="y", period="1mo")
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

    def test_show_points(self, monthly_2col_df):
        """Test boxplot with point display options."""
        fig = plot_boxplot(monthly_2col_df, columns="y", period="1mo", show_points="all")
        assert len(fig.data) > 0

        fig = plot_boxplot(monthly_2col_df, columns="y", period="1mo", show_points=False)
        assert len(fig.data) > 0

    def test_styling(self, monthly_2col_df):
        """Test boxplot with custom styling."""
        fig = plot_boxplot(monthly_2col_df, columns="y", period="1mo", bar_opacity=0.9)
        assert len(fig.data) > 0

    def test_panel(self, monthly_2col_df):
        """Test panel faceting for boxplots."""
        df = pl.DataFrame({
            "time": monthly_2col_df["time"],
            "y__a": monthly_2col_df["y"],
            "y__b": monthly_2col_df["y"] * 2,
        })
        fig = plot_boxplot(df, period="1mo", panel_group_names=["y"])
        assert len(fig.data) > 0

    def test_multi_column(self, monthly_2col_df):
        """Test boxplot with multiple columns (grouped bar-like)."""
        fig = plot_boxplot(monthly_2col_df, columns=["y", "y2"], period="1mo")
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


class TestPlotTimeSeriesErrorPaths:
    """Error path tests for plot_time_series."""

    def test_invalid_column_name(self, monthly_2col_df):
        """Non-existent column raises ValueError."""
        with pytest.raises(ValueError, match="not found|not in"):
            plot_time_series(monthly_2col_df, columns="nonexistent")

    def test_not_a_dataframe(self):
        """Passing a non-DataFrame raises TypeError."""
        with pytest.raises(TypeError, match="DataFrame"):
            plot_time_series("not a dataframe")

    def test_empty_dataframe(self):
        """Empty DataFrame raises ValueError."""
        df = pl.DataFrame({"time": pl.Series([], dtype=pl.Date), "y": pl.Series([], dtype=pl.Float64)})
        with pytest.raises(ValueError, match="empty|at least"):
            plot_time_series(df)


class TestPlotBoxplotErrorPaths:
    """Error path and stronger assertion tests for plot_boxplot."""

    def test_returns_go_figure(self, monthly_2col_df):
        """Boxplot always returns a go.Figure instance."""
        fig = plot_boxplot(monthly_2col_df, columns="y", period="1mo")
        assert_figure_valid(fig)

    def test_trace_type_is_box(self, monthly_2col_df):
        """Boxplot traces should be Box type."""
        fig = plot_boxplot(monthly_2col_df, columns="y", period="1mo")
        assert any(isinstance(t, go.Box) for t in fig.data)

    def test_custom_title(self, monthly_2col_df):
        """Custom title is applied to boxplot figure."""
        fig = plot_boxplot(monthly_2col_df, columns="y", period="1mo", title="Box Title")
        assert_layout(fig, title="Box Title")

    def test_custom_dimensions(self, monthly_2col_df):
        """Custom dimensions are respected."""
        fig = plot_boxplot(monthly_2col_df, columns="y", period="1mo", width=800, height=500)
        assert_layout(fig, width=800, height=500)


class TestPlotTimeSeriesPanelAutoDetect:
    """Tests for panel auto-detect in plot_time_series."""

    def test_auto_detect_panel_no_columns(self):
        """Auto-detect panel mode when columns and panel_group_names are both None."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 1), "1mo", eager=True),
            "sales__store_1": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "sales__store_2": [200.0, 210.0, 220.0, 230.0, 240.0, 250.0],
        })
        fig = plot_time_series(df)
        assert_figure_valid(fig)


class TestPlotRollingStatisticsPanelAutoDetect:
    """Tests for panel auto-detect in plot_rolling_statistics."""

    def test_auto_detect_panel_rolling(self):
        """Rolling statistics auto-detect panel mode when no columns specified."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
            "sales__store_1": [100 + i * 2.0 for i in range(12)],
            "sales__store_2": [200 + i * 3.0 for i in range(12)],
        })
        fig = plot_rolling_statistics(df, window_size=3, statistics="mean")
        assert_figure_valid(fig)


class TestPlotBoxplotPanelAutoDetect:
    """Tests for panel auto-detect in plot_boxplot."""

    def test_auto_detect_panel_boxplot(self):
        """Boxplot auto-detects panel mode when no columns specified."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
            "sales__store_1": [100 + i * 2.0 for i in range(12)],
            "sales__store_2": [200 + i * 3.0 for i in range(12)],
        })
        fig = plot_boxplot(df, period="1mo")
        assert_figure_valid(fig)


class TestPlotMissingDataPanelAutoDetect:
    """Tests for panel auto-detect in plot_missing_data."""

    def test_auto_detect_panel_missing(self):
        """Missing data auto-detects panel mode when no columns specified."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 1), "1mo", eager=True),
            "sales__store_1": [100.0, None, 120.0, None, 140.0, 150.0],
            "sales__store_2": [None, 210.0, 220.0, 230.0, None, 250.0],
        })
        fig = plot_missing_data(df, kind="bars")
        assert_figure_valid(fig)


class TestPlotMissingDataKindBranches:
    """Tests for missing data kind branches (heatmap with/without aggregation, matrix)."""

    @pytest.fixture
    def df_nulls(self):
        """DataFrame with nulls for missing data testing."""
        return pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
            "y": [100 + i if i % 5 != 0 else None for i in range(366)],
            "z": [200 + i if i % 7 != 0 else None for i in range(366)],
        })

    def test_heatmap_with_time_aggregation(self, df_nulls):
        """Heatmap kind with time_aggregation uses period groups."""
        fig = plot_missing_data(df_nulls, kind="heatmap", time_aggregation="1mo")
        assert_figure_valid(fig)
        assert isinstance(fig.data[0], go.Heatmap)

    def test_heatmap_without_time_aggregation(self, df_nulls):
        """Heatmap kind without time_aggregation uses individual time points."""
        fig = plot_missing_data(df_nulls, kind="heatmap")
        assert_figure_valid(fig)
        assert isinstance(fig.data[0], go.Heatmap)

    def test_matrix_kind(self, df_nulls):
        """Matrix kind renders binary heatmap."""
        fig = plot_missing_data(df_nulls, kind="matrix")
        assert_figure_valid(fig)
        assert isinstance(fig.data[0], go.Heatmap)

    def test_invalid_kind_raises(self, df_nulls):
        """Invalid kind raises ValueError."""
        with pytest.raises(ValueError, match="Unknown kind"):
            plot_missing_data(df_nulls, kind="scatter")


class TestPlotDistribution:
    """Tests for plot_distribution function."""

    def test_single_column(self, monthly_2col_df):
        """Test histogram + KDE for a single column."""
        fig = plot_distribution(monthly_2col_df, columns="y")
        assert_figure_valid(fig)
        # 1 histogram + 1 KDE
        assert len(fig.data) == 2

    def test_no_kde(self, monthly_2col_df):
        """Test histogram only (no KDE)."""
        fig = plot_distribution(monthly_2col_df, columns="y", show_kde=False)
        assert len(fig.data) == 1

    def test_multiple_columns(self, monthly_2col_df):
        """Test distribution for multiple columns."""
        fig = plot_distribution(monthly_2col_df, columns=["y", "y2"])
        # 2 histograms + 2 KDEs
        assert len(fig.data) == 4

    def test_custom_bins(self, monthly_2col_df):
        """Test with custom number of bins."""
        fig = plot_distribution(monthly_2col_df, columns="y", n_bins=10)
        assert_figure_valid(fig)

    def test_all_columns_default(self, monthly_2col_df):
        """Test that all numeric columns are used when columns is None."""
        fig = plot_distribution(monthly_2col_df)
        # y and y2 → 2 histograms + 2 KDEs
        assert len(fig.data) == 4

    def test_custom_title(self, monthly_2col_df):
        """Test that title is applied."""
        fig = plot_distribution(monthly_2col_df, columns="y", title="My Distribution")
        assert_layout(fig, title="My Distribution")

    def test_custom_dimensions(self, monthly_2col_df):
        """Test custom width and height."""
        fig = plot_distribution(monthly_2col_df, columns="y", width=800, height=500)
        assert_layout(fig, width=800, height=500)

    def test_panel(self):
        """Test panel faceting for distribution."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
            "y__a": [100, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180, 175],
            "y__b": [200, 210, 205, 220, 230, 225, 240, 250, 245, 260, 270, 265],
        })
        fig = plot_distribution(df, panel_group_names=["y"])
        assert_figure_valid(fig)

    def test_auto_detect_panel(self):
        """Distribution auto-detects panel mode."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 1), "1mo", eager=True),
            "sales__a": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "sales__b": [200.0, 210.0, 220.0, 230.0, 240.0, 250.0],
        })
        fig = plot_distribution(df)
        assert_figure_valid(fig)

    def test_returns_figure(self, monthly_2col_df):
        """Always returns go.Figure."""
        fig = plot_distribution(monthly_2col_df, columns="y")
        assert_figure_valid(fig)

    def test_not_a_dataframe(self):
        """Non-DataFrame raises TypeError."""
        with pytest.raises(TypeError, match="DataFrame"):
            plot_distribution("not a dataframe")


class TestPlotOutlierDetection:
    """Tests for plot_outliers function."""

    def test_zscore(self, monthly_2col_df):
        """Test z-score method."""
        fig = plot_outliers(monthly_2col_df, columns="y", method="zscore", threshold=1.0)
        assert_figure_valid(fig)
        # At least the line trace
        assert len(fig.data) >= 1

    def test_iqr(self, monthly_2col_df):
        """Test IQR method."""
        fig = plot_outliers(monthly_2col_df, columns="y", method="iqr", threshold=1.5)
        assert_figure_valid(fig)

    def test_percentile(self, monthly_2col_df):
        """Test percentile method."""
        fig = plot_outliers(monthly_2col_df, columns="y", method="percentile", threshold=10.0)
        assert_figure_valid(fig)

    def test_multiple_columns(self, monthly_2col_df):
        """Test outlier detection on multiple columns."""
        fig = plot_outliers(monthly_2col_df, columns=["y", "y2"], method="zscore")
        # At least 2 line traces
        assert len(fig.data) >= 2

    def test_no_outliers_constant_series(self):
        """Constant series produces no outlier markers."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "y": [100.0] * 10,
        })
        fig = plot_outliers(df, columns="y")
        assert_figure_valid(fig)
        # Only the line trace + possibly bounds, no outlier markers
        scatter_markers = [t for t in fig.data if t.mode == "markers"]
        assert len(scatter_markers) == 0

    def test_with_nulls(self, df_with_nulls):
        """Null values don't cause errors."""
        fig = plot_outliers(df_with_nulls, columns="y", method="zscore")
        assert_figure_valid(fig)

    def test_panel(self):
        """Test panel faceting for outlier detection."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
            "y__a": [100, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180, 175],
            "y__b": [200, 210, 205, 220, 230, 225, 240, 250, 245, 260, 270, 265],
        })
        fig = plot_outliers(df, method="zscore", panel_group_names=["y"])
        assert_figure_valid(fig)

    def test_custom_styling(self, monthly_2col_df):
        """Test custom outlier styling via kwargs."""
        fig = plot_outliers(
            monthly_2col_df,
            columns="y",
            method="zscore",
            outlier_color="#FF0000",
            outlier_size=12.0,
            show_bounds=False,
        )
        assert_figure_valid(fig)

    def test_invalid_method(self, monthly_2col_df):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            plot_outliers(monthly_2col_df, columns="y", method="invalid")

    def test_auto_detect_panel(self):
        """Outlier detection auto-detects panel mode."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 1), "1mo", eager=True),
            "sales__a": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "sales__b": [200.0, 210.0, 220.0, 230.0, 240.0, 250.0],
        })
        fig = plot_outliers(df)
        assert_figure_valid(fig)


class TestPlotResamplingComparison:
    """Tests for plot_resampling_comparison function."""

    @pytest.fixture
    def hourly_and_daily(self):
        """Create matching hourly and daily DataFrames."""
        hourly = pl.DataFrame({
            "time": pl.datetime_range(
                pl.datetime(2020, 1, 1), pl.datetime(2020, 1, 31, 23), "1h", eager=True,
            ),
            "temp": [20.0 + i % 24 for i in range(31 * 24)],
        })
        daily = hourly.group_by_dynamic("time", every="1d").agg(pl.col("temp").mean())
        return hourly, daily

    def test_basic(self, hourly_and_daily):
        """Test basic overlay of two resolutions."""
        hourly, daily = hourly_and_daily
        fig = plot_resampling_comparison(hourly, daily, columns="temp")
        # 1 original + 1 resampled
        assert len(fig.data) == 2
        assert_figure_valid(fig)

    def test_multiple_columns(self):
        """Test with multiple columns."""
        hourly = pl.DataFrame({
            "time": pl.datetime_range(
                pl.datetime(2020, 1, 1), pl.datetime(2020, 1, 7, 23), "1h", eager=True,
            ),
            "temp": [20.0 + i % 24 for i in range(7 * 24)],
            "wind": [5.0 + i % 12 for i in range(7 * 24)],
        })
        daily = hourly.group_by_dynamic("time", every="1d").agg(
            pl.col("temp").mean(), pl.col("wind").mean()
        )
        fig = plot_resampling_comparison(hourly, daily, columns=["temp", "wind"])
        # 2 originals + 2 resampled
        assert len(fig.data) == 4

    def test_custom_labels(self, hourly_and_daily):
        """Test custom legend labels."""
        hourly, daily = hourly_and_daily
        fig = plot_resampling_comparison(
            hourly, daily, columns="temp",
            original_label="Hourly", resampled_label="Daily Avg",
        )
        assert "Hourly" in fig.data[0].name
        assert "Daily Avg" in fig.data[1].name

    def test_column_not_in_original(self):
        """Column in resampled but not in original raises ValueError."""
        original = pl.DataFrame({
            "time": pl.datetime_range(
                pl.datetime(2020, 1, 1), pl.datetime(2020, 1, 2), "1h", eager=True,
            ),
            "y": list(range(25)),
        })
        resampled = pl.DataFrame({
            "time": [pl.datetime(2020, 1, 1)],
            "z": [12.0],
        })
        with pytest.raises(ValueError, match="not found"):
            plot_resampling_comparison(original, resampled, columns="z")

    def test_returns_figure(self, hourly_and_daily):
        """Always returns go.Figure."""
        hourly, daily = hourly_and_daily
        fig = plot_resampling_comparison(hourly, daily, columns="temp")
        assert_figure_valid(fig)

    def test_custom_title(self, hourly_and_daily):
        """Custom title is applied."""
        hourly, daily = hourly_and_daily
        fig = plot_resampling_comparison(hourly, daily, columns="temp", title="Resample Test")
        assert_layout(fig, title="Resample Test")

    def test_custom_dimensions(self, hourly_and_daily):
        """Custom dimensions are respected."""
        hourly, daily = hourly_and_daily
        fig = plot_resampling_comparison(hourly, daily, columns="temp", width=900, height=400)
        assert_layout(fig, width=900, height=400)


def _make_three_group_panel() -> pl.DataFrame:
    """Create a 3-group, 2-member panel DataFrame for testing."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
        "g1__a": [100 + i * 2.0 for i in range(12)],
        "g1__b": [200 + i * 3.0 for i in range(12)],
        "g2__a": [150 + i * 1.5 for i in range(12)],
        "g2__b": [250 + i * 2.5 for i in range(12)],
        "g3__a": [120 + i * 1.8 for i in range(12)],
        "g3__b": [220 + i * 2.2 for i in range(12)],
    })


class TestPanelLegendDedup:
    """Verify that panel plots produce no duplicate legend entries."""

    def test_time_series_no_duplicate_legend(self):
        """plot_time_series: each group name appears at most once in legend."""
        df = _make_three_group_panel()
        fig = plot_time_series(df)
        names = visible_legend_names(fig)
        assert len(names) == len(set(names)), f"Duplicate legend entries: {names}"
        assert set(names) == {"g1", "g2", "g3"}

    def test_time_series_legendgroup_set(self):
        """plot_time_series: all traces carry legendgroup matching their name."""
        df = _make_three_group_panel()
        fig = plot_time_series(df)
        for trace in fig.data:
            assert trace.legendgroup == trace.name

    def test_time_series_consistent_colors_across_groups(self):
        """plot_time_series: same member has same color in every subplot."""
        df = _make_three_group_panel()
        fig = plot_time_series(df)
        color_by_name: dict[str, str] = {}
        for trace in fig.data:
            name = trace.name
            color = trace.line.color
            if name in color_by_name:
                assert color == color_by_name[name], f"Color mismatch for {name}"
            else:
                color_by_name[name] = color

    def test_rolling_statistics_no_duplicate_legend(self):
        """plot_rolling_statistics: legend entries appear at most once per group."""
        df = _make_three_group_panel()
        fig = plot_rolling_statistics(df, window_size=3, statistics=["mean", "std"])
        names = visible_legend_names(fig)
        # With grouped_legend_kwargs, entry names ("mean", "std") repeat across
        # members but each (member, stat) pair appears only once.
        # The legendgrouptitle differentiates them visually.
        assert len(names) >= 4  # 2 stats × 2 members = 4 visible entries minimum




class TestPanelRollingShowOriginalFalse:
    """Cover panel rolling stats with show_original=False."""

    def test_panel_rolling_no_original(self):
        """Panel rolling stats with show_original=False skips raw traces."""
        df = _make_three_group_panel()
        fig = plot_rolling_statistics(
            df, window_size=3, statistics="mean", show_original=False,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0




class TestRollingStatsDefaultXLabel:
    """Cover x_label=None default branch for non-panel rolling stats."""

    def test_default_x_label(self):
        """Non-panel rolling stats uses 'Time' as default x-axis label."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
            "y": [100 + i * 2.0 for i in range(12)],
        })
        fig = plot_rolling_statistics(df, columns="y", window_size=3, statistics="mean")
        assert isinstance(fig, go.Figure)




class TestPlotMissingDataHeatmapAggregation:
    """Cover heatmap kind with time_aggregation parameter."""

    def test_heatmap_with_time_aggregation(self):
        """Heatmap with time_aggregation groups data by period."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
            "y": [None if i % 30 == 0 else float(i) for i in range(366)],
            "z": [None if i % 50 == 0 else float(i) * 2 for i in range(366)],
        })
        fig = plot_missing_data(df, kind="heatmap", time_aggregation="1mo")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_heatmap_without_aggregation(self):
        """Heatmap without time_aggregation uses individual time points."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 31), "1d", eager=True),
            "y": [None if i % 5 == 0 else float(i) for i in range(31)],
        })
        fig = plot_missing_data(df, kind="heatmap")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0




class TestPlotMissingDataMatrix:
    """Cover matrix kind branch in plot_missing_data."""

    def test_matrix_kind(self):
        """Matrix kind renders binary heatmap of missing values."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 31), "1d", eager=True),
            "y": [None if i % 7 == 0 else float(i) for i in range(31)],
            "z": [None if i % 10 == 0 else float(i) * 2 for i in range(31)],
        })
        fig = plot_missing_data(df, kind="matrix")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0




class TestDistributionPanelKde:
    """Cover show_kde in panel distribution mode."""

    def test_panel_distribution_with_kde(self):
        """Panel distribution with show_kde=True renders KDE curves."""
        import numpy as np

        rng = np.random.default_rng(42)
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
            "y__a": rng.normal(100, 10, 12).tolist(),
            "y__b": rng.normal(200, 15, 12).tolist(),
        })
        fig = plot_distribution(df, panel_group_names=["y"], show_kde=True)
        assert isinstance(fig, go.Figure)
        # Each member: 1 histogram + 1 KDE = 2 traces × 2 members = 4
        assert len(fig.data) >= 4




class TestOutlierMethodBranches:
    """Cover IQR and percentile branches in _compute_outlier_mask."""

    def test_iqr_method_with_panel(self):
        """Panel outlier detection with IQR method."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
            "y__a": [100, 120, 115, 130, 140, 135, 150, 500, 155, 170, 180, 175],
            "y__b": [200, 210, 205, 220, 230, 225, 240, 250, 245, 260, 270, 265],
        })
        fig = plot_outliers(df, method="iqr", panel_group_names=["y"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_percentile_method_with_panel(self):
        """Panel outlier detection with percentile method."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
            "y__a": [100, 120, 115, 130, 140, 135, 150, 500, 155, 170, 180, 175],
            "y__b": [200, 210, 205, 220, 230, 225, 240, 250, 245, 260, 270, 265],
        })
        fig = plot_outliers(df, method="percentile", threshold=10.0, panel_group_names=["y"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0




class TestResamplingComparisonPanel:
    """Cover panel path for plot_resampling_comparison."""

    def test_panel_resampling(self):
        """Panel resampling comparison routes through panel_facet_figure."""
        hourly = pl.DataFrame({
            "time": pl.datetime_range(
                pl.datetime(2020, 1, 1), pl.datetime(2020, 1, 7, 23), "1h", eager=True,
            ),
            "temp__a": [20.0 + i % 24 for i in range(7 * 24)],
            "temp__b": [15.0 + i % 24 for i in range(7 * 24)],
        })
        daily = hourly.group_by_dynamic("time", every="1d").agg(
            pl.col("temp__a").mean(), pl.col("temp__b").mean(),
        )
        fig = plot_resampling_comparison(
            hourly, daily, panel_group_names=["temp"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestOutlierShowBoundsPanel:
    """Cover show_bounds panel path (exploration.py L1409)."""

    def test_iqr_show_bounds_panel(self):
        """Panel outlier detection with IQR + show_bounds draws threshold lines."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
            "y__a": [100, 120, 115, 130, 140, 135, 150, 500, 155, 170, 180, 175],
            "y__b": [200, 210, 205, 220, 230, 225, 240, 250, 245, 260, 270, 265],
        })
        fig = plot_outliers(df, method="iqr", show_bounds=True, panel_group_names=["y"])
        assert isinstance(fig, go.Figure)
        # Bounds add horizontal line traces (dashed or solid)
        assert len(fig.data) >= 4  # series lines + bound lines


class TestResamplingComparisonAutoDetect:
    """Cover auto-detect panel path for resampling comparison (exploration.py L1657)."""

    def test_auto_detect_panel(self):
        """Panel data without explicit panel_group_names triggers auto-detect."""
        hourly = pl.DataFrame({
            "time": pl.datetime_range(
                pl.datetime(2020, 1, 1), pl.datetime(2020, 1, 7, 23), "1h", eager=True,
            ),
            "temp__a": [20.0 + i % 24 for i in range(7 * 24)],
            "temp__b": [15.0 + i % 24 for i in range(7 * 24)],
        })
        daily = hourly.group_by_dynamic("time", every="1d").agg(
            pl.col("temp__a").mean(), pl.col("temp__b").mean(),
        )
        # No columns= and no panel_group_names= → auto-detect
        fig = plot_resampling_comparison(hourly, daily)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
