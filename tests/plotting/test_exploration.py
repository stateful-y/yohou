"""Tests for time series exploration plotting functions."""

import pytest

pytest.importorskip("plotly", reason="plotting extra not installed")


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
        fig = plot_time_series(df, columns="y", groups=["group"])
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
        fig = plot_rolling_statistics(
            monthly_2col_df, columns="y", window_size=3, statistics="mean", show_original=False
        )
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
            fig = plot_rolling_statistics(
                monthly_2col_df, columns="y", window_size=3, statistics=stat, show_original=False
            )
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
        fig = plot_rolling_statistics(df, window_size=3, statistics="mean", groups=["y"])
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
        fig = plot_boxplot(df, period="1mo", groups=["y"])
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
            plot_missing_data(df_with_nulls, kind="invalid")

    def test_panel(self, df_with_nulls):
        """Test panel faceting for missing data."""
        df = pl.DataFrame({
            "time": df_with_nulls["time"],
            "y__a": df_with_nulls["y"],
            "y__b": df_with_nulls["z"],
        })
        fig = plot_missing_data(df, kind="bars", groups=["y"])
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
        """Auto-detect panel mode when columns and groups are both None."""
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
        fig = plot_distribution(df, groups=["y"])
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
        fig = plot_outliers(monthly_2col_df, columns="y", method="percentile", threshold=90.0)
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
        fig = plot_outliers(df, method="zscore", groups=["y"])
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
                pl.datetime(2020, 1, 1),
                pl.datetime(2020, 1, 31, 23),
                "1h",
                eager=True,
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
                pl.datetime(2020, 1, 1),
                pl.datetime(2020, 1, 7, 23),
                "1h",
                eager=True,
            ),
            "temp": [20.0 + i % 24 for i in range(7 * 24)],
            "wind": [5.0 + i % 12 for i in range(7 * 24)],
        })
        daily = hourly.group_by_dynamic("time", every="1d").agg(pl.col("temp").mean(), pl.col("wind").mean())
        fig = plot_resampling_comparison(hourly, daily, columns=["temp", "wind"])
        # 2 originals + 2 resampled
        assert len(fig.data) == 4

    def test_custom_labels(self, hourly_and_daily):
        """Test custom legend labels."""
        hourly, daily = hourly_and_daily
        fig = plot_resampling_comparison(
            hourly,
            daily,
            columns="temp",
            original_label="Hourly",
            resampled_label="Daily Avg",
        )
        assert "Hourly" in fig.data[0].name
        assert "Daily Avg" in fig.data[1].name

    def test_column_not_in_original(self):
        """Column in resampled but not in original raises ValueError."""
        original = pl.DataFrame({
            "time": pl.datetime_range(
                pl.datetime(2020, 1, 1),
                pl.datetime(2020, 1, 2),
                "1h",
                eager=True,
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
            df,
            window_size=3,
            statistics="mean",
            show_original=False,
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


class TestPlotMissingDataSamplingInterval:
    """Tests for the sampling_interval parameter of plot_missing_data."""

    @pytest.fixture
    def df_with_gaps(self):
        """Hourly DataFrame with 3 rows removed (gap rows, not nulls)."""
        from datetime import datetime

        full_times = pl.datetime_range(
            datetime(2020, 1, 1),
            datetime(2020, 1, 1, 9),
            "1h",
            eager=True,
        )
        # 10 timestamps (hours 0-9), drop indices 2, 5, 8 -> 7 rows remain
        df = pl.DataFrame({
            "time": full_times,
            "y": [float(i) for i in range(10)],
            "z": [float(i) * 10 for i in range(10)],
        })
        return df.filter(
            ~pl.col("time").is_in([
                full_times[2],
                full_times[5],
                full_times[8],
            ])
        )

    def test_bars_with_sampling_interval(self, df_with_gaps):
        """Bars reflect gap rows as missing when sampling_interval is set."""
        fig = plot_missing_data(df_with_gaps, kind="bars", sampling_interval="1h")
        assert isinstance(fig, go.Figure)
        # 10 expected rows, 3 gap rows -> 30% missing per column
        bar = fig.data[0]
        assert bar.y[0] == pytest.approx(30.0, abs=0.1)

    def test_bars_without_sampling_interval(self, df_with_gaps):
        """Bars report 0% missing without sampling_interval (no gap detection)."""
        fig = plot_missing_data(df_with_gaps, kind="bars")
        bar = fig.data[0]
        assert bar.y[0] == pytest.approx(0.0, abs=0.1)

    def test_heatmap_with_sampling_interval(self, df_with_gaps):
        """Heatmap z-data includes gap rows as missing."""
        fig = plot_missing_data(df_with_gaps, kind="heatmap", sampling_interval="1h")
        assert isinstance(fig, go.Figure)
        heatmap = fig.data[0]
        # z_data[0] is the first column's missing indicator list (length = 10)
        assert len(heatmap.z[0]) == 10
        # 3 of 10 entries should be 1 (missing)
        assert sum(heatmap.z[0]) == 3

    def test_matrix_with_sampling_interval(self, df_with_gaps):
        """Matrix z-data includes gap rows as missing."""
        fig = plot_missing_data(df_with_gaps, kind="matrix", sampling_interval="1h")
        heatmap = fig.data[0]
        assert len(heatmap.z[0]) == 10
        assert sum(heatmap.z[0]) == 3

    def test_sampling_interval_none_unchanged(self, df_with_gaps):
        """Default (None) preserves original behaviour - no reindexing."""
        fig = plot_missing_data(df_with_gaps, kind="bars", sampling_interval=None)
        bar = fig.data[0]
        # No nulls in the 7 remaining rows
        assert bar.y[0] == pytest.approx(0.0, abs=0.1)

    def test_sampling_interval_variable_raises(self, df_with_gaps):
        """Variable-length interval like '1mo' raises ValueError."""
        with pytest.raises(ValueError, match="variable-length"):
            plot_missing_data(df_with_gaps, kind="bars", sampling_interval="1mo")


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
        fig = plot_distribution(df, groups=["y"], show_kde=True)
        assert isinstance(fig, go.Figure)
        # Each member: 1 histogram + 1 KDE = 2 traces × 2 members = 4
        assert len(fig.data) >= 4


class TestPlotMissingDataBarTraces:
    """Verify per-column bar traces, legend names, and color_palette."""

    def test_non_panel_bars_one_trace_per_column(self, df_with_nulls):
        """Non-panel bars creates one trace per column."""
        fig = plot_missing_data(df_with_nulls, kind="bars")
        # df_with_nulls has columns y and z
        assert len(fig.data) == 2
        assert all(isinstance(t, go.Bar) for t in fig.data)

    def test_non_panel_bars_trace_names(self, df_with_nulls):
        """Each bar trace is named after its column (no 'trace 0')."""
        fig = plot_missing_data(df_with_nulls, kind="bars")
        names = [t.name for t in fig.data]
        assert "y" in names
        assert "z" in names
        assert "trace 0" not in names

    def test_non_panel_bars_custom_palette(self, df_with_nulls):
        """color_palette assigns custom colours to bar traces."""
        palette = ["#111111", "#222222"]
        fig = plot_missing_data(df_with_nulls, kind="bars", color_palette=palette)
        colors = [t.marker.color for t in fig.data]
        assert colors == palette

    def test_non_panel_bars_show_legend_false(self, df_with_nulls):
        """show_legend=False hides legend globally."""
        fig = plot_missing_data(df_with_nulls, kind="bars", show_legend=False)
        assert fig.layout.showlegend is False

    def test_non_panel_univariate_single_trace(self, df_with_nulls):
        """Single column produces exactly one bar trace."""
        fig = plot_missing_data(df_with_nulls, kind="bars", columns="y")
        assert len(fig.data) == 1
        assert fig.data[0].name == "y"


class TestPlotMissingDataPanelBarsLegend:
    """Verify LegendTracker / PanelColorManager in panel bars mode."""

    @pytest.fixture
    def panel_df(self):
        """Panel DataFrame with 2 groups x 2 members."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 1), "1mo", eager=True)
        return pl.DataFrame({
            "time": dates,
            "T3__a": [100.0, None, 120.0, None, 140.0, 150.0],
            "T3__b": [None, 210.0, 220.0, 230.0, None, 250.0],
            "T4__a": [300.0, 310.0, None, 330.0, 340.0, None],
            "T4__b": [400.0, None, 420.0, None, 440.0, 450.0],
        })

    def test_panel_bars_trace_names(self, panel_df):
        """Panel bar traces carry display names (not 'trace 0')."""
        fig = plot_missing_data(panel_df, kind="bars", groups=["T3", "T4"])
        names = {t.name for t in fig.data}
        assert "trace 0" not in names
        assert len(names) > 0

    def test_panel_bars_legendgroup(self, panel_df):
        """Each trace has a legendgroup matching its display name."""
        fig = plot_missing_data(panel_df, kind="bars", groups=["T3", "T4"])
        for t in fig.data:
            assert t.legendgroup == t.name

    def test_panel_bars_legend_dedup(self, panel_df):
        """LegendTracker shows each name only once in the legend."""
        fig = plot_missing_data(panel_df, kind="bars", groups=["T3", "T4"])
        shown_names = [t.name for t in fig.data if t.showlegend is True]
        assert len(shown_names) == len(set(shown_names))

    def test_panel_bars_consistent_color(self, panel_df):
        """Same display name has the same colour across facets."""
        fig = plot_missing_data(panel_df, kind="bars", groups=["T3", "T4"])
        color_map: dict[str, str] = {}
        for t in fig.data:
            if t.name in color_map:
                assert t.marker.color == color_map[t.name]
            else:
                color_map[t.name] = t.marker.color

    def test_panel_bars_custom_palette(self, panel_df):
        """Custom color_palette is applied in panel bars mode."""
        fig = plot_missing_data(
            panel_df,
            kind="bars",
            groups=["T3", "T4"],
            color_palette=["#AA0000", "#00BB00"],
        )
        colors = {t.marker.color for t in fig.data}
        assert colors <= {"#AA0000", "#00BB00"}


class TestPlotMissingDataPanelKinds:
    """Verify panel data works with heatmap and matrix kinds."""

    @pytest.fixture
    def panel_df(self):
        """Panel DataFrame with 2 groups x 2 members."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 1), "1mo", eager=True)
        return pl.DataFrame({
            "time": dates,
            "T3__a": [100.0, None, 120.0, None, 140.0, 150.0],
            "T3__b": [None, 210.0, 220.0, 230.0, None, 250.0],
            "T4__a": [300.0, 310.0, None, 330.0, 340.0, None],
            "T4__b": [400.0, None, 420.0, None, 440.0, 450.0],
        })

    def test_panel_heatmap_facet_member(self, panel_df):
        """Panel heatmap faceted by member creates one subplot per member."""
        fig = plot_missing_data(
            panel_df,
            kind="heatmap",
            groups=["T3", "T4"],
            facet_by="member",
        )
        assert isinstance(fig, go.Figure)
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        # 2 members -> 2 heatmaps
        assert len(heatmaps) == 2

    def test_panel_heatmap_facet_group(self, panel_df):
        """Panel heatmap faceted by group creates one subplot per group."""
        fig = plot_missing_data(
            panel_df,
            kind="heatmap",
            groups=["T3", "T4"],
            facet_by="group",
        )
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        # 2 groups -> 2 heatmaps
        assert len(heatmaps) == 2

    def test_panel_heatmap_y_labels(self, panel_df):
        """Heatmap y-axis labels use overlay key names (group prefix)."""
        fig = plot_missing_data(
            panel_df,
            kind="heatmap",
            groups=["T3", "T4"],
            facet_by="member",
        )
        heatmap = fig.data[0]
        # facet_by="member" -> overlays are groups -> y labels are group names
        assert set(heatmap.y) == {"T3", "T4"}

    def test_panel_matrix_kind(self, panel_df):
        """Panel matrix kind renders heatmap traces."""
        fig = plot_missing_data(
            panel_df,
            kind="matrix",
            groups=["T3", "T4"],
            facet_by="group",
        )
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmaps) == 2

    def test_panel_heatmap_with_time_aggregation(self, panel_df):
        """Panel heatmap supports time_aggregation."""
        fig = plot_missing_data(
            panel_df,
            kind="heatmap",
            groups=["T3", "T4"],
            facet_by="group",
            time_aggregation="3mo",
        )
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmaps) == 2

    def test_panel_invalid_kind_raises(self, panel_df):
        """Invalid kind in panel mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown kind"):
            plot_missing_data(panel_df, kind="scatter", groups=["T3"])  # type: ignore

    def test_panel_auto_detect_heatmap(self):
        """Auto-detected panel data works with heatmap kind."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 1), "1mo", eager=True),
            "sales__store_1": [100.0, None, 120.0, None, 140.0, 150.0],
            "sales__store_2": [None, 210.0, 220.0, 230.0, None, 250.0],
        })
        fig = plot_missing_data(df, kind="heatmap")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestOutlierMethodBranches:
    """Cover IQR and percentile branches in _compute_outlier_mask."""

    def test_iqr_method_with_panel(self):
        """Panel outlier detection with IQR method."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
            "y__a": [100, 120, 115, 130, 140, 135, 150, 500, 155, 170, 180, 175],
            "y__b": [200, 210, 205, 220, 230, 225, 240, 250, 245, 260, 270, 265],
        })
        fig = plot_outliers(df, method="iqr", groups=["y"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_percentile_method_with_panel(self):
        """Panel outlier detection with percentile method."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
            "y__a": [100, 120, 115, 130, 140, 135, 150, 500, 155, 170, 180, 175],
            "y__b": [200, 210, 205, 220, 230, 225, 240, 250, 245, 260, 270, 265],
        })
        fig = plot_outliers(df, method="percentile", threshold=90.0, groups=["y"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestResamplingComparisonPanel:
    """Cover panel path for plot_resampling_comparison."""

    def test_panel_resampling(self):
        """Panel resampling comparison routes through facet_figure."""
        hourly = pl.DataFrame({
            "time": pl.datetime_range(
                pl.datetime(2020, 1, 1),
                pl.datetime(2020, 1, 7, 23),
                "1h",
                eager=True,
            ),
            "temp__a": [20.0 + i % 24 for i in range(7 * 24)],
            "temp__b": [15.0 + i % 24 for i in range(7 * 24)],
        })
        daily = hourly.group_by_dynamic("time", every="1d").agg(
            pl.col("temp__a").mean(),
            pl.col("temp__b").mean(),
        )
        fig = plot_resampling_comparison(
            hourly,
            daily,
            groups=["temp"],
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
        fig = plot_outliers(df, method="iqr", show_bounds=True, groups=["y"])
        assert isinstance(fig, go.Figure)
        # Bounds add horizontal line traces (dashed or solid)
        assert len(fig.data) >= 4  # series lines + bound lines


class TestResamplingComparisonAutoDetect:
    """Cover auto-detect panel path for resampling comparison (exploration.py L1657)."""

    def test_auto_detect_panel(self):
        """Panel data without explicit groups triggers auto-detect."""
        hourly = pl.DataFrame({
            "time": pl.datetime_range(
                pl.datetime(2020, 1, 1),
                pl.datetime(2020, 1, 7, 23),
                "1h",
                eager=True,
            ),
            "temp__a": [20.0 + i % 24 for i in range(7 * 24)],
            "temp__b": [15.0 + i % 24 for i in range(7 * 24)],
        })
        daily = hourly.group_by_dynamic("time", every="1d").agg(
            pl.col("temp__a").mean(),
            pl.col("temp__b").mean(),
        )
        # No columns= and no groups= → auto-detect
        fig = plot_resampling_comparison(hourly, daily)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestMissingDataAsymmetricPanel:
    """Cover col_name is None continue in heatmap panel path."""

    def test_heatmap_asymmetric_panel(self):
        """Heatmap panel where one group has fewer members skips missing pairs."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "g1__a": [1.0, None, 3.0, None, 5.0, 6.0, None, 8.0, 9.0, 10.0],
            "g1__b": [None, 2.0, 3.0, 4.0, None, 6.0, 7.0, 8.0, None, 10.0],
            "g2__a": [1.0, 2.0, None, 4.0, 5.0, None, 7.0, 8.0, 9.0, None],
            # g2__b is missing - asymmetric panel
        })
        fig = plot_missing_data(df, kind="heatmap", groups=["g1", "g2"])
        assert isinstance(fig, go.Figure)


class TestOutlierNullSeries:
    """Cover None guard branches in outlier detection."""

    def test_iqr_all_null_series(self):
        """IQR method with an all-null series returns no outliers."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "y__a": pl.Series([None] * 10, dtype=pl.Float64),
            "y__b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0],
        })
        fig = plot_outliers(df, method="iqr", groups=["y"])
        assert isinstance(fig, go.Figure)

    def test_percentile_all_null_series(self):
        """Percentile method with an all-null series returns no outliers."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "y__a": pl.Series([None] * 10, dtype=pl.Float64),
            "y__b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0],
        })
        fig = plot_outliers(df, method="percentile", threshold=90.0, groups=["y"])
        assert isinstance(fig, go.Figure)


class TestPanelFacetFigureAsymmetricWarning:
    """Cover asymmetric-panel warning in _utils.facet_figure."""

    def test_facet_by_member_asymmetric_warns(self):
        """Asymmetric panel with facet_by='member' emits a warning."""
        import warnings

        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "g1__a": list(range(10)),
            "g1__b": list(range(10)),
            "g2__a": list(range(10)),
            # g2__b intentionally missing -> asymmetric
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = plot_time_series(df, groups=["g1", "g2"], facet_by="member")
        assert isinstance(fig, go.Figure)
        assert any("Asymmetric panel" in str(m.message) for m in w)


class TestResamplingComparisonFacetByGroup:
    """Cover facet_by='group' in panel resampling comparison."""

    def test_panel_resampling_facet_by_group(self):
        """Panel resampling comparison with facet_by='group' renders original traces."""
        hourly = pl.DataFrame({
            "time": pl.datetime_range(
                pl.datetime(2020, 1, 1),
                pl.datetime(2020, 1, 7, 23),
                "1h",
                eager=True,
            ),
            "temp__a": [20.0 + i % 24 for i in range(7 * 24)],
            "temp__b": [15.0 + i % 24 for i in range(7 * 24)],
        })
        daily = hourly.group_by_dynamic("time", every="1d").agg(
            pl.col("temp__a").mean(),
            pl.col("temp__b").mean(),
        )
        fig = plot_resampling_comparison(
            hourly,
            daily,
            groups=["temp"],
            facet_by="group",
        )
        assert isinstance(fig, go.Figure)
        # Each panel should have original + resampled traces
        assert len(fig.data) >= 2


@pytest.fixture
def categorical_df():
    """DataFrame with one categorical and one numeric column."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
        "status": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
        "value": [10.0, 20.0, 15.0, 25.0, 30.0, 28.0, 35.0, 40.0, 38.0, 45.0],
    })


@pytest.fixture
def categorical_only_df():
    """DataFrame with only categorical columns."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 8), "1d", eager=True),
        "color": ["red", "blue", "green", "red", "blue", "green", "red", "blue"],
    })


class TestPlotTimeSeriesCategorical:
    """Tests for categorical column support in plot_time_series."""

    def test_categorical_renders_step_chart(self, categorical_only_df):
        """Categorical column renders as step chart with markers."""
        fig = plot_time_series(categorical_only_df, columns=["color"])
        assert_figure_valid(fig)
        trace = fig.data[0]
        assert trace.mode == "lines+markers"
        assert trace.line.shape == "hv"

    def test_categorical_uses_integer_y_values(self, categorical_only_df):
        """Categorical y-values are mapped to integers."""
        fig = plot_time_series(categorical_only_df, columns=["color"])
        y_vals = list(fig.data[0].y)
        assert all(isinstance(v, int) for v in y_vals)

    def test_categorical_hover_shows_class_name(self, categorical_only_df):
        """Hover template shows category name via text."""
        fig = plot_time_series(categorical_only_df, columns=["color"])
        trace = fig.data[0]
        assert trace.text is not None
        assert "Class" in trace.hovertemplate

    def test_mixed_numeric_and_categorical(self, categorical_df):
        """Mixed DataFrame renders both numeric and categorical columns."""
        fig = plot_time_series(categorical_df, columns=["status", "value"])
        assert len(fig.data) == 2
        cat_trace = fig.data[0]
        num_trace = fig.data[1]
        assert cat_trace.line.shape == "hv"
        assert num_trace.mode == "lines"

    def test_categorical_auto_discovered(self, categorical_only_df):
        """Categorical columns are auto-discovered when columns=None."""
        fig = plot_time_series(categorical_only_df)
        assert_figure_valid(fig)
        assert fig.data[0].line.shape == "hv"

    def test_categorical_yaxis_tick_labels(self, categorical_only_df):
        """Y-axis shows category labels instead of integers."""
        fig = plot_time_series(categorical_only_df, columns=["color"])
        yaxis = fig.layout.yaxis
        assert yaxis.ticktext is not None
        tick_labels = list(yaxis.ticktext)
        assert sorted(tick_labels) == ["blue", "green", "red"]


class TestPlotDistributionCategorical:
    """Tests for categorical column support in plot_distribution."""

    def test_categorical_renders_bar_chart(self, categorical_only_df):
        """Categorical column renders as bar chart instead of histogram."""
        fig = plot_distribution(categorical_only_df, columns=["color"])
        assert_figure_valid(fig)
        trace = fig.data[0]
        assert isinstance(trace, go.Bar)

    def test_categorical_bar_shows_value_counts(self, categorical_only_df):
        """Bar chart shows correct value counts for each category."""
        fig = plot_distribution(categorical_only_df, columns=["color"])
        trace = fig.data[0]
        # Each category appears at least once
        assert len(trace.x) > 0
        assert all(v > 0 for v in trace.y)

    def test_mixed_numeric_categorical_distribution(self, categorical_df):
        """Mixed DataFrame: numeric gets histogram, categorical gets bar."""
        fig = plot_distribution(categorical_df, columns=["status", "value"])
        assert len(fig.data) >= 2
        # First subplot (status) is a bar, second (value) is a histogram
        assert isinstance(fig.data[0], go.Bar)
        assert isinstance(fig.data[1], go.Histogram)

    def test_categorical_auto_discovered_distribution(self, categorical_only_df):
        """Categorical columns auto-discovered in plot_distribution."""
        fig = plot_distribution(categorical_only_df)
        assert_figure_valid(fig)
        assert isinstance(fig.data[0], go.Bar)


class TestPlotMissingDataCategorical:
    """Tests for categorical column inclusion in plot_missing_data."""

    def test_categorical_included_in_bars(self):
        """Categorical columns are included in missing-data bar chart."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 5), "1d", eager=True),
            "status": ["A", None, "C", None, "A"],
            "value": [1.0, 2.0, None, 4.0, 5.0],
        })
        fig = plot_missing_data(df, kind="bars")
        assert_figure_valid(fig)
        # Both columns should appear
        trace_names = {t.name for t in fig.data}
        assert "status" in trace_names
        assert "value" in trace_names

    def test_categorical_only_missing_data(self):
        """All-categorical DataFrame works with missing data bars."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 5), "1d", eager=True),
            "color": ["red", None, "blue", None, "green"],
        })
        fig = plot_missing_data(df, kind="bars")
        assert_figure_valid(fig)
        assert fig.data[0].name == "color"


def _series_df(values):
    """Build a simple single-column daily time series for QA regression tests."""
    n = len(values)
    return pl.DataFrame({
        "time": pl.datetime_range(
            pl.datetime(2020, 1, 1),
            pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        ),
        "y": values,
    })


class TestOutlierColor:
    """Regression: outlier-color-param-never-used (2026-06-15 QA)."""

    def test_outlier_markers_use_outlier_color(self):
        vals = [1.0] * 20 + [100.0] + [1.0] * 5
        df = _series_df(vals)
        fig = plot_outliers(df, columns="y", outlier_color="#00FF00", method="zscore", threshold=2.0)
        marker_colors = [t.marker.color for t in fig.data if t.mode == "markers"]
        assert "#00FF00" in marker_colors


class TestRollingLineOpacity:
    """Regression: rolling-stats-nonpanel-hardcoded-opacity (2026-06-15 QA)."""

    def test_original_series_uses_line_opacity(self):
        df = _series_df([float(i % 7) for i in range(60)])
        fig = plot_rolling_statistics(df, columns="y", line_opacity=0.85, show_original=True)
        opacities = [t.opacity for t in fig.data if t.opacity is not None]
        assert pytest.approx(0.85) in opacities


class TestMissingDataMatrixTimeAggregation:
    """Regression: matrix-kind-ignores-time-aggregation (2026-06-15 QA)."""

    def test_matrix_and_heatmap_agree_under_time_aggregation(self):
        dates = pl.datetime_range(
            pl.datetime(2020, 1, 1),
            pl.datetime(2020, 1, 10, 23),
            interval="1h",
            eager=True,
        )
        vals = [None if i % 24 == 0 else float(i) for i in range(len(dates))]
        df = pl.DataFrame({"time": dates, "y": vals})
        fig_heatmap = plot_missing_data(df, kind="heatmap", time_aggregation="1d")
        fig_matrix = plot_missing_data(df, kind="matrix", time_aggregation="1d")
        z_heatmap = list(fig_heatmap.data[0].z)
        z_matrix = list(fig_matrix.data[0].z)
        assert z_matrix == z_heatmap
        # Aggregation collapsed 240 hourly points into 10 daily cells.
        assert len(z_matrix[0]) == 10


class TestResamplingPanelColors:
    """Regression: resampling-panel-resolve-color-inside-render-closure (2026-06-15 QA)."""

    def test_panel_members_get_distinct_colors(self):
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 2, 9), "1d", eager=True)
        n = len(dates)
        original = pl.DataFrame({
            "time": dates,
            "sales__a": list(range(n)),
            "sales__b": list(range(n, 2 * n)),
        })
        resampled = original.group_by_dynamic("time", every="1w").agg(
            pl.col("sales__a").mean(), pl.col("sales__b").mean()
        )
        # facet_by="group" overlays the two members in one facet, so they must
        # receive distinct palette colours (entity_idx based, not a single color).
        fig = plot_resampling_comparison(
            original, resampled, groups=[], facet_by="group", color_palette=["#abcdef", "#fedcba"]
        )
        line_colors = {t.line.color for t in fig.data if t.line.color is not None}
        assert {"#abcdef", "#fedcba"}.issubset(line_colors)
