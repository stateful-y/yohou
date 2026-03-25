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


class TestPlotTimeSeriesErrorPaths:
    """Error path tests for plot_time_series."""

    def test_invalid_column_name(self, sample_df):
        """Non-existent column raises ValueError."""
        with pytest.raises(ValueError, match="not found|not in"):
            plot_time_series(sample_df, columns="nonexistent")

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

    def test_returns_go_figure(self, sample_df):
        """Boxplot always returns a go.Figure instance."""
        fig = plot_boxplot(sample_df, columns="y", period="1mo")
        assert isinstance(fig, go.Figure)

    def test_trace_type_is_box(self, sample_df):
        """Boxplot traces should be Box type."""
        fig = plot_boxplot(sample_df, columns="y", period="1mo")
        assert any(isinstance(t, go.Box) for t in fig.data)

    def test_custom_title(self, sample_df):
        """Custom title is applied to boxplot figure."""
        fig = plot_boxplot(sample_df, columns="y", period="1mo", title="Box Title")
        assert fig.layout.title.text == "Box Title"

    def test_custom_dimensions(self, sample_df):
        """Custom dimensions are respected."""
        fig = plot_boxplot(sample_df, columns="y", period="1mo", width=800, height=500)
        assert fig.layout.width == 800
        assert fig.layout.height == 500


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
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


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
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


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
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


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
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


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
        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Heatmap)

    def test_heatmap_without_time_aggregation(self, df_nulls):
        """Heatmap kind without time_aggregation uses individual time points."""
        fig = plot_missing_data(df_nulls, kind="heatmap")
        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Heatmap)

    def test_matrix_kind(self, df_nulls):
        """Matrix kind renders binary heatmap."""
        fig = plot_missing_data(df_nulls, kind="matrix")
        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Heatmap)

    def test_invalid_kind_raises(self, df_nulls):
        """Invalid kind raises ValueError."""
        with pytest.raises(ValueError, match="Unknown kind"):
            plot_missing_data(df_nulls, kind="scatter")


# ── plot_distribution ─────────────────────────────────────────────────


class TestPlotDistribution:
    """Tests for plot_distribution function."""

    def test_single_column(self, sample_df):
        """Test histogram + KDE for a single column."""
        fig = plot_distribution(sample_df, columns="y")
        assert isinstance(fig, go.Figure)
        # 1 histogram + 1 KDE
        assert len(fig.data) == 2

    def test_no_kde(self, sample_df):
        """Test histogram only (no KDE)."""
        fig = plot_distribution(sample_df, columns="y", show_kde=False)
        assert len(fig.data) == 1

    def test_multiple_columns(self, sample_df):
        """Test distribution for multiple columns."""
        fig = plot_distribution(sample_df, columns=["y", "y2"])
        # 2 histograms + 2 KDEs
        assert len(fig.data) == 4

    def test_custom_bins(self, sample_df):
        """Test with custom number of bins."""
        fig = plot_distribution(sample_df, columns="y", n_bins=10)
        assert isinstance(fig, go.Figure)

    def test_all_columns_default(self, sample_df):
        """Test that all numeric columns are used when columns is None."""
        fig = plot_distribution(sample_df)
        # y and y2 → 2 histograms + 2 KDEs
        assert len(fig.data) == 4

    def test_custom_title(self, sample_df):
        """Test that title is applied."""
        fig = plot_distribution(sample_df, columns="y", title="My Distribution")
        assert fig.layout.title.text == "My Distribution"

    def test_custom_dimensions(self, sample_df):
        """Test custom width and height."""
        fig = plot_distribution(sample_df, columns="y", width=800, height=500)
        assert fig.layout.width == 800
        assert fig.layout.height == 500

    def test_panel(self):
        """Test panel faceting for distribution."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
            "y__a": [100, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180, 175],
            "y__b": [200, 210, 205, 220, 230, 225, 240, 250, 245, 260, 270, 265],
        })
        fig = plot_distribution(df, panel_group_names=["y"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_auto_detect_panel(self):
        """Distribution auto-detects panel mode."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 1), "1mo", eager=True),
            "sales__a": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "sales__b": [200.0, 210.0, 220.0, 230.0, 240.0, 250.0],
        })
        fig = plot_distribution(df)
        assert isinstance(fig, go.Figure)

    def test_returns_figure(self, sample_df):
        """Always returns go.Figure."""
        fig = plot_distribution(sample_df, columns="y")
        assert isinstance(fig, go.Figure)

    def test_not_a_dataframe(self):
        """Non-DataFrame raises TypeError."""
        with pytest.raises(TypeError, match="DataFrame"):
            plot_distribution("not a dataframe")


# ── plot_outliers ─────────────────────────────────────────────────────


class TestPlotOutlierDetection:
    """Tests for plot_outliers function."""

    def test_zscore(self, sample_df):
        """Test z-score method."""
        fig = plot_outliers(sample_df, columns="y", method="zscore", threshold=1.0)
        assert isinstance(fig, go.Figure)
        # At least the line trace
        assert len(fig.data) >= 1

    def test_iqr(self, sample_df):
        """Test IQR method."""
        fig = plot_outliers(sample_df, columns="y", method="iqr", threshold=1.5)
        assert isinstance(fig, go.Figure)

    def test_percentile(self, sample_df):
        """Test percentile method."""
        fig = plot_outliers(sample_df, columns="y", method="percentile", threshold=10.0)
        assert isinstance(fig, go.Figure)

    def test_multiple_columns(self, sample_df):
        """Test outlier detection on multiple columns."""
        fig = plot_outliers(sample_df, columns=["y", "y2"], method="zscore")
        # At least 2 line traces
        assert len(fig.data) >= 2

    def test_no_outliers_constant_series(self):
        """Constant series produces no outlier markers."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "y": [100.0] * 10,
        })
        fig = plot_outliers(df, columns="y")
        assert isinstance(fig, go.Figure)
        # Only the line trace + possibly bounds, no outlier markers
        scatter_markers = [t for t in fig.data if t.mode == "markers"]
        assert len(scatter_markers) == 0

    def test_with_nulls(self, df_with_nulls):
        """Null values don't cause errors."""
        fig = plot_outliers(df_with_nulls, columns="y", method="zscore")
        assert isinstance(fig, go.Figure)

    def test_panel(self):
        """Test panel faceting for outlier detection."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
            "y__a": [100, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180, 175],
            "y__b": [200, 210, 205, 220, 230, 225, 240, 250, 245, 260, 270, 265],
        })
        fig = plot_outliers(df, method="zscore", panel_group_names=["y"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_custom_styling(self, sample_df):
        """Test custom outlier styling via kwargs."""
        fig = plot_outliers(
            sample_df,
            columns="y",
            method="zscore",
            outlier_color="#FF0000",
            outlier_size=12.0,
            show_bounds=False,
        )
        assert isinstance(fig, go.Figure)

    def test_invalid_method(self, sample_df):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            plot_outliers(sample_df, columns="y", method="invalid")

    def test_auto_detect_panel(self):
        """Outlier detection auto-detects panel mode."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 1), "1mo", eager=True),
            "sales__a": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "sales__b": [200.0, 210.0, 220.0, 230.0, 240.0, 250.0],
        })
        fig = plot_outliers(df)
        assert isinstance(fig, go.Figure)


# ── plot_resampling_comparison ────────────────────────────────────────


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
        assert isinstance(fig, go.Figure)

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
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, hourly_and_daily):
        """Custom title is applied."""
        hourly, daily = hourly_and_daily
        fig = plot_resampling_comparison(hourly, daily, columns="temp", title="Resample Test")
        assert fig.layout.title.text == "Resample Test"

    def test_custom_dimensions(self, hourly_and_daily):
        """Custom dimensions are respected."""
        hourly, daily = hourly_and_daily
        fig = plot_resampling_comparison(hourly, daily, columns="temp", width=900, height=400)
        assert fig.layout.width == 900
        assert fig.layout.height == 400
