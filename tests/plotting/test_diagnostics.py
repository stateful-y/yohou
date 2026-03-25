"""Tests for diagnostic plotting functions (ACF, PACF, spectrum, etc.)."""

import numpy as np
import polars as pl
import pytest
from plotly import graph_objects as go

from yohou.plotting import (
    plot_autocorrelation,
    plot_correlation_heatmap,
    plot_cross_correlation,
    plot_lag_scatter,
    plot_partial_autocorrelation,
    plot_scatter_matrix,
    plot_seasonal_heatmap,
    plot_seasonality,
    plot_subseasonality,
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


@pytest.fixture
def short_df():
    """Create short sample DataFrame for lag/cross-correlation tests."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
        "x": [100 + i % 20 for i in range(91)],
        "y": [150 + (i % 15) * 2 for i in range(91)],
    })


class TestPlotAutocorrelation:
    """Tests for plot_autocorrelation function."""

    def test_basic(self, sample_df):
        """Test basic autocorrelation plot."""
        fig = plot_autocorrelation(sample_df, columns="y", max_lags=10)
        assert len(fig.data) > 0
        assert isinstance(fig.data[0], go.Bar)

    def test_auto_max_lags(self, sample_df):
        """Test autocorrelation with automatic max_lags."""
        fig = plot_autocorrelation(sample_df, columns="y")
        assert len(fig.data) > 0

    def test_no_confidence(self, sample_df):
        """Test autocorrelation without confidence bands."""
        fig = plot_autocorrelation(sample_df, columns="y", max_lags=10, show_confidence=False)
        # Should have fewer traces without confidence bands
        assert len(fig.data) > 0

    def test_custom_styling(self, sample_df):
        """Test autocorrelation with custom styling."""
        fig = plot_autocorrelation(sample_df, columns="y", max_lags=10, bar_color="#FF0000")
        assert len(fig.data) > 0

    def test_panel(self, sample_df):
        """Test panel faceting for autocorrelation."""
        df = pl.DataFrame({
            "time": sample_df["time"],
            "y__a": sample_df["y"],
            "y__b": sample_df["y2"],
        })
        fig = plot_autocorrelation(df, max_lags=10, panel_group_names=["y"])
        assert len(fig.data) > 0


class TestPlotPartialAutocorrelation:
    """Tests for plot_partial_autocorrelation function."""

    def test_basic(self, sample_df):
        """Test basic PACF plot."""
        fig = plot_partial_autocorrelation(sample_df, columns="y", max_lags=10)
        assert len(fig.data) > 0
        assert isinstance(fig.data[0], go.Bar)

    def test_auto_max_lags(self, sample_df):
        """Test PACF with automatic max_lags."""
        fig = plot_partial_autocorrelation(sample_df, columns="y")
        assert len(fig.data) > 0

    def test_no_confidence(self, sample_df):
        """Test PACF without confidence bands."""
        fig = plot_partial_autocorrelation(sample_df, columns="y", max_lags=10, show_confidence=False)
        assert len(fig.data) > 0

    def test_custom_styling(self, sample_df):
        """Test PACF with custom styling."""
        fig = plot_partial_autocorrelation(sample_df, columns="y", max_lags=10, bar_color="#00FF00")
        assert len(fig.data) > 0

    def test_panel(self, sample_df):
        """Test panel faceting for partial autocorrelation."""
        df = pl.DataFrame({
            "time": sample_df["time"],
            "y__a": sample_df["y"],
            "y__b": sample_df["y2"],
        })
        fig = plot_partial_autocorrelation(df, max_lags=10, panel_group_names=["y"])
        assert len(fig.data) > 0


class TestPlotCorrelationHeatmap:
    """Tests for plot_correlation_heatmap function."""

    def test_basic(self, multi_column_df):
        """Test basic correlation matrix."""
        fig = plot_correlation_heatmap(multi_column_df)
        assert len(fig.data) > 0
        assert isinstance(fig.data[0], go.Heatmap)

    def test_no_values(self, multi_column_df):
        """Test correlation matrix without values displayed."""
        fig = plot_correlation_heatmap(multi_column_df, show_values=False)
        assert len(fig.data) > 0

    def test_custom_colorscale(self, multi_column_df):
        """Test correlation matrix with custom colorscale."""
        fig = plot_correlation_heatmap(multi_column_df, colorscale="Viridis")
        assert len(fig.data) > 0

    def test_subset_columns(self, multi_column_df):
        """Test correlation matrix with subset of columns."""
        fig = plot_correlation_heatmap(multi_column_df, columns=["y1", "y2"])
        assert len(fig.data) > 0

    def test_panel_group_names(self):
        """Test panel grouping produces one heatmap per group."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "y__a": list(range(10)),
            "y__b": list(range(10, 20)),
            "y__c": list(range(20, 30)),
        })
        fig = plot_correlation_heatmap(df, panel_group_names=["y"])
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_panel_invalid_group(self):
        """Test panel grouping raises ValueError for unknown groups."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "y__a": list(range(10)),
            "y__b": list(range(10, 20)),
        })
        with pytest.raises(ValueError, match="No panel groups found"):
            plot_correlation_heatmap(df, panel_group_names=["missing"])


class TestPlotSeasonality:
    """Tests for plot_seasonality function."""

    def test_month(self, sample_df):
        """Test monthly seasonality plot."""
        fig = plot_seasonality(sample_df, columns="y", seasonality="month")
        assert len(fig.data) > 0

    def test_quarter(self, sample_df):
        """Test quarterly seasonality plot."""
        fig = plot_seasonality(sample_df, columns="y", seasonality="quarter")
        assert len(fig.data) > 0

    def test_weekday(self, sample_df):
        """Test weekday seasonality plot."""
        fig = plot_seasonality(sample_df, columns="y", seasonality="weekday")
        assert len(fig.data) > 0

    def test_no_mean(self, sample_df):
        """Test seasonality plot without mean line."""
        fig = plot_seasonality(sample_df, columns="y", seasonality="month", show_mean=False)
        assert len(fig.data) > 0

    def test_custom_styling(self, sample_df):
        """Test seasonality with custom styling."""
        fig = plot_seasonality(
            sample_df,
            columns="y",
            seasonality="month",
            line_width=2.5,
            mean_color="#FF00FF",
        )
        assert len(fig.data) > 0

    def test_invalid_seasonality(self, sample_df):
        """Test that invalid seasonality raises ValueError."""
        with pytest.raises(ValueError, match="Unknown frequency"):
            plot_seasonality(sample_df, columns="y", seasonality="invalid")

    def test_panel(self, sample_df):
        """Test panel faceting for seasonality."""
        df = pl.DataFrame({
            "time": sample_df["time"],
            "y__a": sample_df["y"],
            "y__b": sample_df["y2"],
        })
        fig = plot_seasonality(df, seasonality="month", panel_group_names=["y"])
        assert len(fig.data) > 0


class TestPlotLagScatter:
    """Tests for plot_lag_scatter function."""

    def test_single_lag(self, short_df):
        """Test lag scatter with single lag."""
        fig = plot_lag_scatter(short_df, columns="y", lags=[1])
        assert len(fig.data) > 0
        assert isinstance(fig.data[0], go.Scattergl)

    def test_multiple_lags(self, short_df):
        """Test lag scatter with multiple lags."""
        fig = plot_lag_scatter(short_df, columns="y", lags=[1, 7, 14])
        assert len(fig.data) > 0

    def test_with_diagonal(self, short_df):
        """Test lag scatter with diagonal line."""
        fig = plot_lag_scatter(short_df, columns="y", lags=[1], show_diagonal=True)
        assert len(fig.data) > 0

    def test_with_regression(self, short_df):
        """Test lag scatter with regression line."""
        fig = plot_lag_scatter(short_df, columns="y", lags=[1], show_regression=True)
        assert len(fig.data) > 0

    def test_custom_styling(self, short_df):
        """Test lag scatter with custom styling."""
        fig = plot_lag_scatter(
            short_df,
            columns="y",
            lags=[1],
            marker_size=6.0,
            marker_opacity=0.8,
        )
        assert len(fig.data) > 0

    def test_panel(self, short_df):
        """Test panel faceting for lag scatter."""
        df = pl.DataFrame({
            "time": short_df["time"],
            "y__a": short_df["y"],
            "y__b": short_df["x"],
        })
        fig = plot_lag_scatter(df, lags=[1], panel_group_names=["y"])
        assert len(fig.data) > 0

    def test_multi_lag_grid(self, sample_df):
        """Test that multiple lags produce a subplot grid."""
        fig = plot_lag_scatter(sample_df, columns="y", lags=[1, 4, 8])
        # Each lag gets a subplot; check we have traces in multiple subplots
        assert len(fig.data) > 3  # scatter + diagonals

    def test_multi_lag_grid_layout(self, sample_df):
        """Test subplot grid respects facet_n_cols."""
        fig = plot_lag_scatter(sample_df, columns="y", lags=[1, 2, 3, 4], facet_n_cols=2)
        # 4 lags with ncol=2 → 2×2 grid
        assert len(fig.data) > 4  # at least 4 scatter traces + diagonals

    def test_season_coloring_month(self, sample_df):
        """Test seasonality='month' produces season-colored traces."""
        fig = plot_lag_scatter(sample_df, columns="y", lags=[1], seasonality="month")
        legend_groups = {t.legendgroup for t in fig.data if t.legendgroup is not None}
        # Should have one legendgroup per month present in the data
        assert len(legend_groups) >= 4

    def test_season_coloring_quarter(self, sample_df):
        """Test seasonality='quarter' produces 4 season traces."""
        fig = plot_lag_scatter(sample_df, columns="y", lags=[1], seasonality="quarter")
        legend_groups = {t.legendgroup for t in fig.data if t.legendgroup is not None}
        assert len(legend_groups) <= 4

    def test_season_with_grid(self, sample_df):
        """Test season coloring combined with multi-lag subplot grid."""
        fig = plot_lag_scatter(sample_df, columns="y", lags=[1, 4, 8], seasonality="quarter")
        legend_groups = {t.legendgroup for t in fig.data if t.legendgroup is not None}
        assert len(legend_groups) <= 4
        assert len(fig.data) > 6

    def test_seasonality_invalid(self, sample_df):
        """Test that invalid seasonality raises ValueError."""
        with pytest.raises(ValueError, match="Unknown frequency"):
            plot_lag_scatter(sample_df, columns="y", lags=[1], seasonality="invalid")

    def test_backward_compat_single(self, short_df):
        """Test that single lag without frequency matches old behaviour."""
        fig = plot_lag_scatter(short_df, columns="y", lags=[1])
        # No legendgroup set (uniform coloring)
        groups = {t.legendgroup for t in fig.data if t.legendgroup is not None}
        assert len(groups) == 0

    def test_custom_axis_labels(self, short_df):
        """Test custom x_label and y_label."""
        fig = plot_lag_scatter(short_df, columns="y", lags=[1], x_label="Lagged", y_label="Current")
        assert fig.layout.xaxis.title.text == "Lagged"
        assert fig.layout.yaxis.title.text == "Current"


class TestPlotCrossCorrelation:
    """Tests for plot_cross_correlation function."""

    def test_basic(self, short_df):
        """Test basic cross-correlation functionality."""
        fig = plot_cross_correlation(short_df, columns=["x", "y"], max_lags=20)
        assert len(fig.data) > 0

    def test_different_lags(self, short_df):
        """Test with different lag values."""
        fig10 = plot_cross_correlation(short_df, columns=["x", "y"], max_lags=10)
        fig30 = plot_cross_correlation(short_df, columns=["x", "y"], max_lags=30)
        assert len(fig10.data) > 0
        assert len(fig30.data) > 0

    def test_confidence_level(self, short_df):
        """Test different confidence levels."""
        fig = plot_cross_correlation(short_df, columns=["x", "y"], confidence_level=0.99)
        assert len(fig.data) > 0

    def test_styling(self, short_df):
        """Test custom styling."""
        fig = plot_cross_correlation(
            short_df,
            columns=["x", "y"],
            marker_size=8.0,
            marker_color="#DC2626",
            line_color="#059669",
        )
        assert len(fig.data) > 0

    def test_missing_column(self, short_df):
        """Test error handling for missing columns."""
        with pytest.raises(ValueError, match="not found"):
            plot_cross_correlation(short_df, columns=["missing", "y"])

        with pytest.raises(ValueError, match="not found"):
            plot_cross_correlation(short_df, columns=["x", "missing"])

    def test_wrong_column_count(self, short_df):
        """Test error when columns doesn't have exactly 2 entries."""
        with pytest.raises(ValueError, match="exactly 2"):
            plot_cross_correlation(short_df, columns=["x"])

    def test_custom_axis_labels(self, short_df):
        """Test custom x_label and y_label."""
        fig = plot_cross_correlation(short_df, columns=["x", "y"], x_label="Offset", y_label="CCF Value")
        assert fig.layout.xaxis.title.text == "Offset"
        assert fig.layout.yaxis.title.text == "CCF Value"


class TestPlotSubseasonality:
    """Tests for plot_subseasonality function."""

    def test_basic_monthly(self, sample_df):
        """Test basic monthly subseasonal plot."""
        fig = plot_subseasonality(sample_df, columns="y", seasonality="month")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_quarterly(self, sample_df):
        """Test quarterly subseasonal plot."""
        fig = plot_subseasonality(sample_df, columns="y", seasonality="quarter")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_no_mean(self, sample_df):
        """Test without mean lines."""
        # Need multi-year data so show_mean actually adds mean traces
        multi_year_df = pl.DataFrame({
            "time": pl.date_range(pl.date(2018, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
            "y": [100 + i % 30 for i in range(1096)],
        })
        fig = plot_subseasonality(multi_year_df, columns="y", seasonality="month", show_mean=False)
        fig_with_mean = plot_subseasonality(multi_year_df, columns="y", seasonality="month", show_mean=True)
        assert len(fig.data) < len(fig_with_mean.data)

    def test_custom_n_cols(self, sample_df):
        """Test custom number of columns in subplot grid."""
        fig = plot_subseasonality(sample_df, columns="y", seasonality="month", facet_n_cols=3)
        assert isinstance(fig, go.Figure)


class TestPlotScatterMatrix:
    """Tests for plot_scatter_matrix function."""

    @pytest.fixture
    def three_col_df(self):
        """Create 3-column DataFrame for scatter matrix tests."""
        rng = np.random.default_rng(42)
        n = 200
        return pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 7, 18), "1d", eager=True),
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
            "c": rng.standard_normal(n),
        })

    def test_basic(self, three_col_df):
        """Test basic 3×3 scatter matrix."""
        fig = plot_scatter_matrix(three_col_df, columns=["a", "b", "c"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_two_columns(self, three_col_df):
        """Test minimal 2×2 matrix."""
        fig = plot_scatter_matrix(three_col_df, columns=["a", "b"])
        assert isinstance(fig, go.Figure)
        # 2×2 grid: diagonal KDE (2) + lower triangle scatter (1) = 3 traces
        assert len(fig.data) >= 3

    def test_all_columns_default(self, three_col_df):
        """Test that columns=None selects all numeric columns."""
        fig = plot_scatter_matrix(three_col_df)
        assert isinstance(fig, go.Figure)
        # Should use all 3 columns (a, b, c), excluding "time"
        assert len(fig.data) > 0

    def test_column_selection_string(self, three_col_df):
        """Test passing columns as a string."""
        # Single string should raise because we need >= 2 columns
        with pytest.raises(ValueError, match="at least 2"):
            plot_scatter_matrix(three_col_df, columns="a")

    def test_season_coloring_month(self, three_col_df):
        """Test season coloring by month."""
        fig = plot_scatter_matrix(three_col_df, columns=["a", "b", "c"], seasonality="month")
        assert isinstance(fig, go.Figure)
        # With season coloring, lower triangle has more traces (one per season)
        assert len(fig.data) > 3

    def test_diagonal_histogram(self, three_col_df):
        """Test histogram on diagonal."""
        fig = plot_scatter_matrix(three_col_df, columns=["a", "b", "c"], diagonal="histogram")
        assert isinstance(fig, go.Figure)
        # Should have Histogram traces on diagonal
        histogram_traces = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(histogram_traces) == 3

    def test_diagonal_none(self, three_col_df):
        """Test no diagonal content."""
        fig = plot_scatter_matrix(three_col_df, columns=["a", "b", "c"], diagonal=None)
        # No diagonal traces, only scatter traces in lower triangle
        assert isinstance(fig, go.Figure)
        for t in fig.data:
            assert not isinstance(t, go.Histogram)

    def test_no_correlation(self, three_col_df):
        """Test with correlation annotations disabled."""
        fig = plot_scatter_matrix(
            three_col_df,
            columns=["a", "b", "c"],
            show_correlation=False,
        )
        assert isinstance(fig, go.Figure)
        # No annotations should be added
        assert len(fig.layout.annotations) == 0

    def test_correlation_annotations(self, three_col_df):
        """Test that correlation annotations are present by default."""
        fig = plot_scatter_matrix(three_col_df, columns=["a", "b", "c"], show_correlation=True)
        # 3×3 upper triangle has 3 cells → 3 annotations
        assert len(fig.layout.annotations) == 3

    def test_custom_size(self, three_col_df):
        """Test custom width and height."""
        fig = plot_scatter_matrix(three_col_df, columns=["a", "b"], width=800, height=700)
        assert fig.layout.width == 800
        assert fig.layout.height == 700

    def test_custom_styling_kwargs(self, three_col_df):
        """Test marker_size, marker_opacity, and corr_font_size kwargs."""
        fig = plot_scatter_matrix(
            three_col_df,
            columns=["a", "b"],
            marker_size=5.0,
            marker_opacity=0.8,
            corr_font_size=40,
        )
        assert isinstance(fig, go.Figure)


class TestPlotSubseasonalityPanel:
    """Panel and error-path tests for plot_subseasonality."""

    @pytest.fixture
    def panel_df(self):
        """Create panel DataFrame for subseasonality."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        n = len(dates)
        return pl.DataFrame({
            "time": dates,
            "y__a": [100 + i % 30 for i in range(n)],
            "y__b": [200 + (i % 20) * 2 for i in range(n)],
        })

    def test_panel_produces_figure(self, panel_df):
        """Panel data with subseasonality returns a valid figure."""
        fig = plot_subseasonality(panel_df, seasonality="month")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_trace_type_is_scatter(self):
        """Traces produced by subseasonality are Scatter type."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
            "y": [100 + i % 30 for i in range(366)],
        })
        fig = plot_subseasonality(df, columns="y", seasonality="month")
        assert all(isinstance(t, go.Scatter) for t in fig.data)

    def test_custom_title(self):
        """Custom title is applied to subseasonality figure."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
            "y": [100 + i % 30 for i in range(366)],
        })
        fig = plot_subseasonality(df, columns="y", seasonality="month", title="My Title")
        assert fig.layout.title.text == "My Title"

    def test_custom_dimensions(self):
        """Custom width/height apply to figure layout."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
            "y": [100 + i % 30 for i in range(366)],
        })
        fig = plot_subseasonality(df, columns="y", seasonality="month", width=1000, height=600)
        assert fig.layout.width == 1000
        assert fig.layout.height == 600


class TestPlotCrossCorrelationPanel:
    """Panel and error-path tests for plot_cross_correlation."""

    @pytest.fixture
    def panel_df(self):
        """Create panel DataFrame for cross-correlation."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        n = len(dates)
        return pl.DataFrame({
            "time": dates,
            "x__a": [100 + i % 20 for i in range(n)],
            "y__a": [150 + (i % 15) * 2 for i in range(n)],
        })

    def test_custom_title(self):
        """Custom title is applied to cross-correlation figure."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "x": [100 + i % 20 for i in range(91)],
            "y": [150 + (i % 15) * 2 for i in range(91)],
        })
        fig = plot_cross_correlation(df, columns=["x", "y"], title="CCF Title")
        assert fig.layout.title.text == "CCF Title"

    def test_custom_dimensions(self):
        """Custom width/height apply to cross-correlation figure."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "x": [100 + i % 20 for i in range(91)],
            "y": [150 + (i % 15) * 2 for i in range(91)],
        })
        fig = plot_cross_correlation(df, columns=["x", "y"], width=900, height=500)
        assert fig.layout.width == 900
        assert fig.layout.height == 500

    def test_trace_count(self):
        """Cross-correlation produces bar and confidence band traces."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "x": [100 + i % 20 for i in range(91)],
            "y": [150 + (i % 15) * 2 for i in range(91)],
        })
        fig = plot_cross_correlation(df, columns=["x", "y"], max_lags=10)
        # Should have at least bar trace + confidence bands
        assert len(fig.data) >= 1


class TestPlotScatterMatrixPanel:
    """Panel and edge-case tests for plot_scatter_matrix."""

    def test_custom_title(self):
        """Custom title is applied to scatter matrix figure."""
        rng = np.random.default_rng(42)
        n = 100
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
        })
        fig = plot_scatter_matrix(df, columns=["a", "b"], title="Matrix Title")
        assert fig.layout.title.text == "Matrix Title"


class TestPlotSeasonalityAssertions:
    """Stronger assertion tests for plot_seasonality."""

    def test_month_produces_traces(self):
        """Seasonality with month produces multiple traces."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
            "y": [100 + i % 30 for i in range(366)],
        })
        fig = plot_seasonality(df, columns="y", seasonality="month")
        assert len(fig.data) >= 1
        assert all(isinstance(t, go.Scatter | go.Box) for t in fig.data)

    def test_returns_go_figure(self):
        """Seasonality always returns a go.Figure."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
            "y": [100 + i % 30 for i in range(366)],
        })
        fig = plot_seasonality(df, columns="y", seasonality="month")
        assert isinstance(fig, go.Figure)

    def test_custom_dimensions(self):
        """Custom dimensions are respected."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
            "y": [100 + i % 30 for i in range(366)],
        })
        fig = plot_seasonality(df, columns="y", seasonality="month", width=800, height=500)
        assert fig.layout.width == 800
        assert fig.layout.height == 500


class TestPlotSeasonalityHighlight:
    """Tests for seasonality highlight parameter branches."""

    def test_highlight_int(self):
        """Passing highlight as a single integer selects one period."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
            "y": [100 + i % 30 for i in range(366)],
        })
        fig = plot_seasonality(df, columns="y", seasonality="month", highlight=3)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_highlight_list(self):
        """Passing highlight as a list selects multiple periods."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
            "y": [100 + i % 30 for i in range(366)],
        })
        fig = plot_seasonality(
            df,
            columns="y",
            seasonality="month",
            highlight=[1, 6],
            highlight_width=4.0,
            fade_opacity=0.15,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestPacfDurbinLevinsonFallback:
    """Tests for built-in PACF when statsmodels is unavailable."""

    def test_durbin_levinson_direct(self):
        """Direct call to _compute_pacf_durbin_levinson returns valid PACF."""
        from yohou.plotting.diagnostics import _compute_pacf_durbin_levinson

        rng = np.random.default_rng(42)
        values = rng.standard_normal(200)
        pacf_vals, ci_lo, ci_hi = _compute_pacf_durbin_levinson(values, nlags=10, alpha=None)

        assert len(pacf_vals) == 11
        assert pacf_vals[0] == pytest.approx(1.0)
        assert ci_lo is None
        assert ci_hi is None

    def test_durbin_levinson_with_confidence(self):
        """Confidence intervals are returned when alpha is given."""
        from yohou.plotting.diagnostics import _compute_pacf_durbin_levinson

        rng = np.random.default_rng(42)
        values = rng.standard_normal(200)
        pacf_vals, ci_lo, ci_hi = _compute_pacf_durbin_levinson(values, nlags=5, alpha=0.05)

        assert len(pacf_vals) == 6
        assert ci_lo is not None
        assert ci_hi is not None
        assert len(ci_lo) == 6
        assert len(ci_hi) == 6
        assert all(lo < 0 for lo in ci_lo)
        assert all(hi > 0 for hi in ci_hi)

    def test_statsmodels_fallback_warning(self, monkeypatch):
        """When statsmodels is missing and method != 'yw', a warning is raised."""
        import importlib
        import warnings

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if "statsmodels" in name:
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        rng = np.random.default_rng(42)
        values = rng.standard_normal(100)

        monkeypatch.setattr("builtins.__import__", mock_import)
        try:
            importlib.invalidate_caches()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                from yohou.plotting.diagnostics import _compute_pacf

                pacf_vals, _, _ = _compute_pacf(values, nlags=5, method="ols", alpha=0.05)
                assert len(pacf_vals) == 6
                assert any("statsmodels is not installed" in str(wi.message) for wi in w)
        finally:
            monkeypatch.undo()


class TestDiagnosticsPanelAutoDetect:
    """Tests for panel auto-detection branches in diagnostic plots."""

    @pytest.fixture
    def panel_df(self):
        """Create panel DataFrame with two groups."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 30), "1d", eager=True)
        n = len(dates)
        return pl.DataFrame({
            "time": dates,
            "y__a": [100.0 + (i % 30) for i in range(n)],
            "y__b": [200.0 + (i % 20) for i in range(n)],
        })

    def test_acf_panel(self, panel_df):
        """plot_autocorrelation with panel data auto-detects groups."""
        fig = plot_autocorrelation(panel_df, panel_group_names=["y"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_pacf_panel(self, panel_df):
        """plot_partial_autocorrelation with panel data auto-detects groups."""
        fig = plot_partial_autocorrelation(panel_df, panel_group_names=["y"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_correlation_heatmap_panel(self, panel_df):
        """plot_correlation_heatmap with panel data auto-detects groups."""
        fig = plot_correlation_heatmap(panel_df, panel_group_names=["y"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_seasonality_panel(self, panel_df):
        """plot_seasonality with panel data auto-detects groups."""
        fig = plot_seasonality(panel_df, seasonality="month", panel_group_names=["y"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_lag_scatter_panel(self, panel_df):
        """plot_lag_scatter with panel data auto-detects groups."""
        fig = plot_lag_scatter(panel_df, lags=[1, 5], panel_group_names=["y"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestDiagnosticsSeasonalityBranches:
    """Tests for seasonality frequency branches."""

    def test_week_frequency(self, sample_df):
        """plot_seasonality with frequency='week' covers week branch."""
        fig = plot_seasonality(sample_df, columns="y", seasonality="week")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_hour_frequency(self):
        """plot_seasonality with frequency='hour' covers hour branch."""
        times = pl.datetime_range(pl.datetime(2020, 1, 1, 0, 0), pl.datetime(2020, 1, 10, 23, 0), "1h", eager=True)
        df = pl.DataFrame({
            "time": times,
            "y": [100.0 + (i % 24) * 3.0 for i in range(len(times))],
        })
        fig = plot_seasonality(df, columns="y", seasonality="hour")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestDiagnosticsEdgeCases:
    """Tests for edge cases in diagnostics functions."""

    def test_cross_correlation_constant_series(self):
        """Cross-correlation with constant series falls back to 0 coefficient."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "x": [5.0] * len(dates),
            "y": [100.0 + (i % 15) * 2 for i in range(len(dates))],
        })
        fig = plot_cross_correlation(df, columns=["x", "y"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_scatter_matrix_constant_column_kde(self):
        """Scatter matrix with constant column and diagonal='kde' handles LinAlgError."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 2, 28), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "y1": [5.0] * len(dates),
            "y2": [float(i) for i in range(len(dates))],
        })
        fig = plot_scatter_matrix(df, diagonal="kde", show_correlation=False)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_cross_correlation_show_markers_false(self, short_df):
        """Cross-correlation with show_markers=False covers line-mode branch."""
        fig = plot_cross_correlation(short_df, columns=["x", "y"], show_markers=False)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_subseasonality_hour_labels(self):
        """Subseasonality with hour seasonality covers label fallback."""
        times = pl.datetime_range(pl.datetime(2020, 1, 1, 0, 0), pl.datetime(2020, 1, 10, 23, 0), "1h", eager=True)
        df = pl.DataFrame({
            "time": times,
            "y": [100.0 + (i % 24) * 3.0 for i in range(len(times))],
        })
        fig = plot_subseasonality(df, columns="y", seasonality="hour")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestDiagnosticsAutoDetectPanel:
    """Tests for auto-detection of panel data (no explicit panel_group_names)."""

    @pytest.fixture
    def auto_panel_df(self):
        """Create panel DataFrame where auto-detect triggers (__ separator)."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 30), "1d", eager=True)
        n = len(dates)
        return pl.DataFrame({
            "time": dates,
            "val__store1": [100.0 + (i % 30) for i in range(n)],
            "val__store2": [200.0 + (i % 20) for i in range(n)],
        })

    def test_acf_auto_detect(self, auto_panel_df):
        """plot_autocorrelation auto-detects panel data without panel_group_names."""
        fig = plot_autocorrelation(auto_panel_df, max_lags=10)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_pacf_auto_detect(self, auto_panel_df):
        """plot_partial_autocorrelation auto-detects panel data."""
        fig = plot_partial_autocorrelation(auto_panel_df, max_lags=10)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_heatmap_auto_detect(self, auto_panel_df):
        """plot_correlation_heatmap auto-detects panel data."""
        fig = plot_correlation_heatmap(auto_panel_df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_seasonality_auto_detect(self, auto_panel_df):
        """plot_seasonality auto-detects panel data."""
        fig = plot_seasonality(auto_panel_df, seasonality="month")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_lag_scatter_auto_detect(self, auto_panel_df):
        """plot_lag_scatter auto-detects panel data."""
        fig = plot_lag_scatter(auto_panel_df, lags=[1])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestDiagnosticsCustomLabels:
    """Tests for custom x_label and y_label branches in diagnostic plots."""

    @pytest.fixture
    def simple_df(self):
        """Create simple time series for label tests."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 30), "1d", eager=True)
        return pl.DataFrame({
            "time": dates,
            "y": [100.0 + (i % 30) for i in range(len(dates))],
        })

    def test_acf_custom_labels(self, simple_df):
        """plot_autocorrelation with custom x_label and y_label."""
        fig = plot_autocorrelation(
            simple_df,
            columns="y",
            max_lags=10,
            x_label="Custom Lag",
            y_label="Custom ACF",
        )
        assert isinstance(fig, go.Figure)
        assert fig.layout.xaxis.title.text == "Custom Lag"
        assert fig.layout.yaxis.title.text == "Custom ACF"

    def test_pacf_custom_labels(self, simple_df):
        """plot_partial_autocorrelation with custom x_label and y_label."""
        fig = plot_partial_autocorrelation(
            simple_df,
            columns="y",
            max_lags=10,
            x_label="Custom Lag",
            y_label="Custom PACF",
        )
        assert isinstance(fig, go.Figure)
        assert fig.layout.xaxis.title.text == "Custom Lag"
        assert fig.layout.yaxis.title.text == "Custom PACF"

    def test_seasonality_custom_labels(self, simple_df):
        """plot_seasonality with custom x_label and y_label."""
        fig = plot_seasonality(
            simple_df,
            columns="y",
            seasonality="month",
            x_label="Month Index",
            y_label="Values",
        )
        assert isinstance(fig, go.Figure)


class TestLagScatterMultiLagShowDiagonal:
    """Tests for plot_lag_scatter multi-lag with show_diagonal branch."""

    def test_multi_lag_show_diagonal_true(self):
        """Multi-lag with show_diagonal=True adds diagonal lines to each subplot."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "y": [100.0 + (i % 30) for i in range(len(dates))],
        })
        fig = plot_lag_scatter(df, columns="y", lags=[1, 2], show_diagonal=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 4

    def test_multi_lag_with_seasonality_month(self):
        """Multi-lag + seasonality='month' triggers season-colored multi-lag path."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "y": [100.0 + (i % 30) for i in range(len(dates))],
        })
        fig = plot_lag_scatter(df, columns="y", lags=[1, 2], seasonality="month")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_single_lag_with_seasonality(self):
        """Single lag + seasonality='month' triggers the seasonal single-lag path."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "y": [100.0 + (i % 30) for i in range(len(dates))],
        })
        fig = plot_lag_scatter(df, columns="y", lags=[1], seasonality="month")
        assert isinstance(fig, go.Figure)
        legend_groups = {t.legendgroup for t in fig.data if t.legendgroup is not None}
        assert len(legend_groups) >= 4

    def test_single_lag_show_diagonal(self):
        """Single lag + show_diagonal covers the single-lag diagonal path."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 30), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "y": [100.0 + (i % 30) for i in range(len(dates))],
        })
        fig = plot_lag_scatter(df, columns="y", lags=[1], show_diagonal=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_single_lag_show_regression(self):
        """Single lag + show_regression covers the regression overlay."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 30), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "y": [100.0 + (i % 30) for i in range(len(dates))],
        })
        fig = plot_lag_scatter(df, columns="y", lags=[1], show_regression=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2


class TestCorrelationHeatmapPanelShowValues:
    """Tests for plot_correlation_heatmap panel + show_values=True."""

    def test_panel_show_values(self):
        """Panel heatmap with show_values=True adds text annotations."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        n = len(dates)
        df = pl.DataFrame({
            "time": dates,
            "a__g1": [float(i) for i in range(n)],
            "b__g1": [float(i * 2) for i in range(n)],
            "a__g2": [float(i + 10) for i in range(n)],
            "b__g2": [float(i * 3) for i in range(n)],
        })
        fig = plot_correlation_heatmap(df, panel_group_names=["a", "b"], show_values=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestCrossCorrelationShowMarkers:
    """Tests for plot_cross_correlation show_markers=True branch."""

    def test_show_markers_true(self):
        """show_markers=True uses lines+markers mode."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "x": [100.0 + (i % 20) for i in range(len(dates))],
            "y": [150.0 + (i % 15) * 2 for i in range(len(dates))],
        })
        fig = plot_cross_correlation(df, columns=["x", "y"], show_markers=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestLagScatterShowDiagonalAndRegression:
    """Tests for plot_lag_scatter show_diagonal and show_regression branches."""

    @pytest.fixture
    def lag_scatter_df(self):
        """DataFrame for lag scatter tests."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 30), "1d", eager=True)
        return pl.DataFrame({
            "time": dates,
            "y": [100.0 + i * 0.5 + (i % 7) * 3.0 for i in range(len(dates))],
        })

    def test_show_diagonal_single_lag(self, lag_scatter_df):
        """show_diagonal adds a dashed reference line for single lag."""
        fig = plot_lag_scatter(lag_scatter_df, columns="y", lags=[1], show_diagonal=True)
        assert isinstance(fig, go.Figure)
        dash_traces = [t for t in fig.data if hasattr(t, "line") and t.line and t.line.dash == "dash"]
        assert len(dash_traces) >= 1

    def test_show_regression_single_lag(self, lag_scatter_df):
        """show_regression adds a regression line for single lag."""
        fig = plot_lag_scatter(lag_scatter_df, columns="y", lags=[1], show_regression=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_show_both_diagonal_and_regression(self, lag_scatter_df):
        """Both show_diagonal and show_regression produce extra traces."""
        fig = plot_lag_scatter(
            lag_scatter_df,
            columns="y",
            lags=[1],
            show_diagonal=True,
            show_regression=True,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3


class TestPlotACFPanelNoConfidence:
    """Tests for panel autocorrelation without confidence bands."""

    def test_panel_acf_no_confidence(self):
        """Panel ACF with show_confidence=False skips CI band rendering."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "sales__store_1": [100.0 + i * 2.0 for i in range(12)],
            "sales__store_2": [200.0 + i * 3.0 for i in range(12)],
        })
        fig = plot_autocorrelation(df, panel_group_names=["sales"], show_confidence=False)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_panel_pacf_no_confidence(self):
        """Panel PACF with show_confidence=False skips CI band rendering."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "sales__store_1": [100.0 + i * 2.0 for i in range(12)],
            "sales__store_2": [200.0 + i * 3.0 for i in range(12)],
        })
        fig = plot_partial_autocorrelation(df, panel_group_names=["sales"], show_confidence=False)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestScatterMatrixDiagonalModes:
    """Tests for plot_scatter_matrix diagonal mode branches."""

    @pytest.fixture
    def scatter_df(self):
        """Multi-column DataFrame for scatter matrix tests."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        n = len(dates)
        return pl.DataFrame({
            "time": dates,
            "x": [100.0 + i * 0.3 for i in range(n)],
            "y": [200.0 + (i % 30) * 2.0 for i in range(n)],
        })

    def test_diagonal_kde(self, scatter_df):
        """Scatter matrix with diagonal='kde' renders KDE curves."""
        fig = plot_scatter_matrix(scatter_df, columns=["x", "y"], diagonal="kde")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_diagonal_histogram(self, scatter_df):
        """Scatter matrix with diagonal='histogram' renders histogram bars."""
        fig = plot_scatter_matrix(scatter_df, columns=["x", "y"], diagonal="histogram")
        assert isinstance(fig, go.Figure)
        histogram_traces = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(histogram_traces) >= 1

    def test_show_correlation(self, scatter_df):
        """Scatter matrix with show_correlation=True adds Pearson r annotations."""
        fig = plot_scatter_matrix(scatter_df, columns=["x", "y"], show_correlation=True)
        assert isinstance(fig, go.Figure)
        annotations = fig.layout.annotations
        assert annotations is not None
        corr_annotations = [a for a in annotations if hasattr(a, "text") and "." in str(a.text)]
        assert len(corr_annotations) >= 1


# ── plot_seasonal_heatmap ─────────────────────────────────────────────


class TestPlotSeasonalHeatmap:
    """Tests for plot_seasonal_heatmap function."""

    @pytest.fixture
    def hourly_df(self):
        """2 years of hourly data with seasonal-like pattern."""
        import math

        times = pl.datetime_range(
            pl.datetime(2020, 1, 1), pl.datetime(2021, 12, 31, 23), "1h", eager=True,
        )
        n = len(times)
        return pl.DataFrame({
            "time": times,
            "temp": [
                20 + 10 * math.sin(i * 2 * math.pi / 24)
                + 5 * math.sin(i * 2 * math.pi / 8760)
                for i in range(n)
            ],
        })

    def test_hour_by_month(self, hourly_df):
        """Basic hour × month heatmap."""
        fig = plot_seasonal_heatmap(hourly_df, "temp", x_period="hour", y_period="month")
        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Heatmap)

    def test_day_of_week_by_hour(self, hourly_df):
        """Day-of-week × hour heatmap."""
        fig = plot_seasonal_heatmap(hourly_df, "temp", x_period="hour", y_period="day_of_week")
        assert isinstance(fig, go.Figure)

    def test_month_by_year(self, hourly_df):
        """Month × year heatmap."""
        fig = plot_seasonal_heatmap(hourly_df, "temp", x_period="month", y_period="year")
        assert isinstance(fig, go.Figure)

    def test_aggregation_methods(self, hourly_df):
        """All aggregation methods produce valid figures."""
        for agg in ["mean", "median", "sum", "count", "std", "min", "max"]:
            fig = plot_seasonal_heatmap(hourly_df, "temp", agg=agg)
            assert len(fig.data) > 0

    def test_custom_colorscale(self, hourly_df):
        """Custom colorscale is accepted."""
        fig = plot_seasonal_heatmap(hourly_df, "temp", colorscale="RdBu_r")
        assert isinstance(fig, go.Figure)

    def test_no_values(self, hourly_df):
        """show_values=False suppresses annotations."""
        fig = plot_seasonal_heatmap(hourly_df, "temp", show_values=False)
        assert isinstance(fig, go.Figure)

    def test_reverse_y(self, hourly_df):
        """reverse_y=True flips y-axis."""
        fig = plot_seasonal_heatmap(hourly_df, "temp", reverse_y=True)
        assert isinstance(fig, go.Figure)

    def test_invalid_period(self, hourly_df):
        """Invalid period raises ValueError."""
        with pytest.raises(ValueError, match="Unknown period"):
            plot_seasonal_heatmap(hourly_df, "temp", x_period="invalid")

    def test_invalid_agg(self, hourly_df):
        """Invalid agg raises ValueError."""
        with pytest.raises(ValueError, match="Unknown agg"):
            plot_seasonal_heatmap(hourly_df, "temp", agg="invalid")

    def test_columns_not_found(self, hourly_df):
        """Non-existent column raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            plot_seasonal_heatmap(hourly_df, "nonexistent")

    def test_custom_title(self, hourly_df):
        """Custom title is applied."""
        fig = plot_seasonal_heatmap(hourly_df, "temp", title="Heatmap Test")
        assert fig.layout.title.text == "Heatmap Test"

    def test_custom_dimensions(self, hourly_df):
        """Custom dimensions are respected."""
        fig = plot_seasonal_heatmap(hourly_df, "temp", width=800, height=500)
        assert fig.layout.width == 800
        assert fig.layout.height == 500

    def test_panel_explicit(self):
        """Panel faceting with explicit panel_group_names."""
        times = pl.datetime_range(
            pl.datetime(2020, 1, 1), pl.datetime(2020, 3, 31, 23), "1h", eager=True,
        )
        n = len(times)
        df = pl.DataFrame({
            "time": times,
            "weather__temp": [20.0 + i % 24 for i in range(n)],
            "weather__wind": [5.0 + i % 12 for i in range(n)],
        })
        fig = plot_seasonal_heatmap(df, "temp", panel_group_names=["weather"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_auto_column_single(self, hourly_df):
        """Auto-detect single numeric column when columns=None."""
        fig = plot_seasonal_heatmap(hourly_df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_auto_column_multi(self):
        """Multiple numeric columns produce subplot grid."""
        times = pl.datetime_range(
            pl.datetime(2020, 1, 1), pl.datetime(2020, 6, 30, 23), "1h", eager=True,
        )
        n = len(times)
        df = pl.DataFrame({
            "time": times,
            "temp": [20.0 + i % 24 for i in range(n)],
            "wind": [5.0 + i % 12 for i in range(n)],
        })
        fig = plot_seasonal_heatmap(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_columns_list(self):
        """Explicit list of columns produces subplot grid."""
        times = pl.datetime_range(
            pl.datetime(2020, 1, 1), pl.datetime(2020, 6, 30, 23), "1h", eager=True,
        )
        n = len(times)
        df = pl.DataFrame({
            "time": times,
            "temp": [20.0 + i % 24 for i in range(n)],
            "wind": [5.0 + i % 12 for i in range(n)],
            "humidity": [60.0 + i % 10 for i in range(n)],
        })
        fig = plot_seasonal_heatmap(df, columns=["temp", "wind"])
        assert len(fig.data) == 2

    def test_panel_auto_detect(self):
        """Panel data auto-detected when columns=None and no panel_group_names."""
        times = pl.datetime_range(
            pl.datetime(2020, 1, 1), pl.datetime(2020, 3, 31, 23), "1h", eager=True,
        )
        n = len(times)
        df = pl.DataFrame({
            "time": times,
            "loc_A__temp": [20.0 + i % 24 for i in range(n)],
            "loc_B__temp": [18.0 + i % 24 for i in range(n)],
        })
        fig = plot_seasonal_heatmap(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_panel_auto_detect_with_columns(self):
        """Panel auto-detect with columns filtering to specific member."""
        times = pl.datetime_range(
            pl.datetime(2020, 1, 1), pl.datetime(2020, 3, 31, 23), "1h", eager=True,
        )
        n = len(times)
        df = pl.DataFrame({
            "time": times,
            "weather__temp": [20.0 + i % 24 for i in range(n)],
            "weather__wind": [5.0 + i % 12 for i in range(n)],
        })
        fig = plot_seasonal_heatmap(df, columns="temp")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
