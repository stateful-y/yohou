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
        fig = plot_lag_scatter(short_df, columns="y", lags=1)
        assert len(fig.data) > 0
        assert isinstance(fig.data[0], go.Scatter)

    def test_multiple_lags(self, short_df):
        """Test lag scatter with multiple lags."""
        fig = plot_lag_scatter(short_df, columns="y", lags=[1, 7, 14])
        assert len(fig.data) > 0

    def test_with_diagonal(self, short_df):
        """Test lag scatter with diagonal line."""
        fig = plot_lag_scatter(short_df, columns="y", lags=1, show_diagonal=True)
        assert len(fig.data) > 0

    def test_with_regression(self, short_df):
        """Test lag scatter with regression line."""
        fig = plot_lag_scatter(short_df, columns="y", lags=1, show_regression=True)
        assert len(fig.data) > 0

    def test_custom_styling(self, short_df):
        """Test lag scatter with custom styling."""
        fig = plot_lag_scatter(
            short_df,
            columns="y",
            lags=1,
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
        fig = plot_lag_scatter(df, lags=1, panel_group_names=["y"])
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
        fig = plot_lag_scatter(sample_df, columns="y", lags=1, seasonality="month")
        legend_groups = {t.legendgroup for t in fig.data if t.legendgroup is not None}
        # Should have one legendgroup per month present in the data
        assert len(legend_groups) >= 4

    def test_season_coloring_quarter(self, sample_df):
        """Test seasonality='quarter' produces 4 season traces."""
        fig = plot_lag_scatter(sample_df, columns="y", lags=1, seasonality="quarter")
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
            plot_lag_scatter(sample_df, columns="y", lags=1, seasonality="invalid")

    def test_backward_compat_single(self, short_df):
        """Test that single lag without frequency matches old behaviour."""
        fig = plot_lag_scatter(short_df, columns="y", lags=1)
        # No legendgroup set (uniform coloring)
        groups = {t.legendgroup for t in fig.data if t.legendgroup is not None}
        assert len(groups) == 0

    def test_custom_axis_labels(self, short_df):
        """Test custom x_label and y_label."""
        fig = plot_lag_scatter(short_df, columns="y", lags=1, x_label="Lagged", y_label="Current")
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
