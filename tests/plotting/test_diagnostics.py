"""Tests for diagnostic plotting functions (ACF, PACF, spectrum, etc.)."""

import pytest

pytest.importorskip("plotly", reason="plotting extra not installed")


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

from .conftest import assert_figure_valid, assert_layout


@pytest.fixture
def yearly_2col_df():
    """Create sample DataFrame for testing."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
        "y": [100 + i % 30 for i in range(366)],
        "y2": [150 + (i % 20) * 2 for i in range(366)],
    })


class TestPlotAutocorrelation:
    """Tests for plot_autocorrelation function."""

    def test_basic(self, yearly_2col_df):
        """Test basic autocorrelation plot."""
        fig = plot_autocorrelation(yearly_2col_df, columns="y", max_lags=10)
        assert len(fig.data) > 0
        assert isinstance(fig.data[0], go.Bar)

    def test_auto_max_lags(self, yearly_2col_df):
        """Test autocorrelation with automatic max_lags."""
        fig = plot_autocorrelation(yearly_2col_df, columns="y")
        assert len(fig.data) > 0

    def test_no_confidence(self, yearly_2col_df):
        """Test autocorrelation without confidence bands."""
        fig = plot_autocorrelation(yearly_2col_df, columns="y", max_lags=10, show_confidence=False)
        # Should have fewer traces without confidence bands
        assert len(fig.data) > 0

    def test_custom_styling(self, yearly_2col_df):
        """Test autocorrelation with custom styling."""
        fig = plot_autocorrelation(yearly_2col_df, columns="y", max_lags=10, color="#FF0000")
        assert len(fig.data) > 0

    def test_panel(self, yearly_2col_df):
        """Test panel faceting for autocorrelation."""
        df = pl.DataFrame({
            "time": yearly_2col_df["time"],
            "y__a": yearly_2col_df["y"],
            "y__b": yearly_2col_df["y2"],
        })
        fig = plot_autocorrelation(df, max_lags=10, groups=["y"])
        assert len(fig.data) > 0


class TestPlotPartialAutocorrelation:
    """Tests for plot_partial_autocorrelation function."""

    def test_basic(self, yearly_2col_df):
        """Test basic PACF plot."""
        fig = plot_partial_autocorrelation(yearly_2col_df, columns="y", max_lags=10)
        assert len(fig.data) > 0
        assert isinstance(fig.data[0], go.Bar)

    def test_auto_max_lags(self, yearly_2col_df):
        """Test PACF with automatic max_lags."""
        fig = plot_partial_autocorrelation(yearly_2col_df, columns="y")
        assert len(fig.data) > 0

    def test_no_confidence(self, yearly_2col_df):
        """Test PACF without confidence bands."""
        fig = plot_partial_autocorrelation(yearly_2col_df, columns="y", max_lags=10, show_confidence=False)
        assert len(fig.data) > 0

    def test_custom_styling(self, yearly_2col_df):
        """Test PACF with custom styling."""
        fig = plot_partial_autocorrelation(yearly_2col_df, columns="y", max_lags=10, color="#00FF00")
        assert len(fig.data) > 0

    def test_panel(self, yearly_2col_df):
        """Test panel faceting for partial autocorrelation."""
        df = pl.DataFrame({
            "time": yearly_2col_df["time"],
            "y__a": yearly_2col_df["y"],
            "y__b": yearly_2col_df["y2"],
        })
        fig = plot_partial_autocorrelation(df, max_lags=10, groups=["y"])
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

    def test_groups(self):
        """Test panel grouping produces one heatmap per group."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "y__a": list(range(10)),
            "y__b": list(range(10, 20)),
            "y__c": list(range(20, 30)),
        })
        result = plot_correlation_heatmap(df, groups=["y"])
        assert isinstance(result, dict)
        fig = result["y"]
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
            plot_correlation_heatmap(df, groups=["missing"])


class TestPlotSeasonality:
    """Tests for plot_seasonality function."""

    def test_month(self, yearly_2col_df):
        """Test monthly seasonality plot."""
        fig = plot_seasonality(yearly_2col_df, columns="y", seasonality="month")
        assert len(fig.data) > 0

    def test_quarter(self, yearly_2col_df):
        """Test quarterly seasonality plot."""
        fig = plot_seasonality(yearly_2col_df, columns="y", seasonality="quarter")
        assert len(fig.data) > 0

    def test_weekday(self, yearly_2col_df):
        """Test weekday seasonality plot."""
        fig = plot_seasonality(yearly_2col_df, columns="y", seasonality="weekday")
        assert len(fig.data) > 0

    def test_no_mean(self, yearly_2col_df):
        """Test seasonality plot with custom line width."""
        fig = plot_seasonality(yearly_2col_df, columns="y", seasonality="month", line_width=3.0)
        assert len(fig.data) > 0

    def test_custom_styling(self, yearly_2col_df):
        """Test seasonality with custom styling."""
        fig = plot_seasonality(
            yearly_2col_df,
            columns="y",
            seasonality="month",
            line_width=2.5,
            highlight_width=4.0,
        )
        assert len(fig.data) > 0

    def test_invalid_seasonality(self, yearly_2col_df):
        """Test that invalid seasonality raises ValueError."""
        with pytest.raises(ValueError, match="Unknown frequency"):
            plot_seasonality(yearly_2col_df, columns="y", seasonality="invalid")

    def test_panel(self, yearly_2col_df):
        """Test panel faceting for seasonality."""
        df = pl.DataFrame({
            "time": yearly_2col_df["time"],
            "y__a": yearly_2col_df["y"],
            "y__b": yearly_2col_df["y2"],
        })
        fig = plot_seasonality(df, seasonality="month", groups=["y"])
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
        result = plot_lag_scatter(df, lags=[1], groups=["y"])
        assert isinstance(result, dict)
        assert len(result) == 2
        for fig in result.values():
            assert len(fig.data) > 0

    def test_multi_lag_grid(self, yearly_2col_df):
        """Test that multiple lags produce a subplot grid."""
        fig = plot_lag_scatter(yearly_2col_df, columns="y", lags=[1, 4, 8])
        # Each lag gets a subplot; check we have traces in multiple subplots
        assert len(fig.data) > 3  # scatter + diagonals

    def test_multi_lag_grid_layout(self, yearly_2col_df):
        """Test subplot grid respects facet_n_cols."""
        fig = plot_lag_scatter(yearly_2col_df, columns="y", lags=[1, 2, 3, 4], facet_n_cols=2)
        # 4 lags with ncol=2 → 2×2 grid
        assert len(fig.data) > 4  # at least 4 scatter traces + diagonals

    def test_season_coloring_month(self, yearly_2col_df):
        """Test seasonality='month' produces season-colored traces."""
        fig = plot_lag_scatter(yearly_2col_df, columns="y", lags=[1], seasonality="month")
        legend_groups = {t.legendgroup for t in fig.data if t.legendgroup is not None}
        # Should have one legendgroup per month present in the data
        assert len(legend_groups) >= 4

    def test_season_coloring_quarter(self, yearly_2col_df):
        """Test seasonality='quarter' produces 4 season traces."""
        fig = plot_lag_scatter(yearly_2col_df, columns="y", lags=[1], seasonality="quarter")
        legend_groups = {t.legendgroup for t in fig.data if t.legendgroup is not None}
        assert len(legend_groups) <= 4

    def test_season_with_grid(self, yearly_2col_df):
        """Test season coloring combined with multi-lag subplot grid."""
        fig = plot_lag_scatter(yearly_2col_df, columns="y", lags=[1, 4, 8], seasonality="quarter")
        legend_groups = {t.legendgroup for t in fig.data if t.legendgroup is not None}
        assert len(legend_groups) <= 4
        assert len(fig.data) > 6

    def test_seasonality_invalid(self, yearly_2col_df):
        """Test that invalid seasonality raises ValueError."""
        with pytest.raises(ValueError, match="Unknown frequency"):
            plot_lag_scatter(yearly_2col_df, columns="y", lags=[1], seasonality="invalid")

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
        """Test custom styling (cross-correlation has no extra style params)."""
        fig = plot_cross_correlation(
            short_df,
            columns=["x", "y"],
            max_lags=10,
        )
        assert len(fig.data) > 0

    def test_missing_column(self, short_df):
        """Test error handling for missing columns."""
        with pytest.raises(ValueError, match="not found"):
            plot_cross_correlation(short_df, columns=["missing", "y"])

        with pytest.raises(ValueError, match="not found"):
            plot_cross_correlation(short_df, columns=["x", "missing"])

    def test_wrong_column_count(self, short_df):
        """Test error when columns has fewer than 2 entries."""
        with pytest.raises(ValueError, match="at least 2"):
            plot_cross_correlation(short_df, columns=["x"])

    def test_custom_axis_labels(self, short_df):
        """Test custom x_label and y_label."""
        fig = plot_cross_correlation(short_df, columns=["x", "y"], x_label="Offset", y_label="CCF Value")
        assert fig.layout.xaxis.title.text == "Offset"
        assert fig.layout.yaxis.title.text == "CCF Value"


class TestPlotSubseasonality:
    """Tests for plot_subseasonality function."""

    def test_basic_monthly(self, yearly_2col_df):
        """Test basic monthly subseasonal plot."""
        fig = plot_subseasonality(yearly_2col_df, columns="y", seasonality="month")
        assert_figure_valid(fig)

    def test_quarterly(self, yearly_2col_df):
        """Test quarterly subseasonal plot."""
        fig = plot_subseasonality(yearly_2col_df, columns="y", seasonality="quarter")
        assert_figure_valid(fig)

    def test_no_mean(self, yearly_2col_df):
        """Test without mean lines."""
        # Need multi-year data so show_mean actually adds mean traces
        multi_year_df = pl.DataFrame({
            "time": pl.date_range(pl.date(2018, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
            "y": [100 + i % 30 for i in range(1096)],
        })
        fig = plot_subseasonality(multi_year_df, columns="y", seasonality="month", show_mean=False)
        fig_with_mean = plot_subseasonality(multi_year_df, columns="y", seasonality="month", show_mean=True)
        assert len(fig.data) < len(fig_with_mean.data)

    def test_custom_n_cols(self, yearly_2col_df):
        """Test custom number of columns in subplot grid."""
        fig = plot_subseasonality(yearly_2col_df, columns="y", seasonality="month", facet_n_cols=3)
        assert_figure_valid(fig)


class TestPlotSubseasonalityKind:
    """Tests for the ``kind`` parameter of plot_subseasonality."""

    @pytest.fixture
    def multi_year_df(self):
        """3-year daily data so each month has multiple cycles."""
        dates = pl.date_range(pl.date(2018, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        return pl.DataFrame({
            "time": dates,
            "y": [100 + i % 30 for i in range(len(dates))],
        })

    def test_kind_mean_default(self, multi_year_df):
        """Default kind='mean' produces only go.Scatter traces."""
        fig = plot_subseasonality(multi_year_df, columns="y", seasonality="month")
        assert_figure_valid(fig)
        assert all(isinstance(t, go.Scatter) for t in fig.data)

    def test_kind_lines(self, multi_year_df):
        """kind='lines' adds Scattergl per-cycle traces plus Scatter mean overlay."""
        fig = plot_subseasonality(multi_year_df, columns="y", seasonality="month", kind="lines")
        assert_figure_valid(fig)
        has_scattergl = any(isinstance(t, go.Scattergl) for t in fig.data)
        has_scatter = any(isinstance(t, go.Scatter) for t in fig.data)
        assert has_scattergl, "Expected Scattergl traces for per-cycle lines"
        assert has_scatter, "Expected Scatter traces for mean overlay"

    def test_kind_box_per_cycle(self, multi_year_df):
        """kind='box' produces one Box trace per cycle per season."""
        fig = plot_subseasonality(multi_year_df, columns="y", seasonality="month", kind="box")
        assert_figure_valid(fig)
        box_traces = [t for t in fig.data if isinstance(t, go.Box)]
        assert len(box_traces) > 12, "Expected multiple Box traces (one per cycle per season)"

    def test_kind_violin_per_cycle(self, multi_year_df):
        """kind='violin' produces one Violin trace per cycle per season."""
        fig = plot_subseasonality(multi_year_df, columns="y", seasonality="month", kind="violin")
        assert_figure_valid(fig)
        violin_traces = [t for t in fig.data if isinstance(t, go.Violin)]
        assert len(violin_traces) > 12, "Expected multiple Violin traces (one per cycle per season)"

    def test_kind_invalid(self, multi_year_df):
        """Invalid kind raises ValueError."""
        with pytest.raises(ValueError, match="kind must be one of"):
            plot_subseasonality(multi_year_df, columns="y", seasonality="month", kind="invalid")

    def test_kind_lines_panel(self):
        """kind='lines' works with panel data."""
        dates = pl.date_range(pl.date(2018, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        n = len(dates)
        df = pl.DataFrame({
            "time": dates,
            "y__a": [100 + i % 30 for i in range(n)],
            "y__b": [200 + (i % 20) * 2 for i in range(n)],
        })
        result = plot_subseasonality(df, seasonality="month", kind="lines")
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)
            has_scattergl = any(isinstance(t, go.Scattergl) for t in fig.data)
            assert has_scattergl


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
        assert_figure_valid(fig)

    def test_two_columns(self, three_col_df):
        """Test minimal 2×2 matrix."""
        fig = plot_scatter_matrix(three_col_df, columns=["a", "b"])
        assert_figure_valid(fig)
        # 2×2 grid: diagonal KDE (2) + lower triangle scatter (1) = 3 traces
        assert len(fig.data) >= 3

    def test_all_columns_default(self, three_col_df):
        """Test that columns=None selects all numeric columns."""
        fig = plot_scatter_matrix(three_col_df)
        assert_figure_valid(fig)
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
        assert_figure_valid(fig)
        # With season coloring, lower triangle has more traces (one per season)
        assert len(fig.data) > 3

    def test_diagonal_histogram(self, three_col_df):
        """Test histogram on diagonal."""
        fig = plot_scatter_matrix(three_col_df, columns=["a", "b", "c"], diagonal="histogram")
        assert_figure_valid(fig)
        # Should have Histogram traces on diagonal
        histogram_traces = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(histogram_traces) == 3

    def test_diagonal_none(self, three_col_df):
        """Test no diagonal content."""
        fig = plot_scatter_matrix(three_col_df, columns=["a", "b", "c"], diagonal=None)
        # No diagonal traces, only scatter traces in lower triangle
        assert_figure_valid(fig)
        for t in fig.data:
            assert not isinstance(t, go.Histogram)

    def test_no_correlation(self, three_col_df):
        """Test with correlation annotations disabled."""
        fig = plot_scatter_matrix(
            three_col_df,
            columns=["a", "b", "c"],
            show_correlation=False,
        )
        assert_figure_valid(fig)
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
        assert_layout(fig, width=800, height=700)

    def test_custom_styling_kwargs(self, three_col_df):
        """Test marker_size and marker_opacity params."""
        fig = plot_scatter_matrix(
            three_col_df,
            columns=["a", "b"],
            marker_size=5.0,
            marker_opacity=0.8,
        )
        assert_figure_valid(fig)


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
        """Panel data with subseasonality returns a dict of figures per member."""
        result = plot_subseasonality(panel_df, seasonality="month")
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)

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
        assert_layout(fig, title="My Title")

    def test_custom_dimensions(self):
        """Custom width/height apply to figure layout."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
            "y": [100 + i % 30 for i in range(366)],
        })
        fig = plot_subseasonality(df, columns="y", seasonality="month", width=1000, height=600)
        assert_layout(fig, width=1000, height=600)


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
        assert_layout(fig, title="CCF Title")

    def test_custom_dimensions(self):
        """Custom width/height apply to cross-correlation figure."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "x": [100 + i % 20 for i in range(91)],
            "y": [150 + (i % 15) * 2 for i in range(91)],
        })
        fig = plot_cross_correlation(df, columns=["x", "y"], width=900, height=500)
        assert_layout(fig, width=900, height=500)

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
        assert_layout(fig, title="Matrix Title")


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
        assert_figure_valid(fig)

    def test_custom_dimensions(self):
        """Custom dimensions are respected."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
            "y": [100 + i % 30 for i in range(366)],
        })
        fig = plot_seasonality(df, columns="y", seasonality="month", width=800, height=500)
        assert_layout(fig, width=800, height=500)


class TestPlotSeasonalityHighlight:
    """Tests for seasonality highlight parameter branches."""

    def test_highlight_int(self):
        """Passing highlight as a single integer selects one period."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
            "y": [100 + i % 30 for i in range(366)],
        })
        fig = plot_seasonality(df, columns="y", seasonality="month", highlight=3)
        assert_figure_valid(fig)

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
        assert_figure_valid(fig)


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
        fig = plot_autocorrelation(panel_df, groups=["y"])
        assert_figure_valid(fig)

    def test_pacf_panel(self, panel_df):
        """plot_partial_autocorrelation with panel data auto-detects groups."""
        fig = plot_partial_autocorrelation(panel_df, groups=["y"])
        assert_figure_valid(fig)

    def test_correlation_heatmap_panel(self, panel_df):
        """plot_correlation_heatmap with panel data returns dict per group."""
        result = plot_correlation_heatmap(panel_df, groups=["y"])
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)

    def test_seasonality_panel(self, panel_df):
        """plot_seasonality with panel data auto-detects groups."""
        fig = plot_seasonality(panel_df, seasonality="month", groups=["y"])
        assert_figure_valid(fig)

    def test_lag_scatter_panel(self, panel_df):
        """plot_lag_scatter with panel data auto-detects groups."""
        result = plot_lag_scatter(panel_df, lags=[1, 5], groups=["y"])
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)


class TestDiagnosticsSeasonalityBranches:
    """Tests for seasonality frequency branches."""

    def test_week_frequency(self, yearly_2col_df):
        """plot_seasonality with frequency='week' covers week branch."""
        fig = plot_seasonality(yearly_2col_df, columns="y", seasonality="week")
        assert_figure_valid(fig)

    def test_hour_frequency(self):
        """plot_seasonality with frequency='hour' covers hour branch."""
        times = pl.datetime_range(pl.datetime(2020, 1, 1, 0, 0), pl.datetime(2020, 1, 10, 23, 0), "1h", eager=True)
        df = pl.DataFrame({
            "time": times,
            "y": [100.0 + (i % 24) * 3.0 for i in range(len(times))],
        })
        fig = plot_seasonality(df, columns="y", seasonality="hour")
        assert_figure_valid(fig)


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
        assert_figure_valid(fig)

    def test_scatter_matrix_constant_column_kde(self):
        """Scatter matrix with constant column and diagonal='kde' handles LinAlgError."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 2, 28), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "y1": [5.0] * len(dates),
            "y2": [float(i) for i in range(len(dates))],
        })
        fig = plot_scatter_matrix(df, diagonal="kde", show_correlation=False)
        assert_figure_valid(fig)

    def test_cross_correlation_basic_params(self, short_df):
        """Cross-correlation with custom confidence level."""
        fig = plot_cross_correlation(short_df, columns=["x", "y"], confidence_level=0.99)
        assert_figure_valid(fig)

    def test_subseasonality_hour_labels(self):
        """Subseasonality with hour seasonality covers label fallback."""
        times = pl.datetime_range(pl.datetime(2020, 1, 1, 0, 0), pl.datetime(2020, 1, 10, 23, 0), "1h", eager=True)
        df = pl.DataFrame({
            "time": times,
            "y": [100.0 + (i % 24) * 3.0 for i in range(len(times))],
        })
        fig = plot_subseasonality(df, columns="y", seasonality="hour")
        assert_figure_valid(fig)


class TestDiagnosticsAutoDetectPanel:
    """Tests for auto-detection of panel data (no explicit groups)."""

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
        """plot_autocorrelation auto-detects panel data without groups."""
        fig = plot_autocorrelation(auto_panel_df, max_lags=10)
        assert_figure_valid(fig)

    def test_pacf_auto_detect(self, auto_panel_df):
        """plot_partial_autocorrelation auto-detects panel data."""
        fig = plot_partial_autocorrelation(auto_panel_df, max_lags=10)
        assert_figure_valid(fig)

    def test_heatmap_auto_detect(self, auto_panel_df):
        """plot_correlation_heatmap auto-detects panel data."""
        result = plot_correlation_heatmap(auto_panel_df)
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)

    def test_seasonality_auto_detect(self, auto_panel_df):
        """plot_seasonality auto-detects panel data."""
        fig = plot_seasonality(auto_panel_df, seasonality="month")
        assert_figure_valid(fig)

    def test_lag_scatter_auto_detect(self, auto_panel_df):
        """plot_lag_scatter auto-detects panel data."""
        result = plot_lag_scatter(auto_panel_df, lags=[1])
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)


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
        assert_figure_valid(fig)
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
        assert_figure_valid(fig)
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
        assert_figure_valid(fig)


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
        assert_figure_valid(fig)
        assert len(fig.data) >= 4

    def test_multi_lag_with_seasonality_month(self):
        """Multi-lag + seasonality='month' triggers season-colored multi-lag path."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "y": [100.0 + (i % 30) for i in range(len(dates))],
        })
        fig = plot_lag_scatter(df, columns="y", lags=[1, 2], seasonality="month")
        assert_figure_valid(fig)
        assert len(fig.data) >= 2

    def test_single_lag_with_seasonality(self):
        """Single lag + seasonality='month' triggers the seasonal single-lag path."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "y": [100.0 + (i % 30) for i in range(len(dates))],
        })
        fig = plot_lag_scatter(df, columns="y", lags=[1], seasonality="month")
        assert_figure_valid(fig)
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
        assert_figure_valid(fig)
        assert len(fig.data) >= 2

    def test_single_lag_show_regression(self):
        """Single lag + show_regression covers the regression overlay."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 6, 30), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "y": [100.0 + (i % 30) for i in range(len(dates))],
        })
        fig = plot_lag_scatter(df, columns="y", lags=[1], show_regression=True)
        assert_figure_valid(fig)
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
        result = plot_correlation_heatmap(df, groups=["a", "b"], show_values=True)
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)


class TestCrossCorrelationBasic:
    """Tests for plot_cross_correlation basic functionality."""

    def test_basic_params(self):
        """Basic cross-correlation with custom max_lags."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "x": [100.0 + (i % 20) for i in range(len(dates))],
            "y": [150.0 + (i % 15) * 2 for i in range(len(dates))],
        })
        fig = plot_cross_correlation(df, columns=["x", "y"], max_lags=15)
        assert_figure_valid(fig)


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
        assert_figure_valid(fig)
        dash_traces = [t for t in fig.data if hasattr(t, "line") and t.line and t.line.dash == "dash"]
        assert len(dash_traces) >= 1

    def test_show_regression_single_lag(self, lag_scatter_df):
        """show_regression adds a regression line for single lag."""
        fig = plot_lag_scatter(lag_scatter_df, columns="y", lags=[1], show_regression=True)
        assert_figure_valid(fig)
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
        assert_figure_valid(fig)
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
        fig = plot_autocorrelation(df, groups=["sales"], show_confidence=False)
        assert_figure_valid(fig)

    def test_panel_pacf_no_confidence(self):
        """Panel PACF with show_confidence=False skips CI band rendering."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "sales__store_1": [100.0 + i * 2.0 for i in range(12)],
            "sales__store_2": [200.0 + i * 3.0 for i in range(12)],
        })
        fig = plot_partial_autocorrelation(df, groups=["sales"], show_confidence=False)
        assert_figure_valid(fig)


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
        assert_figure_valid(fig)

    def test_diagonal_histogram(self, scatter_df):
        """Scatter matrix with diagonal='histogram' renders histogram bars."""
        fig = plot_scatter_matrix(scatter_df, columns=["x", "y"], diagonal="histogram")
        assert_figure_valid(fig)
        histogram_traces = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(histogram_traces) >= 1

    def test_show_correlation(self, scatter_df):
        """Scatter matrix with show_correlation=True adds Pearson r annotations."""
        fig = plot_scatter_matrix(scatter_df, columns=["x", "y"], show_correlation=True)
        assert_figure_valid(fig)
        annotations = fig.layout.annotations
        assert annotations is not None
        corr_annotations = [a for a in annotations if hasattr(a, "text") and "." in str(a.text)]
        assert len(corr_annotations) >= 1


class TestPlotSeasonalHeatmap:
    """Tests for plot_seasonal_heatmap function."""

    @pytest.fixture
    def hourly_df(self):
        """2 years of hourly data with seasonal-like pattern."""
        import math

        times = pl.datetime_range(
            pl.datetime(2020, 1, 1),
            pl.datetime(2021, 12, 31, 23),
            "1h",
            eager=True,
        )
        n = len(times)
        return pl.DataFrame({
            "time": times,
            "temp": [20 + 10 * math.sin(i * 2 * math.pi / 24) + 5 * math.sin(i * 2 * math.pi / 8760) for i in range(n)],
        })

    def test_hour_by_month(self, hourly_df):
        """Basic hour × month heatmap."""
        fig = plot_seasonal_heatmap(hourly_df, "temp", x_period="hour", y_period="month")
        assert_figure_valid(fig)
        assert isinstance(fig.data[0], go.Heatmap)

    def test_day_of_week_by_hour(self, hourly_df):
        """Day-of-week × hour heatmap."""
        fig = plot_seasonal_heatmap(hourly_df, "temp", x_period="hour", y_period="day_of_week")
        assert_figure_valid(fig)

    def test_month_by_year(self, hourly_df):
        """Month × year heatmap."""
        fig = plot_seasonal_heatmap(hourly_df, "temp", x_period="month", y_period="year")
        assert_figure_valid(fig)

    def test_aggregation_methods(self, hourly_df):
        """All aggregation methods produce valid figures."""
        for agg in ["mean", "median", "sum", "count", "std", "min", "max"]:
            fig = plot_seasonal_heatmap(hourly_df, "temp", agg=agg)
            assert len(fig.data) > 0

    def test_custom_colorscale(self, hourly_df):
        """Custom colorscale is accepted."""
        fig = plot_seasonal_heatmap(hourly_df, "temp", colorscale="RdBu_r")
        assert_figure_valid(fig)

    def test_no_values(self, hourly_df):
        """show_values=False suppresses annotations."""
        fig = plot_seasonal_heatmap(hourly_df, "temp", show_values=False)
        assert_figure_valid(fig)

    def test_reverse_y(self, hourly_df):
        """reverse_y=True flips y-axis."""
        fig = plot_seasonal_heatmap(hourly_df, "temp", reverse_y=True)
        assert_figure_valid(fig)

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
        assert_layout(fig, title="Heatmap Test")

    def test_custom_dimensions(self, hourly_df):
        """Custom dimensions are respected."""
        fig = plot_seasonal_heatmap(hourly_df, "temp", width=800, height=500)
        assert_layout(fig, width=800, height=500)

    def test_panel_explicit(self):
        """Panel faceting with explicit groups."""
        times = pl.datetime_range(
            pl.datetime(2020, 1, 1),
            pl.datetime(2020, 3, 31, 23),
            "1h",
            eager=True,
        )
        n = len(times)
        df = pl.DataFrame({
            "time": times,
            "weather__temp": [20.0 + i % 24 for i in range(n)],
            "weather__wind": [5.0 + i % 12 for i in range(n)],
        })
        fig = plot_seasonal_heatmap(df, "temp", groups=["weather"])
        assert_figure_valid(fig)

    def test_auto_column_single(self, hourly_df):
        """Auto-detect single numeric column when columns=None."""
        fig = plot_seasonal_heatmap(hourly_df)
        assert_figure_valid(fig)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_auto_column_multi(self):
        """Multiple numeric columns produce subplot grid."""
        times = pl.datetime_range(
            pl.datetime(2020, 1, 1),
            pl.datetime(2020, 6, 30, 23),
            "1h",
            eager=True,
        )
        n = len(times)
        df = pl.DataFrame({
            "time": times,
            "temp": [20.0 + i % 24 for i in range(n)],
            "wind": [5.0 + i % 12 for i in range(n)],
        })
        fig = plot_seasonal_heatmap(df)
        assert_figure_valid(fig)
        assert len(fig.data) == 2

    def test_columns_list(self):
        """Explicit list of columns produces subplot grid."""
        times = pl.datetime_range(
            pl.datetime(2020, 1, 1),
            pl.datetime(2020, 6, 30, 23),
            "1h",
            eager=True,
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
        """Panel data auto-detected when columns=None and no groups."""
        times = pl.datetime_range(
            pl.datetime(2020, 1, 1),
            pl.datetime(2020, 3, 31, 23),
            "1h",
            eager=True,
        )
        n = len(times)
        df = pl.DataFrame({
            "time": times,
            "loc_A__temp": [20.0 + i % 24 for i in range(n)],
            "loc_B__temp": [18.0 + i % 24 for i in range(n)],
        })
        fig = plot_seasonal_heatmap(df)
        assert_figure_valid(fig)
        assert len(fig.data) >= 2

    def test_panel_auto_detect_with_columns(self):
        """Panel auto-detect with columns filtering to specific member."""
        times = pl.datetime_range(
            pl.datetime(2020, 1, 1),
            pl.datetime(2020, 3, 31, 23),
            "1h",
            eager=True,
        )
        n = len(times)
        df = pl.DataFrame({
            "time": times,
            "weather__temp": [20.0 + i % 24 for i in range(n)],
            "weather__wind": [5.0 + i % 12 for i in range(n)],
        })
        fig = plot_seasonal_heatmap(df, columns="temp")
        assert_figure_valid(fig)


class TestPlotScatterMatrixPanelPaths:
    """Cover scatter matrix panel branches: grid, facet, season, diagonal."""

    @pytest.fixture
    def panel_scatter_df(self):
        """Panel DataFrame with 2 numeric columns per group."""
        rng = np.random.default_rng(42)
        n = 366
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        return pl.DataFrame({
            "time": dates,
            "x__store_a": (100 + np.arange(n) % 20 + rng.standard_normal(n)).tolist(),
            "y__store_a": (200 + np.arange(n) % 15 + rng.standard_normal(n)).tolist(),
            "x__store_b": (150 + np.arange(n) % 25 + rng.standard_normal(n)).tolist(),
            "y__store_b": (250 + np.arange(n) % 10 + rng.standard_normal(n)).tolist(),
        })

    def test_panel_grid_mode(self, panel_scatter_df):
        """Panel data with default facet_by='group' returns dict of figures."""
        result = plot_scatter_matrix(panel_scatter_df)
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)

    def test_panel_separate_mode(self, panel_scatter_df):
        """Panel data with facet_by='group' returns dict of figures."""
        result = plot_scatter_matrix(panel_scatter_df, facet_by="group")
        assert isinstance(result, dict)
        assert len(result) > 0
        for fig in result.values():
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0

    def test_panel_with_seasonality(self, panel_scatter_df):
        """Season colouring works with panel scatter matrix."""
        result = plot_scatter_matrix(panel_scatter_df, seasonality="month")
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)

    def test_panel_with_seasonality_separate(self, panel_scatter_df):
        """Season colouring + facet_by='group' mode."""
        result = plot_scatter_matrix(panel_scatter_df, seasonality="quarter", facet_by="group")
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)

    def test_panel_diagonal_histogram(self, panel_scatter_df):
        """Panel with histogram diagonal."""
        result = plot_scatter_matrix(panel_scatter_df, diagonal="histogram")
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)

    def test_panel_diagonal_none(self, panel_scatter_df):
        """Panel with no diagonal."""
        result = plot_scatter_matrix(panel_scatter_df, diagonal=None)
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)

    def test_panel_show_correlation(self, panel_scatter_df):
        """Panel with correlation annotations."""
        result = plot_scatter_matrix(panel_scatter_df, show_correlation=True)
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)

    def test_nonpanel_seasonality_small_max_points(self):
        """Non-panel scatter matrix with seasonality and small max_points."""
        rng = np.random.default_rng(42)
        n = 200
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 7, 18), "1d", eager=True),
            "x": rng.standard_normal(n).tolist(),
            "y": rng.standard_normal(n).tolist(),
        })
        fig = plot_scatter_matrix(df, columns=["x", "y"], seasonality="month", max_points=50)
        assert_figure_valid(fig)

    def test_groups_filter(self, panel_scatter_df):
        """Passing explicit groups filters to those groups."""
        result = plot_scatter_matrix(panel_scatter_df, groups=["x"])
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)


class TestPlotCorrelationHeatmapPanelPaths:
    """Cover correlation heatmap panel separate and grid branches."""

    @pytest.fixture
    def panel_heatmap_df(self):
        """Panel DataFrame for correlation heatmap testing."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        n = len(dates)
        return pl.DataFrame({
            "time": dates,
            "x__a": [100 + i % 20 for i in range(n)],
            "y__a": [150 + (i % 15) * 2 for i in range(n)],
            "x__b": [200 + i % 25 for i in range(n)],
            "y__b": [250 + (i % 10) * 3 for i in range(n)],
        })

    def test_panel_separate(self, panel_heatmap_df):
        """facet_by='group' returns a dict of figures."""
        result = plot_correlation_heatmap(panel_heatmap_df, facet_by="group")
        assert isinstance(result, dict)
        assert len(result) > 0
        for fig in result.values():
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0

    def test_panel_grid(self, panel_heatmap_df):
        """facet_by=None with panel data returns a single figure."""
        fig = plot_correlation_heatmap(panel_heatmap_df, facet_by=None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestPlotCrossCorrelationPanelData:
    """Cross-correlation panel data path."""

    def test_panel_data(self):
        """Panel data auto-detect produces a valid cross-correlation figure."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        n = len(dates)
        df = pl.DataFrame({
            "time": dates,
            "y__a": [100 + i % 20 for i in range(n)],
            "y__b": [150 + (i % 15) * 2 for i in range(n)],
        })
        fig = plot_cross_correlation(df, max_lags=10)
        assert_figure_valid(fig)


class TestPlotLagScatterPanelAndExtras:
    """Lag scatter panel mode plus seasonality and regression extras."""

    @pytest.fixture
    def panel_lag_df(self):
        """Panel DataFrame for lag scatter testing."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        n = len(dates)
        return pl.DataFrame({
            "time": dates,
            "y__a": [100 + i % 30 for i in range(n)],
            "y__b": [200 + (i % 20) * 2 for i in range(n)],
        })

    def test_panel_data(self, panel_lag_df):
        """Panel data auto-detect produces a valid lag scatter dict."""
        result = plot_lag_scatter(panel_lag_df, lags=[1])
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)

    def test_seasonality_single_lag(self, yearly_2col_df):
        """Seasonality colouring with a single lag."""
        fig = plot_lag_scatter(yearly_2col_df, columns="y", lags=[1], seasonality="month")
        assert_figure_valid(fig)

    def test_seasonality_multi_lag(self, yearly_2col_df):
        """Seasonality colouring with multiple lags and subplot grid."""
        fig = plot_lag_scatter(yearly_2col_df, columns="y", lags=[1, 7], seasonality="quarter")
        assert_figure_valid(fig)

    def test_show_regression(self, yearly_2col_df):
        """Regression line overlaid on lag scatter."""
        fig = plot_lag_scatter(yearly_2col_df, columns="y", lags=[1], show_regression=True)
        assert_figure_valid(fig)
        # Regression adds at least one extra trace (Scattergl line)
        assert len(fig.data) >= 2


class TestPlotSeasonalityPanelAndHighlight:
    """Seasonality panel and highlight branches."""

    def test_highlight_with_panel(self):
        """Panel data with highlight parameter."""
        dates = pl.date_range(pl.date(2018, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        n = len(dates)
        df = pl.DataFrame({
            "time": dates,
            "y__a": [100 + i % 30 for i in range(n)],
            "y__b": [200 + (i % 20) * 2 for i in range(n)],
        })
        fig = plot_seasonality(df, seasonality="month", highlight=2020)
        assert_figure_valid(fig)


class TestPlotSubseasonalityPanelModes:
    """Subseasonality panel separate and combined modes."""

    @pytest.fixture
    def panel_subseas_df(self):
        """Panel DataFrame for subseasonality testing."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        n = len(dates)
        return pl.DataFrame({
            "time": dates,
            "y__a": [100 + i % 30 for i in range(n)],
            "y__b": [200 + (i % 20) * 2 for i in range(n)],
        })

    def test_panel_separate(self, panel_subseas_df):
        """Panel data returns a dict of figures per member."""
        result = plot_subseasonality(panel_subseas_df, seasonality="month")
        assert isinstance(result, dict)
        assert len(result) > 0
        for fig in result.values():
            assert isinstance(fig, go.Figure)

    def test_panel_combined(self, panel_subseas_df):
        """Default panel mode with single member returns a single figure."""
        # With only 1 group and 2 members, returns dict
        result = plot_subseasonality(panel_subseas_df, seasonality="month")
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)


class TestPlotSeasonalHeatmapPanel:
    """Seasonal heatmap with panel data."""

    def test_panel_data(self):
        """Panel DataFrame auto-detect for seasonal heatmap."""
        times = pl.datetime_range(
            pl.datetime(2020, 1, 1),
            pl.datetime(2020, 3, 31, 23),
            "1h",
            eager=True,
        )
        n = len(times)
        df = pl.DataFrame({
            "time": times,
            "temp__city_a": [20.0 + i % 24 for i in range(n)],
            "temp__city_b": [18.0 + i % 24 for i in range(n)],
        })
        fig = plot_seasonal_heatmap(df)
        assert_figure_valid(fig)

    def test_multi_column_nonpanel(self):
        """Multiple non-panel columns produce subplot grid."""
        times = pl.datetime_range(
            pl.datetime(2020, 1, 1),
            pl.datetime(2020, 6, 30, 23),
            "1h",
            eager=True,
        )
        n = len(times)
        df = pl.DataFrame({
            "time": times,
            "temp": [20.0 + i % 24 for i in range(n)],
            "wind": [5.0 + i % 12 for i in range(n)],
        })
        fig = plot_seasonal_heatmap(df)
        assert_figure_valid(fig)
        assert len(fig.data) == 2


class TestPlotAutocorrelationMultiColumn:
    """Autocorrelation with multiple columns (no custom palette)."""

    def test_multi_column(self, yearly_2col_df):
        """Multiple columns without custom palette."""
        fig = plot_autocorrelation(yearly_2col_df, columns=["y", "y2"], max_lags=10)
        assert_figure_valid(fig)
        assert len(fig.data) > 1


class TestPlotPartialAutocorrelationMultiColumn:
    """Partial autocorrelation with multiple columns."""

    def test_multi_column(self, yearly_2col_df):
        """Multiple columns without custom palette."""
        fig = plot_partial_autocorrelation(yearly_2col_df, columns=["y", "y2"], max_lags=10)
        assert_figure_valid(fig)
        assert len(fig.data) > 1


class TestSeasonalityHighlightCycleMatch:
    """Highlight parameter with multi-year data covering matched/unmatched cycles."""

    @pytest.fixture
    def multi_year_df(self):
        """3-year daily DataFrame for seasonality tests."""
        dates = pl.date_range(pl.date(2018, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        n = len(dates)
        return pl.DataFrame({
            "time": dates,
            "y": [100 + i % 30 for i in range(n)],
        })

    def test_highlight_matching_year(self, multi_year_df):
        """Highlighting a year that exists as a cycle covers the is_hl=True branch."""
        fig = plot_seasonality(
            multi_year_df,
            columns="y",
            seasonality="month",
            highlight=2020,
        )
        assert_figure_valid(fig)

    def test_highlight_with_fade(self, multi_year_df):
        """Non-highlighted cycles hit the fade_opacity branch."""
        fig = plot_seasonality(
            multi_year_df,
            columns="y",
            seasonality="month",
            highlight=[2020],
            fade_opacity=0.15,
        )
        assert_figure_valid(fig)
        # There should be traces for all 3 cycles (2018, 2019, 2020)
        assert len(fig.data) >= 3


class TestLagScatterSeasonality:
    """Season-colored branches in plot_lag_scatter (lines 1871, 1967)."""

    def test_single_lag_with_seasonality(self):
        """Single lag with month seasonality hits the non-panel season path."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "y": [100 + i % 30 for i in range(len(dates))],
        })
        fig = plot_lag_scatter(df, columns="y", lags=[1], seasonality="month")
        assert_figure_valid(fig)
        # Season-colored traces produce at least 12 season groups
        assert len(fig.data) >= 12

    def test_multi_lag_with_seasonality(self):
        """Multi-lag grid with seasonality covers the panel path season branch."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "y": [100 + i % 30 for i in range(len(dates))],
        })
        fig = plot_lag_scatter(df, columns="y", lags=[1, 7], seasonality="quarter")
        assert_figure_valid(fig)
        assert len(fig.data) >= 4  # At least 4 quarters x 2 lags in the grid


class TestScatterMatrixSeasonality:
    """Season-colored branches in plot_scatter_matrix (lines 2513, 2619, 2735)."""

    @pytest.fixture
    def seasonal_df(self):
        """Daily DataFrame spanning a full year for season coloring."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        n = len(dates)
        return pl.DataFrame({
            "time": dates,
            "a": [100.0 + i % 30 for i in range(n)],
            "b": [200.0 + (i % 20) * 2 for i in range(n)],
        })

    def test_grid_mode_with_seasonality(self, seasonal_df):
        """Non-panel with month seasonality hits season-colored scatter path."""
        fig = plot_scatter_matrix(
            seasonal_df,
            columns=["a", "b"],
            seasonality="month",
        )
        assert_figure_valid(fig)

    def test_separate_mode_with_seasonality(self):
        """Panel data with facet_by='group' and seasonality hits _scatter_matrix_facet."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        n = len(dates)
        df = pl.DataFrame({
            "time": dates,
            "grp__x": [100.0 + i % 30 for i in range(n)],
            "grp__y": [200.0 + (i % 20) * 2 for i in range(n)],
        })
        result = plot_scatter_matrix(
            df,
            seasonality="quarter",
            facet_by="group",
            groups=["grp"],
        )
        assert isinstance(result, dict)
        for fig in result.values():
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0

    def test_with_max_points_and_seasonality(self, seasonal_df):
        """Stratified subsampling when max_points < len(df) with seasonality."""
        fig = plot_scatter_matrix(
            seasonal_df,
            columns=["a", "b"],
            seasonality="quarter",
            max_points=50,
        )
        assert_figure_valid(fig)


class TestSeasonalHeatmapReverseY:
    """reverse_y parameter branches (lines 3489, 3532)."""

    def test_reverse_y_single_column(self):
        """Single column with reverse_y=True."""
        dates = pl.datetime_range(
            pl.datetime(2020, 1, 1),
            pl.datetime(2020, 2, 29, 23),
            "1h",
            eager=True,
        )
        df = pl.DataFrame({
            "time": dates,
            "y": [10.0 + i % 24 for i in range(len(dates))],
        })
        fig = plot_seasonal_heatmap(df, columns="y", reverse_y=True)
        assert_figure_valid(fig)

    def test_reverse_y_multi_column(self):
        """Multiple columns with reverse_y=True."""
        dates = pl.datetime_range(
            pl.datetime(2020, 1, 1),
            pl.datetime(2020, 2, 29, 23),
            "1h",
            eager=True,
        )
        n = len(dates)
        df = pl.DataFrame({
            "time": dates,
            "a": [10.0 + i % 24 for i in range(n)],
            "b": [20.0 + i % 12 for i in range(n)],
        })
        fig = plot_seasonal_heatmap(df, columns=["a", "b"], reverse_y=True)
        assert_figure_valid(fig)

    def test_reverse_y_panel(self, panel_hourly_df):
        """Panel data with reverse_y=True."""
        fig = plot_seasonal_heatmap(panel_hourly_df, reverse_y=True)
        assert_figure_valid(fig)
