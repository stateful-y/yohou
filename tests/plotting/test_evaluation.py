"""Tests for model evaluation plotting functions."""

import pytest

pytest.importorskip("plotly", reason="plotting extra not installed")


from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from plotly import graph_objects as go

from yohou.metrics import IntervalScore, MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError
from yohou.plotting import (
    plot_calibration,
    plot_group_scores,
    plot_residuals,
    plot_score_distribution,
    plot_score_heatmap,
    plot_score_per_step,
    plot_score_per_vintage,
    plot_score_summary,
    plot_score_time_series,
)
from yohou.weighting import LinearDecayWeighter, LookupWeighter, TableWeighter

from .conftest import assert_figure_valid, assert_layout, visible_legend_names


@pytest.fixture
def sample_residuals_data():
    """Create sample y_pred / y_truth DataFrames for residual tests."""
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
    y_truth = pl.DataFrame({
        "time": dates,
        "y": [100 + i for i in range(91)],
    })
    y_pred = pl.DataFrame({
        "time": dates,
        "y": [100 + i + (i % 3) for i in range(91)],
    })
    return {"y_truth": y_truth, "y_pred": y_pred}


@pytest.fixture
def sample_interval_data():
    """Create sample interval prediction data."""
    n = 50
    rng = np.random.default_rng(42)
    y_vals = rng.standard_normal(n)
    return {
        "y_truth": pl.DataFrame({"y": y_vals}),
        "y_pred_int": pl.DataFrame({
            "y_upper_0.9": y_vals + 1.65 + rng.uniform(0, 0.3, n),
            "y_lower_0.9": y_vals - 1.65 - rng.uniform(0, 0.3, n),
            "y_upper_0.95": y_vals + 1.96 + rng.uniform(0, 0.3, n),
            "y_lower_0.95": y_vals - 1.96 - rng.uniform(0, 0.3, n),
        }),
    }


@pytest.fixture
def multi_column_interval_data():
    """Create multi-column interval data for calibration tests."""
    n = 30
    rng = np.random.default_rng(123)
    y_vals = rng.standard_normal(n) * 10 + 100
    return {
        "y_truth": pl.DataFrame({"value": y_vals}),
        "y_pred_int": pl.DataFrame({
            "value_upper_0.9": y_vals + 5,
            "value_lower_0.9": y_vals - 5,
        }),
    }


@pytest.fixture
def multi_target_interval_data():
    """Create interval data with two target columns (a, b)."""
    n = 40
    rng = np.random.default_rng(77)
    a_vals = rng.standard_normal(n)
    b_vals = rng.standard_normal(n) * 2 + 5
    return {
        "y_truth": pl.DataFrame({"a": a_vals, "b": b_vals}),
        "y_pred_int": pl.DataFrame({
            "a_upper_0.9": a_vals + 1.65,
            "a_lower_0.9": a_vals - 1.65,
            "b_upper_0.9": b_vals + 3.3,
            "b_lower_0.9": b_vals - 3.3,
        }),
    }


@pytest.fixture
def panel_interval_data():
    """Create panel interval data with two groups (x, y) each having member a."""
    n = 30
    rng = np.random.default_rng(99)
    xa = rng.standard_normal(n)
    ya = rng.standard_normal(n) + 3
    return {
        "y_truth": pl.DataFrame({"x__a": xa, "y__a": ya}),
        "y_pred_int": pl.DataFrame({
            "x__a_upper_0.9": xa + 1.65,
            "x__a_lower_0.9": xa - 1.65,
            "y__a_upper_0.9": ya + 1.65,
            "y__a_lower_0.9": ya - 1.65,
        }),
    }


@pytest.fixture
def sample_forecast_data():
    """Create sample forecast data for score time series tests."""
    from datetime import datetime

    times = [datetime(2020, 1, 1 + i) for i in range(10)]
    return {
        "y_truth": pl.DataFrame({
            "time": times,
            "value": [10.0, 20.0, 30.0, 25.0, 35.0, 40.0, 38.0, 45.0, 50.0, 48.0],
        }),
        "y_pred": pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 10,
            "time": times,
            "value": [12.0, 19.0, 28.0, 27.0, 33.0, 42.0, 36.0, 43.0, 52.0, 46.0],
        }),
    }


@pytest.fixture
def multi_model_forecast_data(sample_forecast_data):
    """Create multi-model forecast data."""
    from datetime import datetime

    times = [datetime(2020, 1, 1 + i) for i in range(10)]
    y_pred_b = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * 10,
        "time": times,
        "value": [11.0, 21.0, 29.0, 26.0, 34.0, 41.0, 37.0, 44.0, 51.0, 47.0],
    })
    return {
        "y_truth": sample_forecast_data["y_truth"],
        "y_preds": {
            "Model A": sample_forecast_data["y_pred"],
            "Model B": y_pred_b,
        },
    }


class TestPlotResiduals:
    """Tests for plot_residuals function."""

    def test_basic(self, sample_residuals_data):
        """Test basic residual diagnostics plot."""
        fig = plot_residuals(sample_residuals_data["y_pred"], sample_residuals_data["y_truth"])
        assert_figure_valid(fig)
        assert_layout(fig, title="Residual Diagnostics")

    def test_explicit_column(self, sample_residuals_data):
        """Test with explicit columns parameter."""
        fig = plot_residuals(sample_residuals_data["y_pred"], sample_residuals_data["y_truth"], columns="y")
        assert_figure_valid(fig)

    def test_missing_column(self, sample_residuals_data):
        """Test error when column is missing from y_pred."""
        with pytest.raises(ValueError, match="not found"):
            plot_residuals(sample_residuals_data["y_pred"], sample_residuals_data["y_truth"], columns="nonexistent")

    def test_panel_residuals(self):
        """Test panel residuals plot with panel-formatted data."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        y_truth = pl.DataFrame({
            "time": dates,
            "y__a": [100 + i for i in range(10)],
            "y__b": [200 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "y__a": [100 + i + (i % 3) for i in range(10)],
            "y__b": [200 + i - (i % 2) for i in range(10)],
        })
        fig = plot_residuals(y_pred, y_truth, groups=["y"])
        assert_figure_valid(fig)  # At least one trace for panel facets

    def test_custom_title(self, sample_residuals_data):
        """Test custom title."""
        fig = plot_residuals(
            sample_residuals_data["y_pred"], sample_residuals_data["y_truth"], title="Custom Residuals"
        )
        assert_layout(fig, title="Custom Residuals")

    def test_custom_dimensions(self, sample_residuals_data):
        """Test custom width and height."""
        fig = plot_residuals(sample_residuals_data["y_pred"], sample_residuals_data["y_truth"], width=1200, height=800)
        assert_layout(fig, width=1200, height=800)

    def test_multi_column_produces_facets(self):
        """Test that multiple common columns produce a faceted figure."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 20), "1d", eager=True)
        y_truth = pl.DataFrame({
            "time": dates,
            "a": list(range(20)),
            "b": list(range(20, 40)),
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "a": [i + 1 for i in range(20)],
            "b": [i + 20 - 1 for i in range(20)],
        })
        fig = plot_residuals(y_pred, y_truth, columns=["a", "b"])
        assert_figure_valid(fig)
        # 2 scatter traces + 2 hline shapes (one per column facet)
        assert len(fig.data) == 2
        assert_layout(fig, title="Residual Diagnostics")

    def test_multi_column_default_auto_detects(self):
        """Test that columns=None with multiple common cols produces facets."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        y_truth = pl.DataFrame({
            "time": dates,
            "x": list(range(10)),
            "y": list(range(10, 20)),
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "x": [i + 1 for i in range(10)],
            "y": [i + 10 - 1 for i in range(10)],
        })
        # Two common columns -> faceted (not 4-panel)
        fig = plot_residuals(y_pred, y_truth)
        assert len(fig.data) == 2

    def test_facet_n_cols(self):
        """Test facet_n_cols controls the subplot grid."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        y_truth = pl.DataFrame({
            "time": dates,
            "a": list(range(10)),
            "b": list(range(10)),
            "c": list(range(10)),
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "a": [i + 1 for i in range(10)],
            "b": [i + 1 for i in range(10)],
            "c": [i + 1 for i in range(10)],
        })
        fig = plot_residuals(y_pred, y_truth, columns=["a", "b", "c"], facet_n_cols=3)
        # 3 columns in a single row → 3 traces
        assert len(fig.data) == 3

    def test_panel_with_columns_filter(self):
        """Test panel mode with columns acting as member postfix filter."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        y_truth = pl.DataFrame({
            "time": dates,
            "y__a": [100 + i for i in range(10)],
            "y__b": [200 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "y__a": [100 + i + 1 for i in range(10)],
            "y__b": [200 + i - 1 for i in range(10)],
        })
        # Filter to member "a" only
        fig = plot_residuals(y_pred, y_truth, groups=["y"], columns=["a"])
        assert_figure_valid(fig)
        # Single panel column → 4-panel diagnostics (5 traces: scatter, scatter, hist, scatter, line)
        assert len(fig.data) >= 4

    def test_single_column_gives_four_panels(self, sample_residuals_data):
        """Test that single column still produces 4-panel diagnostics layout."""
        fig = plot_residuals(sample_residuals_data["y_pred"], sample_residuals_data["y_truth"])
        # 4-panel: residual scatter, fitted scatter, histogram, QQ scatter, QQ ref line = 5
        assert len(fig.data) == 5


class TestPlotCalibration:
    """Tests for plot_calibration function."""

    def test_basic(self, sample_interval_data):
        """Test basic calibration plot."""
        fig = plot_calibration(
            sample_interval_data["y_pred_int"],
            sample_interval_data["y_truth"],
            coverage_rates=[0.9, 0.95],
        )
        assert_figure_valid(fig)
        assert len(fig.data) == 2  # empirical + reference line

    def test_single_coverage(self, sample_interval_data):
        """Test calibration with single coverage rate."""
        fig = plot_calibration(
            sample_interval_data["y_pred_int"],
            sample_interval_data["y_truth"],
            coverage_rates=[0.9],
        )
        assert len(fig.data) == 2

    def test_auto_detect_coverage_rates(self, sample_interval_data):
        """Omitting coverage_rates plots every rate present in y_pred."""
        fig = plot_calibration(
            sample_interval_data["y_pred_int"],
            sample_interval_data["y_truth"],
            type="interval",
        )
        assert len(fig.data) == 2  # empirical + reference line
        assert sorted(fig.data[0].x) == [0.9, 0.95]

    def test_quantile_type(self, sample_interval_data):
        """type='quantile' reinterprets each bound as a one-sided quantile."""
        fig = plot_calibration(
            sample_interval_data["y_pred_int"],
            sample_interval_data["y_truth"],
            type="quantile",
        )
        assert len(fig.data) == 2  # empirical + reference line
        # rates 0.9/0.95 -> quantile levels (1 -+ rate)/2, no median column here
        assert sorted(round(v, 3) for v in fig.data[0].x) == [0.025, 0.05, 0.95, 0.975]
        assert fig.layout.xaxis.title.text == "Nominal quantile level"

    def test_quantile_type_includes_median(self):
        """The median/point column contributes the 0.5 quantile level."""
        n = 200
        rng = np.random.default_rng(0)
        y_vals = rng.standard_normal(n)
        y_truth = pl.DataFrame({"y": y_vals})
        y_pred = pl.DataFrame({
            "y": y_vals,  # median / point column
            "y_lower_0.9": y_vals - 1.65,
            "y_upper_0.9": y_vals + 1.65,
        })
        fig = plot_calibration(y_pred, y_truth, type="quantile")
        assert sorted(round(v, 3) for v in fig.data[0].x) == [0.05, 0.5, 0.95]

    def test_custom_target_column(self, multi_column_interval_data):
        """Test with explicit target column name."""
        fig = plot_calibration(
            multi_column_interval_data["y_pred_int"],
            multi_column_interval_data["y_truth"],
            coverage_rates=[0.9],
            columns="value",
        )
        assert len(fig.data) == 2

    def test_missing_target_column(self, sample_interval_data):
        """Test error when target column is missing."""
        with pytest.raises(ValueError, match="not found"):
            plot_calibration(
                sample_interval_data["y_pred_int"],
                sample_interval_data["y_truth"],
                coverage_rates=[0.9],
                columns="nonexistent",
            )

    def test_missing_interval_columns(self, sample_interval_data):
        """Test error when interval columns are missing."""
        y_pred_bad = pl.DataFrame({"y": [1.0, 2.0, 3.0]})
        y_truth = pl.DataFrame({"y": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="not found"):
            plot_calibration(y_pred_bad, y_truth, coverage_rates=[0.9])

    def test_custom_title(self, sample_interval_data):
        """Test custom title."""
        fig = plot_calibration(
            sample_interval_data["y_pred_int"],
            sample_interval_data["y_truth"],
            coverage_rates=[0.9],
            title="Custom Cal",
        )
        assert_layout(fig, title="Custom Cal")

    def test_custom_labels(self, sample_interval_data):
        """Test custom axis labels."""
        fig = plot_calibration(
            sample_interval_data["y_pred_int"],
            sample_interval_data["y_truth"],
            coverage_rates=[0.9],
            x_label="Coverage",
            y_label="Error",
        )
        assert fig.layout.xaxis.title.text == "Coverage"
        assert fig.layout.yaxis.title.text == "Error"

    def test_default_title(self, sample_interval_data):
        """Test default (quantile) title is 'Quantile calibration'."""
        fig = plot_calibration(
            sample_interval_data["y_pred_int"],
            sample_interval_data["y_truth"],
            coverage_rates=[0.9],
        )
        assert_layout(fig, title="Quantile calibration")

    def test_interval_default_title(self, sample_interval_data):
        """type='interval' default title is 'Calibration plot'."""
        fig = plot_calibration(
            sample_interval_data["y_pred_int"],
            sample_interval_data["y_truth"],
            coverage_rates=[0.9],
            type="interval",
        )
        assert_layout(fig, title="Calibration plot")

    def test_custom_styling(self, sample_interval_data):
        """Test custom styling kwargs."""
        fig = plot_calibration(
            sample_interval_data["y_pred_int"],
            sample_interval_data["y_truth"],
            coverage_rates=[0.9],
            line_width=3.0,
            reference_dash="dot",
        )
        assert len(fig.data) >= 2

    def test_no_legend(self, sample_interval_data):
        """Test hiding legend."""
        fig = plot_calibration(
            sample_interval_data["y_pred_int"],
            sample_interval_data["y_truth"],
            coverage_rates=[0.9],
            show_legend=False,
        )
        assert fig.layout.showlegend is False

    def test_custom_color_palette(self, sample_interval_data):
        """Test custom color palette."""
        fig = plot_calibration(
            sample_interval_data["y_pred_int"],
            sample_interval_data["y_truth"],
            coverage_rates=[0.9],
            color_palette=["#FF0000"],
        )
        assert fig.data[0].line.color == "#FF0000"

    def test_custom_reference_color(self, sample_interval_data):
        """Test custom reference line color."""
        fig = plot_calibration(
            sample_interval_data["y_pred_int"],
            sample_interval_data["y_truth"],
            coverage_rates=[0.9],
            reference_color="#00FF00",
        )
        assert fig.data[1].line.color == "#00FF00"

    def test_custom_dimensions(self, sample_interval_data):
        """Test custom dimensions."""
        fig = plot_calibration(
            sample_interval_data["y_pred_int"],
            sample_interval_data["y_truth"],
            coverage_rates=[0.9],
            width=800,
            height=600,
        )
        assert_layout(fig, width=800, height=600)

    def test_hover_template(self, sample_interval_data):
        """Test that hover templates are set."""
        fig = plot_calibration(
            sample_interval_data["y_pred_int"],
            sample_interval_data["y_truth"],
            coverage_rates=[0.9],
        )
        assert "Empirical" in fig.data[0].hovertemplate
        assert "Perfect" in fig.data[1].hovertemplate

    def test_multi_column(self, multi_target_interval_data):
        """Test calibration with multiple target columns on one plot."""
        fig = plot_calibration(
            multi_target_interval_data["y_pred_int"],
            multi_target_interval_data["y_truth"],
            coverage_rates=[0.9],
        )
        # 2 column traces + 1 reference line = 3
        assert len(fig.data) == 3

    def test_multi_column_explicit(self, multi_target_interval_data):
        """Test multi-column calibration with explicit column list."""
        fig = plot_calibration(
            multi_target_interval_data["y_pred_int"],
            multi_target_interval_data["y_truth"],
            coverage_rates=[0.9],
            columns=["a"],
        )
        # 1 column trace + 1 reference = 2
        assert len(fig.data) == 2

    def test_panel_facets(self, panel_interval_data):
        """Test panel faceted calibration subplots."""
        fig = plot_calibration(
            panel_interval_data["y_pred_int"],
            panel_interval_data["y_truth"],
            coverage_rates=[0.9],
            groups=["x", "y"],
        )
        # 2 panels × (1 empirical + 1 reference) = 4
        assert len(fig.data) == 4

    def test_panel_member_filter(self, panel_interval_data):
        """Test panel facets with member filter."""
        fig = plot_calibration(
            panel_interval_data["y_pred_int"],
            panel_interval_data["y_truth"],
            coverage_rates=[0.9],
            groups=["x"],
            columns="a",
        )
        # 1 panel × 2 traces = 2
        assert len(fig.data) == 2


class TestPlotScoreTimeSeries:
    """Tests for plot_score_time_series function."""

    def test_basic(self, sample_forecast_data):
        """Test basic score time series plot."""
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(
            scorer,
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
        )
        assert_figure_valid(fig)
        assert len(fig.data) == 1

    def test_multi_model(self, multi_model_forecast_data):
        """Test with multiple models as dict."""
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(
            scorer,
            multi_model_forecast_data["y_truth"],
            multi_model_forecast_data["y_preds"],
        )
        assert len(fig.data) == 2

    def test_invalid_y_pred_type(self, sample_forecast_data):
        """Test error for invalid y_pred type."""
        scorer = MeanAbsoluteError()
        with pytest.raises(TypeError, match="must be pl.DataFrame or dict"):
            plot_score_time_series(
                scorer,
                sample_forecast_data["y_truth"],
                "not_a_df",
            )

    def test_custom_title(self, sample_forecast_data):
        """Test custom title."""
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(
            scorer,
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            title="Custom Title",
        )
        assert_layout(fig, title="Custom Title")

    def test_default_title(self, sample_forecast_data):
        """Test default title from scorer name."""
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(
            scorer,
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
        )
        assert "MeanAbsoluteError" in fig.layout.title.text

    def test_custom_labels(self, sample_forecast_data):
        """Test custom axis labels."""
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(
            scorer,
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            x_label="Date",
            y_label="Error",
        )
        assert fig.layout.xaxis.title.text == "Date"
        assert fig.layout.yaxis.title.text == "Error"

    def test_custom_dimensions(self, sample_forecast_data):
        """Test custom dimensions."""
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(
            scorer,
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            width=900,
            height=500,
        )
        assert_layout(fig, width=900, height=500)

    def test_styling_kwargs(self, sample_forecast_data):
        """Test styling kwargs."""
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(
            scorer,
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            line_width=3.0,
            line_dash="dash",
            show_markers=True,
        )
        assert fig.data[0].line.width == 3.0
        assert fig.data[0].line.dash == "dash"

    def test_no_legend(self, sample_forecast_data):
        """Test hiding legend."""
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(
            scorer,
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            show_legend=False,
        )
        assert fig.layout.showlegend is False

    def test_custom_color_palette(self, multi_model_forecast_data):
        """Test custom color palette."""
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(
            scorer,
            multi_model_forecast_data["y_truth"],
            multi_model_forecast_data["y_preds"],
            color_palette=["#FF0000", "#0000FF"],
        )
        assert fig.data[0].line.color == "#FF0000"
        assert fig.data[1].line.color == "#0000FF"

    def test_forecast_name_default(self, sample_forecast_data):
        """Test default forecast name for single DataFrame."""
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(
            scorer,
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
        )
        assert fig.data[0].name == "Forecast"


class TestPlotScoreDistribution:
    """Tests for plot_score_distribution function."""

    def test_histogram(self, sample_forecast_data):
        """Test histogram mode."""
        fig = plot_score_distribution(
            MeanAbsoluteError(),
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            kind="histogram",
        )
        assert_figure_valid(fig)
        assert any(isinstance(t, go.Histogram) for t in fig.data)

    def test_kde(self, sample_forecast_data):
        """Test KDE mode."""
        fig = plot_score_distribution(
            MeanAbsoluteError(),
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            kind="kde",
        )
        assert_figure_valid(fig)
        assert any(isinstance(t, go.Scatter) for t in fig.data)

    def test_both(self, sample_forecast_data):
        """Test combined histogram + KDE mode."""
        fig = plot_score_distribution(
            MeanAbsoluteError(),
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            kind="both",
        )
        assert_figure_valid(fig)
        assert any(isinstance(t, go.Histogram) for t in fig.data)
        assert any(isinstance(t, go.Scatter) for t in fig.data)

    def test_multi_model(self, multi_model_forecast_data):
        """Test multi-model overlay."""
        fig = plot_score_distribution(
            MeanAbsoluteError(),
            multi_model_forecast_data["y_truth"],
            multi_model_forecast_data["y_preds"],
        )
        assert_figure_valid(fig)
        # At least one trace per model
        assert len(fig.data) >= 2

    def test_no_mean_no_zero(self, sample_forecast_data):
        """Test with mean and zero lines disabled."""
        fig = plot_score_distribution(
            MeanAbsoluteError(),
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            show_mean=False,
            show_zero=False,
        )
        assert_figure_valid(fig)
        # No vlines should be added (no shapes)
        shapes = fig.layout.shapes or ()
        assert len(shapes) == 0

    def test_custom_bins(self, sample_forecast_data):
        """Test custom number of bins."""
        fig = plot_score_distribution(
            MeanAbsoluteError(),
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            n_bins=10,
        )
        assert_figure_valid(fig)

    def test_invalid_kind(self, sample_forecast_data):
        """Test that invalid kind raises ValueError."""
        with pytest.raises(ValueError, match="kind must be one of"):
            plot_score_distribution(
                MeanAbsoluteError(),
                sample_forecast_data["y_truth"],
                sample_forecast_data["y_pred"],
                kind="boxplot",
            )

    def test_default_title(self, sample_forecast_data):
        """Test default title uses scorer name."""
        fig = plot_score_distribution(
            MeanAbsoluteError(),
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
        )
        assert "MeanAbsoluteError" in fig.layout.title.text


class TestPlotScorePerHorizon:
    """Tests for plot_score_per_step function."""

    def test_line(self, sample_forecast_data):
        """Test line chart mode."""
        fig = plot_score_per_step(
            MeanAbsoluteError(),
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            kind="line",
        )
        assert_figure_valid(fig)
        assert isinstance(fig.data[0], go.Scatter)

    def test_bar(self, sample_forecast_data):
        """Test bar chart mode."""
        fig = plot_score_per_step(
            MeanAbsoluteError(),
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            kind="bar",
        )
        assert_figure_valid(fig)
        assert any(isinstance(t, go.Bar) for t in fig.data)

    def test_multi_model(self, multi_model_forecast_data):
        """Test multi-model comparison."""
        fig = plot_score_per_step(
            MeanAbsoluteError(),
            multi_model_forecast_data["y_truth"],
            multi_model_forecast_data["y_preds"],
        )
        assert_figure_valid(fig)
        assert len(fig.data) >= 2

    def test_show_trend(self, sample_forecast_data):
        """Test trend line overlay."""
        fig = plot_score_per_step(
            MeanAbsoluteError(),
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            show_trend=True,
        )
        assert_figure_valid(fig)
        # Should have 2 traces: score + trend
        assert len(fig.data) == 2

    def test_single_step(self):
        """Test with a single horizon step."""
        from datetime import datetime

        y_truth = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "value": [10.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "value": [12.0],
        })
        fig = plot_score_per_step(MeanAbsoluteError(), y_truth, y_pred)
        assert_figure_valid(fig)

    def test_invalid_kind(self, sample_forecast_data):
        """Test that invalid kind raises ValueError."""
        with pytest.raises(ValueError, match="kind must be one of"):
            plot_score_per_step(
                MeanAbsoluteError(),
                sample_forecast_data["y_truth"],
                sample_forecast_data["y_pred"],
                kind="scatter",
            )

    def test_default_title(self, sample_forecast_data):
        """Test default title uses scorer name."""
        fig = plot_score_per_step(
            MeanAbsoluteError(),
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
        )
        assert "MeanAbsoluteError" in fig.layout.title.text

    def test_custom_palette(self, sample_forecast_data):
        """Test custom color palette."""
        fig = plot_score_per_step(
            MeanAbsoluteError(),
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            color_palette=["#FF0000"],
        )
        assert_figure_valid(fig)

    @pytest.fixture
    def walk_forward(self):
        """Three vintages of a four-step horizon, the shape a backtest leaves."""
        horizon, n_vintages = 4, 3
        times, vintages = [], []
        for v in range(n_vintages):
            origin = datetime(2020, 1, 1) + timedelta(hours=v * horizon)
            for step in range(1, horizon + 1):
                vintages.append(origin)
                times.append(origin + timedelta(hours=step))
        y_truth = pl.DataFrame({"time": times, "value": [10.0] * len(times)})
        # error grows with the step, so a correct profile is monotonic
        y_pred = pl.DataFrame({
            "vintage_time": vintages,
            "time": times,
            "value": [10.0 + (t - v).total_seconds() / 3600 for t, v in zip(times, vintages, strict=True)],
        })
        return {"y_truth": y_truth, "y_pred": y_pred, "horizon": horizon}

    def test_axis_is_the_horizon_not_the_row_count(self, walk_forward):
        """The step axis must span the horizon, not horizon x vintages.

        Componentwise aggregation keeps one row per (time, vintage); numbering
        those rows drew every vintage's profile end to end under an axis
        labelled as a single horizon.
        """
        fig = plot_score_per_step(MeanAbsoluteError(), walk_forward["y_truth"], walk_forward["y_pred"])
        steps = list(fig.data[0].x)
        assert steps == list(range(1, walk_forward["horizon"] + 1))

    def test_each_step_averages_over_its_vintages(self, walk_forward):
        """Step k is the mean over every vintage's step k, so the profile is
        the horizon decay rather than a sawtooth."""
        fig = plot_score_per_step(MeanAbsoluteError(), walk_forward["y_truth"], walk_forward["y_pred"])
        assert list(fig.data[0].y) == [1.0, 2.0, 3.0, 4.0]


class TestPlotScoreTimeSeriesPanel:
    """Panel data tests for plot_score_time_series."""

    @pytest.fixture
    def panel_forecast(self):
        """Create panel forecast data for score time series."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(10)]
        y_truth = pl.DataFrame({
            "time": times,
            "value__a": [10.0 + i for i in range(10)],
            "value__b": [20.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 10,
            "time": times,
            "value__a": [12.0 + i for i in range(10)],
            "value__b": [19.0 + i for i in range(10)],
        })
        return {"y_truth": y_truth, "y_pred": y_pred}

    def test_panel_produces_figure(self, panel_forecast):
        """Panel forecast data produces a single aggregated score line."""
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(
            scorer,
            panel_forecast["y_truth"],
            panel_forecast["y_pred"],
            groups=["value"],
        )
        assert_figure_valid(fig)
        # The panel score is aggregated across members into one line trace.
        assert len(fig.data) == 1

    def test_panel_with_step_and_vintage_weight(self, panel_forecast):
        """Panel score time series forwards step_weight and vintage_weight."""
        scorer = MeanAbsoluteError(
            step_weighter=LookupWeighter(mapping={}, default=1.0),
            vintage_weighter=LookupWeighter(mapping={}, default=1.0),
        )
        fig = plot_score_time_series(
            scorer,
            panel_forecast["y_truth"],
            panel_forecast["y_pred"],
            groups=["value"],
        )
        assert_figure_valid(fig)


class TestPlotScoreDistributionPanel:
    """Panel data tests for plot_score_distribution."""

    @pytest.fixture
    def panel_forecast(self):
        """Create panel forecast data for score distribution."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(20)]
        y_truth = pl.DataFrame({
            "time": times,
            "value__a": [10.0 + i for i in range(20)],
            "value__b": [20.0 + i for i in range(20)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 20,
            "time": times,
            "value__a": [12.0 + i for i in range(20)],
            "value__b": [19.0 + i for i in range(20)],
        })
        return {"y_truth": y_truth, "y_pred": y_pred}

    def test_panel_produces_figure(self, panel_forecast):
        """Panel data produces a valid score distribution figure."""
        scorer = MeanAbsoluteError()
        fig = plot_score_distribution(
            scorer,
            panel_forecast["y_truth"],
            panel_forecast["y_pred"],
            groups=["value"],
        )
        assert_figure_valid(fig)


class TestPlotScorePerHorizonPanel:
    """Panel data tests for plot_score_per_step."""

    @pytest.fixture
    def panel_forecast(self):
        """Create panel forecast data for score per horizon."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(10)]
        y_truth = pl.DataFrame({
            "time": times,
            "value__a": [10.0 + i for i in range(10)],
            "value__b": [20.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 10,
            "time": times,
            "value__a": [12.0 + i for i in range(10)],
            "value__b": [19.0 + i for i in range(10)],
        })
        return {"y_truth": y_truth, "y_pred": y_pred}

    def test_panel_produces_figure(self, panel_forecast):
        """Panel data produces a valid score per horizon figure."""
        scorer = MeanAbsoluteError()
        fig = plot_score_per_step(
            scorer,
            panel_forecast["y_truth"],
            panel_forecast["y_pred"],
            groups=["value"],
        )
        assert_figure_valid(fig)


class TestPlotResidualsColumnResolution:
    """Tests for plot_residuals column resolution paths."""

    @pytest.fixture
    def standard_data(self):
        """Create single-column residual data."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        y_truth = pl.DataFrame({"time": dates, "y": [100 + i for i in range(91)]})
        y_pred = pl.DataFrame({"time": dates, "y": [102 + i for i in range(91)]})
        return y_truth, y_pred

    def test_explicit_columns_parameter(self, standard_data):
        """Passing columns explicitly resolves target columns."""
        y_truth, y_pred = standard_data
        fig = plot_residuals(y_pred, y_truth, columns="y")
        assert_figure_valid(fig)

    def test_no_common_columns_error(self):
        """Mismatched DataFrames with no common columns raise ValueError."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        y_truth = pl.DataFrame({"time": dates, "a": list(range(10))})
        y_pred = pl.DataFrame({"time": dates, "b": list(range(10))})
        with pytest.raises(ValueError, match="No common non-time columns"):
            plot_residuals(y_pred, y_truth)

    def test_column_not_in_pred_error(self):
        """Requesting a column absent from y_pred raises ValueError."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        y_truth = pl.DataFrame({"time": dates, "a": list(range(10))})
        y_pred = pl.DataFrame({"time": dates, "b": list(range(10))})
        with pytest.raises(ValueError, match="not found in y_pred"):
            plot_residuals(y_pred, y_truth, columns="a")

    def test_column_not_in_truth_error(self):
        """Requesting a column absent from y_truth raises ValueError."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        y_truth = pl.DataFrame({"time": dates, "a": list(range(10))})
        y_pred = pl.DataFrame({"time": dates, "a": list(range(10)), "b": list(range(10))})
        with pytest.raises(ValueError, match="not found in y_truth"):
            plot_residuals(y_pred, y_truth, columns="b")

    def test_groups(self):
        """Panel group names resolve panel columns correctly."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        y_truth = pl.DataFrame({
            "time": dates,
            "y__a": [100 + i for i in range(91)],
            "y__b": [200 + i for i in range(91)],
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "y__a": [102 + i for i in range(91)],
            "y__b": [198 + i for i in range(91)],
        })
        fig = plot_residuals(y_pred, y_truth, groups=["y"])
        assert_figure_valid(fig)


class TestPlotCalibrationNonPanel:
    """Tests for plot_calibration non-panel column resolution."""

    def test_auto_detect_target_columns(self):
        """Non-panel calibration auto-detects target columns from y_truth."""
        n = 50
        rng = np.random.default_rng(42)
        y_vals = rng.standard_normal(n)
        y_truth = pl.DataFrame({"y": y_vals})
        y_pred_int = pl.DataFrame({
            "y_upper_0.9": y_vals + 1.65,
            "y_lower_0.9": y_vals - 1.65,
        })
        fig = plot_calibration(y_pred_int, y_truth, coverage_rates=[0.9])
        assert_figure_valid(fig)
        assert len(fig.data) >= 2

    def test_no_target_columns_error(self):
        """y_truth with only time column raises ValueError."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 5), "1d", eager=True)
        y_truth = pl.DataFrame({"time": dates})
        y_pred_int = pl.DataFrame({
            "y_upper_0.9": [1.0] * 5,
            "y_lower_0.9": [-1.0] * 5,
        })
        with pytest.raises(ValueError, match="no non-time columns"):
            plot_calibration(y_pred_int, y_truth, coverage_rates=[0.9])


class TestPlotScoreTimeSeriesNonPanel:
    """Tests for plot_score_time_series non-panel execution path."""

    @pytest.fixture
    def simple_data(self):
        """Create simple single-column truth/pred pair."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        y_truth = pl.DataFrame({"time": dates, "y": [100 + i for i in range(91)]})
        y_pred = pl.DataFrame({"time": dates, "y": [102 + i for i in range(91)]})
        return y_truth, y_pred

    def test_single_model(self, simple_data):
        """Non-panel single model produces valid score time series."""
        y_truth, y_pred = simple_data
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(scorer, y_truth, y_pred)
        assert_figure_valid(fig)

    def test_multi_model_dict(self, simple_data):
        """Multi-model dict produces traces for each model."""
        y_truth, y_pred = simple_data
        y_pred2 = y_pred.with_columns((pl.col("y") + 1).alias("y"))
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(scorer, y_truth, {"M1": y_pred, "M2": y_pred2})
        assert_figure_valid(fig)
        names = [t.name for t in fig.data if t.name is not None]
        assert "M1" in names

    def test_non_panel_with_step_and_vintage_weight(self, simple_data):
        """Non-panel score time series forwards step_weight and vintage_weight."""
        y_truth, y_pred = simple_data
        scorer = MeanAbsoluteError(
            step_weighter=LookupWeighter(mapping={}, default=1.0),
            vintage_weighter=LookupWeighter(mapping={}, default=1.0),
        )
        fig = plot_score_time_series(
            scorer,
            y_truth,
            y_pred,
        )
        assert_figure_valid(fig)


class TestPlotScoreTimeSeriesMultiComponent:
    """Tests for plot_score_time_series with multi-column (multi-component) data."""

    @pytest.fixture
    def multi_col_data(self):
        """Create multi-column truth/pred pair for mean aggregation path."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(10)]
        y_truth = pl.DataFrame({
            "time": times,
            "a": [10.0 + i for i in range(10)],
            "b": [20.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 10,
            "time": times,
            "a": [12.0 + i for i in range(10)],
            "b": [19.0 + i for i in range(10)],
        })
        return y_truth, y_pred

    def test_multi_component_aggregation(self, multi_col_data):
        """Multi-column data triggers mean aggregation across components."""
        y_truth, y_pred = multi_col_data
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(scorer, y_truth, y_pred)
        assert_figure_valid(fig)

    def test_multi_component_multi_model(self, multi_col_data):
        """Multi-column data with multiple models."""
        y_truth, y_pred = multi_col_data
        y_pred2 = y_pred.with_columns(
            (pl.col("a") + 1).alias("a"),
            (pl.col("b") + 1).alias("b"),
        )
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(scorer, y_truth, {"M1": y_pred, "M2": y_pred2})
        assert_figure_valid(fig)
        assert len(fig.data) >= 2

    def test_show_markers(self, multi_col_data):
        """show_markers=True adds markers to traces."""
        y_truth, y_pred = multi_col_data
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(scorer, y_truth, y_pred, show_markers=True)
        assert_figure_valid(fig)
        assert fig.data[0].mode == "lines+markers"


class TestPlotScoreTimeSeriesAutoPanel:
    """Tests for plot_score_time_series auto-detecting panel data."""

    def test_auto_panel_detection(self):
        """Passing panel data without groups auto-detects."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        y_truth = pl.DataFrame({
            "time": dates,
            "value__a": [100 + i for i in range(91)],
            "value__b": [200 + i for i in range(91)],
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "value__a": [102 + i for i in range(91)],
            "value__b": [198 + i for i in range(91)],
        })
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(scorer, y_truth, y_pred)
        assert_figure_valid(fig)


class TestPlotResidualsAutoPanel:
    """Tests for plot_residuals panel resolution paths."""

    def test_panel_explicit_empty_groups_error(self):
        """Passing empty groups raises ValueError."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 20), "1d", eager=True)
        y_truth = pl.DataFrame({
            "time": dates,
            "y__a": [100.0 + i for i in range(20)],
            "y__b": [200.0 + i for i in range(20)],
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "y__a": [102.0 + i for i in range(20)],
            "y__b": [198.0 + i for i in range(20)],
        })
        with pytest.raises(ValueError, match="panel columns"):
            plot_residuals(y_pred, y_truth, groups=[])

    def test_panel_with_columns(self):
        """Panel data with explicit columns parameter."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 20), "1d", eager=True)
        y_truth = pl.DataFrame({
            "time": dates,
            "y__a": [100.0 + i for i in range(20)],
            "y__b": [200.0 + i for i in range(20)],
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "y__a": [102.0 + i for i in range(20)],
            "y__b": [198.0 + i for i in range(20)],
        })
        fig = plot_residuals(y_pred, y_truth, columns=["y__a", "y__b"])
        assert_figure_valid(fig)


class TestPlotScorePerHorizonBarMultiModel:
    """Tests for plot_score_per_step bar mode with multiple models."""

    def test_bar_multi_model_grouped(self):
        """Bar mode with multiple models sets barmode=group."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(10)]
        y_truth = pl.DataFrame({
            "time": times,
            "value": [10.0 + i for i in range(10)],
        })
        y_pred_a = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 10,
            "time": times,
            "value": [12.0 + i for i in range(10)],
        })
        y_pred_b = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 10,
            "time": times,
            "value": [11.0 + i for i in range(10)],
        })
        scorer = MeanAbsoluteError()
        fig = plot_score_per_step(
            scorer,
            y_truth,
            {"A": y_pred_a, "B": y_pred_b},
            kind="bar",
        )
        assert_figure_valid(fig)
        assert len(fig.data) >= 2
        assert fig.layout.barmode == "group"

    def test_multi_component_bar(self):
        """Multi-column data with bar kind uses horizontal mean."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(10)]
        y_truth = pl.DataFrame({
            "time": times,
            "a": [10.0 + i for i in range(10)],
            "b": [20.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 10,
            "time": times,
            "a": [12.0 + i for i in range(10)],
            "b": [19.0 + i for i in range(10)],
        })
        scorer = MeanAbsoluteError()
        fig = plot_score_per_step(scorer, y_truth, y_pred, kind="bar")
        assert_figure_valid(fig)
        assert any(isinstance(t, go.Bar) for t in fig.data)


class TestPlotScoreDistributionEdgeCases:
    """Edge case tests for plot_score_distribution."""

    def test_multi_component_distribution(self):
        """Multi-column data flattens scores before histogram/kde."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(20)]
        y_truth = pl.DataFrame({
            "time": times,
            "a": [10.0 + i for i in range(20)],
            "b": [20.0 + i for i in range(20)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 20,
            "time": times,
            "a": [12.0 + i for i in range(20)],
            "b": [19.0 + i for i in range(20)],
        })
        scorer = MeanAbsoluteError()
        fig = plot_score_distribution(scorer, y_truth, y_pred, kind="both")
        assert_figure_valid(fig)
        assert any(isinstance(t, go.Histogram) for t in fig.data)
        assert any(isinstance(t, go.Scatter) for t in fig.data)

    def test_distribution_with_mean_line(self):
        """show_mean=True adds vertical mean line."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(15)]
        y_truth = pl.DataFrame({
            "time": times,
            "value": [10.0 + i for i in range(15)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 15,
            "time": times,
            "value": [12.0 + i for i in range(15)],
        })
        scorer = MeanAbsoluteError()
        fig = plot_score_distribution(scorer, y_truth, y_pred, show_mean=True, kind="both")
        assert_figure_valid(fig)
        shapes = fig.layout.shapes or ()
        assert len(shapes) >= 1


class TestPlotResidualsExplicitColumns:
    """Tests for plot_residuals with explicit columns parameter (L286-L294)."""

    def test_explicit_columns_selects_subset(self):
        """Passing columns= selects only the requested columns."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 20), "1d", eager=True)
        y_truth = pl.DataFrame({
            "time": dates,
            "a": [100.0 + i for i in range(20)],
            "b": [200.0 + i for i in range(20)],
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "a": [102.0 + i for i in range(20)],
            "b": [198.0 + i for i in range(20)],
        })
        fig = plot_residuals(y_pred, y_truth, columns=["a"])
        assert_figure_valid(fig)

    def test_auto_detect_panel_residuals(self):
        """Panel data without explicit groups auto-detects and raises."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 20), "1d", eager=True)
        y_truth = pl.DataFrame({
            "time": dates,
            "val__g1": [100.0 + i for i in range(20)],
            "val__g2": [200.0 + i for i in range(20)],
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "val__g1": [102.0 + i for i in range(20)],
            "val__g2": [198.0 + i for i in range(20)],
        })
        with pytest.raises(ValueError, match="panel columns"):
            plot_residuals(y_pred, y_truth)


class TestPlotCalibrationAutoPanel:
    """Tests for plot_calibration auto-detecting panel data (L612)."""

    def test_auto_detect_panel_calibration(self):
        """Panel data without groups triggers auto-detect branch."""
        n = 30
        rng = np.random.default_rng(42)
        xa = rng.standard_normal(n) * 10 + 100
        ya = rng.standard_normal(n) * 10 + 200
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 30), "1d", eager=True)
        y_truth = pl.DataFrame({
            "time": dates,
            "val__g1": xa,
            "val__g2": ya,
        })
        y_pred_int = pl.DataFrame({
            "time": dates,
            "val__g1_upper_0.9": xa + 5,
            "val__g1_lower_0.9": xa - 5,
            "val__g2_upper_0.9": ya + 5,
            "val__g2_lower_0.9": ya - 5,
        })
        fig = plot_calibration(y_pred_int, y_truth, coverage_rates=[0.9])
        assert_figure_valid(fig)


class TestPlotScoreTimeSeriesTimeWeight:
    """Tests for plot_score_time_series with time_weight parameter (L1114)."""

    def test_callable_time_weight(self):
        """Passing time_weight as a callable forwards it to scorer."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        y_truth = pl.DataFrame({"time": dates, "y": [100.0 + i for i in range(91)]})
        y_pred = pl.DataFrame({"time": dates, "y": [102.0 + i for i in range(91)]})
        scorer = MeanAbsoluteError(time_weighter=LinearDecayWeighter())
        fig = plot_score_time_series(scorer, y_truth, y_pred)
        assert_figure_valid(fig)

    def test_dataframe_time_weight(self):
        """Passing time_weight as a DataFrame forwards it to scorer."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        y_truth = pl.DataFrame({"time": dates, "y": [100.0 + i for i in range(91)]})
        y_pred = pl.DataFrame({"time": dates, "y": [102.0 + i for i in range(91)]})
        tw = pl.DataFrame({"time": dates, "weight": [1.0] * 91})
        scorer = MeanAbsoluteError(time_weighter=TableWeighter(frame=tw, on="time"))
        fig = plot_score_time_series(scorer, y_truth, y_pred)
        assert_figure_valid(fig)


class TestPlotScoreTimeSeriesIntervalScorer:
    """Tests for plot_score_time_series with interval scorer isinstance branch (L1038)."""

    def test_interval_scorer_sets_coveragewise(self):
        """IntervalScorer triggers componentwise+coveragewise aggregation."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 30), "1d", eager=True)
        n = len(dates)
        rng = np.random.default_rng(42)
        vals = rng.standard_normal(n) * 10 + 100
        y_truth = pl.DataFrame({"time": dates, "y": vals})
        y_pred = pl.DataFrame({
            "time": dates,
            "y_upper_0.9": vals + 5,
            "y_lower_0.9": vals - 5,
        })
        scorer = IntervalScore(coverage_rates=[0.9])
        fig = plot_score_time_series(scorer, y_truth, y_pred)
        assert_figure_valid(fig)


class TestPlotScoreDistributionIntervalScorer:
    """Tests for plot_score_distribution with interval scorer (L1466)."""

    def test_interval_scorer_distribution(self):
        """IntervalScorer with plot_score_distribution triggers coveragewise branch."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 30), "1d", eager=True)
        n = len(dates)
        rng = np.random.default_rng(42)
        vals = rng.standard_normal(n) * 10 + 100
        y_truth = pl.DataFrame({"time": dates, "y": vals})
        y_pred = pl.DataFrame({
            "time": dates,
            "y_upper_0.9": vals + 5,
            "y_lower_0.9": vals - 5,
        })
        scorer = IntervalScore(coverage_rates=[0.9])
        fig = plot_score_distribution(scorer, y_truth, y_pred)
        assert_figure_valid(fig)


class TestPlotScorePerHorizonIntervalScorer:
    """Tests for plot_score_per_step with interval scorer (L1716)."""

    def test_interval_scorer_per_horizon(self):
        """IntervalScorer with plot_score_per_step triggers coveragewise branch."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(20)]
        rng = np.random.default_rng(42)
        vals = rng.standard_normal(20) * 10 + 100
        y_truth = pl.DataFrame({"time": times, "y": vals})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 20,
            "time": times,
            "y_upper_0.9": vals + 5,
            "y_lower_0.9": vals - 5,
        })
        scorer = IntervalScore(coverage_rates=[0.9])
        fig = plot_score_per_step(scorer, y_truth, y_pred)
        assert_figure_valid(fig)


class TestPlotScorePerHorizonShowTrend:
    """Tests for plot_score_per_step show_trend branch (L1799-L1809)."""

    def test_trend_line_overlay(self):
        """show_trend=True adds a linear trend trace."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(10)]
        y_truth = pl.DataFrame({
            "time": times,
            "value": [10.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 10,
            "time": times,
            "value": [12.0 + i for i in range(10)],
        })
        scorer = MeanAbsoluteError()
        fig = plot_score_per_step(scorer, y_truth, y_pred, show_trend=True)
        assert_figure_valid(fig)
        trace_names = [t.name for t in fig.data if t.name is not None]
        assert any("trend" in name.lower() for name in trace_names)


class TestPlotScoreDistributionKDEOnly:
    """Tests for plot_score_distribution kind='kde' branch (L1519-L1537)."""

    def test_kde_only(self):
        """kind='kde' produces scatter line traces."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(20)]
        y_truth = pl.DataFrame({"time": times, "y": [10.0 + i * 0.5 for i in range(20)]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 20,
            "time": times,
            "y": [12.0 + i * 0.3 for i in range(20)],
        })
        scorer = MeanAbsoluteError()
        fig = plot_score_distribution(scorer, y_truth, y_pred, kind="kde")
        assert_figure_valid(fig)

    def test_show_mean_false(self):
        """show_mean=False omits the mean vertical line but keeps zero line."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(20)]
        y_truth = pl.DataFrame({"time": times, "y": [10.0 + i for i in range(20)]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 20,
            "time": times,
            "y": [12.0 + i for i in range(20)],
        })
        scorer = MeanAbsoluteError()
        fig_no_mean = plot_score_distribution(
            scorer,
            y_truth,
            y_pred,
            kind="histogram",
            show_mean=False,
        )
        fig_mean = plot_score_distribution(
            scorer,
            y_truth,
            y_pred,
            kind="histogram",
            show_mean=True,
        )
        assert isinstance(fig_no_mean, go.Figure)
        shapes_no = len(fig_no_mean.layout.shapes or ())
        shapes_yes = len(fig_mean.layout.shapes or ())
        assert shapes_no < shapes_yes

    def test_show_zero_false(self):
        """show_zero=False omits the zero reference line."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(20)]
        y_truth = pl.DataFrame({"time": times, "y": [10.0 + i for i in range(20)]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 20,
            "time": times,
            "y": [12.0 + i for i in range(20)],
        })
        scorer = MeanAbsoluteError()
        fig_no_zero = plot_score_distribution(
            scorer,
            y_truth,
            y_pred,
            kind="histogram",
            show_zero=False,
        )
        fig_zero = plot_score_distribution(
            scorer,
            y_truth,
            y_pred,
            kind="histogram",
            show_zero=True,
        )
        assert isinstance(fig_no_zero, go.Figure)
        shapes_no = fig_no_zero.layout.shapes or ()
        shapes_yes = fig_zero.layout.shapes or ()
        assert len(shapes_no) < len(shapes_yes)


class TestPlotResidualsColumnsParam:
    """Tests for plot_residuals with explicit columns parameter."""

    def test_residuals_columns_string(self):
        """plot_residuals with columns as string selects that column for residuals."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(10)]
        y_truth = pl.DataFrame({
            "time": times,
            "a": [10.0 + i for i in range(10)],
            "b": [20.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 10,
            "time": times,
            "a": [11.0 + i for i in range(10)],
            "b": [19.0 + i for i in range(10)],
        })
        fig = plot_residuals(y_pred, y_truth, columns="a")
        assert_figure_valid(fig)
        # Residuals are y_truth - y_pred; column "a" gives (10 + i) - (11 + i) = -1.
        residual_trace = next(t for t in fig.data if t.name == "Residuals")
        assert {round(float(v), 6) for v in residual_trace.y} == {-1.0}


class TestPlotScoreTimeSeriesPanelGroupScoreCols:
    """Tests for panel score time series with single and multiple group score columns."""

    def test_panel_single_group_score_col(self):
        """Panel score time series with exactly one score column per group."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(5)]
        y_truth = pl.DataFrame({
            "time": times,
            "sales__store_1": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": times,
            "sales__store_1": [11.0, 21.0, 31.0, 41.0, 51.0],
        })
        scorer = MeanAbsoluteError(aggregation_method="componentwise")
        fig = plot_score_time_series(
            scorer,
            y_truth,
            {"model": y_pred},
            groups=["sales"],
        )
        assert_figure_valid(fig)

    def test_panel_multi_group_score_cols(self):
        """Panel score time series with multiple score columns per group averages them."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(5)]
        y_truth = pl.DataFrame({
            "time": times,
            "sales__store_1": [10.0, 20.0, 30.0, 40.0, 50.0],
            "sales__store_2": [15.0, 25.0, 35.0, 45.0, 55.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": times,
            "sales__store_1": [11.0, 21.0, 31.0, 41.0, 51.0],
            "sales__store_2": [16.0, 26.0, 36.0, 46.0, 56.0],
        })
        scorer = MeanAbsoluteError(aggregation_method="componentwise")
        fig = plot_score_time_series(
            scorer,
            y_truth,
            {"model": y_pred},
            groups=["sales"],
        )
        assert_figure_valid(fig)


class TestPanelLegendDedupResiduals:
    """Verify panel residuals produce no duplicate legend entries."""

    def test_panel_residuals_legend_dedup(self):
        """Each member appears at most once in legend for panel residuals."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        y_truth = pl.DataFrame({
            "time": dates,
            "g1__a": [100.0 + i for i in range(10)],
            "g1__b": [200.0 + i for i in range(10)],
            "g2__a": [150.0 + i for i in range(10)],
            "g2__b": [250.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "g1__a": [101.0 + i for i in range(10)],
            "g1__b": [199.0 + i for i in range(10)],
            "g2__a": [151.0 + i for i in range(10)],
            "g2__b": [249.0 + i for i in range(10)],
        })
        fig = plot_residuals(y_pred, y_truth, groups=["g1", "g2"])
        names = visible_legend_names(fig)
        assert len(names) == len(set(names)), f"Duplicate legend entries: {names}"


class TestPlotResidualsPanelMissingTruthCols:
    """Cover plot_residuals panel path where y_truth is missing columns (lines 308-309)."""

    def test_panel_column_missing_from_y_truth(self):
        """Panel columns validated from y_pred but absent in y_truth raises ValueError."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 5), "1d", eager=True)
        y_truth = pl.DataFrame({
            "time": dates,
            "y__a": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "y__a": [1.1, 2.1, 3.1, 4.1, 5.1],
            "y__b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        with pytest.raises(ValueError, match="not found in y_truth"):
            plot_residuals(y_pred, y_truth, groups=["y"])


class TestPlotCalibrationAutoDetectPanel:
    """Cover plot_calibration auto-detect panel path (lines 628-632)."""

    def test_auto_detect_panel_calibration(self):
        """Panel-formatted y_truth auto-detects panel groups without explicit groups."""
        n = 30
        rng = np.random.default_rng(42)
        xa = rng.standard_normal(n)
        ya = rng.standard_normal(n) + 3
        y_truth = pl.DataFrame({"x__a": xa, "y__a": ya})
        y_pred_int = pl.DataFrame({
            "x__a_upper_0.9": xa + 1.65,
            "x__a_lower_0.9": xa - 1.65,
            "y__a_upper_0.9": ya + 1.65,
            "y__a_lower_0.9": ya - 1.65,
        })
        fig = plot_calibration(y_pred_int, y_truth, coverage_rates=[0.9])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestPlotScoreTimeSeriesPanelColumnFilter:
    """Cover _plot_score_time_series_panel column filtering paths (lines 851-902)."""

    @pytest.fixture
    def panel_multi_member(self):
        """Panel data with two groups, each having two members."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(8)]
        y_truth = pl.DataFrame({
            "time": times,
            "sales__store_1": [10.0 + i for i in range(8)],
            "sales__store_2": [20.0 + i for i in range(8)],
            "revenue__store_1": [100.0 + i for i in range(8)],
            "revenue__store_2": [200.0 + i for i in range(8)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 8,
            "time": times,
            "sales__store_1": [11.0 + i for i in range(8)],
            "sales__store_2": [21.0 + i for i in range(8)],
            "revenue__store_1": [101.0 + i for i in range(8)],
            "revenue__store_2": [201.0 + i for i in range(8)],
        })
        return y_truth, y_pred

    def test_panel_with_columns_filter_match(self, panel_multi_member):
        """Panel score time series with columns filter matching scorer output."""
        y_truth, y_pred = panel_multi_member
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(
            scorer,
            y_truth,
            y_pred,
            groups=["sales", "revenue"],
            columns="mae",
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_panel_with_columns_filter_skip(self, panel_multi_member):
        """Panel column filter that matches no scorer cols skips groups."""
        y_truth, y_pred = panel_multi_member
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(
            scorer,
            y_truth,
            y_pred,
            groups=["sales"],
            columns="nonexistent",
        )
        assert isinstance(fig, go.Figure)
        # All groups skipped via continue → 0 traces
        assert len(fig.data) == 0

    def test_panel_multi_model_with_columns(self, panel_multi_member):
        """Panel score time series with multi-model dict and columns filter."""
        y_truth, y_pred = panel_multi_member
        y_pred2 = y_pred.with_columns(
            (pl.col("sales__store_1") + 2).alias("sales__store_1"),
            (pl.col("sales__store_2") + 2).alias("sales__store_2"),
        )
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(
            scorer,
            y_truth,
            {"M1": y_pred, "M2": y_pred2},
            groups=["sales"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_panel_single_group_score_col(self):
        """Panel with single member per group covers single-column path."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(5)]
        y_truth = pl.DataFrame({
            "time": times,
            "grp__x": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": times,
            "grp__x": [11.0, 21.0, 31.0, 41.0, 51.0],
        })
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(
            scorer,
            y_truth,
            y_pred,
            groups=["grp"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestPlotScoreTimeSeriesNonPanelColumns:
    """Cover plot_score_time_series non-panel column filtering (lines 1155-1179)."""

    @pytest.fixture
    def multi_col_data(self):
        """Multi-column truth/pred data."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(10)]
        y_truth = pl.DataFrame({
            "time": times,
            "a": [10.0 + i for i in range(10)],
            "b": [20.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 10,
            "time": times,
            "a": [11.0 + i for i in range(10)],
            "b": [21.0 + i for i in range(10)],
        })
        return y_truth, y_pred

    def test_columns_filter_single(self, multi_col_data):
        """Filtering to scorer output column in non-panel mode."""
        y_truth, y_pred = multi_col_data
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(scorer, y_truth, y_pred, columns="mae")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_columns_filter_nonexistent(self, multi_col_data):
        """Non-existent column filter raises ValueError."""
        y_truth, y_pred = multi_col_data
        scorer = MeanAbsoluteError()
        with pytest.raises(ValueError, match="None of the requested columns"):
            plot_score_time_series(scorer, y_truth, y_pred, columns="nonexistent")

    def test_multi_column_aggregation(self, multi_col_data):
        """Multi-column data without filter aggregates across columns."""
        y_truth, y_pred = multi_col_data
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(scorer, y_truth, y_pred)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestPlotScoreDistributionMultiColumn:
    """Cover plot_score_distribution multi-column paths (lines 1560-1568, 1647-1654)."""

    @pytest.fixture
    def multi_col_data(self):
        """Multi-column data for score distribution."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(20)]
        y_truth = pl.DataFrame({
            "time": times,
            "a": [10.0 + i for i in range(20)],
            "b": [20.0 + i for i in range(20)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 20,
            "time": times,
            "a": [11.0 + i for i in range(20)],
            "b": [21.0 + i for i in range(20)],
        })
        return y_truth, y_pred

    def test_multi_column_scores(self, multi_col_data):
        """Multi-column data flattens scores across columns."""
        y_truth, y_pred = multi_col_data
        fig = plot_score_distribution(MeanAbsoluteError(), y_truth, y_pred)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_columns_filter(self, multi_col_data):
        """Non-panel with columns filter selects specific columns."""
        y_truth, y_pred = multi_col_data
        fig = plot_score_distribution(MeanAbsoluteError(), y_truth, y_pred, columns="a")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_columns_filter_list(self, multi_col_data):
        """Non-panel with columns filter as list."""
        y_truth, y_pred = multi_col_data
        fig = plot_score_distribution(MeanAbsoluteError(), y_truth, y_pred, columns=["a", "b"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestPlotScoreDistributionPanelAutoDetect:
    """Cover plot_score_distribution panel auto-detect (lines 1688-1691)."""

    def test_auto_detect_panel(self):
        """Panel-formatted data auto-detects panel groups."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(15)]
        y_truth = pl.DataFrame({
            "time": times,
            "grp__x": [10.0 + i for i in range(15)],
            "grp__y": [20.0 + i for i in range(15)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 15,
            "time": times,
            "grp__x": [11.0 + i for i in range(15)],
            "grp__y": [21.0 + i for i in range(15)],
        })
        fig = plot_score_distribution(MeanAbsoluteError(), y_truth, y_pred)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_panel_with_columns_filter(self):
        """Panel data with columns filter selects specific members."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(15)]
        y_truth = pl.DataFrame({
            "time": times,
            "grp__x": [10.0 + i for i in range(15)],
            "grp__y": [20.0 + i for i in range(15)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 15,
            "time": times,
            "grp__x": [11.0 + i for i in range(15)],
            "grp__y": [21.0 + i for i in range(15)],
        })
        fig = plot_score_distribution(
            MeanAbsoluteError(),
            y_truth,
            y_pred,
            groups=["grp"],
            columns="x",
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestPlotScorePerHorizonMultiColumn:
    """Cover plot_score_per_step multi-column paths (lines 1858-1867)."""

    @pytest.fixture
    def multi_col_data(self):
        """Multi-column data for per-horizon scoring."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(10)]
        y_truth = pl.DataFrame({
            "time": times,
            "a": [10.0 + i for i in range(10)],
            "b": [20.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 10,
            "time": times,
            "a": [11.0 + i for i in range(10)],
            "b": [21.0 + i for i in range(10)],
        })
        return y_truth, y_pred

    def test_multi_column_scores(self, multi_col_data):
        """Multi-column data averages scores across columns per horizon step."""
        y_truth, y_pred = multi_col_data
        fig = plot_score_per_step(MeanAbsoluteError(), y_truth, y_pred)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_columns_filter(self, multi_col_data):
        """Non-panel with columns filter selects specific column."""
        y_truth, y_pred = multi_col_data
        fig = plot_score_per_step(MeanAbsoluteError(), y_truth, y_pred, columns="a")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_columns_filter_list(self, multi_col_data):
        """Non-panel with columns filter as list."""
        y_truth, y_pred = multi_col_data
        fig = plot_score_per_step(MeanAbsoluteError(), y_truth, y_pred, columns=["a", "b"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestPlotScorePerHorizonPanelAutoDetect:
    """Cover plot_score_per_step panel auto-detect and column filter (lines 1917-1963)."""

    def test_auto_detect_panel(self):
        """Panel-formatted data auto-detects panel groups for per-horizon."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(10)]
        y_truth = pl.DataFrame({
            "time": times,
            "grp__x": [10.0 + i for i in range(10)],
            "grp__y": [20.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 10,
            "time": times,
            "grp__x": [11.0 + i for i in range(10)],
            "grp__y": [21.0 + i for i in range(10)],
        })
        fig = plot_score_per_step(MeanAbsoluteError(), y_truth, y_pred)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_panel_with_columns_filter(self):
        """Panel data with columns filter selects specific members."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(10)]
        y_truth = pl.DataFrame({
            "time": times,
            "grp__x": [10.0 + i for i in range(10)],
            "grp__y": [20.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 10,
            "time": times,
            "grp__x": [11.0 + i for i in range(10)],
            "grp__y": [21.0 + i for i in range(10)],
        })
        fig = plot_score_per_step(
            MeanAbsoluteError(),
            y_truth,
            y_pred,
            groups=["grp"],
            columns="x",
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_non_panel_columns_filter(self):
        """Non-panel columns filter path."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(10)]
        y_truth = pl.DataFrame({
            "time": times,
            "val1": [10.0 + i for i in range(10)],
            "val2": [20.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 10,
            "time": times,
            "val1": [11.0 + i for i in range(10)],
            "val2": [21.0 + i for i in range(10)],
        })
        fig = plot_score_per_step(MeanAbsoluteError(), y_truth, y_pred, columns="val1")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestPlotCalibrationTypeErrors:
    """Cover plot_calibration type validation paths (lines 628-632)."""

    def test_y_truth_not_dataframe(self):
        """Passing non-DataFrame y_truth raises TypeError."""
        y_pred_int = pl.DataFrame({"y_upper_0.9": [1.0], "y_lower_0.9": [-1.0]})
        with pytest.raises(TypeError, match="Expected pl.DataFrame for y_truth"):
            plot_calibration(y_pred_int, "not_a_df", coverage_rates=[0.9])

    def test_y_pred_int_not_dataframe(self):
        """Passing non-DataFrame y_pred raises TypeError."""
        y_truth = pl.DataFrame({"y": [1.0]})
        with pytest.raises(TypeError, match="Expected pl.DataFrame for y_pred"):
            plot_calibration("not_a_df", y_truth, coverage_rates=[0.9])


class TestPlotScoreTimeSeriesPanelEdgeCases:
    """Cover _plot_score_time_series_panel edge cases (lines 851-852, 870)."""

    def test_no_matching_panel_groups(self):
        """Panel group names that don't match any detected groups raise ValueError."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(5)]
        y_truth = pl.DataFrame({
            "time": times,
            "grp__x": [10.0 + i for i in range(5)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": times,
            "grp__x": [11.0 + i for i in range(5)],
        })
        scorer = MeanAbsoluteError()
        with pytest.raises(ValueError, match="No panel groups found"):
            plot_score_time_series(scorer, y_truth, y_pred, groups=["nonexistent"])

    def test_panel_with_time_weight(self):
        """Panel score time series with time_weight parameter."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(5)]
        y_truth = pl.DataFrame({
            "time": times,
            "grp__x": [10.0 + i for i in range(5)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": times,
            "grp__x": [11.0 + i for i in range(5)],
        })
        # time_weight as a DataFrame with uniform weights
        weights = pl.DataFrame({"time": times, "weight": [1.0] * 5})
        scorer = MeanAbsoluteError(time_weighter=TableWeighter(frame=weights, on="time"))
        fig = plot_score_time_series(
            scorer,
            y_truth,
            y_pred,
            groups=["grp"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestPlotScoreTimeSeriesVintageFacet:
    """Tests for facet_by='vintage' in plot_score_time_series."""

    @pytest.fixture()
    def multi_vintage_data(self):
        from datetime import datetime

        times = [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)]
        y_truth = pl.DataFrame({"time": times, "value": [10.0, 20.0, 30.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 30)] * 3 + [datetime(2019, 12, 31)] * 3,
            "time": times * 2,
            "value": [12.0, 19.0, 28.0, 11.0, 21.0, 29.0],
        }).sort("time", "vintage_time")
        return y_truth, y_pred

    def test_creates_one_subplot_per_vintage(self, multi_vintage_data):
        """Each unique vintage_time gets its own subplot."""
        y_truth, y_pred = multi_vintage_data
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(scorer, y_truth, y_pred, facet_by="vintage")
        assert isinstance(fig, go.Figure)
        # 2 vintages → 2 traces (one per subplot for the single model)
        assert len(fig.data) == 2
        # Each trace has 3 time points
        for trace in fig.data:
            assert len(trace.x) == 3

    def test_multi_model_vintage_facet(self, multi_vintage_data):
        """Multiple models with vintage faceting overlay models within each vintage subplot."""
        y_truth, y_pred = multi_vintage_data
        y_pred2 = y_pred.with_columns(pl.col("value") + 1)
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(scorer, y_truth, {"A": y_pred, "B": y_pred2}, facet_by="vintage")
        # 2 vintages × 2 models = 4 traces
        assert len(fig.data) == 4

    def test_single_vintage_raises(self):
        """Single vintage data with facet_by='vintage' raises ValueError."""
        from datetime import datetime

        y_truth = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [10.0, 20.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [11.0, 21.0],
        })
        scorer = MeanAbsoluteError()
        with pytest.raises(ValueError, match="multiple vintages"):
            plot_score_time_series(scorer, y_truth, y_pred, facet_by="vintage")

    def test_multi_scorer_vintage_raises(self, multi_vintage_data):
        """Multi-scorer with facet_by='vintage' raises ValueError."""
        y_truth, y_pred = multi_vintage_data
        scorers = {"mae": MeanAbsoluteError(), "mse": MeanSquaredError()}
        with pytest.raises(ValueError, match="Multi-scorer.*vintage"):
            plot_score_time_series(scorers, y_truth, y_pred, facet_by="vintage")

    def test_legend_deduplication(self, multi_vintage_data):
        """Model name appears in legend only once across vintage subplots."""
        y_truth, y_pred = multi_vintage_data
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(scorer, y_truth, y_pred, facet_by="vintage")
        showlegend_count = sum(1 for t in fig.data if t.showlegend is not False)
        assert showlegend_count == 1

    def test_panel_with_vintage_raises(self):
        """Panel data (groups) combined with facet_by='vintage' raises ValueError."""
        from datetime import datetime

        times = [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)]
        y_truth = pl.DataFrame({"time": times, "grp__x": [10.0, 20.0, 30.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 30)] * 3 + [datetime(2019, 12, 31)] * 3,
            "time": times * 2,
            "grp__x": [12.0, 19.0, 28.0, 11.0, 21.0, 29.0],
        }).sort("time", "vintage_time")
        scorer = MeanAbsoluteError()
        with pytest.raises(ValueError, match="cannot be combined with panel"):
            plot_score_time_series(scorer, y_truth, y_pred, groups=["grp"], facet_by="vintage")


class TestPlotScoreTimeSeriesVintageOverlay:
    """Tests for multi-vintage data without facet_by='vintage' (zigzag fix)."""

    @pytest.fixture()
    def multi_vintage_data(self):
        from datetime import datetime

        times = [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)]
        y_truth = pl.DataFrame({"time": times, "value": [10.0, 20.0, 30.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 28)] * 3 + [datetime(2019, 12, 29)] * 3 + [datetime(2019, 12, 30)] * 3,
            "time": times * 3,
            "value": [13.0, 18.0, 27.0, 12.0, 19.0, 28.0, 11.0, 21.0, 29.0],
        }).sort("time", "vintage_time")
        return y_truth, y_pred

    def test_multi_model_overlay_splits_by_vintage(self, multi_vintage_data):
        """Single scorer + multi-model with 3 vintages creates 3 traces per model."""
        y_truth, y_pred = multi_vintage_data
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(scorer, y_truth, {"A": y_pred, "B": y_pred}, facet_by=None)
        # 2 models x 3 vintages = 6 traces
        assert len(fig.data) == 6

    def test_multi_scorer_overlay_splits_by_vintage(self, multi_vintage_data):
        """Multi-scorer + single model with 3 vintages creates 3 traces per scorer."""
        y_truth, y_pred = multi_vintage_data
        scorers = {"mae": MeanAbsoluteError(), "mse": MeanSquaredError()}
        fig = plot_score_time_series(scorers, y_truth, y_pred, facet_by=None)
        # 2 scorers x 3 vintages = 6 traces
        assert len(fig.data) == 6

    def test_newest_vintage_showlegend(self, multi_vintage_data):
        """Only the newest vintage trace per model shows in the legend."""
        y_truth, y_pred = multi_vintage_data
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(scorer, y_truth, y_pred, facet_by=None)
        # 3 vintages, only newest shows
        legend_count = sum(1 for t in fig.data if t.showlegend is not False)
        assert legend_count == 1

    def test_oldest_vintage_has_lower_opacity(self, multi_vintage_data):
        """Oldest vintage trace has opacity ~ 0.3, newest ~ 1.0."""
        y_truth, y_pred = multi_vintage_data
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(scorer, y_truth, y_pred, facet_by=None)
        opacities = [t.opacity for t in fig.data]
        assert min(opacities) == pytest.approx(0.3, abs=0.01)
        assert max(opacities) == pytest.approx(1.0, abs=0.01)

    def test_vintage_hover_includes_vintage(self, multi_vintage_data):
        """Hover template includes 'Vintage:' for multi-vintage traces."""
        y_truth, y_pred = multi_vintage_data
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(scorer, y_truth, y_pred, facet_by=None)
        for trace in fig.data:
            assert "Vintage:" in trace.hovertemplate

    def test_single_vintage_no_split(self):
        """Single-vintage data draws one trace per model (no splitting)."""
        from datetime import datetime

        times = [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)]
        y_truth = pl.DataFrame({"time": times, "value": [10.0, 20.0, 30.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": times,
            "value": [11.0, 21.0, 29.0],
        })
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(scorer, y_truth, y_pred, facet_by=None)
        assert len(fig.data) == 1
        assert "Vintage:" not in fig.data[0].hovertemplate

    def test_faceted_overlay_splits_by_vintage(self, multi_vintage_data):
        """Multi-scorer + multi-model with vintages splits within facet subplots."""
        y_truth, y_pred = multi_vintage_data
        scorers = {"mae": MeanAbsoluteError(), "mse": MeanSquaredError()}
        fig = plot_score_time_series(scorers, y_truth, {"A": y_pred, "B": y_pred}, facet_by=None)
        # 2 facets x 2 overlays x 3 vintages = 12 traces
        assert len(fig.data) == 12


class TestPlotScorePerHorizonPanelBarMultiModel:
    """Cover bar + multi-model panel path in plot_score_per_step (line 1954)."""

    def test_panel_bar_multi_model(self):
        """Panel bar chart with multiple models triggers barmode='group'."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(8)]
        y_truth = pl.DataFrame({
            "time": times,
            "grp__x": [10.0 + i for i in range(8)],
        })
        y_pred_a = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 8,
            "time": times,
            "grp__x": [11.0 + i for i in range(8)],
        })
        y_pred_b = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 8,
            "time": times,
            "grp__x": [12.0 + i for i in range(8)],
        })
        fig = plot_score_per_step(
            MeanAbsoluteError(),
            y_truth,
            {"M1": y_pred_a, "M2": y_pred_b},
            groups=["grp"],
            kind="bar",
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2


class TestMultiComponentScoreBranches:
    """Cover multi-score-column aggregation paths in score_time_series,
    score_distribution, and score_per_horizon (eval lines 1203, 1596, 1902).
    """

    @pytest.fixture
    def multi_col_data(self):
        """Truth and pred DataFrames with two value columns."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(30)]
        y_truth = pl.DataFrame({
            "time": times,
            "y1": [10.0 + i for i in range(30)],
            "y2": [20.0 + i * 2 for i in range(30)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 30,
            "time": times,
            "y1": [11.0 + i for i in range(30)],
            "y2": [22.0 + i * 2 for i in range(30)],
        })
        return y_truth, y_pred

    def test_score_time_series_multi_column(self, multi_col_data):
        """Multiple score columns trigger mean aggregation in score_time_series."""
        y_truth, y_pred = multi_col_data
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(scorer, y_truth, y_pred)
        assert_figure_valid(fig)

    def test_score_distribution_multi_column(self, multi_col_data):
        """Multiple score columns trigger flatten path in score_distribution."""
        y_truth, y_pred = multi_col_data
        scorer = MeanAbsoluteError()
        fig = plot_score_distribution(scorer, y_truth, y_pred)
        assert_figure_valid(fig)

    def test_score_per_horizon_multi_column(self, multi_col_data):
        """Multiple score columns trigger mean_horizontal in score_per_horizon."""
        y_truth, y_pred = multi_col_data
        scorer = MeanAbsoluteError()
        fig = plot_score_per_step(scorer, y_truth, y_pred)
        assert_figure_valid(fig)


class TestScorerReturnValidation:
    """Tests for scorer return-type checks in componentwise aggregation."""

    @pytest.fixture
    def panel_data(self):
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(10)]
        y_truth = pl.DataFrame({
            "time": times,
            "sales__store_1": [10.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 10,
            "time": times,
            "sales__store_1": [11.0 + i for i in range(10)],
        })
        return y_truth, {"model_a": y_pred}

    @pytest.fixture
    def simple_data(self):
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(10)]
        y_truth = pl.DataFrame({"time": times, "y": [10.0 + i for i in range(10)]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 10,
            "time": times,
            "y": [11.0 + i for i in range(10)],
        })
        return y_truth, {"model_a": y_pred}

    def test_panel_score_time_series_non_dataframe_raises(self, panel_data):
        """Panel score_time_series raises TypeError when scorer.score returns non-DataFrame."""
        from unittest.mock import patch

        y_truth, y_pred = panel_data
        scorer = MeanAbsoluteError()
        with (
            patch.object(MeanAbsoluteError, "score", return_value=42.0),
            pytest.raises(TypeError, match="Scorer must return DataFrame"),
        ):
            plot_score_time_series(scorer, y_truth, y_pred, groups=["sales"])

    def test_panel_score_time_series_no_time_col_raises(self, panel_data):
        """Panel score_time_series raises ValueError when scorer.score returns no 'time' column."""
        from unittest.mock import patch

        y_truth, y_pred = panel_data
        scorer = MeanAbsoluteError()
        bad_df = pl.DataFrame({"sales__store_1": [1.0] * 10})
        with (
            patch.object(MeanAbsoluteError, "score", return_value=bad_df),
            pytest.raises(ValueError, match="'time' column"),
        ):
            plot_score_time_series(scorer, y_truth, y_pred, groups=["sales"])

    def test_score_time_series_non_dataframe_raises(self, simple_data):
        """Non-panel score_time_series raises TypeError when scorer.score returns non-DataFrame."""
        from unittest.mock import patch

        y_truth, y_pred = simple_data
        scorer = MeanAbsoluteError()
        with (
            patch.object(MeanAbsoluteError, "score", return_value=42.0),
            pytest.raises(TypeError, match="Scorer must return DataFrame"),
        ):
            plot_score_time_series(scorer, y_truth, y_pred)

    def test_score_time_series_no_time_col_raises(self, simple_data):
        """Non-panel score_time_series raises ValueError when scorer.score returns no 'time' column."""
        from unittest.mock import patch

        y_truth, y_pred = simple_data
        scorer = MeanAbsoluteError()
        bad_df = pl.DataFrame({"y": [1.0] * 10})
        with (
            patch.object(MeanAbsoluteError, "score", return_value=bad_df),
            pytest.raises(ValueError, match="'time' column"),
        ):
            plot_score_time_series(scorer, y_truth, y_pred)

    def test_score_distribution_non_dataframe_raises(self, simple_data):
        """plot_score_distribution raises TypeError when scorer.score returns non-DataFrame."""
        from unittest.mock import patch

        y_truth, y_pred = simple_data
        scorer = MeanAbsoluteError()
        with (
            patch.object(MeanAbsoluteError, "score", return_value=42.0),
            pytest.raises(TypeError, match="Scorer must return DataFrame"),
        ):
            plot_score_distribution(scorer, y_truth, y_pred)

    def test_per_horizon_non_dataframe_raises(self, simple_data):
        """plot_score_per_step raises TypeError when scorer.score returns non-DataFrame."""
        from unittest.mock import patch

        y_truth, y_pred = simple_data
        scorer = MeanAbsoluteError()
        with (
            patch.object(MeanAbsoluteError, "score", return_value=42.0),
            pytest.raises(TypeError, match="Scorer must return DataFrame"),
        ):
            plot_score_per_step(scorer, y_truth, y_pred)


class TestPanelMultiMemberScoring:
    """Tests for multi-member panel scoring (multi-column else branches)."""

    @pytest.fixture
    def panel_multi_member(self):
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(15)]
        y_truth = pl.DataFrame({
            "time": times,
            "sales__store_1": [10.0 + i for i in range(15)],
            "sales__store_2": [20.0 + i for i in range(15)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 15,
            "time": times,
            "sales__store_1": [11.0 + i for i in range(15)],
            "sales__store_2": [21.0 + i for i in range(15)],
        })
        return y_truth, {"model_a": y_pred}

    def test_panel_score_distribution_multi_member(self, panel_multi_member):
        """Score distribution multi-member panel triggers multi-column flatten."""
        y_truth, y_pred = panel_multi_member
        scorer = MeanAbsoluteError()
        fig = plot_score_distribution(scorer, y_truth, y_pred, groups=["sales"])
        assert_figure_valid(fig)

    def test_panel_score_per_horizon_multi_member(self, panel_multi_member):
        """Per-horizon multi-member panel triggers multi-column mean_horizontal."""
        y_truth, y_pred = panel_multi_member
        scorer = MeanAbsoluteError()
        fig = plot_score_per_step(scorer, y_truth, y_pred, groups=["sales"])
        assert_figure_valid(fig)

    def test_panel_score_time_series_filtered_empty_group(self, panel_multi_member):
        """Panel score_time_series with column filter leaving no matches skips group."""
        y_truth, y_pred = panel_multi_member
        scorer = MeanAbsoluteError()
        fig = plot_score_time_series(
            scorer,
            y_truth,
            y_pred,
            groups=["sales"],
            columns="nonexistent_member",
        )
        assert isinstance(fig, go.Figure)


class TestPlotReliabilityDiagram:
    """Tests for plot_calibration with class-probability data."""

    @pytest.fixture
    def reliability_data(self):
        """Create sample data for reliability diagram tests."""
        n = 50
        rng = np.random.default_rng(42)
        times = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 2, 19), "1d", eager=True)
        labels = rng.choice(["sunny", "rainy", "cloudy"], size=n)
        probs_sunny = rng.uniform(0, 1, n)
        probs_rainy = rng.uniform(0, 1, n)
        probs_cloudy = rng.uniform(0, 1, n)
        total = probs_sunny + probs_rainy + probs_cloudy
        y_pred = pl.DataFrame({
            "time": times,
            "w_proba_sunny": probs_sunny / total,
            "w_proba_rainy": probs_rainy / total,
            "w_proba_cloudy": probs_cloudy / total,
        })
        y_truth = pl.DataFrame({
            "time": times,
            "w": labels.tolist(),
        })
        return y_pred, y_truth

    def test_basic(self, reliability_data):
        """Basic reliability diagram returns Figure."""
        y_pred, y_truth = reliability_data
        fig = plot_calibration(y_pred, y_truth)
        assert isinstance(fig, go.Figure)
        # 3 class traces + 1 reference diagonal
        assert len(fig.data) == 4

    def test_custom_bins(self, reliability_data):
        """Custom bin count works."""
        y_pred, y_truth = reliability_data
        fig = plot_calibration(y_pred, y_truth, n_bins=5)
        assert isinstance(fig, go.Figure)

    def test_title_propagated(self, reliability_data):
        """Custom title is applied."""
        y_pred, y_truth = reliability_data
        fig = plot_calibration(y_pred, y_truth, title="My Reliability")
        assert fig.layout.title.text == "My Reliability"

    def test_no_proba_columns_raises(self):
        """DataFrame without proba columns and no coverage_rates raises ValueError."""
        from datetime import datetime

        df = pl.DataFrame({"time": [datetime(2020, 1, 1)], "a": [1.0]})
        y_truth = pl.DataFrame({"time": [datetime(2020, 1, 1)], "a": ["x"]})
        with pytest.raises(ValueError, match="No interval columns found"):
            plot_calibration(df, y_truth)

    def test_multi_target_without_target_raises(self):
        """Ambiguous multi-target raises ValueError."""
        from datetime import datetime

        df = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "a_proba_x": [0.5],
            "a_proba_y": [0.5],
            "b_proba_x": [0.5],
            "b_proba_y": [0.5],
        })
        y_truth = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "a": ["x"],
            "b": ["x"],
        })
        with pytest.raises(ValueError, match="Multiple targets"):
            plot_calibration(df, y_truth)

    def test_invalid_target_raises(self, reliability_data):
        """Non-existent target raises ValueError."""
        y_pred, y_truth = reliability_data
        with pytest.raises(ValueError, match="Target 'invalid'"):
            plot_calibration(y_pred, y_truth, target="invalid")

    def test_target_not_in_truth_raises(self, reliability_data):
        """Target in pred but not in truth raises ValueError."""
        y_pred, y_truth = reliability_data
        y_truth_bad = y_truth.rename({"w": "other"})
        with pytest.raises(ValueError, match="not found in y_truth"):
            plot_calibration(y_pred, y_truth_bad)


@pytest.fixture
def multi_scorer_data():
    """Create data suitable for multi-scorer tests."""
    from datetime import datetime

    times = [datetime(2020, 1, 1 + i) for i in range(10)]
    y_truth = pl.DataFrame({
        "time": times,
        "value": [10.0, 20.0, 30.0, 25.0, 35.0, 40.0, 38.0, 45.0, 50.0, 48.0],
    })
    y_pred_a = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * 10,
        "time": times,
        "value": [12.0, 19.0, 28.0, 27.0, 33.0, 42.0, 36.0, 43.0, 52.0, 46.0],
    })
    y_pred_b = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * 10,
        "time": times,
        "value": [11.0, 21.0, 29.0, 26.0, 34.0, 41.0, 37.0, 44.0, 51.0, 47.0],
    })
    return {
        "y_truth": y_truth,
        "y_pred": y_pred_a,
        "y_preds": {"Model A": y_pred_a, "Model B": y_pred_b},
    }


@pytest.fixture
def multi_vintage_data():
    """Create data with multiple vintages for per-vintage tests."""
    from datetime import datetime

    times = [datetime(2020, 1, 1 + i) for i in range(5)]
    y_truth = pl.DataFrame({
        "time": times,
        "value": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    # Build multi-vintage predictions: each vintage is a separate forecast
    # sorted by vintage_time then time
    rows = []
    for ot, offsets in [
        (datetime(2019, 12, 29), [2.0, -1.0, -2.0, 2.0, -2.0]),
        (datetime(2019, 12, 30), [1.0, 1.0, -1.0, 1.0, -1.0]),
        (datetime(2019, 12, 31), [0.5, 0.5, 0.5, 0.5, 0.5]),
    ]:
        for i in range(5):
            rows.append({
                "vintage_time": ot,
                "time": times[i],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0][i] + offsets[i],
            })
    y_pred = pl.DataFrame(rows).sort("time", "vintage_time")
    return {"y_truth": y_truth, "y_pred": y_pred}


@pytest.fixture
def panel_group_data():
    """Create panel data with groups for group score tests."""
    from datetime import datetime

    times = [datetime(2020, 1, 1 + i) for i in range(5)]
    y_truth = pl.DataFrame({
        "time": times,
        "region__east": [10.0, 20.0, 30.0, 40.0, 50.0],
        "region__west": [15.0, 25.0, 35.0, 45.0, 55.0],
    })
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * 5,
        "time": times,
        "region__east": [12.0, 19.0, 28.0, 42.0, 48.0],
        "region__west": [14.0, 26.0, 33.0, 46.0, 53.0],
    })
    return {"y_truth": y_truth, "y_pred": y_pred}


class TestMultiScorerTimeSeries:
    """Tests for multi-scorer support in plot_score_time_series."""

    def test_dict_scorer_single_model(self, multi_scorer_data):
        """Dict scorer with single model overlays scorers."""
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_score_time_series(
            scorers,
            multi_scorer_data["y_truth"],
            multi_scorer_data["y_pred"],
        )
        assert_figure_valid(fig)
        assert len(fig.data) == 2  # one trace per scorer

    def test_dict_scorer_dict_model_faceted(self, multi_scorer_data):
        """Dict scorer with dict y_pred creates faceted subplots."""
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_score_time_series(
            scorers,
            multi_scorer_data["y_truth"],
            multi_scorer_data["y_preds"],
        )
        assert_figure_valid(fig)
        # 2 models * 2 scorers = 4 traces
        assert len(fig.data) == 4

    def test_compare_by_model(self, multi_scorer_data):
        """compare_by='model' facets by scorer, overlays models."""
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_score_time_series(
            scorers,
            multi_scorer_data["y_truth"],
            multi_scorer_data["y_preds"],
            compare_by="model",
        )
        assert_figure_valid(fig)
        assert len(fig.data) == 4

    def test_single_scorer_backward_compat(self, multi_scorer_data):
        """Single scorer still works as before."""
        fig = plot_score_time_series(
            MeanAbsoluteError(),
            multi_scorer_data["y_truth"],
            multi_scorer_data["y_preds"],
        )
        assert_figure_valid(fig)
        assert len(fig.data) == 2  # one trace per model

    def test_multi_scorer_panel_raises(self, multi_scorer_data):
        """Multi-scorer with panel data raises ValueError."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(5)]
        y_truth_panel = pl.DataFrame({
            "time": times,
            "grp__a": [1.0] * 5,
            "grp__b": [2.0] * 5,
        })
        y_pred_panel = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": times,
            "grp__a": [1.1] * 5,
            "grp__b": [2.1] * 5,
        })
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        with pytest.raises(ValueError, match="Multi-scorer"):
            plot_score_time_series(scorers, y_truth_panel, y_pred_panel)


class TestMultiScorerDistribution:
    """Tests for multi-scorer support in plot_score_distribution."""

    def test_dict_scorer_single_model(self, multi_scorer_data):
        """Dict scorer with single model overlays scorers."""
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_score_distribution(
            scorers,
            multi_scorer_data["y_truth"],
            multi_scorer_data["y_pred"],
        )
        assert_figure_valid(fig)

    def test_dict_scorer_dict_model(self, multi_scorer_data):
        """Dict scorer with dict y_pred creates faceted subplots."""
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_score_distribution(
            scorers,
            multi_scorer_data["y_truth"],
            multi_scorer_data["y_preds"],
        )
        assert_figure_valid(fig)

    def test_kde_mode(self, multi_scorer_data):
        """Multi-scorer with KDE mode."""
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_score_distribution(
            scorers,
            multi_scorer_data["y_truth"],
            multi_scorer_data["y_pred"],
            kind="kde",
        )
        assert_figure_valid(fig)


class TestMultiScorerPerHorizon:
    """Tests for multi-scorer support in plot_score_per_step."""

    def test_dict_scorer_single_model(self, multi_scorer_data):
        """Dict scorer with single model overlays scorers."""
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_score_per_step(
            scorers,
            multi_scorer_data["y_truth"],
            multi_scorer_data["y_pred"],
        )
        assert_figure_valid(fig)
        assert len(fig.data) == 2

    def test_dict_scorer_dict_model(self, multi_scorer_data):
        """Dict scorer with dict y_pred creates faceted subplots."""
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_score_per_step(
            scorers,
            multi_scorer_data["y_truth"],
            multi_scorer_data["y_preds"],
        )
        assert_figure_valid(fig)

    def test_bar_mode(self, multi_scorer_data):
        """Multi-scorer with bar mode."""
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_score_per_step(
            scorers,
            multi_scorer_data["y_truth"],
            multi_scorer_data["y_pred"],
            kind="bar",
        )
        assert_figure_valid(fig)


class TestPlotScoreSummary:
    """Tests for plot_score_summary function."""

    def test_summary_single_scorer(self, multi_scorer_data):
        """plot_score_summary with single scorer and dict y_pred."""
        fig = plot_score_summary(
            MeanAbsoluteError(),
            multi_scorer_data["y_truth"],
            multi_scorer_data["y_preds"],
        )
        assert_figure_valid(fig)
        assert any(isinstance(t, go.Bar) for t in fig.data)

    def test_summary_multi_scorer(self, multi_scorer_data):
        """plot_score_summary with dict scorers and dict y_pred."""
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_score_summary(
            scorers,
            multi_scorer_data["y_truth"],
            multi_scorer_data["y_preds"],
        )
        assert_figure_valid(fig)
        # 2 models
        assert len(fig.data) == 2

    def test_summary_default_title(self, multi_scorer_data):
        """plot_score_summary uses 'Model Comparison' as default title."""
        fig = plot_score_summary(
            MeanAbsoluteError(),
            multi_scorer_data["y_truth"],
            multi_scorer_data["y_preds"],
        )
        assert_layout(fig, title="Model Comparison")

    def test_summary_sort_ascending(self, multi_scorer_data):
        """plot_score_summary with sort_ascending."""
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_score_summary(
            scorers,
            multi_scorer_data["y_truth"],
            multi_scorer_data["y_pred"],
            sort_ascending=True,
        )
        assert_figure_valid(fig)

    def test_summary_sort_descending(self, multi_scorer_data):
        """sort_ascending=False orders score categories from highest to lowest."""
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        ascending = plot_score_summary(
            scorers,
            multi_scorer_data["y_truth"],
            multi_scorer_data["y_preds"],
            sort_ascending=True,
        )
        descending = plot_score_summary(
            scorers,
            multi_scorer_data["y_truth"],
            multi_scorer_data["y_preds"],
            sort_ascending=False,
        )
        asc_bar = next(t for t in ascending.data if isinstance(t, go.Bar))
        desc_bar = next(t for t in descending.data if isinstance(t, go.Bar))
        # Descending sort reverses the category order relative to ascending,
        # so the first descending category is the last ascending one.
        assert list(desc_bar.x) == list(asc_bar.x)[::-1]
        assert desc_bar.y[0] >= desc_bar.y[-1]


class TestPlotScorePerVintage:
    """Tests for plot_score_per_vintage function."""

    def test_basic(self, multi_vintage_data):
        """Basic per-vintage plot with single scorer."""
        fig = plot_score_per_vintage(
            MeanAbsoluteError(),
            multi_vintage_data["y_truth"],
            multi_vintage_data["y_pred"],
        )
        assert_figure_valid(fig)

    def test_bar_kind(self, multi_vintage_data):
        """Per-vintage bar chart."""
        from yohou.plotting import plot_score_per_vintage

        fig = plot_score_per_vintage(
            MeanAbsoluteError(),
            multi_vintage_data["y_truth"],
            multi_vintage_data["y_pred"],
            kind="bar",
        )
        assert_figure_valid(fig)

    def test_invalid_kind_raises(self, multi_vintage_data):
        """An unsupported kind raises ValueError."""
        with pytest.raises(ValueError, match="kind must be one of"):
            plot_score_per_vintage(
                MeanAbsoluteError(),
                multi_vintage_data["y_truth"],
                multi_vintage_data["y_pred"],
                kind="scatter",
            )

    def test_multi_scorer(self, multi_vintage_data):
        """Dict scorer overlays scorers."""
        from yohou.plotting import plot_score_per_vintage

        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_score_per_vintage(
            scorers,
            multi_vintage_data["y_truth"],
            multi_vintage_data["y_pred"],
        )
        assert_figure_valid(fig)
        assert len(fig.data) == 2

    def test_single_vintage_raises(self, multi_scorer_data):
        """Single vintage raises ValueError."""
        from yohou.plotting import plot_score_per_vintage

        with pytest.raises(ValueError, match="single vintage"):
            plot_score_per_vintage(
                MeanAbsoluteError(),
                multi_scorer_data["y_truth"],
                multi_scorer_data["y_pred"],
            )

    def test_show_trend(self, multi_vintage_data):
        """Trend line is displayed."""
        from yohou.plotting import plot_score_per_vintage

        fig = plot_score_per_vintage(
            MeanAbsoluteError(),
            multi_vintage_data["y_truth"],
            multi_vintage_data["y_pred"],
            show_trend=True,
        )
        assert_figure_valid(fig)
        trace_names = [t.name for t in fig.data if t.name is not None]
        assert any("trend" in name.lower() for name in trace_names)


class TestPlotScoreHeatmap:
    """Tests for plot_score_heatmap function."""

    def test_basic(self, multi_vintage_data):
        """Basic heatmap: step x vintage."""

        fig = plot_score_heatmap(
            MeanAbsoluteError(),
            multi_vintage_data["y_truth"],
            multi_vintage_data["y_pred"],
        )
        assert_figure_valid(fig)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_inverted_dims(self, multi_vintage_data):
        """Heatmap with x=vintage, y=step."""

        fig = plot_score_heatmap(
            MeanAbsoluteError(),
            multi_vintage_data["y_truth"],
            multi_vintage_data["y_pred"],
            x_dim="vintage",
            y_dim="step",
        )
        assert_figure_valid(fig)

    def test_same_dims_raises(self, multi_vintage_data):
        """Same x_dim and y_dim raises ValueError."""

        with pytest.raises(ValueError, match="must differ"):
            plot_score_heatmap(
                MeanAbsoluteError(),
                multi_vintage_data["y_truth"],
                multi_vintage_data["y_pred"],
                x_dim="step",
                y_dim="step",
            )

    def test_custom_colorscale(self, multi_vintage_data):
        """Custom colorscale."""

        fig = plot_score_heatmap(
            MeanAbsoluteError(),
            multi_vintage_data["y_truth"],
            multi_vintage_data["y_pred"],
            color_palette="Viridis",
        )
        assert_figure_valid(fig)

    def test_no_text(self, multi_vintage_data):
        """Heatmap without text annotations."""

        fig = plot_score_heatmap(
            MeanAbsoluteError(),
            multi_vintage_data["y_truth"],
            multi_vintage_data["y_pred"],
            show_text=False,
        )
        assert_figure_valid(fig)


class TestPlotGroupScores:
    """Tests for plot_group_scores function."""

    def test_basic_bar(self, panel_group_data):
        """Basic bar chart per group."""

        fig = plot_group_scores(
            MeanAbsoluteError(),
            panel_group_data["y_truth"],
            panel_group_data["y_pred"],
        )
        assert_figure_valid(fig)
        assert any(isinstance(t, go.Bar) for t in fig.data)

    def test_box_kind(self, panel_group_data):
        """Box plot per group."""

        fig = plot_group_scores(
            MeanAbsoluteError(),
            panel_group_data["y_truth"],
            panel_group_data["y_pred"],
            kind="box",
        )
        assert_figure_valid(fig)
        assert any(isinstance(t, go.Box) for t in fig.data)

    def test_multi_scorer_bar(self, panel_group_data):
        """Multi-scorer bar chart."""

        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_group_scores(
            scorers,
            panel_group_data["y_truth"],
            panel_group_data["y_pred"],
        )
        assert_figure_valid(fig)
        # 2 scorers
        assert len(fig.data) == 2

    def test_non_panel_raises(self, multi_scorer_data):
        """Non-panel data raises ValueError."""

        with pytest.raises(ValueError, match="panel group columns"):
            plot_group_scores(
                MeanAbsoluteError(),
                multi_scorer_data["y_truth"],
                multi_scorer_data["y_pred"],
            )

    def test_sort_ascending(self, panel_group_data):
        """Sorted bar chart."""

        fig = plot_group_scores(
            MeanAbsoluteError(),
            panel_group_data["y_truth"],
            panel_group_data["y_pred"],
            sort_ascending=True,
        )
        assert_figure_valid(fig)

    def test_heatmap_kind(self, panel_group_data):
        """Heatmap per group."""

        fig = plot_group_scores(
            MeanAbsoluteError(),
            panel_group_data["y_truth"],
            panel_group_data["y_pred"],
            kind="heatmap",
        )
        assert_figure_valid(fig)
        assert any(isinstance(t, go.Heatmap) for t in fig.data)

    def test_multi_scorer_multi_model_bar(self, panel_group_data):
        """Multi-scorer dict with multi-model dict produces bar chart."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(5)]
        y_pred_b = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": times,
            "region__east": [13.0, 18.0, 29.0, 41.0, 49.0],
            "region__west": [16.0, 24.0, 34.0, 44.0, 54.0],
        })
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        y_preds = {"Model A": panel_group_data["y_pred"], "Model B": y_pred_b}
        fig = plot_group_scores(
            scorers,
            panel_group_data["y_truth"],
            y_preds,
        )
        assert_figure_valid(fig)
        # 2 scorers x 2 models = 4 series
        assert len(fig.data) == 4

    def test_multi_scorer_heatmap(self, panel_group_data):
        """Multi-scorer dict produces heatmap with scorer columns."""
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_group_scores(
            scorers,
            panel_group_data["y_truth"],
            panel_group_data["y_pred"],
            kind="heatmap",
        )
        assert_figure_valid(fig)
        assert any(isinstance(t, go.Heatmap) for t in fig.data)

    def test_invalid_kind_raises(self, panel_group_data):
        """Invalid kind raises ValueError."""
        with pytest.raises(ValueError, match="kind must be"):
            plot_group_scores(
                MeanAbsoluteError(),
                panel_group_data["y_truth"],
                panel_group_data["y_pred"],
                kind="scatter",
            )

    @pytest.mark.parametrize("kind", ["bar", "box", "heatmap"])
    def test_dataframe_scorer_reduced_to_scalar(self, panel_group_data, kind):
        """A scorer that returns a per-component DataFrame is reduced to one score per group."""
        scorer = MeanAbsoluteError(aggregation_method=["componentwise"])
        fig = plot_group_scores(
            scorer,
            panel_group_data["y_truth"],
            panel_group_data["y_pred"],
            kind=kind,
        )
        assert_figure_valid(fig)


class TestMultiScorerMultiModelFaceted:
    """Tests for multi-scorer + multi-model faceted subplot paths."""

    def test_distribution_compare_by_model(self, multi_scorer_data):
        """Distribution with compare_by='model' facets by scorer."""
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_score_distribution(
            scorers,
            multi_scorer_data["y_truth"],
            multi_scorer_data["y_preds"],
            compare_by="model",
        )
        assert_figure_valid(fig)

    def test_horizon_compare_by_model(self, multi_scorer_data):
        """Per-horizon with compare_by='model' facets by scorer."""
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_score_per_step(
            scorers,
            multi_scorer_data["y_truth"],
            multi_scorer_data["y_preds"],
            compare_by="model",
        )
        assert_figure_valid(fig)


@pytest.fixture
def multi_vintage_multi_model_data():
    """Create multi-vintage data with two models for faceted vintage tests."""
    from datetime import datetime

    times = [datetime(2020, 1, 1 + i) for i in range(5)]
    y_truth = pl.DataFrame({
        "time": times,
        "value": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    rows_a = []
    rows_b = []
    for ot, offsets_a, offsets_b in [
        (datetime(2019, 12, 29), [2.0, -1.0, -2.0, 2.0, -2.0], [3.0, -2.0, -1.0, 1.0, -3.0]),
        (datetime(2019, 12, 30), [1.0, 1.0, -1.0, 1.0, -1.0], [0.5, 2.0, -0.5, 0.5, -0.5]),
        (datetime(2019, 12, 31), [0.5, 0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0]),
    ]:
        for i in range(5):
            base = [10.0, 20.0, 30.0, 40.0, 50.0][i]
            rows_a.append({"vintage_time": ot, "time": times[i], "value": base + offsets_a[i]})
            rows_b.append({"vintage_time": ot, "time": times[i], "value": base + offsets_b[i]})
    y_pred_a = pl.DataFrame(rows_a).sort("time", "vintage_time")
    y_pred_b = pl.DataFrame(rows_b).sort("time", "vintage_time")
    return {
        "y_truth": y_truth,
        "y_pred_a": y_pred_a,
        "y_preds": {"Model A": y_pred_a, "Model B": y_pred_b},
    }


class TestPlotScorePerVintageFaceted:
    """Tests for multi-scorer + multi-model faceted vintage plots."""

    def test_multi_scorer_multi_model(self, multi_vintage_multi_model_data):
        """Multi-scorer + multi-model creates faceted subplots."""
        d = multi_vintage_multi_model_data
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_score_per_vintage(scorers, d["y_truth"], d["y_preds"])
        assert_figure_valid(fig)
        # Should have at least 2 * 2 = 4 traces (2 models overlay in each facet)
        assert len(fig.data) >= 4

    def test_multi_scorer_multi_model_compare_by_model(self, multi_vintage_multi_model_data):
        """compare_by='model' facets by scorer, overlays models."""
        d = multi_vintage_multi_model_data
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_score_per_vintage(
            scorers,
            d["y_truth"],
            d["y_preds"],
            compare_by="model",
        )
        assert_figure_valid(fig)

    def test_multi_scorer_single_model(self, multi_vintage_multi_model_data):
        """Multi-scorer with single model overlays scorers."""
        d = multi_vintage_multi_model_data
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_score_per_vintage(
            scorers,
            d["y_truth"],
            d["y_pred_a"],
        )
        assert_figure_valid(fig)
        assert len(fig.data) == 2

    def test_multi_scorer_multi_model_bar(self, multi_vintage_multi_model_data):
        """Multi-scorer + multi-model bar mode faceted."""
        d = multi_vintage_multi_model_data
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        fig = plot_score_per_vintage(
            scorers,
            d["y_truth"],
            d["y_preds"],
            kind="bar",
        )
        assert_figure_valid(fig)

    def test_multi_model_overlay(self, multi_vintage_multi_model_data):
        """Single scorer + multi-model overlays models."""
        d = multi_vintage_multi_model_data
        fig = plot_score_per_vintage(
            MeanAbsoluteError(),
            d["y_truth"],
            d["y_preds"],
        )
        assert_figure_valid(fig)
        # 2 models = 2 traces per-vintage overlay
        assert len(fig.data) >= 2


class TestPlotScoreHeatmapValidation:
    """Tests for error paths in plot_score_heatmap."""

    def test_invalid_dim_raises(self, multi_vintage_data):
        """Invalid dim value raises ValueError."""
        with pytest.raises(ValueError, match="must be one of"):
            plot_score_heatmap(
                MeanAbsoluteError(),
                multi_vintage_data["y_truth"],
                multi_vintage_data["y_pred"],
                x_dim="invalid",
            )

    def test_missing_vintage_time_raises(self):
        """y_pred without vintage_time raises ValueError."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(5)]
        y_truth = pl.DataFrame({"time": times, "value": [1.0] * 5})
        y_pred = pl.DataFrame({"time": times, "value": [1.1] * 5})
        with pytest.raises(ValueError, match="vintage_time"):
            plot_score_heatmap(MeanAbsoluteError(), y_truth, y_pred)

    def test_single_vintage_raises(self):
        """Single vintage raises ValueError."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(5)]
        y_truth = pl.DataFrame({"time": times, "value": [1.0] * 5})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": times,
            "value": [1.1] * 5,
        })
        with pytest.raises(ValueError, match="single vintage"):
            plot_score_heatmap(MeanAbsoluteError(), y_truth, y_pred)


class TestPanelMultiScorerErrors:
    """Tests for multi-scorer with panel data error paths."""

    @pytest.fixture
    def panel_forecast(self):
        """Panel data for error tests."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(5)]
        y_truth = pl.DataFrame({
            "time": times,
            "region__east": [10.0, 20.0, 30.0, 40.0, 50.0],
            "region__west": [15.0, 25.0, 35.0, 45.0, 55.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": times,
            "region__east": [12.0, 19.0, 28.0, 42.0, 48.0],
            "region__west": [14.0, 26.0, 33.0, 46.0, 53.0],
        })
        return y_truth, y_pred

    def test_distribution_panel_multi_scorer_raises(self, panel_forecast):
        """Multi-scorer with panel in distribution raises ValueError."""
        y_truth, y_pred = panel_forecast
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        with pytest.raises(ValueError, match="Multi-scorer is not supported with panel data"):
            plot_score_distribution(scorers, y_truth, y_pred)

    def test_horizon_panel_multi_scorer_raises(self, panel_forecast):
        """Multi-scorer with panel in per-horizon raises ValueError."""
        y_truth, y_pred = panel_forecast
        scorers = {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()}
        with pytest.raises(ValueError, match="Multi-scorer is not supported with panel data"):
            plot_score_per_step(scorers, y_truth, y_pred)


class TestWarnLargeGrid:
    """Tests for _warn_large_grid warning."""

    def test_large_grid_warns(self, multi_scorer_data):
        """Many scorers x many models triggers UserWarning."""
        # 13 scorers x 2 models = 26 > 12
        scorers = {f"scorer_{i}": MeanAbsoluteError() for i in range(7)}
        with pytest.warns(UserWarning, match="Large facet grid"):
            plot_score_distribution(
                scorers,
                multi_scorer_data["y_truth"],
                multi_scorer_data["y_preds"],
            )


class TestSummaryKindDataFrame:
    """Tests for plot_score_summary when scorer returns DataFrame."""

    def test_summary_with_multivariate_scorer(self):
        """plot_score_summary handles scorer that returns DataFrame."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(5)]
        y_truth = pl.DataFrame({
            "time": times,
            "a": [10.0, 20.0, 30.0, 40.0, 50.0],
            "b": [15.0, 25.0, 35.0, 45.0, 55.0],
        })
        y_pred_a = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": times,
            "a": [12.0, 19.0, 28.0, 42.0, 48.0],
            "b": [14.0, 26.0, 33.0, 46.0, 53.0],
        })
        y_pred_b = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": times,
            "a": [11.0, 21.0, 29.0, 41.0, 49.0],
            "b": [16.0, 24.0, 34.0, 44.0, 54.0],
        })
        # aggregation_method="componentwise" so score returns DataFrame, not scalar
        scorer = MeanAbsoluteError(aggregation_method="componentwise")
        fig = plot_score_summary(
            scorer,
            y_truth,
            {"Model A": y_pred_a, "Model B": y_pred_b},
        )
        assert_figure_valid(fig)


class TestGroupScoresBoxKind:
    """Tests for plot_group_scores kind='box' with panel data."""

    def test_box_with_multi_vintage(self):
        """Box plot with panel data and multiple vintages."""
        from datetime import datetime

        times = [datetime(2020, 1, 1 + i) for i in range(5)]
        y_truth = pl.DataFrame({
            "time": times,
            "region__east": [10.0, 20.0, 30.0, 40.0, 50.0],
            "region__west": [15.0, 25.0, 35.0, 45.0, 55.0],
        })
        rows = []
        for ot in [datetime(2019, 12, 30), datetime(2019, 12, 31)]:
            for i in range(5):
                rows.append({
                    "vintage_time": ot,
                    "time": times[i],
                    "region__east": [10.0, 20.0, 30.0, 40.0, 50.0][i] + 2.0,
                    "region__west": [15.0, 25.0, 35.0, 45.0, 55.0][i] + 1.0,
                })
        y_pred = pl.DataFrame(rows).sort("time", "vintage_time")
        fig = plot_group_scores(
            MeanAbsoluteError(),
            y_truth,
            y_pred,
            kind="box",
        )
        assert_figure_valid(fig)
        assert any(isinstance(t, go.Box) for t in fig.data)

    def test_box_distribute_by_vintage(self, panel_group_data):
        """Box plot with distribute_by='vintage'."""
        fig = plot_group_scores(
            MeanAbsoluteError(),
            panel_group_data["y_truth"],
            panel_group_data["y_pred"],
            kind="box",
            distribute_by="vintage",
        )
        assert_figure_valid(fig)


class TestCalibrationClassProbaParams:
    """plot_calibration forwards its public styling params (such as reference_color) in class-probability mode."""

    def test_reference_color_is_forwarded(self):
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 20), "1d", eager=True)
        n = len(dates)
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, n)
        y_pred = pl.DataFrame({
            "time": dates,
            "target_proba_yes": p.tolist(),
            "target_proba_no": (1 - p).tolist(),
        })
        truth = ["yes" if pi > 0.5 else "no" for pi in p]
        y_truth = pl.DataFrame({"time": dates, "target": truth})
        fig = plot_calibration(y_pred, y_truth, target="target", reference_color="#123456")
        ref_colors = [t.line.color for t in fig.data if t.line.color is not None]
        assert "#123456" in ref_colors


class TestScoreDistributionKdeWarning:
    """A KDE failure warns instead of silently omitting the trace."""

    def test_zero_variance_scores_warn(self):
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        n = len(dates)
        y_truth = pl.DataFrame({"time": dates, "y": [5.0] * n})
        y_pred = pl.DataFrame({"time": dates, "y": [5.0] * n})
        with pytest.warns(UserWarning, match="KDE"):
            plot_score_distribution(MeanAbsoluteError(), y_truth, y_pred, kind="kde")
