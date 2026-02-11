"""Tests for model evaluation plotting functions."""

import numpy as np
import polars as pl
import pytest
from plotly import graph_objects as go

from yohou.metrics import MeanAbsoluteError
from yohou.plotting import (
    plot_calibration,
    plot_model_comparison_bar,
    plot_residual_time_series,
    plot_score_distribution,
    plot_score_per_horizon,
    plot_score_time_series,
)


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
            "observed_time": [datetime(2019, 12, 31)] * 10,
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
        "observed_time": [datetime(2019, 12, 31)] * 10,
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


@pytest.fixture
def sample_comparison_results():
    """Create sample model comparison results."""
    return {
        "Naive": {"MAE": 12.3, "RMSE": 15.1, "MAPE": 8.5},
        "LinearRegression": {"MAE": 8.7, "RMSE": 10.4, "MAPE": 5.2},
        "Ridge": {"MAE": 9.1, "RMSE": 11.0, "MAPE": 5.8},
    }


class TestPlotResidualTimeSeries:
    """Tests for plot_residual_time_series function."""

    def test_basic(self, sample_residuals_data):
        """Test basic residual diagnostics plot."""
        fig = plot_residual_time_series(sample_residuals_data["y_pred"], sample_residuals_data["y_truth"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == "Residual Diagnostics"

    def test_explicit_column(self, sample_residuals_data):
        """Test with explicit columns parameter."""
        fig = plot_residual_time_series(sample_residuals_data["y_pred"], sample_residuals_data["y_truth"], columns="y")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_missing_column(self, sample_residuals_data):
        """Test error when column is missing from y_pred."""
        with pytest.raises(ValueError, match="not found"):
            plot_residual_time_series(
                sample_residuals_data["y_pred"], sample_residuals_data["y_truth"], columns="nonexistent"
            )

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
        fig = plot_residual_time_series(y_pred, y_truth, panel_group_names=["y"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1  # At least one trace for panel facets

    def test_custom_title(self, sample_residuals_data):
        """Test custom title."""
        fig = plot_residual_time_series(
            sample_residuals_data["y_pred"], sample_residuals_data["y_truth"], title="Custom Residuals"
        )
        assert fig.layout.title.text == "Custom Residuals"

    def test_custom_dimensions(self, sample_residuals_data):
        """Test custom width and height."""
        fig = plot_residual_time_series(
            sample_residuals_data["y_pred"], sample_residuals_data["y_truth"], width=1200, height=800
        )
        assert fig.layout.width == 1200
        assert fig.layout.height == 800

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
        fig = plot_residual_time_series(y_pred, y_truth, columns=["a", "b"])
        assert isinstance(fig, go.Figure)
        # 2 scatter traces + 2 hline shapes (one per column facet)
        assert len(fig.data) == 2
        assert fig.layout.title.text == "Residual Diagnostics"

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
        fig = plot_residual_time_series(y_pred, y_truth)
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
        fig = plot_residual_time_series(y_pred, y_truth, columns=["a", "b", "c"], facet_n_cols=3)
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
        fig = plot_residual_time_series(y_pred, y_truth, panel_group_names=["y"], columns=["a"])
        assert isinstance(fig, go.Figure)
        # Single panel column → 4-panel diagnostics (5 traces: scatter, scatter, hist, scatter, line)
        assert len(fig.data) >= 4

    def test_single_column_gives_four_panels(self, sample_residuals_data):
        """Test that single column still produces 4-panel diagnostics layout."""
        fig = plot_residual_time_series(sample_residuals_data["y_pred"], sample_residuals_data["y_truth"])
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
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # empirical + reference line

    def test_single_coverage(self, sample_interval_data):
        """Test calibration with single coverage rate."""
        fig = plot_calibration(
            sample_interval_data["y_pred_int"],
            sample_interval_data["y_truth"],
            coverage_rates=[0.9],
        )
        assert len(fig.data) == 2

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
        assert fig.layout.title.text == "Custom Cal"

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
        """Test default title is 'Calibration plot'."""
        fig = plot_calibration(
            sample_interval_data["y_pred_int"],
            sample_interval_data["y_truth"],
            coverage_rates=[0.9],
        )
        assert fig.layout.title.text == "Calibration plot"

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
        assert fig.layout.width == 800
        assert fig.layout.height == 600

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
            panel_group_names=["x", "y"],
        )
        # 2 panels × (1 empirical + 1 reference) = 4
        assert len(fig.data) == 4

    def test_panel_member_filter(self, panel_interval_data):
        """Test panel facets with member filter."""
        fig = plot_calibration(
            panel_interval_data["y_pred_int"],
            panel_interval_data["y_truth"],
            coverage_rates=[0.9],
            panel_group_names=["x"],
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
        assert isinstance(fig, go.Figure)
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
        assert fig.layout.title.text == "Custom Title"

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
        assert fig.layout.width == 900
        assert fig.layout.height == 500

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


class TestPlotModelComparisonBar:
    """Tests for plot_model_comparison_bar function."""

    def test_basic(self, sample_comparison_results):
        """Test basic model comparison bar chart."""
        fig = plot_model_comparison_bar(sample_comparison_results)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # 3 models

    def test_group_by_scorer(self, sample_comparison_results):
        """Test grouping by scorer (default)."""
        fig = plot_model_comparison_bar(sample_comparison_results, group_by="scorer")
        assert len(fig.data) == 3

    def test_group_by_model(self, sample_comparison_results):
        """Test grouping by model."""
        fig = plot_model_comparison_bar(sample_comparison_results, group_by="model")
        # 3 scorers as series
        assert len(fig.data) == 3

    def test_horizontal_orientation(self, sample_comparison_results):
        """Test horizontal bar chart."""
        fig = plot_model_comparison_bar(sample_comparison_results, orientation="horizontal")
        assert len(fig.data) == 3

    def test_invalid_group_by(self, sample_comparison_results):
        """Test error for invalid group_by."""
        with pytest.raises(ValueError, match="group_by must be one of"):
            plot_model_comparison_bar(sample_comparison_results, group_by="invalid")

    def test_invalid_orientation(self, sample_comparison_results):
        """Test error for invalid orientation."""
        with pytest.raises(ValueError, match="orientation must be one of"):
            plot_model_comparison_bar(sample_comparison_results, orientation="diagonal")

    def test_empty_results(self):
        """Test error for empty results."""
        with pytest.raises(ValueError, match="non-empty"):
            plot_model_comparison_bar({})

    def test_custom_title(self, sample_comparison_results):
        """Test custom title."""
        fig = plot_model_comparison_bar(sample_comparison_results, title="My Comparison")
        assert fig.layout.title.text == "My Comparison"

    def test_default_title(self, sample_comparison_results):
        """Test default title."""
        fig = plot_model_comparison_bar(sample_comparison_results)
        assert fig.layout.title.text == "Model Comparison"

    def test_sort_by(self, sample_comparison_results):
        """Test sorting by a model."""
        fig = plot_model_comparison_bar(
            sample_comparison_results,
            sort_by="LinearRegression",
            ascending=True,
        )
        assert len(fig.data) == 3

    def test_custom_color_palette(self, sample_comparison_results):
        """Test custom color palette."""
        fig = plot_model_comparison_bar(
            sample_comparison_results,
            color_palette=["#FF0000", "#00FF00", "#0000FF"],
        )
        assert len(fig.data) == 3

    def test_no_text_auto(self, sample_comparison_results):
        """Test without text annotations."""
        fig = plot_model_comparison_bar(
            sample_comparison_results,
            text_auto=False,
        )
        assert len(fig.data) == 3

    def test_custom_dimensions(self, sample_comparison_results):
        """Test custom dimensions."""
        fig = plot_model_comparison_bar(
            sample_comparison_results,
            width=1000,
            height=700,
        )
        assert fig.layout.width == 1000
        assert fig.layout.height == 700


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
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert any(isinstance(t, go.Histogram) for t in fig.data)

    def test_kde(self, sample_forecast_data):
        """Test KDE mode."""
        fig = plot_score_distribution(
            MeanAbsoluteError(),
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            kind="kde",
        )
        assert isinstance(fig, go.Figure)
        assert any(isinstance(t, go.Scatter) for t in fig.data)

    def test_both(self, sample_forecast_data):
        """Test combined histogram + KDE mode."""
        fig = plot_score_distribution(
            MeanAbsoluteError(),
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            kind="both",
        )
        assert isinstance(fig, go.Figure)
        assert any(isinstance(t, go.Histogram) for t in fig.data)
        assert any(isinstance(t, go.Scatter) for t in fig.data)

    def test_multi_model(self, multi_model_forecast_data):
        """Test multi-model overlay."""
        fig = plot_score_distribution(
            MeanAbsoluteError(),
            multi_model_forecast_data["y_truth"],
            multi_model_forecast_data["y_preds"],
        )
        assert isinstance(fig, go.Figure)
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
        assert isinstance(fig, go.Figure)
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
        assert isinstance(fig, go.Figure)

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
    """Tests for plot_score_per_horizon function."""

    def test_line(self, sample_forecast_data):
        """Test line chart mode."""
        fig = plot_score_per_horizon(
            MeanAbsoluteError(),
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            kind="line",
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert isinstance(fig.data[0], go.Scatter)

    def test_bar(self, sample_forecast_data):
        """Test bar chart mode."""
        fig = plot_score_per_horizon(
            MeanAbsoluteError(),
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            kind="bar",
        )
        assert isinstance(fig, go.Figure)
        assert any(isinstance(t, go.Bar) for t in fig.data)

    def test_multi_model(self, multi_model_forecast_data):
        """Test multi-model comparison."""
        fig = plot_score_per_horizon(
            MeanAbsoluteError(),
            multi_model_forecast_data["y_truth"],
            multi_model_forecast_data["y_preds"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_show_trend(self, sample_forecast_data):
        """Test trend line overlay."""
        fig = plot_score_per_horizon(
            MeanAbsoluteError(),
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            show_trend=True,
        )
        assert isinstance(fig, go.Figure)
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
            "observed_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "value": [12.0],
        })
        fig = plot_score_per_horizon(MeanAbsoluteError(), y_truth, y_pred)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_invalid_kind(self, sample_forecast_data):
        """Test that invalid kind raises ValueError."""
        with pytest.raises(ValueError, match="kind must be one of"):
            plot_score_per_horizon(
                MeanAbsoluteError(),
                sample_forecast_data["y_truth"],
                sample_forecast_data["y_pred"],
                kind="scatter",
            )

    def test_default_title(self, sample_forecast_data):
        """Test default title uses scorer name."""
        fig = plot_score_per_horizon(
            MeanAbsoluteError(),
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
        )
        assert "MeanAbsoluteError" in fig.layout.title.text

    def test_custom_palette(self, sample_forecast_data):
        """Test custom color palette."""
        fig = plot_score_per_horizon(
            MeanAbsoluteError(),
            sample_forecast_data["y_truth"],
            sample_forecast_data["y_pred"],
            color_palette=["#FF0000"],
        )
        assert isinstance(fig, go.Figure)
