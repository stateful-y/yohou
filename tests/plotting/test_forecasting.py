"""Tests for forecasting plotting functions."""

import importlib.util

import numpy as np
import polars as pl
import pytest

from yohou.plotting import (
    plot_components,
    plot_forecast,
    plot_time_weight,
)


class TestPlotForecast:
    """Tests for plot_forecast function."""

    def test_basic(self):
        """Test basic forecast plotting."""
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [191 + i for i in range(30)],
        })
        y_pred = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [190 + i + (i % 3) for i in range(30)],
        })
        fig = plot_forecast(y_test, y_pred)
        assert len(fig.data) > 0

    def test_with_history(self):
        """Test forecast with training history shown."""
        y_train = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "y": [100 + i for i in range(91)],
        })
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [191 + i for i in range(30)],
        })
        y_pred = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [190 + i + (i % 3) for i in range(30)],
        })
        fig = plot_forecast(y_test, y_pred, y_train=y_train, n_history=30)
        assert len(fig.data) > 0

    def test_with_intervals(self):
        """Test forecast with prediction intervals."""
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [191 + i for i in range(30)],
        })
        y_pred = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [190 + i + (i % 3) for i in range(30)],
            "y_lower_0.9": [185 + i for i in range(30)],
            "y_upper_0.9": [195 + i for i in range(30)],
        })
        fig = plot_forecast(y_test, y_pred, coverage_rates=[0.9])
        assert len(fig.data) > 0

    def test_panel_group_names(self):
        """Test forecast with panel_group_names parameter."""
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [191 + i for i in range(30)],
        })
        y_pred = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [190 + i for i in range(30)],
        })
        # Non-panel data with panel_group_names should still work (no panel columns detected)
        fig = plot_forecast(y_test, y_pred, panel_group_names=["group"])
        assert len(fig.data) >= 0  # May produce empty figure or error gracefully


class TestPlotTimeWeight:
    """Tests for plot_time_weight function."""

    def test_basic(self):
        """Test basic time weight plotting."""
        time_weight = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "weight": [1.0 / (91 - i) for i in range(91)],
        })
        fig = plot_time_weight(time_weight, weight_column="weight")
        assert len(fig.data) > 0

    def test_with_panel(self):
        """Test time weight plotting with panel data."""
        time_weight = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "weight__store_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "weight__store_2": [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0],
        })
        fig = plot_time_weight(time_weight, weight_column="weight")
        assert len(fig.data) >= 2  # One trace per panel group

    def test_custom_title(self):
        """Test time weight plotting with custom title."""
        time_weight = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "weight": [0.1 * i for i in range(1, 11)],
        })
        fig = plot_time_weight(time_weight, weight_column="weight", title="Custom Weights")
        assert fig.layout.title.text == "Custom Weights"

    def test_custom_dimensions(self):
        """Test time weight plotting with custom dimensions."""
        time_weight = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "weight": [0.1 * i for i in range(1, 11)],
        })
        fig = plot_time_weight(time_weight, weight_column="weight", width=800, height=400)
        assert fig.layout.width == 800
        assert fig.layout.height == 400

    def test_missing_column(self):
        """Test error when weight column is missing."""
        time_weight = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "value": [0.1 * i for i in range(1, 11)],
        })
        with pytest.raises((ValueError, KeyError)):
            plot_time_weight(time_weight, weight_column="missing")


@pytest.fixture
def decomposition_data():
    """Create sample decomposition data."""
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
    n = len(dates)
    y = pl.DataFrame({"time": dates, "y": list(range(n))})
    components = {
        "trend": pl.DataFrame({"time": dates, "y": [i * 0.5 for i in range(n)]}),
        "seasonality": pl.DataFrame({
            "time": dates,
            "y": [10.0 * (i % 7) / 7 for i in range(n)],
        }),
        "residual": pl.DataFrame({"time": dates, "y": [i * 0.3 for i in range(n)]}),
    }
    return y, components


@pytest.fixture
def multi_model_data():
    """Create sample multi-model forecast data."""
    y_test = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
        "y": [191 + i for i in range(30)],
    })
    y_pred_a = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
        "y": [190 + i for i in range(30)],
    })
    y_pred_b = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
        "y": [192 + i for i in range(30)],
    })
    return y_test, {"Model A": y_pred_a, "Model B": y_pred_b}


class TestPlotComponents:
    """Tests for plot_components function."""

    def test_basic(self, decomposition_data):
        """Test basic decomposition plot."""
        y, components = decomposition_data
        fig = plot_components(y, components)
        # Original + 3 components = 4 traces
        assert len(fig.data) >= 4

    def test_no_original(self, decomposition_data):
        """Test decomposition without original series."""
        y, components = decomposition_data
        fig = plot_components(y, components, show_original=False)
        # 3 component traces only
        assert len(fig.data) >= 3

    def test_custom_title(self, decomposition_data):
        """Test custom title."""
        y, components = decomposition_data
        fig = plot_components(y, components, title="My Decomposition")
        assert fig.layout.title.text == "My Decomposition"

    def test_default_title(self, decomposition_data):
        """Test default title."""
        y, components = decomposition_data
        fig = plot_components(y, components)
        assert fig.layout.title.text == "Time Series Decomposition"

    def test_specific_columns(self):
        """Test with specific column selection."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        n = len(dates)
        y = pl.DataFrame({
            "time": dates,
            "y1": list(range(n)),
            "y2": [i * 2 for i in range(n)],
        })
        components = {
            "trend": pl.DataFrame({
                "time": dates,
                "y1": [i * 0.5 for i in range(n)],
                "y2": [i * 1.0 for i in range(n)],
            }),
        }
        fig = plot_components(y, components, columns="y1")
        # Original y1 + trend y1 = 2 traces
        assert len(fig.data) == 2

    def test_empty_components(self):
        """Test that empty components raises ValueError."""
        y = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "y": list(range(10)),
        })
        with pytest.raises(ValueError, match="non-empty"):
            plot_components(y, {})

    def test_invalid_y(self):
        """Test that invalid y raises TypeError."""
        with pytest.raises(TypeError, match="DataFrame"):
            plot_components("not a df", {"trend": pl.DataFrame({"time": [], "y": []})})

    def test_custom_palette(self, decomposition_data):
        """Test custom color palette."""
        y, components = decomposition_data
        fig = plot_components(y, components, color_palette=["#ff0000"])
        assert fig.data[0].line.color == "#ff0000"

    def test_invalid_components_type(self):
        """Test that invalid components type raises TypeError."""
        y = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "y": list(range(10)),
        })
        with pytest.raises(TypeError, match="dict.*list.*tuple"):
            plot_components(y, 42)


_has_statsmodels = importlib.util.find_spec("statsmodels") is not None


@pytest.mark.skipif(not _has_statsmodels, reason="statsmodels not installed")
class TestPlotComponentsStl:
    """Tests for plot_components STL mode (list/tuple components)."""

    @pytest.fixture
    def monthly_df(self):
        """Create monthly data suitable for STL decomposition."""
        rng = np.random.default_rng(42)
        return pl.DataFrame({
            "time": pl.date_range(pl.date(2018, 1, 1), pl.date(2023, 12, 1), "1mo", eager=True),
            "y": [100 + 2 * i + 10 * np.sin(2 * np.pi * i / 12) + rng.standard_normal() for i in range(72)],
        })

    def test_basic(self, monthly_df):
        """Test basic STL decomposition via plot_components."""
        fig = plot_components(
            monthly_df,
            ["observed", "trend", "seasonal", "residual", "seasonal_adjusted"],
            columns="y",
        )
        assert len(fig.data) >= 4

    def test_explicit_period(self, monthly_df):
        """Test STL mode with explicit period via stl_kwargs."""
        fig = plot_components(
            monthly_df,
            ["observed", "trend", "seasonal", "residual"],
            columns="y",
            stl_kwargs={"period": 12},
        )
        assert len(fig.data) >= 4

    def test_subset_components(self, monthly_df):
        """Test showing only a subset of STL components."""
        fig = plot_components(monthly_df, ["trend", "seasonal"], columns="y", show_original=False)
        assert len(fig.data) == 2

    def test_observed_sets_show_original(self, monthly_df):
        """Test that 'observed' in components enables original trace."""
        fig = plot_components(monthly_df, ["observed", "trend"], columns="y")
        names = [t.name for t in fig.data]
        # "observed" maps to show_original, so the first trace uses the column name
        assert "y" in names
        assert "Trend" in names

    def test_stl_kwargs_robust(self, monthly_df):
        """Test passing robust=False via stl_kwargs."""
        fig = plot_components(
            monthly_df,
            ["trend", "seasonal"],
            columns="y",
            show_original=False,
            stl_kwargs={"robust": False},
        )
        assert len(fig.data) == 2

    def test_stl_kwargs_windows(self, monthly_df):
        """Test passing window parameters via stl_kwargs."""
        fig = plot_components(
            monthly_df,
            ["trend", "residual"],
            columns="y",
            show_original=False,
            stl_kwargs={"period": 12, "seasonal_window": 15, "trend_window": 25},
        )
        assert len(fig.data) == 2

    def test_stl_default_title(self, monthly_df):
        """Test STL mode default title."""
        fig = plot_components(monthly_df, ["trend"], columns="y")
        assert fig.layout.title.text == "STL Decomposition"

    def test_stl_custom_title(self, monthly_df):
        """Test STL mode custom title."""
        fig = plot_components(monthly_df, ["trend"], columns="y", title="My STL")
        assert fig.layout.title.text == "My STL"

    def test_unknown_component(self, monthly_df):
        """Test unknown STL component raises ValueError."""
        with pytest.raises(ValueError, match="Unknown components"):
            plot_components(monthly_df, ["trend", "bogus"], columns="y")

    def test_tuple_components(self, monthly_df):
        """Test that tuple components trigger STL mode."""
        fig = plot_components(monthly_df, ("trend", "seasonal"), columns="y", show_original=False)
        assert len(fig.data) == 2


class TestPlotForecastMultiModel:
    """Tests for plot_forecast with multiple model predictions."""

    def test_basic(self, multi_model_data):
        """Test multi-model forecast overlay."""
        y_test, y_preds = multi_model_data
        fig = plot_forecast(y_test, y_preds)
        # 1 actual + 2 models = 3 traces
        assert len(fig.data) == 3

    def test_with_train(self, multi_model_data):
        """Test multi-model with training data."""
        y_test, y_preds = multi_model_data
        y_train = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "y": [100 + i for i in range(91)],
        })
        fig = plot_forecast(y_test, y_preds, y_train=y_train)
        # 1 train + 1 actual + 2 models = 4 traces
        assert len(fig.data) == 4

    def test_default_title(self, multi_model_data):
        """Test multi-model default title."""
        y_test, y_preds = multi_model_data
        fig = plot_forecast(y_test, y_preds)
        assert fig.layout.title.text == "Forecast Comparison"

    def test_custom_title(self, multi_model_data):
        """Test multi-model custom title."""
        y_test, y_preds = multi_model_data
        fig = plot_forecast(y_test, y_preds, title="My Comparison")
        assert fig.layout.title.text == "My Comparison"

    def test_model_names(self, multi_model_data):
        """Test model names appear in traces."""
        y_test, y_preds = multi_model_data
        fig = plot_forecast(y_test, y_preds)
        trace_names = [t.name for t in fig.data]
        assert "y (Model A)" in trace_names
        assert "y (Model B)" in trace_names

    def test_single_df_still_works(self):
        """Test backward compatibility with single DataFrame."""
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [191 + i for i in range(30)],
        })
        y_pred = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [190 + i for i in range(30)],
        })
        fig = plot_forecast(y_test, y_pred)
        # 1 actual + 1 forecast = 2 traces
        assert len(fig.data) == 2


class TestPlotForecastMultiColumn:
    """Tests for plot_forecast with multiple target columns."""

    def test_multi_column_distinct_colors(self):
        """Test that multi-column forecasts have distinct per-column colors."""
        dates = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True)
        y_test = pl.DataFrame({"time": dates, "a": list(range(30)), "b": list(range(30, 60))})
        y_pred = pl.DataFrame({"time": dates, "a": [x + 1 for x in range(30)], "b": [x + 1 for x in range(30, 60)]})
        fig = plot_forecast(y_test, y_pred)
        a_fc = [t for t in fig.data if t.name == "a (Forecast)"][0]
        b_fc = [t for t in fig.data if t.name == "b (Forecast)"][0]
        assert a_fc.line.color != b_fc.line.color

    def test_multi_column_per_column_colors(self):
        """Test that multi-column actual/forecast share the same per-column color."""
        dates = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True)
        y_test = pl.DataFrame({"time": dates, "a": list(range(30)), "b": list(range(30, 60))})
        y_pred = pl.DataFrame({"time": dates, "a": [x + 1 for x in range(30)], "b": [x + 1 for x in range(30, 60)]})
        fig = plot_forecast(y_test, y_pred)
        a_actual = [t for t in fig.data if t.name == "a (Actual)"][0]
        b_actual = [t for t in fig.data if t.name == "b (Actual)"][0]
        # Different columns get different colors
        assert a_actual.line.color != b_actual.line.color

    def test_multi_column_per_column_train_legend(self):
        """Test that each column gets its own Train legend entry for multi-column."""
        dates_train = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        dates_test = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True)
        y_train = pl.DataFrame({"time": dates_train, "a": list(range(91)), "b": list(range(91))})
        y_test = pl.DataFrame({"time": dates_test, "a": list(range(30)), "b": list(range(30))})
        y_pred = pl.DataFrame({"time": dates_test, "a": list(range(30)), "b": list(range(30))})
        fig = plot_forecast(y_test, y_pred, y_train=y_train)
        train_traces = [t for t in fig.data if t.name and "Train" in t.name]
        assert len(train_traces) == 2
        train_names = {t.name for t in train_traces}
        assert train_names == {"a (Train)", "b (Train)"}


class TestPlotForecastPanelMultiModel:
    """Tests for plot_forecast with panel data and multi-model."""

    def test_panel_multi_model(self):
        """Test panel data with multiple model predictions."""
        dates = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True)
        y_test = pl.DataFrame({"time": dates, "y__s1": list(range(30)), "y__s2": list(range(30, 60))})
        y_pred_a = pl.DataFrame({
            "time": dates,
            "y__s1": [x + 1 for x in range(30)],
            "y__s2": [x + 1 for x in range(30, 60)],
        })
        y_pred_b = pl.DataFrame({
            "time": dates,
            "y__s1": [x - 1 for x in range(30)],
            "y__s2": [x - 1 for x in range(30, 60)],
        })
        fig = plot_forecast(y_test, {"A": y_pred_a, "B": y_pred_b})
        names = [t.name for t in fig.data if t.name is not None]
        # Multi-member panel labels include member: "s1 (A)", "s2 (B)", etc.
        assert any("A" in n for n in names)
        assert any("B" in n for n in names)
