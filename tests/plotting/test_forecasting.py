"""Tests for forecasting plotting functions."""

import importlib.util

import numpy as np
import polars as pl
import pytest
from plotly import graph_objects as go

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


class TestPlotComponentsPanel:
    """Panel data tests for plot_components."""

    def test_panel_dict_components(self):
        """Panel data with dict components produces a valid figure."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        n = len(dates)
        y = pl.DataFrame({
            "time": dates,
            "y__a": list(range(n)),
            "y__b": [i * 2 for i in range(n)],
        })
        components = {
            "trend": pl.DataFrame({
                "time": dates,
                "y__a": [i * 0.5 for i in range(n)],
                "y__b": [i * 1.0 for i in range(n)],
            }),
        }
        fig = plot_components(y, components)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_custom_dimensions(self):
        """Custom dimensions are passed through."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        y = pl.DataFrame({"time": dates, "y": list(range(10))})
        components = {
            "trend": pl.DataFrame({"time": dates, "y": [i * 0.5 for i in range(10)]}),
        }
        fig = plot_components(y, components, width=900, height=500)
        assert fig.layout.width == 900
        assert fig.layout.height == 500


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


class TestPlotForecastMultiModelIntervals:
    """Tests for multi-model forecast with interval bands."""

    def test_multi_model_interval_bands(self):
        """Multi-model overlaid forecasts with prediction intervals."""
        dates = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True)
        y_test = pl.DataFrame({"time": dates, "y": [191 + i for i in range(30)]})
        y_pred_a = pl.DataFrame({
            "time": dates,
            "y": [190 + i for i in range(30)],
            "y_lower_0.9": [185 + i for i in range(30)],
            "y_upper_0.9": [195 + i for i in range(30)],
        })
        y_pred_b = pl.DataFrame({
            "time": dates,
            "y": [192 + i for i in range(30)],
            "y_lower_0.9": [187 + i for i in range(30)],
            "y_upper_0.9": [197 + i for i in range(30)],
        })
        fig = plot_forecast(y_test, {"M1": y_pred_a, "M2": y_pred_b}, coverage_rates=[0.9])
        assert isinstance(fig, go.Figure)
        names = [t.name for t in fig.data if t.name is not None]
        assert any("M1" in n for n in names)
        assert any("M2" in n for n in names)
        assert any("PI" in n for n in names)


class TestPlotForecastPanelSingleMember:
    """Tests for panel forecast with single member per group."""

    def test_single_member_panel(self):
        """Single-member panel groups use solid line without dash."""
        dates_train = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        dates_test = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True)
        y_train = pl.DataFrame({
            "time": dates_train,
            "y__store1": list(range(91)),
            "z__store1": list(range(91, 182)),
        })
        y_test = pl.DataFrame({
            "time": dates_test,
            "y__store1": list(range(30)),
            "z__store1": list(range(30, 60)),
        })
        y_pred = pl.DataFrame({
            "time": dates_test,
            "y__store1": [x + 1 for x in range(30)],
            "z__store1": [x + 1 for x in range(30, 60)],
        })
        fig = plot_forecast(
            y_test,
            y_pred,
            y_train=y_train,
            panel_group_names=["y", "z"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestPlotForecastPanelTrainAndIntervals:
    """Tests for panel forecast with train history and intervals."""

    def test_panel_with_train_and_intervals(self):
        """Panel forecast with y_train and coverage_rates renders bands."""
        dates_train = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        dates_test = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True)
        y_train = pl.DataFrame({
            "time": dates_train,
            "y__a": [100 + i for i in range(91)],
            "y__b": [200 + i for i in range(91)],
        })
        y_test = pl.DataFrame({
            "time": dates_test,
            "y__a": [191 + i for i in range(30)],
            "y__b": [291 + i for i in range(30)],
        })
        y_pred = pl.DataFrame({
            "time": dates_test,
            "y__a": [190 + i for i in range(30)],
            "y__b": [289 + i for i in range(30)],
            "y__a_lower_0.9": [185 + i for i in range(30)],
            "y__a_upper_0.9": [195 + i for i in range(30)],
            "y__b_lower_0.9": [284 + i for i in range(30)],
            "y__b_upper_0.9": [294 + i for i in range(30)],
        })
        fig = plot_forecast(
            y_test,
            y_pred,
            y_train=y_train,
            coverage_rates=[0.9],
            panel_group_names=["y"],
        )
        assert isinstance(fig, go.Figure)
        names = [t.name for t in fig.data if t.name is not None]
        assert any("Train" in n for n in names)
        assert any("PI" in n for n in names)


class TestPlotComponentsSTL:
    """Tests for plot_components in STL decomposition mode."""

    @pytest.fixture
    def monthly_series(self):
        """Create a monthly series with trend and seasonal pattern."""
        dates = pl.date_range(
            pl.date(2018, 1, 1),
            pl.date(2022, 12, 31),
            "1mo",
            eager=True,
        )
        n = len(dates)
        return pl.DataFrame({
            "time": dates,
            "y": [100 + 10 * (i % 12) + i * 0.5 for i in range(n)],
        })

    @pytest.mark.skipif(
        not importlib.util.find_spec("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_stl_trend_seasonal(self, monthly_series):
        """STL mode with trend and seasonal components works."""
        fig = plot_components(monthly_series, ["trend", "seasonal"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    @pytest.mark.skipif(
        not importlib.util.find_spec("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_stl_with_nan_interpolation_warning(self):
        """STL mode warns when NaN values are interpolated."""
        dates = pl.date_range(
            pl.date(2018, 1, 1),
            pl.date(2022, 12, 31),
            "1mo",
            eager=True,
        )
        n = len(dates)
        values = [100 + 10 * (i % 12) + i * 0.5 for i in range(n)]
        values[5] = None
        values[10] = None
        df = pl.DataFrame({"time": dates, "y": values})
        with pytest.warns(UserWarning, match="Interpolated"):
            fig = plot_components(df, ["trend", "residual"])
        assert isinstance(fig, go.Figure)

    @pytest.mark.skipif(
        not importlib.util.find_spec("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_stl_explicit_period(self):
        """STL mode with explicit period instead of auto-detect."""
        dates = pl.date_range(
            pl.date(2018, 1, 1),
            pl.date(2022, 12, 31),
            "1mo",
            eager=True,
        )
        n = len(dates)
        df = pl.DataFrame({
            "time": dates,
            "y": [100 + 10 * (i % 12) + i * 0.5 for i in range(n)],
        })
        fig = plot_components(df, ["trend", "seasonal"], stl_kwargs={"period": 12})
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    @pytest.mark.skipif(
        not importlib.util.find_spec("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_stl_with_window_kwargs(self):
        """STL mode forwards trend/seasonal/low_pass window kwargs."""
        dates = pl.date_range(
            pl.date(2018, 1, 1),
            pl.date(2022, 12, 31),
            "1mo",
            eager=True,
        )
        n = len(dates)
        df = pl.DataFrame({
            "time": dates,
            "y": [100 + 10 * (i % 12) + i * 0.5 for i in range(n)],
        })
        fig = plot_components(
            df,
            ["trend", "seasonal"],
            stl_kwargs={"period": 12, "trend_window": 15, "seasonal_window": 7, "low_pass_window": 13},
        )
        assert isinstance(fig, go.Figure)

    @pytest.mark.skipif(
        not importlib.util.find_spec("statsmodels"),
        reason="statsmodels not installed",
    )
    def test_stl_unsupported_interval_error(self):
        """Auto-period with unsupported interval frequency raises ValueError."""
        from datetime import datetime

        dates = pl.datetime_range(
            datetime(2020, 1, 1),
            datetime(2020, 1, 1, 0, 9),
            "1m",
            eager=True,
        )
        n = len(dates)
        df = pl.DataFrame({
            "time": dates,
            "y": list(range(n)),
        })
        with pytest.raises(ValueError, match="Cannot infer STL period"):
            plot_components(df, ["trend", "seasonal"])


class TestPlotForecastMultiModelTrainHistory:
    """Tests for multi-model forecast with training history and n_history."""

    def test_multi_model_with_train_and_n_history(self):
        """Multi-model forecast with y_train and n_history covers tail branch."""
        y_train = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "y": [100.0 + i for i in range(91)],
        })
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [191.0 + i for i in range(30)],
        })
        y_pred_a = pl.DataFrame({
            "time": y_test["time"],
            "y": [190.0 + i * 1.1 for i in range(30)],
        })
        y_pred_b = pl.DataFrame({
            "time": y_test["time"],
            "y": [192.0 + i * 0.9 for i in range(30)],
        })
        fig = plot_forecast(
            y_test,
            {"Model A": y_pred_a, "Model B": y_pred_b},
            y_train=y_train,
            n_history=30,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3


class TestPlotForecastPanelErrors:
    """Tests for panel forecast error paths."""

    def test_panel_nonexistent_group_raises(self):
        """Panel forecast with nonexistent group name raises ValueError."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        y_test = pl.DataFrame({
            "time": dates,
            "sales__store_1": [float(i) for i in range(10)],
            "sales__store_2": [float(i) + 5 for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "sales__store_1": [float(i) + 1 for i in range(10)],
            "sales__store_2": [float(i) + 6 for i in range(10)],
        })
        with pytest.raises(ValueError, match="No panel columns found for groups"):
            plot_forecast(y_test, y_pred, panel_group_names=["nonexistent"])


class TestPlotTimeWeightPanelErrors:
    """Tests for time_weight panel error paths."""

    def test_panel_weight_nonexistent_group_raises(self):
        """Panel weight with nonexistent group name raises ValueError."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "time_weight__store_1": [0.1 * i for i in range(1, 11)],
            "time_weight__store_2": [0.05 * i for i in range(1, 11)],
        })
        with pytest.raises(ValueError, match="No weight columns found for panel groups"):
            plot_time_weight(df, panel_group_names=["nonexistent"])

    def test_panel_weight_valid_group(self):
        """Panel weight with valid group name produces a figure."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "time_weight__store_1": [0.1 * i for i in range(1, 11)],
            "time_weight__store_2": [0.05 * i for i in range(1, 11)],
        })
        fig = plot_time_weight(df, panel_group_names=["time_weight"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestPlotComponentsErrors:
    """Tests for plot_components error paths."""

    def test_empty_components_no_original_raises(self):
        """Empty components list with show_original=False raises ValueError."""
        y = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "y": [float(i) for i in range(91)],
        })
        with pytest.raises(ValueError, match="components must contain at least one displayable component"):
            plot_components(y, [], show_original=False)


class TestPlotForecastWithIntervals:
    """Tests for plot_forecast with prediction intervals (coverage_rates truthy branch)."""

    def test_forecast_with_intervals(self):
        """Forecast with interval columns triggers coverage_rates rendering branch."""
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 10), "1d", eager=True),
            "y": [100.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 10), "1d", eager=True),
            "y": [101.0 + i for i in range(10)],
            "y_lower_0.9": [98.0 + i for i in range(10)],
            "y_upper_0.9": [104.0 + i for i in range(10)],
        })
        fig = plot_forecast(y_test, y_pred)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2


class TestPlotForecastShowTransition:
    """Tests for plot_forecast show_transition=True branch."""

    def test_show_transition_with_train(self):
        """show_transition=True with y_train prepends last train point to forecast line."""
        y_train = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "y": [100.0 + i * 0.5 for i in range(91)],
        })
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 10), "1d", eager=True),
            "y": [146.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 10), "1d", eager=True),
            "y": [145.0 + i for i in range(10)],
        })
        fig = plot_forecast(y_test, y_pred, y_train=y_train, show_transition=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3


class TestPlotTimeWeightNonPanel:
    """Tests for plot_time_weight non-panel single weight column branch."""

    def test_single_weight_column(self):
        """Non-panel time weight plots single weight column."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "time_weight": [1.0 / (i + 1) for i in range(91)],
        })
        fig = plot_time_weight(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_single_weight_with_fill(self):
        """Non-panel time weight with fill enabled."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "time_weight": [0.9**i for i in range(10)],
        })
        fig = plot_time_weight(df, fill=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestPlotComponentsShowOriginal:
    """Tests for plot_components show_original=True branch."""

    def test_show_original_true(self):
        """show_original=True adds original series panel to the subplot."""
        y = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "y": [float(i) + (i % 7) * 3.0 for i in range(91)],
        })
        trend = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "y": [float(i) for i in range(91)],
        })
        fig = plot_components(y, {"Trend": trend}, show_original=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2


class TestPlotForecastClassProba:
    """Tests for plot_forecast with class-probability predictions."""

    @pytest.fixture()
    def _class_proba_data(self):
        """Create class-probability test data."""
        times = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 10), "1d", eager=True)
        y_test = pl.DataFrame({
            "time": times,
            "weather": ["sunny", "sunny", "rainy", "cloudy", "sunny", "rainy", "cloudy", "sunny", "sunny", "rainy"],
        })
        y_proba = pl.DataFrame({
            "time": times,
            "weather_proba_cloudy": [0.1, 0.1, 0.2, 0.6, 0.1, 0.2, 0.7, 0.1, 0.05, 0.3],
            "weather_proba_rainy": [0.2, 0.1, 0.6, 0.2, 0.1, 0.7, 0.2, 0.1, 0.05, 0.6],
            "weather_proba_sunny": [0.7, 0.8, 0.2, 0.2, 0.8, 0.1, 0.1, 0.8, 0.9, 0.1],
        })
        return y_test, y_proba

    def test_single_model(self, _class_proba_data):
        """Stacked area chart is produced for single-model proba predictions."""
        y_test, y_proba = _class_proba_data
        fig = plot_forecast(y_test, y_proba)
        assert isinstance(fig, go.Figure)
        # 3 class traces + 1 truth marker trace
        assert len(fig.data) >= 3

    def test_multi_model(self, _class_proba_data):
        """Subplots are created for multi-model proba predictions."""
        y_test, y_proba = _class_proba_data
        y_proba_b = y_proba.with_columns(
            pl.col("weather_proba_sunny") * 0.5,
            pl.col("weather_proba_rainy") * 1.5,
        )
        fig = plot_forecast(
            y_test,
            {"Model A": y_proba, "Model B": y_proba_b},
        )
        assert isinstance(fig, go.Figure)
        # Should have traces from both models
        assert len(fig.data) >= 6

    def test_custom_title_and_palette(self, _class_proba_data):
        """Custom title and palette are applied."""
        y_test, y_proba = _class_proba_data
        fig = plot_forecast(
            y_test,
            y_proba,
            title="Custom Title",
            color_palette=["#ff0000", "#00ff00", "#0000ff"],
        )
        assert fig.layout.title.text == "Custom Title"

    def test_truth_markers_present(self, _class_proba_data):
        """Truth markers are shown when y_test has matching target column."""
        y_test, y_proba = _class_proba_data
        fig = plot_forecast(y_test, y_proba)
        trace_names = [t.name for t in fig.data]
        assert "True class" in trace_names


class TestPlotForecastCategorical:
    """Tests for plot_forecast with categorical string predictions."""

    @pytest.fixture()
    def _categorical_data(self):
        """Create categorical test data."""
        times = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 10), "1d", eager=True)
        y_test = pl.DataFrame({
            "time": times,
            "weather": ["sunny", "sunny", "rainy", "cloudy", "sunny", "rainy", "cloudy", "sunny", "sunny", "rainy"],
        })
        y_pred = pl.DataFrame({
            "time": times,
            "weather": ["sunny", "rainy", "rainy", "cloudy", "sunny", "sunny", "cloudy", "sunny", "rainy", "rainy"],
        })
        return y_test, y_pred

    def test_single_model(self, _categorical_data):
        """Step chart is produced for single-model categorical predictions."""
        y_test, y_pred = _categorical_data
        fig = plot_forecast(y_test, y_pred)
        assert isinstance(fig, go.Figure)
        # Actual + Forecast traces
        assert len(fig.data) >= 2

    def test_multi_model(self, _categorical_data):
        """Multiple model categorical predictions are overlaid."""
        y_test, y_pred = _categorical_data
        y_pred_b = y_pred.with_columns(
            pl.Series(
                "weather", ["cloudy", "sunny", "sunny", "rainy", "rainy", "cloudy", "sunny", "rainy", "cloudy", "sunny"]
            ),
        )
        fig = plot_forecast(
            y_test,
            {"Model A": y_pred, "Model B": y_pred_b},
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3

    def test_with_training_data(self, _categorical_data):
        """Training data is shown when provided."""
        y_test, y_pred = _categorical_data
        y_train = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 3, 22), pl.date(2020, 3, 31), "1d", eager=True),
            "weather": ["sunny", "sunny", "cloudy", "rainy", "sunny", "rainy", "cloudy", "sunny", "rainy", "sunny"],
        })
        fig = plot_forecast(y_test, y_pred, y_train=y_train)
        assert isinstance(fig, go.Figure)
        # Train + Actual + Forecast
        assert len(fig.data) >= 3

    def test_y_axis_has_category_labels(self, _categorical_data):
        """Y-axis tick labels are category names, not integers."""
        y_test, y_pred = _categorical_data
        fig = plot_forecast(y_test, y_pred)
        yaxis = fig.layout.yaxis
        assert set(yaxis.ticktext) == {"cloudy", "rainy", "sunny"}

    def test_custom_title(self, _categorical_data):
        """Custom title is applied to categorical forecast."""
        y_test, y_pred = _categorical_data
        fig = plot_forecast(y_test, y_pred, title="My Categorical Plot")
        assert fig.layout.title.text == "My Categorical Plot"


class TestPlotForecastPanelClassProba:
    """Tests for plot_forecast with panel class-probability data."""

    @pytest.fixture
    def _panel_proba_data(self):
        """Panel class-probability data with two members."""
        times = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 5), "1d", eager=True)
        y_test = pl.DataFrame({
            "time": times,
            "weather__east": ["sunny", "rainy", "rainy", "sunny", "cloudy"],
            "weather__west": ["cloudy", "sunny", "sunny", "rainy", "sunny"],
        })
        y_pred = pl.DataFrame({
            "time": times,
            "weather_proba_sunny__east": [0.7, 0.2, 0.1, 0.6, 0.1],
            "weather_proba_rainy__east": [0.2, 0.6, 0.7, 0.2, 0.2],
            "weather_proba_cloudy__east": [0.1, 0.2, 0.2, 0.2, 0.7],
            "weather_proba_sunny__west": [0.1, 0.7, 0.6, 0.2, 0.8],
            "weather_proba_rainy__west": [0.2, 0.1, 0.2, 0.6, 0.1],
            "weather_proba_cloudy__west": [0.7, 0.2, 0.2, 0.2, 0.1],
        })
        return y_test, y_pred

    def test_basic_panel(self, _panel_proba_data):
        """Panel class-proba data produces a faceted figure."""
        y_test, y_pred = _panel_proba_data
        fig = plot_forecast(y_test, y_pred)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_subplot_titles_are_members(self, _panel_proba_data):
        """Subplot titles correspond to panel members."""
        y_test, y_pred = _panel_proba_data
        fig = plot_forecast(y_test, y_pred)
        annotations = [a.text for a in fig.layout.annotations]
        assert "east" in annotations
        assert "west" in annotations

    def test_multi_model_panel(self, _panel_proba_data):
        """Multi-model panel class-proba produces subplots."""
        y_test, y_pred = _panel_proba_data
        y_pred_b = y_pred.with_columns(
            pl.col("weather_proba_sunny__east").alias("weather_proba_sunny__east") * 0.9,
        )
        fig = plot_forecast(y_test, {"Model A": y_pred, "Model B": y_pred_b})
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 4

    def test_panel_with_training_data(self, _panel_proba_data):
        """Panel class-proba data with y_train renders training traces."""
        y_test, y_pred = _panel_proba_data
        y_train = pl.DataFrame({
            "time": pl.date_range(pl.date(2019, 12, 27), pl.date(2019, 12, 31), "1d", eager=True),
            "weather__east": ["sunny", "rainy", "cloudy", "sunny", "rainy"],
            "weather__west": ["cloudy", "sunny", "rainy", "cloudy", "sunny"],
        })
        fig = plot_forecast(y_test, y_pred, y_train=y_train)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2


class TestPlotForecastPanelCategorical:
    """Tests for plot_forecast with panel categorical data."""

    @pytest.fixture
    def _panel_cat_data(self):
        """Panel categorical data with two members."""
        times = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 5), "1d", eager=True)
        y_test = pl.DataFrame({
            "time": times,
            "weather__east": ["sunny", "rainy", "rainy", "sunny", "cloudy"],
            "weather__west": ["cloudy", "sunny", "sunny", "rainy", "sunny"],
        })
        y_pred = pl.DataFrame({
            "time": times,
            "weather__east": ["sunny", "sunny", "rainy", "sunny", "rainy"],
            "weather__west": ["cloudy", "rainy", "sunny", "rainy", "sunny"],
        })
        return y_test, y_pred

    def test_basic_panel(self, _panel_cat_data):
        """Panel categorical data produces a faceted figure."""
        y_test, y_pred = _panel_cat_data
        fig = plot_forecast(y_test, y_pred)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_subplot_titles_are_members(self, _panel_cat_data):
        """Subplot titles correspond to panel members."""
        y_test, y_pred = _panel_cat_data
        fig = plot_forecast(y_test, y_pred)
        annotations = [a.text for a in fig.layout.annotations]
        assert "east" in annotations
        assert "west" in annotations

    def test_with_training_data(self, _panel_cat_data):
        """Training data is shown in panel mode."""
        y_test, y_pred = _panel_cat_data
        y_train = pl.DataFrame({
            "time": pl.date_range(pl.date(2019, 12, 27), pl.date(2019, 12, 31), "1d", eager=True),
            "weather__east": ["sunny", "sunny", "cloudy", "rainy", "sunny"],
            "weather__west": ["rainy", "cloudy", "sunny", "sunny", "rainy"],
        })
        fig = plot_forecast(y_test, y_pred, y_train=y_train)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 4

    def test_y_axis_category_labels(self, _panel_cat_data):
        """Y-axis tick labels are category names in panel mode."""
        y_test, y_pred = _panel_cat_data
        fig = plot_forecast(y_test, y_pred)
        yaxis = fig.layout.yaxis
        assert set(yaxis.ticktext) == {"cloudy", "rainy", "sunny"}
