"""Tests for plotly-resampler integration: config system, figure factories, and parameter threading."""

import polars as pl
import pytest
from plotly import graph_objects as go
from plotly_resampler import FigureResampler, FigureWidgetResampler
from plotly_resampler.aggregation.aggregators import LTTB, MinMaxAggregator
from plotly_resampler.aggregation.gap_handlers import NoGapHandler

from yohou.plotting._utils import (
    _build_resampler_kwargs,
    _create_figure,
    _create_subplots,
    _fill_trace_kwargs,
    _get_resampler_mode,
    _global_config,
    config_context,
    get_config,
    set_config,
)

_DEFAULT_CONFIG = {
    "resampler": False,
    "resampler_n_shown_samples": None,
    "resampler_downsampler": None,
    "resampler_gap_handler": None,
    "resampler_trace_prefix_suffix": None,
    "resampler_show_mean_aggregation_size": None,
}


# Fixtures


@pytest.fixture(autouse=True)
def _reset_config():
    """Ensure every test starts and ends with the default config."""
    old = _global_config.copy()
    _global_config.clear()
    _global_config.update(_DEFAULT_CONFIG)
    yield
    _global_config.clear()
    _global_config.update(old)


@pytest.fixture
def monthly_1col_df():
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1mo", eager=True),
        "y": [100, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180, 175],
    })


# Config system


class TestGetConfig:
    def test_default(self):
        assert get_config() == _DEFAULT_CONFIG

    def test_returns_copy(self):
        cfg = get_config()
        cfg["resampler"] = True
        assert get_config()["resampler"] is False


class TestSetConfig:
    def test_set_true(self):
        set_config(resampler=True)
        assert get_config()["resampler"] is True

    def test_set_widget(self):
        set_config(resampler="widget")
        assert get_config()["resampler"] == "widget"

    def test_set_false(self):
        set_config(resampler=True)
        set_config(resampler=False)
        assert get_config()["resampler"] is False

    def test_none_leaves_unchanged(self):
        set_config(resampler=True)
        set_config(resampler=None)
        assert get_config()["resampler"] is True


class TestConfigContext:
    def test_temporary_override(self):
        with config_context(resampler=True):
            assert get_config()["resampler"] is True
        assert get_config()["resampler"] is False

    def test_widget_override(self):
        with config_context(resampler="widget"):
            assert get_config()["resampler"] == "widget"
        assert get_config()["resampler"] is False

    def test_restores_on_exception(self):
        with pytest.raises(RuntimeError), config_context(resampler=True):
            raise RuntimeError("boom")
        assert get_config()["resampler"] is False

    def test_none_leaves_unchanged(self):
        set_config(resampler=True)
        with config_context(resampler=None):
            assert get_config()["resampler"] is True
        assert get_config()["resampler"] is True

    def test_nested_contexts(self):
        with config_context(resampler=True):
            with config_context(resampler="widget"):
                assert get_config()["resampler"] == "widget"
            assert get_config()["resampler"] is True
        assert get_config()["resampler"] is False


# _get_resampler_mode


class TestGetResamplerMode:
    def test_explicit_true(self):
        assert _get_resampler_mode(True) is True

    def test_explicit_false(self):
        assert _get_resampler_mode(False) is False

    def test_explicit_widget(self):
        assert _get_resampler_mode("widget") == "widget"

    def test_none_reads_config(self):
        assert _get_resampler_mode(None) is False
        set_config(resampler=True)
        assert _get_resampler_mode(None) is True


# _create_figure


class TestCreateFigure:
    def test_default_returns_plain(self):
        fig = _create_figure()
        assert type(fig) is go.Figure

    def test_false_returns_plain(self):
        fig = _create_figure(False)
        assert type(fig) is go.Figure

    def test_true_returns_resampler(self):
        fig = _create_figure(True)
        assert isinstance(fig, FigureResampler)

    def test_widget_returns_widget_resampler(self):
        pytest.importorskip("anywidget")
        fig = _create_figure("widget")
        assert isinstance(fig, FigureWidgetResampler)

    def test_none_respects_global_config(self):
        fig = _create_figure(None)
        assert type(fig) is go.Figure

        set_config(resampler=True)
        fig = _create_figure(None)
        assert isinstance(fig, FigureResampler)


# _create_subplots


class TestCreateSubplots:
    def test_default_returns_plain(self):
        fig = _create_subplots(rows=1, cols=2)
        assert type(fig) is go.Figure

    def test_true_returns_resampler(self):
        fig = _create_subplots(True, rows=2, cols=1)
        assert isinstance(fig, FigureResampler)

    def test_widget_returns_widget_resampler(self):
        pytest.importorskip("anywidget")
        fig = _create_subplots("widget", rows=1, cols=1)
        assert isinstance(fig, FigureWidgetResampler)


# Resampler parameter pass-through in public functions


class TestExplorationResampler:
    """Test that exploration functions accept and honour the resampler param."""

    def test_plot_time_series_resampler(self, monthly_1col_df):
        from yohou.plotting import plot_time_series

        fig = plot_time_series(monthly_1col_df, columns="y", resampler=True)
        assert isinstance(fig, FigureResampler)

    def test_plot_time_series_widget(self, monthly_1col_df):
        pytest.importorskip("anywidget")
        from yohou.plotting import plot_time_series

        fig = plot_time_series(monthly_1col_df, columns="y", resampler="widget")
        assert isinstance(fig, FigureWidgetResampler)

    def test_plot_time_series_default(self, monthly_1col_df):
        from yohou.plotting import plot_time_series

        fig = plot_time_series(monthly_1col_df, columns="y")
        assert type(fig) is go.Figure

    def test_plot_rolling_statistics_resampler(self, monthly_1col_df):
        from yohou.plotting import plot_rolling_statistics

        fig = plot_rolling_statistics(monthly_1col_df, columns="y", window_size=3, resampler=True)
        assert isinstance(fig, FigureResampler)

    def test_plot_outliers_resampler(self, monthly_1col_df):
        from yohou.plotting import plot_outliers

        fig = plot_outliers(monthly_1col_df, columns="y", resampler=True)
        assert isinstance(fig, FigureResampler)


class TestForecastingResampler:
    """Test that forecasting functions accept and honour the resampler param."""

    def test_plot_forecast_resampler(self, monthly_1col_df):
        from yohou.plotting import plot_forecast

        y_pred = (
            monthly_1col_df.rename({"y": "y_pred"})
            .with_columns(
                pl.col("y_pred").alias("y_pred"),
            )
            .select("time", pl.col("y_pred").alias("y"))
        )
        fig = plot_forecast(monthly_1col_df, y_pred, resampler=True)
        assert isinstance(fig, FigureResampler)


class TestConfigContextIntegration:
    """Test that config_context affects figure creation in plot functions."""

    def test_config_context_enables_resampler(self, monthly_1col_df):
        from yohou.plotting import config_context, plot_time_series

        with config_context(resampler=True):
            fig = plot_time_series(monthly_1col_df, columns="y")
        assert isinstance(fig, FigureResampler)

    def test_config_context_widget(self, monthly_1col_df):
        pytest.importorskip("anywidget")
        from yohou.plotting import config_context, plot_time_series

        with config_context(resampler="widget"):
            fig = plot_time_series(monthly_1col_df, columns="y")
        assert isinstance(fig, FigureWidgetResampler)


# Extended config keys


class TestSetConfigExtended:
    def test_n_shown_samples(self):
        set_config(resampler_n_shown_samples=500)
        assert get_config()["resampler_n_shown_samples"] == 500

    def test_downsampler(self):
        ds = LTTB()
        set_config(resampler_downsampler=ds)
        assert get_config()["resampler_downsampler"] is ds

    def test_gap_handler(self):
        gh = NoGapHandler()
        set_config(resampler_gap_handler=gh)
        assert get_config()["resampler_gap_handler"] is gh

    def test_trace_prefix_suffix(self):
        set_config(resampler_trace_prefix_suffix=("[RS] ", " ~"))
        assert get_config()["resampler_trace_prefix_suffix"] == ("[RS] ", " ~")

    def test_show_mean_aggregation_size(self):
        set_config(resampler_show_mean_aggregation_size=False)
        assert get_config()["resampler_show_mean_aggregation_size"] is False

    def test_none_leaves_extended_unchanged(self):
        set_config(resampler_n_shown_samples=500)
        set_config(resampler_n_shown_samples=None)
        # None does NOT reset - it leaves the current value
        assert get_config()["resampler_n_shown_samples"] == 500


class TestConfigContextExtended:
    def test_temporary_n_shown_samples(self):
        with config_context(resampler_n_shown_samples=2000):
            assert get_config()["resampler_n_shown_samples"] == 2000
        assert get_config()["resampler_n_shown_samples"] is None

    def test_temporary_downsampler(self):
        ds = MinMaxAggregator()
        with config_context(resampler_downsampler=ds):
            assert get_config()["resampler_downsampler"] is ds
        assert get_config()["resampler_downsampler"] is None

    def test_restores_extended_on_exception(self):
        with pytest.raises(RuntimeError), config_context(resampler_n_shown_samples=500):
            raise RuntimeError("boom")
        assert get_config()["resampler_n_shown_samples"] is None


# _build_resampler_kwargs


class TestBuildResamplerKwargs:
    def test_empty_when_all_none(self):
        assert _build_resampler_kwargs() == {}

    def test_n_shown_samples(self):
        set_config(resampler_n_shown_samples=2000)
        kwargs = _build_resampler_kwargs()
        assert kwargs == {"default_n_shown_samples": 2000}

    def test_downsampler(self):
        ds = LTTB()
        set_config(resampler_downsampler=ds)
        kwargs = _build_resampler_kwargs()
        assert kwargs == {"default_downsampler": ds}

    def test_gap_handler(self):
        gh = NoGapHandler()
        set_config(resampler_gap_handler=gh)
        kwargs = _build_resampler_kwargs()
        assert kwargs == {"default_gap_handler": gh}

    def test_multiple_keys(self):
        ds = MinMaxAggregator()
        set_config(resampler_n_shown_samples=500, resampler_downsampler=ds)
        kwargs = _build_resampler_kwargs()
        assert kwargs["default_n_shown_samples"] == 500
        assert kwargs["default_downsampler"] is ds
        assert len(kwargs) == 2

    def test_prefix_suffix_and_aggregation_size(self):
        set_config(
            resampler_trace_prefix_suffix=("[RS] ", ""),
            resampler_show_mean_aggregation_size=False,
        )
        kwargs = _build_resampler_kwargs()
        assert kwargs["resampled_trace_prefix_suffix"] == ("[RS] ", "")
        assert kwargs["show_mean_aggregation_size"] is False


class TestCreateFigureForwardsConfig:
    def test_n_shown_samples_forwarded(self):
        set_config(resampler_n_shown_samples=777)
        fig = _create_figure(True)
        assert isinstance(fig, FigureResampler)
        # FigureResampler stores the default as _global_n_shown_samples
        assert fig._global_n_shown_samples == 777

    def test_downsampler_forwarded(self):
        ds = LTTB()
        set_config(resampler_downsampler=ds)
        fig = _create_figure(True)
        assert fig._global_downsampler is ds

    def test_gap_handler_forwarded(self):
        gh = NoGapHandler()
        set_config(resampler_gap_handler=gh)
        fig = _create_figure(True)
        assert isinstance(fig._global_gap_handler, NoGapHandler)


# _fill_trace_kwargs


class TestFillTraceKwargs:
    def test_plain_figure_returns_empty(self):
        fig = go.Figure()
        assert _fill_trace_kwargs(fig) == {}

    def test_resampler_figure_returns_no_gap_handler(self):
        fig = _create_figure(True)
        result = _fill_trace_kwargs(fig)
        assert "gap_handler" in result
        assert isinstance(result["gap_handler"], NoGapHandler)


# Re-exports


class TestReExports:
    def test_aggregators_importable(self):
        from yohou.plotting import LTTB, EveryNthPoint, MinMaxAggregator, MinMaxLTTB

        assert MinMaxLTTB is not None
        assert LTTB is not None
        assert MinMaxAggregator is not None
        assert EveryNthPoint is not None

    def test_gap_handlers_importable(self):
        from yohou.plotting import MedDiffGapHandler, NoGapHandler

        assert MedDiffGapHandler is not None
        assert NoGapHandler is not None
