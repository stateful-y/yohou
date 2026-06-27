"""Tests for the global ``line_shape`` plotting config.

Covers the config plumbing (``set_config`` / ``get_config`` / ``config_context``),
the ``_apply_line_shape`` finaliser, and end-to-end application across the
faceted and direct ``apply_default_layout`` finalisation paths.
"""

import pytest

pytest.importorskip("plotly", reason="plotting extra not installed")

import polars as pl  # noqa: E402
from plotly import graph_objects as go  # noqa: E402

from yohou.plotting import (  # noqa: E402
    config_context,
    get_config,
    plot_autocorrelation,
    plot_calibration,
    plot_phase,
    plot_residuals,
    plot_spectrum,
    plot_time_series,
    set_config,
)
from yohou.plotting._utils import (  # noqa: E402
    VALID_LINE_SHAPES,
    _apply_line_shape,
    _global_config,
    _resolve_line_shape,
)


@pytest.fixture(autouse=True)
def _reset_config():
    """Ensure every test starts and ends with the default config."""
    old = _global_config.copy()
    yield
    _global_config.clear()
    _global_config.update(old)


def _line_trace_shapes(fig: go.Figure) -> set:
    """Return the distinct ``line.shape`` of every scatter trace that draws lines."""
    shapes = set()
    for trace in fig.data:
        if trace.type in ("scatter", "scattergl"):
            mode = trace.mode
            if mode is None or "lines" in mode:
                shapes.add(trace.line.shape)
    return shapes


class TestConfigPlumbing:
    def test_default_is_none(self):
        assert get_config()["line_shape"] is None

    def test_set_config_sets_value(self):
        set_config(line_shape="hv")
        assert get_config()["line_shape"] == "hv"

    def test_set_config_none_leaves_unchanged(self):
        set_config(line_shape="hv")
        set_config(line_shape=None)
        assert get_config()["line_shape"] == "hv"

    @pytest.mark.parametrize("shape", sorted(VALID_LINE_SHAPES))
    def test_set_config_accepts_all_valid_shapes(self, shape):
        set_config(line_shape=shape)
        assert get_config()["line_shape"] == shape

    def test_set_config_rejects_invalid_shape(self):
        with pytest.raises(ValueError, match="line_shape must be one of"):
            set_config(line_shape="zigzag")

    def test_config_context_overrides_then_restores(self):
        with config_context(line_shape="hv"):
            assert get_config()["line_shape"] == "hv"
        assert get_config()["line_shape"] is None

    def test_config_context_restores_on_exception(self):
        with pytest.raises(RuntimeError), config_context(line_shape="hv"):
            raise RuntimeError("boom")
        assert get_config()["line_shape"] is None

    def test_config_context_validates(self):
        with pytest.raises(ValueError, match="line_shape must be one of"), config_context(line_shape="bogus"):
            pass


class TestApplyLineShape:
    def test_resolve_explicit_wins_over_config(self):
        set_config(line_shape="hv")
        assert _resolve_line_shape("vh") == "vh"

    def test_resolve_falls_back_to_config(self):
        set_config(line_shape="hv")
        assert _resolve_line_shape() == "hv"

    def test_resolve_none_when_unset(self):
        assert _resolve_line_shape() is None

    def test_overrides_scatter_traces(self):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2], mode="lines", line={"shape": "linear"}))
        _apply_line_shape(fig, "hv")
        assert fig.data[0].line.shape == "hv"

    def test_overrides_explicit_per_trace_shape(self):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2], line={"shape": "hv"}))
        _apply_line_shape(fig, "spline")
        assert fig.data[0].line.shape == "spline"

    def test_skips_non_scatter_traces(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], mode="lines"))
        fig.add_trace(go.Bar(x=[1, 2], y=[1, 2]))
        # Must not raise on the Bar trace.
        _apply_line_shape(fig, "hv")
        assert fig.data[0].line.shape == "hv"
        assert fig.data[1].type == "bar"

    def test_none_is_noop_preserving_explicit_shape(self):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2], line={"shape": "hv"}))
        _apply_line_shape(fig, None)
        assert fig.data[0].line.shape == "hv"

    def test_reads_global_config_when_no_arg(self):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2], mode="lines"))
        with config_context(line_shape="vh"):
            _apply_line_shape(fig)
        assert fig.data[0].line.shape == "vh"


@pytest.fixture
def _y_truth_pred():
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
    y_truth = pl.DataFrame({"time": dates, "y": [100 + i for i in range(91)]})
    y_pred = pl.DataFrame({"time": dates, "y": [100 + i + (i % 3) for i in range(91)]})
    return y_pred, y_truth


class TestEndToEndOverride:
    """Setting the config forces the shape on every scatter line trace."""

    def test_plot_time_series_facet_path(self, simple_df):
        with config_context(line_shape="vh"):
            fig = plot_time_series(simple_df)
        assert _line_trace_shapes(fig) == {"vh"}

    def test_plot_autocorrelation_facet_path(self, simple_df):
        with config_context(line_shape="hv"):
            fig = plot_autocorrelation(simple_df)
        assert _line_trace_shapes(fig) == {"hv"}

    def test_plot_spectrum_signal_module(self, short_df):
        with config_context(line_shape="hv"):
            fig = plot_spectrum(short_df, columns=["x"])
        assert _line_trace_shapes(fig) == {"hv"}

    def test_plot_phase_signal_module(self, short_df):
        with config_context(line_shape="hvh"):
            fig = plot_phase(short_df, columns=["x"])
        assert _line_trace_shapes(fig) == {"hvh"}

    def test_plot_residuals_direct_layout_path(self, _y_truth_pred):
        y_pred, y_truth = _y_truth_pred
        with config_context(line_shape="vh"):
            fig = plot_residuals(y_pred, y_truth)
        assert _line_trace_shapes(fig) == {"vh"}


class TestDefaultPreserved:
    """Without the config set, each function keeps its built-in line shape."""

    def test_numeric_time_series_stays_linear(self, simple_df):
        fig = plot_time_series(simple_df)
        # Plotly represents the linear default as an unset (None) shape.
        assert _line_trace_shapes(fig) == {None}

    def test_categorical_step_chart_keeps_hv(self):
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 5), "1d", eager=True),
            "state": ["on", "off", "on", "off", "on"],
        })
        fig = plot_time_series(df, columns=["state"])
        assert fig.data[0].line.shape == "hv"


class TestCalibrationExempt:
    """Calibration reliability diagrams ignore the global line_shape override."""

    @staticmethod
    def _interval_data():
        y_vals = [float(i) for i in range(30)]
        y_truth = pl.DataFrame({"value": y_vals})
        y_pred_int = pl.DataFrame({
            "value_lower_0.9": [v - 2 for v in y_vals],
            "value_upper_0.9": [v + 2 for v in y_vals],
            "value_lower_0.95": [v - 3 for v in y_vals],
            "value_upper_0.95": [v + 3 for v in y_vals],
        })
        return y_pred_int, y_truth

    def test_calibration_stays_linear_under_hv(self):
        y_pred_int, y_truth = self._interval_data()
        with config_context(line_shape="hv"):
            fig = plot_calibration(y_pred_int, y_truth, coverage_rates=[0.9, 0.95])
        # The reliability curve and the diagonal must never become step lines.
        assert "hv" not in _line_trace_shapes(fig)
        assert _line_trace_shapes(fig) == {None}

    def test_calibration_unchanged_with_and_without_config(self):
        y_pred_int, y_truth = self._interval_data()
        plain = plot_calibration(y_pred_int, y_truth, coverage_rates=[0.9, 0.95])
        with config_context(line_shape="hv"):
            overridden = plot_calibration(y_pred_int, y_truth, coverage_rates=[0.9, 0.95])
        assert _line_trace_shapes(plain) == _line_trace_shapes(overridden)
