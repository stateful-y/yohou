"""Tests for signal plotting functions."""

import pytest

pytest.importorskip("plotly", reason="plotting extra not installed")


import polars as pl
import pytest
from plotly import graph_objects as go

from yohou.plotting import plot_phase, plot_spectrum

from .conftest import assert_figure_valid, assert_layout


@pytest.fixture
def sine_wave_df():
    """Create sample DataFrame for testing."""
    import numpy as np

    n = 100
    t = np.linspace(0, 2 * np.pi * 4, n)
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
        "y": (np.sin(t) + 0.5 * np.cos(2 * t)).tolist(),
    })


class TestPlotPhase:
    """Tests for plot_phase function."""

    def test_basic(self, sine_wave_df):
        """Test basic phase spectrum plot."""
        fig = plot_phase(sine_wave_df, columns="y")
        assert_figure_valid(fig)

    def test_unwrap(self, sine_wave_df):
        """Test with phase unwrapping enabled."""
        fig = plot_phase(sine_wave_df, columns="y", unwrap=True)
        assert_figure_valid(fig)

    def test_degrees(self, sine_wave_df):
        """Test degrees mode."""
        fig = plot_phase(sine_wave_df, columns="y", angle_unit="degree")
        assert_figure_valid(fig)
        # Y-axis label should mention degrees
        y_text = fig.layout.yaxis.title.text
        assert "deg" in y_text.lower() or "°" in y_text

    def test_radians(self, sine_wave_df):
        """Test radians mode (default)."""
        fig = plot_phase(sine_wave_df, columns="y", angle_unit="radian")
        assert_figure_valid(fig)
        y_text = fig.layout.yaxis.title.text
        assert "radian" in y_text.lower()

    def test_multiple_columns(self, sine_wave_df):
        """Test with multiple columns."""
        import numpy as np

        n = len(sine_wave_df)
        df = sine_wave_df.with_columns(pl.Series("y2", (np.cos(np.linspace(0, 4 * np.pi, n))).tolist()))
        fig = plot_phase(df, columns=["y", "y2"])
        assert len(fig.data) >= 2

    def test_panel(self):
        """Test panel data faceting."""
        import numpy as np

        n = 50
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 2, 19), "1d", eager=True),
            "y__a": np.sin(np.linspace(0, 4 * np.pi, n)).tolist(),
            "y__b": np.cos(np.linspace(0, 4 * np.pi, n)).tolist(),
        })
        fig = plot_phase(df, groups=["y"])
        assert_figure_valid(fig)
        assert len(fig.data) >= 2


class TestPlotSpectrum:
    """Tests for plot_spectrum function."""

    def test_basic(self):
        """Test basic periodogram functionality."""
        import numpy as np

        t = np.arange(100)
        y = np.sin(2 * np.pi * 0.1 * t)
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
            "y": y,
        })

        fig = plot_spectrum(df, columns="y")
        assert len(fig.data) >= 1
        assert fig.data[0].type == "scatter"

    def test_log_scale(self):
        """Test log scale option."""
        import numpy as np

        t = np.arange(100)
        y = np.sin(2 * np.pi * 0.1 * t)
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
            "y": y,
        })

        fig = plot_spectrum(df, log_scale=True)
        assert len(fig.data) >= 1
        assert fig.layout.yaxis.type == "log"

    def test_peaks(self):
        """Test peak detection."""
        import numpy as np

        t = np.arange(100)
        y = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.25 * t)
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
            "y": y,
        })

        fig = plot_spectrum(df, show_peaks=True, n_peaks=2)
        assert len(fig.data) >= 2  # Line trace + peak markers

    def test_multiple_columns(self):
        """Test plotting multiple columns."""
        import numpy as np

        t = np.arange(100)
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
            "y1": np.sin(2 * np.pi * 0.1 * t),
            "y2": np.sin(2 * np.pi * 0.2 * t),
        })

        fig = plot_spectrum(df, columns=["y1", "y2"])
        assert len(fig.data) >= 2

    def test_panel(self, short_df):
        """Test panel faceting for periodogram."""
        df = pl.DataFrame({
            "time": short_df["time"],
            "y__a": short_df["y"],
            "y__b": short_df["x"],
        })
        fig = plot_spectrum(df, groups=["y"])
        assert len(fig.data) > 0


class TestPlotPhaseAssertions:
    """Error path and stronger assertion tests for plot_phase."""

    def test_returns_go_figure(self):
        """Phase plot always returns a go.Figure instance."""
        import numpy as np

        n = 50
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 2, 19), "1d", eager=True),
            "y": np.sin(np.linspace(0, 4 * np.pi, n)).tolist(),
        })
        fig = plot_phase(df, columns="y")
        assert_figure_valid(fig)

    def test_trace_type_is_scatter(self):
        """Phase plot traces are Scatter type."""
        import numpy as np

        n = 50
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 2, 19), "1d", eager=True),
            "y": np.sin(np.linspace(0, 4 * np.pi, n)).tolist(),
        })
        fig = plot_phase(df, columns="y")
        assert all(isinstance(t, go.Scatter) for t in fig.data)

    def test_custom_title(self):
        """Custom title is applied to phase figure."""
        import numpy as np

        n = 50
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 2, 19), "1d", eager=True),
            "y": np.sin(np.linspace(0, 4 * np.pi, n)).tolist(),
        })
        fig = plot_phase(df, columns="y", title="Phase Title")
        assert_layout(fig, title="Phase Title")

    def test_custom_dimensions(self):
        """Custom dimensions are respected."""
        import numpy as np

        n = 50
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 2, 19), "1d", eager=True),
            "y": np.sin(np.linspace(0, 4 * np.pi, n)).tolist(),
        })
        fig = plot_phase(df, columns="y", width=900, height=600)
        assert_layout(fig, width=900, height=600)


class TestPlotSpectrumAssertions:
    """Error path and stronger assertion tests for plot_spectrum."""

    def test_returns_go_figure(self):
        """Spectrum always returns a go.Figure instance."""
        import numpy as np

        t = np.arange(100)
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
            "y": np.sin(2 * np.pi * 0.1 * t),
        })
        fig = plot_spectrum(df, columns="y")
        assert_figure_valid(fig)

    def test_custom_title(self):
        """Custom title is applied to spectrum figure."""
        import numpy as np

        t = np.arange(100)
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
            "y": np.sin(2 * np.pi * 0.1 * t),
        })
        fig = plot_spectrum(df, columns="y", title="Spectrum Title")
        assert_layout(fig, title="Spectrum Title")

    def test_custom_dimensions(self):
        """Custom dimensions are respected."""
        import numpy as np

        t = np.arange(100)
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
            "y": np.sin(2 * np.pi * 0.1 * t),
        })
        fig = plot_spectrum(df, columns="y", width=1000, height=500)
        assert_layout(fig, width=1000, height=500)

    def test_invalid_dimensions_raise(self):
        """Invalid width/height raise ValueError."""
        import numpy as np

        t = np.arange(100)
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
            "y": np.sin(2 * np.pi * 0.1 * t),
        })
        with pytest.raises(ValueError, match="width"):
            plot_spectrum(df, columns="y", width=0)


class TestConnectGaps:
    """Tests for connect_gaps parameter across signal functions."""

    def test_plot_phase_connect_gaps_false(self):
        """connect_gaps=False (default) leaves Scatter traces without connectgaps."""
        import numpy as np

        n = 50
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 2, 19), "1d", eager=True),
            "y": np.sin(np.linspace(0, 4 * np.pi, n)).tolist(),
        })
        fig = plot_phase(df, columns="y", connect_gaps=False)
        assert_figure_valid(fig)
        assert not any(t.connectgaps for t in fig.data if hasattr(t, "connectgaps"))

    def test_plot_phase_connect_gaps_true(self):
        """connect_gaps=True sets connectgaps on all Scatter traces."""
        import numpy as np

        n = 50
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 2, 19), "1d", eager=True),
            "y": np.sin(np.linspace(0, 4 * np.pi, n)).tolist(),
        })
        fig = plot_phase(df, columns="y", connect_gaps=True)
        assert_figure_valid(fig)
        assert all(t.connectgaps for t in fig.data if hasattr(t, "connectgaps"))

    def test_plot_spectrum_connect_gaps_true(self):
        """connect_gaps=True does not raise for plot_spectrum."""
        import numpy as np

        t = np.arange(100)
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
            "y": np.sin(2 * np.pi * 0.1 * t),
        })
        fig = plot_spectrum(df, columns="y", connect_gaps=True)
        assert_figure_valid(fig)


class TestPlotPhaseAutoDetectPanel:
    """Test auto-detect panel path in plot_phase (line 113)."""

    def test_auto_detect_panel(self):
        """Phase auto-detects panel data when no columns/groups given."""
        import numpy as np

        n = 50
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 2, 19), "1d", eager=True),
            "y__a": np.sin(np.linspace(0, 4 * np.pi, n)).tolist(),
            "y__b": np.cos(np.linspace(0, 4 * np.pi, n)).tolist(),
        })
        fig = plot_phase(df)
        assert_figure_valid(fig)
        assert len(fig.data) >= 2


class TestPlotPhaseUnwrapDegreePanel:
    """Test panel phase with unwrap=True and angle_unit='degree' (lines 127-130)."""

    def test_panel_unwrap_and_degree(self):
        """Panel phase with both unwrap and degree covers lines 127-130."""
        import numpy as np

        n = 50
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 2, 19), "1d", eager=True),
            "y__a": np.sin(np.linspace(0, 4 * np.pi, n)).tolist(),
            "y__b": np.cos(np.linspace(0, 4 * np.pi, n)).tolist(),
        })
        fig = plot_phase(df, groups=["y"], unwrap=True, angle_unit="degree")
        assert_figure_valid(fig)
        assert len(fig.data) >= 2


class TestPlotPhaseUnwrapDegreeNonPanel:
    """Test non-panel phase with unwrap=True and degree (lines 175-177)."""

    def test_non_panel_unwrap_and_degree(self):
        """Non-panel phase with unwrap=True and degree covers lines 175-177."""
        import numpy as np

        n = 50
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 2, 19), "1d", eager=True),
            "y": np.sin(np.linspace(0, 4 * np.pi, n)).tolist(),
        })
        fig = plot_phase(df, columns="y", unwrap=True, angle_unit="degree")
        assert_figure_valid(fig)
        y_text = fig.layout.yaxis.title.text
        assert "deg" in y_text.lower() or "°" in y_text


class TestPlotSpectrumAutoDetectPanel:
    """Test auto-detect panel path in plot_spectrum (line 304)."""

    def test_auto_detect_panel(self):
        """Spectrum auto-detects panel data when no columns/groups given."""
        import numpy as np

        n = 50
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 2, 19), "1d", eager=True),
            "y__a": np.sin(np.linspace(0, 4 * np.pi, n)).tolist(),
            "y__b": np.cos(np.linspace(0, 4 * np.pi, n)).tolist(),
        })
        fig = plot_spectrum(df)
        assert_figure_valid(fig)
        assert len(fig.data) >= 2
