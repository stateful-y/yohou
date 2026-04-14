"""Tests for lazy import behaviour.

Verifies that ``import yohou`` succeeds even when optional plotting
dependencies (plotly, plotly-resampler) are not installed.
"""

import importlib.util
import subprocess
import sys

import pytest


@pytest.mark.integration
def test_import_yohou_without_plotly():
    """``import yohou`` must not fail when plotly is absent."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys;"
                "sys.modules['plotly'] = None;"
                "sys.modules['plotly.graph_objects'] = None;"
                "sys.modules['plotly.subplots'] = None;"
                "import yohou;"
                "print(yohou.__version__)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"import yohou failed without plotly:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


@pytest.mark.integration
def test_import_yohou_plotting_without_plotly_raises():
    """Accessing ``yohou.plotting`` without plotly must raise ImportError."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys;"
                "sys.modules['plotly'] = None;"
                "sys.modules['plotly.graph_objects'] = None;"
                "sys.modules['plotly.subplots'] = None;"
                "import yohou;"
                "yohou.plotting"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "yohou[plotting]" in result.stderr


class TestLazySubmoduleImport:
    """Tests for __getattr__ lazy import in yohou.__init__."""

    @pytest.mark.skipif(
        importlib.util.find_spec("plotly") is None,
        reason="plotly not installed",
    )
    def test_lazy_plotting_import(self):
        """Accessing yohou.plotting via __getattr__ returns the plotting module."""
        import yohou

        mod = yohou.__getattr__("plotting")
        assert hasattr(mod, "plot_forecast")

    def test_nonexistent_attr_raises(self):
        """Accessing a nonexistent attribute via __getattr__ raises AttributeError."""
        import yohou

        with pytest.raises(AttributeError, match="has no attribute"):
            yohou.__getattr__("nonexistent_submodule_xyz")
