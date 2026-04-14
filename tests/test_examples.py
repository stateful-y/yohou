"""Tests for marimo notebook examples.

This module validates that all notebook examples in the examples/ directory
can be executed successfully as Python scripts.

Marimo notebooks can be run directly with Python:
https://docs.marimo.io/getting_started/quickstart/#run-as-scripts
"""

import importlib.util
import pathlib
import subprocess

import pytest

# Discover all example notebooks
EXAMPLES_DIR = pathlib.Path(__file__).parent.parent / "examples"

# Optional-dependency availability flags
_has_yohou_optuna = importlib.util.find_spec("yohou_optuna") is not None
_has_statsmodels = importlib.util.find_spec("statsmodels") is not None
_has_catboost = importlib.util.find_spec("catboost") is not None

# Notebooks that require optional dependencies or have known issues, keyed by filename
_NOTEBOOK_MARKS: dict[str, list[pytest.MarkDecorator]] = {
    "optuna_search.py": [pytest.mark.xfail(not _has_yohou_optuna, reason="requires yohou-optuna", strict=True)],
}


def _collect_notebooks(subdir: str | None = None) -> list:
    """Collect notebook .py files, excluding __init__.py."""
    search_dir = EXAMPLES_DIR / subdir if subdir else EXAMPLES_DIR
    paths = sorted(p for p in search_dir.glob("*.py") if p.name != "__init__.py" and not p.name.startswith("_"))
    params: list = []
    for p in paths:
        marks = _NOTEBOOK_MARKS.get(p.name)
        if marks:
            params.append(pytest.param(p, marks=marks))
        else:
            params.append(p)
    return params


def _run_notebook(notebook_file: pathlib.Path) -> None:
    """Run a marimo notebook as a script and assert success."""
    result = subprocess.run(
        ["python", str(notebook_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Notebook {notebook_file.name} failed with:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


@pytest.mark.example
class TestTopLevelExamples:
    """Tests for top-level example notebooks."""

    @pytest.mark.parametrize("notebook_file", _collect_notebooks())
    def test_notebook_runs_without_error(self, notebook_file: pathlib.Path) -> None:
        """Test that a top-level notebook example runs successfully as a script."""
        _run_notebook(notebook_file)


@pytest.mark.example
class TestSubdirExamples:
    """Tests for example notebooks in subdirectories."""

    @pytest.mark.parametrize("notebook_file", _collect_notebooks("point"))
    def test_point_example(self, notebook_file: pathlib.Path) -> None:
        """Test that a point/ notebook example runs successfully."""
        _run_notebook(notebook_file)

    @pytest.mark.parametrize("notebook_file", _collect_notebooks("interval"))
    def test_interval_example(self, notebook_file: pathlib.Path) -> None:
        """Test that an interval/ notebook example runs successfully."""
        _run_notebook(notebook_file)

    @pytest.mark.parametrize("notebook_file", _collect_notebooks("metrics"))
    def test_metrics_example(self, notebook_file: pathlib.Path) -> None:
        """Test that a metrics/ notebook example runs successfully."""
        _run_notebook(notebook_file)

    @pytest.mark.parametrize("notebook_file", _collect_notebooks("model_selection"))
    def test_model_selection_example(self, notebook_file: pathlib.Path) -> None:
        """Test that a model_selection/ notebook example runs successfully."""
        _run_notebook(notebook_file)

    @pytest.mark.parametrize("notebook_file", _collect_notebooks("preprocessing"))
    def test_preprocessing_example(self, notebook_file: pathlib.Path) -> None:
        """Test that a preprocessing/ notebook example runs successfully."""
        _run_notebook(notebook_file)

    @pytest.mark.parametrize("notebook_file", _collect_notebooks("stationarity"))
    def test_stationarity_example(self, notebook_file: pathlib.Path) -> None:
        """Test that a stationarity/ notebook example runs successfully."""
        _run_notebook(notebook_file)

    @pytest.mark.parametrize("notebook_file", _collect_notebooks("compose"))
    def test_compose_example(self, notebook_file: pathlib.Path) -> None:
        """Test that a compose/ notebook example runs successfully."""
        _run_notebook(notebook_file)

    @pytest.mark.parametrize("notebook_file", _collect_notebooks("plotting"))
    def test_plotting_example(self, notebook_file: pathlib.Path) -> None:
        """Test that a plotting/ notebook example runs successfully."""
        pytest.importorskip("plotly_resampler", reason="plotting extra not installed")
        _run_notebook(notebook_file)
