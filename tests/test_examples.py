"""Tests for marimo notebook examples.

This module validates that all notebook examples in the examples/ directory
can be executed successfully as Python scripts.

Marimo notebooks can be run directly with Python:
https://docs.marimo.io/getting_started/quickstart/#run-as-scripts
"""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

# Discover all example notebooks
EXAMPLES_DIR = pathlib.Path(__file__).parent.parent / "examples"

# Optional-dependency availability flags
_has_statsmodels = importlib.util.find_spec("statsmodels") is not None
_has_catboost = importlib.util.find_spec("catboost") is not None

# Notebooks that require optional dependencies or have known issues, keyed by filename
_NOTEBOOK_MARKS: dict[str, list[pytest.MarkDecorator]] = {
    "catboost_forecasting.py": [pytest.mark.skipif(not _has_catboost, reason="catboost not installed")],
    "catboost_multiquantile.py": [pytest.mark.skipif(not _has_catboost, reason="catboost not installed")],
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
        [sys.executable, str(notebook_file)],
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

    @pytest.mark.parametrize("notebook_file", _collect_notebooks("getting-started"))
    def test_getting_started_example(self, notebook_file: pathlib.Path) -> None:
        """Test that a getting-started/ notebook example runs successfully."""
        _run_notebook(notebook_file)

    @pytest.mark.parametrize("notebook_file", _collect_notebooks("data-features"))
    def test_data_features_example(self, notebook_file: pathlib.Path) -> None:
        """Test that a data-features/ notebook example runs successfully."""
        _run_notebook(notebook_file)

    @pytest.mark.parametrize("notebook_file", _collect_notebooks("forecasting-models"))
    def test_forecasting_models_example(self, notebook_file: pathlib.Path) -> None:
        """Test that a forecasting-models/ notebook example runs successfully."""
        _run_notebook(notebook_file)

    @pytest.mark.parametrize("notebook_file", _collect_notebooks("evaluation-search"))
    def test_evaluation_search_example(self, notebook_file: pathlib.Path) -> None:
        """Test that an evaluation-search/ notebook example runs successfully."""
        _run_notebook(notebook_file)

    @pytest.mark.parametrize("notebook_file", _collect_notebooks("panel-data"))
    def test_panel_data_example(self, notebook_file: pathlib.Path) -> None:
        """Test that a panel-data/ notebook example runs successfully."""
        _run_notebook(notebook_file)

    @pytest.mark.parametrize("notebook_file", _collect_notebooks("visualization"))
    def test_visualization_example(self, notebook_file: pathlib.Path) -> None:
        """Test that a visualization/ notebook example runs successfully."""
        _run_notebook(notebook_file)
