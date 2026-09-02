"""Static audit of PEP 723 dependency blocks in marimo notebook examples.

Checks that every notebook's PEP 723 ``dependencies`` block covers all
packages imported by that notebook, accounting for yohou's base and
optional extras.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
import tomllib

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).parent.parent / "examples"

# Mapping from PyPI install name (lowercase, normalised) to importable top-level
# module name. Add an entry here when a new package is introduced to the examples.
INSTALL_TO_IMPORT: dict[str, str] = {
    "scikit-learn": "sklearn",
    "sklearn-wrap": "sklearn_wrap",
    "scipy": "scipy",
    "polars": "polars",
    "pydantic": "pydantic",
    "plotly": "plotly",
    "plotly-resampler": "plotly_resampler",
    "anywidget": "anywidget",
    "pytz": "pytz",
    "statsmodels": "statsmodels",
    "catboost": "catboost",
    "lightgbm": "lightgbm",
    "numpy": "numpy",
    "marimo": "marimo",
}

# Imports covered by the base yohou package (no extras required).
_YOHOU_BASE_IMPORTS: frozenset[str] = frozenset({
    "sklearn",
    "sklearn_wrap",
    "scipy",
    "polars",
    "pydantic",
    "joblib",
})

# Imports added by each yohou optional extra.
_YOHOU_EXTRAS_IMPORTS: dict[str, frozenset[str]] = {
    "plotting": frozenset({
        "anywidget",
        "plotly",
        "plotly_resampler",
        "pytz",
        "statsmodels",
    }),
}

# All import names recognised by the mapping. Any notebook import that is not in
# this set is unknown and must be added to INSTALL_TO_IMPORT.
_ALL_KNOWN_IMPORTS: frozenset[str] = (
    frozenset(INSTALL_TO_IMPORT.values()) | _YOHOU_BASE_IMPORTS | frozenset().union(*_YOHOU_EXTRAS_IMPORTS.values())
)


def _parse_pep723_deps(path: pathlib.Path) -> list[str]:
    """Extract the ``dependencies`` list from a PEP 723 inline script block."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^# /// script\s*\n((?:#[^\n]*\n)+?)# ///", text, re.MULTILINE)
    if not match:
        return []
    raw = match.group(1)
    toml_text = re.sub(r"^# ?", "", raw, flags=re.MULTILINE)
    data = tomllib.loads(toml_text)
    return data.get("dependencies", [])


# Synthetic marker used to require the plotting extra when a notebook imports
# from yohou.plotting (which is only available with yohou[plotting]).
_YOHOU_PLOTTING_MARKER = "_yohou_plotting"


def _get_notebook_imports(path: pathlib.Path) -> set[str]:
    """Return all top-level package names imported by the notebook.

    Excludes stdlib modules, ``marimo``, and ``yohou`` internals, but adds a
    synthetic ``_yohou_plotting`` marker when the notebook imports from
    ``yohou.plotting`` (which requires the ``plotting`` extra).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    uses_yohou_plotting = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                if mod == "yohou.plotting" or mod.startswith("yohou.plotting."):
                    uses_yohou_plotting = True
                imports.add(mod.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "yohou.plotting" or node.module.startswith("yohou.plotting."):
                uses_yohou_plotting = True
            imports.add(node.module.split(".")[0])
    stdlib = sys.stdlib_module_names
    external = {imp for imp in imports if imp not in stdlib and imp != "marimo" and not imp.startswith("yohou")}
    if uses_yohou_plotting:
        external.add(_YOHOU_PLOTTING_MARKER)
    return external


def _resolve_covered_imports(deps: list[str]) -> set[str]:
    """Expand PEP 723 dep strings into the set of importable module names they cover."""
    covered: set[str] = set()
    for dep in deps:
        m = re.match(r"^([A-Za-z0-9_.-]+)(?:\[([^\]]+)\])?", dep)
        if not m:
            continue
        name = m.group(1).lower()
        extras_str = m.group(2)
        extras = [e.strip() for e in extras_str.split(",")] if extras_str else []

        if name == "yohou":
            covered |= _YOHOU_BASE_IMPORTS
            for extra in extras:
                covered |= _YOHOU_EXTRAS_IMPORTS.get(extra, frozenset())
            if "plotting" in extras:
                covered.add(_YOHOU_PLOTTING_MARKER)
        elif name in INSTALL_TO_IMPORT:
            covered.add(INSTALL_TO_IMPORT[name])

    return covered


def _collect_notebooks() -> list[pathlib.Path]:
    return sorted(p for p in EXAMPLES_DIR.rglob("*.py") if p.name != "__init__.py" and not p.name.startswith("_"))


@pytest.mark.parametrize(
    "notebook",
    _collect_notebooks(),
    ids=lambda p: str(p.relative_to(EXAMPLES_DIR.parent)),
)
def test_pep723_deps_are_complete(notebook: pathlib.Path) -> None:
    deps = _parse_pep723_deps(notebook)
    if not deps:
        pytest.skip("No PEP 723 dependencies block found")

    actual_imports = _get_notebook_imports(notebook)
    covered = _resolve_covered_imports(deps)

    unknown = actual_imports - _ALL_KNOWN_IMPORTS - {_YOHOU_PLOTTING_MARKER}
    if unknown:
        pytest.fail(
            f"{notebook.relative_to(EXAMPLES_DIR.parent)}: imports {sorted(unknown)} "
            f"are not in INSTALL_TO_IMPORT in tests/test_notebook_pep723.py. "
            f"Add them to the mapping."
        )

    missing = actual_imports - covered
    if missing:
        human_missing = [
            "yohou.plotting (requires yohou[plotting])" if m == _YOHOU_PLOTTING_MARKER else m for m in sorted(missing)
        ]
        pytest.fail(
            f"{notebook.relative_to(EXAMPLES_DIR.parent)}: {human_missing} "
            f"are not covered by the declared PEP 723 dependencies {deps!r}. "
            f"Add the corresponding package(s) to the notebook's PEP 723 block."
        )
