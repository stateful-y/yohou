"""Scikit-learn-compatible time series forecasting framework built on polars."""

import re
from importlib.metadata import version

from sklearn import set_config

from yohou.utils._compat import COMPOSITE_METHODS, METHODS, SIMPLE_METHODS

__version__ = version(__name__)

# Enable metadata routing globally for all Yohou estimators
set_config(enable_metadata_routing=True)

# Patch numpydoc's See Also parser to tolerate MkDocs link syntax.
# Yohou docstrings use `[`Name`][qualified.path]` in See Also sections
# so that mkdocs-autorefs can generate hyperlinks in the rendered docs.
# The numpydoc parser bundled with sklearn only accepts bare names or
# Sphinx :role:`name` syntax, so MkDocs links, bullet markers, and
# backtick-only references cause a ValueError when sklearn tries to
# render the HTML repr.  This patch normalizes content lines *before*
# parsing, leaving the original __doc__ strings (and therefore mkdocs
# output) untouched.
try:
    from sklearn.externals._numpydoc.docscrape import NumpyDocString

    _original_parse_see_also = NumpyDocString._parse_see_also
    _MKDOCS_LINK_RE = re.compile(r"\[`?[^\]]*`?\]\[([A-Za-z_][A-Za-z0-9_.]*)\]")
    _BULLET_PREFIX_RE = re.compile(r"^(\s*)-\s+")
    _BACKTICK_NAME_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)`")

    def _parse_see_also_strip_backticks(self, content):  # noqa: ANN001, ANN202
        """Normalize MkDocs links and backticks before delegating to the original parser."""
        cleaned = [_MKDOCS_LINK_RE.sub(r"\1", line) for line in content]
        cleaned = [_BULLET_PREFIX_RE.sub(r"\1", line) for line in cleaned]
        cleaned = [_BACKTICK_NAME_RE.sub(r"\1", line) for line in cleaned]
        return _original_parse_see_also(self, cleaned)

    NumpyDocString._parse_see_also = _parse_see_also_strip_backticks  # ty: ignore[invalid-assignment]
except (ImportError, AttributeError):
    pass

# Extend sklearn's metadata routing to support Yohou-specific methods
# This allows MethodMapping to route metadata for custom methods:
# - observe_transform: Composite method like fit_transform (observe + transform)
# - rewind_transform: Composite method for rewinding state and transforming (rewind + transform)
# - observe_predict: Composite method for forecasters (observe + predict)
# - predict_interval: Interval forecasting method
# - observe_predict_interval: Composite method for interval forecasting (observe + predict_interval)
# - predict_class_proba: Class-probability forecasting method
# - observe_predict_class_proba: Composite method for class-probability forecasting (observe + predict_class_proba)
# Note: 'observe' itself is NOT routed - it's a memory management operation

for method in [
    "observe_transform",
    "rewind_transform",
    "observe_predict",
    "predict_interval",
    "observe_predict_interval",
    "predict_class_proba",
    "observe_predict_class_proba",
]:
    if method not in SIMPLE_METHODS:
        SIMPLE_METHODS.append(method)
    if method not in METHODS:
        METHODS.append(method)

# Mark as composite methods (params route to constituent methods)
COMPOSITE_METHODS["observe_transform"] = ["observe", "transform"]
COMPOSITE_METHODS["rewind_transform"] = ["rewind", "transform"]
COMPOSITE_METHODS["observe_predict"] = ["observe", "predict"]
COMPOSITE_METHODS["observe_predict_interval"] = ["observe", "predict_interval"]
COMPOSITE_METHODS["observe_predict_class_proba"] = ["observe", "predict_class_proba"]

from yohou import (  # noqa: E402
    base,
    class_proba,
    compose,
    datasets,
    ensemble,
    interval,
    metrics,
    model_selection,
    point,
    preprocessing,
    stationarity,
    testing,
    utils,
)

# Lazy-load plotting subpackage so that `import yohou` works without plotly.
_LAZY_SUBMODULES = {"plotting"}


def __getattr__(name: str):  # noqa: ANN202
    """Lazy import for optional submodules that require extra dependencies."""
    if name in _LAZY_SUBMODULES:
        import importlib  # noqa: PLC0415

        return importlib.import_module(f"yohou.{name}")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "__version__",
    "base",
    "class_proba",
    "compose",
    "datasets",
    "ensemble",
    "interval",
    "metrics",
    "model_selection",
    "plotting",
    "point",
    "preprocessing",
    "stationarity",
    "testing",
    "utils",
]
