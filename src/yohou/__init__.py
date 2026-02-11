"""Scikit-learn-compatible time series forecasting framework built on polars."""

from importlib.metadata import version

from sklearn import set_config
from sklearn.utils._metadata_requests import COMPOSITE_METHODS, METHODS, SIMPLE_METHODS

__version__ = version(__name__)

# Enable metadata routing globally for all Yohou estimators
set_config(enable_metadata_routing=True)

# Extend sklearn's metadata routing to support Yohou-specific methods
# This allows MethodMapping to route metadata for custom methods:
# - observe_transform: Composite method like fit_transform (observe + transform)
# - rewind_transform: Composite method for rewinding state and transforming (rewind + transform)
# - observe_predict: Composite method for forecasters (observe + predict)
# - predict_interval: Interval forecasting method
# - observe_predict_interval: Composite method for interval forecasting (observe + predict_interval)
# Note: 'observe' itself is NOT routed - it's a memory management operation

for method in [
    "observe_transform",
    "rewind_transform",
    "observe_predict",
    "predict_interval",
    "observe_predict_interval",
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

from yohou import (  # noqa: E402
    base,
    compose,
    datasets,
    interval,
    metrics,
    model_selection,
    plotting,
    point,
    preprocessing,
    stationarity,
    testing,
    utils,
)

__all__ = [
    "__version__",
    "base",
    "compose",
    "datasets",
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
