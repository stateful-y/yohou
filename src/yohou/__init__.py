"""Scikit-learn-compatible time series forecasting framework built on polars."""

from importlib.metadata import version

from sklearn import set_config

__version__ = version(__name__)

# Enable metadata routing globally for all Yohou estimators
set_config(enable_metadata_routing=True)

# Extend sklearn's metadata routing to support Yohou-specific methods
# This allows MethodMapping to route metadata for custom methods:
# - update_transform: Composite method like fit_transform (update + transform)
# - update_predict: Composite method for forecasters (update + predict)
# - predict_interval: Interval forecasting method
# - update_predict_interval: Composite method for interval forecasting (update + predict_interval)
# Note: 'update' itself is NOT routed - it's a memory management operation
from sklearn.utils._metadata_requests import COMPOSITE_METHODS, METHODS, SIMPLE_METHODS

if "update_transform" not in SIMPLE_METHODS:
    SIMPLE_METHODS.extend([
        "update_transform",
        "update_predict",
        "predict_interval",
        "update_predict_interval",
    ])
    # Also extend METHODS (used by MethodMapping validation)
    METHODS.extend([
        "update_transform",
        "update_predict",
        "predict_interval",
        "update_predict_interval",
    ])
    # Mark as composite methods (params route to constituent methods)
    COMPOSITE_METHODS["update_transform"] = ["update", "transform"]
    COMPOSITE_METHODS["update_predict"] = ["update", "predict"]
    COMPOSITE_METHODS["update_predict_interval"] = ["update", "predict_interval"]

from yohou import (
    base,
    datasets,
    forecaster,
    interval_forecaster,
    metrics,
    model_selection,
    pipeline,
    plotting,
    point_forecaster,
    preprocessing,
    utils,
)

__all__ = [
    "__version__",
    "analysis",
    "base",
    "datasets",
    "forecaster",
    "interval_forecaster",
    "metrics",
    "model_selection",
    "point_forecaster",
    "pipeline",
    "plotting",
    "preprocessing",
    "utils",
]
