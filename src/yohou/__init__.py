"""Scikit-learn-compatible time series forecasting framework built on polars."""

from sklearn import set_config

# Enable metadata routing globally for all Yohou estimators
set_config(enable_metadata_routing=True)

# Extend sklearn's metadata routing to support Yohou-specific methods
# This allows MethodMapping to route metadata for custom methods:
# - update_transform: Composite method like fit_transform (update + transform)
# - update_predict: Composite method for forecasters (update + predict)
# Note: 'update' itself is NOT routed - it's a memory management operation
from sklearn.utils._metadata_requests import COMPOSITE_METHODS, SIMPLE_METHODS

if "update_transform" not in SIMPLE_METHODS:
    SIMPLE_METHODS.extend(["update_transform", "update_predict"])
    # Mark as composite methods (params route to constituent methods)
    COMPOSITE_METHODS["update_transform"] = ["update", "transform"]
    COMPOSITE_METHODS["update_predict"] = ["update", "predict"]

from yohou import (
    analysis,
    base,
    interval_forecaster,
    metrics,
    model_selection,
    pipeline,
    point_forecaster,
    preprocessing,
    utils,
)

__all__ = [
    "analysis",
    "base",
    "interval_forecaster",
    "metrics",
    "model_selection",
    "point_forecaster",
    "pipeline",
    "preprocessing",
    "utils",
]
