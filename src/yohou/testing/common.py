"""Common check functions shared across forecasters and transformers.

This module provides checks that apply to both forecasters and transformers,
including metadata routing validation.
"""

import importlib.util

if importlib.util.find_spec("polars") is None:
    raise ImportError("polars.testing is required for yohou.testing module. Install with: uv sync --group tests")

from sklearn.utils.metadata_routing import MetadataRequest, MetadataRouter

from .metadata_routing import assert_request_is_empty

__all__ = ["check_metadata_routing_default_request", "check_metadata_routing_get_metadata_routing"]


def check_metadata_routing_default_request(estimator_fitted) -> None:
    """Check that by default metadata routing request is empty.

    Tests:
    - get_metadata_routing() returns MetadataRouter or MetadataRequest
    - Default requests are empty (all metadata values are None)

    Parameters
    ----------
    estimator_fitted : BaseForecaster | BaseTransformer
        A fitted estimator instance

    Raises
    ------
    AssertionError
        If routing structure is incorrect or requests are not empty

    """
    # Routing is always enabled in Yohou - no check needed

    router = estimator_fitted.get_metadata_routing()
    assert isinstance(router, MetadataRouter | MetadataRequest), (
        f"Expected MetadataRouter or MetadataRequest, got {type(router)}"
    )

    # Check requests are empty (with possible exclusions for defaults)
    exclude = {}  # Can add specific exclusions per estimator type
    assert_request_is_empty(router, exclude=exclude)


def check_metadata_routing_get_metadata_routing(estimator_fitted) -> None:
    """Check that get_metadata_routing() is implemented correctly.

    Tests:
    - Method exists and returns MetadataRouter or MetadataRequest
    - Router has an owner set

    Parameters
    ----------
    estimator_fitted : BaseForecaster | BaseTransformer
        A fitted estimator instance

    Raises
    ------
    AssertionError
        If get_metadata_routing implementation is incorrect

    """
    assert hasattr(estimator_fitted, "get_metadata_routing"), (
        f"{type(estimator_fitted).__name__} must implement get_metadata_routing()"
    )

    router = estimator_fitted.get_metadata_routing()

    assert isinstance(router, MetadataRouter | MetadataRequest), (
        f"get_metadata_routing() must return MetadataRouter or MetadataRequest, got {type(router)}"
    )

    # Check owner is set
    assert router.owner is not None, "Router must have an owner set"
