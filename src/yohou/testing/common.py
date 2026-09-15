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
    - Default requests are empty (all metadata values are None), except the
      ``forecasting_horizon`` fit request a ``produces_step_columns`` transformer
      must declare

    Parameters
    ----------
    estimator_fitted : BaseForecaster | BaseActualTransformer
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

    _assert_default_requests_empty(router)


def _produces_step_columns(owner) -> bool:
    """Whether the owner of a metadata request declares step-column output."""
    get_tags = getattr(owner, "__sklearn_tags__", None)
    if not callable(get_tags):
        return False
    transformer_tags = getattr(get_tags(), "transformer_tags", None)
    return bool(transformer_tags is not None and transformer_tags.produces_step_columns)


def _assert_default_requests_empty(request) -> None:
    """Assert every default request in a routing tree is empty, bar the step-output exemption.

    A transformer tagged ``produces_step_columns`` requests ``forecasting_horizon`` on
    ``fit`` by default, because a reduction forecaster routes its fit horizon to it.
    That one request is the whole exemption, and it applies wherever the transformer
    sits: bare, inside a composite, or inside a forecaster's ``actual_transformer``.
    Every other request, on it or on anything else in the tree, must still be empty.

    Parameters
    ----------
    request : MetadataRequest or MetadataRouter
        A request, or a router whose route mappings are walked recursively.

    Raises
    ------
    AssertionError
        If a default request is not empty.

    """
    if isinstance(request, MetadataRouter):
        for _, route_mapping in request:
            _assert_default_requests_empty(route_mapping.router)
        return

    if not _produces_step_columns(request.owner):
        assert_request_is_empty(request)
        return

    fit_requests = dict(request.fit.requests)
    assert fit_requests.get("forecasting_horizon") is True, (
        "A transformer tagged produces_step_columns must request forecasting_horizon on fit by default, "
        f"got {fit_requests.get('forecasting_horizon')!r}"
    )
    other = [
        prop
        for prop, alias in fit_requests.items()
        if prop != "forecasting_horizon" and (isinstance(alias, str) or alias is not None)
    ]
    assert not other, f"Method fit has non-empty requests: {other}"
    assert_request_is_empty(request, exclude=["fit"])


def check_metadata_routing_get_metadata_routing(estimator_fitted) -> None:
    """Check that get_metadata_routing() is implemented correctly.

    Tests:
    - Method exists and returns MetadataRouter or MetadataRequest
    - Router has an owner set

    Parameters
    ----------
    estimator_fitted : BaseForecaster | BaseActualTransformer
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
