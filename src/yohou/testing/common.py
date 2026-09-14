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

    # A step-output transformer requests ``forecasting_horizon`` on ``fit`` by default:
    # the horizon is fit metadata that a reduction forecaster routes to it. That one
    # request is the whole exemption; every other request must still be empty.
    transformer_tags = getattr(estimator_fitted.__sklearn_tags__(), "transformer_tags", None)
    if isinstance(router, MetadataRequest) and transformer_tags is not None and transformer_tags.produces_step_columns:
        fit_requests = dict(router.fit.requests)
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
        assert_request_is_empty(router, exclude=["fit"])
        return

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
