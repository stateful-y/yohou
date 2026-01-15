"""Metadata routing utilities for testing Yohou estimators.

This module provides test utilities for verifying metadata routing behavior
in Yohou estimators, following sklearn's testing patterns.
"""

import inspect
from collections import defaultdict
from functools import partial

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.utils._metadata_requests import SIMPLE_METHODS
from sklearn.utils.metadata_routing import MetadataRouter

try:
    import polars as pl
except ImportError as e:
    raise ImportError(
        "polars.testing is required for yohou.testing module. "
        "Install with: uv sync --group tests"
    ) from e


class _Registry(list):
    """Registry to track which estimators have been called.

    This is used to verify metadata is routed to the correct sub-estimators.
    Overrides __deepcopy__ and __copy__ to ensure references are preserved
    during cloning.
    """

    def __deepcopy__(self, memo):
        """Deep copy returns self to preserve registry reference.

        Parameters
        ----------
        memo : dict
            Memoization dictionary

        Returns
        -------
        self : _Registry
            The same registry instance

        """
        return self

    def __copy__(self):
        """Shallow copy returns self to preserve registry reference.

        Returns
        -------
        self : _Registry
            The same registry instance

        """
        return self


def record_metadata(obj, record_default: bool = True, **kwargs) -> None:
    """Utility function to store passed metadata to a method of obj.

    If record_default is False, kwargs whose values are "default" are skipped.
    This is so that checks on keyword arguments whose default was not changed
    are skipped.

    Parameters
    ----------
    obj : object
        Object to record metadata on (usually has a registry attribute)
    record_default : bool, default=True
        Whether to record kwargs with "default" values
    **kwargs : dict
        Metadata to record

    """
    stack = inspect.stack()
    callee = stack[1].function
    caller = stack[2].function
    if not hasattr(obj, "_records"):
        obj._records = defaultdict(lambda: defaultdict(list))
    if not record_default:
        kwargs = {
            key: val
            for key, val in kwargs.items()
            if not isinstance(val, str) or (val != "default")
        }
    obj._records[callee][caller].append(kwargs)


# Partial application for common use case
record_metadata_not_default = partial(record_metadata, record_default=False)


def check_recorded_metadata(
    obj, method: str, parent: str, split_params: tuple = (), **kwargs
) -> None:
    """Check whether the expected metadata is passed to the object's method.

    Parameters
    ----------
    obj : estimator object
        Sub-estimator to check routed params for
    method : str
        Sub-estimator's method where metadata is routed to (callee)
    parent : str
        The parent method which called `method` (caller)
    split_params : tuple, default=empty
        Parameters which should be checked as subsets of the original values
        (used for CV splits where each fold gets a subset)
    **kwargs : dict
        Expected metadata that should have been passed

    Raises
    ------
    AssertionError
        If recorded metadata doesn't match expected metadata

    """
    all_records = getattr(obj, "_records", dict()).get(method, dict()).get(parent, list())
    for record in all_records:
        # Check that metadata names match
        assert set(kwargs.keys()) == set(record.keys()), (
            f"Expected {kwargs.keys()} vs {record.keys()}"
        )
        for key, value in kwargs.items():
            recorded_value = record[key]
            # For split_params, check if recorded is a subset of original
            if key in split_params and recorded_value is not None:
                if isinstance(value, pl.Series):
                    value = value.to_numpy()
                if isinstance(recorded_value, pl.Series):
                    recorded_value = recorded_value.to_numpy()
                assert np.isin(recorded_value, value).all()
            else:
                if isinstance(recorded_value, np.ndarray):
                    assert_array_equal(recorded_value, value)
                elif isinstance(recorded_value, pl.Series):
                    assert recorded_value.equals(value)
                else:
                    assert recorded_value is value, (
                        f"Expected {recorded_value} vs {value}. Method: {method}"
                    )


def assert_request_is_empty(metadata_request, exclude=None) -> None:
    """Check if a metadata request dict is empty.

    One can exclude a method or a list of methods from the check using the
    ``exclude`` parameter. If metadata_request is a MetadataRouter, then
    ``exclude`` can be of the form ``{"object" : [method, ...]}``.

    Parameters
    ----------
    metadata_request : MetadataRequest | MetadataRouter
        The request object to check
    exclude : str | list[str] | dict, optional
        Methods or objects to exclude from the check

    Raises
    ------
    AssertionError
        If metadata request is not empty

    """
    if isinstance(metadata_request, MetadataRouter):
        for name, route_mapping in metadata_request:
            if exclude is not None and name in exclude:
                _exclude = exclude[name]
            else:
                _exclude = None
            assert_request_is_empty(route_mapping.router, exclude=_exclude)
        return

    exclude = [] if exclude is None else exclude
    # Yohou methods: fit, predict, update, update_predict, update_transform, transform, score
    yohou_methods = [
        "fit",
        "predict",
        "update",
        "update_predict",
        "update_transform",
        "transform",
        "inverse_transform",
        "score",
    ]
    for method in yohou_methods:
        if method in exclude:
            continue
        mmr = getattr(metadata_request, method, None)
        if mmr is None:
            continue
        props = [
            prop
            for prop, alias in mmr.requests.items()
            if isinstance(alias, str) or alias is not None
        ]
        assert not props, f"Method {method} has non-empty requests: {props}"


def assert_request_equal(request, dictionary: dict) -> None:
    """Assert metadata request matches expected dictionary.

    Parameters
    ----------
    request : MetadataRequest
        The request object to check
    dictionary : dict
        Expected requests in the form {"method": {"param": value}}

    Raises
    ------
    AssertionError
        If request doesn't match expected dictionary

    """
    for method, requests in dictionary.items():
        mmr = getattr(request, method)
        assert mmr.requests == requests, f"Method {method}: expected {requests}, got {mmr.requests}"

    empty_methods = [
        method
        for method in SIMPLE_METHODS + ["update", "update_predict", "update_transform"]
        if method not in dictionary
    ]
    for method in empty_methods:
        mmr = getattr(request, method, None)
        if mmr is not None:
            assert not len(mmr.requests), f"Method {method} should be empty but has {mmr.requests}"
