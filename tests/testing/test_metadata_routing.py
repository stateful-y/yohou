"""Tests for yohou.testing.metadata_routing utility functions."""

import pytest
from sklearn import set_config
from sklearn.base import BaseEstimator
from sklearn.utils.metadata_routing import MetadataRequest, MetadataRouter

from yohou.testing.metadata_routing import (
    assert_request_is_empty,
    check_recorded_metadata,
    record_metadata,
)


# Enable metadata routing for validation
def setup_module():
    set_config(enable_metadata_routing=True)


class RecordingEstimator(BaseEstimator):
    """Mock estimator that records metadata."""

    def fit(self, X=None, y=None, **kwargs):
        """Fit with metadata recording."""
        record_metadata(self, **kwargs)
        return self


def test_assert_request_is_empty_with_empty_request():
    """Test assert_request_is_empty passes for empty request."""
    request = MetadataRequest(owner="test")

    # Should not raise - request is empty
    assert_request_is_empty(request)


def test_assert_request_is_empty_with_non_empty_request():
    """Test assert_request_is_empty fails for non-empty request."""
    request = MetadataRequest(owner="test")
    request.fit.add_request(param="sample_weight", alias=True)

    # Should raise AssertionError
    with pytest.raises(AssertionError, match="Method fit has non-empty requests"):
        assert_request_is_empty(request)


def test_assert_request_is_empty_with_exclude():
    """Test assert_request_is_empty with exclude parameter."""
    request = MetadataRequest(owner="test")
    request.fit.add_request(param="sample_weight", alias=True)

    # Should not raise - sample_weight is excluded
    assert_request_is_empty(request, exclude={"sample_weight": True})


def test_assert_request_is_empty_with_router():
    """Test assert_request_is_empty works with MetadataRouter."""
    router = MetadataRouter(owner="test")

    # Should not raise - router is empty
    assert_request_is_empty(router)


def test_check_recorded_metadata_success():
    """Test check_recorded_metadata passes when expectation matches."""
    est = RecordingEstimator()
    # est.set_fit_request(sample_weight=True) # Not needed for direct call test

    sample_weight = [1.0, 1.0]
    est.fit(sample_weight=sample_weight)

    # Check recorded metadata (parent is name of this test function)
    check_recorded_metadata(
        est,
        method="fit",
        parent="test_check_recorded_metadata_success",
        sample_weight=sample_weight,
    )


def test_check_recorded_metadata_failure():
    """Test check_recorded_metadata fails when expectation mismatch."""
    est = RecordingEstimator()
    # est.set_fit_request(sample_weight=True)

    est.fit(sample_weight=[1.0])

    # Should raise AssertionError because expected doesn't match recorded
    with pytest.raises(AssertionError):
        check_recorded_metadata(
            est,
            method="fit",
            parent="test_check_recorded_metadata_failure",
            sample_weight=[2.0],
        )


def test_check_recorded_metadata_missing_param_failure():
    """Test check fails when expected metadata is not recorded."""
    est = RecordingEstimator()
    # No request set, so no metadata passed (or ignored)

    est.fit()  # No metadata recorded

    with pytest.raises(AssertionError):
        check_recorded_metadata(
            est,
            method="fit",
            parent="test_check_recorded_metadata_missing_param_failure",
            sample_weight=[1.0],
        )
