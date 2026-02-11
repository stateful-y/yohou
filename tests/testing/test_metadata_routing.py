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
    """Enable metadata routing for all tests in this module."""
    set_config(enable_metadata_routing=True)


class RecordingEstimator(BaseEstimator):
    """Mock estimator that records metadata."""

    def fit(self, X=None, y=None, **kwargs):
        """Fit with metadata recording."""
        record_metadata(self, **kwargs)
        return self


class TestAssertRequestIsEmpty:
    """Tests for assert_request_is_empty utility."""

    def test_with_empty_request(self):
        """Test assert_request_is_empty passes for empty request."""
        request = MetadataRequest(owner="test")

        # Should not raise - request is empty
        assert_request_is_empty(request)

    def test_with_non_empty_request(self):
        """Test assert_request_is_empty fails for non-empty request."""
        request = MetadataRequest(owner="test")
        request.fit.add_request(param="sample_weight", alias=True)

        # Should raise AssertionError
        with pytest.raises(AssertionError, match="Method fit has non-empty requests"):
            assert_request_is_empty(request)

    def test_with_exclude(self):
        """Test assert_request_is_empty with exclude parameter."""
        request = MetadataRequest(owner="test")
        request.fit.add_request(param="sample_weight", alias=True)

        # Should not raise - sample_weight is excluded
        assert_request_is_empty(request, exclude={"sample_weight": True})

    def test_with_router(self):
        """Test assert_request_is_empty works with MetadataRouter."""
        router = MetadataRouter(owner="test")

        # Should not raise - router is empty
        assert_request_is_empty(router)


class TestCheckRecordedMetadata:
    """Tests for check_recorded_metadata utility."""

    def test_success(self):
        """Test check_recorded_metadata passes when expectation matches."""
        est = RecordingEstimator()

        sample_weight = [1.0, 1.0]
        est.fit(sample_weight=sample_weight)

        # Check recorded metadata (parent is name of this test function)
        check_recorded_metadata(
            est,
            method="fit",
            parent="test_success",
            sample_weight=sample_weight,
        )

    def test_failure(self):
        """Test check_recorded_metadata fails when expectation mismatch."""
        est = RecordingEstimator()

        est.fit(sample_weight=[1.0])

        # Should raise AssertionError because expected doesn't match recorded
        with pytest.raises(AssertionError):
            check_recorded_metadata(
                est,
                method="fit",
                parent="test_failure",
                sample_weight=[2.0],
            )

    def test_missing_param_failure(self):
        """Test check fails when expected metadata is not recorded."""
        est = RecordingEstimator()

        est.fit()  # No metadata recorded

        with pytest.raises(AssertionError):
            check_recorded_metadata(
                est,
                method="fit",
                parent="test_missing_param_failure",
                sample_weight=[1.0],
            )
