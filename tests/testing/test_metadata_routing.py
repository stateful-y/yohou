"""Tests for yohou.testing.metadata_routing utility functions."""

import copy

import numpy as np
import polars as pl
import pytest
from sklearn import set_config
from sklearn.base import BaseEstimator
from sklearn.utils.metadata_routing import MetadataRequest, MetadataRouter

from yohou.testing.metadata_routing import (
    _Registry,
    assert_request_equal,
    assert_request_is_empty,
    check_recorded_metadata,
    record_metadata,
    record_metadata_not_default,
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


class TestAssertRequestEqual:
    """Tests for assert_request_equal utility."""

    def test_matching_request(self):
        """Matching method/param pair passes assertion."""
        request = MetadataRequest(owner="test")
        request.fit.add_request(param="sample_weight", alias=True)
        assert_request_equal(request, {"fit": {"sample_weight": True}})

    def test_unlisted_methods_empty(self):
        """Methods not in the dictionary must be empty."""
        request = MetadataRequest(owner="test")
        request.fit.add_request(param="sample_weight", alias=True)
        request.predict.add_request(param="extra", alias="alias_name")
        with pytest.raises(AssertionError, match="predict"):
            assert_request_equal(request, {"fit": {"sample_weight": True}})

    def test_mismatch_raises(self):
        """Mismatched param value raises AssertionError."""
        request = MetadataRequest(owner="test")
        request.fit.add_request(param="sample_weight", alias=True)
        with pytest.raises(AssertionError, match="fit"):
            assert_request_equal(request, {"fit": {"sample_weight": False}})


class TestRegistryCopy:
    """Tests for _Registry copy semantics."""

    def test_deepcopy_returns_same(self):
        """Deep copy returns the same registry instance."""
        reg = _Registry()
        reg.append("item")
        deep = copy.deepcopy(reg)
        assert deep is reg
        assert deep == ["item"]

    def test_copy_returns_same(self):
        """Shallow copy returns the same registry instance."""
        reg = _Registry()
        reg.append("other")
        shallow = copy.copy(reg)
        assert shallow is reg


class _NotDefaultRecorder:
    """Helper that wraps record_metadata_not_default in a named method."""

    def record(self, **params):
        """Call record_metadata_not_default so stack frames are predictable."""
        record_metadata_not_default(self, **params)


class TestRecordMetadataNotDefault:
    """Tests for record_metadata_not_default partial."""

    def test_skips_default_string(self):
        """Parameters with value 'default' are skipped."""
        obj = _NotDefaultRecorder()
        obj.record(sample_weight="default", real_param=42)
        records = obj._records["record"]["test_skips_default_string"]
        assert len(records) == 1
        assert "sample_weight" not in records[0]
        assert records[0]["real_param"] == 42

    def test_keeps_non_default(self):
        """Non-default values are always recorded."""
        obj = _NotDefaultRecorder()
        obj.record(a=1, b="hello")
        records = obj._records["record"]["test_keeps_non_default"]
        assert records[0] == {"a": 1, "b": "hello"}


class TestCheckRecordedMetadataAdvanced:
    """Tests for split_params and polars Series in check_recorded_metadata."""

    def test_split_params_numpy(self):
        """split_params validates subset relationship for numpy arrays."""
        est = RecordingEstimator()
        full = np.array([1, 2, 3, 4, 5])
        est._records = {"fit": {"caller": [{"weight": np.array([1, 3])}]}}
        check_recorded_metadata(
            est,
            method="fit",
            parent="caller",
            split_params=("weight",),
            weight=full,
        )

    def test_split_params_polars_series(self):
        """split_params validates subset for polars Series recorded data."""
        est = RecordingEstimator()
        full = pl.Series("w", [10, 20, 30, 40])
        subset = pl.Series("w", [10, 30])
        est._records = {"fit": {"caller": [{"weight": subset}]}}
        check_recorded_metadata(
            est,
            method="fit",
            parent="caller",
            split_params=("weight",),
            weight=full,
        )

    def test_polars_series_equality(self):
        """Polars Series compared by equals() when not in split_params."""
        est = RecordingEstimator()
        s = pl.Series("val", [1, 2, 3])
        est._records = {"fit": {"caller": [{"data": s}]}}
        check_recorded_metadata(est, method="fit", parent="caller", data=s)
