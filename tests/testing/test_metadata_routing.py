"""Tests for yohou.testing.metadata_routing utility functions."""

import pytest
from sklearn.utils.metadata_routing import MetadataRequest, MetadataRouter

from yohou.point_forecaster.naive import SeasonalNaive
from yohou.preprocessing.window import LagTransformer
from yohou.testing.metadata_routing import (
    assert_request_is_empty,
    check_recorded_metadata,
)


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


def test_check_recorded_metadata_forecaster(y_X_factory):
    """Test check_recorded_metadata for forecasters."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    forecaster = SeasonalNaive(seasonality=12)
    forecaster.fit(y[:40], X[:40], forecasting_horizon=3)

    # Set up metadata recording
    forecaster.set_fit_request(sample_weight=True)
    sample_weight = [1.0] * 40

    # Should not raise
    check_recorded_metadata(
        forecaster,
        method="fit",
        expected={"sample_weight": sample_weight},
        y=y[:40],
        X=X[:40],
        forecasting_horizon=3,
        sample_weight=sample_weight,
    )


def test_check_recorded_metadata_transformer(y_X_factory):
    """Test check_recorded_metadata for transformers."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    transformer = LagTransformer(lag=3)
    transformer.fit(X[:40], y[:40])

    # Set up metadata recording
    transformer.set_fit_request(sample_weight=True)
    sample_weight = [1.0] * 40

    # Should not raise
    check_recorded_metadata(
        transformer,
        method="fit",
        expected={"sample_weight": sample_weight},
        X=X[:40],
        y=y[:40],
        sample_weight=sample_weight,
    )


def test_check_recorded_metadata_missing_param(y_X_factory):
    """Test check fails when expected metadata is not recorded."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    forecaster = SeasonalNaive(seasonality=12)

    # Fit without setting request
    forecaster.fit(y[:40], X[:40], forecasting_horizon=3)

    # Should raise AssertionError - metadata not recorded
    with pytest.raises(AssertionError):
        check_recorded_metadata(
            forecaster,
            method="fit",
            expected={"sample_weight": [1.0] * 40},
            y=y[:40],
            X=X[:40],
            forecasting_horizon=3,
            sample_weight=[1.0] * 40,
        )
