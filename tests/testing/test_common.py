"""Tests for yohou.testing.common check functions."""

import pytest

from yohou.point.naive import SeasonalNaive
from yohou.preprocessing.window import LagTransformer
from yohou.testing.common import (
    check_metadata_routing_default_request,
    check_metadata_routing_get_metadata_routing,
)


class MockEstimatorNoGetMetadataRouting:
    """Mock estimator without get_metadata_routing."""

    pass


class MockEstimatorBadReturn:
    """Mock estimator with bad get_metadata_routing return."""

    def get_metadata_routing(self):
        """Return invalid type."""
        return "not a router"


class TestMetadataRoutingDefaultRequest:
    """Tests for check_metadata_routing_default_request."""

    def test_forecaster(self, y_X_factory):
        """Test check_metadata_routing_default_request for forecasters."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        forecaster = SeasonalNaive(seasonality=12)
        forecaster.fit(y[:40], X[:40], forecasting_horizon=3)

        # Should not raise - default requests are empty
        check_metadata_routing_default_request(forecaster)

    def test_transformer(self, y_X_factory):
        """Test check_metadata_routing_default_request for transformers."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise - default requests are empty
        check_metadata_routing_default_request(transformer)


class TestMetadataRoutingGetMetadataRouting:
    """Tests for check_metadata_routing_get_metadata_routing."""

    def test_forecaster(self, y_X_factory):
        """Test check_metadata_routing_get_metadata_routing for forecasters."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        forecaster = SeasonalNaive(seasonality=12)
        forecaster.fit(y[:40], X[:40], forecasting_horizon=3)

        # Should not raise; the check verifies the router type and owner itself.
        check_metadata_routing_get_metadata_routing(forecaster)

    def test_transformer(self, y_X_factory):
        """Test check_metadata_routing_get_metadata_routing for transformers."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise; the check verifies the router type and owner itself.
        check_metadata_routing_get_metadata_routing(transformer)

    def test_missing_method(self):
        """Test check fails when get_metadata_routing is missing."""
        estimator = MockEstimatorNoGetMetadataRouting()

        with pytest.raises(AssertionError, match="must implement get_metadata_routing"):
            check_metadata_routing_get_metadata_routing(estimator)

    def test_bad_return(self):
        """Test check fails when get_metadata_routing returns wrong type."""
        estimator = MockEstimatorBadReturn()

        with pytest.raises(AssertionError, match="must return MetadataRouter or MetadataRequest"):
            check_metadata_routing_get_metadata_routing(estimator)
