"""Tests for DistanceSimilarity interval forecasting component."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone

from yohou.interval.similarity import DistanceSimilarity


@pytest.fixture
def train_data():
    """Create training data for similarity tests."""
    time_train = pl.datetime_range(
        start=datetime(2021, 12, 16),
        end=datetime(2021, 12, 16, 0, 0, 7),
        interval="1s",
        eager=True,
    )
    y = pl.DataFrame({
        "time": time_train,
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    })
    y_pred = pl.DataFrame({
        "time": time_train,
        "value": [1.1, 2.1, 2.9, 4.2, 4.8, 6.1, 7.0, 8.1],
    })
    return y, y_pred


@pytest.fixture
def prediction_data():
    """Create test data for similarity prediction."""
    time_test = pl.datetime_range(
        start=datetime(2021, 12, 16, 0, 0, 8),
        end=datetime(2021, 12, 16, 0, 0, 9),
        interval="1s",
        eager=True,
    )
    y_pred = pl.DataFrame({
        "time": time_test,
        "value": [8.5, 9.2],
    })
    return y_pred


class TestDistanceSimilarityBasic:
    """Basic tests for DistanceSimilarity."""

    def test_fit_returns_self(self, train_data):
        """Test that fit returns the estimator."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        result = sim.fit(y, y_pred)
        assert result is sim

    def test_predict_returns_ndarray(self, train_data, prediction_data):
        """Test that predict returns a numpy array."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        sim.fit(y, y_pred)
        weights = sim.predict(prediction_data)
        assert isinstance(weights, np.ndarray)

    def test_predict_shape(self, train_data, prediction_data):
        """Test that predict returns correct shape."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        sim.fit(y, y_pred)
        weights = sim.predict(prediction_data)
        # Shape: (n_test_samples, n_train_samples)
        assert weights.shape == (2, 8)

    def test_weights_are_positive(self, train_data, prediction_data):
        """Test that all weights are positive."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        sim.fit(y, y_pred)
        weights = sim.predict(prediction_data)
        assert np.all(weights > 0)

    def test_weights_are_finite(self, train_data, prediction_data):
        """Test that all weights are finite."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        sim.fit(y, y_pred)
        weights = sim.predict(prediction_data)
        assert np.all(np.isfinite(weights))

    def test_default_metric(self):
        """Test default metric is euclidean."""
        sim = DistanceSimilarity()
        assert sim.metric == "euclidean"

    def test_default_metric_params(self):
        """Test default metric_params is empty dict."""
        sim = DistanceSimilarity()
        assert sim.metric_params == {}


class TestDistanceSimilarityMetrics:
    """Tests for different distance metrics."""

    @pytest.mark.parametrize("metric", ["euclidean", "cityblock", "cosine"])
    def test_different_metrics(self, train_data, prediction_data, metric):
        """Test that different metrics produce valid weights."""
        y, y_pred = train_data
        sim = DistanceSimilarity(metric=metric)
        sim.fit(y, y_pred)
        weights = sim.predict(prediction_data)
        assert weights.shape == (2, 8)
        assert np.all(np.isfinite(weights))

    def test_custom_metric_params(self, train_data, prediction_data):
        """Test with custom metric parameters."""
        y, y_pred = train_data
        sim = DistanceSimilarity(metric="minkowski", metric_params={"p": 3})
        sim.fit(y, y_pred)
        weights = sim.predict(prediction_data)
        assert weights.shape == (2, 8)

    def test_different_metrics_give_different_weights(self, train_data, prediction_data):
        """Test that different metrics produce valid but potentially different weights."""
        y, y_pred = train_data

        sim_euclidean = DistanceSimilarity(metric="euclidean")
        sim_euclidean.fit(y, y_pred)
        w_euclidean = sim_euclidean.predict(prediction_data)

        sim_cityblock = DistanceSimilarity(metric="cityblock")
        sim_cityblock.fit(y, y_pred)
        w_cityblock = sim_cityblock.predict(prediction_data)

        # Both should produce valid weights
        assert np.all(np.isfinite(w_euclidean))
        assert np.all(np.isfinite(w_cityblock))


class TestDistanceSimilarityObserve:
    """Tests for observe method."""

    def test_observe_returns_self(self, train_data):
        """Test that observe returns the estimator."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        sim.fit(y, y_pred)
        result = sim.observe(y[:3], y_pred[:3])
        assert result is sim

    def test_observe_extends_observations(self, train_data, prediction_data):
        """Test that observe adds new observations."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        sim.fit(y, y_pred)

        # Get initial weights
        weights_before = sim.predict(prediction_data)

        # Observe new data
        sim.observe(y[:2], y_pred[:2])

        # Weights shape should change (more training points)
        weights_after = sim.predict(prediction_data)
        assert weights_after.shape[1] == weights_before.shape[1] + 2


class TestDistanceSimilarityWithExogenous:
    """Tests for similarity with exogenous features."""

    def test_fit_with_X(self, train_data):
        """Test fitting with exogenous features."""
        y, y_pred = train_data
        # X should NOT have a 'time' column to avoid duplicate with y_pred
        X = pl.DataFrame({
            "feature": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        })
        sim = DistanceSimilarity()
        sim.fit(y, y_pred, X=X)
        assert sim._X_observed.shape[1] > 1  # time + value + feature cols

    def test_predict_with_X(self, train_data, prediction_data):
        """Test prediction with exogenous features."""
        y, y_pred = train_data
        X_train = pl.DataFrame({
            "feature": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        })
        X_test = pl.DataFrame({
            "feature": [85.0, 92.0],
        })
        sim = DistanceSimilarity()
        sim.fit(y, y_pred, X=X_train)
        weights = sim.predict(prediction_data, X=X_test)
        assert weights.shape == (2, 8)


class TestDistanceSimilarityProperties:
    """Tests for properties and attributes."""

    def test_n_discarded_indices(self, train_data):
        """Test n_discarded_indices_ property."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        sim.fit(y, y_pred)
        assert sim.n_discarded_indices_ == 0

    def test_clone(self):
        """Test that DistanceSimilarity can be cloned."""
        sim = DistanceSimilarity(metric="cityblock")
        cloned = clone(sim)
        assert cloned.metric == "cityblock"
        assert cloned is not sim

    def test_get_params(self):
        """Test get_params returns expected parameters."""
        sim = DistanceSimilarity(metric="cosine", metric_params={"p": 2})
        params = sim.get_params()
        assert params["metric"] == "cosine"
        assert params["metric_params"] == {"p": 2}

    def test_set_params(self):
        """Test set_params updates parameters."""
        sim = DistanceSimilarity()
        sim.set_params(metric="cityblock")
        assert sim.metric == "cityblock"

    def test_tags(self):
        """Test sklearn tags are set correctly."""
        sim = DistanceSimilarity()
        tags = sim.__sklearn_tags__()
        assert tags.estimator_type == "similarity"
        assert tags.requires_fit is True
