"""Tests for interval utility functions."""

import numpy as np
import pytest

from yohou.interval.utils import weighted_quantile


def test_weighted_quantile_reexported_from_package():
    """weighted_quantile is reachable via the public yohou.interval namespace."""
    import yohou.interval as interval_pkg

    assert "weighted_quantile" in interval_pkg.__all__
    assert interval_pkg.weighted_quantile is weighted_quantile


class TestWeightedQuantile:
    """Tests for weighted_quantile function."""

    def test_basic_median(self):
        """Test weighted quantile acts as median with equal weights."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        result = weighted_quantile(x, q=0.5, weights=weights)
        assert isinstance(result, float)

    def test_high_quantile(self):
        """Small q selects the largest element under the 1-q CDF threshold.

        With equal weights on [1, 2, 3, 4, 5], q=0.1 walks the sorted CDF until
        it reaches 1 - q = 0.9, which lands on the largest element, 5.0.
        """
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        result = weighted_quantile(x, q=0.1, weights=weights)
        assert result == pytest.approx(5.0)

    def test_low_quantile(self):
        """Large q selects the smallest element under the 1-q CDF threshold.

        With equal weights on [1, 2, 3, 4, 5], q=0.9 reaches 1 - q = 0.1 at the
        first sorted element, 1.0.
        """
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        result = weighted_quantile(x, q=0.9, weights=weights)
        assert result == pytest.approx(1.0)

    def test_concentrated_weight(self):
        """Test with weight concentrated on one element."""
        x = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 1.0, 0.0])
        result = weighted_quantile(x, q=0.5, weights=weights)
        assert result == pytest.approx(2.0)

    def test_returns_float(self):
        """Test that result is always a float."""
        x = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.5, 0.3, 0.2])
        result = weighted_quantile(x, q=0.5, weights=weights)
        assert isinstance(result, float)

    def test_zero_weights_raises(self):
        """Test that all-zero weights raise a descriptive ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="sum to zero"):
            weighted_quantile(x, q=0.1, weights=weights)

    def test_small_weights_normalizes(self):
        """Test that small (but non-zero) weights are normalized and produce a finite result."""
        x = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.01, 0.01, 0.01])
        result = weighted_quantile(x, q=0.1, weights=weights)
        assert np.isfinite(result)

    def test_single_element(self):
        """Test with single element array."""
        x = np.array([5.0])
        weights = np.array([1.0])
        result = weighted_quantile(x, q=0.5, weights=weights)
        assert result == pytest.approx(5.0)

    def test_unsorted_input(self):
        """Unsorted input yields the same median as sorted input.

        The function sorts internally, so [5, 1, 3, 2, 4] with equal weights and
        q=0.5 must give the median element, 3.0.
        """
        x = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        result = weighted_quantile(x, q=0.5, weights=weights)
        assert result == pytest.approx(3.0)

    def test_boundary_q_zero(self):
        """Test with q=0 (100th percentile)."""
        x = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.4, 0.3, 0.3])
        result = weighted_quantile(x, q=0.0, weights=weights)
        assert isinstance(result, float)

    def test_large_array(self):
        """Test with larger array for numerical stability."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        weights = rng.random(100)
        weights = weights / weights.sum()  # Normalize
        result = weighted_quantile(x, q=0.5, weights=weights)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_shape_mismatch_raises(self):
        """Test that mismatched x and weights lengths raise ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="same length"):
            weighted_quantile(x, q=0.5, weights=weights)
