"""Tests for interval utility functions."""

import warnings

import numpy as np
import polars as pl
import pytest

from yohou.interval.utils import (
    required_calibration_size,
    warn_if_calibration_too_small,
    weighted_quantile,
)


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


class TestConformalCorrection:
    """The (n+1) correction and the rate-resolution guard."""

    @pytest.mark.parametrize(
        ("rate", "symmetric", "expected"),
        [
            (0.9, True, 9),
            (0.99, True, 99),
            (0.9, False, 19),
            (0.99, False, 199),
            (1.0, True, 0),
        ],
    )
    def test_required_calibration_size(self, rate, symmetric, expected):
        """``n >= q / (1 - q)`` is where ``ceil((n+1) * q)`` still fits in ``n``."""
        assert required_calibration_size(rate, symmetric) == expected

    @pytest.mark.parametrize(("rate", "symmetric"), [(0.9, True), (0.99, True), (0.9, False), (0.8, False)])
    def test_required_size_is_exactly_where_the_index_fits(self, rate, symmetric):
        required = required_calibration_size(rate, symmetric)
        tail = rate if symmetric else 1.0 - (1.0 - rate) / 2.0

        assert int(np.ceil((required + 1) * tail)) <= required
        assert int(np.ceil(required * tail)) > required - 1

    def test_warns_below_the_required_size_and_is_silent_above(self):
        with pytest.warns(UserWarning, match="calibration scores per value column"):
            assert warn_if_calibration_too_small(10, 0.99, symmetric=True, step=1)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            assert not warn_if_calibration_too_small(200, 0.99, symmetric=True, step=1)

    def test_weighted_quantile_does_not_renormalize(self):
        """Reserved test-point mass must survive into the quantile.

        Renormalizing the row to sum to 1 cancels the reservation and costs
        exactly one order statistic, which is the weighted analogue of dropping
        the ``(n+1)`` correction.
        """
        n = 20
        x = np.arange(n, dtype=float)
        reserved = np.full(n, 1.0 / (n + 1))

        picked = weighted_quantile(x, 0.1, reserved)
        conformal_index = int(np.ceil((n + 1) * 0.9))

        assert picked == pytest.approx(x[conformal_index - 1])

    def test_weighted_quantile_falls_back_when_mass_is_short(self):
        """Too little calibration mass means an unbounded bound; return the widest finite one."""
        n = 5
        x = np.arange(n, dtype=float)
        # Total mass 5/21 never reaches 0.99.
        assert weighted_quantile(x, 0.01, np.full(n, 1.0 / 21.0)) == pytest.approx(x[-1])

    @pytest.mark.parametrize("rate", [0.5, 0.8, 0.9, 0.95])
    def test_weighted_and_unweighted_paths_select_the_same_score(self, rate):
        """The two reductions must agree, or configuring a similarity would shift the bound.

        The unweighted path takes ``ceil((n + 1) * rate)`` directly; the weighted
        path reaches the same index because uniform reserved-mass weights of
        ``1 / (n + 1)`` accumulate to ``rate`` at exactly that point.
        """
        from yohou.metrics.conformity_base import BaseConformityScorer

        rng = np.random.default_rng(5)
        scores = np.sort(rng.normal(0, 1, 40))
        frame = pl.DataFrame({"value": scores})

        unweighted = BaseConformityScorer._compute_symmetric_quantiles(frame, rate)[0]
        reserved = np.full(scores.size, 1.0 / (scores.size + 1))
        weighted = weighted_quantile(scores, 1.0 - rate, reserved)

        assert weighted == pytest.approx(unweighted)
