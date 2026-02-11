"""Tests for time-based weighting utilities."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from yohou.utils.weighting import (
    compose_weights,
    exponential_decay_weight,
    linear_decay_weight,
    seasonal_emphasis_weight,
    validate_callable_signature,
)


@pytest.fixture
def daily_times():
    """Return a polars Series of 10 daily datetimes."""
    start = datetime(2024, 1, 1)
    return pl.Series("time", [start + timedelta(days=i) for i in range(10)])


@pytest.fixture
def short_times():
    """Return a polars Series of 3 daily datetimes."""
    start = datetime(2024, 1, 1)
    return pl.Series("time", [start + timedelta(days=i) for i in range(3)])


@pytest.fixture
def single_time():
    """Return a polars Series of 1 datetime."""
    return pl.Series("time", [datetime(2024, 1, 1)])


class TestExponentialDecayWeight:
    """Tests for exponential_decay_weight."""

    def test_most_recent_has_weight_one(self, daily_times):
        """Most recent time point should have weight 1.0."""
        weight_fn = exponential_decay_weight(half_life=3)
        weights = weight_fn(daily_times)
        assert weights[-1] == pytest.approx(1.0)

    def test_oldest_has_lowest_weight(self, daily_times):
        """Oldest time point should have the lowest weight."""
        weight_fn = exponential_decay_weight(half_life=3)
        weights = weight_fn(daily_times)
        assert weights[0] < weights[-1]
        assert weights[0] > 0  # Never zero for exponential

    def test_monotonically_increasing(self, daily_times):
        """Weights should be monotonically increasing (older → newer)."""
        weight_fn = exponential_decay_weight(half_life=3)
        weights = weight_fn(daily_times)
        arr = weights.to_numpy()
        assert np.all(np.diff(arr) > 0)

    def test_half_life_numeric(self, short_times):
        """Numeric half_life should produce correct decay pattern."""
        weight_fn = exponential_decay_weight(half_life=1)
        weights = weight_fn(short_times)
        # At half_life=1 day: weight at t-1 = 0.5, t-2 = 0.25
        assert weights[2] == pytest.approx(1.0)
        assert weights[1] == pytest.approx(0.5)
        assert weights[0] == pytest.approx(0.25)

    def test_half_life_timedelta(self, short_times):
        """Timedelta half_life should work equivalently to numeric."""
        weight_fn = exponential_decay_weight(half_life=timedelta(days=1))
        weights = weight_fn(short_times)
        assert weights[2] == pytest.approx(1.0)
        assert weights[1] == pytest.approx(0.5)

    def test_returns_float64_series(self, daily_times):
        """Output should be Float64 polars Series named 'weight'."""
        weight_fn = exponential_decay_weight(half_life=3)
        weights = weight_fn(daily_times)
        assert weights.dtype == pl.Float64
        assert weights.name == "weight"

    def test_all_positive(self, daily_times):
        """All weights should be strictly positive."""
        weight_fn = exponential_decay_weight(half_life=0.5)
        weights = weight_fn(daily_times)
        assert (weights > 0).all()


class TestLinearDecayWeight:
    """Tests for linear_decay_weight."""

    def test_most_recent_has_weight_one(self, daily_times):
        """Most recent time point should have weight 1.0."""
        weight_fn = linear_decay_weight()
        weights = weight_fn(daily_times)
        assert weights[-1] == pytest.approx(1.0)

    def test_oldest_has_weight_zero(self, daily_times):
        """Oldest time point should have weight 0.0 with no max_steps."""
        weight_fn = linear_decay_weight()
        weights = weight_fn(daily_times)
        assert weights[0] == pytest.approx(0.0)

    def test_monotonically_increasing(self, daily_times):
        """Weights should be monotonically non-decreasing."""
        weight_fn = linear_decay_weight()
        weights = weight_fn(daily_times)
        arr = weights.to_numpy()
        assert np.all(np.diff(arr) >= 0)

    def test_max_steps_truncates(self, daily_times):
        """With max_steps, older times should get weight 0."""
        weight_fn = linear_decay_weight(max_steps=3)
        weights = weight_fn(daily_times)
        # Only last 3 points should have weight > 0
        arr = weights.to_numpy()
        assert arr[-1] == pytest.approx(1.0)
        assert np.all(arr[:-3] == 0.0)

    def test_single_point(self, single_time):
        """Single time point should have weight 1.0."""
        weight_fn = linear_decay_weight()
        weights = weight_fn(single_time)
        assert weights[0] == pytest.approx(1.0)

    def test_returns_float64_series(self, daily_times):
        """Output should be Float64 polars Series named 'weight'."""
        weight_fn = linear_decay_weight()
        weights = weight_fn(daily_times)
        assert weights.dtype == pl.Float64
        assert weights.name == "weight"


class TestSeasonalEmphasisWeight:
    """Tests for seasonal_emphasis_weight."""

    def test_most_recent_gets_emphasis(self, daily_times):
        """Most recent time point should always get emphasis weight."""
        weight_fn = seasonal_emphasis_weight(seasonality=7, emphasis=2.0)
        weights = weight_fn(daily_times)
        assert weights[-1] == pytest.approx(2.0)

    def test_in_phase_gets_emphasis(self):
        """Times matching the seasonal pattern should get emphasis."""
        start = datetime(2024, 1, 1)
        times = pl.Series("time", [start + timedelta(days=i) for i in range(14)])
        weight_fn = seasonal_emphasis_weight(seasonality=7, emphasis=3.0)
        weights = weight_fn(times)
        arr = weights.to_numpy()
        # Last index is 13 (0-indexed), so in-phase: 13, 6
        assert arr[13] == pytest.approx(3.0)
        assert arr[6] == pytest.approx(3.0)
        # Out-of-phase points
        assert arr[5] == pytest.approx(1.0)
        assert arr[7] == pytest.approx(1.0)

    def test_out_of_phase_gets_base_weight(self, daily_times):
        """Non-matching times should get weight 1.0."""
        weight_fn = seasonal_emphasis_weight(seasonality=7, emphasis=2.0)
        weights = weight_fn(daily_times)
        # Check a non-seasonal point
        arr = weights.to_numpy()
        out_of_phase_mask = arr == 1.0
        assert out_of_phase_mask.any()

    def test_multiple_seasonalities(self, daily_times):
        """Multiple seasonalities should combine with OR logic."""
        weight_fn = seasonal_emphasis_weight(seasonality=[3, 7], emphasis=2.0)
        weights = weight_fn(daily_times)
        arr = weights.to_numpy()
        # More points should be emphasized with multiple seasonalities
        n_emphasized = np.sum(arr > 1.0)
        assert n_emphasized >= 1

    def test_single_point(self, single_time):
        """Single time point should have weight 1.0."""
        weight_fn = seasonal_emphasis_weight(seasonality=7, emphasis=2.0)
        weights = weight_fn(single_time)
        assert weights[0] == pytest.approx(1.0)

    def test_returns_float64_series(self, daily_times):
        """Output should be Float64 polars Series named 'weight'."""
        weight_fn = seasonal_emphasis_weight(seasonality=7, emphasis=2.0)
        weights = weight_fn(daily_times)
        assert weights.dtype == pl.Float64
        assert weights.name == "weight"


class TestComposeWeights:
    """Tests for compose_weights."""

    def test_single_function(self, daily_times):
        """Composing a single function should return same weights."""
        single_fn = exponential_decay_weight(half_life=3)
        composed_fn = compose_weights(single_fn)

        single_weights = single_fn(daily_times)
        composed_weights = composed_fn(daily_times)
        np.testing.assert_allclose(composed_weights.to_numpy(), single_weights.to_numpy())

    def test_two_functions_multiply(self, daily_times):
        """Two functions should be multiplied element-wise."""
        fn1 = exponential_decay_weight(half_life=3)
        fn2 = linear_decay_weight()
        composed_fn = compose_weights(fn1, fn2)

        w1 = fn1(daily_times)
        w2 = fn2(daily_times)
        composed = composed_fn(daily_times)

        expected = (w1 * w2).to_numpy()
        np.testing.assert_allclose(composed.to_numpy(), expected)

    def test_empty_raises(self):
        """Composing zero functions should raise ValueError."""
        with pytest.raises(ValueError, match="At least one weight function"):
            compose_weights()

    def test_returns_float64_series(self, daily_times):
        """Output should be Float64 polars Series named 'weight'."""
        fn = compose_weights(exponential_decay_weight(half_life=3))
        weights = fn(daily_times)
        assert weights.dtype == pl.Float64
        assert weights.name == "weight"


class TestValidateCallableSignature:
    """Tests for validate_callable_signature."""

    def test_one_param(self):
        """Callable with 1 parameter should return 1."""

        def fn(time):
            return time

        assert validate_callable_signature(fn) == 1

    def test_two_params(self):
        """Callable with 2 parameters should return 2."""

        def fn(time, group_name):
            return time

        assert validate_callable_signature(fn) == 2

    def test_zero_params_raises(self):
        """Callable with 0 parameters should raise ValueError."""

        def fn():
            return None

        with pytest.raises(ValueError, match="must accept either 1 parameter"):
            validate_callable_signature(fn)

    def test_three_params_raises(self):
        """Callable with 3 parameters should raise ValueError."""

        def fn(a, b, c):
            return a

        with pytest.raises(ValueError, match="must accept either 1 parameter"):
            validate_callable_signature(fn)

    def test_lambda_one_param(self):
        """Lambda with 1 parameter should work."""
        assert validate_callable_signature(lambda t: t) == 1

    def test_lambda_two_params(self):
        """Lambda with 2 parameters should work."""
        assert validate_callable_signature(lambda t, g: t) == 2
