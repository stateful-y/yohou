"""Tests for time-based weighting utilities."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from yohou.utils.weighting import (
    combine_weight_vectors,
    compose_weights,
    exponential_decay_weight,
    linear_decay_weight,
    normalize_weights,
    resolve_dict_weights,
    resolve_weight_to_array,
    seasonal_emphasis_weight,
    validate_callable_signature,
    validate_weight_array,
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


class TestNormalizeWeights:
    """Tests for normalize_weights."""

    def test_uniform_unchanged(self):
        """Uniform weights should remain uniform after normalization."""
        w = np.array([2.0, 2.0, 2.0])
        result = normalize_weights(w)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0])
        assert result.sum() == pytest.approx(3.0)

    def test_nonuniform_sums_to_n(self):
        """Non-uniform weights should sum to n after normalization."""
        w = np.array([1.0, 3.0, 6.0])
        result = normalize_weights(w)
        assert result.sum() == pytest.approx(3.0)

    def test_single_element(self):
        """Single element normalizes to 1."""
        w = np.array([5.0])
        result = normalize_weights(w)
        assert result[0] == pytest.approx(1.0)

    def test_zero_sum_raises(self):
        """Weights summing to zero should raise ValueError."""
        w = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="sum is zero"):
            normalize_weights(w)


class TestValidateWeightArray:
    """Tests for validate_weight_array."""

    def test_valid_array_passes(self):
        """Normal positive array passes without error."""
        validate_weight_array(np.array([1.0, 2.0, 3.0]))

    def test_array_with_zeros_and_nonzeros_passes(self):
        """Array with some zeros but not all passes."""
        validate_weight_array(np.array([0.0, 1.0, 0.0]))

    def test_nan_raises(self):
        """NaN values should raise ValueError."""
        with pytest.raises(ValueError, match="contains NaN"):
            validate_weight_array(np.array([1.0, np.nan, 3.0]))

    def test_negative_raises(self):
        """Negative values should raise ValueError."""
        with pytest.raises(ValueError, match="negative values"):
            validate_weight_array(np.array([1.0, -0.5, 3.0]))

    def test_infinite_raises(self):
        """Infinite values should raise ValueError."""
        with pytest.raises(ValueError, match="infinite values"):
            validate_weight_array(np.array([1.0, np.inf, 3.0]))

    def test_all_zeros_raises(self):
        """All-zero weights should raise ValueError."""
        with pytest.raises(ValueError, match="All weights are zero"):
            validate_weight_array(np.array([0.0, 0.0]))

    def test_custom_name_in_message(self):
        """Custom name appears in error messages."""
        with pytest.raises(ValueError, match="my_weight"):
            validate_weight_array(np.array([np.nan]), name="my_weight")


class TestResolveDictWeights:
    """Tests for resolve_dict_weights."""

    def test_exact_keys(self):
        """Dict keys matching array produce correct weights."""
        result = resolve_dict_weights({1: 2.0, 2: 3.0}, [1, 2, 1])
        np.testing.assert_array_equal(result, [2.0, 3.0, 2.0])

    def test_missing_keys_use_default(self):
        """Missing keys fall back to default=1.0."""
        result = resolve_dict_weights({1: 5.0}, [1, 2, 3])
        np.testing.assert_array_equal(result, [5.0, 1.0, 1.0])

    def test_wildcard_overrides_default(self):
        """Wildcard '*' key overrides the default value."""
        result = resolve_dict_weights({"*": 0.0, 1: 2.0}, [1, 2, 3])
        np.testing.assert_array_equal(result, [2.0, 0.0, 0.0])

    def test_empty_dict_uses_default(self):
        """Empty dict gives all defaults."""
        result = resolve_dict_weights({}, [1, 2])
        np.testing.assert_array_equal(result, [1.0, 1.0])

    def test_datetime_keys(self):
        """Dict with datetime keys works."""
        d1, d2 = datetime(2024, 1, 1), datetime(2024, 1, 2)
        result = resolve_dict_weights({d1: 2.0, d2: 3.0}, [d1, d2])
        np.testing.assert_array_equal(result, [2.0, 3.0])


class TestCombineWeightVectors:
    """Tests for combine_weight_vectors."""

    def test_single_array(self):
        """Single non-None array returned normalized."""
        result = combine_weight_vectors(np.array([2.0, 4.0]), n=2)
        assert result is not None
        assert result.sum() == pytest.approx(2.0)

    def test_two_arrays_multiplied(self):
        """Two arrays are multiplied element-wise then normalized."""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 1.0])
        result = combine_weight_vectors(a, b, n=2)
        assert result is not None
        # products: [3, 2], normalize to sum=2: [6/5, 4/5]
        np.testing.assert_allclose(result, [3 * 2 / 5, 2 * 2 / 5])

    def test_all_none_returns_none(self):
        """All None inputs return None."""
        assert combine_weight_vectors(None, None, n=5) is None

    def test_none_mixed_with_array(self):
        """None entries are ignored; only non-None arrays used."""
        a = np.array([1.0, 3.0])
        result = combine_weight_vectors(None, a, None, n=2)
        assert result is not None
        assert result.sum() == pytest.approx(2.0)

    def test_zero_product_raises(self):
        """Weights producing all-zero product should raise ValueError."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="zero"):
            combine_weight_vectors(a, b, n=2)


class TestResolveWeightToArray:
    """Tests for resolve_weight_to_array."""

    @pytest.fixture
    def time_series(self):
        """Return a time series for testing."""
        return pl.Series("time", [datetime(2024, 1, i) for i in range(1, 6)])

    def test_callable_1_param(self, time_series):
        """1-parameter callable is resolved correctly."""

        def wfn(t: pl.Series) -> pl.Series:
            return pl.Series("w", [1.0] * len(t))

        result = resolve_weight_to_array(wfn, time_series, "time")
        assert len(result) == 5
        np.testing.assert_array_equal(result, [1.0] * 5)

    def test_callable_2_param(self, time_series):
        """2-parameter callable passes group_name correctly."""

        def wfn(t: pl.Series, g: str) -> pl.Series:
            val = 2.0 if g == "groupA" else 1.0
            return pl.Series("w", [val] * len(t))

        result = resolve_weight_to_array(wfn, time_series, "time", group_name="groupA")
        np.testing.assert_array_equal(result, [2.0] * 5)

    def test_dataframe(self, time_series):
        """DataFrame with 'time' and 'weight' columns resolves correctly."""
        tw_df = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 6)],
            "weight": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = resolve_weight_to_array(tw_df, time_series, "time")
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_dict(self, time_series):
        """Dict with datetime keys resolves correctly."""
        d = {datetime(2024, 1, i): float(i) for i in range(1, 6)}
        result = resolve_weight_to_array(d, time_series, "time")
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_dict_wildcard(self, time_series):
        """Dict with '*' wildcard applies default to unmatched keys."""
        d = {datetime(2024, 1, 1): 10.0, "*": 0.5}
        result = resolve_weight_to_array(d, time_series, "time")
        assert result[0] == pytest.approx(10.0)
        assert all(w == pytest.approx(0.5) for w in result[1:])

    def test_invalid_type_raises(self, time_series):
        """Invalid weight type raises ValueError."""
        with pytest.raises(ValueError, match="callable.*DataFrame.*dict.*None"):
            resolve_weight_to_array("bad", time_series, "time")

    def test_callable_wrong_length_raises(self, time_series):
        """Callable returning wrong number of weights raises ValueError."""

        def wfn(t: pl.Series) -> pl.Series:
            return pl.Series("w", [1.0, 2.0])

        with pytest.raises(ValueError, match="weights.*expected"):
            resolve_weight_to_array(wfn, time_series, "time")

    def test_dataframe_missing_weight_col_raises(self, time_series):
        """DataFrame without 'weight' column raises ValueError."""
        bad_df = pl.DataFrame({
            "time": [datetime(2024, 1, 1)],
            "bad_col": [1.0],
        })
        with pytest.raises(ValueError, match="weight.*column"):
            resolve_weight_to_array(bad_df, time_series, "time")

    def test_dataframe_unmatched_keys_raises(self, time_series):
        """DataFrame with mismatched keys raises ValueError for NaN."""
        bad_df = pl.DataFrame({
            "time": [datetime(2025, 1, 1)],
            "weight": [1.0],
        })
        with pytest.raises(ValueError, match="no values for"):
            resolve_weight_to_array(bad_df, time_series, "time")
