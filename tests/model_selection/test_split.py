"""Tests for time series cross-validation splitters."""

import numpy as np
import polars as pl
import pytest

from yohou.model_selection import (
    BaseSplitter,
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
)


def test_basesplitter_cannot_instantiate():
    """Cannot instantiate abstract BaseSplitter."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseSplitter()


def test_basesplitter_panel_detection(y_X_factory):
    """Test panel data detection in BaseSplitter."""
    from datetime import datetime, timedelta

    # Create panel data
    y = pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=99),
            interval="1d",
            eager=True,
        ),
        "sales__store_1": np.arange(100),
        "sales__store_2": np.arange(100, 200),
    })

    splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)
    splits = list(splitter.split(y))

    # Panel detection works, splitter handles panel data
    assert len(splits) == 3


def test_expanding_window_basic_split(y_X_factory):
    """Test basic expanding window split."""
    y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

    splitter = ExpandingWindowSplitter(n_splits=5, test_size=10)
    splits = list(splitter.split(y, X))

    # Should have 5 splits with test_size=10
    assert len(splits) == 5

    # Check first split: train on [0:50], test on [50:60]
    train_idx, test_idx = splits[0]
    assert len(test_idx) == 10
    assert test_idx[0] == 50
    assert train_idx[-1] + 1 == test_idx[0]  # Continuity

    # Check last split: train on [0:90], test on [90:100]
    train_idx, test_idx = splits[-1]
    assert len(test_idx) == 10
    assert test_idx[0] == 90
    assert len(train_idx) == 90


def test_expanding_window_max_train_size(y_X_factory):
    """Test max_train_size parameter limits training set."""
    y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

    splitter = ExpandingWindowSplitter(n_splits=3, test_size=10, max_train_size=30)
    splits = list(splitter.split(y, X))

    # Training sets should be capped at 30
    for train_idx, test_idx in splits:
        assert len(train_idx) <= 30

    # Check that the cap is actually applied (not all splits should be < 30)
    train_sizes = [len(train) for train, _ in splits]
    assert max(train_sizes) == 30  # At least one split hits the cap


def test_expanding_window_get_n_splits(y_X_factory):
    """Test get_n_splits returns correct count."""
    y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

    splitter = ExpandingWindowSplitter(n_splits=5, test_size=10)
    assert splitter.get_n_splits(y) == 5

    splitter = ExpandingWindowSplitter(n_splits=3, test_size=15)
    assert splitter.get_n_splits(y) == 3


def test_expanding_window_insufficient_data(y_X_factory):
    """Test behavior when splits don't fit - skips invalid splits."""
    y, _ = y_X_factory(length=20, n_targets=1, n_features=0, seed=42)

    splitter = ExpandingWindowSplitter(n_splits=5, test_size=10)
    splits = list(splitter.split(y))

    # With only 20 samples and requesting 5 splits of test_size=10,
    # splits with negative test indices are skipped, leaving 2 valid splits
    assert len(splits) == 2

    # First split: test indices [0:10], train indices [] (empty)
    train_idx, test_idx = splits[0]
    assert len(test_idx) == 10
    assert test_idx[0] == 0
    assert len(train_idx) == 0  # No training data before first test

    # Second split: test indices [10:20], train indices [0:10]
    train_idx, test_idx = splits[1]
    assert len(test_idx) == 10
    assert test_idx[0] == 10
    assert len(train_idx) == 10


def test_expanding_window_train_index_expansion(y_X_factory):
    """Test training set expands correctly without max_train_size."""
    y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

    splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)
    splits = list(splitter.split(y))

    # Training sizes should grow (no max limit)
    train_sizes = [len(train) for train, _ in splits]
    assert train_sizes[0] < train_sizes[1] < train_sizes[2]


def test_expanding_window_gap_insertion(y_X_factory):
    """Test gap parameter inserts gap between train and test."""
    y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

    splitter = ExpandingWindowSplitter(n_splits=3, test_size=10, gap=5)
    splits = list(splitter.split(y, X))

    # Check gap is inserted correctly
    for train_idx, test_idx in splits:
        # Gap of 5 means test starts 5 indices after where train ends
        gap_size = test_idx[0] - train_idx[-1] - 1
        assert gap_size == 5


def test_expanding_window_gap_preserves_split_count(y_X_factory):
    """Test gap doesn't change number of splits, only index positions."""
    y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

    splitter = ExpandingWindowSplitter(n_splits=5, test_size=10, gap=10)
    base_splitter = ExpandingWindowSplitter(n_splits=5, test_size=10)

    splits = list(splitter.split(y))
    base_splits = list(base_splitter.split(y))

    # Gap should not change the number of splits
    assert len(splits) == len(base_splits) == 5

    # But indices should be different due to gap
    for (train_gap, test_gap), (train_base, test_base) in zip(splits, base_splits, strict=False):
        # Test indices should remain the same (anchored to end)
        assert test_gap[0] == test_base[0]
        # Train ends earlier (gap samples before test)
        assert train_gap[-1] == train_base[-1] - 10


def test_expanding_window_gap_zero_equivalent_to_none(y_X_factory):
    """Test gap=0 produces same result as gap=None."""
    y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

    splitter_gap_zero = ExpandingWindowSplitter(n_splits=3, test_size=10, gap=0)
    splitter_gap_none = ExpandingWindowSplitter(n_splits=3, test_size=10, gap=None)
    splitter_no_gap = ExpandingWindowSplitter(n_splits=3, test_size=10)

    splits_zero = list(splitter_gap_zero.split(y))
    splits_none = list(splitter_gap_none.split(y))
    splits_no_gap = list(splitter_no_gap.split(y))

    # All three should produce identical results
    for (t1, te1), (t2, te2), (t3, te3) in zip(splits_zero, splits_none, splits_no_gap, strict=False):
        assert np.array_equal(t1, t2)
        assert np.array_equal(t1, t3)
        assert np.array_equal(te1, te2)
        assert np.array_equal(te1, te3)


def test_expanding_window_gap_get_n_splits_no_data_required(y_X_factory):
    """Test get_n_splits works without y even when gap > 0."""
    y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

    splitter = ExpandingWindowSplitter(n_splits=3, test_size=10, gap=5)

    # Should work even without y since gap doesn't affect count
    n_splits_no_y = splitter.get_n_splits(y=None)
    n_splits_with_y = splitter.get_n_splits(y)

    assert n_splits_no_y == n_splits_with_y == 3


def test_sliding_window_basic_split(y_X_factory):
    """Test basic sliding window split."""
    y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

    splitter = SlidingWindowSplitter(train_size=30, test_size=10)
    splits = list(splitter.split(y, X))

    # Should have 7 splits with constant train_size=30
    assert len(splits) == 7

    # Check all splits have constant train size
    for train_idx, test_idx in splits:
        assert len(train_idx) == 30
        assert len(test_idx) == 10
        assert train_idx[-1] + 1 == test_idx[0]  # Continuity


def test_sliding_window_step_parameter(y_X_factory):
    """Test stride parameter controls slide amount."""
    y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

    splitter = SlidingWindowSplitter(train_size=30, test_size=10, stride=20)
    splits = list(splitter.split(y, X))

    # With stride=20: train windows slide by 20 samples
    assert len(splits) >= 3

    # All train sizes should be constant
    train_sizes = [len(train) for train, _ in splits]
    assert all(size == 30 for size in train_sizes)


def test_sliding_window_train_size_consistency(y_X_factory):
    """Test training size remains constant across splits."""
    y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

    splitter = SlidingWindowSplitter(train_size=30, test_size=10)
    splits = list(splitter.split(y))

    train_sizes = [len(train) for train, _ in splits]
    assert all(size == 30 for size in train_sizes)


def test_sliding_window_get_n_splits(y_X_factory):
    """Test get_n_splits returns correct count."""
    y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

    splitter = SlidingWindowSplitter(train_size=30, test_size=10)
    n_splits = splitter.get_n_splits(y)
    assert n_splits >= 1

    splitter = SlidingWindowSplitter(train_size=30, test_size=10, stride=20)
    n_splits_stride = splitter.get_n_splits(y)
    assert n_splits_stride >= 1


def test_sliding_window_insufficient_data(y_X_factory):
    """Test error when insufficient data."""
    y, _ = y_X_factory(length=20, n_targets=1, n_features=0, seed=42)

    splitter = SlidingWindowSplitter(train_size=30, test_size=10)
    with pytest.raises(ValueError, match="greater than n_samples"):
        list(splitter.split(y))


def test_sliding_window_gap_insertion(y_X_factory):
    """Test gap parameter inserts gap between train and test."""
    y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

    splitter = SlidingWindowSplitter(train_size=30, test_size=10, gap=5)
    splits = list(splitter.split(y, X))

    # All splits should have constant train size
    train_sizes = [len(train) for train, _ in splits]
    assert all(size == 30 for size in train_sizes)

    # Gap should be maintained
    for train_idx, test_idx in splits:
        assert test_idx[0] - train_idx[-1] == 6  # gap + 1 (continuity)


def test_sliding_window_gap_with_stride(y_X_factory):
    """Test gap and stride work independently."""
    y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

    splitter = SlidingWindowSplitter(train_size=30, test_size=10, stride=20, gap=5)
    splits = list(splitter.split(y))

    # Should have valid splits
    assert len(splits) >= 2

    # All train sizes constant
    train_sizes = [len(train) for train, _ in splits]
    assert all(size == 30 for size in train_sizes)

    # Gap maintained across all splits
    for train_idx, test_idx in splits:
        assert test_idx[0] - train_idx[-1] == 6  # gap + 1


def test_sliding_window_gap_zero_equivalent_to_none(y_X_factory):
    """Test gap=0 produces same result as gap=None."""
    y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

    splitter_gap_zero = SlidingWindowSplitter(train_size=30, test_size=10, gap=0)
    splitter_gap_none = SlidingWindowSplitter(train_size=30, test_size=10, gap=None)
    splitter_no_gap = SlidingWindowSplitter(train_size=30, test_size=10)

    splits_zero = list(splitter_gap_zero.split(y))
    splits_none = list(splitter_gap_none.split(y))
    splits_no_gap = list(splitter_no_gap.split(y))

    # All three should produce identical results
    for (t1, te1), (t2, te2), (t3, te3) in zip(splits_zero, splits_none, splits_no_gap, strict=False):
        assert np.array_equal(t1, t2)
        assert np.array_equal(t1, t3)
        assert np.array_equal(te1, te2)
        assert np.array_equal(te1, te3)


def test_sliding_window_gap_insufficient_data(y_X_factory):
    """Test error when gap + train + test exceeds data."""
    y, _ = y_X_factory(length=40, n_targets=1, n_features=0, seed=42)

    splitter = SlidingWindowSplitter(train_size=30, test_size=10, gap=5)
    with pytest.raises(ValueError, match="train_size.*gap.*test_size.*greater than n_samples"):
        list(splitter.split(y))


def test_splitter_integration_expanding_window_with_forecaster(y_X_factory):
    """Test ExpandingWindowSplitter with forecaster."""
    from yohou.point_forecaster.naive import SeasonalNaive

    y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

    splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)
    forecaster = SeasonalNaive(seasonality=1)

    # Simulate cross-validation
    scores = []
    for train_idx, test_idx in splitter.split(y, X):
        y_train, X_train = y[train_idx], X[train_idx]
        y_test, X_test = y[test_idx], X[test_idx]

        forecaster.fit(y_train, X_train, forecasting_horizon=len(test_idx))
        y_pred = forecaster.predict(forecasting_horizon=len(test_idx), X=X_test)

        # Simple MAE
        y_test_cols = [c for c in y_test.columns if c != "time"]
        y_pred_cols = [c for c in y_pred.columns if c not in ("time", "observed_time")]
        error = np.abs(y_test.select(y_test_cols).to_numpy() - y_pred.select(y_pred_cols).to_numpy())
        scores.append(np.mean(error))

    assert len(scores) == 3
    assert all(score >= 0 for score in scores)


def test_splitter_integration_sliding_window_with_forecaster(y_X_factory):
    """Test SlidingWindowSplitter with forecaster."""
    from yohou.point_forecaster.naive import SeasonalNaive

    y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

    splitter = SlidingWindowSplitter(train_size=30, test_size=10)
    forecaster = SeasonalNaive(seasonality=1)

    # Simulate cross-validation
    scores = []
    for train_idx, test_idx in splitter.split(y, X):
        y_train, X_train = y[train_idx], X[train_idx]
        y_test, X_test = y[test_idx], X[test_idx]

        forecaster.fit(y_train, X_train, forecasting_horizon=len(test_idx))
        y_pred = forecaster.predict(forecasting_horizon=len(test_idx), X=X_test)

        # Simple MAE
        y_test_cols = [c for c in y_test.columns if c != "time"]
        y_pred_cols = [c for c in y_pred.columns if c not in ("time", "observed_time")]
        error = np.abs(y_test.select(y_test_cols).to_numpy() - y_pred.select(y_pred_cols).to_numpy())
        scores.append(np.mean(error))

    assert len(scores) >= 1
    assert all(score >= 0 for score in scores)


def test_splitter_integration_gap_prevents_leakage(y_X_factory):
    """Test gap parameter prevents data leakage."""
    y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

    splitter = ExpandingWindowSplitter(n_splits=3, test_size=10, gap=5)

    for train_idx, test_idx in splitter.split(y, X):
        # Gap region is between train end and test start
        # Gap size should be 5 (the number of excluded indices)
        gap_size = test_idx[0] - train_idx[-1] - 1
        assert gap_size == 5
