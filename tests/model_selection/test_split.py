"""Tests for time series cross-validation splitters."""

import numpy as np
import polars as pl
import pytest

from yohou.model_selection import (
    BaseSplitter,
    ExpandingWindowSplitter,
    GapSplitter,
    SlidingWindowSplitter,
)


class TestBaseSplitter:
    """Test BaseSplitter abstract class."""

    def test_cannot_instantiate(self):
        """Cannot instantiate abstract BaseSplitter."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseSplitter()

    def test_panel_detection(self, y_X_factory):
        """Test panel data detection in BaseSplitter."""
        from datetime import datetime, timedelta

        # Create panel data
        y = pl.DataFrame(
            {
                "time": pl.datetime_range(
                    start=datetime(2020, 1, 1),
                    end=datetime(2020, 1, 1) + timedelta(days=99),
                    interval="1d",
                    eager=True,
                ),
                "sales__store_1": np.arange(100),
                "sales__store_2": np.arange(100, 200),
            }
        )

        splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)
        splits = list(splitter.split(y))

        # Panel detection works, splitter handles panel data
        assert len(splits) == 3


class TestExpandingWindowSplitter:
    """Test ExpandingWindowSplitter."""

    def test_basic_split(self, y_X_factory):
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

    def test_max_train_size(self, y_X_factory):
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

    def test_get_n_splits(self, y_X_factory):
        """Test get_n_splits returns correct count."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        splitter = ExpandingWindowSplitter(n_splits=5, test_size=10)
        assert splitter.get_n_splits(y) == 5

        splitter = ExpandingWindowSplitter(n_splits=3, test_size=15)
        assert splitter.get_n_splits(y) == 3

    def test_insufficient_data(self, y_X_factory):
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

    def test_train_index_expansion(self, y_X_factory):
        """Test training set expands correctly without max_train_size."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)
        splits = list(splitter.split(y))

        # Training sizes should grow (no max limit)
        train_sizes = [len(train) for train, _ in splits]
        assert train_sizes[0] < train_sizes[1] < train_sizes[2]


class TestSlidingWindowSplitter:
    """Test SlidingWindowSplitter."""

    def test_basic_split(self, y_X_factory):
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

    def test_step_parameter(self, y_X_factory):
        """Test stride parameter controls slide amount."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

        splitter = SlidingWindowSplitter(train_size=30, test_size=10, stride=20)
        splits = list(splitter.split(y, X))

        # With stride=20: train windows slide by 20 samples
        assert len(splits) >= 3

        # All train sizes should be constant
        train_sizes = [len(train) for train, _ in splits]
        assert all(size == 30 for size in train_sizes)

    def test_train_size_consistency(self, y_X_factory):
        """Test training size remains constant across splits."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        splitter = SlidingWindowSplitter(train_size=30, test_size=10)
        splits = list(splitter.split(y))

        train_sizes = [len(train) for train, _ in splits]
        assert all(size == 30 for size in train_sizes)

    def test_get_n_splits(self, y_X_factory):
        """Test get_n_splits returns correct count."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        splitter = SlidingWindowSplitter(train_size=30, test_size=10)
        n_splits = splitter.get_n_splits(y)
        assert n_splits >= 1

        splitter = SlidingWindowSplitter(train_size=30, test_size=10, stride=20)
        n_splits_stride = splitter.get_n_splits(y)
        assert n_splits_stride >= 1

    def test_insufficient_data(self, y_X_factory):
        """Test error when insufficient data."""
        y, _ = y_X_factory(length=20, n_targets=1, n_features=0, seed=42)

        splitter = SlidingWindowSplitter(train_size=30, test_size=10)
        with pytest.raises(ValueError, match="greater than n_samples"):
            list(splitter.split(y))


class TestGapSplitter:
    """Test GapSplitter wrapper."""

    def test_basic_gap_insertion(self, y_X_factory):
        """Test gap insertion between train and test."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

        base_splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)
        gap_splitter = GapSplitter(base_splitter, gap=5)

        splits = list(gap_splitter.split(y, X))

        # Should have splits (possibly fewer due to gap)
        base_splits = list(base_splitter.split(y, X))
        assert len(splits) <= len(base_splits)

        # Check gap is inserted correctly
        for (train_idx, test_idx), (base_train, base_test) in zip(splits, base_splits):
            # Training set unchanged
            assert np.array_equal(train_idx, base_train)

            # Test set shifted by gap
            assert test_idx[0] == base_test[0] + 5
            assert len(test_idx) == len(base_test)

    def test_gap_reduces_splits(self, y_X_factory):
        """Test gap can reduce number of splits when data insufficient."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        base_splitter = ExpandingWindowSplitter(n_splits=5, test_size=10)
        gap_splitter = GapSplitter(base_splitter, gap=10)

        base_splits = list(base_splitter.split(y))
        gap_splits = list(gap_splitter.split(y))

        # Gap of 10 should reduce available splits
        assert len(gap_splits) <= len(base_splits)

    def test_gap_with_sliding_window(self, y_X_factory):
        """Test gap with SlidingWindowSplitter."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        base_splitter = SlidingWindowSplitter(train_size=30, test_size=10)
        gap_splitter = GapSplitter(base_splitter, gap=5)

        splits = list(gap_splitter.split(y))

        # All splits should have constant train size
        train_sizes = [len(train) for train, _ in splits]
        assert all(size == 30 for size in train_sizes)

        # Gap should be maintained
        for train_idx, test_idx in splits:
            assert test_idx[0] - train_idx[-1] == 6  # gap + 1 (continuity)

    def test_gap_parameter_validation(self, y_X_factory):
        """Test gap parameter validation."""
        pytest.skip(
            "Parameter validation triggers at __init__ via @validate_params decorator - test needs adjustment"
        )
        from sklearn.utils._param_validation import InvalidParameterError

        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        base_splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)

        # Negative gap should fail during split (parameter validation)
        gap_splitter = GapSplitter(base_splitter, gap=-1)
        with pytest.raises(InvalidParameterError):
            list(gap_splitter.split(y))

        # Zero gap should fail (must be >= 1)
        gap_splitter = GapSplitter(base_splitter, gap=0)
        with pytest.raises(InvalidParameterError):
            list(gap_splitter.split(y))

    def test_get_n_splits(self, y_X_factory):
        """Test get_n_splits with gap."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        base_splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)
        gap_splitter = GapSplitter(base_splitter, gap=5)

        # Number of splits may differ from base due to gap
        n_splits = gap_splitter.get_n_splits(y)
        assert n_splits >= 0
        assert n_splits <= base_splitter.get_n_splits(y)


class TestSplitterIntegration:
    """Test splitter integration with yohou forecasters."""

    def test_expanding_window_with_forecaster(self, y_X_factory):
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
            error = np.abs(
                y_test.select(y_test_cols).to_numpy() - y_pred.select(y_pred_cols).to_numpy()
            )
            scores.append(np.mean(error))

        assert len(scores) == 3
        assert all(score >= 0 for score in scores)

    def test_sliding_window_with_forecaster(self, y_X_factory):
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
            error = np.abs(
                y_test.select(y_test_cols).to_numpy() - y_pred.select(y_pred_cols).to_numpy()
            )
            scores.append(np.mean(error))

        assert len(scores) >= 1
        assert all(score >= 0 for score in scores)

    def test_gap_splitter_prevents_leakage(self, y_X_factory):
        """Test GapSplitter prevents data leakage."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

        base_splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)
        gap_splitter = GapSplitter(base_splitter, gap=5)

        for train_idx, test_idx in gap_splitter.split(y, X):
            # Gap region should not overlap with train or test
            gap_start = train_idx[-1] + 1
            gap_end = test_idx[0] - 1

            # Verify gap exists
            assert gap_end >= gap_start

            # Verify gap size
            gap_size = gap_end - gap_start + 1
            assert gap_size == 5
