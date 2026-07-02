"""Tests for time series cross-validation splitters."""

import numpy as np
import polars as pl
import pytest

from yohou.model_selection import (
    BaseSplitter,
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
)


class TestBaseSplitter:
    """Tests for BaseSplitter abstract class."""

    def test_basesplitter_cannot_instantiate(self):
        """Cannot instantiate abstract BaseSplitter."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseSplitter()

    def test_basesplitter_panel_detection(self, y_X_factory):
        """Test panel data detection in BaseSplitter."""
        from datetime import datetime, timedelta

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

        assert len(splits) == 3


class TestExpandingWindowBasic:
    """Tests for basic ExpandingWindowSplitter functionality."""

    def test_expanding_window_basic_split(self, y_X_factory):
        """Test basic expanding window split."""
        y, X_actual = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

        splitter = ExpandingWindowSplitter(n_splits=5, test_size=10)
        splits = list(splitter.split(y, X_actual))

        assert len(splits) == 5

        # Check first split: train on [0:50], test on [50:60].
        # (Temporal ordering train[-1] < test[0] is owned by the systematic
        # check_splitter_produces_valid_indices; assert only the concrete
        # boundary values here.)
        train_idx, test_idx = splits[0]
        assert len(test_idx) == 10
        assert test_idx[0] == 50

        # Check last split: train on [0:90], test on [90:100]
        train_idx, test_idx = splits[-1]
        assert len(test_idx) == 10
        assert test_idx[0] == 90
        assert len(train_idx) == 90

    def test_expanding_window_max_train_size(self, y_X_factory):
        """Test max_train_size parameter limits training set."""
        y, X_actual = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

        splitter = ExpandingWindowSplitter(n_splits=3, test_size=10, max_train_size=30)
        splits = list(splitter.split(y, X_actual))

        for train_idx, _test_idx in splits:
            assert len(train_idx) <= 30

        train_sizes = [len(train) for train, _ in splits]
        assert max(train_sizes) == 30

    def test_expanding_window_all_test_no_train_raises(self, y_X_factory):
        """The boundary n_samples == n_splits * test_size must not yield an empty train.

        When n_samples == n_splits * test_size, the first fold's training set
        would be empty, so split() must raise rather than yield a fold whose
        train index starts at 0.
        """
        y, _ = y_X_factory(length=20, n_targets=1, n_features=0, seed=42)

        splitter = ExpandingWindowSplitter(n_splits=2, test_size=10)
        with pytest.raises(ValueError, match="non-overlapping test windows"):
            list(splitter.split(y))

    def test_expanding_window_split_count_matches_get_n_splits(self, y_X_factory):
        """split() yields exactly get_n_splits() folds when the data supports it."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        splitter = ExpandingWindowSplitter(n_splits=5, test_size=10)
        n_yielded = len(list(splitter.split(y)))
        assert n_yielded == splitter.get_n_splits(y) == 5

    def test_expanding_window_train_index_expansion(self, y_X_factory):
        """Test training set expands correctly without max_train_size."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)
        splits = list(splitter.split(y))

        train_sizes = [len(train) for train, _ in splits]
        assert train_sizes[0] < train_sizes[1] < train_sizes[2]


class TestSlidingWindowBasic:
    """Tests for basic SlidingWindowSplitter functionality."""

    def test_sliding_window_basic_split(self, y_X_factory):
        """Test basic sliding window split."""
        y, X_actual = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

        splitter = SlidingWindowSplitter(n_splits=7, test_size=10)
        splits = list(splitter.split(y, X_actual))

        assert len(splits) == 7

        # Temporal ordering train[-1] < test[0] is owned by the systematic
        # check_splitter_produces_valid_indices; assert only window sizes here.
        for train_idx, test_idx in splits:
            assert len(train_idx) == 30  # auto-computed: 100 - 10 - 6*10
            assert len(test_idx) == 10

    def test_sliding_window_step_parameter(self, y_X_factory):
        """Test stride parameter controls slide amount."""
        y, X_actual = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

        splitter = SlidingWindowSplitter(n_splits=4, train_size=30, test_size=10, stride=20)
        splits = list(splitter.split(y, X_actual))

        assert len(splits) == 4

        train_sizes = [len(train) for train, _ in splits]
        assert all(size == 30 for size in train_sizes)

    def test_sliding_window_train_size_consistency(self, y_X_factory):
        """Test training size remains constant across splits."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        splitter = SlidingWindowSplitter(n_splits=7, test_size=10)
        splits = list(splitter.split(y))

        train_sizes = [len(train) for train, _ in splits]
        assert all(size == train_sizes[0] for size in train_sizes)

    def test_sliding_window_explicit_train_size_exceeds_data(self, y_X_factory):
        """An explicit train_size + test_size > n_samples is rejected by split()."""
        y, _ = y_X_factory(length=20, n_targets=1, n_features=0, seed=42)

        # split() resolves train_size first, so the error originates in
        # _resolve_train_size, not the defensive guard in _iter_test_indices.
        splitter = SlidingWindowSplitter(n_splits=3, train_size=30, test_size=10)
        with pytest.raises(ValueError, match="greater than n_samples"):
            list(splitter.split(y))

    def test_sliding_window_iter_test_indices_defensive_guard(self, y_X_factory):
        """_iter_test_indices guards against train_size_ + test_size > n_samples.

        split() validates via _resolve_train_size first, so this guard is only
        reachable by calling _iter_test_indices directly with an oversized
        train_size_ patched onto the splitter (matching the source comment).
        """
        y, _ = y_X_factory(length=20, n_targets=1, n_features=0, seed=42)

        splitter = SlidingWindowSplitter(n_splits=3, test_size=10)
        splitter.train_size_ = 30  # bypass _resolve_train_size
        with pytest.raises(ValueError, match="greater than n_samples"):
            list(splitter._iter_test_indices(y))

    def test_sliding_window_train_size_auto_computed(self, y_X_factory):
        """Test train_size_ is computed when train_size is None."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        splitter = SlidingWindowSplitter(n_splits=5, test_size=10)
        splits = list(splitter.split(y))

        # train_size_ should be set: 100 - 0 - 10 - 4*10 = 50
        assert splitter.train_size_ == 50
        assert len(splits) == 5
        assert all(len(train) == 50 for train, _ in splits)

    def test_sliding_window_conflict_raises(self, y_X_factory):
        """Test ValueError when n_splits exceeds what data supports."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        # n_splits=10, train_size=30, test_size=10, stride=10, gap=0
        # max_splits: (100-30-0-10)//10 + 1 = 7, so 10 > 7 → error
        splitter = SlidingWindowSplitter(n_splits=10, train_size=30, test_size=10)
        with pytest.raises(ValueError, match="Inconsistent parameters"):
            list(splitter.split(y))

    def test_sliding_window_n_splits_within_max_ok(self, y_X_factory):
        """Test n_splits < max_splits works fine (no exact match required)."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        # max_splits: (100-30-0-10)//10 + 1 = 7, so 5 <= 7 → ok
        splitter = SlidingWindowSplitter(n_splits=5, train_size=30, test_size=10)
        splits = list(splitter.split(y))
        assert len(splits) == 5

    def test_sliding_window_auto_train_size_too_small(self, y_X_factory):
        """Test error when auto-computed train_size would be < 1."""
        y, _ = y_X_factory(length=20, n_targets=1, n_features=0, seed=42)

        splitter = SlidingWindowSplitter(n_splits=10, test_size=10)
        with pytest.raises(ValueError, match="computed train_size"):
            list(splitter.split(y))


class TestSplitterIntegration:
    """Integration tests for splitters with forecasters."""

    def test_expanding_window_with_forecaster(self, y_X_factory):
        """Test ExpandingWindowSplitter with forecaster."""
        from yohou.point.naive import SeasonalNaive

        y, X_actual = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

        splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)
        forecaster = SeasonalNaive(seasonality=1)

        scores = []
        for train_idx, test_idx in splitter.split(y, X_actual):
            y_train, X_actual_train = y[train_idx], X_actual[train_idx]
            y_test, _X_actual_test = y[test_idx], X_actual[test_idx]

            forecaster.fit(y_train, X_actual_train, forecasting_horizon=len(test_idx))
            y_pred = forecaster.predict(forecasting_horizon=len(test_idx))

            y_test_cols = [c for c in y_test.columns if c != "time"]
            y_pred_cols = [c for c in y_pred.columns if c not in ("time", "vintage_time")]
            error = np.abs(y_test.select(y_test_cols).to_numpy() - y_pred.select(y_pred_cols).to_numpy())
            scores.append(np.mean(error))

        assert len(scores) == 3
        assert all(score >= 0 for score in scores)

    def test_sliding_window_with_forecaster(self, y_X_factory):
        """Test SlidingWindowSplitter with forecaster."""
        from yohou.point.naive import SeasonalNaive

        y, X_actual = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

        splitter = SlidingWindowSplitter(n_splits=7, test_size=10)
        forecaster = SeasonalNaive(seasonality=1)

        scores = []
        for train_idx, test_idx in splitter.split(y, X_actual):
            y_train, X_actual_train = y[train_idx], X_actual[train_idx]
            y_test, _X_test = y[test_idx], X_actual[test_idx]

            forecaster.fit(y_train, X_actual_train, forecasting_horizon=len(test_idx))
            y_pred = forecaster.predict(forecasting_horizon=len(test_idx))

            y_test_cols = [c for c in y_test.columns if c != "time"]
            y_pred_cols = [c for c in y_pred.columns if c not in ("time", "vintage_time")]
            error = np.abs(y_test.select(y_test_cols).to_numpy() - y_pred.select(y_pred_cols).to_numpy())
            scores.append(np.mean(error))

        assert len(scores) == 7
        assert all(score >= 0 for score in scores)
