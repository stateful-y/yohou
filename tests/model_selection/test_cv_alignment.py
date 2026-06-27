"""Tests for test_size / stride alignment warnings and check_cv_alignment."""

import warnings

import pytest

from yohou.model_selection import (
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
    check_cv_alignment,
)


class TestStrideWarning:
    """SlidingWindowSplitter warns when test_size % stride != 0."""

    def test_no_warning_when_aligned(self, y_X_factory):
        """No warning when test_size is a multiple of stride."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        splitter = SlidingWindowSplitter(n_splits=3, test_size=12, stride=4)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            list(splitter.split(y))

    def test_warning_when_misaligned(self, y_X_factory):
        """Warning emitted when test_size is not a multiple of stride."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        splitter = SlidingWindowSplitter(n_splits=3, test_size=10, stride=4)

        with pytest.warns(UserWarning, match="test_size=10 is not a multiple of stride=4"):
            list(splitter.split(y))

    def test_no_warning_default_stride(self, y_X_factory):
        """No warning when stride defaults to test_size (always aligned)."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        splitter = SlidingWindowSplitter(n_splits=3, test_size=10)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            list(splitter.split(y))


class TestCheckCvAlignment:
    """Tests for check_cv_alignment utility."""

    def test_expanding_aligned(self):
        """ExpandingWindowSplitter with test_size=fh is balanced."""
        cv = ExpandingWindowSplitter(n_splits=3, test_size=5)
        info = check_cv_alignment(cv, forecasting_horizon=5)
        assert info["is_balanced"] is True
        # test_size=stride=5 => ceil(5/5)=1 observe step => initial + 1 = 2 vintages.
        assert info["n_vintages"] == 2

    def test_expanding_default_test_size(self):
        """ExpandingWindowSplitter with test_size=None uses fh."""
        cv = ExpandingWindowSplitter(n_splits=3)
        info = check_cv_alignment(cv, forecasting_horizon=5)
        assert info["is_balanced"] is True

    def test_sliding_aligned(self):
        """SlidingWindowSplitter with test_size divisible by stride."""
        cv = SlidingWindowSplitter(n_splits=3, test_size=12, stride=4)
        info = check_cv_alignment(cv, forecasting_horizon=4)
        assert info["is_balanced"] is True
        assert info["n_vintages"] == 1 + 3  # initial + 12/4

    def test_sliding_misaligned(self):
        """SlidingWindowSplitter with test_size not divisible by stride."""
        cv = SlidingWindowSplitter(n_splits=3, test_size=10, stride=4)
        info = check_cv_alignment(cv, forecasting_horizon=4)
        assert info["is_balanced"] is False

    def test_step_counts_structure(self):
        """step_counts maps each step to number of vintages and balances per step."""
        cv = SlidingWindowSplitter(n_splits=3, test_size=9, stride=3)
        info = check_cv_alignment(cv, forecasting_horizon=3)
        # test_size=9, stride=3 => ceil(9/3)=3 observe steps => initial + 3 = 4 vintages.
        assert info["n_vintages"] == 4
        assert set(info["step_counts"].keys()) == {1, 2, 3}
        # Every step is forecast by every vintage when aligned.
        assert all(count == info["n_vintages"] for count in info["step_counts"].values())
        # Each vintage forecasts exactly forecasting_horizon steps.
        assert info["steps_per_vintage"] == [3] * info["n_vintages"]

    def test_unknown_splitter_returns_none(self):
        """Unknown splitter type returns None for all fields."""
        from yohou.model_selection import BaseSplitter

        class CustomSplitter(BaseSplitter):
            def split(self, y, X=None):
                yield from ()

            def _iter_test_indices(self, y, X=None):
                yield from ()

            def get_n_splits(self, y=None, X=None):
                return 0

        cv = CustomSplitter()
        info = check_cv_alignment(cv, forecasting_horizon=3)
        assert info["is_balanced"] is None
        assert info["n_vintages"] is None
