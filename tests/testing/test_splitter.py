"""Tests for splitter check functions.

Meta-tests that validate the check functions themselves work correctly.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from conftest import run_checks
from yohou.model_selection import ExpandingWindowSplitter, SlidingWindowSplitter
from yohou.testing import (
    _yield_yohou_splitter_checks,
    check_splitter_n_splits_consistency,
    check_splitter_non_overlapping_tests,
    check_splitter_parameter_constraints,
    check_splitter_produces_valid_indices,
    check_splitter_tags_accessible_before_fit,
    check_splitter_tags_match_capabilities,
    check_splitter_tags_static_after_fit,
)


@pytest.fixture
def y_data():
    """Generate test time series data."""
    return pl.DataFrame({
        "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)],
        "value": range(100),
    })


@pytest.fixture
def splitters():
    """Generate test splitter instances."""
    return [
        ExpandingWindowSplitter(n_splits=3, test_size=10),
        SlidingWindowSplitter(n_splits=3, test_size=10),
        ExpandingWindowSplitter(n_splits=3, test_size=10, gap=2),
    ]


class TestSplitterChecks:
    """Tests for splitter check functions."""

    def test_tags_accessible_before_fit(self):
        """Test that tags are accessible on splitter instances."""
        check_splitter_tags_accessible_before_fit(ExpandingWindowSplitter(n_splits=3, test_size=10))
        check_splitter_tags_accessible_before_fit(SlidingWindowSplitter(n_splits=3, test_size=10))
        check_splitter_tags_accessible_before_fit(ExpandingWindowSplitter(n_splits=3, test_size=10, gap=2))

    def test_tags_static_after_fit(self, splitters, y_data):
        """Test that tags don't change after split()."""
        for splitter in splitters:
            check_splitter_tags_static_after_fit(splitter, y_data)

    def test_tags_match_capabilities(self, splitters, y_data):
        """Test that tags match actual splitter behavior."""
        # Test ExpandingWindowSplitter (no gap)
        check_splitter_tags_match_capabilities(
            splitters[0],
            y_data,
            expected_tags={"splitter_type": "expanding"},
        )

        # Test SlidingWindowSplitter
        check_splitter_tags_match_capabilities(
            splitters[1],
            y_data,
            expected_tags={"splitter_type": "sliding"},
        )

        # Test ExpandingWindowSplitter with gap
        check_splitter_tags_match_capabilities(
            splitters[2],
            y_data,
            expected_tags={"splitter_type": "expanding"},
        )

    def test_produces_valid_indices(self, splitters, y_data):
        """Test that all splitters produce valid indices."""
        for splitter in splitters:
            check_splitter_produces_valid_indices(splitter, y_data)

    def test_n_splits_consistency(self, splitters, y_data):
        """Test that get_n_splits() matches actual split count."""
        for splitter in splitters:
            check_splitter_n_splits_consistency(splitter, y_data)

    def test_non_overlapping_tests(self, splitters, y_data):
        """Test that test sets don't overlap."""
        for splitter in splitters:
            check_splitter_non_overlapping_tests(splitter, y_data)

    @pytest.mark.parametrize(
        "splitter_class,param_name,invalid_values",
        [
            (ExpandingWindowSplitter, "n_splits", [1, 0, -1]),
            (ExpandingWindowSplitter, "test_size", [0, -1]),
            (ExpandingWindowSplitter, "gap", [-1]),
            (SlidingWindowSplitter, "n_splits", [1, 0, -1]),
            (SlidingWindowSplitter, "test_size", [0, -1]),
            (SlidingWindowSplitter, "gap", [-1]),
        ],
    )
    def test_parameter_constraints(self, splitter_class, param_name, invalid_values):
        """Test that parameter constraints are enforced."""
        check_splitter_parameter_constraints(splitter_class, param_name, invalid_values)

    def test_yield_yohou_splitter_checks(self, splitters, y_data):
        """Test that generator produces checks for all splitters."""
        for splitter in splitters:
            checks = list(_yield_yohou_splitter_checks(splitter, y_data))

            # Should yield multiple checks
            assert len(checks) >= 6, f"Expected at least 6 checks, got {len(checks)}"

            # Each check should be a tuple (name, func, kwargs)
            for check_name, check_func, check_kwargs in checks:
                assert isinstance(check_name, str), "Check name should be string"
                assert callable(check_func), "Check function should be callable"
                assert isinstance(check_kwargs, dict), "Check kwargs should be dict"

    def test_systematic_expanding_window_checks(self, y_data):
        """Systematic test using generator for ExpandingWindowSplitter."""
        splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)

        run_checks(
            splitter,
            _yield_yohou_splitter_checks(splitter, y_data),
        )
