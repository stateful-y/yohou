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
    check_splitter_parameter_constraints,
    check_splitter_tags_match_capabilities,
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
    ]


class TestSplitterChecks:
    """Tests for splitter check functions.

    The happy-path tag/index/consistency checks are exercised end-to-end by
    test_systematic_*_checks below (which run the full generator on both
    splitters), so only the checks that take arguments not supplied by the
    generator (expected_tags, hardcoded invalid constraint values) are tested
    in isolation here.
    """

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

    def test_tags_match_capabilities_without_expected_tags(self, y_data):
        """check_splitter_tags_match_capabilities with expected_tags=None skips tag validation."""
        splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)
        check_splitter_tags_match_capabilities(splitter, y_data, expected_tags=None)

    @pytest.mark.parametrize(
        "splitter_class,param_name,invalid_values",
        [
            (ExpandingWindowSplitter, "n_splits", [1, 0, -1]),
            (ExpandingWindowSplitter, "test_size", [0, -1]),
            (SlidingWindowSplitter, "n_splits", [1, 0, -1]),
            (SlidingWindowSplitter, "test_size", [0, -1]),
        ],
    )
    def test_parameter_constraints(self, splitter_class, param_name, invalid_values):
        """Test that parameter constraints are enforced."""
        check_splitter_parameter_constraints(splitter_class, param_name, invalid_values)

    def test_yield_yohou_splitter_checks(self, splitters, y_data):
        """Generator yields well-formed tuples and the exact expected check count.

        ExpandingWindowSplitter exposes n_splits/test_size constraints (9 checks);
        SlidingWindowSplitter additionally exposes train_size (10 checks). Pinning
        the exact count catches a regression that silently drops a yielded check.
        """
        expected_counts = {
            "ExpandingWindowSplitter": 9,
            "SlidingWindowSplitter": 10,
        }
        for splitter in splitters:
            checks = list(_yield_yohou_splitter_checks(splitter, y_data))

            expected = expected_counts[type(splitter).__name__]
            assert len(checks) == expected, f"{type(splitter).__name__}: expected {expected} checks, got {len(checks)}"

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

    def test_systematic_sliding_window_checks(self, y_data):
        """Systematic test using generator for SlidingWindowSplitter."""
        splitter = SlidingWindowSplitter(n_splits=3, test_size=10)

        run_checks(
            splitter,
            _yield_yohou_splitter_checks(splitter, y_data),
        )


class TestSplitterPanelDataChecks:
    """Tests for panel data support splitter checks."""

    @pytest.fixture
    def y_panel(self):
        """Generate panel time series data."""
        return pl.DataFrame({
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)],
            "sales__store_1": list(range(100)),
            "sales__store_2": list(range(100, 200)),
        })

    def test_panel_support_expanding_window(self, y_panel):
        """Expanding window splitter handles panel data correctly."""
        splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)
        from yohou.testing.splitter import check_splitter_panel_data_support

        check_splitter_panel_data_support(splitter, y_panel)

    def test_panel_support_sliding_window(self, y_panel):
        """Sliding window splitter handles panel data correctly."""
        splitter = SlidingWindowSplitter(n_splits=3, test_size=10)
        from yohou.testing.splitter import check_splitter_panel_data_support

        check_splitter_panel_data_support(splitter, y_panel)
