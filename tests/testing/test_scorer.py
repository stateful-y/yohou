"""Tests for scorer check functions.

Meta-tests that validate the check functions themselves work correctly.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from yohou.metrics import MeanAbsoluteError, MeanSquaredError
from yohou.testing import (
    _yield_yohou_scorer_checks,
    check_scorer_aggregation_methods,
    check_scorer_lower_is_better,
    check_scorer_parameter_validation,
    check_scorer_tags_accessible_before_fit,
    check_scorer_tags_match_capabilities,
    check_scorer_tags_static_after_fit,
)


@pytest.fixture
def y_truth():
    """Generate ground truth data."""
    return pl.DataFrame({
        "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
        "value": range(10),
    })


@pytest.fixture
def y_pred():
    """Generate prediction data."""
    return pl.DataFrame({
        "vintage_time": [datetime(2020, 1, 1)] * 10,
        "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)],
        "value": [i + 0.1 for i in range(10)],
    })


@pytest.fixture
def scorers():
    """Generate test scorer instances."""
    return [
        MeanAbsoluteError(),
        MeanSquaredError(),
    ]


class TestScorerChecks:
    """Tests for scorer check functions."""

    def test_tags_accessible_before_fit(self):
        """Test that tags are accessible on scorer instances."""
        check_scorer_tags_accessible_before_fit(MeanAbsoluteError())
        check_scorer_tags_accessible_before_fit(MeanSquaredError())

    def test_tags_static_after_fit(self, scorers, y_truth):
        """Test that tags don't change after fit()."""
        for scorer in scorers:
            check_scorer_tags_static_after_fit(scorer, y_truth)

    def test_tags_match_capabilities(self, scorers, y_truth, y_pred):
        """Test that tags match actual scorer behavior."""
        for scorer in scorers:
            check_scorer_tags_match_capabilities(
                scorer,
                y_truth,
                y_pred,
                expected_tags={"prediction_type": "point", "lower_is_better": True},
            )

    def test_lower_is_better(self, scorers):
        """Test that lower_is_better tag is boolean."""
        for scorer in scorers:
            check_scorer_lower_is_better(scorer)

    def test_aggregation_methods(self, scorers, y_truth, y_pred):
        """Test that aggregation methods work."""
        for scorer in scorers:
            check_scorer_aggregation_methods(
                scorer, y_truth, y_pred, aggregation_methods=["stepwise", "vintagewise", "componentwise"]
            )

    @pytest.mark.parametrize(
        "scorer_class,param_name,invalid_value,error_match",
        [
            (MeanAbsoluteError, "groups", ["nonexistent"], "groups"),
            (MeanAbsoluteError, "components", ["nonexistent"], "components"),
            (MeanAbsoluteError, "aggregation_method", [["invalid"]], "aggregation_method"),
        ],
    )
    def test_parameter_validation(self, scorer_class, param_name, invalid_value, error_match):
        """Test that parameter validation is enforced."""
        check_scorer_parameter_validation(scorer_class, param_name, invalid_value, error_match)

    def test_yield_yohou_scorer_checks(self, scorers, y_truth, y_pred):
        """Generator yields well-formed tuples and emits every expected point-scorer check.

        Execution of each check on these scorers is covered systematically in
        test_scorer_registry_checks.py; this test guards generator completeness
        so a regression dropping (e.g.) the fitted-guard or multi-vintage check
        is caught.
        """
        # Names that must always be emitted for a point scorer.
        required_names = {
            "check_scorer_lower_is_better",
            "check_scorer_aggregation_methods",
            "check_scorer_tags_accessible_before_fit",
            "check_scorer_tags_static_after_fit",
            "check_scorer_tags_match_capabilities",
            "check_scorer_methods_call_check_is_fitted",
            "check_scorer_multi_vintage",
        }
        for scorer in scorers:
            checks = list(_yield_yohou_scorer_checks(scorer, y_truth, y_pred))

            # Each check should be a tuple (name, func, kwargs)
            for check_name, check_func, check_kwargs in checks:
                assert isinstance(check_name, str), "Check name should be string"
                assert callable(check_func), "Check function should be callable"
                assert isinstance(check_kwargs, dict), "Check kwargs should be dict"

            names = {name for name, _, _ in checks}
            missing = required_names - names
            assert not missing, f"Generator dropped expected checks: {missing}"
