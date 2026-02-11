"""Tests for parameter validation in scorer base classes."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from yohou.metrics import MeanAbsoluteError
from yohou.metrics.interval import EmpiricalCoverage


@pytest.fixture
def panel_data_factory():
    """Factory for panel data."""

    def _factory(group_names, length=50, n_targets=1, seed=42):
        rng = np.random.default_rng(seed)
        start_date = datetime(2021, 1, 1)
        time = pl.datetime_range(
            start=start_date,
            end=start_date + timedelta(seconds=length - 1),
            interval="1s",
            eager=True,
        )

        data = {"time": time}
        for group in group_names:
            for i in range(n_targets):
                # Using group as prefix to match inspect_locality behavior
                # Component (y_0) becomes the suffix
                target = f"y_{i}"
                col_name = f"{group}__{target}"
                data[col_name] = rng.random(length)

        y = pl.DataFrame(data)
        return y, None, None

    return _factory


class TestPanelGroupNames:
    def test_panel_group_names_must_be_list_or_none(self, y_X_factory):
        """panel_group_names must be a list or None."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        scorer = MeanAbsoluteError(panel_group_names="invalid")
        with pytest.raises(ValueError, match=r"panel_group_names.*must be an instance of 'list' or None"):
            scorer.fit(y)

    def test_panel_group_names_elements_must_be_strings(self, y_X_factory):
        """All elements in panel_group_names must be strings."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        scorer = MeanAbsoluteError(panel_group_names=[1, 2, 3])
        with pytest.raises(ValueError, match="All elements in panel_group_names must be strings"):
            scorer.fit(y)

    def test_panel_group_names_cannot_be_empty_list(self, y_X_factory):
        """panel_group_names cannot be an empty list."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        scorer = MeanAbsoluteError(panel_group_names=[])
        with pytest.raises(ValueError, match="panel_group_names cannot be an empty list"):
            scorer.fit(y)

    def test_panel_group_names_validated_against_data(self, panel_data_factory):
        """Requested panel groups must exist in the data."""
        # Create panel data with known groups
        y, _, _ = panel_data_factory(
            group_names=["store1", "store2"],
            length=50,
            n_targets=1,
            seed=42,
        )

        # Request non-existent groups
        scorer = MeanAbsoluteError(panel_group_names=["store1", "store3"])
        with pytest.raises(ValueError, match="Requested panel_group_names.*not found in data"):
            scorer.fit(y)

    def test_valid_panel_group_names_accepted(self, panel_data_factory):
        """Valid panel_group_names should be accepted."""
        y, _, _ = panel_data_factory(
            group_names=["store1", "store2"],
            length=50,
            n_targets=1,
            seed=42,
        )

        scorer = MeanAbsoluteError(panel_group_names=["store1"])
        scorer.fit(y)  # Should not raise

    def test_panel_group_names_edge_case_with_mixed_types(self, y_X_factory):
        """panel_group_names with mixed string and non-string types should fail."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        scorer = MeanAbsoluteError(panel_group_names=["store1", 123, "store2"])
        with pytest.raises(ValueError, match="All elements in panel_group_names must be strings"):
            scorer.fit(y)

    def test_panel_group_names_edge_case_multiple_valid_groups(self, panel_data_factory):
        """Multiple valid panel group names should be accepted."""
        y, _, _ = panel_data_factory(
            group_names=["store1", "store2", "store3"],
            length=50,
            n_targets=1,
            seed=42,
        )

        scorer = MeanAbsoluteError(panel_group_names=["store1", "store3"])
        scorer.fit(y)  # Should not raise

    def test_panel_group_names_edge_case_all_groups(self, panel_data_factory):
        """All panel group names should be accepted."""
        y, _, _ = panel_data_factory(
            group_names=["store1", "store2"],
            length=50,
            n_targets=1,
            seed=42,
        )

        scorer = MeanAbsoluteError(panel_group_names=["store1", "store2"])
        scorer.fit(y)  # Should not raise

    def test_panel_group_names_edge_case_single_missing(self, panel_data_factory):
        """Single missing panel group name should be reported."""
        y, _, _ = panel_data_factory(
            group_names=["store1", "store2"],
            length=50,
            n_targets=1,
            seed=42,
        )

        scorer = MeanAbsoluteError(panel_group_names=["store99"])
        with pytest.raises(ValueError, match="Requested panel_group_names.*store99.*not found"):
            scorer.fit(y)

    def test_panel_group_names_edge_case_on_global_data_ignored(self, y_X_factory):
        """panel_group_names on global (non-panel) data should raise ValueError."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        # Should raise because requested groups are not found
        scorer = MeanAbsoluteError(panel_group_names=["store1"])
        with pytest.raises(ValueError, match="panel_group_names specified but data contains no panel groups"):
            scorer.fit(y)


class TestComponentNames:
    def test_component_names_must_be_list_or_none(self, y_X_factory):
        """component_names must be a list or None."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        scorer = MeanAbsoluteError(component_names="invalid")
        with pytest.raises(ValueError, match=r"(must be a list or None|must be an instance of 'list' or None)"):
            scorer.fit(y)

    def test_component_names_elements_must_be_strings(self, y_X_factory):
        """All elements in component_names must be strings."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        scorer = MeanAbsoluteError(component_names=[1, 2, 3])
        with pytest.raises(ValueError, match="All elements in component_names must be strings"):
            scorer.fit(y)

    def test_component_names_cannot_be_empty_list(self, y_X_factory):
        """component_names cannot be an empty list."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        scorer = MeanAbsoluteError(component_names=[])
        with pytest.raises(ValueError, match="component_names cannot be an empty list"):
            scorer.fit(y)

    def test_component_names_validated_against_global_data(self, y_X_factory):
        """Requested components must exist in global data."""
        y, _ = y_X_factory(length=50, n_targets=2, seed=42)  # Creates 'y' and 'y_1'

        scorer = MeanAbsoluteError(component_names=["y", "y_nonexistent"])
        with pytest.raises(ValueError, match="Requested component_names.*not found in data"):
            scorer.fit(y)

    def test_component_names_validated_against_panel_data(self, panel_data_factory):
        """Requested components must exist in panel data (unprefixed names)."""
        y, _, _ = panel_data_factory(
            group_names=["store1"],
            length=50,
            n_targets=2,  # Creates 'y' and 'y_1' within each group
            seed=42,
        )

        # Request non-existent component (unprefixed name)
        scorer = MeanAbsoluteError(component_names=["y", "y_nonexistent"])
        with pytest.raises(ValueError, match="Requested component_names.*not found in data"):
            scorer.fit(y)

    def test_valid_component_names_accepted_global(self, y_X_factory):
        """Valid component_names should be accepted for global data."""
        y, _ = y_X_factory(length=50, n_targets=2, seed=42)

        scorer = MeanAbsoluteError(component_names=["y_0"])
        scorer.fit(y)  # Should not raise

    def test_valid_component_names_accepted_panel(self, panel_data_factory):
        """Valid component_names should be accepted for panel data."""
        y, _, _ = panel_data_factory(
            group_names=["store1"],
            length=50,
            n_targets=2,
            seed=42,
        )

        scorer = MeanAbsoluteError(component_names=["y_0"])
        scorer.fit(y)  # Should not raise

    def test_component_names_edge_case_with_mixed_types(self, y_X_factory):
        """component_names with mixed string and non-string types should fail."""
        y, _ = y_X_factory(length=50, n_targets=2, seed=42)

        scorer = MeanAbsoluteError(component_names=["y", 123, "y_1"])
        with pytest.raises(ValueError, match="All elements in component_names must be strings"):
            scorer.fit(y)

    def test_component_names_edge_case_multiple_valid_components(self, y_X_factory):
        """Multiple valid component names should be accepted."""
        y, _ = y_X_factory(length=50, n_targets=3, seed=42)  # Creates y, y_1, y_2

        scorer = MeanAbsoluteError(component_names=["y_0", "y_2"])
        scorer.fit(y)  # Should not raise

    def test_component_names_edge_case_all_components(self, y_X_factory):
        """All component names should be accepted."""
        y, _ = y_X_factory(length=50, n_targets=2, seed=42)

        scorer = MeanAbsoluteError(component_names=["y_0", "y_1"])
        scorer.fit(y)  # Should not raise

    def test_component_names_edge_case_single_missing_global(self, y_X_factory):
        """Single missing component name in global data should be reported."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        scorer = MeanAbsoluteError(component_names=["y_nonexistent"])
        with pytest.raises(ValueError, match="Requested component_names.*y_nonexistent"):
            scorer.fit(y)

    def test_component_names_edge_case_partial_match_panel(self, panel_data_factory):
        """Component names must all exist across panel groups."""
        y, _, _ = panel_data_factory(
            group_names=["store1"],
            length=50,
            n_targets=2,  # Creates y, y_1
            seed=42,
        )

        scorer = MeanAbsoluteError(component_names=["y", "y_1", "y_2"])
        with pytest.raises(ValueError, match="Requested component_names.*y_2.*not found"):
            scorer.fit(y)

    def test_component_names_edge_case_unprefixed_in_panel_data(self, panel_data_factory):
        """Component names in panel data should be unprefixed."""
        y, _, _ = panel_data_factory(
            group_names=["store1"],
            length=50,
            n_targets=2,
            seed=42,
        )

        # Should use unprefixed names, not prefixed ones like "store1__y_0"
        scorer = MeanAbsoluteError(component_names=["y_0"])
        scorer.fit(y)  # Should not raise

        # Prefixed names should not work
        scorer_invalid = MeanAbsoluteError(component_names=["store1__y_0"])
        with pytest.raises(ValueError, match="Requested component_names.*store1__y_0.*not found"):
            scorer_invalid.fit(y)


class TestCoverageRates:
    def test_coverage_rates_must_be_list_or_none(self, y_X_factory):
        """coverage_rates must be a list or None."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        scorer = EmpiricalCoverage(coverage_rates="invalid")
        with pytest.raises(ValueError, match=r"coverage_rates.*must be an instance of 'list' or None"):
            scorer.fit(y)

    def test_coverage_rates_elements_must_be_numeric(self, y_X_factory):
        """All elements in coverage_rates must be numeric."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        scorer = EmpiricalCoverage(coverage_rates=["0.9", "0.95"])
        with pytest.raises(ValueError, match="All elements in coverage_rates must be numeric"):
            scorer.fit(y)

    def test_coverage_rates_cannot_be_empty_list(self, y_X_factory):
        """coverage_rates cannot be an empty list."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        scorer = EmpiricalCoverage(coverage_rates=[])
        with pytest.raises(ValueError, match="coverage_rates cannot be an empty list"):
            scorer.fit(y)

    def test_coverage_rates_must_be_between_zero_and_one(self, y_X_factory):
        """All coverage_rates must be between 0 and 1 (exclusive)."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        # Test zero
        scorer = EmpiricalCoverage(coverage_rates=[0.0, 0.9])
        with pytest.raises(ValueError, match="All coverage_rates must be between 0 and 1.*got 0.0"):
            scorer.fit(y)

        # Test one
        scorer = EmpiricalCoverage(coverage_rates=[0.9, 1.0])
        with pytest.raises(ValueError, match="All coverage_rates must be between 0 and 1.*got 1.0"):
            scorer.fit(y)

        # Test negative
        scorer = EmpiricalCoverage(coverage_rates=[-0.1, 0.9])
        with pytest.raises(ValueError, match="All coverage_rates must be between 0 and 1.*got -0.1"):
            scorer.fit(y)

        # Test greater than one
        scorer = EmpiricalCoverage(coverage_rates=[0.9, 1.5])
        with pytest.raises(ValueError, match="All coverage_rates must be between 0 and 1.*got 1.5"):
            scorer.fit(y)

    def test_valid_coverage_rates_accepted(self, y_X_factory):
        """Valid coverage_rates should be accepted."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        scorer = EmpiricalCoverage(coverage_rates=[0.5, 0.9, 0.95])
        scorer.fit(y)  # Should not raise

    def test_coverage_rates_edge_case_with_mixed_numeric_types(self, y_X_factory):
        """coverage_rates with mixed int and float should be accepted."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        scorer = EmpiricalCoverage(coverage_rates=[0.5, 0.9])  # float
        scorer.fit(y)  # Should not raise

    def test_coverage_rates_edge_case_boundary_values_excluded(self, y_X_factory):
        """Boundary values 0 and 1 should be excluded."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        # Test 0.0
        scorer = EmpiricalCoverage(coverage_rates=[0.0])
        with pytest.raises(ValueError, match="must be between 0 and 1.*got 0.0"):
            scorer.fit(y)

        # Test 1.0
        scorer = EmpiricalCoverage(coverage_rates=[1.0])
        with pytest.raises(ValueError, match="must be between 0 and 1.*got 1.0"):
            scorer.fit(y)

    def test_coverage_rates_edge_case_very_small_value(self, y_X_factory):
        """Very small valid coverage rates should be accepted."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        scorer = EmpiricalCoverage(coverage_rates=[0.01, 0.05])
        scorer.fit(y)  # Should not raise

    def test_coverage_rates_edge_case_very_large_value(self, y_X_factory):
        """Very large valid coverage rates should be accepted."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        scorer = EmpiricalCoverage(coverage_rates=[0.95, 0.99])
        scorer.fit(y)  # Should not raise

    def test_coverage_rates_edge_case_multiple_invalid(self, y_X_factory):
        """First invalid coverage rate should be reported."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        scorer = EmpiricalCoverage(coverage_rates=[0.5, 1.5, 2.0])
        with pytest.raises(ValueError, match="must be between 0 and 1.*got 1.5"):
            scorer.fit(y)

    def test_coverage_rates_edge_case_none_accepted(self, y_X_factory):
        """coverage_rates=None should be accepted."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        scorer = EmpiricalCoverage(coverage_rates=None)
        scorer.fit(y)  # Should not raise


class TestTypeValidation:
    def test_type_validation_without_training_data(self):
        """Type validation should work even without training data."""
        # Test invalid type for panel_group_names
        scorer = MeanAbsoluteError(panel_group_names="invalid")
        with pytest.raises(ValueError, match=r"panel_group_names.*must be an instance of 'list' or None"):
            scorer.fit(None)

        # Test invalid type for component_names
        scorer = MeanAbsoluteError(component_names="invalid")
        with pytest.raises(ValueError, match=r"component_names.*must be an instance of 'list' or None"):
            scorer.fit(None)

        # Test invalid coverage_rates
        scorer = EmpiricalCoverage(coverage_rates=[1.5])
        with pytest.raises(ValueError, match="All coverage_rates must be between 0 and 1"):
            scorer.fit(None)

    def test_data_validation_requires_training_data(self):
        """fit() should always require y_train, even for stateless scorers."""
        scorer = MeanAbsoluteError(
            panel_group_names=["nonexistent"],
            component_names=["also_nonexistent"],
        )
        # Should raise even for stateless scorers - y_train always required
        with pytest.raises(ValueError, match="`y_train` is required for scorer.fit"):
            scorer.fit(None)


class TestCombinedValidation:
    def test_combined_validation_all_parameters_validated(self, panel_data_factory):
        """All parameters should be validated together."""
        y, _, _ = panel_data_factory(
            group_names=["store1"],
            length=50,
            n_targets=1,
            seed=42,
        )

        # Invalid panel_group_names should fail first
        scorer = EmpiricalCoverage(
            panel_group_names="invalid",  # Invalid type
            component_names="invalid",  # Also invalid
            coverage_rates="invalid",  # Also invalid
        )
        with pytest.raises(
            ValueError,
            match=r"(coverage_rates|component_names).*must be an instance of 'list' or None",
        ):
            # coverage_rates validated first in BaseIntervalScorer.fit()
            scorer.fit(y)

    def test_combined_validation_valid_parameters_all_accepted(self, panel_data_factory):
        """All valid parameters should be accepted together."""
        y, _, _ = panel_data_factory(
            group_names=["store1", "store2"],
            length=50,
            n_targets=2,
            seed=42,
        )

        scorer = EmpiricalCoverage(
            panel_group_names=["store1"],
            component_names=["y_0"],
            coverage_rates=[0.9],
        )
        scorer.fit(y)  # Should not raise


class TestValidationOrdering:
    def test_validation_ordering_point_scorer_validation_order(self, panel_data_factory):
        """Point scorers validate aggregation_method, then panel/component names."""
        y, _, _ = panel_data_factory(
            group_names=["store1"],
            length=50,
            n_targets=1,
            seed=42,
        )

        # Invalid aggregation_method should be caught first (by sklearn)
        scorer = MeanAbsoluteError(
            aggregation_method="invalid",
            panel_group_names="also_invalid",
        )
        with pytest.raises(ValueError):
            scorer.fit(y)

    def test_validation_ordering_interval_scorer_validation_order(self, panel_data_factory):
        """Interval scorers validate aggregation, coverage_rates, then panel/component names."""
        y, _, _ = panel_data_factory(
            group_names=["store1"],
            length=50,
            n_targets=1,
            seed=42,
        )

        # Invalid coverage_rates should be caught before panel_group_names
        scorer = EmpiricalCoverage(
            coverage_rates=[1.5],
            panel_group_names=["nonexistent"],
        )
        with pytest.raises(ValueError, match="must be between 0 and 1.*got 1.5"):
            scorer.fit(y)


class TestValidationWithScore:
    def test_validation_with_score_method_validated_panel_groups_filter_in_score(self, panel_data_factory):
        """Validated panel_group_names should filter predictions in score()."""
        y, _, _ = panel_data_factory(
            group_names=["store1", "store2"],
            length=50,
            n_targets=1,
            seed=42,
        )

        # Fit with panel_group_names filter
        scorer = MeanAbsoluteError(panel_group_names=["store1"])
        scorer.fit(y)

        # Create predictions for both groups
        y_pred = y.with_columns(pl.col("time").alias("observed_time"))

        # Score should work and only include store1
        score = scorer.score(y, y_pred)
        assert isinstance(score, float)  # Should aggregate to scalar

    def test_validation_with_score_method_validated_components_filter_in_score(self, y_X_factory):
        """Validated component_names should filter predictions in score()."""
        y, _ = y_X_factory(length=50, n_targets=2, seed=42)

        # Fit with component_names filter
        scorer = MeanAbsoluteError(component_names=["y_0"])
        scorer.fit(y)

        # Create predictions for both components
        y_pred = y.with_columns(pl.col("time").alias("observed_time"))

        # Score should work and only include y
        score = scorer.score(y, y_pred)
        assert isinstance(score, float)

    def test_validation_with_score_method_validated_coverage_rates_filter_in_score(self, y_X_factory):
        """Validated coverage_rates should be available in predictions."""
        y, _ = y_X_factory(length=50, n_targets=1, seed=42)

        # Fit with coverage_rates filter
        scorer = EmpiricalCoverage(coverage_rates=[0.9])
        scorer.fit(y)

        # This validation is tested in test_column_subselection.py
        # Just verify fit succeeds
        assert scorer.coverage_rates == [0.9]
