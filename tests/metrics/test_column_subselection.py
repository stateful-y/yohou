"""Tests for scorer column subselection via panel_group_names, component_names, and coverage_rates."""

from datetime import datetime

import polars as pl
import pytest

from yohou.metrics import (
    EmpiricalCoverage,
    MeanAbsoluteError,
    MeanIntervalWidth,
)


@pytest.fixture
def panel_point_data():
    """Panel point forecast data for testing."""
    y_true = pl.DataFrame(
        {
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "sales__store_1": [10.0, 15.0, 20.0],
            "sales__store_2": [12.0, 18.0, 24.0],
            "demand__store_1": [5.0, 7.0, 9.0],
            "demand__store_2": [6.0, 8.0, 10.0],
        }
    )
    y_pred = pl.DataFrame(
        {
            "observed_time": [datetime(2019, 12, 31)] * 3,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            # Sales predictions have error of 1.0
            "sales__store_1": [11.0, 14.0, 19.0],
            "sales__store_2": [13.0, 17.0, 23.0],
            # Demand predictions: store_1 has error of 2.0, store_2 has error of 3.0
            "demand__store_1": [7.0, 9.0, 11.0],
            "demand__store_2": [9.0, 11.0, 13.0],
        }
    )
    return y_true, y_pred


@pytest.fixture
def panel_interval_data():
    """Panel interval forecast data for testing."""
    y_true = pl.DataFrame(
        {
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "sales__store_1": [10.0, 15.0],
            "sales__store_2": [12.0, 18.0],
        }
    )
    y_pred = pl.DataFrame(
        {
            "observed_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "sales__store_1_lower_0.9": [8.0, 13.0],
            "sales__store_1_upper_0.9": [12.0, 17.0],
            "sales__store_1_lower_0.95": [7.0, 12.0],
            "sales__store_1_upper_0.95": [13.0, 18.0],
            "sales__store_2_lower_0.9": [10.0, 16.0],
            "sales__store_2_upper_0.9": [14.0, 20.0],
            "sales__store_2_lower_0.95": [9.0, 15.0],
            "sales__store_2_upper_0.95": [15.0, 21.0],
        }
    )
    return y_true, y_pred


@pytest.fixture
def global_multi_component_data():
    """Global data with multiple components for testing."""
    y_true = pl.DataFrame(
        {
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "sales": [10.0, 15.0, 20.0],
            "demand": [5.0, 7.0, 9.0],
        }
    )
    y_pred = pl.DataFrame(
        {
            "observed_time": [datetime(2019, 12, 31)] * 3,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "sales": [11.0, 14.0, 19.0],  # Error of 1.0
            "demand": [7.0, 9.0, 11.0],   # Error of 2.0
        }
    )
    return y_true, y_pred


# ============================================================================
# Point Scorers - Panel Group Filtering
# ============================================================================


def test_point_scorer_filter_panel_group(panel_point_data):
    """Test filtering by panel_group_names for point scorers."""
    y_true, y_pred = panel_point_data

    # Score only sales group (includes both sales__store_1 and sales__store_2)
    mae_sales = MeanAbsoluteError(panel_group_names=["sales"])
    mae_sales.fit(y_true)
    score_sales = mae_sales.score(y_true, y_pred)

    # Score only demand group (includes both demand__store_1 and demand__store_2)
    mae_demand = MeanAbsoluteError(panel_group_names=["demand"])
    mae_demand.fit(y_true)
    score_demand = mae_demand.score(y_true, y_pred)

    # Score all groups
    mae_all = MeanAbsoluteError()
    mae_all.fit(y_true)
    score_all = mae_all.score(y_true, y_pred)

    # sales and demand scores should differ from each other
    assert score_sales != score_demand

    # They should also differ from all groups score (which averages both)
    assert score_sales != score_all or score_demand != score_all


def test_point_scorer_filter_component_names_panel(panel_point_data):
    """Test filtering by component_names for panel data."""
    y_true, y_pred = panel_point_data

    # Score only store_1 components
    mae_store1 = MeanAbsoluteError(component_names=["store_1"])
    mae_store1.fit(y_true)
    score_store1 = mae_store1.score(y_true, y_pred)

    # Score all components
    mae_all = MeanAbsoluteError()
    mae_all.fit(y_true)
    score_all = mae_all.score(y_true, y_pred)

    # Filtering should produce different score
    assert score_store1 != score_all


def test_point_scorer_filter_both_panel_and_component(panel_point_data):
    """Test filtering by both panel_group_names and component_names."""
    y_true, y_pred = panel_point_data

    # Score only sales group, store_1 component
    mae_filtered = MeanAbsoluteError(
        panel_group_names=["sales"], component_names=["store_1"]
    )
    mae_filtered.fit(y_true)
    score_filtered = mae_filtered.score(y_true, y_pred)

    # Score all
    mae_all = MeanAbsoluteError()
    mae_all.fit(y_true)
    score_all = mae_all.score(y_true, y_pred)

    # Should be different
    assert score_filtered != score_all
    assert isinstance(score_filtered, float)


# ============================================================================
# Point Scorers - Component Name Filtering (Global Data)
# ============================================================================


def test_point_scorer_filter_component_names_global(global_multi_component_data):
    """Test filtering by component_names for global data."""
    y_true, y_pred = global_multi_component_data

    # Score only sales component
    mae_sales = MeanAbsoluteError(component_names=["sales"])
    mae_sales.fit(y_true)
    score_sales = mae_sales.score(y_true, y_pred)

    # Score only demand component
    mae_demand = MeanAbsoluteError(component_names=["demand"])
    mae_demand.fit(y_true)
    score_demand = mae_demand.score(y_true, y_pred)

    # Score all components
    mae_all = MeanAbsoluteError()
    mae_all.fit(y_true)
    score_all = mae_all.score(y_true, y_pred)

    # Each filtered score should differ
    assert score_sales != score_demand
    assert score_sales != score_all
    assert score_demand != score_all


# ============================================================================
# Interval Scorers - Coverage Rate Filtering
# ============================================================================


def test_interval_scorer_filter_coverage_rates(panel_interval_data):
    """Test filtering by coverage_rates for interval scorers."""
    y_true, y_pred = panel_interval_data

    # Score only 0.9 coverage
    cov_09 = EmpiricalCoverage(coverage_rates=[0.9])
    cov_09.fit(y_true)
    score_09 = cov_09.score(y_true, y_pred)

    # Score only 0.95 coverage
    cov_095 = EmpiricalCoverage(coverage_rates=[0.95])
    cov_095.fit(y_true)
    score_095 = cov_095.score(y_true, y_pred)

    # Score all coverage rates (should return dict unless fully aggregated)
    cov_all = EmpiricalCoverage(aggregation_method=["timewise", "componentwise"])
    cov_all.fit(y_true)
    score_all = cov_all.score(y_true, y_pred)

    # When filtering to single rate, result should be float (fully aggregated with coveragewise)
    assert isinstance(score_09, float)
    assert isinstance(score_095, float)

    # When not aggregating coveragewise, result should be dict
    assert isinstance(score_all, dict)
    assert 0.9 in score_all
    assert 0.95 in score_all


def test_interval_scorer_filter_panel_group(panel_interval_data):
    """Test filtering by panel_group_names for interval scorers."""
    y_true, y_pred = panel_interval_data

    # Score only sales group
    width_sales = MeanIntervalWidth(panel_group_names=["sales"])
    width_sales.fit(y_true)
    score_sales = width_sales.score(y_true, y_pred)

    # Score all groups
    width_all = MeanIntervalWidth()
    width_all.fit(y_true)
    score_all = width_all.score(y_true, y_pred)

    # Should produce consistent results since only sales group exists
    assert isinstance(score_sales, float)
    assert isinstance(score_all, float)


def test_interval_scorer_filter_component_names(panel_interval_data):
    """Test filtering by component_names for interval scorers."""
    y_true, y_pred = panel_interval_data

    # Score only store_1
    cov_store1 = EmpiricalCoverage(component_names=["store_1"])
    cov_store1.fit(y_true)
    score_store1 = cov_store1.score(y_true, y_pred)

    # Score only store_2
    cov_store2 = EmpiricalCoverage(component_names=["store_2"])
    cov_store2.fit(y_true)
    score_store2 = cov_store2.score(y_true, y_pred)

    # Both should be valid floats
    assert isinstance(score_store1, float)
    assert isinstance(score_store2, float)


def test_interval_scorer_filter_all_three(panel_interval_data):
    """Test filtering by panel_group_names, component_names, and coverage_rates."""
    y_true, y_pred = panel_interval_data

    # Filter by all three parameters
    cov_filtered = EmpiricalCoverage(
        panel_group_names=["sales"],
        component_names=["store_1"],
        coverage_rates=[0.9],
    )
    cov_filtered.fit(y_true)
    score_filtered = cov_filtered.score(y_true, y_pred)

    # Should produce scalar result
    assert isinstance(score_filtered, float)


# ============================================================================
# Edge Cases
# ============================================================================


def test_empty_filter_returns_all_data(panel_point_data):
    """Test that None filters return all data (default behavior)."""
    y_true, y_pred = panel_point_data

    # Explicit None parameters
    mae_none = MeanAbsoluteError(
        panel_group_names=None, component_names=None
    )
    mae_none.fit(y_true)
    score_none = mae_none.score(y_true, y_pred)

    # Default parameters (should be same)
    mae_default = MeanAbsoluteError()
    mae_default.fit(y_true)
    score_default = mae_default.score(y_true, y_pred)

    # Should produce same result
    assert score_none == score_default


def test_nonexistent_panel_group_produces_empty_result(panel_point_data):
    """Test that filtering for non-existent panel group raises ValueError."""
    y_true, y_pred = panel_point_data

    # Filter for non-existent group
    mae_invalid = MeanAbsoluteError(panel_group_names=["nonexistent"])
    
    with pytest.raises(ValueError, match="(Invalid|Requested) panel_group_names.*not found"):
        mae_invalid.fit(y_true)
        mae_invalid.score(y_true, y_pred)


def test_nonexistent_component_produces_empty_result(global_multi_component_data):
    """Test that filtering for non-existent component raises ValueError."""
    y_true, y_pred = global_multi_component_data

    # Filter for non-existent component
    mae_invalid = MeanAbsoluteError(component_names=["nonexistent"])
    
    with pytest.raises(ValueError, match="(Invalid|Requested) component_names.*not found"):
        mae_invalid.fit(y_true)
        mae_invalid.score(y_true, y_pred)


def test_invalid_coverage_rate_raises_error(panel_interval_data):
    """Test that requesting non-existent coverage_rates raises an error."""
    y_true, y_pred = panel_interval_data

    # Request coverage rate that doesn't exist in predictions
    cov_invalid = EmpiricalCoverage(coverage_rates=[0.99])

    # Should raise ValueError about missing rate
    import pytest

    with pytest.raises(ValueError, match="coverage_rates.*not found"):
        cov_invalid.fit(y_true)
        cov_invalid.score(y_true, y_pred)
