"""Comprehensive tests for aggregation_method parameter across all scorers."""

from datetime import datetime

import polars as pl
import pytest

from yohou.metrics import (
    EmpiricalCoverage,
    IntervalScore,
    MeanAbsoluteError,
    MeanIntervalWidth,
    MeanSquaredError,
    PinballLoss,
    RootMeanSquaredError,
)


@pytest.fixture
def point_data():
    """Create sample point prediction data with multiple components."""
    y_true = pl.DataFrame({
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "value1": [10.0, 20.0, 30.0],
        "value2": [15.0, 25.0, 35.0],
    })
    y_pred = pl.DataFrame({
        "observed_time": [datetime(2019, 12, 31)] * 3,
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "value1": [12.0, 18.0, 32.0],
        "value2": [14.0, 26.0, 36.0],
    })
    return y_true, y_pred


@pytest.fixture
def interval_data():
    """Create sample interval prediction data with multiple components."""
    y_true = pl.DataFrame({
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "value1": [10.0, 20.0, 30.0],
        "value2": [15.0, 25.0, 35.0],
    })
    y_pred = pl.DataFrame({
        "observed_time": [datetime(2019, 12, 31)] * 3,
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "value1_lower_0.9": [8.0, 18.0, 28.0],
        "value1_upper_0.9": [12.0, 22.0, 32.0],
        "value2_lower_0.9": [13.0, 23.0, 33.0],
        "value2_upper_0.9": [17.0, 27.0, 37.0],
    })
    return y_true, y_pred


@pytest.fixture
def multi_rate_interval_data():
    """Create sample interval prediction data with multiple rates."""
    y_true = pl.DataFrame({
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        "value": [10.0, 20.0],
    })
    y_pred = pl.DataFrame({
        "observed_time": [datetime(2019, 12, 31)] * 2,
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        "value_lower_0.9": [8.0, 18.0],
        "value_upper_0.9": [12.0, 22.0],
        "value_lower_0.95": [7.0, 17.0],
        "value_upper_0.95": [13.0, 23.0],
    })
    return y_true, y_pred


@pytest.fixture
def panel_point_data():
    """Create sample panel point prediction data."""
    y_true = pl.DataFrame({
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        "sales__store_1": [10.0, 20.0],
        "sales__store_2": [15.0, 25.0],
    })
    y_pred = pl.DataFrame({
        "observed_time": [datetime(2019, 12, 31)] * 2,
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        "sales__store_1": [12.0, 18.0],
        "sales__store_2": [14.0, 26.0],
    })
    return y_true, y_pred


@pytest.fixture
def simple_data():
    """Create simple test data for validation tests."""
    y_true = pl.DataFrame({
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        "value1": [10.0, 20.0],
    })
    return y_true


@pytest.fixture
def equivalence_simple_data():
    """Create simple test data."""
    y_true = pl.DataFrame({
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        "value1": [10.0, 20.0],
        "value2": [15.0, 25.0],
    })
    y_pred = pl.DataFrame({
        "observed_time": [datetime(2019, 12, 31)] * 2,
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        "value1": [11.0, 19.0],
        "value2": [14.0, 26.0],
    })
    return y_true, y_pred


class TestPointScorerAggregation:
    def test_pointscorer_aggregation_all_returns_scalar(self, point_data):
        """Test that aggregation_method='all' returns a scalar."""
        y_true, y_pred = point_data
        mae = MeanAbsoluteError(aggregation_method="all")
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)
        assert isinstance(result, float), "aggregation_method='all' should return scalar"

    def test_pointscorer_aggregation_timewise_returns_dataframe_with_components(self, point_data):
        """Test that aggregation_method=['timewise'] returns per-component DataFrame."""
        y_true, y_pred = point_data
        mae = MeanAbsoluteError(aggregation_method=["timewise"])
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)

        assert isinstance(result, pl.DataFrame), "Should return DataFrame"
        assert result.shape == (1, 2), "Should have 1 row and 2 component columns"
        assert set(result.columns) == {"value1", "value2"}, "Should have component columns"

    def test_pointscorer_aggregation_componentwise_returns_dataframe_with_time(self, point_data):
        """Test that aggregation_method=['componentwise'] returns per-timestep DataFrame."""
        y_true, y_pred = point_data
        mae = MeanAbsoluteError(aggregation_method=["componentwise"])
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)

        assert isinstance(result, pl.DataFrame), "Should return DataFrame"
        assert result.shape == (3, 2), "Should have 3 rows and 2 columns (time + metric)"
        assert "time" in result.columns, "Should have time column"
        assert "mae" in result.columns, "Should have mae column"

    def test_pointscorer_aggregation_timewise_componentwise_returns_scalar(self, point_data):
        """Test that both dimensions aggregated returns scalar."""
        y_true, y_pred = point_data
        mae = MeanAbsoluteError(aggregation_method=["timewise", "componentwise"])
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)
        assert isinstance(result, float), "Should return scalar when both dimensions aggregated"

    def test_pointscorer_aggregation_all_equivalent_to_both_dimensions(self, point_data):
        """Test that 'all' is equivalent to ['timewise', 'componentwise']."""
        y_true, y_pred = point_data

        mae_all = MeanAbsoluteError(aggregation_method="all")
        mae_both = MeanAbsoluteError(aggregation_method=["timewise", "componentwise"])

        mae_all.fit(y_true)
        result_all = mae_all.score(y_true, y_pred)
        mae_both.fit(y_true)
        result_both = mae_both.score(y_true, y_pred)

        assert result_all == result_both, "'all' should equal ['timewise', 'componentwise']"

    def test_pointscorer_aggregation_with_multiple_scorers(self, point_data):
        """Test aggregation works across different scorer types."""
        y_true, y_pred = point_data

        scorers = [
            MeanAbsoluteError(aggregation_method="all"),
            MeanSquaredError(aggregation_method="all"),
            RootMeanSquaredError(aggregation_method="all"),
        ]

        for scorer in scorers:
            scorer.fit(y_true)
            result = scorer.score(y_true, y_pred)
            assert isinstance(result, float), f"{scorer.__class__.__name__} should return scalar"


class TestPanelAggregation:
    def test_panelaggregation_point_metric_componentwise_panel(self):
        # Setup panel data
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "g1__val": [10.0, 20.0],
            "g2__val": [100.0, 200.0],
        })
        y_pred = pl.DataFrame({
            "observed_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "g1__val": [12.0, 18.0],  # Errors: 2, 2 -> MAE=2
            "g2__val": [110.0, 190.0],  # Errors: 10, 10 -> MAE=10
        })

        scorer = MeanAbsoluteError(aggregation_method=["componentwise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)

        # Expected columns: time, g1__mae, g2__mae
        assert "time" in result.columns
        assert "g1__mae" in result.columns
        assert "g2__mae" in result.columns
        assert "score" not in result.columns

        assert result["g1__mae"].to_list() == [2.0, 2.0]
        assert result["g2__mae"].to_list() == [10.0, 10.0]

    def test_panelaggregation_point_metric_componentwise_panel_rmse(self):
        # RMSE involves sqrt
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "g1__val": [0.0],
        })
        y_pred = pl.DataFrame({
            "observed_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "g1__val": [4.0],  # Error 4, SqError 16, MeanSqError 16, RMSE 4
        })

        scorer = RootMeanSquaredError(aggregation_method=["componentwise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)

        assert "g1__rmse" in result.columns
        assert result["g1__rmse"][0] == 4.0

    def test_panelaggregation_interval_metric_componentwise_panel(self):
        # Setup panel data
        y_true = pl.DataFrame({"time": [datetime(2020, 1, 1)], "g1__val": [10.0], "g2__val": [100.0]})
        y_pred = pl.DataFrame({
            "observed_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "g1__val_lower_0.9": [8.0],
            "g1__val_upper_0.9": [12.0],  # Width 4
            "g2__val_lower_0.9": [90.0],
            "g2__val_upper_0.9": [110.0],  # Width 20
        })

        # MeanIntervalWidth
        # Include coveragewise to aggregate over rates (even if single rate) and get 'width' suffix
        scorer = MeanIntervalWidth(aggregation_method=["componentwise", "coveragewise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)

        assert "time" in result.columns
        assert "g1__width" in result.columns
        assert "g2__width" in result.columns

        assert result["g1__width"][0] == 4.0
        assert result["g2__width"][0] == 20.0

    def test_panelaggregation_interval_metric_componentwise_panel_per_rate(self):
        # Setup panel data with multiple rates
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "g1__val": [10.0],
        })
        y_pred = pl.DataFrame({
            "observed_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "g1__val_lower_0.9": [8.0],
            "g1__val_upper_0.9": [12.0],  # Width 4
            "g1__val_lower_0.5": [9.0],
            "g1__val_upper_0.5": [11.0],  # Width 2
        })

        # MeanIntervalWidth without coveragewise in aggregation_method
        scorer = MeanIntervalWidth(aggregation_method=["componentwise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)

        assert "time" in result.columns
        # With coveragewise not in aggregation, BaseIntervalScorer returns rate columns per group
        # g1__rate_0.9, g1__rate_0.5 (BaseIntervalScorer implementation detail)
        assert "g1__rate_0.9" in result.columns
        assert "g1__rate_0.5" in result.columns

        assert result["g1__rate_0.9"][0] == 4.0
        assert result["g1__rate_0.5"][0] == 2.0

    def test_panelaggregation_interval_score_componentwise_panel(self):
        # IntervalScore
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "g1__val": [10.0],
        })
        y_pred = pl.DataFrame({
            "observed_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "g1__val_lower_0.9": [8.0],
            "g1__val_upper_0.9": [12.0],  # Width 4, in interval
        })

        scorer = IntervalScore(aggregation_method=["componentwise", "coveragewise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)

        # Check column name
        # Base returns g1__score
        # IntervalScore renames __score -> __interval_score
        assert "g1__interval_score" in result.columns

    def test_panelaggregation_all_aggregation_with_panel_weights(self):
        """Test that 'all' aggregation respects panel groups (hierarchical aggregation)."""
        # Scenario:
        # Group A: 10 columns, error=1.0
        # Group B: 1 column, error=100.0

        # Flat mean: (10*1 + 100)/11 = 10.0
        # Group mean: (1.0 + 100.0)/2 = 50.5

        y_true_data = {"time": [datetime(2020, 1, 1)]}
        y_pred_data = {"observed_time": [datetime(2019, 12, 31)], "time": [datetime(2020, 1, 1)]}

        # Group A
        for i in range(10):
            y_true_data[f"groupA__c{i}"] = [10.0]
            y_pred_data[f"groupA__c{i}"] = [11.0]  # Error 1.0

        # Group B
        y_true_data["groupB__c0"] = [10.0]
        y_pred_data["groupB__c0"] = [110.0]  # Error 100.0

        y_true = pl.DataFrame(y_true_data)
        y_pred = pl.DataFrame(y_pred_data)

        # Equal weights (default)
        scorer = MeanAbsoluteError(aggregation_method="all")
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)

        # Should be hierarchical mean (50.5), not flat mean (10.0)
        assert result == 50.5

        # Custom weights
        weights = {"groupA": 0.9, "groupB": 0.1}
        scorer_weighted = MeanAbsoluteError(aggregation_method="all", panel_group_weight=weights)
        scorer.fit(y_true)
        scorer_weighted.fit(y_true)
        result_weighted = scorer_weighted.score(y_true, y_pred)

        # Weighted: 1.0 * 0.9 + 100.0 * 0.1 = 0.9 + 10.0 = 10.9
        # Weighted average formula is sum(w*x)/sum(w)
        # sum(w) = 1.0
        assert abs(result_weighted - 10.9) < 1e-10


class TestIntervalScorerAggregation:
    def test_intervalscorer_aggregation_coveragewise_accepted_in_fit(self, multi_rate_interval_data):
        """Test that 'coveragewise' is accepted in fit for interval scorers."""
        y_true, _ = multi_rate_interval_data
        cov = EmpiricalCoverage(aggregation_method=["coveragewise"])
        # Should not raise
        cov.fit(y_true)
        assert cov.aggregation_method == ["coveragewise"]

    def test_intervalscorer_aggregation_invalid_method_rejected_in_fit(self, multi_rate_interval_data):
        """Test that invalid method is rejected in fit for interval scorers."""
        y_true, _ = multi_rate_interval_data
        cov = EmpiricalCoverage(aggregation_method=["invalid"])
        with pytest.raises(ValueError, match="Invalid aggregation_method 'invalid' in list"):
            cov.fit(y_true)

    def test_intervalscorer_aggregation_all_returns_scalar(self, interval_data):
        """Test that aggregation_method='all' returns a scalar."""
        y_true, y_pred = interval_data
        coverage = EmpiricalCoverage(aggregation_method="all")
        coverage.fit(y_true)
        result = coverage.score(y_true, y_pred)
        assert isinstance(result, float), "aggregation_method='all' should return scalar"

    def test_intervalscorer_aggregation_timewise_returns_dataframe(self, interval_data):
        """Test that aggregation_method=['timewise'] returns per-component DataFrame."""
        y_true, y_pred = interval_data
        coverage = EmpiricalCoverage(aggregation_method=["timewise"])
        coverage.fit(y_true)
        result = coverage.score(y_true, y_pred)

        assert isinstance(result, pl.DataFrame), "Should return DataFrame"
        assert result.shape == (1, 2), "Should have 1 row and 2 component columns"

    def test_intervalscorer_aggregation_componentwise_returns_dataframe(self, interval_data):
        """Test that aggregation_method=['componentwise'] returns per-timestep DataFrame."""
        y_true, y_pred = interval_data
        coverage = EmpiricalCoverage(aggregation_method=["componentwise"])
        coverage.fit(y_true)
        result = coverage.score(y_true, y_pred)

        assert isinstance(result, pl.DataFrame), "Should return DataFrame"
        assert result.shape == (3, 2), "Should have 3 rows (timesteps)"
        assert "time" in result.columns, "Should have time column"

    def test_intervalscorer_aggregation_without_coveragewise(self, multi_rate_interval_data):
        """Test aggregation without 'coveragewise' (returns dict)."""
        y_true, y_pred = multi_rate_interval_data

        # Explicitly exclude 'coveragewise' to get per-rate scores
        # "all" would include "coveragewise"
        coverage = EmpiricalCoverage(aggregation_method=["timewise", "componentwise", "groupwise"])
        coverage.fit(y_true)
        result = coverage.score(y_true, y_pred)

        assert isinstance(result, dict), "Should return dict when coveragewise is excluded"
        assert 0.9 in result, "Should have coverage rate 0.9"
        assert 0.95 in result, "Should have coverage rate 0.95"
        assert all(isinstance(v, float) for v in result.values()), "Values should be floats"

    def test_intervalscorer_aggregation_timewise_per_rate(self, multi_rate_interval_data):
        """Test timewise aggregation without coveragewise returns DataFrame with rate columns."""
        y_true, y_pred = multi_rate_interval_data

        # aggregation_method=["timewise"] does not include "coveragewise", so it returns per rate
        coverage = EmpiricalCoverage(aggregation_method=["timewise"])
        coverage.fit(y_true)
        result = coverage.score(y_true, y_pred)

        assert isinstance(result, pl.DataFrame), "Should return DataFrame"
        # Should have columns like value_rate_0.9, value_rate_0.95
        assert any("rate_0.9" in col for col in result.columns), "Should have rate-specific columns"

    def test_intervalscorer_aggregation_with_multiple_interval_scorers(self, interval_data):
        """Test aggregation works across different interval scorer types."""
        y_true, y_pred = interval_data

        scorers = [
            EmpiricalCoverage(aggregation_method="all"),
            MeanIntervalWidth(aggregation_method="all"),
            IntervalScore(aggregation_method="all"),
            PinballLoss(aggregation_method="all"),
        ]

        for scorer in scorers:
            scorer.fit(y_true)
            result = scorer.score(y_true, y_pred)
            assert isinstance(result, float | dict), f"{scorer.__class__.__name__} should return scalar or dict"

    def test_intervalscorer_aggregation_coveragewise_aggregation_explicit(self, multi_rate_interval_data):
        """Test explicit aggregation_method='coveragewise'."""
        y_true, y_pred = multi_rate_interval_data

        # Case 1: Explicitly exclude coveragewise (return per rate)
        # now we must explicitly manage the list.
        # "timewise", "componentwise" -> no "coveragewise" -> return dict of scalars
        coverage_explicit_ex = EmpiricalCoverage(aggregation_method=["timewise", "componentwise"])
        coverage_explicit_ex.fit(y_true)
        result_ex = coverage_explicit_ex.score(y_true, y_pred)

        # Should be a dict of scalars (timewise+componentwise reduces everything but rate)
        assert isinstance(result_ex, dict), "Should return dict when coveragewise is excluded"
        assert 0.9 in result_ex
        assert 0.95 in result_ex
        assert isinstance(result_ex[0.9], float)

        # Case 2: Explicitly include coveragewise (scalar)
        coverage_with_cov = EmpiricalCoverage(aggregation_method=["timewise", "componentwise", "coveragewise"])
        coverage_with_cov.fit(y_true)
        result_with_cov = coverage_with_cov.score(y_true, y_pred)
        assert isinstance(result_with_cov, float), "Should return scalar when coveragewise is included"


class TestPanelDataAggregation:
    def test_paneldata_aggregation_all_returns_scalar(self, panel_point_data):
        """Test that panel data with aggregation_method='all' returns scalar."""
        y_true, y_pred = panel_point_data
        mae = MeanAbsoluteError(aggregation_method="all")
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)
        assert isinstance(result, float), "Panel data with 'all' should return scalar"

    def test_paneldata_aggregation_timewise_returns_dataframe(self, panel_point_data):
        """Test that panel data with timewise aggregation returns per-group-component DataFrame."""
        y_true, y_pred = panel_point_data
        mae = MeanAbsoluteError(aggregation_method=["timewise"])
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)

        assert isinstance(result, pl.DataFrame), "Should return DataFrame"
        # Should have panel group columns
        assert "sales__store_1" in result.columns, "Should have panel group columns"
        assert "sales__store_2" in result.columns, "Should have panel group columns"

    def test_paneldata_aggregation_componentwise_returns_dataframe(self, panel_point_data):
        """Test that panel data with componentwise aggregation returns per-timestep DataFrame."""
        y_true, y_pred = panel_point_data
        mae = MeanAbsoluteError(aggregation_method=["componentwise"])
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)

        assert isinstance(result, pl.DataFrame), "Should return DataFrame"
        assert "time" in result.columns, "Should have time column"
        assert result.shape[0] == 2, "Should have 2 rows (timesteps)"


class TestValidation:
    def test_validation_invalid_list_element_raises_at_fit_time(self, simple_data):
        """Test that invalid list elements are caught at fit time."""
        mae = MeanAbsoluteError(aggregation_method=["invalid"])
        with pytest.raises(ValueError, match="Invalid aggregation_method 'invalid' in list"):
            mae.fit(simple_data)

    def test_validation_mixed_valid_invalid_raises_at_fit_time(self, simple_data):
        """Test that mixed valid/invalid list elements are caught at fit time."""
        mae = MeanAbsoluteError(aggregation_method=["timewise", "bad_method"])
        with pytest.raises(ValueError, match="Invalid aggregation_method 'bad_method' in list"):
            mae.fit(simple_data)

    def test_validation_empty_list_raises_at_fit_time(self, simple_data):
        """Test that empty list is caught at fit time."""
        mae = MeanAbsoluteError(aggregation_method=[])
        with pytest.raises(ValueError, match="aggregation_method list cannot be empty"):
            mae.fit(simple_data)

    def test_validation_invalid_string_raises_at_fit_time(self, simple_data):
        """Test that invalid strings are caught by sklearn at fit time."""
        mae = MeanAbsoluteError(aggregation_method="invalid_string")
        with pytest.raises(ValueError, match="aggregation_method.*must be"):
            mae.fit(simple_data)

    def test_validation_invalid_type_raises_at_fit_time(self, simple_data):
        """Test that invalid types are caught by sklearn at fit time."""
        mae = MeanAbsoluteError(aggregation_method=123)
        with pytest.raises(ValueError, match="aggregation_method.*must be"):
            mae.fit(simple_data)

    def test_validation_valid_single_methods_accepted(self):
        """Test that all valid single methods are accepted as strings or lists."""
        valid_methods = ["timewise", "componentwise", "groupwise"]

        for method in valid_methods:
            # Test as list
            mae_list = MeanAbsoluteError(aggregation_method=[method])
            assert mae_list.aggregation_method == [method]

            # Test as string
            mae_str = MeanAbsoluteError(aggregation_method=method)
            assert mae_str.aggregation_method == method

    def test_validation_valid_combinations_accepted(self):
        """Test that valid combinations are accepted."""
        valid_combinations = [
            ["timewise"],
            ["componentwise"],
            ["groupwise"],
            ["timewise", "componentwise"],
            ["timewise", "groupwise"],
            ["componentwise", "groupwise"],
            ["timewise", "componentwise", "groupwise"],
        ]

        for combination in valid_combinations:
            mae = MeanAbsoluteError(aggregation_method=combination)
            assert mae.aggregation_method == combination

    def test_validation_all_string_accepted(self):
        """Test that 'all' string is accepted."""
        mae = MeanAbsoluteError(aggregation_method="all")
        assert mae.aggregation_method == "all"

    def test_validation_default_is_all(self):
        """Test that default aggregation_method is 'all'."""
        mae = MeanAbsoluteError()
        assert mae.aggregation_method == "all"


class TestGroupwiseAggregation:
    """Tests for groupwise aggregation on panel data."""

    @pytest.fixture
    def panel_multivariate(self):
        """Panel data with 2 groups × 2 members."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "g1__val_a": [10.0, 20.0, 30.0],
            "g1__val_b": [100.0, 200.0, 300.0],
            "g2__val_a": [12.0, 22.0, 32.0],
            "g2__val_b": [110.0, 210.0, 310.0],
        })
        y_pred = pl.DataFrame({
            "observed_time": [datetime(2019, 12, 31)] * 3,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            # g1__val_a errors: |10-11|=1, |20-19|=1, |30-31|=1 → all 1.0
            "g1__val_a": [11.0, 19.0, 31.0],
            # g1__val_b errors: |100-103|=3, |200-197|=3, |300-303|=3 → all 3.0
            "g1__val_b": [103.0, 197.0, 303.0],
            # g2__val_a errors: |12-17|=5, |22-17|=5, |32-37|=5 → all 5.0
            "g2__val_a": [17.0, 17.0, 37.0],
            # g2__val_b errors: |110-117|=7, |210-203|=7, |310-317|=7 → all 7.0
            "g2__val_b": [117.0, 203.0, 317.0],
        })
        return y_true, y_pred

    def test_groupwise_only_returns_per_component_dataframe(self, panel_multivariate):
        """Groupwise alone returns per-component per-timestep DataFrame."""
        y_true, y_pred = panel_multivariate
        mae = MeanAbsoluteError(aggregation_method="groupwise")
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)

        assert isinstance(result, pl.DataFrame)
        # Columns: time + unprefixed component names
        assert "time" in result.columns
        assert "val_a" in result.columns
        assert "val_b" in result.columns
        assert len(result.columns) == 3
        assert len(result) == 3  # 3 timesteps

    def test_groupwise_only_values_are_group_means(self, panel_multivariate):
        """Groupwise averages errors across groups (g1, g2) for each component."""
        y_true, y_pred = panel_multivariate
        mae = MeanAbsoluteError(aggregation_method="groupwise")
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)

        # val_a: mean(g1=1.0, g2=5.0) = 3.0 at every timestep
        assert result["val_a"].to_list() == [3.0, 3.0, 3.0]
        # val_b: mean(g1=3.0, g2=7.0) = 5.0 at every timestep
        assert result["val_b"].to_list() == [5.0, 5.0, 5.0]

    def test_groupwise_with_panel_group_weight(self, panel_multivariate):
        """Groupwise respects panel_group_weight."""
        y_true, y_pred = panel_multivariate
        mae = MeanAbsoluteError(
            aggregation_method="groupwise",
            panel_group_weight={"g1": 3.0, "g2": 1.0},
        )
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)

        # val_a: weighted mean = (1.0*3 + 5.0*1) / 4 = 8/4 = 2.0
        assert result["val_a"].to_list() == [2.0, 2.0, 2.0]
        # val_b: weighted mean = (3.0*3 + 7.0*1) / 4 = 16/4 = 4.0
        assert result["val_b"].to_list() == [4.0, 4.0, 4.0]

    def test_groupwise_timewise_returns_single_row(self, panel_multivariate):
        """Groupwise + timewise → single row with component means."""
        y_true, y_pred = panel_multivariate
        mae = MeanAbsoluteError(aggregation_method=["groupwise", "timewise"])
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1
        assert "val_a" in result.columns
        assert "val_b" in result.columns
        # val_a mean across time: 3.0, val_b mean across time: 5.0
        assert result["val_a"][0] == 3.0
        assert result["val_b"][0] == 5.0

    def test_groupwise_componentwise_returns_per_timestep_scalar(self, panel_multivariate):
        """Groupwise + componentwise → per-timestep DataFrame with single score col."""
        y_true, y_pred = panel_multivariate
        mae = MeanAbsoluteError(aggregation_method=["groupwise", "componentwise"])
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)

        assert isinstance(result, pl.DataFrame)
        assert "time" in result.columns
        # After groupwise (val_a=3, val_b=5) then componentwise: mean(3,5)=4.0
        assert result["mae"].to_list() == [4.0, 4.0, 4.0]

    def test_groupwise_differs_from_componentwise_on_panel(self):
        """Groupwise and componentwise give different results on panel data."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "g1__val_a": [10.0],
            "g1__val_b": [100.0],
            "g2__val_a": [10.0],
        })
        y_pred = pl.DataFrame({
            "observed_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "g1__val_a": [11.0],  # error 1
            "g1__val_b": [103.0],  # error 3
            "g2__val_a": [15.0],  # error 5
        })

        # Componentwise: g1 mean = (1+3)/2 = 2.0, g2 mean = 5.0
        mae_cw = MeanAbsoluteError(aggregation_method=["componentwise"])
        mae_cw.fit(y_true)
        result_cw = mae_cw.score(y_true, y_pred)
        assert "g1__mae" in result_cw.columns
        assert "g2__mae" in result_cw.columns

        # Groupwise: val_a = mean(1,5) = 3.0, val_b = 3.0 (only g1)
        mae_gw = MeanAbsoluteError(aggregation_method="groupwise")
        mae_gw.fit(y_true)
        result_gw = mae_gw.score(y_true, y_pred)
        assert "val_a" in result_gw.columns
        assert "val_b" in result_gw.columns

    def test_groupwise_interval_scorer(self):
        """Groupwise works for interval scorers on panel data."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "g1__val": [10.0, 20.0],
            "g2__val": [15.0, 25.0],
        })
        y_pred = pl.DataFrame({
            "observed_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "g1__val_lower_0.9": [8.0, 18.0],
            "g1__val_upper_0.9": [12.0, 22.0],
            "g2__val_lower_0.9": [13.0, 23.0],
            "g2__val_upper_0.9": [17.0, 27.0],
        })

        coverage = EmpiricalCoverage(aggregation_method=["groupwise", "coveragewise"])
        coverage.fit(y_true)
        result = coverage.score(y_true, y_pred)

        assert isinstance(result, pl.DataFrame)
        assert "time" in result.columns
        assert "val" in result.columns
        assert len(result) == 2


class TestEquivalence:
    def test_equivalence_all_equals_manual_aggregation_timewise(self, equivalence_simple_data):
        """Test that 'all' equals manual aggregation of timewise results."""
        y_true, y_pred = equivalence_simple_data

        # Get scalar with 'all'
        mae_all = MeanAbsoluteError(aggregation_method="all")
        mae_all.fit(y_true)
        result_all = mae_all.score(y_true, y_pred)

        # Get per-component with timewise, then average manually
        mae_timewise = MeanAbsoluteError(aggregation_method=["timewise"])
        mae_timewise.fit(y_true)
        result_timewise = mae_timewise.score(y_true, y_pred)
        manual_avg = float(result_timewise.mean().to_numpy().mean())

        assert abs(result_all - manual_avg) < 1e-10, "Should be mathematically equivalent"

    def test_equivalence_all_equals_manual_aggregation_componentwise(self, equivalence_simple_data):
        """Test that 'all' equals manual aggregation of componentwise results."""
        y_true, y_pred = equivalence_simple_data

        # Get scalar with 'all'
        mae_all = MeanAbsoluteError(aggregation_method="all")
        mae_all.fit(y_true)
        result_all = mae_all.score(y_true, y_pred)

        # Get per-timestep with componentwise, then average manually
        mae_componentwise = MeanAbsoluteError(aggregation_method=["componentwise"])
        mae_componentwise.fit(y_true)
        result_componentwise = mae_componentwise.score(y_true, y_pred)
        manual_avg = result_componentwise["mae"].mean()

        assert abs(result_all - manual_avg) < 1e-10, "Should be mathematically equivalent"
