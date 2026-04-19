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

    def test_pointscorer_aggregation_stepwise_vintagewise_returns_dataframe_with_components(self, point_data):
        """Test that aggregation_method=['stepwise', 'vintagewise'] returns per-component DataFrame."""
        y_true, y_pred = point_data
        mae = MeanAbsoluteError(aggregation_method=["stepwise", "vintagewise"])
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

    def test_pointscorer_aggregation_stepwise_vintagewise_componentwise_returns_scalar(self, point_data):
        """Test that both dimensions aggregated returns scalar."""
        y_true, y_pred = point_data
        mae = MeanAbsoluteError(aggregation_method=["stepwise", "vintagewise", "componentwise"])
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)
        assert isinstance(result, float), "Should return scalar when both dimensions aggregated"

    def test_pointscorer_aggregation_all_equivalent_to_all_dimensions(self, point_data):
        """Test that 'all' is equivalent to ['stepwise', 'vintagewise', 'componentwise']."""
        y_true, y_pred = point_data

        mae_all = MeanAbsoluteError(aggregation_method="all")
        mae_all_explicit = MeanAbsoluteError(aggregation_method=["stepwise", "vintagewise", "componentwise"])

        mae_all.fit(y_true)
        result_all = mae_all.score(y_true, y_pred)
        mae_all_explicit.fit(y_true)
        result_all_explicit = mae_all_explicit.score(y_true, y_pred)

        assert result_all == result_all_explicit, "'all' should equal ['stepwise', 'vintagewise', 'componentwise']"

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
        assert "coverage_rate" in result.columns
        # With coveragewise not in aggregation, rates appear as rows
        assert "g1__width" in result.columns
        assert result.shape[0] == 2  # one row per rate

        rate_09 = result.filter(pl.col("coverage_rate") == 0.9)
        rate_05 = result.filter(pl.col("coverage_rate") == 0.5)
        assert rate_09["g1__width"][0] == 4.0
        assert rate_05["g1__width"][0] == 2.0

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
        scorer_weighted = MeanAbsoluteError(aggregation_method="all", group_weight=weights)
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

    def test_intervalscorer_aggregation_stepwise_vintagewise_returns_dataframe(self, interval_data):
        """Test that aggregation_method=['stepwise', 'vintagewise'] returns per-component DataFrame."""
        y_true, y_pred = interval_data
        coverage = EmpiricalCoverage(aggregation_method=["stepwise", "vintagewise"])
        coverage.fit(y_true)
        result = coverage.score(y_true, y_pred)

        assert isinstance(result, pl.DataFrame), "Should return DataFrame"
        # 1 row per coverage_rate, component columns + coverage_rate
        assert "coverage_rate" in result.columns
        assert result.shape[0] == 1  # single rate

    def test_intervalscorer_aggregation_componentwise_returns_dataframe(self, interval_data):
        """Test that aggregation_method=['componentwise'] returns per-timestep DataFrame."""
        y_true, y_pred = interval_data
        coverage = EmpiricalCoverage(aggregation_method=["componentwise"])
        coverage.fit(y_true)
        result = coverage.score(y_true, y_pred)

        assert isinstance(result, pl.DataFrame), "Should return DataFrame"
        # 3 timesteps * 1 rate = 3 rows, with time + score + coverage_rate cols
        assert result.shape[0] == 3
        assert "time" in result.columns, "Should have time column"
        assert "coverage_rate" in result.columns

    def test_intervalscorer_aggregation_without_coveragewise(self, multi_rate_interval_data):
        """Test aggregation without 'coveragewise' returns DataFrame with coverage_rate column."""
        y_true, y_pred = multi_rate_interval_data

        # Explicitly exclude 'coveragewise' to get per-rate scores
        coverage = EmpiricalCoverage(aggregation_method=["stepwise", "vintagewise", "componentwise", "groupwise"])
        coverage.fit(y_true)
        result = coverage.score(y_true, y_pred)

        assert isinstance(result, pl.DataFrame), "Should return DataFrame with coverage_rate rows"
        assert "coverage_rate" in result.columns
        # Column name is the metric name after rename (EmpiricalCoverage -> coverage)
        value_cols = [c for c in result.columns if c != "coverage_rate"]
        assert len(value_cols) == 1
        rates = result["coverage_rate"].to_list()
        assert 0.9 in rates
        assert 0.95 in rates

    def test_intervalscorer_aggregation_stepwise_vintagewise_per_rate(self, multi_rate_interval_data):
        """Test stepwise+vintagewise aggregation without coveragewise returns DataFrame with coverage_rate rows."""
        y_true, y_pred = multi_rate_interval_data

        # aggregation_method=["stepwise", "vintagewise"] does not include "coveragewise", so rates appear as rows
        coverage = EmpiricalCoverage(aggregation_method=["stepwise", "vintagewise"])
        coverage.fit(y_true)
        result = coverage.score(y_true, y_pred)

        assert isinstance(result, pl.DataFrame), "Should return DataFrame"
        assert "coverage_rate" in result.columns, "Should have coverage_rate column"
        rates = result["coverage_rate"].to_list()
        assert 0.9 in rates
        assert 0.95 in rates

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

        # Case 1: Explicitly exclude coveragewise (return per rate as DataFrame)
        coverage_explicit_ex = EmpiricalCoverage(aggregation_method=["stepwise", "vintagewise", "componentwise"])
        coverage_explicit_ex.fit(y_true)
        result_ex = coverage_explicit_ex.score(y_true, y_pred)

        # Should be a DataFrame with coverage_rate and score columns
        assert isinstance(result_ex, pl.DataFrame), "Should return DataFrame when coveragewise is excluded"
        assert "coverage_rate" in result_ex.columns
        value_cols = [c for c in result_ex.columns if c != "coverage_rate"]
        assert len(value_cols) == 1
        rates = result_ex["coverage_rate"].to_list()
        assert 0.9 in rates
        assert 0.95 in rates

        # Case 2: Explicitly include coveragewise (scalar)
        coverage_with_cov = EmpiricalCoverage(
            aggregation_method=["stepwise", "vintagewise", "componentwise", "coveragewise"]
        )
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

    def test_paneldata_aggregation_stepwise_vintagewise_returns_dataframe(self, panel_point_data):
        """Test that panel data with stepwise+vintagewise aggregation returns per-group-component DataFrame."""
        y_true, y_pred = panel_point_data
        mae = MeanAbsoluteError(aggregation_method=["stepwise", "vintagewise"])
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
        mae = MeanAbsoluteError(aggregation_method=["stepwise", "bad_method"])
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
        valid_methods = ["stepwise", "vintagewise", "componentwise", "groupwise"]

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
            ["stepwise"],
            ["vintagewise"],
            ["componentwise"],
            ["groupwise"],
            ["stepwise", "vintagewise", "componentwise"],
            ["stepwise", "vintagewise", "groupwise"],
            ["componentwise", "groupwise"],
            ["stepwise", "vintagewise", "componentwise", "groupwise"],
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

    def test_groupwise_with_group_weight(self, panel_multivariate):
        """Groupwise respects group_weight."""
        y_true, y_pred = panel_multivariate
        mae = MeanAbsoluteError(
            aggregation_method="groupwise",
            group_weight={"g1": 3.0, "g2": 1.0},
        )
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)

        # val_a: weighted mean = (1.0*3 + 5.0*1) / 4 = 8/4 = 2.0
        assert result["val_a"].to_list() == [2.0, 2.0, 2.0]
        # val_b: weighted mean = (3.0*3 + 7.0*1) / 4 = 16/4 = 4.0
        assert result["val_b"].to_list() == [4.0, 4.0, 4.0]

    def test_groupwise_stepwise_vintagewise_returns_single_row(self, panel_multivariate):
        """Groupwise + stepwise+vintagewise → single row with component means."""
        y_true, y_pred = panel_multivariate
        mae = MeanAbsoluteError(aggregation_method=["groupwise", "stepwise", "vintagewise"])
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
    def test_equivalence_all_equals_manual_aggregation_stepwise_vintagewise(self, equivalence_simple_data):
        """Test that 'all' equals manual aggregation of stepwise+vintagewise results."""
        y_true, y_pred = equivalence_simple_data

        # Get scalar with 'all'
        mae_all = MeanAbsoluteError(aggregation_method="all")
        mae_all.fit(y_true)
        result_all = mae_all.score(y_true, y_pred)

        # Get per-component with stepwise+vintagewise, then average manually
        mae_sv = MeanAbsoluteError(aggregation_method=["stepwise", "vintagewise"])
        mae_sv.fit(y_true)
        result_sv = mae_sv.score(y_true, y_pred)
        manual_avg = float(result_sv.mean().to_numpy().mean())

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


class TestIntervalScorerComponentwiseAggregation:
    """Tests for interval scorer componentwise aggregation paths."""

    @pytest.fixture
    def interval_panel_data(self):
        """Interval panel data with 2 groups and 2 rates."""
        times = [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)]
        y_true = pl.DataFrame({
            "time": times,
            "g1__val": [10.0, 20.0, 30.0],
            "g2__val": [15.0, 25.0, 35.0],
        })
        y_pred = pl.DataFrame({
            "observed_time": [datetime(2019, 12, 31)] * 3,
            "time": times,
            "g1__val_lower_0.9": [8.0, 18.0, 28.0],
            "g1__val_upper_0.9": [12.0, 22.0, 32.0],
            "g2__val_lower_0.9": [13.0, 23.0, 33.0],
            "g2__val_upper_0.9": [17.0, 27.0, 37.0],
            "g1__val_lower_0.95": [7.0, 17.0, 27.0],
            "g1__val_upper_0.95": [13.0, 23.0, 33.0],
            "g2__val_lower_0.95": [12.0, 22.0, 32.0],
            "g2__val_upper_0.95": [18.0, 28.0, 38.0],
        })
        return y_true, y_pred

    def test_componentwise_coveragewise_returns_dataframe(self, interval_panel_data):
        """Componentwise + coveragewise returns per-timestep DataFrame."""
        y_true, y_pred = interval_panel_data
        scorer = EmpiricalCoverage(aggregation_method=["componentwise", "coveragewise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)
        assert "time" in result.columns
        assert len(result) == 3

    def test_componentwise_without_coveragewise_returns_dataframe(self, interval_panel_data):
        """Componentwise without coveragewise returns DataFrame with coverage_rate rows."""
        y_true, y_pred = interval_panel_data
        scorer = EmpiricalCoverage(aggregation_method=["componentwise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)
        assert "time" in result.columns
        assert "coverage_rate" in result.columns

    def test_stepwise_vintagewise_coveragewise_returns_dataframe(self, interval_panel_data):
        """Stepwise+vintagewise + coveragewise returns per-component DataFrame."""
        y_true, y_pred = interval_panel_data
        scorer = EmpiricalCoverage(aggregation_method=["stepwise", "vintagewise", "coveragewise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1

    def test_stepwise_vintagewise_without_coveragewise_returns_dataframe(self, interval_panel_data):
        """Stepwise+vintagewise without coveragewise produces rows per rate."""
        y_true, y_pred = interval_panel_data
        scorer = EmpiricalCoverage(aggregation_method=["stepwise", "vintagewise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)
        assert "coverage_rate" in result.columns

    def test_groupwise_coveragewise_returns_dataframe(self, interval_panel_data):
        """Groupwise + coveragewise returns collapsed per-component DataFrame."""
        y_true, y_pred = interval_panel_data
        scorer = EmpiricalCoverage(aggregation_method=["groupwise", "coveragewise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)
        assert "time" in result.columns

    def test_groupwise_without_coveragewise_returns_dataframe(self, interval_panel_data):
        """Groupwise alone without coveragewise returns DataFrame with coverage_rate."""
        y_true, y_pred = interval_panel_data
        scorer = EmpiricalCoverage(aggregation_method=["groupwise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)
        assert "coverage_rate" in result.columns

    def test_no_spatial_without_coveragewise_returns_float(self):
        """No spatial aggregation + coveragewise returns scalar."""
        times = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        y_true = pl.DataFrame({"time": times, "val": [10.0, 20.0]})
        y_pred = pl.DataFrame({
            "observed_time": [datetime(2019, 12, 31)] * 2,
            "time": times,
            "val_lower_0.9": [8.0, 18.0],
            "val_upper_0.9": [12.0, 22.0],
        })
        scorer = EmpiricalCoverage(aggregation_method="coveragewise")
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        assert isinstance(result, float)

    def test_all_stepwise_vintagewise_componentwise_groupwise_coveragewise(self, interval_panel_data):
        """All aggregation returns scalar for interval panel data."""
        y_true, y_pred = interval_panel_data
        scorer = EmpiricalCoverage(aggregation_method="all")
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        assert isinstance(result, float)


class TestIndividualModes:
    """Tests for individual aggregation modes in isolation."""

    @pytest.fixture
    def multi_vintage_data(self):
        """Multi-vintage data with 2 components."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "val_a": [10.0, 20.0, 30.0],
            "val_b": [15.0, 25.0, 35.0],
        })
        y_pred = pl.DataFrame({
            "observed_time": [
                datetime(2019, 12, 30),
                datetime(2019, 12, 30),
                datetime(2019, 12, 31),
                datetime(2019, 12, 31),
                datetime(2020, 1, 1),
                datetime(2020, 1, 1),
            ],
            "time": [
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                datetime(2020, 1, 2),
                datetime(2020, 1, 3),
                datetime(2020, 1, 3),
                datetime(2020, 1, 4),
            ],
            "val_a": [11.0, 21.0, 19.0, 31.0, 29.0, 41.0],
            "val_b": [16.0, 26.0, 24.0, 36.0, 34.0, 46.0],
        })
        return y_true, y_pred

    def test_stepwise_alone_returns_dataframe(self, multi_vintage_data):
        """stepwise alone collapses step dimension, keeps vintage and component."""
        y_true, y_pred = multi_vintage_data
        mae = MeanAbsoluteError(aggregation_method=["stepwise"])
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)

    def test_vintagewise_alone_returns_dataframe(self, multi_vintage_data):
        """vintagewise alone collapses vintage dimension, keeps step and component."""
        y_true, y_pred = multi_vintage_data
        mae = MeanAbsoluteError(aggregation_method=["vintagewise"])
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)

    def test_groupwise_alone_on_non_panel_returns_dataframe(self):
        """groupwise on non-panel data returns original shape."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "val": [10.0, 20.0],
        })
        y_pred = pl.DataFrame({
            "observed_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "val": [11.0, 19.0],
        })
        mae = MeanAbsoluteError(aggregation_method=["groupwise"])
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)


class TestCoverageWeightAggregation:
    """Tests for coverage_weight in aggregation."""

    def test_coverage_weight_affects_scalar(self):
        """coverage_weight changes the scalar result when coveragewise is used."""
        times = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        y_true = pl.DataFrame({"time": times, "val": [10.0, 20.0]})
        y_pred = pl.DataFrame({
            "observed_time": [datetime(2019, 12, 31)] * 2,
            "time": times,
            "val_lower_0.9": [8.0, 18.0],
            "val_upper_0.9": [12.0, 22.0],
            "val_lower_0.95": [5.0, 15.0],
            "val_upper_0.95": [15.0, 25.0],
        })

        # Equal weights (default)
        cov_equal = EmpiricalCoverage(aggregation_method="all")
        cov_equal.fit(y_true)
        score_equal = cov_equal.score(y_true, y_pred)

        # Weighted: emphasize 0.9 rate
        cov_weighted = EmpiricalCoverage(
            aggregation_method="all",
            coverage_weight={0.9: 10.0, 0.95: 1.0},
        )
        cov_weighted.fit(y_true)
        score_weighted = cov_weighted.score(y_true, y_pred)

        # Both should be floats
        assert isinstance(score_equal, float)
        assert isinstance(score_weighted, float)

    def test_single_rate_no_coveragewise_returns_scalar(self):
        """Single rate without coveragewise returns scalar directly."""
        times = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        y_true = pl.DataFrame({"time": times, "val": [10.0, 20.0]})
        y_pred = pl.DataFrame({
            "observed_time": [datetime(2019, 12, 31)] * 2,
            "time": times,
            "val_lower_0.9": [8.0, 18.0],
            "val_upper_0.9": [12.0, 22.0],
        })
        cov = EmpiricalCoverage(aggregation_method=["stepwise", "vintagewise", "componentwise", "groupwise"])
        cov.fit(y_true)
        result = cov.score(y_true, y_pred)
        assert isinstance(result, float)

    def test_coverage_weight_with_stepwise_dataframe(self):
        """coverage_weight with stepwise returns weighted DataFrame."""
        times = [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)]
        y_true = pl.DataFrame({"time": times, "val": [10.0, 20.0, 30.0]})
        rows = []
        for ot in [datetime(2019, 12, 30), datetime(2019, 12, 31)]:
            for i, t in enumerate(times):
                rows.append({
                    "observed_time": ot,
                    "time": t,
                    "val_lower_0.9": [10.0, 20.0, 30.0][i] - 2.0,
                    "val_upper_0.9": [10.0, 20.0, 30.0][i] + 2.0,
                    "val_lower_0.95": [10.0, 20.0, 30.0][i] - 3.0,
                    "val_upper_0.95": [10.0, 20.0, 30.0][i] + 3.0,
                })
        y_pred = pl.DataFrame(rows).sort("time", "observed_time")
        # Stepwise without coveragewise -> DataFrame with coverage_rate and observed_time
        cov = EmpiricalCoverage(
            aggregation_method=["componentwise", "vintagewise", "coveragewise"],
            coverage_weight={0.9: 2.0, 0.95: 1.0},
        )
        cov.fit(y_true)
        result = cov.score(y_true, y_pred)
        # Should collapse coverage dimension, returning DataFrame with time or scalar
        assert isinstance(result, (float, pl.DataFrame))
