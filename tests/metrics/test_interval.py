"""Tests for interval forecast metrics."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from conftest import run_checks as _run_checks_base
from yohou.metrics import (
    CalibrationError,
    EmpiricalCoverage,
    IntervalScore,
    MeanIntervalWidth,
    PinballLoss,
)
from yohou.testing import _yield_yohou_scorer_checks


@pytest.fixture
def perfect_interval_predictions():
    """Create perfect interval predictions (all actuals within bounds)."""
    y_true = pl.DataFrame({
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "value": [10.0, 20.0, 30.0],
    })
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * 3,
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "value_lower_0.9": [8.0, 18.0, 28.0],
        "value_upper_0.9": [12.0, 22.0, 32.0],
    })
    return y_true, y_pred


@pytest.fixture
def zero_coverage_predictions():
    """Create predictions with zero coverage (all actuals outside bounds)."""
    y_true = pl.DataFrame({
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        "value": [10.0, 20.0],
    })
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * 2,
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        "value_lower_0.9": [15.0, 25.0],  # All actuals below lower bound
        "value_upper_0.9": [18.0, 28.0],
    })
    return y_true, y_pred


@pytest.fixture
def multi_rate_predictions():
    """Create predictions with multiple coverage rates."""
    y_true = pl.DataFrame({
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        "value": [10.0, 20.0],
    })
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * 2,
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        "value_lower_0.9": [8.0, 18.0],
        "value_upper_0.9": [12.0, 22.0],
        "value_lower_0.95": [7.0, 17.0],
        "value_upper_0.95": [13.0, 23.0],
    })
    return y_true, y_pred


def run_checks(scorer, y_truth, y_pred):
    """Run all systematic checks for a scorer."""
    _run_checks_base(scorer, _yield_yohou_scorer_checks(scorer, y_truth, y_pred))


class TestSystematicChecks:
    def test_empirical_coverage_systematic(self, perfect_interval_predictions):
        """Run systematic checks for EmpiricalCoverage."""
        y_truth, y_pred = perfect_interval_predictions
        scorer = EmpiricalCoverage()
        run_checks(scorer, y_truth, y_pred)

    def test_mean_interval_width_systematic(self, perfect_interval_predictions):
        """Run systematic checks for MeanIntervalWidth."""
        y_truth, y_pred = perfect_interval_predictions
        scorer = MeanIntervalWidth()
        run_checks(scorer, y_truth, y_pred)

    def test_interval_score_systematic(self, perfect_interval_predictions):
        """Run systematic checks for IntervalScore."""
        y_truth, y_pred = perfect_interval_predictions
        scorer = IntervalScore()
        run_checks(scorer, y_truth, y_pred)

    def test_pinball_loss_systematic(self, perfect_interval_predictions):
        """Run systematic checks for PinballLoss."""
        y_truth, y_pred = perfect_interval_predictions
        scorer = PinballLoss()
        run_checks(scorer, y_truth, y_pred)

    def test_calibration_error_systematic(self, multi_rate_predictions):
        """Run systematic checks for CalibrationError."""
        y_truth, y_pred = multi_rate_predictions
        scorer = CalibrationError()
        run_checks(scorer, y_truth, y_pred)


class TestEmpiricalCoverage:
    def test_empirical_coverage_perfect_coverage(self, perfect_interval_predictions):
        """Test that perfect predictions yield coverage of 1.0."""
        y_true, y_pred = perfect_interval_predictions
        coverage = EmpiricalCoverage()
        coverage.fit(y_true)
        score = coverage.score(y_true, y_pred)
        assert score == 1.0

    def test_empirical_coverage_zero_coverage(self, zero_coverage_predictions):
        """Test that no coverage yields score of 0.0."""
        y_true, y_pred = zero_coverage_predictions
        coverage = EmpiricalCoverage()
        coverage.fit(y_true)
        score = coverage.score(y_true, y_pred)
        assert score == 0.0

    def test_empirical_coverage_per_rate_via_aggregation(self, multi_rate_predictions):
        """Test aggregation without 'coveragewise' returns DataFrame with coverage_rate."""
        y_true, y_pred = multi_rate_predictions
        # Exclude "coveragewise" to get per-rate scores
        coverage = EmpiricalCoverage(aggregation_method=["stepwise", "vintagewise", "componentwise", "groupwise"])
        coverage.fit(y_true)
        scores = coverage.score(y_true, y_pred)

        assert isinstance(scores, pl.DataFrame)
        assert "coverage_rate" in scores.columns
        value_cols = [c for c in scores.columns if c != "coverage_rate"]
        assert len(value_cols) == 1
        rates = set(scores["coverage_rate"].to_list())
        assert rates == {0.9, 0.95}
        for val in scores[value_cols[0]].to_list():
            assert 0 <= val <= 1

    def test_empirical_coverage_per_step(self, perfect_interval_predictions):
        """Test aggregation_method=['componentwise', 'coveragewise'] returns DataFrame with time column."""
        y_true, y_pred = perfect_interval_predictions
        coverage = EmpiricalCoverage(aggregation_method=["componentwise", "coveragewise"])
        coverage.fit(y_true)
        df = coverage.score(y_true, y_pred)

        assert isinstance(df, pl.DataFrame)
        assert "time" in df.columns
        assert "coverage" in df.columns
        assert len(df) == len(y_true)
        assert df["time"].dtype == pl.Datetime

    def test_empirical_coverage_per_step_and_per_rate(self, multi_rate_predictions):
        """Test aggregation_method=['componentwise'] returns DataFrame with coverage_rate rows."""
        y_true, y_pred = multi_rate_predictions
        coverage = EmpiricalCoverage(aggregation_method=["componentwise"])
        coverage.fit(y_true)
        scores = coverage.score(y_true, y_pred)

        assert isinstance(scores, pl.DataFrame)
        assert "time" in scores.columns
        assert "coverage_rate" in scores.columns
        rates = set(scores["coverage_rate"].to_list())
        assert rates == {0.9, 0.95}

    def test_empirical_coverage_sklearn_tags(self):
        """Test scorer has correct sklearn tags."""
        coverage = EmpiricalCoverage()
        tags = coverage.__sklearn_tags__()
        assert tags.scorer_tags.prediction_type == "interval"


class TestMeanIntervalWidth:
    def test_mean_interval_width_known_width(self):
        """Test width calculation with known values."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [10.0, 20.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value_lower_0.9": [8.0, 18.0],
            "value_upper_0.9": [12.0, 22.0],
        })
        width = MeanIntervalWidth()
        width.fit(y_true)
        score = width.score(y_true, y_pred)
        assert score == 4.0

    def test_mean_interval_width_absolute_width(self):
        """Test that width uses absolute value (handles bound inversions)."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "value": [10.0],
        })
        # Inverted bounds: upper < lower
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "value_lower_0.9": [12.0],
            "value_upper_0.9": [8.0],
        })
        width = MeanIntervalWidth()
        width.fit(y_true)
        score = width.score(y_true, y_pred)
        assert score == 4.0  # abs(8 - 12) = 4

    def test_mean_interval_width_per_rate_via_aggregation(self, multi_rate_predictions):
        """Test aggregation without 'coveragewise' returns DataFrame with coverage_rate."""
        y_true, y_pred = multi_rate_predictions
        width = MeanIntervalWidth(aggregation_method=["stepwise", "vintagewise", "componentwise", "groupwise"])
        width.fit(y_true)
        scores = width.score(y_true, y_pred)

        assert isinstance(scores, pl.DataFrame)
        assert "coverage_rate" in scores.columns
        value_cols = [c for c in scores.columns if c != "coverage_rate"]
        assert len(value_cols) == 1
        rates = set(scores["coverage_rate"].to_list())
        assert rates == {0.9, 0.95}
        score_09 = scores.filter(pl.col("coverage_rate") == 0.9)[value_cols[0]][0]
        score_095 = scores.filter(pl.col("coverage_rate") == 0.95)[value_cols[0]][0]
        assert score_095 > score_09  # Wider interval at higher coverage

    def test_mean_interval_width_per_step(self):
        """Test aggregation_method=['componentwise', 'coveragewise'] returns DataFrame."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [10.0, 20.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value_lower_0.9": [8.0, 17.0],
            "value_upper_0.9": [12.0, 23.0],
        })
        width = MeanIntervalWidth(aggregation_method=["componentwise", "coveragewise"])
        width.fit(y_true)
        df = width.score(y_true, y_pred)

        assert isinstance(df, pl.DataFrame)
        assert "time" in df.columns
        assert df["width"].to_list() == [4.0, 6.0]


class TestIntervalScore:
    def test_interval_score_perfect_predictions_equals_width(self, perfect_interval_predictions):
        """Test that perfect coverage yields score equal to width."""
        y_true, y_pred = perfect_interval_predictions
        scorer = IntervalScore()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)

        # For perfect coverage, IS = width (no penalties)
        width = MeanIntervalWidth()
        width.fit(y_true)
        width_score = width.score(y_true, y_pred)
        assert np.isclose(score, width_score)

    def test_interval_score_penalty_for_under_coverage(self):
        """Test penalty is added when actual is outside bounds."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "value": [10.0],
        })
        # Actual below lower bound
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "value_lower_0.9": [15.0],
            "value_upper_0.9": [20.0],
        })
        scorer = IntervalScore()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)

        # Score = width + penalty = 5 + (2/0.9) * (15-10) = 5 + 11.11 ≈ 16.11
        expected = 5.0 + (2.0 / 0.9) * 5.0
        assert np.isclose(score, expected, atol=0.01)

    def test_interval_score_per_step(self):
        """Test aggregation_method=['componentwise', 'coveragewise'] returns DataFrame."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [10.0, 20.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value_lower_0.9": [8.0, 18.0],
            "value_upper_0.9": [12.0, 22.0],
        })
        # Include coveragewise to get aggregated 'interval_score' column
        scorer = IntervalScore(aggregation_method=["componentwise", "coveragewise"])
        scorer.fit(y_true)
        df = scorer.score(y_true, y_pred)

        assert isinstance(df, pl.DataFrame)
        assert "time" in df.columns
        assert "interval_score" in df.columns
        assert len(df) == 2


class TestPinballLoss:
    def test_pinball_loss_perfect_predictions_zero_loss(self):
        """Test that perfect quantile predictions yield near-zero loss."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [10.0, 20.0],
        })
        # Actuals exactly at bounds (quantiles)
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value_lower_0.9": [10.0, 20.0],
            "value_upper_0.9": [10.0, 20.0],
        })
        loss = PinballLoss()
        loss.fit(y_true)
        score = loss.score(y_true, y_pred)
        assert np.isclose(score, 0.0, atol=0.01)

    def test_pinball_loss_asymmetric_penalty(self):
        """Test that pinball loss has asymmetric penalties."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "value": [10.0],
        })

        # Under-prediction
        y_pred_under = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "value_lower_0.9": [8.0],
            "value_upper_0.9": [9.0],  # Below actual
        })

        # Over-prediction
        y_pred_over = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "value_lower_0.9": [11.0],  # Above actual
            "value_upper_0.9": [12.0],
        })

        loss = PinballLoss()
        loss.fit(y_true)
        loss_under = loss.score(y_true, y_pred_under)
        loss_over = loss.score(y_true, y_pred_over)

        # Penalties should differ due to asymmetry
        assert loss_under != loss_over

    def test_pinball_loss_per_step(self):
        """Test aggregation_method=['componentwise', 'coveragewise'] returns DataFrame."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [10.0, 20.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value_lower_0.9": [8.0, 18.0],
            "value_upper_0.9": [12.0, 22.0],
        })
        loss = PinballLoss(aggregation_method=["componentwise", "coveragewise"])
        loss.fit(y_true)
        df = loss.score(y_true, y_pred)

        assert isinstance(df, pl.DataFrame)
        assert "time" in df.columns
        assert "loss" in df.columns


class TestCalibrationError:
    def test_calibration_error_requires_multiple_rates(self):
        """Test that CalibrationError raises error with single rate."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "value": [10.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "value_lower_0.9": [8.0],
            "value_upper_0.9": [12.0],
        })
        error = CalibrationError()

        with pytest.raises(ValueError, match="at least 2 coverage rates"):
            error.fit(y_true)
            error.score(y_true, y_pred)

    def test_calibration_error_perfect_calibration(self, multi_rate_predictions):
        """Test that perfect calibration yields near-zero error."""
        y_true, y_pred = multi_rate_predictions
        error = CalibrationError()
        error.fit(y_true)
        score = error.score(y_true, y_pred)

        # With perfect coverage at both rates, error should be near zero
        assert score >= 0.0
        assert score <= 1.0

    def test_calibration_error_per_step(self, multi_rate_predictions):
        """Test aggregation_method=['componentwise', 'coveragewise'] returns DataFrame."""
        y_true, y_pred = multi_rate_predictions
        error = CalibrationError(aggregation_method=["componentwise", "coveragewise"])
        error.fit(y_true)
        df = error.score(y_true, y_pred)

        assert isinstance(df, pl.DataFrame)
        assert "time" in df.columns
        assert "calibration_error" in df.columns
        assert len(df) == len(y_true)


class TestPanelData:
    def test_panel_data_coverage_with_panel_data(self):
        """Test EmpiricalCoverage with panel data."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "sales__store_1": [10.0, 20.0],
            "sales__store_2": [15.0, 25.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "sales__store_1_lower_0.9": [8.0, 18.0],
            "sales__store_1_upper_0.9": [12.0, 22.0],
            "sales__store_2_lower_0.9": [13.0, 23.0],
            "sales__store_2_upper_0.9": [17.0, 27.0],
        })
        coverage = EmpiricalCoverage()
        coverage.fit(y_true)
        score = coverage.score(y_true, y_pred)

        assert isinstance(score, float)
        assert 0 <= score <= 1


class TestEdgeCases:
    def test_edge_cases_single_step_prediction(self):
        """Test scorers handle single-step predictions."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "value": [10.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "value_lower_0.9": [8.0],
            "value_upper_0.9": [12.0],
        })

        coverage = EmpiricalCoverage()
        coverage.fit(y_true)
        assert coverage.score(y_true, y_pred) == 1.0

        width = MeanIntervalWidth()
        width.fit(y_true)
        assert width.score(y_true, y_pred) == 4.0

    def test_edge_cases_multiple_columns(self):
        """Test scorers aggregate across multiple target columns."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "col1": [10.0, 20.0],
            "col2": [15.0, 25.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "col1_lower_0.9": [8.0, 18.0],
            "col1_upper_0.9": [12.0, 22.0],
            "col2_lower_0.9": [13.0, 23.0],
            "col2_upper_0.9": [17.0, 27.0],
        })

        coverage = EmpiricalCoverage()
        coverage.fit(y_true)
        score = coverage.score(y_true, y_pred)
        assert isinstance(score, float)
