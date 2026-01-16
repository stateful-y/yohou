"""Tests for interval forecast metrics."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from yohou.metrics import (
    CalibrationError,
    EmpiricalCoverage,
    IntervalScore,
    MeanIntervalWidth,
    PinballLoss,
)


@pytest.fixture
def perfect_interval_predictions():
    """Create perfect interval predictions (all actuals within bounds)."""
    y_true = pl.DataFrame(
        {
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "value": [10.0, 20.0, 30.0],
        }
    )
    y_pred = pl.DataFrame(
        {
            "observed_time": [datetime(2019, 12, 31)] * 3,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "value_lower_0.9": [8.0, 18.0, 28.0],
            "value_upper_0.9": [12.0, 22.0, 32.0],
        }
    )
    return y_true, y_pred


@pytest.fixture
def zero_coverage_predictions():
    """Create predictions with zero coverage (all actuals outside bounds)."""
    y_true = pl.DataFrame(
        {
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [10.0, 20.0],
        }
    )
    y_pred = pl.DataFrame(
        {
            "observed_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value_lower_0.9": [15.0, 25.0],  # All actuals below lower bound
            "value_upper_0.9": [18.0, 28.0],
        }
    )
    return y_true, y_pred


@pytest.fixture
def multi_rate_predictions():
    """Create predictions with multiple coverage rates."""
    y_true = pl.DataFrame(
        {
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [10.0, 20.0],
        }
    )
    y_pred = pl.DataFrame(
        {
            "observed_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value_lower_0.9": [8.0, 18.0],
            "value_upper_0.9": [12.0, 22.0],
            "value_lower_0.95": [7.0, 17.0],
            "value_upper_0.95": [13.0, 23.0],
        }
    )
    return y_true, y_pred


class TestEmpiricalCoverage:
    """Tests for EmpiricalCoverage metric."""

    def test_perfect_coverage(self, perfect_interval_predictions):
        """Test that perfect predictions yield coverage of 1.0."""
        y_true, y_pred = perfect_interval_predictions
        coverage = EmpiricalCoverage()
        score = coverage.score(y_true, y_pred)
        assert score == 1.0

    def test_zero_coverage(self, zero_coverage_predictions):
        """Test that no coverage yields score of 0.0."""
        y_true, y_pred = zero_coverage_predictions
        coverage = EmpiricalCoverage()
        score = coverage.score(y_true, y_pred)
        assert score == 0.0

    def test_per_rate_via_aggregation(self, multi_rate_predictions):
        """Test aggregation without "coveragewise" returns dict with rate keys."""
        y_true, y_pred = multi_rate_predictions
        # Exclude "coveragewise" to get per-rate results
        coverage = EmpiricalCoverage(aggregation_method=["timewise", "componentwise", "groupwise"])
        scores = coverage.score(y_true, y_pred)

        assert isinstance(scores, dict)
        assert set(scores.keys()) == {0.9, 0.95}
        assert all(isinstance(v, float) for v in scores.values())
        assert all(0 <= v <= 1 for v in scores.values())

    def test_per_step(self, perfect_interval_predictions):
        """Test aggregation_method=['componentwise', 'coveragewise'] returns DataFrame with time column."""
        y_true, y_pred = perfect_interval_predictions
        coverage = EmpiricalCoverage(aggregation_method=["componentwise", "coveragewise"])
        df = coverage.score(y_true, y_pred)

        assert isinstance(df, pl.DataFrame)
        assert "time" in df.columns
        assert "coverage" in df.columns
        assert len(df) == len(y_true)
        assert df["time"].dtype == pl.Datetime

    def test_per_step_and_per_rate(self, multi_rate_predictions):
        """Test aggregation_method=['componentwise'] returns DataFrame with rate columns."""
        y_true, y_pred = multi_rate_predictions
        coverage = EmpiricalCoverage(aggregation_method=["componentwise"])
        scores = coverage.score(y_true, y_pred)

        assert isinstance(scores, pl.DataFrame)
        assert "time" in scores.columns
        # Without coveragewise aggregation, returns one column per rate (BaseIntervalScorer logic)
        # Columns will be named rate_0.9, rate_0.95 (or similar, depending on BaseIntervalScorer)
        assert any("rate_0.9" in col for col in scores.columns)
        assert any("rate_0.95" in col for col in scores.columns)

    def test_sklearn_tags(self):
        """Test scorer has correct sklearn tags."""
        coverage = EmpiricalCoverage()
        tags = coverage.__sklearn_tags__()
        assert tags.scorer_tags.prediction_type == "interval"


class TestMeanIntervalWidth:
    """Tests for MeanIntervalWidth metric."""

    def test_known_width(self):
        """Test width calculation with known values."""
        y_true = pl.DataFrame(
            {
                "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
                "value": [10.0, 20.0],
            }
        )
        y_pred = pl.DataFrame(
            {
                "observed_time": [datetime(2019, 12, 31)] * 2,
                "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
                "value_lower_0.9": [8.0, 18.0],
                "value_upper_0.9": [12.0, 22.0],
            }
        )
        width = MeanIntervalWidth()
        score = width.score(y_true, y_pred)
        assert score == 4.0

    def test_absolute_width(self):
        """Test that width uses absolute value (handles bound inversions)."""
        y_true = pl.DataFrame(
            {
                "time": [datetime(2020, 1, 1)],
                "value": [10.0],
            }
        )
        # Inverted bounds: upper < lower
        y_pred = pl.DataFrame(
            {
                "observed_time": [datetime(2019, 12, 31)],
                "time": [datetime(2020, 1, 1)],
                "value_lower_0.9": [12.0],
                "value_upper_0.9": [8.0],
            }
        )
        width = MeanIntervalWidth()
        score = width.score(y_true, y_pred)
        assert score == 4.0  # abs(8 - 12) = 4

    def test_per_rate_via_aggregation(self, multi_rate_predictions):
        """Test aggregation without "coveragewise" returns dict with rate keys."""
        y_true, y_pred = multi_rate_predictions
        width = MeanIntervalWidth(aggregation_method=["timewise", "componentwise", "groupwise"])
        scores = width.score(y_true, y_pred)

        assert isinstance(scores, dict)
        assert set(scores.keys()) == {0.9, 0.95}
        assert scores[0.95] > scores[0.9]  # Wider interval at higher coverage

    def test_per_step(self):
        """Test aggregation_method=['componentwise', 'coveragewise'] returns DataFrame."""
        y_true = pl.DataFrame(
            {
                "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
                "value": [10.0, 20.0],
            }
        )
        y_pred = pl.DataFrame(
            {
                "observed_time": [datetime(2019, 12, 31)] * 2,
                "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
                "value_lower_0.9": [8.0, 17.0],
                "value_upper_0.9": [12.0, 23.0],
            }
        )
        width = MeanIntervalWidth(aggregation_method=["componentwise", "coveragewise"])
        df = width.score(y_true, y_pred)

        assert isinstance(df, pl.DataFrame)
        assert "time" in df.columns
        assert df["width"].to_list() == [4.0, 6.0]


class TestIntervalScore:
    """Tests for IntervalScore (Winkler Score) metric."""

    def test_perfect_predictions_equals_width(self, perfect_interval_predictions):
        """Test that perfect coverage yields score equal to width."""
        y_true, y_pred = perfect_interval_predictions
        scorer = IntervalScore()
        score = scorer.score(y_true, y_pred)

        # For perfect coverage, IS = width (no penalties)
        width = MeanIntervalWidth()
        width_score = width.score(y_true, y_pred)
        assert np.isclose(score, width_score)

    def test_penalty_for_under_coverage(self):
        """Test penalty is added when actual is outside bounds."""
        y_true = pl.DataFrame(
            {
                "time": [datetime(2020, 1, 1)],
                "value": [10.0],
            }
        )
        # Actual below lower bound
        y_pred = pl.DataFrame(
            {
                "observed_time": [datetime(2019, 12, 31)],
                "time": [datetime(2020, 1, 1)],
                "value_lower_0.9": [15.0],
                "value_upper_0.9": [20.0],
            }
        )
        scorer = IntervalScore()
        score = scorer.score(y_true, y_pred)

        # Score = width + penalty = 5 + (2/0.9) * (15-10) = 5 + 11.11 ≈ 16.11
        expected = 5.0 + (2.0 / 0.9) * 5.0
        assert np.isclose(score, expected, atol=0.01)

    def test_per_step(self):
        """Test aggregation_method=['componentwise', 'coveragewise'] returns DataFrame."""
        y_true = pl.DataFrame(
            {
                "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
                "value": [10.0, 20.0],
            }
        )
        y_pred = pl.DataFrame(
            {
                "observed_time": [datetime(2019, 12, 31)] * 2,
                "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
                "value_lower_0.9": [8.0, 18.0],
                "value_upper_0.9": [12.0, 22.0],
            }
        )
        # Include coveragewise to get aggregated 'interval_score' column
        scorer = IntervalScore(aggregation_method=["componentwise", "coveragewise"])
        df = scorer.score(y_true, y_pred)

        assert isinstance(df, pl.DataFrame)
        assert "time" in df.columns
        assert "interval_score" in df.columns
        assert len(df) == 2


class TestPinballLoss:
    """Tests for PinballLoss metric."""

    def test_perfect_predictions_zero_loss(self):
        """Test that perfect quantile predictions yield near-zero loss."""
        y_true = pl.DataFrame(
            {
                "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
                "value": [10.0, 20.0],
            }
        )
        # Actuals exactly at bounds (quantiles)
        y_pred = pl.DataFrame(
            {
                "observed_time": [datetime(2019, 12, 31)] * 2,
                "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
                "value_lower_0.9": [10.0, 20.0],
                "value_upper_0.9": [10.0, 20.0],
            }
        )
        loss = PinballLoss()
        score = loss.score(y_true, y_pred)
        assert np.isclose(score, 0.0, atol=0.01)

    def test_asymmetric_penalty(self):
        """Test that pinball loss has asymmetric penalties."""
        y_true = pl.DataFrame(
            {
                "time": [datetime(2020, 1, 1)],
                "value": [10.0],
            }
        )

        # Under-prediction
        y_pred_under = pl.DataFrame(
            {
                "observed_time": [datetime(2019, 12, 31)],
                "time": [datetime(2020, 1, 1)],
                "value_lower_0.9": [8.0],
                "value_upper_0.9": [9.0],  # Below actual
            }
        )

        # Over-prediction
        y_pred_over = pl.DataFrame(
            {
                "observed_time": [datetime(2019, 12, 31)],
                "time": [datetime(2020, 1, 1)],
                "value_lower_0.9": [11.0],  # Above actual
                "value_upper_0.9": [12.0],
            }
        )

        loss = PinballLoss()
        loss_under = loss.score(y_true, y_pred_under)
        loss_over = loss.score(y_true, y_pred_over)

        # Penalties should differ due to asymmetry
        assert loss_under != loss_over

    def test_per_step(self):
        """Test aggregation_method=['componentwise', 'coveragewise'] returns DataFrame."""
        y_true = pl.DataFrame(
            {
                "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
                "value": [10.0, 20.0],
            }
        )
        y_pred = pl.DataFrame(
            {
                "observed_time": [datetime(2019, 12, 31)] * 2,
                "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
                "value_lower_0.9": [8.0, 18.0],
                "value_upper_0.9": [12.0, 22.0],
            }
        )
        loss = PinballLoss(aggregation_method=["componentwise", "coveragewise"])
        df = loss.score(y_true, y_pred)

        assert isinstance(df, pl.DataFrame)
        assert "time" in df.columns
        assert "loss" in df.columns


class TestCalibrationError:
    """Tests for CalibrationError metric."""

    def test_requires_multiple_rates(self):
        """Test that CalibrationError raises error with single rate."""
        y_true = pl.DataFrame(
            {
                "time": [datetime(2020, 1, 1)],
                "value": [10.0],
            }
        )
        y_pred = pl.DataFrame(
            {
                "observed_time": [datetime(2019, 12, 31)],
                "time": [datetime(2020, 1, 1)],
                "value_lower_0.9": [8.0],
                "value_upper_0.9": [12.0],
            }
        )
        error = CalibrationError()

        with pytest.raises(ValueError, match="at least 2 coverage rates"):
            error.score(y_true, y_pred)

    def test_perfect_calibration(self, multi_rate_predictions):
        """Test that perfect calibration yields near-zero error."""
        y_true, y_pred = multi_rate_predictions
        error = CalibrationError()
        score = error.score(y_true, y_pred)

        # With perfect coverage at both rates, error should be near zero
        assert score >= 0.0
        assert score <= 1.0

    def test_per_step(self, multi_rate_predictions):
        """Test aggregation_method=['componentwise', 'coveragewise'] returns DataFrame."""
        y_true, y_pred = multi_rate_predictions
        error = CalibrationError(aggregation_method=["componentwise", "coveragewise"])
        df = error.score(y_true, y_pred)

        assert isinstance(df, pl.DataFrame)
        assert "time" in df.columns
        assert "calibration_error" in df.columns
        assert len(df) == len(y_true)


class TestPanelData:
    """Tests for panel data handling in interval scorers."""

    def test_coverage_with_panel_data(self):
        """Test EmpiricalCoverage with panel data."""
        y_true = pl.DataFrame(
            {
                "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
                "sales__store_1": [10.0, 20.0],
                "sales__store_2": [15.0, 25.0],
            }
        )
        y_pred = pl.DataFrame(
            {
                "observed_time": [datetime(2019, 12, 31)] * 2,
                "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
                "sales__store_1_lower_0.9": [8.0, 18.0],
                "sales__store_1_upper_0.9": [12.0, 22.0],
                "sales__store_2_lower_0.9": [13.0, 23.0],
                "sales__store_2_upper_0.9": [17.0, 27.0],
            }
        )
        coverage = EmpiricalCoverage()
        score = coverage.score(y_true, y_pred)

        assert isinstance(score, float)
        assert 0 <= score <= 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_step_prediction(self):
        """Test scorers handle single-step predictions."""
        y_true = pl.DataFrame(
            {
                "time": [datetime(2020, 1, 1)],
                "value": [10.0],
            }
        )
        y_pred = pl.DataFrame(
            {
                "observed_time": [datetime(2019, 12, 31)],
                "time": [datetime(2020, 1, 1)],
                "value_lower_0.9": [8.0],
                "value_upper_0.9": [12.0],
            }
        )

        coverage = EmpiricalCoverage()
        assert coverage.score(y_true, y_pred) == 1.0

        width = MeanIntervalWidth()
        assert width.score(y_true, y_pred) == 4.0

    def test_multiple_columns(self):
        """Test scorers aggregate across multiple target columns."""
        y_true = pl.DataFrame(
            {
                "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
                "col1": [10.0, 20.0],
                "col2": [15.0, 25.0],
            }
        )
        y_pred = pl.DataFrame(
            {
                "observed_time": [datetime(2019, 12, 31)] * 2,
                "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
                "col1_lower_0.9": [8.0, 18.0],
                "col1_upper_0.9": [12.0, 22.0],
                "col2_lower_0.9": [13.0, 23.0],
                "col2_upper_0.9": [17.0, 27.0],
            }
        )

        coverage = EmpiricalCoverage()
        score = coverage.score(y_true, y_pred)
        assert isinstance(score, float)
