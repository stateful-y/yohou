"""Tests for interval forecast metrics."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from conftest import run_checks as _run_checks_base
from yohou.metrics import (
    CalibrationError,
    ContinuousRankedProbabilityScore,
    EmpiricalCoverage,
    IntervalScore,
    MeanIntervalWidth,
    PinballLoss,
    get_scorer,
    make_scorer,
)
from yohou.metrics.interval import _pinball_loss, _trapezoidal_weights
from yohou.testing import _yield_yohou_scorer_checks
from yohou.weighting import LookupWeighter


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

    def test_crps_systematic(self, multi_rate_predictions):
        """Run systematic checks for ContinuousRankedProbabilityScore."""
        y_truth, y_pred = multi_rate_predictions
        scorer = ContinuousRankedProbabilityScore()
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

    def test_empirical_coverage_coverage_rates_filter(self, multi_rate_predictions):
        """coverage_rates=[0.9] scores only the 0.9-rate columns of a multi-rate y_pred."""
        y_true, y_pred = multi_rate_predictions

        filtered = EmpiricalCoverage(coverage_rates=[0.9])
        filtered.fit(y_true)
        filtered_score = filtered.score(y_true, y_pred)

        # Scoring an unfiltered scorer on a y_pred that contains only the 0.9
        # columns must give the identical result.
        y_pred_only_09 = y_pred.select(["vintage_time", "time", "value_lower_0.9", "value_upper_0.9"])
        reference = EmpiricalCoverage()
        reference.fit(y_true)
        reference_score = reference.score(y_true, y_pred_only_09)

        assert filtered_score == pytest.approx(reference_score)


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

    def test_interval_score_raises_for_zero_coverage_rate(self):
        """Test IntervalScore raises ValueError for coverage_rate=0."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "value": [10.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "value_lower_0.0": [10.0],
            "value_upper_0.0": [10.0],
        })
        scorer = IntervalScore(coverage_rates=[0.0])
        scorer.fit(y_true)
        with pytest.raises(ValueError, match="coverage_rate=0"):
            scorer.score(y_true, y_pred)

    def test_interval_score_time_weighter_multi_rate(self):
        """time_weighter changes the score on multi-rate y_pred (n_rates>1 weight tiling).

        Two coverage rates and two timesteps with very different per-step
        interval scores (step 1 covered, step 2 badly under-covered). Up-weighting
        the worse step must change the weighted scalar, exercising the
        ``n_rates > 1`` tiling path in ``_apply_weights``.
        """
        times = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        y_true = pl.DataFrame({"time": times, "value": [10.0, 20.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": times,
            "value_lower_0.9": [8.0, 30.0],
            "value_upper_0.9": [12.0, 35.0],
            "value_lower_0.95": [7.0, 31.0],
            "value_upper_0.95": [13.0, 36.0],
        })

        unweighted = IntervalScore()
        unweighted.fit(y_true)
        unweighted_score = unweighted.score(y_true, y_pred)

        weighted = IntervalScore(time_weighter=LookupWeighter(mapping={times[0]: 1.0, times[1]: 5.0}))
        weighted.fit(y_true)
        weighted_score = weighted.score(y_true, y_pred)

        assert weighted_score != pytest.approx(unweighted_score)


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

    def test_calibration_error_matches_coverage_deviation(self):
        """Score equals mean over rates of |empirical_coverage - rate|.

        Five timesteps, identical interval [0, 2] at both rates; three of five
        truths fall inside, so empirical coverage is 0.6 at each rate.
        CalibError = (|0.6 - 0.5| + |0.6 - 0.9|) / 2 = 0.2. The buggy per-row
        formula mean(|indicator - rate|) returns 0.46 instead.
        """
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]
        y_true = pl.DataFrame({"time": times, "value": [1.0, 1.0, 1.0, 100.0, 100.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": times,
            "value_lower_0.5": [0.0] * 5,
            "value_upper_0.5": [2.0] * 5,
            "value_lower_0.9": [0.0] * 5,
            "value_upper_0.9": [2.0] * 5,
        })
        error = CalibrationError(coverage_rates=[0.5, 0.9])
        error.fit(y_true)
        score = error.score(y_true, y_pred)

        assert score == pytest.approx(0.2)

    def test_calibration_error_perfect_calibration_is_zero(self):
        """Empirical coverage equal to the nominal rate at every rate yields exactly 0.

        Ten timesteps with nested intervals: [0, 4.5] covers 5/10 (rate 0.5) and
        [0, 8.5] covers 9/10 (rate 0.9), so coverage matches nominal at both rates.
        """
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)]
        y_true = pl.DataFrame({"time": times, "value": [float(i) for i in range(10)]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 10,
            "time": times,
            "value_lower_0.5": [0.0] * 10,
            "value_upper_0.5": [4.5] * 10,
            "value_lower_0.9": [0.0] * 10,
            "value_upper_0.9": [8.5] * 10,
        })
        error = CalibrationError(coverage_rates=[0.5, 0.9])
        error.fit(y_true)

        assert error.score(y_true, y_pred) == pytest.approx(0.0)

    def test_calibration_error_per_component_coverage(self):
        """Components retained: each component's error uses its own empirical coverage.

        Component ``a`` is covered in 3/5 rows (coverage 0.6 -> error 0.2);
        component ``b`` is covered in all 5 rows (coverage 1.0 -> error
        (|1 - 0.5| + |1 - 0.9|) / 2 = 0.3). Aggregation collapses every
        dimension except components.
        """
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]
        y_true = pl.DataFrame({
            "time": times,
            "a": [1.0, 1.0, 1.0, 100.0, 100.0],
            "b": [1.0, 1.0, 1.0, 1.0, 1.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": times,
            "a_lower_0.5": [0.0] * 5,
            "a_upper_0.5": [2.0] * 5,
            "a_lower_0.9": [0.0] * 5,
            "a_upper_0.9": [2.0] * 5,
            "b_lower_0.5": [0.0] * 5,
            "b_upper_0.5": [2.0] * 5,
            "b_lower_0.9": [0.0] * 5,
            "b_upper_0.9": [2.0] * 5,
        })
        error = CalibrationError(
            aggregation_method=["stepwise", "vintagewise", "groupwise", "coveragewise"],
            coverage_rates=[0.5, 0.9],
        )
        error.fit(y_true)
        df = error.score(y_true, y_pred)

        assert isinstance(df, pl.DataFrame)
        row = df.to_dicts()[0]
        assert row["a"] == pytest.approx(0.2)
        assert row["b"] == pytest.approx(0.3)


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


class TestTrapezoidalWeights:
    def test_weights_sum_to_one(self):
        """Trapezoidal weights sum to 1.0 for various rate sets."""
        for rates in [[0.5, 0.9], [0.5, 0.9, 0.95], [0.1, 0.3, 0.5, 0.7, 0.9]]:
            weights = _trapezoidal_weights(rates)
            assert abs(sum(weights.values()) - 1.0) < 1e-10, f"Weights don't sum to 1 for {rates}"

    def test_known_values(self):
        """Trapezoidal weights match hand-computed values for [0.5, 0.9, 0.95]."""
        weights = _trapezoidal_weights([0.5, 0.9, 0.95])
        assert abs(weights[0.5] - 0.70) < 1e-10
        assert abs(weights[0.9] - 0.225) < 1e-10
        assert abs(weights[0.95] - 0.075) < 1e-10

    def test_two_rates(self):
        """Trapezoidal weights for [0.5, 0.9] are not equal."""
        weights = _trapezoidal_weights([0.5, 0.9])
        assert abs(weights[0.5] - 0.70) < 1e-10
        assert abs(weights[0.9] - 0.30) < 1e-10


class TestContinuousRankedProbabilityScore:
    def test_requires_two_rates(self):
        """CRPS raises ValueError with fewer than 2 coverage rates."""
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
        crps = ContinuousRankedProbabilityScore()
        crps.fit(y_true)
        with pytest.raises(ValueError, match="at least 2 coverage rates"):
            crps.score(y_true, y_pred)

    def test_perfect_predictions_zero_score(self):
        """CRPS is 0 when predictions exactly match truth at all bounds."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [10.0, 20.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value_lower_0.5": [10.0, 20.0],
            "value_upper_0.5": [10.0, 20.0],
            "value_lower_0.9": [10.0, 20.0],
            "value_upper_0.9": [10.0, 20.0],
        })
        crps = ContinuousRankedProbabilityScore()
        crps.fit(y_true)
        assert crps.score(y_true, y_pred) == 0.0

    def test_point_predictions_half_mae(self):
        """CRPS of point predictions (lower=upper) equals half the MAE."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [10.0, 20.0],
        })
        # Point predictions: lower = upper = constant offset from truth
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value_lower_0.5": [12.0, 22.0],
            "value_upper_0.5": [12.0, 22.0],
            "value_lower_0.9": [12.0, 22.0],
            "value_upper_0.9": [12.0, 22.0],
        })
        crps = ContinuousRankedProbabilityScore()
        crps.fit(y_true)
        crps_score = crps.score(y_true, y_pred)
        mae = 2.0  # |10-12| + |20-22| / 2
        assert abs(crps_score - mae / 2) < 1e-10

    def test_wider_intervals_higher_crps(self):
        """Wider intervals produce higher CRPS (monotonicity)."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [10.0, 20.0],
        })
        y_pred_narrow = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value_lower_0.5": [9.5, 19.5],
            "value_upper_0.5": [10.5, 20.5],
            "value_lower_0.9": [9.0, 19.0],
            "value_upper_0.9": [11.0, 21.0],
        })
        y_pred_wide = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value_lower_0.5": [5.0, 15.0],
            "value_upper_0.5": [15.0, 25.0],
            "value_lower_0.9": [0.0, 10.0],
            "value_upper_0.9": [20.0, 30.0],
        })
        crps = ContinuousRankedProbabilityScore()
        crps.fit(y_true)
        score_narrow = crps.score(y_true, y_pred_narrow)
        score_wide = crps.score(y_true, y_pred_wide)
        assert score_wide > score_narrow

    def test_mean_vs_trapezoidal_differ(self, multi_rate_predictions):
        """Mean and trapezoidal integration give different results."""
        y_true, y_pred = multi_rate_predictions
        crps_mean = ContinuousRankedProbabilityScore(integration="mean")
        crps_mean.fit(y_true)
        crps_trap = ContinuousRankedProbabilityScore(integration="trapezoidal")
        crps_trap.fit(y_true)
        assert crps_mean.score(y_true, y_pred) != crps_trap.score(y_true, y_pred)

    def test_hand_computed_value(self):
        """CRPS matches hand-computed value for a simple case."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "value": [10.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)],
            "time": [datetime(2020, 1, 1)],
            "value_lower_0.5": [9.0],
            "value_upper_0.5": [11.0],
            "value_lower_0.9": [8.0],
            "value_upper_0.9": [12.0],
        })
        # y=10, perfectly centered.
        # Rate 0.5: tau_l=0.25, tau_u=0.75
        #   pinball_l = 0.25*(10-9) = 0.25, pinball_u = 0.25*(12-10)... wait
        #   L_lower: tau=0.25, y=10, q=9 → y>=q → 0.25*(10-9)=0.25
        #   L_upper: tau=0.75, y=10, q=11 → y<q → 0.25*(11-10)=0.25
        #   (L_lower + L_upper)/2 = 0.25
        # Rate 0.9: tau_l=0.05, tau_u=0.95
        #   L_lower: tau=0.05, y=10, q=8 → y>=q → 0.05*(10-8)=0.10
        #   L_upper: tau=0.95, y=10, q=12 → y<q → 0.05*(12-10)=0.10
        #   (L_lower + L_upper)/2 = 0.10
        # Mean CRPS = (0.25 + 0.10) / 2 = 0.175
        crps = ContinuousRankedProbabilityScore()
        crps.fit(y_true)
        assert abs(crps.score(y_true, y_pred) - 0.175) < 1e-10

    def test_aggregation_coveragewise_always_collapsed(self):
        """Coveragewise is always collapsed, even with partial aggregation."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [10.0, 20.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value_lower_0.5": [9.0, 19.0],
            "value_upper_0.5": [11.0, 21.0],
            "value_lower_0.9": [8.0, 18.0],
            "value_upper_0.9": [12.0, 22.0],
        })
        crps = ContinuousRankedProbabilityScore(aggregation_method=["stepwise"])
        crps.fit(y_true)
        result = crps.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)
        assert "coverage_rate" not in result.columns

    def test_get_scorer_registry(self):
        """get_scorer('crps') returns ContinuousRankedProbabilityScore."""
        scorer = get_scorer("crps")
        assert isinstance(scorer, ContinuousRankedProbabilityScore)

    def test_make_scorer_with_integration(self):
        """make_scorer passes integration parameter correctly."""
        scorer = make_scorer("crps", integration="trapezoidal")
        assert isinstance(scorer, ContinuousRankedProbabilityScore)
        assert scorer.integration == "trapezoidal"

    def test_coverage_rates_dict_rejected(self):
        """Dict coverage_rates rejected by parameter validation."""
        crps = ContinuousRankedProbabilityScore(coverage_rates={0.5: 1.0, 0.9: 2.0})
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "value": [10.0],
        })
        with pytest.raises(ValueError, match="coverage_rates"):
            crps.fit(y_true)

    def test_panel_data(self):
        """CRPS works with panel data."""
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "sales__store_1": [10.0, 20.0],
            "sales__store_2": [15.0, 25.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "sales__store_1_lower_0.5": [9.0, 19.0],
            "sales__store_1_upper_0.5": [11.0, 21.0],
            "sales__store_1_lower_0.9": [8.0, 18.0],
            "sales__store_1_upper_0.9": [12.0, 22.0],
            "sales__store_2_lower_0.5": [14.0, 24.0],
            "sales__store_2_upper_0.5": [16.0, 26.0],
            "sales__store_2_lower_0.9": [13.0, 23.0],
            "sales__store_2_upper_0.9": [17.0, 27.0],
        })
        crps = ContinuousRankedProbabilityScore()
        crps.fit(y_true)
        score = crps.score(y_true, y_pred)
        assert isinstance(score, float)
        assert score >= 0.0


class TestCRPSTrapezoidalEdgeCases:
    def test_crps_trapezoidal_no_coverage_rate_column(self):
        """_collapse_coverage_rates returns df unchanged when no coverage_rate column."""
        crps = ContinuousRankedProbabilityScore(integration="trapezoidal")
        df = pl.DataFrame({"value": [1.0, 2.0, 3.0]})
        result = crps._collapse_coverage_rates(df)
        assert result.equals(df)

    def test_crps_trapezoidal_empty_dataframe(self):
        """_collapse_coverage_rates drops coverage_rate from empty df."""
        crps = ContinuousRankedProbabilityScore(integration="trapezoidal")
        df = pl.DataFrame({"value": pl.Series([], dtype=pl.Float64), "coverage_rate": pl.Series([], dtype=pl.Float64)})
        result = crps._collapse_coverage_rates(df)
        assert "coverage_rate" not in result.columns
        assert len(result) == 0


class TestCollapseCoverageRatesEdgeCases:
    """Cover _collapse_coverage_rates with interval scorers."""

    def test_interval_all_dims_collapses_coverage(self):
        """EmpiricalCoverage with all dims including coveragewise produces scalar."""
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(3)]
        y_true = pl.DataFrame({"time": times, "value": [10.0, 20.0, 30.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": times,
            "value_lower_0.5": [7.0, 15.0, 25.0],
            "value_upper_0.5": [13.0, 25.0, 35.0],
            "value_lower_0.9": [5.0, 12.0, 22.0],
            "value_upper_0.9": [15.0, 28.0, 38.0],
        })
        scorer = EmpiricalCoverage()
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        assert isinstance(result, float)


@pytest.fixture
def multi_vintage_interval_data():
    """Multi-vintage interval data (2 vintages, 3 rows each, 2 rates)."""
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(3)]
    vt1 = datetime(2019, 12, 31)
    vt2 = datetime(2019, 12, 30)
    y_true = pl.DataFrame({"time": times, "value": [10.0, 20.0, 30.0]})
    y_pred = pl.DataFrame({
        "vintage_time": [vt1] * 3 + [vt2] * 3,
        "time": times * 2,
        "value_lower_0.5": [8.0, 17.0, 27.0, 8.5, 17.5, 27.5],
        "value_upper_0.5": [14.0, 23.0, 33.0, 14.5, 23.5, 33.5],
        "value_lower_0.9": [5.0, 12.0, 22.0, 5.5, 12.5, 22.5],
        "value_upper_0.9": [15.0, 28.0, 38.0, 15.5, 28.5, 38.5],
    })
    return y_true, y_pred


class TestIntervalMultiVintageCoverage:
    """Cover interval scorer paths with multi-vintage data and coverage_rate."""

    def test_stepwise_vintagewise_componentwise_no_coveragewise(self, multi_vintage_interval_data):
        """Collapse steps+vintages+components but keep coverage_rate."""
        y_true, y_pred = multi_vintage_interval_data
        scorer = EmpiricalCoverage(aggregation_method=["stepwise", "vintagewise", "componentwise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)
        assert "coverage_rate" in result.columns

    def test_stepwise_vintagewise_componentwise_weighted(self, multi_vintage_interval_data):
        """Weighted vintage collapse with coverage_rate meta column."""
        y_true, y_pred = multi_vintage_interval_data
        vt1 = datetime(2019, 12, 31)
        vt2 = datetime(2019, 12, 30)
        scorer = EmpiricalCoverage(
            aggregation_method=["stepwise", "vintagewise", "componentwise"],
            vintage_weighter=LookupWeighter(mapping={vt1: 2.0, vt2: 1.0}),
        )
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)
        assert "coverage_rate" in result.columns

    def test_vintagewise_componentwise_partial_collapse(self, multi_vintage_interval_data):
        """Partial collapse: vintagewise without stepwise, coverage_rate as meta."""
        y_true, y_pred = multi_vintage_interval_data
        scorer = EmpiricalCoverage(aggregation_method=["vintagewise", "componentwise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)


class TestPinballLossScalar:
    """_pinball_loss returns a float for scalar inputs (both sides of the quantile)."""

    def test_scalar_above_quantile(self):
        """When y >= q the loss is ``tau * (y - q)``."""
        assert _pinball_loss(0.5, 3.0, 2.0) == pytest.approx(0.5 * (3.0 - 2.0))

    def test_scalar_below_quantile(self):
        """When y < q the loss is ``(1 - tau) * (q - y)``."""
        assert _pinball_loss(0.5, 1.0, 2.0) == pytest.approx(0.5 * (2.0 - 1.0))


class TestIntervalScorerWeighterSignatureConsistency:
    """CalibrationError and CRPS must expose the same weighter params as siblings.

    Every other ``BaseIntervalScorer`` (EmpiricalCoverage, IntervalScore, ...)
    accepts ``time_weighter``, ``step_weighter``, and ``vintage_weighter``.
    These two previously omitted all three from their constructors, breaking
    the cross-family signature contract.
    """

    @pytest.mark.parametrize("scorer_cls", [CalibrationError, ContinuousRankedProbabilityScore])
    def test_accepts_and_stores_weighters(self, scorer_cls):
        """Constructor accepts and stores time/step/vintage weighters as-is."""
        tw = LookupWeighter(mapping={}, default=1.0)
        sw = LookupWeighter(mapping={}, default=1.0)
        vw = LookupWeighter(mapping={}, default=1.0)
        scorer = scorer_cls(time_weighter=tw, step_weighter=sw, vintage_weighter=vw)
        assert scorer.time_weighter is tw
        assert scorer.step_weighter is sw
        assert scorer.vintage_weighter is vw

    @pytest.mark.parametrize("scorer_cls", [CalibrationError, ContinuousRankedProbabilityScore])
    def test_weighter_params_in_get_params(self, scorer_cls):
        """time/step/vintage weighters appear in get_params (sklearn introspection)."""
        params = scorer_cls().get_params()
        assert "time_weighter" in params
        assert "step_weighter" in params
        assert "vintage_weighter" in params

    def test_crps_time_weighter_changes_score(self):
        """A time_weighter that up-weights the worse timestep changes the CRPS scalar."""
        times = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        y_true = pl.DataFrame({"time": times, "value": [10.0, 20.0]})
        # Step 1 well-centred, step 2 badly off so the two steps differ strongly.
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": times,
            "value_lower_0.5": [9.0, 30.0],
            "value_upper_0.5": [11.0, 35.0],
            "value_lower_0.9": [8.0, 31.0],
            "value_upper_0.9": [12.0, 36.0],
        })

        unweighted = ContinuousRankedProbabilityScore()
        unweighted.fit(y_true)
        unweighted_score = unweighted.score(y_true, y_pred)

        weighted = ContinuousRankedProbabilityScore(
            time_weighter=LookupWeighter(mapping={times[0]: 1.0, times[1]: 5.0})
        )
        weighted.fit(y_true)
        weighted_score = weighted.score(y_true, y_pred)

        assert weighted_score != pytest.approx(unweighted_score)

    def test_calibration_error_vintage_weighter_changes_score(self):
        """A vintage_weighter that reweights vintages changes the calibration error."""
        times = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        vt1 = datetime(2019, 12, 31)
        vt2 = datetime(2019, 12, 30)
        y_true = pl.DataFrame({"time": times, "value": [10.0, 20.0]})
        # Vintage vt1 covers both rates; vintage vt2 misses, so per-vintage
        # calibration error differs and vintage weighting matters.
        y_pred = pl.DataFrame({
            "vintage_time": [vt1] * 2 + [vt2] * 2,
            "time": times * 2,
            "value_lower_0.9": [8.0, 18.0, 30.0, 40.0],
            "value_upper_0.9": [12.0, 22.0, 35.0, 45.0],
            "value_lower_0.95": [7.0, 17.0, 31.0, 41.0],
            "value_upper_0.95": [13.0, 23.0, 36.0, 46.0],
        })

        unweighted = CalibrationError()
        unweighted.fit(y_true)
        unweighted_score = unweighted.score(y_true, y_pred)

        weighted = CalibrationError(vintage_weighter=LookupWeighter(mapping={vt1: 5.0, vt2: 1.0}))
        weighted.fit(y_true)
        weighted_score = weighted.score(y_true, y_pred)

        assert weighted_score != pytest.approx(unweighted_score)
