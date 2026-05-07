"""Tests targeting uncovered code paths in base.py, point.py, and classification.py."""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from conftest import run_checks

from yohou.metrics.base import BaseScorer
from yohou.metrics.point import (
    MeanAbsoluteError,
    MeanDirectionalAccuracy,
    MedianAbsoluteError,
    R2Score,
)
from yohou.metrics.classification import ROCAuC
from yohou.metrics.interval import EmpiricalCoverage
from yohou.testing import _yield_yohou_scorer_checks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_data():
    """Simple 5-row point data with single vintage."""
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]
    y_true = pl.DataFrame({"time": times, "value": [10.0, 20.0, 30.0, 40.0, 50.0]})
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * 5,
        "time": times,
        "value": [12.0, 18.0, 33.0, 38.0, 55.0],
    })
    return y_true, y_pred


@pytest.fixture
def multi_vintage_data():
    """Multi-vintage point data (2 vintages, 3 rows each)."""
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(3)]
    vt1 = datetime(2019, 12, 31)
    vt2 = datetime(2019, 12, 30)
    y_true = pl.DataFrame({"time": times, "value": [10.0, 20.0, 30.0]})
    y_pred = pl.DataFrame({
        "vintage_time": [vt1] * 3 + [vt2] * 3,
        "time": times * 2,
        "value": [12.0, 18.0, 33.0, 11.0, 19.0, 28.0],
    })
    return y_true, y_pred


@pytest.fixture
def class_data():
    """Classification data for ranking scorer tests."""
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]
    y_true = pl.DataFrame({
        "time": times,
        "weather": ["sunny", "rainy", "cloudy", "sunny", "rainy"],
    })
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * 5,
        "time": times,
        "weather_proba_sunny": [0.7, 0.1, 0.2, 0.6, 0.1],
        "weather_proba_rainy": [0.2, 0.8, 0.1, 0.3, 0.8],
        "weather_proba_cloudy": [0.1, 0.1, 0.7, 0.1, 0.1],
    })
    return y_true, y_pred


@pytest.fixture
def multi_vintage_class_data():
    """Multi-vintage classification data."""
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]
    vt1 = datetime(2019, 12, 31)
    vt2 = datetime(2019, 12, 30)
    y_true = pl.DataFrame({
        "time": times,
        "weather": ["sunny", "rainy", "cloudy", "sunny", "rainy"],
    })
    y_pred = pl.DataFrame({
        "vintage_time": [vt1] * 5 + [vt2] * 5,
        "time": times * 2,
        "weather_proba_sunny": [0.7, 0.1, 0.2, 0.6, 0.1, 0.8, 0.2, 0.1, 0.7, 0.05],
        "weather_proba_rainy": [0.2, 0.8, 0.1, 0.3, 0.8, 0.1, 0.7, 0.2, 0.2, 0.85],
        "weather_proba_cloudy": [0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1],
    })
    return y_true, y_pred


# ---------------------------------------------------------------------------
# _reject_weights (line 955): Pattern 2 scorers reject time_weight/step_weight
# ---------------------------------------------------------------------------


class TestRejectWeights:
    """Cover _reject_weights raising TypeError."""

    def test_r2_rejects_time_weight(self, simple_data):
        """R2Score.score() rejects time_weight via _reject_weights."""
        y_true, y_pred = simple_data
        scorer = R2Score()
        scorer.fit(y_true)
        with pytest.raises(TypeError, match="does not support sample weights"):
            scorer.score(y_true, y_pred, time_weight=lambda ts: pl.Series([1.0] * len(ts)))

    def test_mda_rejects_step_weight(self, simple_data):
        """MeanDirectionalAccuracy.score() rejects step_weight."""
        y_true, y_pred = simple_data
        scorer = MeanDirectionalAccuracy()
        scorer.fit(y_true)
        with pytest.raises(TypeError, match="does not support sample weights"):
            scorer.score(y_true, y_pred, step_weight=lambda ts: pl.Series([1.0] * len(ts)))


# ---------------------------------------------------------------------------
# _resolve_vintage_weight_to_context (lines 712-724): Pattern 2 vintage weight
# ---------------------------------------------------------------------------


class TestResolveVintageWeightToContext:
    """Cover _resolve_vintage_weight_to_context for Pattern 2 scorers."""

    def test_r2_with_vintage_weight_callable(self, multi_vintage_data):
        """R2Score supports vintage_weight via _resolve_vintage_weight_to_context."""
        y_true, y_pred = multi_vintage_data
        scorer = R2Score()
        scorer.fit(y_true)
        scorer.set_score_request(vintage_weight=True)

        score_plain = scorer.score(y_true, y_pred)
        score_weighted = scorer.score(
            y_true, y_pred,
            vintage_weight=lambda vt: pl.Series([2.0, 1.0] * (len(vt) // 2 + 1))[:len(vt)],
        )
        assert isinstance(score_plain, float)
        assert isinstance(score_weighted, float)

    def test_r2_vintage_weight_none_context(self, simple_data):
        """R2Score with vintage_weight=None just returns context unchanged."""
        y_true, y_pred = simple_data
        scorer = R2Score()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert isinstance(score, float)

    def test_median_with_vintage_weight(self, multi_vintage_data):
        """MedianAbsoluteError supports vintage_weight."""
        y_true, y_pred = multi_vintage_data
        scorer = MedianAbsoluteError()
        scorer.fit(y_true)
        scorer.set_score_request(vintage_weight=True)

        score_weighted = scorer.score(
            y_true, y_pred,
            vintage_weight={datetime(2019, 12, 31): 2.0, datetime(2019, 12, 30): 1.0},
        )
        assert isinstance(score_weighted, float)


# ---------------------------------------------------------------------------
# _map_per_vintage edge cases (lines 1065, 1079, 1084)
# ---------------------------------------------------------------------------


class TestMapPerVintageEdgeCases:
    """Cover _map_per_vintage None return paths."""

    def test_mda_single_row_returns_zero(self):
        """MDA returns 0.0 for single-row data (line 1513 in point.py)."""
        times = [datetime(2020, 1, 1)]
        y_true = pl.DataFrame({"time": times, "value": [10.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)],
            "time": times,
            "value": [12.0],
        })
        scorer = MeanDirectionalAccuracy()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert score == 0.0

    def test_mda_multi_vintage_one_skipped(self):
        """MDA skips vintages with <2 rows (line 1079 in base.py)."""
        times_3 = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(3)]
        vt1 = datetime(2019, 12, 31)
        vt2 = datetime(2019, 12, 30)

        y_true = pl.DataFrame({"time": times_3, "value": [10.0, 20.0, 30.0]})
        # vt1 covers all 3 times; vt2 covers only the first time (1 row)
        y_pred = pl.DataFrame({
            "vintage_time": [vt1] * 3 + [vt2],
            "time": times_3 + [times_3[0]],
            "value": [12.0, 22.0, 28.0, 11.0],
        })
        scorer = MeanDirectionalAccuracy()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert isinstance(score, float)

    def test_mda_all_vintages_skipped_raises(self):
        """All vintages skipped raises ValueError (line 1084 in base.py)."""
        t1 = datetime(2020, 1, 1)
        t2 = datetime(2020, 1, 2)
        vt1 = datetime(2019, 12, 31)
        vt2 = datetime(2019, 12, 30)

        # 2 time points so y_truth >= 2 rows after expansion, but each
        # vintage only covers one of the times (1 row per vintage slice).
        y_true = pl.DataFrame({"time": [t1, t2], "value": [10.0, 20.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [vt1, vt2],
            "time": [t1, t2],
            "value": [12.0, 21.0],
        })
        scorer = MeanDirectionalAccuracy()
        scorer.fit(y_true)
        with pytest.raises(ValueError, match="All vintage groups were skipped"):
            scorer.score(y_true, y_pred)


# ---------------------------------------------------------------------------
# _validate_parameters: non-string in list (line 819)
# ---------------------------------------------------------------------------


class TestValidateParametersNonStringList:
    """Cover aggregation_method list containing non-string elements."""

    def test_non_string_list_element_raises(self):
        """Non-string element in aggregation_method list raises ValueError."""
        scorer = MeanAbsoluteError(aggregation_method=["stepwise", 42])
        y_true = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "value": [1.0],
        })
        with pytest.raises(ValueError, match="All elements in aggregation_method must be strings"):
            scorer.fit(y_true)


# ---------------------------------------------------------------------------
# _collapse_vintage_dimension with meta_cols (lines 389, 396, 404)
# ---------------------------------------------------------------------------


class TestCollapseVintageDimensionWithMeta:
    """Cover weighted/unweighted vintage collapse paths with metadata columns."""

    def test_unweighted_vintage_collapse_with_forecasting_step(self):
        """Unweighted vintage collapse preserving forecasting_step (line 404)."""
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(3)]
        vt1 = datetime(2019, 12, 31)
        vt2 = datetime(2019, 12, 30)

        y_true = pl.DataFrame({"time": times, "value": [10.0, 20.0, 30.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [vt1] * 3 + [vt2] * 3,
            "time": times * 2,
            "value": [12.0, 18.0, 33.0, 11.0, 19.0, 28.0],
        })
        scorer = MeanAbsoluteError(aggregation_method=["vintagewise", "componentwise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)

    def test_weighted_vintage_collapse_with_forecasting_step(self):
        """Weighted vintage collapse preserving forecasting_step (line 396)."""
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(3)]
        vt1 = datetime(2019, 12, 31)
        vt2 = datetime(2019, 12, 30)

        y_true = pl.DataFrame({"time": times, "value": [10.0, 20.0, 30.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [vt1] * 3 + [vt2] * 3,
            "time": times * 2,
            "value": [12.0, 18.0, 33.0, 11.0, 19.0, 28.0],
        })
        scorer = MeanAbsoluteError(aggregation_method=["vintagewise", "componentwise"])
        scorer.fit(y_true)
        scorer.set_score_request(vintage_weight=True)
        result = scorer.score(
            y_true, y_pred,
            vintage_weight={vt1: 2.0, vt2: 1.0},
        )
        assert isinstance(result, pl.DataFrame)


# ---------------------------------------------------------------------------
# _collapse_vintage_dimension: weighted vintage collapse with meta_cols (lines 396, 404)
# ---------------------------------------------------------------------------


class TestCollapseVintageWeighted:
    """Cover weighted vintage collapse when meta_cols are present."""

    def test_weighted_vintage_collapse_stepwise_vintagewise(self):
        """Weighted vintage collapse with stepwise+vintagewise (lines 389, 396)."""
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(3)]
        vt1 = datetime(2019, 12, 31)
        vt2 = datetime(2019, 12, 30)

        y_true = pl.DataFrame({"time": times, "value": [10.0, 20.0, 30.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [vt1] * 3 + [vt2] * 3,
            "time": times * 2,
            "value": [12.0, 18.0, 33.0, 11.0, 19.0, 28.0],
        })
        # stepwise+vintagewise+componentwise collapses all dimensions to scalar
        scorer = MeanAbsoluteError(aggregation_method=["stepwise", "vintagewise", "componentwise"])
        scorer.fit(y_true)
        scorer.set_score_request(vintage_weight=True)
        result = scorer.score(
            y_true, y_pred,
            vintage_weight={vt1: 2.0, vt2: 1.0},
        )
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _collapse_coverage_rates: coveragewise on interval scorer
# Covers line 231 indirectly (the branch fires for all non-interval paths)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Interval scorer: multi-vintage with coverage_rate meta column
# Covers _collapse_rows_with with coverage_rate (lines 319-320),
# _collapse_rows_with partial collapse with no dim_values (line 342),
# _collapse_vintage_dimension weighted/unweighted with meta_cols (lines 389, 396, 404)
# ---------------------------------------------------------------------------


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

    def test_stepwise_vintagewise_componentwise_no_coveragewise(
        self, multi_vintage_interval_data
    ):
        """Collapse steps+vintages+components but keep coverage_rate (lines 319-320, 404)."""
        y_true, y_pred = multi_vintage_interval_data
        scorer = EmpiricalCoverage(
            aggregation_method=["stepwise", "vintagewise", "componentwise"]
        )
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        # coverage_rate stays because coveragewise is not in dims
        assert isinstance(result, pl.DataFrame)
        assert "coverage_rate" in result.columns

    def test_stepwise_vintagewise_componentwise_weighted(
        self, multi_vintage_interval_data
    ):
        """Weighted vintage collapse with coverage_rate meta column (lines 319-320, 389, 396)."""
        y_true, y_pred = multi_vintage_interval_data
        vt1 = datetime(2019, 12, 31)
        vt2 = datetime(2019, 12, 30)
        scorer = EmpiricalCoverage(
            aggregation_method=["stepwise", "vintagewise", "componentwise"]
        )
        scorer.fit(y_true)
        scorer.set_score_request(vintage_weight=True)
        result = scorer.score(
            y_true, y_pred,
            vintage_weight={vt1: 2.0, vt2: 1.0},
        )
        assert isinstance(result, pl.DataFrame)
        assert "coverage_rate" in result.columns

    def test_vintagewise_componentwise_partial_collapse(
        self, multi_vintage_interval_data
    ):
        """Partial collapse: vintagewise without stepwise, coverage_rate as meta (line 342)."""
        y_true, y_pred = multi_vintage_interval_data
        scorer = EmpiricalCoverage(
            aggregation_method=["vintagewise", "componentwise"]
        )
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)


# ---------------------------------------------------------------------------
# Ranking scorer: _resolve_combined_weights dict warning (lines 2549-2555)
# and multi-vintage path (line 2444)
# ---------------------------------------------------------------------------


class TestRankingScorerCoverage:
    """Cover BaseRankingScorer uncovered paths."""

    def test_roc_auc_resolve_combined_weights_dict_only_skipped(self):
        """_resolve_combined_weights returns None when only dict weights are provided."""
        result = ROCAuC._resolve_combined_weights(
            tw={"group_A": np.array([1.0, 2.0])},
            sw=None,
            n=2,
        )
        assert result is None

    def test_roc_auc_resolve_combined_weights_dict_warns(self):
        """_resolve_combined_weights warns when receiving dict weights (lines 2549-2555)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ROCAuC._resolve_combined_weights(
                tw={"group_A": np.array([1.0, 2.0])},
                sw=np.array([1.0, 2.0]),
                n=2,
            )
        assert len(w) == 1
        assert "Panel-aware (dict) weights are not supported" in str(w[0].message)
        assert result is not None  # sw was still applied

    def test_roc_auc_multi_vintage(self, multi_vintage_class_data):
        """ROCAuC computes across multiple vintages (lines 2441-2457)."""
        y_true, y_pred = multi_vintage_class_data
        scorer = ROCAuC()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert isinstance(score, float)
        assert np.isfinite(score)


# ---------------------------------------------------------------------------
# _yield_yohou_scorer_checks: exercises scorer.py check functions
# (covers lines 56-65, 95-107, 137-155, 186-227, 251-256, 289-313,
#  342-348, 377-383, 414-428, 461-498, 525-548, 580-602)
# ---------------------------------------------------------------------------


class TestScorerChecks:
    """Run _yield_yohou_scorer_checks on representative scorers to cover testing/scorer.py."""

    def test_mae_checks(self, simple_data):
        """Run all yohou scorer checks on MeanAbsoluteError."""
        y_true, y_pred = simple_data
        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        run_checks(scorer, _yield_yohou_scorer_checks(scorer, y_true, y_pred))

    def test_r2_checks(self, simple_data):
        """Run all yohou scorer checks on R2Score (Pattern 2)."""
        y_true, y_pred = simple_data
        scorer = R2Score()
        scorer.fit(y_true)
        run_checks(scorer, _yield_yohou_scorer_checks(scorer, y_true, y_pred))

    def test_mda_checks(self):
        """Run all yohou scorer checks on MeanDirectionalAccuracy."""
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]
        y_true = pl.DataFrame({"time": times, "value": [10.0, 20.0, 30.0, 40.0, 50.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": times,
            "value": [12.0, 22.0, 28.0, 42.0, 48.0],
        })
        scorer = MeanDirectionalAccuracy()
        scorer.fit(y_true)
        run_checks(scorer, _yield_yohou_scorer_checks(scorer, y_true, y_pred))

    def test_roc_auc_checks(self, class_data):
        """Run all yohou scorer checks on ROCAuC (ranking scorer)."""
        y_true, y_pred = class_data
        scorer = ROCAuC()
        scorer.fit(y_true)
        run_checks(scorer, _yield_yohou_scorer_checks(scorer, y_true, y_pred))

    def test_empirical_coverage_checks(self):
        """Run all yohou scorer checks on EmpiricalCoverage (interval scorer)."""
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]
        y_true = pl.DataFrame({"time": times, "value": [10.0, 20.0, 30.0, 40.0, 50.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": times,
            "value_lower_0.9": [8.0, 17.0, 27.0, 37.0, 47.0],
            "value_upper_0.9": [14.0, 23.0, 33.0, 43.0, 53.0],
        })
        scorer = EmpiricalCoverage()
        scorer.fit(y_true)
        run_checks(scorer, _yield_yohou_scorer_checks(scorer, y_true, y_pred))

    def test_multivariate_mae_checks(self):
        """Run scorer checks on MAE with multivariate data (component filtering)."""
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]
        y_true = pl.DataFrame({
            "time": times,
            "temp": [10.0, 20.0, 30.0, 40.0, 50.0],
            "wind": [5.0, 10.0, 15.0, 20.0, 25.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": times,
            "temp": [12.0, 18.0, 33.0, 38.0, 55.0],
            "wind": [6.0, 9.0, 14.0, 22.0, 23.0],
        })
        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        run_checks(scorer, _yield_yohou_scorer_checks(scorer, y_true, y_pred))

    def test_panel_mae_checks(self):
        """Run scorer checks on MAE with panel data (group filtering)."""
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]
        y_true = pl.DataFrame({
            "time": times,
            "A__value": [10.0, 20.0, 30.0, 40.0, 50.0],
            "B__value": [100.0, 200.0, 300.0, 400.0, 500.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": times,
            "A__value": [12.0, 18.0, 33.0, 38.0, 55.0],
            "B__value": [110.0, 190.0, 310.0, 390.0, 510.0],
        })
        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        run_checks(scorer, _yield_yohou_scorer_checks(scorer, y_true, y_pred))
