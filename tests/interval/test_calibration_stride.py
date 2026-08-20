"""Tests for the ``calibration_stride`` parameter of SplitConformalForecaster.

A configured stride restricts the calibration replay to origins
``calibration_stride`` rows apart, tail-anchored on the last calibration row,
while still observing every calibration row. Step ``h`` collects
``C/k - ceil(h/k) + 1`` conformity scores, and fit refuses configurations
whose deepest step falls below ``MIN_STRIDED_SCORES_PER_STEP``.
"""

import math
from datetime import datetime

import polars as pl
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

import yohou.interval.split_conformal as sc_module
from yohou.interval import SplitConformalForecaster
from yohou.interval.split_conformal import MIN_STRIDED_SCORES_PER_STEP
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.preprocessing import LagTransformer


def _series(n: int) -> pl.DataFrame:
    times = pl.datetime_range(
        datetime(2026, 1, 1),
        pl.select(pl.lit(datetime(2026, 1, 1)).dt.offset_by(f"{n - 1}h")).item(),
        interval="1h",
        eager=True,
    )
    return pl.DataFrame({"time": times, "value": [float(i % 24) for i in range(n)]})


def _per_step_counts(fc) -> dict[int, int]:
    counts = fc.conformity_scores_.group_by("step").len().sort("step")
    return dict(zip(counts["step"].to_list(), counts["len"].to_list(), strict=True))


class TestCalibrationStrideReplay:
    @pytest.fixture(autouse=True)
    def _tiny_floor(self, monkeypatch):
        """Lower the score floor so small fixtures exercise the replay, not the guard."""
        monkeypatch.setattr(sc_module, "MIN_STRIDED_SCORES_PER_STEP", 2)

    def test_default_none_keeps_stride_one_counts(self):
        """Without a stride every calibration row scores: C - h + 1 per step."""
        fc = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=24),
            calibration_size=12,
        )
        fc.fit(y=_series(60), forecasting_horizon=4, coverage_rates=[0.5])
        counts = _per_step_counts(fc)
        assert counts == {h: 12 - h + 1 for h in range(1, 5)}

    def test_strided_counts_match_the_formula(self):
        """With stride k, step h collects C/k - ceil(h/k) + 1 scores."""
        fc = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=24),
            calibration_size=12,
            calibration_stride=3,
        )
        fc.fit(y=_series(60), forecasting_horizon=4, coverage_rates=[0.5])
        counts = _per_step_counts(fc)
        assert counts == {h: 12 // 3 - math.ceil(h / 3) + 1 for h in range(1, 5)}

    def test_strided_origins_are_tail_anchored_and_k_apart(self):
        """Step-1 score times sit k rows apart and the block end is covered."""
        y = _series(60)
        fc = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=24),
            calibration_size=12,
            calibration_stride=3,
        )
        fc.fit(y=y, forecasting_horizon=4, coverage_rates=[0.5])
        step1_times = fc.conformity_scores_.filter(pl.col("step") == 1).sort("time")["time"].to_list()
        calib_times = y["time"][-12:].to_list()
        # Origins are train_end, +3, +6, +9 (the +12 origin's targets fall past
        # the block), so step-1 targets are calib rows 0, 3, 6, 9.
        assert step1_times == [calib_times[i] for i in (0, 3, 6, 9)]

    def test_strided_replay_still_observes_the_whole_block(self):
        """The wrapped forecaster ends the replay observed through fit's end."""
        y = _series(60)
        fc = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=24),
            calibration_size=12,
            calibration_stride=4,
        )
        # fit itself runs _check_replay_left_the_block_observed; assert directly too.
        fc.fit(y=y, forecasting_horizon=4, coverage_rates=[0.5])
        assert fc.point_forecaster_.observed_time_ == y["time"].max()

    def test_stride_skips_the_bulk_path(self):
        """A direct reduction stack takes bulk unstrided and batched when strided."""
        point = PointReductionForecaster(
            estimator=LinearRegression(),
            reduction_strategy="direct",
            actual_transformer=LagTransformer(lag=[1, 2]),
        )
        y = _series(80)
        unstrided = SplitConformalForecaster(
            point_forecaster=point,
            calibration_size=12,
        )
        unstrided.fit(y=y, forecasting_horizon=3, coverage_rates=[0.5])
        assert unstrided.replay_path_ == "bulk"

        strided = SplitConformalForecaster(
            point_forecaster=point,
            calibration_size=12,
            calibration_stride=3,
        )
        strided.fit(y=y, forecasting_horizon=3, coverage_rates=[0.5])
        assert strided.replay_path_ == "batched"
        counts = _per_step_counts(strided)
        assert counts == {h: 12 // 3 - math.ceil(h / 3) + 1 for h in range(1, 4)}


class TestCalibrationStrideValidation:
    def test_misaligned_calibration_size_rejected(self):
        fc = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(),
            calibration_size=10,
            calibration_stride=3,
        )
        with pytest.raises(ValueError, match="not a multiple"):
            fc.fit(y=_series(40), forecasting_horizon=2, coverage_rates=[0.5])

    def test_floor_guardrail_names_binding_step_and_passing_size(self):
        """Thin strided calibration raises with the worst step and the fix."""
        fc = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(),
            calibration_size=20,
            calibration_stride=2,
        )
        # worst = 20/2 - ceil(4/2) + 1 = 9; needed = (30 - 1 + 2) * 2 = 62.
        with pytest.raises(ValueError, match=r"floor of 30") as excinfo:
            fc.fit(y=_series(60), forecasting_horizon=4, coverage_rates=[0.5])
        assert "62" in str(excinfo.value)
        assert "3 to 4" in str(excinfo.value)

    def test_nominal_count_clearing_the_floor_is_not_enough(self):
        """The deepest step binds even when C/k itself clears the floor."""
        fc = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(),
            calibration_size=60,
            calibration_stride=2,
        )
        # nominal 60/2 = 30 >= floor, but worst = 30 - 2 + 1 = 29 < 30.
        with pytest.raises(ValueError, match=r"29 conformity scores"):
            fc.fit(y=_series(120), forecasting_horizon=4, coverage_rates=[0.5])

    def test_unstrided_thin_calibration_is_unaffected(self):
        """The floor applies only when a stride is configured."""
        fc = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=24),
            calibration_size=12,
        )
        fc.fit(y=_series(60), forecasting_horizon=4, coverage_rates=[0.5])
        assert fc.replay_path_ in {"bulk", "batched", "rolling"}

    def test_clone_round_trip(self):
        fc = SplitConformalForecaster(calibration_stride=24)
        assert clone(fc).calibration_stride == 24

    def test_floor_constant_is_the_documented_default(self):
        assert MIN_STRIDED_SCORES_PER_STEP == 30


class TestCoverageAwareFloor:
    """The per-step requirement scales with the requested coverage rates.

    The flat floor keeps tail quantiles stable at moderate coverage; the
    coverage-driven validity minimum ``ceil(1/t) - 1`` keeps the tail
    quantile an interior order statistic, with tail mass ``t = 1 - cr``
    for a symmetric conformity scorer and ``(1 - cr) / 2`` for an
    asymmetric one.
    """

    def test_high_coverage_raises_the_requirement_above_the_flat_floor(self):
        """0.99 with a symmetric scorer needs 99 scores, not 30."""
        from yohou.metrics.conformity import AbsoluteResidual

        fc = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(),
            conformity_scorer=AbsoluteResidual(),
            calibration_size=120,
            calibration_stride=2,
        )
        # worst = 120/2 - 2 + 1 = 59: clears the flat floor of 30, but the
        # 0.99 tail (mass 0.01) needs ceil(1/0.01) - 1 = 99 scores.
        with pytest.raises(ValueError, match=r"coverage rate 0\.99") as excinfo:
            fc.fit(y=_series(240), forecasting_horizon=4, coverage_rates=[0.5, 0.99])
        assert "99 scores" in str(excinfo.value)

    def test_high_coverage_passes_with_enough_scores(self):
        """A window sized for the 0.99 tail fits."""
        from yohou.metrics.conformity import AbsoluteResidual

        fc = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=24),
            conformity_scorer=AbsoluteResidual(),
            calibration_size=204,
            calibration_stride=2,
        )
        # worst = 204/2 - 2 + 1 = 101 >= 99.
        fc.fit(y=_series(320), forecasting_horizon=4, coverage_rates=[0.99])
        assert fc.replay_path_ in {"bulk", "batched", "rolling"}

    def test_asymmetric_scorer_doubles_the_tail_requirement(self, monkeypatch):
        """The same coverage rate needs twice the scores under signed residuals."""
        from yohou.metrics.conformity import AbsoluteResidual, Residual

        # Lower the flat floor so only the coverage term decides: the block
        # yields worst = 22/2 - 2 + 1 = 10 scores. Coverage 0.9 needs 9 with a
        # symmetric scorer (tail 0.1) and 19 with an asymmetric one (tail 0.05).
        monkeypatch.setattr(sc_module, "MIN_STRIDED_SCORES_PER_STEP", 2)

        symmetric = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=24),
            conformity_scorer=AbsoluteResidual(),
            calibration_size=22,
            calibration_stride=2,
        )
        symmetric.fit(y=_series(90), forecasting_horizon=4, coverage_rates=[0.9])

        asymmetric = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=24),
            conformity_scorer=Residual(),
            calibration_size=22,
            calibration_stride=2,
        )
        with pytest.raises(ValueError, match="asymmetric") as excinfo:
            asymmetric.fit(y=_series(90), forecasting_horizon=4, coverage_rates=[0.9])
        assert "19 scores" in str(excinfo.value)

    def test_flat_floor_message_survives_when_it_binds(self):
        """Moderate coverage keeps the stability floor as the named bound."""
        fc = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(),
            calibration_size=20,
            calibration_stride=2,
        )
        with pytest.raises(ValueError, match="stability floor of 30"):
            fc.fit(y=_series(60), forecasting_horizon=4, coverage_rates=[0.5])
