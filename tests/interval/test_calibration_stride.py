"""Tests for the ``calibration_stride`` parameter of SplitConformalForecaster.

A configured stride restricts the calibration replay to origins
``calibration_stride`` rows apart, tail-anchored on the last calibration row,
while still observing every calibration row. Step ``h`` collects
``C/k - ceil(h/k) + 1`` conformity scores, and fit refuses configurations
whose deepest step falls below ``ceil(MIN_TAIL_SAMPLES / t)`` scores for any
requested coverage rate, ``t`` being that rate's tail mass.
"""

import math
from datetime import datetime

import polars as pl
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

import yohou.interval.split_conformal as sc_module
from yohou.interval import SplitConformalForecaster
from yohou.interval.split_conformal import MIN_TAIL_SAMPLES
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
    def _no_guard(self, monkeypatch):
        """Disable the score guard so small fixtures exercise the replay, not the guard."""
        monkeypatch.setattr(sc_module, "MIN_TAIL_SAMPLES", 0)

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

    def test_guardrail_names_binding_step_and_passing_size(self):
        """Thin strided calibration raises with the worst step and the fix."""
        fc = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(),
            calibration_size=20,
            calibration_stride=2,
        )
        # The default Residual scorer is asymmetric: cr 0.5 has tail mass 0.25,
        # so ceil(3 / 0.25) = 12 scores are required. worst = 20/2 - 2 + 1 = 9;
        # needed = (12 - 1 + 2) * 2 = 26.
        with pytest.raises(ValueError, match=r"required 12") as excinfo:
            fc.fit(y=_series(60), forecasting_horizon=4, coverage_rates=[0.5])
        assert "26" in str(excinfo.value)
        assert "3 to 4" in str(excinfo.value)
        assert "3 tail samples" in str(excinfo.value)

    def test_unstrided_thin_calibration_is_unaffected(self):
        """The requirement applies only when a stride is configured."""
        fc = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=24),
            calibration_size=12,
        )
        fc.fit(y=_series(60), forecasting_horizon=4, coverage_rates=[0.5])
        assert fc.replay_path_ in {"bulk", "batched", "rolling"}

    def test_clone_round_trip(self):
        fc = SplitConformalForecaster(calibration_stride=24)
        assert clone(fc).calibration_stride == 24

    def test_tail_sample_constant_is_the_documented_default(self):
        assert MIN_TAIL_SAMPLES == 3


class TestCoverageScaledRequirement:
    """The per-step requirement scales with the requested coverage rates.

    Each rate requires ``ceil(MIN_TAIL_SAMPLES / t)`` scores, enough for
    the estimated tail to hold ``MIN_TAIL_SAMPLES`` samples, with tail
    mass ``t = 1 - cr`` for a symmetric conformity scorer and
    ``(1 - cr) / 2`` for an asymmetric one. The largest requirement binds.
    """

    def test_high_coverage_scales_the_requirement(self):
        """0.99 with a symmetric scorer needs 300 scores, not a flat count."""
        from yohou.metrics.conformity import AbsoluteResidual

        fc = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(),
            conformity_scorer=AbsoluteResidual(),
            calibration_size=120,
            calibration_stride=2,
        )
        # worst = 120/2 - 2 + 1 = 59: plenty for cr 0.5, but the 0.99 tail
        # (mass 0.01) needs ceil(3/0.01) = 300 scores.
        with pytest.raises(ValueError, match=r"coverage rate 0\.99") as excinfo:
            fc.fit(y=_series(240), forecasting_horizon=4, coverage_rates=[0.5, 0.99])
        assert "300 scores" in str(excinfo.value)

    def test_requirement_boundary_passes(self):
        """A window whose worst step holds exactly ceil(3/t) scores fits."""
        from yohou.metrics.conformity import AbsoluteResidual

        fc = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=24),
            conformity_scorer=AbsoluteResidual(),
            calibration_size=62,
            calibration_stride=2,
        )
        # cr 0.9 symmetric: tail 0.1 needs 30. worst = 62/2 - 2 + 1 = 30.
        fc.fit(y=_series(120), forecasting_horizon=4, coverage_rates=[0.9])
        assert fc.replay_path_ in {"bulk", "batched", "rolling"}

    def test_deepest_step_binds_at_the_boundary(self):
        """The deepest step binds even when C/k itself clears the requirement."""
        from yohou.metrics.conformity import AbsoluteResidual

        fc = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(),
            conformity_scorer=AbsoluteResidual(),
            calibration_size=60,
            calibration_stride=2,
        )
        # nominal 60/2 = 30 clears the cr 0.9 requirement of 30, but
        # worst = 30 - 2 + 1 = 29.
        with pytest.raises(ValueError, match=r"29 conformity scores"):
            fc.fit(y=_series(120), forecasting_horizon=4, coverage_rates=[0.9])

    def test_asymmetric_scorer_doubles_the_requirement(self):
        """The same coverage rate needs twice the scores under signed residuals."""
        from yohou.metrics.conformity import AbsoluteResidual, Residual

        # worst = 80/2 - 2 + 1 = 39. Coverage 0.9 needs 30 with a symmetric
        # scorer (tail 0.1) and 60 with an asymmetric one (tail 0.05).
        symmetric = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=24),
            conformity_scorer=AbsoluteResidual(),
            calibration_size=80,
            calibration_stride=2,
        )
        symmetric.fit(y=_series(160), forecasting_horizon=4, coverage_rates=[0.9])

        asymmetric = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=24),
            conformity_scorer=Residual(),
            calibration_size=80,
            calibration_stride=2,
        )
        with pytest.raises(ValueError, match="asymmetric") as excinfo:
            asymmetric.fit(y=_series(160), forecasting_horizon=4, coverage_rates=[0.9])
        assert "60 scores" in str(excinfo.value)

    def test_median_only_calibration_is_cheap(self):
        """A central quantile needs few scores: no flat floor inflates it."""
        from yohou.metrics.conformity import AbsoluteResidual

        fc = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=24),
            conformity_scorer=AbsoluteResidual(),
            calibration_size=14,
            calibration_stride=2,
        )
        # cr 0.5 symmetric: tail 0.5 needs ceil(3/0.5) = 6. worst = 7 - 2 + 1 = 6.
        fc.fit(y=_series(60), forecasting_horizon=4, coverage_rates=[0.5])
        assert fc.replay_path_ in {"bulk", "batched", "rolling"}
