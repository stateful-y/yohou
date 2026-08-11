"""Tests for adaptive conformal inference adapters."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from yohou.interval import AdaptiveConformalInference, SplitConformalForecaster
from yohou.interval.similarity import SeasonalSimilarity
from yohou.metrics.conformity import AbsoluteResidual, Residual
from yohou.point import SeasonalNaive


def _series(n: int = 200, seed: int = 42) -> pl.DataFrame:
    """A seasonal daily series with noise."""
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(seed)
    values = [10.0 + 5.0 * np.sin(2 * np.pi * i / 7) + rng.normal(0, 0.5) for i in range(n)]
    return pl.DataFrame({"time": dates, "value": values})


def _fit(adapter=None, scorer=None, similarity=None, horizon=3, calib=50):
    """Fit a SplitConformalForecaster on the first 180 rows of ``_series``."""
    y = _series()
    forecaster = SplitConformalForecaster(
        point_forecaster=SeasonalNaive(seasonality=7),
        calibration_size=calib,
        conformity_scorer=scorer if scorer is not None else Residual(),
        similarity=similarity,
        adapter=adapter,
    )
    forecaster.fit(y[:180], forecasting_horizon=horizon, coverage_rates=[0.9])
    return forecaster, y


def _width(intervals: pl.DataFrame, rate: float = 0.9) -> list[float]:
    """Interval widths for the given coverage rate."""
    return (intervals[f"value_upper_{rate}"] - intervals[f"value_lower_{rate}"]).to_list()


class TestAdaptiveConformalInferenceUnit:
    """The adapter estimator in isolation."""

    def test_fit_seeds_target_level_symmetric(self):
        adapter = AdaptiveConformalInference(step_size=0.1).fit([0.9, 0.8], symmetric=True)
        assert adapter.predict() == {0.9: pytest.approx(0.1), 0.8: pytest.approx(0.2)}

    def test_fit_seeds_half_target_per_tail_asymmetric(self):
        adapter = AdaptiveConformalInference().fit([0.9], symmetric=False)
        lower, upper = adapter.predict()[0.9]
        assert lower == pytest.approx(0.05)
        assert upper == pytest.approx(0.05)

    def test_miscoverage_lowers_level(self):
        adapter = AdaptiveConformalInference(step_size=0.1).fit([0.9], symmetric=True)
        adapter.observe([{0.9: 1.0}])
        assert adapter.predict()[0.9] < 0.1

    def test_coverage_raises_level(self):
        adapter = AdaptiveConformalInference(step_size=0.1).fit([0.9], symmetric=True)
        adapter.observe([{0.9: 0.0}])
        assert adapter.predict()[0.9] > 0.1

    def test_observe_rewind_round_trip(self):
        adapter = AdaptiveConformalInference(step_size=0.1).fit([0.9, 0.8], symmetric=True)
        before = adapter.predict()
        adapter.observe([{0.9: 1.0, 0.8: 0.0}, {0.9: 0.0, 0.8: 1.0}, {0.9: 1.0, 0.8: 1.0}])
        assert adapter.predict() != before
        adapter.rewind(3)
        assert adapter.predict() == before

    def test_rewind_never_drops_below_seed(self):
        adapter = AdaptiveConformalInference(step_size=0.1).fit([0.9], symmetric=True)
        seed = adapter.predict()
        adapter.observe([{0.9: 1.0}])
        adapter.rewind(10)  # more than observed
        assert adapter.predict() == seed

    def test_epsilon_clips_level_floor(self):
        adapter = AdaptiveConformalInference(step_size=0.2, epsilon=0.1).fit([0.9], symmetric=True)
        adapter.observe([{0.9: 1.0}] * 100)
        assert adapter.predict()[0.9] >= 0.1 - 1e-9

    def test_default_epsilon_reaches_zero(self):
        adapter = AdaptiveConformalInference(step_size=0.5).fit([0.9], symmetric=True)
        adapter.observe([{0.9: 1.0}] * 100)
        assert adapter.predict()[0.9] == pytest.approx(0.0)

    def test_asymmetric_tails_adapt_independently(self):
        adapter = AdaptiveConformalInference(step_size=0.1).fit([0.9], symmetric=False)
        # Miss low only: lower level drops, upper level rises.
        adapter.observe([{0.9: (1.0, 0.0)}])
        lower, upper = adapter.predict()[0.9]
        assert lower < 0.05
        assert upper > 0.05

    def test_unfitted_methods_raise(self):
        adapter = AdaptiveConformalInference()
        with pytest.raises(NotFittedError):
            adapter.predict()
        with pytest.raises(NotFittedError):
            adapter.observe([{0.9: 1.0}])
        with pytest.raises(NotFittedError):
            adapter.rewind(1)

    def test_clone_preserves_params(self):
        adapter = AdaptiveConformalInference(step_size=0.2, alpha_pooling="shared", epsilon=0.05)
        assert clone(adapter).get_params() == adapter.get_params()


class TestAdapterInForecaster:
    """The adapter wired into SplitConformalForecaster."""

    def test_adapter_none_is_default(self):
        forecaster, _ = _fit(adapter=None)
        assert not hasattr(forecaster, "adapters_")

    def test_adapter_none_deterministic(self):
        # adapter=None must not perturb the static path.
        f_a, _ = _fit(adapter=None)
        f_b, _ = _fit(adapter=None)
        assert_frame_equal(f_a.predict_interval(), f_b.predict_interval())

    def test_adapter_fits_one_per_step(self):
        forecaster, _ = _fit(adapter=AdaptiveConformalInference(), horizon=3)
        assert set(forecaster.adapters_) == {"step_1", "step_2", "step_3"}

    def test_seeded_adapter_labels_columns_nominally(self):
        forecaster, _ = _fit(adapter=AdaptiveConformalInference())
        cols = forecaster.predict_interval().columns
        assert any(c == "value_lower_0.9" for c in cols)
        assert any(c == "value_upper_0.9" for c in cols)

    def test_observe_adapts_then_rewind_restores(self):
        forecaster, y = _fit(adapter=AdaptiveConformalInference(step_size=0.1))
        new = y[180:188]
        before = forecaster.adapters_["step_1"].predict()
        forecaster.observe(new)
        assert forecaster.adapters_["step_1"].predict() != before
        forecaster.rewind(new)
        assert forecaster.adapters_["step_1"].predict() == before

    def test_untracked_rate_warns_and_falls_back(self):
        forecaster, _ = _fit(adapter=AdaptiveConformalInference())
        with pytest.warns(UserWarning, match="not tracked"):
            intervals = forecaster.predict_interval(coverage_rates=[0.8])
        assert "value_lower_0.8" in intervals.columns

    def test_tracked_rate_does_not_warn(self):
        forecaster, _ = _fit(adapter=AdaptiveConformalInference())
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            forecaster.predict_interval(coverage_rates=[0.9])

    def test_symmetric_scorer_single_level(self):
        forecaster, y = _fit(adapter=AdaptiveConformalInference(), scorer=AbsoluteResidual())
        level = forecaster.adapters_["step_1"].predict()[0.9]
        assert isinstance(level, float)

    def test_asymmetric_scorer_two_tails(self):
        forecaster, y = _fit(adapter=AdaptiveConformalInference(), scorer=Residual())
        level = forecaster.adapters_["step_1"].predict()[0.9]
        assert isinstance(level, tuple) and len(level) == 2

    def test_composes_with_similarity(self):
        forecaster, y = _fit(
            adapter=AdaptiveConformalInference(step_size=0.1),
            scorer=AbsoluteResidual(),
            similarity=SeasonalSimilarity(seasonality=[7.0]),
        )
        # Both add-ons active and the lifecycle runs end to end.
        forecaster.observe(y[180:188])
        intervals = forecaster.predict_interval()
        assert len(intervals) == 3
        assert forecaster.adapters_["step_1"].predict()[0.9] != 0.1

    def test_shared_pooling_ties_steps_together(self):
        forecaster, y = _fit(
            adapter=AdaptiveConformalInference(step_size=0.1, alpha_pooling="shared"),
            scorer=AbsoluteResidual(),
        )
        forecaster.observe(y[180:188])
        levels = [forecaster.adapters_[f"step_{s}"].predict()[0.9] for s in (1, 2, 3)]
        assert levels[0] == pytest.approx(levels[1]) == pytest.approx(levels[2])

    def test_shared_pooling_with_asymmetric_scorer(self):
        # Shared pooling must pool each tail across steps for asymmetric scorers.
        forecaster, y = _fit(
            adapter=AdaptiveConformalInference(step_size=0.1, alpha_pooling="shared"),
            scorer=Residual(),
        )
        forecaster.observe(y[180:188])
        levels = [forecaster.adapters_[f"step_{s}"].predict()[0.9] for s in (1, 2, 3)]
        # Every step shares one (lower, upper) trajectory.
        assert all(isinstance(level, tuple) for level in levels)
        assert levels[0] == pytest.approx(levels[1])
        assert levels[1] == pytest.approx(levels[2])

    def test_per_step_pooling_allows_divergence(self):
        forecaster, y = _fit(
            adapter=AdaptiveConformalInference(step_size=0.1, alpha_pooling="per_step"),
            scorer=AbsoluteResidual(),
        )
        forecaster.observe(y[180:190])
        levels = {s: forecaster.adapters_[f"step_{s}"].predict()[0.9] for s in (1, 2, 3)}
        # Different horizons generally realize different coverage, so at least
        # one step's level should differ from another.
        assert len({round(v, 6) for v in levels.values()}) > 1

    def test_observe_predict_interval_loop_runs(self):
        forecaster, y = _fit(adapter=AdaptiveConformalInference(step_size=0.1), scorer=AbsoluteResidual())
        out = forecaster.observe_predict_interval(y[180:190], stride=1, coverage_rates=[0.9])
        assert "value_lower_0.9" in out.columns
        assert len(out) > 0
