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


def _level(forecaster, step: int = 1, column: str = "value"):
    """Effective level for one (step, value column) pair.

    Adapter clones are keyed by step and then by value column, so a univariate
    fixture reaches its only adapter through the sole column name.
    """
    return forecaster.adapters_[f"step_{step}"][column].predict()


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
        before = _level(forecaster)
        forecaster.observe(new)
        assert _level(forecaster) != before
        forecaster.rewind(new)
        assert _level(forecaster) == before

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
        level = _level(forecaster)[0.9]
        assert isinstance(level, float)

    def test_asymmetric_scorer_two_tails(self):
        forecaster, y = _fit(adapter=AdaptiveConformalInference(), scorer=Residual())
        level = _level(forecaster)[0.9]
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
        assert _level(forecaster)[0.9] != 0.1

    def test_shared_pooling_ties_steps_together(self):
        forecaster, y = _fit(
            adapter=AdaptiveConformalInference(step_size=0.1, alpha_pooling="shared"),
            scorer=AbsoluteResidual(),
        )
        forecaster.observe(y[180:188])
        levels = [_level(forecaster, s)[0.9] for s in (1, 2, 3)]
        assert levels[0] == pytest.approx(levels[1]) == pytest.approx(levels[2])

    def test_shared_pooling_with_asymmetric_scorer(self):
        # Shared pooling must pool each tail across steps for asymmetric scorers.
        forecaster, y = _fit(
            adapter=AdaptiveConformalInference(step_size=0.1, alpha_pooling="shared"),
            scorer=Residual(),
        )
        forecaster.observe(y[180:188])
        levels = [_level(forecaster, s)[0.9] for s in (1, 2, 3)]
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
        levels = {s: _level(forecaster, s)[0.9] for s in (1, 2, 3)}
        # Different horizons generally realize different coverage, so at least
        # one step's level should differ from another.
        assert len({round(v, 6) for v in levels.values()}) > 1

    def test_observe_predict_interval_loop_runs(self):
        forecaster, y = _fit(adapter=AdaptiveConformalInference(step_size=0.1), scorer=AbsoluteResidual())
        out = forecaster.observe_predict_interval(y[180:190], stride=1, coverage_rates=[0.9])
        assert "value_lower_0.9" in out.columns
        assert len(out) > 0


def _two_column_series(n: int = 200) -> pl.DataFrame:
    """Two seasonal panel entities sharing a time index."""
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]

    def series(seed: int) -> list[float]:
        rng = np.random.default_rng(seed)
        return [10.0 + 5.0 * np.sin(2 * np.pi * i / 7) + rng.normal(0, 0.5) for i in range(n)]

    return pl.DataFrame({"time": dates, "a__value": series(1), "b__value": series(2)})


class TestAdapterEntityAxis:
    """Each value column adapts from its own miscoverage.

    Regression coverage for the cross-entity mean: ``_adapter_step_errors`` fed
    ``np.mean(misses)`` over all value columns into a recursion the spec defines
    with a binary indicator, so one chronically miscovered entity dragged every
    other entity's level with it.
    """

    @staticmethod
    def _fit(y: pl.DataFrame) -> SplitConformalForecaster:
        return SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            adapter=AdaptiveConformalInference(step_size=0.1),
        ).fit(y[:180], forecasting_horizon=1, coverage_rates=[0.9])

    def test_one_columns_miscoverage_does_not_affect_the_other(self):
        """Shocking one column must leave the other column's level untouched.

        Compared across two runs rather than before-and-after within one: a
        covered column still advances its own level, because the recursion
        moves by ``step_size * alpha_target`` when the miscoverage indicator is
        zero. The property under test is independence between columns.
        """
        y = _two_column_series()

        def observe_and_read(shock: bool) -> dict[str, object]:
            forecaster = self._fit(y)
            row = y[180:181]
            if shock:
                # "b" lands far outside any calibrated interval; "a" is untouched.
                row = row.with_columns(pl.col("b__value") + 100_000.0)
            forecaster.observe(y=row)
            return {column: adapter.predict()[0.9] for column, adapter in forecaster.adapters_["step_1"].items()}

        quiet, shocked = observe_and_read(shock=False), observe_and_read(shock=True)

        assert shocked["a__value"] == quiet["a__value"], "another column's shock must not reach this column's level"
        assert shocked["b__value"] != quiet["b__value"], "the shocked column's level must respond"

    def test_adapters_are_keyed_by_step_and_column(self):
        y = _two_column_series()
        forecaster = self._fit(y)

        assert set(forecaster.adapters_) == {"step_1"}
        assert set(forecaster.adapters_["step_1"]) == {"a__value", "b__value"}

    def test_a_column_of_a_panel_matches_the_equivalent_single_column_run(self):
        """One column of a multi-column frame must adapt exactly as it would alone."""
        y = _two_column_series()
        panel = self._fit(y)
        univariate = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            adapter=AdaptiveConformalInference(step_size=0.1),
        ).fit(y[:180].select(["time", "a__value"]), forecasting_horizon=1, coverage_rates=[0.9])

        panel.observe(y=y[180:185])
        univariate.observe(y=y[180:185].select(["time", "a__value"]))

        assert (
            panel.adapters_["step_1"]["a__value"].predict()[0.9]
            == univariate.adapters_["step_1"]["a__value"].predict()[0.9]
        )

    def test_shared_pooling_still_lets_columns_diverge(self):
        """``alpha_pooling="shared"`` pools across steps, never across columns."""
        y = _two_column_series()
        forecaster = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            adapter=AdaptiveConformalInference(step_size=0.1, alpha_pooling="shared"),
        ).fit(y[:180], forecasting_horizon=2, coverage_rates=[0.9])

        # Only "b" is shocked, so only "b" should be driven away from "a".
        forecaster.observe(y=y[180:184].with_columns(pl.col("b__value") + 100_000.0))

        levels = {column: adapter.predict()[0.9] for column, adapter in forecaster.adapters_["step_1"].items()}
        assert levels["a__value"] != levels["b__value"], "shared pooling must not fuse the columns"

        # Within a column, the shared trajectory still ties the two steps.
        for column in ("a__value", "b__value"):
            step_1 = forecaster.adapters_["step_1"][column].predict()[0.9]
            step_2 = forecaster.adapters_["step_2"][column].predict()[0.9]
            assert step_1 == pytest.approx(step_2), f"{column} steps should share one trajectory"

    def test_shared_pooling_allocates_one_adapter_per_column(self):
        """Sharing across steps must share the object, not copy it per step."""
        y = _two_column_series()
        forecaster = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            adapter=AdaptiveConformalInference(step_size=0.1, alpha_pooling="shared"),
        ).fit(y[:180], forecasting_horizon=3, coverage_rates=[0.9])

        distinct = {id(a) for step in forecaster.adapters_.values() for a in step.values()}
        assert len(distinct) == 2, "two columns at horizon 3 should need two adapters, not six"
        for column in ("a__value", "b__value"):
            assert forecaster.adapters_["step_1"][column] is forecaster.adapters_["step_3"][column]

    def test_per_step_pooling_allocates_one_adapter_per_step_and_column(self):
        """Deduplication must not merge adapters that are genuinely distinct.

        Asserted on the object count rather than on divergence: a bug that
        merged two adapters would leave the survivors still adapting
        independently, so a divergence check alone would not notice.
        """
        y = _two_column_series()
        forecaster = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            adapter=AdaptiveConformalInference(step_size=0.1, alpha_pooling="per_step"),
        ).fit(y[:180], forecasting_horizon=3, coverage_rates=[0.9])

        distinct = {id(a) for step in forecaster.adapters_.values() for a in step.values()}
        assert len(distinct) == 6, "two columns at horizon 3 should hold six independent adapters"

    def test_shared_adapter_advances_once_per_observed_row(self):
        """The assertion that catches an un-deduplicated observe loop.

        Walking the step keys would advance a shared adapter once per horizon
        step, so a single row would move the level three times at horizon 3.
        That does not raise; it just adapts three times too fast.
        """
        y = _two_column_series()
        forecaster = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            adapter=AdaptiveConformalInference(step_size=0.1, alpha_pooling="shared"),
        ).fit(y[:180], forecasting_horizon=3, coverage_rates=[0.9])

        before = len(forecaster.adapters_["step_1"]["a__value"]._level_history[0.9])
        forecaster.observe(y=y[180:181])
        after = len(forecaster.adapters_["step_1"]["a__value"]._level_history[0.9])

        assert after - before == 1, f"one row should be one update, got {after - before}"

    def test_shared_observe_rewind_round_trip(self):
        y = _two_column_series()
        forecaster = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            adapter=AdaptiveConformalInference(step_size=0.1, alpha_pooling="shared"),
        ).fit(y[:180], forecasting_horizon=3, coverage_rates=[0.9])

        # At least the point forecaster's observation horizon of rows: rewind
        # rebuilds its buffers from the window it is given.
        window = y[180:188]
        before = {c: forecaster.adapters_["step_1"][c].predict()[0.9] for c in ("a__value", "b__value")}
        forecaster.observe(y=window)
        forecaster.rewind(y=window)
        after = {c: forecaster.adapters_["step_1"][c].predict()[0.9] for c in ("a__value", "b__value")}

        assert after == before

    def test_fitted_pooling_axis_is_inspectable(self):
        y = _two_column_series()
        for mode in ("per_step", "shared"):
            forecaster = SplitConformalForecaster(
                point_forecaster=SeasonalNaive(seasonality=7),
                calibration_size=50,
                adapter=AdaptiveConformalInference(alpha_pooling=mode),
            ).fit(y[:180], forecasting_horizon=2, coverage_rates=[0.9])
            assert forecaster.adapter_pooling_ == mode


class TestAdapterPoolingIsDeclared:
    """`alpha_pooling` is part of the contract, not a duck-typed side channel."""

    def test_every_adapter_carries_the_setting(self):
        assert AdaptiveConformalInference().alpha_pooling == "per_step"

    def test_setting_is_addressable_through_the_forecaster(self):
        """The path with no coverage before: search, set_params, and clone."""
        forecaster = SplitConformalForecaster(adapter=AdaptiveConformalInference(alpha_pooling="shared"))

        assert forecaster.get_params(deep=True)["adapter__alpha_pooling"] == "shared"
        forecaster.set_params(adapter__alpha_pooling="per_step")
        assert forecaster.adapter.alpha_pooling == "per_step"
        assert clone(AdaptiveConformalInference(alpha_pooling="shared")).alpha_pooling == "shared"

    def test_an_adapter_without_the_setting_fails_visibly(self):
        """No silent fallback to per-step for an adapter that omits it."""

        class _NoPooling(AdaptiveConformalInference):
            def __init__(self):
                self.step_size = 0.05
                self.epsilon = 0.0

        y = _two_column_series()
        forecaster = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7), calibration_size=50, adapter=_NoPooling()
        )
        with pytest.raises(AttributeError, match="alpha_pooling"):
            forecaster.fit(y[:180], forecasting_horizon=1, coverage_rates=[0.9])
