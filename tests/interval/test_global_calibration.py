"""Tests for global conformal calibration and its prerequisites."""

import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone

from yohou.interval import SplitConformalForecaster, diagnose_global_calibration
from yohou.interval.similarity import SeasonalSimilarity
from yohou.interval.utils import global_calibration_weights, weighted_quantile
from yohou.metrics.conformity import (
    AbsoluteGammaResidual,
    AbsoluteNormalizedResidual,
    AbsoluteResidual,
    GammaResidual,
    NormalizedResidual,
    Residual,
)
from yohou.point import SeasonalNaive


def _panel(n: int = 220, scales=(1.0, 100.0), noise=(0.5, 0.5)) -> pl.DataFrame:
    """A panel whose entities differ in magnitude, volatility, or both."""
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    data = {"time": dates}
    for idx, (scale, rel) in enumerate(zip(scales, noise, strict=True)):
        rng = np.random.default_rng(idx + 1)
        data[f"e{idx}__v"] = [
            scale * (10.0 + 5.0 * np.sin(2 * np.pi * i / 7)) + rng.normal(0, rel * scale) for i in range(n)
        ]
    return pl.DataFrame(data)


def _fit(y: pl.DataFrame, **kwargs) -> SplitConformalForecaster:
    return SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=7), calibration_size=50, **kwargs).fit(
        y[:200], forecasting_horizon=1, coverage_rates=[0.9]
    )


def _widths(forecaster, columns, rate=0.9):
    iv = forecaster.predict_interval(forecasting_horizon=1, coverage_rates=[rate])
    return {c: float(iv[f"{c}_upper_{rate}"][0] - iv[f"{c}_lower_{rate}"][0]) for c in columns}


class TestDefaultIsUnchanged:
    """The opt-in must not move anything for a user who does not name it."""

    def test_default_is_local(self):
        assert SplitConformalForecaster().calibration_strategy == "local"

    def test_parameter_round_trips(self):
        forecaster = SplitConformalForecaster(calibration_strategy="global")
        assert forecaster.get_params()["calibration_strategy"] == "global"
        assert clone(forecaster).calibration_strategy == "global"

    def test_unnamed_matches_explicit_local(self):
        y = _panel()
        assert _widths(_fit(y), ["e0__v", "e1__v"]) == _widths(
            _fit(y, calibration_strategy="local"), ["e0__v", "e1__v"]
        )


class TestComparabilityTag:
    """Only a dispersion-normalized scorer may calibrate globally."""

    @pytest.mark.parametrize("scorer", [Residual(), AbsoluteResidual(), GammaResidual(), AbsoluteGammaResidual()])
    def test_existing_scorers_are_not_comparable(self, scorer):
        """Nothing becomes poolable by accident."""
        assert scorer.__sklearn_tags__().scorer_tags.supports_global_calibration is False

    @pytest.mark.parametrize("scorer", [NormalizedResidual(), AbsoluteNormalizedResidual()])
    def test_normalized_scorers_declare_comparability(self, scorer):
        assert scorer.__sklearn_tags__().scorer_tags.supports_global_calibration is True

    def test_global_calibration_with_an_incomparable_scorer_raises(self):
        """An error, not a warning: the result would be wrong, not imprecise."""
        with pytest.raises(ValueError, match="supports global calibration"):
            _fit(_panel(), calibration_strategy="global")

    def test_local_calibration_accepts_every_scorer(self):
        """The comparability question does not arise per column."""
        for scorer in (Residual(), GammaResidual(), NormalizedResidual()):
            _fit(_panel(), conformity_scorer=scorer, calibration_strategy="local")

    def test_the_gate_rests_on_the_declaration_not_the_type(self):
        """A scorer outside the built-in family can opt in."""

        class _MyComparable(Residual):
            def __sklearn_tags__(self):
                tags = super().__sklearn_tags__()
                tags.scorer_tags.supports_global_calibration = True
                return tags

        _fit(_panel(), conformity_scorer=_MyComparable(), calibration_strategy="global")


class TestNormalizedScorer:
    """The scorer that makes global calibration sound."""

    def test_scale_is_fitted_per_column(self):
        y = _panel(scales=(1.0, 100.0))
        scorer = NormalizedResidual().fit(y)
        scales = scorer.column_scales_
        assert set(scales) == {"e0__v", "e1__v"}
        assert scales["e1__v"] / scales["e0__v"] > 10, "the larger column should get the larger scale"

    def test_scores_are_supports_global_calibration(self):
        """A proportionally equal miss in each column scores about the same."""
        y = _panel(scales=(1.0, 100.0))
        scorer = NormalizedResidual().fit(y)
        pred = y[:1].with_columns(pl.col("e0__v") - 1.0, pl.col("e1__v") - 100.0)
        scores = scorer.score(y[:1], pred)
        ratio = abs(float(scores["e1__v"][0]) / float(scores["e0__v"][0]))
        assert 0.5 < ratio < 2.0, f"scores should be on a common footing, got ratio {ratio}"

    def test_scale_is_frozen_at_fit(self):
        y = _panel()
        scorer = NormalizedResidual().fit(y[:100])
        before = dict(scorer.column_scales_)
        scorer.score(y[100:110], y[100:110])
        assert scorer.column_scales_ == before

    def test_degenerate_column_does_not_divide_by_zero(self):
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(20)]
        y = pl.DataFrame({"time": dates, "flat__v": [3.0] * 20})
        scorer = NormalizedResidual().fit(y)
        assert scorer.column_scales_["flat__v"] > 0
        assert np.isfinite(scorer.score(y[:1], y[:1]).drop("time").to_numpy()).all()

    def test_scoring_an_unfitted_column_names_it(self):
        y = _panel()
        scorer = NormalizedResidual().fit(y.select(["time", "e0__v"]))
        with pytest.raises(ValueError, match="e1__v"):
            scorer.score(y[:1], y[:1])


class TestGlobalCalibration:
    """What global calibration does to the emitted interval."""

    def test_global_calibration_keeps_per_column_widths(self):
        """One quantile, but each column's bound uses its own fitted scale."""
        y = _panel(scales=(1.0, 100.0))
        widths = _widths(
            _fit(y, conformity_scorer=NormalizedResidual(), calibration_strategy="global"), ["e0__v", "e1__v"]
        )
        ratio = widths["e1__v"] / widths["e0__v"]
        assert 50 < ratio < 200, f"global-calibration widths should still track each column's scale, got {widths}"

    def test_global_calibration_changes_the_quantile_not_the_axis(self):
        """Both modes emit one lower and one upper bound per column."""
        y = _panel()
        for strategy in ("local", "global"):
            forecaster = _fit(y, conformity_scorer=NormalizedResidual(), calibration_strategy=strategy)
            iv = forecaster.predict_interval(forecasting_horizon=1, coverage_rates=[0.9])
            for column in ("e0__v", "e1__v"):
                assert f"{column}_lower_0.9" in iv.columns
                assert f"{column}_upper_0.9" in iv.columns

    def test_global_calibration_draws_on_more_scores_than_one_column_has(self):
        """The point of the mode: the two strategies do not agree."""
        y = _panel(scales=(1.0, 1.0), noise=(0.3, 1.2))
        local = _widths(_fit(y, conformity_scorer=NormalizedResidual(), calibration_strategy="local"), ["e0__v"])
        global_cal = _widths(_fit(y, conformity_scorer=NormalizedResidual(), calibration_strategy="global"), ["e0__v"])
        assert local["e0__v"] != global_cal["e0__v"]


class TestInverseScoreCalibrationStrategy:
    """``inverse_score`` takes the forecaster's ``calibration_strategy`` vocabulary."""

    def _fitted_scorer_and_frames(self):
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]
        scores = pl.DataFrame({
            "time": dates,
            "e0__v": [-0.3, -0.1, 0.0, 0.1, 0.3],
            "e1__v": [-1.2, -0.4, 0.0, 0.4, 1.2],
        })
        y_pred = pl.DataFrame({"time": [datetime(2020, 2, 1)], "e0__v": [1.0], "e1__v": [1.0]})
        return NormalizedResidual().fit(scores), scores, y_pred

    def test_local_and_global_draw_from_different_scores(self):
        scorer, scores, y_pred = self._fitted_scorer_and_frames()
        local = scorer.inverse_score(y_pred, scores, 0.6, calibration_strategy="local")
        pooled = scorer.inverse_score(y_pred, scores, 0.6, calibration_strategy="global")
        assert local["e0__v_lower_0.6"][0] != pooled["e0__v_lower_0.6"][0]

    def test_unknown_strategy_raises_instead_of_falling_back_to_local(self):
        scorer, scores, y_pred = self._fitted_scorer_and_frames()
        with pytest.raises(ValueError, match="calibration_strategy"):
            scorer.inverse_score(y_pred, scores, 0.6, calibration_strategy="pooled")


class TestGlobalCalibrationWeights:
    """The construction that lets a similarity compose with global calibration."""

    def test_uniform_affinities_reduce_to_the_plain_stacked_quantile(self):
        """The property that separates the correct construction from naive tiling."""
        n_times, n_columns = 40, 3
        n_stacked = n_times * n_columns
        rng = np.random.default_rng(0)
        scores = rng.normal(0, 1, (n_times, n_columns)).reshape(-1)

        uniform = np.full(n_times, 1.0 / (n_times + 1))
        plain = np.full(n_stacked, 1.0 / (n_stacked + 1))
        for alpha in (0.1, 0.05, 0.01):
            assert weighted_quantile(scores, alpha, global_calibration_weights(uniform, n_columns)) == pytest.approx(
                weighted_quantile(scores, alpha, plain)
            )

    def test_reserved_mass_reflects_the_stacked_count(self):
        n_times, n_columns = 40, 3
        uniform = np.full(n_times, 1.0 / (n_times + 1))
        stacked = global_calibration_weights(uniform, n_columns)
        assert 1.0 - stacked.sum() == pytest.approx(1.0 / (n_times * n_columns + 1))

    def test_every_column_at_a_time_shares_that_time_affinity(self):
        n_columns = 3
        raw = np.exp(-np.arange(40) / 8.0)
        weights = raw / (raw.sum() + 1.0)
        stacked = global_calibration_weights(weights, n_columns)
        assert np.allclose(stacked[:n_columns], stacked[0])

    def test_informative_affinities_still_shift_the_quantile(self):
        """Global calibration must not quietly discard the weighting."""
        n_times, n_columns = 40, 3
        rng = np.random.default_rng(1)
        scores = rng.normal(0, 1, (n_times, n_columns)).reshape(-1)
        raw = np.exp(-np.arange(n_times) / 5.0)
        concentrated = raw / (raw.sum() + 1.0)
        uniform = np.full(n_times, 1.0 / (n_times + 1))
        assert weighted_quantile(scores, 0.1, global_calibration_weights(concentrated, n_columns)) != pytest.approx(
            weighted_quantile(scores, 0.1, global_calibration_weights(uniform, n_columns))
        )

    def test_similarity_composes_with_global_calibration_end_to_end(self):
        y = _panel()
        forecaster = _fit(
            y,
            conformity_scorer=NormalizedResidual(),
            similarity=SeasonalSimilarity(seasonality=[7.0]),
            calibration_strategy="global",
        )
        widths = _widths(forecaster, ["e0__v", "e1__v"])
        assert all(w > 0 for w in widths.values())


class TestResolutionGuardCountsStackedScores:
    """A rate the stacked set can express must not be reported as unreachable."""

    @staticmethod
    def _many_columns(n_cols: int = 12, n: int = 120) -> pl.DataFrame:
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        data = {"time": dates}
        for c in range(n_cols):
            rng = np.random.default_rng(c)
            data[f"e{c}__v"] = [10.0 + 5.0 * np.sin(2 * np.pi * i / 7) + rng.normal(0, 0.5) for i in range(n)]
        return pl.DataFrame(data)

    def _fit_small(self, **kwargs):
        y = self._many_columns()
        return SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7), calibration_size=25, **kwargs
        ).fit(y[:100], forecasting_horizon=1, coverage_rates=[0.99])

    def test_local_warns_on_a_rate_the_column_cannot_express(self):
        forecaster = self._fit_small(conformity_scorer=NormalizedResidual(), calibration_strategy="local")
        with pytest.warns(UserWarning, match="calibration scores per value column"):
            forecaster.predict_interval(forecasting_horizon=1, coverage_rates=[0.99])

    def test_global_calibration_lifts_the_rate_into_range(self):
        forecaster = self._fit_small(conformity_scorer=NormalizedResidual(), calibration_strategy="global")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            forecaster.predict_interval(forecasting_horizon=1, coverage_rates=[0.99])


class TestGlobalCalibrationDiagnostic:
    """It reports the two figures that decide the choice, and decides nothing."""

    def test_reports_correlation_and_heterogeneity(self):
        forecaster = _fit(_panel(), conformity_scorer=NormalizedResidual())
        report = diagnose_global_calibration(forecaster)
        assert {"cross_sectional_correlation", "effective_gain", "score_heterogeneity"} <= set(report)
        assert report["n_columns"] == 2

    def test_does_not_change_the_forecaster(self):
        y = _panel()
        forecaster = _fit(y, conformity_scorer=NormalizedResidual())
        before = _widths(forecaster, ["e0__v", "e1__v"])
        params_before = forecaster.get_params()
        diagnose_global_calibration(forecaster)
        assert forecaster.get_params() == params_before
        assert _widths(forecaster, ["e0__v", "e1__v"]) == before

    def test_needs_something_to_pool(self):
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(200)]
        rng = np.random.default_rng(0)
        y = pl.DataFrame({
            "time": dates,
            "only": [10 + 5 * np.sin(2 * np.pi * i / 7) + rng.normal(0, 0.5) for i in range(200)],
        })
        forecaster = SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=7), calibration_size=50).fit(
            y[:180], forecasting_horizon=1, coverage_rates=[0.9]
        )
        with pytest.raises(ValueError, match="at least two value columns"):
            diagnose_global_calibration(forecaster)


class TestAdapterStaysPerColumn:
    """Global calibration applies to the quantile, never to the adaptive level."""

    def test_levels_remain_per_column_under_global_calibration(self):
        from yohou.interval import AdaptiveConformalInference

        y = _panel()
        forecaster = _fit(
            y,
            conformity_scorer=NormalizedResidual(),
            adapter=AdaptiveConformalInference(step_size=0.1),
            calibration_strategy="global",
        )
        assert set(forecaster.adapters_["step_1"]) == {"e0__v", "e1__v"}


class TestAbsoluteNormalizedScorer:
    """The symmetric variant, end to end."""

    def test_scores_are_absolute(self):
        y = _panel()
        scorer = AbsoluteNormalizedResidual().fit(y)
        over = scorer.score(y[:1], y[:1].with_columns(pl.col("e0__v") + 5.0, pl.col("e1__v") + 500.0))
        under = scorer.score(y[:1], y[:1].with_columns(pl.col("e0__v") - 5.0, pl.col("e1__v") - 500.0))
        assert (over.drop("time").to_numpy() >= 0).all()
        assert (under.drop("time").to_numpy() >= 0).all()

    def test_intervals_are_symmetric_about_the_prediction(self):
        """Its whole point: equidistant bounds, each scaled by its column."""
        y = _panel(scales=(1.0, 100.0))
        forecaster = _fit(y, conformity_scorer=AbsoluteNormalizedResidual())
        iv = forecaster.predict_interval(forecasting_horizon=1, coverage_rates=[0.9])
        point = forecaster.point_forecaster_.predict(forecasting_horizon=1).drop("vintage_time", strict=False)
        for column in ("e0__v", "e1__v"):
            lower = float(iv[f"{column}_lower_0.9"][0])
            upper = float(iv[f"{column}_upper_0.9"][0])
            centre = float(point[column][0])
            assert centre - lower == pytest.approx(upper - centre)

    def test_widths_still_track_each_column(self):
        y = _panel(scales=(1.0, 100.0))
        widths = _widths(_fit(y, conformity_scorer=AbsoluteNormalizedResidual()), ["e0__v", "e1__v"])
        assert 50 < widths["e1__v"] / widths["e0__v"] < 200

    def test_global_calibration_works_with_the_symmetric_variant(self):
        y = _panel(scales=(1.0, 100.0))
        widths = _widths(
            _fit(y, conformity_scorer=AbsoluteNormalizedResidual(), calibration_strategy="global"),
            ["e0__v", "e1__v"],
        )
        assert all(w > 0 for w in widths.values())
        assert 50 < widths["e1__v"] / widths["e0__v"] < 200


def test_global_calibration_weights_accepts_rows_without_reserved_mass():
    """A weight row that already sums to 1 or more carries no reservation to undo.

    Nothing in the library produces such a row, since `_reserve_mass` always
    holds some back, but the helper is public and should not divide by a
    negative remainder if handed raw affinities.
    """
    raw = np.ones(10)
    stacked = global_calibration_weights(raw, 2)

    assert stacked.sum() < 1.0
    assert np.allclose(stacked, stacked[0])
