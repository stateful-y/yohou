"""Tests for sample_weight_alignment strategies and weighter integration in reduction forecasters."""

import inspect
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeClassifier

from base.test_reduction_validation_holdout import WeightRecordingRegressor
from conftest import SampleWeightRecorder
from yohou.base.reduction import BaseReductionForecaster, _StepWeights
from yohou.class_proba import ClassProbaReductionForecaster
from yohou.interval import IntervalReductionForecaster
from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV
from yohou.point import PointReductionForecaster
from yohou.weighting import (
    ExponentialDecayWeighter,
    LinearDecayWeighter,
    LookupWeighter,
    SeasonalEmphasisWeighter,
    TableWeighter,
)


@pytest.fixture
def reduction_data():
    """Create minimal time series data for reduction forecaster tests."""
    n = 50
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    values = [float(i) + np.sin(i * 0.5) for i in range(n)]
    y = pl.DataFrame({"time": times, "value": values})
    return y


def _increasing_table(reduction_data: pl.DataFrame) -> TableWeighter:
    """A TableWeighter giving linearly increasing weights over the data times."""
    frame = pl.DataFrame({
        "time": reduction_data["time"].to_list(),
        "weight": [float(i + 1) for i in range(len(reduction_data))],
    })
    return TableWeighter(frame=frame, on="time")


def _make_forecaster(**kwargs):
    """Create a default PointReductionForecaster with LinearRegression."""
    return PointReductionForecaster(estimator=LinearRegression(), **kwargs)


class TestSampleWeightAlignmentStrategies:
    """Tests for the five sample_weight_alignment strategies."""

    @pytest.mark.parametrize(
        "alignment",
        ["first_step", "mean_step", "weighted_mean_step", "max_weight_step", "min_weight_step"],
    )
    def test_alignment_strategy_produces_fitted_forecaster(self, reduction_data, alignment):
        """Each alignment strategy should produce a fitted forecaster without error."""
        f = _make_forecaster(time_weighter=LinearDecayWeighter(), sample_weight_alignment=alignment)
        f.fit(reduction_data, forecasting_horizon=3)
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3

    def test_different_alignments_produce_different_predictions(self, reduction_data):
        """Different alignment strategies should generally produce different predictions."""
        results = {}
        for alignment in ["first_step", "mean_step", "weighted_mean_step", "max_weight_step", "min_weight_step"]:
            f = _make_forecaster(time_weighter=LinearDecayWeighter(), sample_weight_alignment=alignment)
            f.fit(reduction_data, forecasting_horizon=3)
            pred = f.predict(forecasting_horizon=3)
            results[alignment] = pred["value"].to_numpy()

        pairs_differ = 0
        strategies = list(results.keys())
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i + 1 :]:
                if not np.allclose(results[s1], results[s2]):
                    pairs_differ += 1

        assert pairs_differ > 0, "All alignment strategies produced identical predictions"

    def test_weighted_mean_step_vs_mean_step_differ(self, reduction_data):
        """weighted_mean_step and mean_step should produce different results with non-uniform weights."""
        forecasters = {}
        for alignment in ["mean_step", "weighted_mean_step"]:
            f = _make_forecaster(time_weighter=LinearDecayWeighter(), sample_weight_alignment=alignment)
            f.fit(reduction_data, forecasting_horizon=5)
            forecasters[alignment] = f.predict(forecasting_horizon=5)

        pred_mean = forecasters["mean_step"]["value"].to_numpy()
        pred_wmean = forecasters["weighted_mean_step"]["value"].to_numpy()
        assert not np.allclose(pred_mean, pred_wmean)


class TestTimeWeighterFit:
    """Tests for time_weighter integration with fit()."""

    def test_time_weighter_none_matches_no_weighter(self, reduction_data):
        """Passing time_weighter=None should match omitting it entirely."""
        f1 = _make_forecaster()
        f1.fit(reduction_data, forecasting_horizon=3)
        pred1 = f1.predict(forecasting_horizon=3)

        f2 = _make_forecaster(time_weighter=None)
        f2.fit(reduction_data, forecasting_horizon=3)
        pred2 = f2.predict(forecasting_horizon=3)

        np.testing.assert_array_almost_equal(pred1["value"].to_numpy(), pred2["value"].to_numpy())

    def test_nonuniform_weighter_changes_predictions(self, reduction_data):
        """Non-uniform time_weighter should change predictions vs no weight."""
        f_plain = _make_forecaster()
        f_plain.fit(reduction_data, forecasting_horizon=3)
        pred_plain = f_plain.predict(forecasting_horizon=3)

        f_weighted = _make_forecaster(time_weighter=LinearDecayWeighter())
        f_weighted.fit(reduction_data, forecasting_horizon=3)
        pred_weighted = f_weighted.predict(forecasting_horizon=3)

        assert not np.allclose(pred_plain["value"].to_numpy(), pred_weighted["value"].to_numpy())


class TestTableWeighterFit:
    """Tests for TableWeighter (DataFrame-backed) time weighting in fit()."""

    def test_table_weighter_works(self, reduction_data):
        """A TableWeighter with 'time' and 'weight' columns should work."""
        f = _make_forecaster(time_weighter=_increasing_table(reduction_data))
        f.fit(reduction_data, forecasting_horizon=3)
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3


class TestTimeWeighterValidation:
    """Tests for time_weighter error paths in reduction forecasters."""

    def test_zero_sum_weights_raises(self, reduction_data):
        """Weights summing to zero should raise ValueError."""
        f = _make_forecaster(time_weighter=LookupWeighter(mapping={}, default=0.0))
        with pytest.raises(ValueError, match="zero"):
            f.fit(reduction_data, forecasting_horizon=3)

    def test_estimator_without_sample_weight_raises(self, reduction_data):
        """Estimator that doesn't support sample_weight should raise ValueError."""
        from sklearn.neighbors import KNeighborsRegressor

        f = PointReductionForecaster(estimator=KNeighborsRegressor(), time_weighter=LinearDecayWeighter())
        with pytest.raises(ValueError, match="sample_weight"):
            f.fit(reduction_data, forecasting_horizon=3)


class TestVintageWeighterFit:
    """Tests for vintage_weighter integration with fit()."""

    def test_vintage_weighter_none_matches_no_weighter(self, reduction_data):
        """Passing vintage_weighter=None should match omitting it entirely."""
        f1 = _make_forecaster()
        f1.fit(reduction_data, forecasting_horizon=3)
        pred1 = f1.predict(forecasting_horizon=3)

        f2 = _make_forecaster(vintage_weighter=None)
        f2.fit(reduction_data, forecasting_horizon=3)
        pred2 = f2.predict(forecasting_horizon=3)

        np.testing.assert_array_almost_equal(pred1["value"].to_numpy(), pred2["value"].to_numpy())

    def test_vintage_weighter_works(self, reduction_data):
        """A vintage TableWeighter should produce a fitted forecaster."""
        f = _make_forecaster(vintage_weighter=_increasing_table(reduction_data))
        f.fit(reduction_data, forecasting_horizon=3)
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3


class TestInvalidSampleWeightAlignment:
    """Tests for invalid sample_weight_alignment values."""

    def test_invalid_alignment_raises(self, reduction_data):
        """Invalid sample_weight_alignment should raise an error mentioning the param."""
        f = _make_forecaster(time_weighter=LinearDecayWeighter(), sample_weight_alignment="invalid_strategy")
        with pytest.raises(ValueError, match="sample_weight_alignment"):
            f.fit(reduction_data, forecasting_horizon=3)

    def test_vintage_weighter_dict(self, reduction_data):
        """A LookupWeighter (dict-backed) vintage weighter should work."""
        times = reduction_data["time"].to_list()
        vw = LookupWeighter(mapping={t: float(i + 1) for i, t in enumerate(times)})
        f = _make_forecaster(vintage_weighter=vw)
        f.fit(reduction_data, forecasting_horizon=3)
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)

    def test_vintage_weighter_changes_predictions(self, reduction_data):
        """Non-uniform vintage_weighter should change predictions vs no weight."""
        f_plain = _make_forecaster()
        f_plain.fit(reduction_data, forecasting_horizon=3)
        pred_plain = f_plain.predict(forecasting_horizon=3)

        # Weight only the first 5 observations, zero out the rest.
        times = reduction_data["time"].to_list()
        weights = [1.0 if i < 5 else 0.0 for i in range(len(times))]
        vw = TableWeighter(frame=pl.DataFrame({"time": times, "weight": weights}), on="time")

        f_weighted = _make_forecaster(vintage_weighter=vw)
        f_weighted.fit(reduction_data, forecasting_horizon=3)
        pred_weighted = f_weighted.predict(forecasting_horizon=3)

        assert not np.allclose(pred_plain["value"].to_numpy(), pred_weighted["value"].to_numpy())


class TestCombinedTimeAndVintageWeighter:
    """Tests for combining time_weighter and vintage_weighter."""

    def test_both_weighters_work_together(self, reduction_data):
        """Providing both time_weighter and vintage_weighter should produce a fitted forecaster."""
        f = _make_forecaster(
            time_weighter=LinearDecayWeighter(),
            vintage_weighter=_increasing_table(reduction_data),
        )
        f.fit(reduction_data, forecasting_horizon=3)
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3

    def test_combined_differs_from_individual(self, reduction_data):
        """Combined weights should differ from either individual weight alone."""
        times = reduction_data["time"].to_list()
        weights = [0.01] * len(times)
        weights[-1] = 100.0
        vw = TableWeighter(frame=pl.DataFrame({"time": times, "weight": weights}), on="time")

        f_tw = _make_forecaster(time_weighter=LinearDecayWeighter())
        f_tw.fit(reduction_data, forecasting_horizon=3)
        pred_tw = f_tw.predict(forecasting_horizon=3)["value"].to_numpy()

        f_vw = _make_forecaster(vintage_weighter=vw)
        f_vw.fit(reduction_data, forecasting_horizon=3)
        pred_vw = f_vw.predict(forecasting_horizon=3)["value"].to_numpy()

        f_both = _make_forecaster(time_weighter=LinearDecayWeighter(), vintage_weighter=vw)
        f_both.fit(reduction_data, forecasting_horizon=3)
        pred_both = f_both.predict(forecasting_horizon=3)["value"].to_numpy()

        assert not np.allclose(pred_both, pred_tw) or not np.allclose(pred_both, pred_vw)

    def test_zero_vintage_weight_raises(self, reduction_data):
        """All-zero vintage_weighter should raise ValueError."""
        f = _make_forecaster(vintage_weighter=LookupWeighter(mapping={}, default=0.0))
        with pytest.raises(ValueError, match="zero"):
            f.fit(reduction_data, forecasting_horizon=3)


class TestVintageWeighterDirectStrategies:
    """Tests for vintage_weighter with different reduction strategies."""

    @pytest.mark.parametrize("strategy", ["multi-output", "direct"])
    def test_vintage_weighter_with_strategy(self, reduction_data, strategy):
        """vintage_weighter should work with each reduction strategy."""
        f = PointReductionForecaster(
            estimator=LinearRegression(),
            reduction_strategy=strategy,
            vintage_weighter=_increasing_table(reduction_data),
        )
        f.fit(reduction_data, forecasting_horizon=3)
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3


# Delivered sample_weight per collapse alignment, recorded from the implementation
# before the per-step "matched_step" refactor (14 daily rows, horizon 3, default
# features). Every strategy delivers the same collapsed vector to every estimator.
_GOLDEN_COLLAPSE_WEIGHTS = {
    "first_step": [0.535398230088, 0.58407079646, 0.632743362832, 0.924778761062, 0.730088495575, 0.778761061947, 2.774336283186, 1.119469026549, 0.924778761062, 0.973451327434, 1.022123893805],
    "mean_step": [0.549237170596, 0.671289875173, 0.71705963939, 0.762829403606, 1.342579750347, 1.464632454924, 1.51040221914, 0.945908460472, 0.915395284327, 1.037447988904, 1.083217753121],
    "weighted_mean_step": [0.544524361013, 0.63457808543, 0.709374769032, 0.802449707603, 1.07847070943, 1.393931008828, 1.840360307158, 0.988933154521, 0.91749125485, 1.007544979267, 1.082341662869],
    "max_weight_step": [0.421828908555, 0.616519174041, 0.616519174041, 0.616519174041, 1.849557522124, 1.849557522124, 1.849557522124, 0.746312684366, 0.681415929204, 0.87610619469, 0.87610619469],
    "min_weight_step": [0.672222222222, 0.733333333333, 0.794444444444, 0.916666666667, 0.916666666667, 0.977777777778, 1.161111111111, 1.161111111111, 1.161111111111, 1.222222222222, 1.283333333333],
}  # fmt: skip


def _golden_series() -> tuple[pl.DataFrame, TableWeighter]:
    """Fourteen daily rows and a non-geometric weight table (ramp, period-4 bump, spike)."""
    n = 14
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    y = pl.DataFrame({"time": times, "value": [float(i) + np.sin(i * 0.5) for i in range(n)]})
    weights = [1.0 + 0.1 * i + (0.5 if i % 4 == 0 else 0.0) + (4.0 if i == 7 else 0.0) for i in range(n)]
    return y, TableWeighter(frame=pl.DataFrame({"time": times, "weight": weights}), on="time")


def _delivered_per_step(forecaster) -> list[np.ndarray]:
    """The training sample_weight each fitted estimator received, in step order."""
    estimators = forecaster.estimator_ if isinstance(forecaster.estimator_, list) else [forecaster.estimator_]
    return [np.asarray(est.received_[2]) for est in estimators]


class TestGoldenCollapseWeights:
    """The five collapse alignments keep their pre-refactor delivered weights."""

    def test_golden_vectors_discriminate(self):
        """The pinned vectors are pairwise different and none is constant, so the pin is not vacuous."""
        vectors = [np.asarray(v) for v in _GOLDEN_COLLAPSE_WEIGHTS.values()]
        for i, a in enumerate(vectors):
            assert len(np.unique(np.round(a, 9))) > 2
            for b in vectors[i + 1 :]:
                assert not np.allclose(a, b)

    @pytest.mark.parametrize("strategy", ["multi-output", "direct", "dir-rec"])
    @pytest.mark.parametrize("alignment", list(_GOLDEN_COLLAPSE_WEIGHTS))
    def test_delivered_weights_match_golden(self, alignment, strategy):
        """Every estimator of every strategy receives the pinned vector for a collapse alignment."""
        y, weighter = _golden_series()
        forecaster = PointReductionForecaster(
            estimator=SampleWeightRecorder(),
            reduction_strategy=strategy,
            time_weighter=weighter,
            sample_weight_alignment=alignment,
        ).fit(y, forecasting_horizon=3)
        for delivered in _delivered_per_step(forecaster):
            np.testing.assert_allclose(delivered, _GOLDEN_COLLAPSE_WEIGHTS[alignment], rtol=1e-10)


# ---------------------------------------------------------------------------
# sample_weight_alignment="matched_step" and sample_weight_horizon_decay
# ---------------------------------------------------------------------------

H = 3
FAMILIES = [PointReductionForecaster, ClassProbaReductionForecaster, IntervalReductionForecaster]


def _daily(n: int) -> list[datetime]:
    return [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]


def _y(n: int = 30, values: list | None = None) -> pl.DataFrame:
    """A daily series; positions in the transformed target frame equal row positions (no transformer)."""
    values = values if values is not None else [float(i) + np.sin(i) for i in range(n)]
    return pl.DataFrame({"time": _daily(n), "value": values}, schema={"time": pl.Datetime("us"), "value": pl.Float64})


def _y_panel(n: int = 30, b_values: list | None = None) -> pl.DataFrame:
    b_values = b_values if b_values is not None else [2.0 * i for i in range(n)]
    return pl.DataFrame(
        {"time": _daily(n), "a__value": [float(i) + np.sin(i) for i in range(n)], "b__value": b_values},
        schema={"time": pl.Datetime("us"), "a__value": pl.Float64, "b__value": pl.Float64},
    )


def _table(times: list[datetime], weights, **group_weights) -> TableWeighter:
    """A TableWeighter over ``times``; keyword arguments add ``{group}__weight`` columns."""
    data = {"time": times, "weight": list(map(float, weights))}
    data.update({f"{g}__weight": list(map(float, w)) for g, w in group_weights.items()})
    return TableWeighter(frame=pl.DataFrame(data), on="time")


def _profile(n: int) -> np.ndarray:
    """A non-geometric weight profile: ramp, period-4 bump, one spike."""
    return np.array([1.0 + 0.1 * i + (0.5 if i % 4 == 0 else 0.0) + (4.0 if i == 7 else 0.0) for i in range(n)])


def _normalized(column: np.ndarray) -> np.ndarray:
    return column * (len(column) / column.sum())


def _matched(strategy: str = "direct", estimator=None, **kwargs) -> PointReductionForecaster:
    return PointReductionForecaster(
        estimator=estimator if estimator is not None else SampleWeightRecorder(),
        reduction_strategy=strategy,
        sample_weight_alignment="matched_step",
        **kwargs,
    )


def _spy(monkeypatch, name: str, events: list | None = None) -> list:
    """Record the return value of a BaseReductionForecaster method on every call."""
    calls: list = []
    original = getattr(BaseReductionForecaster, name)

    def wrapper(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        calls.append(out)
        if events is not None:
            events.append(name)
        return out

    monkeypatch.setattr(BaseReductionForecaster, name, wrapper)
    return calls


class TestHorizonDecayParameter:
    """The weighted-mean horizon decay is a tunable constructor parameter."""

    def test_negative_decay_raises(self, reduction_data):
        f = _make_forecaster(time_weighter=LinearDecayWeighter(), sample_weight_horizon_decay=-1.0)
        with pytest.raises(ValueError, match="sample_weight_horizon_decay"):
            f.fit(reduction_data, forecasting_horizon=3)

    @pytest.mark.parametrize("family", FAMILIES, ids=lambda k: k.__name__)
    def test_families_expose_and_clone_new_parameters(self, family):
        assert family().get_params()["sample_weight_horizon_decay"] == 0.5
        cloned = clone(family(sample_weight_alignment="matched_step", sample_weight_horizon_decay=0.25)).get_params()
        assert cloned["sample_weight_alignment"] == "matched_step"
        assert cloned["sample_weight_horizon_decay"] == 0.25

    def test_zero_decay_equals_mean_step(self):
        y, weighter = _golden_series()
        fits = [
            PointReductionForecaster(estimator=SampleWeightRecorder(), time_weighter=weighter, **kw).fit(
                y, forecasting_horizon=H
            )
            for kw in (
                {"sample_weight_alignment": "weighted_mean_step", "sample_weight_horizon_decay": 0.0},
                {"sample_weight_alignment": "mean_step"},
            )
        ]
        np.testing.assert_allclose(_delivered_per_step(fits[0])[0], _delivered_per_step(fits[1])[0])

    def test_decay_changes_weights_and_is_searchable(self):
        y = _y(60)
        weighter = SeasonalEmphasisWeighter(seasonality=4, emphasis=3.0)
        base = PointReductionForecaster(
            estimator=SampleWeightRecorder(),
            reduction_strategy="direct",
            time_weighter=weighter,
            sample_weight_alignment="weighted_mean_step",
        )
        delivered = [
            _delivered_per_step(clone(base).set_params(sample_weight_horizon_decay=d).fit(y, forecasting_horizon=H))[0]
            for d in (0.1, 1.0)
        ]
        assert not np.allclose(delivered[0], delivered[1])

        search = GridSearchCV(
            forecaster=clone(base).set_params(estimator=LinearRegression()),
            param_grid={"sample_weight_horizon_decay": [0.1, 1.0]},
            scoring=MeanAbsoluteError(),
            cv=ExpandingWindowSplitter(n_splits=2, test_size=6),
        ).fit(y, forecasting_horizon=H)
        assert sorted(p["sample_weight_horizon_decay"] for p in search.cv_results_["params"]) == [0.1, 1.0]


class TestStepWeightHelpers:
    """Unit tests of the per-step weight container and its helpers."""

    def test_container_masks_rows_in_lockstep(self):
        weights = _StepWeights(np.arange(10.0).reshape(5, 2), np.array([0, 0, 0, 1, 1]))
        kept = weights[np.array([True, False, True, False, True])]
        assert len(kept) == 3
        np.testing.assert_array_equal(kept.values, [[0.0, 1.0], [4.0, 5.0], [8.0, 9.0]])
        np.testing.assert_array_equal(kept.groups, [0, 0, 1])

    def test_step_sample_weight_normalizes_per_group(self):
        weights = _StepWeights(
            np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0], [1.0, 10.0], [1.0, 30.0]]), np.array([0, 0, 0, 1, 1])
        )
        column = BaseReductionForecaster._step_sample_weight(weights, 2)
        np.testing.assert_allclose(column[:3], np.array([2.0, 4.0, 6.0]) * 3 / 12)
        np.testing.assert_allclose(column[3:], np.array([10.0, 30.0]) * 2 / 40)

    def test_step_sample_weight_passes_vectors_and_none_through(self):
        vector = np.array([1.0, 2.0])
        assert BaseReductionForecaster._step_sample_weight(vector, 1) is vector
        assert BaseReductionForecaster._step_sample_weight(None, 1) is None

    def test_compute_keep_mask_and_precomputed_mask(self):
        X = pl.DataFrame({"a": [1.0, None, 3.0, float("nan")]})
        y_tab = pl.DataFrame({"t": [1.0, 2.0, None, 4.0]})
        assert PointReductionForecaster(nan_handling="pass")._compute_keep_mask(X, y_tab) is None
        f = PointReductionForecaster(nan_handling="drop")
        mask = f._compute_keep_mask(X, y_tab)
        assert mask.to_list() == [True, False, False, False]
        w = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.warns(UserWarning, match="dropped"):
            plain = f._apply_nan_handling(X, y_tab, w)
        with pytest.warns(UserWarning, match="dropped"):
            given = f._apply_nan_handling(X, y_tab, w, mask=mask)
        pl.testing.assert_frame_equal(plain[0], given[0])
        np.testing.assert_array_equal(plain[2], given[2])

    def test_check_step_weights_names_lowest_step_and_set(self):
        f = PointReductionForecaster()
        f.groups_ = None
        weights = _StepWeights(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), np.zeros(2, dtype=int))
        with pytest.raises(ValueError, match=r"training weights for step 2 sum to zero"):
            f._check_step_weights(weights, [None] * 3, "training")
        f._check_step_weights(_StepWeights(np.ones((2, 3)), np.zeros(2, dtype=int)), [None] * 3, "training")
        f._check_step_weights(np.zeros(2), [None] * 3, "training")  # not per-step: no effect


class TestMatchedStepContainer:
    """_process_fit_weights and the stride keep per-step weights and group ids in lockstep."""

    def test_panel_container_carries_group_ids(self, monkeypatch):
        returned = _spy(monkeypatch, "_process_fit_weights")
        f = _matched(time_weighter=SeasonalEmphasisWeighter(seasonality=4, emphasis=3.0)).fit(
            _y_panel(), forecasting_horizon=H
        )
        weights = returned[0]
        assert isinstance(weights, _StepWeights)
        n = 30 - H
        np.testing.assert_array_equal(weights.groups, [0] * n + [1] * n)
        assert len(weights) == f.estimator_[0].received_[0].height

    def test_stride_keeps_surviving_rows_and_ids(self, monkeypatch):
        processed = _spy(monkeypatch, "_process_fit_weights")
        strided = _spy(monkeypatch, "_apply_training_stride")
        _matched(time_weighter=_table(_daily(30), _profile(30)), training_stride=3).fit(_y(), forecasting_horizon=H)
        before, after = processed[0], strided[0][2]
        n_inst = 30 - H
        keep = np.arange(n_inst) % 3 == (n_inst - 1) % 3
        np.testing.assert_array_equal(after.values, before.values[keep])
        np.testing.assert_array_equal(after.groups, before.groups[keep])

    def test_vintage_weighter_alone_takes_collapse_path(self):
        vintage = _table(_daily(30), _profile(30))
        delivered = [
            _delivered_per_step(
                PointReductionForecaster(
                    estimator=SampleWeightRecorder(),
                    reduction_strategy="direct",
                    vintage_weighter=vintage,
                    training_stride=2,
                    sample_weight_alignment=alignment,
                ).fit(_y(), forecasting_horizon=H)
            )
            for alignment in ("matched_step", "first_step")
        ]
        for matched, first in zip(*delivered, strict=True):
            np.testing.assert_allclose(matched, first)


class TestMatchedStepDirect:
    """The direct strategy weights each step's estimator by its own target time."""

    @pytest.mark.parametrize("n_jobs", [None, 2])
    def test_later_steps_use_their_own_target_timestamp(self, n_jobs):
        times = _daily(30)
        spike = 10  # at least H rows in, at most the number of training rows
        f = _matched(time_weighter=LookupWeighter(mapping={times[spike]: 1.0}, default=0.0), n_jobs=n_jobs).fit(
            _y(), forecasting_horizon=H
        )
        for step, delivered in enumerate(_delivered_per_step(f), start=1):
            np.testing.assert_array_equal(np.flatnonzero(delivered), [spike - step])

    def test_step_one_matches_first_step_without_drops(self):
        weighter = _table(_daily(30), _profile(30))
        matched = _delivered_per_step(_matched(time_weighter=weighter).fit(_y(), forecasting_horizon=H))
        first = _delivered_per_step(
            PointReductionForecaster(
                estimator=SampleWeightRecorder(), reduction_strategy="direct", time_weighter=weighter
            ).fit(_y(), forecasting_horizon=H)
        )
        np.testing.assert_allclose(matched[0], first[0])
        assert not np.allclose(matched[2], first[2])

    def test_each_step_sums_to_its_row_count(self):
        f = _matched(time_weighter=_table(_daily(30), _profile(30))).fit(_y(), forecasting_horizon=H)
        for step, delivered in enumerate(_delivered_per_step(f), start=1):
            np.testing.assert_allclose(delivered, _normalized(_profile(30)[step : step + 30 - H]))

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"time_weighter": ExponentialDecayWeighter(half_life=5), "sample_weight_alignment": "mean_step"}],
        ids=["unweighted", "collapse"],
    )
    def test_mask_pass_skipped_without_container(self, monkeypatch, kwargs):
        masks = _spy(monkeypatch, "_compute_keep_mask")
        PointReductionForecaster(estimator=SampleWeightRecorder(), reduction_strategy="direct", **kwargs).fit(
            _y(), forecasting_horizon=H
        )
        assert masks == []

    def test_filtering_happens_inside_the_fits_after_the_check(self, monkeypatch):
        events: list = []
        _spy(monkeypatch, "_check_step_weights", events)
        _spy(monkeypatch, "_apply_nan_handling", events)
        values = [None if i == 12 else float(i) for i in range(30)]
        with pytest.warns(UserWarning, match="dropped"):
            _matched(time_weighter=_table(_daily(30), _profile(30)), nan_handling="drop", n_jobs=1).fit(
                _y(values=values), forecasting_horizon=H
            )
        assert events.count("_apply_nan_handling") == H
        assert max(i for i, e in enumerate(events) if e == "_check_step_weights") < events.index("_apply_nan_handling")


class TestMatchedStepDirRec:
    """dir-rec follows the same per-step rule."""

    def test_dir_rec_matches_direct(self):
        weighter = SeasonalEmphasisWeighter(seasonality=4, emphasis=3.0)
        direct = _delivered_per_step(_matched("direct", time_weighter=weighter).fit(_y(), forecasting_horizon=H))
        dir_rec = _delivered_per_step(_matched("dir-rec", time_weighter=weighter).fit(_y(), forecasting_horizon=H))
        assert not np.allclose(direct[0], direct[2])
        for a, b in zip(direct, dir_rec, strict=True):
            np.testing.assert_allclose(a, b)

    @pytest.mark.parametrize("strategy", ["direct", "dir-rec"])
    def test_step_dependent_nan_rows(self, strategy):
        values = [None if i in (12, 20) else float(i) for i in range(30)]
        with pytest.warns(UserWarning, match="dropped"):
            f = _matched(strategy, time_weighter=SeasonalEmphasisWeighter(seasonality=4), nan_handling="drop").fit(
                _y(values=values), forecasting_horizon=H
            )
        missing = {12, 20}
        weights = SeasonalEmphasisWeighter(seasonality=4).compute_weights(pl.Series(_daily(30))).to_numpy()

        def survives(origin: int, step: int) -> bool:
            # direct drops a row per step; dir-rec drops it if any step's target is missing.
            targets = [step] if strategy == "direct" else range(1, H + 1)
            return origin not in missing and all(origin + h not in missing for h in targets)

        for step, est in enumerate(f.estimator_, start=1):
            origins = np.array([i for i in range(30 - H) if survives(i, step)])
            np.testing.assert_allclose(np.asarray(est.received_[2]), _normalized(weights[origins + step]))

    @pytest.mark.parametrize("strategy", ["direct", "dir-rec"])
    def test_all_dropped_step_keeps_nan_error(self, strategy):
        values = [float(i) if i % 2 == 0 else None for i in range(30)]
        with pytest.raises(ValueError, match="contain NaN"):
            _matched(strategy, time_weighter=SeasonalEmphasisWeighter(seasonality=4), nan_handling="drop").fit(
                _y(values=values), forecasting_horizon=H
            )

    def test_group_with_no_surviving_row_is_not_checked(self):
        b_values = [float(i) if i % 2 == 0 else None for i in range(30)]
        with pytest.warns(UserWarning, match="dropped"):
            f = _matched(time_weighter=SeasonalEmphasisWeighter(seasonality=4), nan_handling="drop").fit(
                _y_panel(b_values=b_values), forecasting_horizon=H
            )
        step_one = np.asarray(f.estimator_[0].received_[2])
        assert len(step_one) == 30 - H  # group b has no step-1 survivor; group a's rows remain
        np.testing.assert_allclose(step_one.sum(), 30 - H)


class TestMatchedStepZeroColumns:
    """A step with no weighted row for some group raises, deterministically."""

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_lookup_below_horizon(self, n_jobs):
        times = _daily(30)
        f = _matched(time_weighter=LookupWeighter(mapping={times[1]: 1.0}, default=0.0), n_jobs=n_jobs)
        with pytest.raises(ValueError, match=r"training weights for step 2 sum to zero"):
            f.fit(_y(), forecasting_horizon=H)

    def test_stride_drops_only_weighted_origin(self):
        times = _daily(30)
        n_inst = 30 - H
        keep = np.arange(n_inst) % 3 == (n_inst - 1) % 3
        # Every step has an origin for this timestamp (H <= origin <= n_inst), but the
        # stride drops step 1's only one.
        origin = next(q for q in range(H, n_inst + 1) if not keep[q - 1])
        weighter = LookupWeighter(mapping={times[origin]: 1.0}, default=0.0)
        _matched(time_weighter=weighter).fit(_y(), forecasting_horizon=H)  # fine without the stride
        with pytest.raises(ValueError, match=r"training weights for step 1 sum to zero"):
            _matched(time_weighter=weighter, training_stride=3).fit(_y(), forecasting_horizon=H)

    def test_one_panel_group_without_weighted_row(self):
        times = _daily(30)
        b_weights = [1.0 if i == 1 else 0.0 for i in range(30)]
        weighter = _table(times, np.ones(30), a=np.ones(30), b=b_weights)
        with pytest.raises(ValueError, match=r"training weights for step 2 for panel group 'b' sum to zero"):
            _matched(time_weighter=weighter).fit(_y_panel(), forecasting_horizon=H)

    def test_evaluation_window_without_weighted_row(self):
        times = _daily(30)
        val = 10
        weighter = LookupWeighter(mapping=dict.fromkeys(times[: 30 - val], 1.0), default=0.0)
        f = _matched(estimator=WeightRecordingRegressor(), time_weighter=weighter, validation_size=val)
        with pytest.raises(ValueError, match=r"evaluation weights for step 1 sum to zero"):
            f.fit(_y(), forecasting_horizon=H)

    def test_training_checked_before_evaluation(self):
        times = _daily(30)
        weighter = LookupWeighter(mapping={times[1]: 1.0, times[2]: 1.0, times[29]: 1.0}, default=0.0)
        f = _matched(estimator=WeightRecordingRegressor(), time_weighter=weighter, validation_size=10)
        with pytest.raises(ValueError, match=r"training weights for step 3 sum to zero"):
            f.fit(_y(), forecasting_horizon=H)


class TestMatchedStepPanelBalance:
    """Each panel group keeps its own balance under matched_step."""

    N, TAIL = 30, 4

    def _weighter(self):
        times = _daily(self.N)
        return _table(times, np.ones(self.N), a=_profile(self.N), b=10.0 * _profile(self.N)[::-1])

    def _profiles(self):
        return {"a": _profile(self.N), "b": 10.0 * _profile(self.N)[::-1]}

    def _y(self):
        return _y_panel(self.N, b_values=[2.0 * i if i < self.N - self.TAIL else None for i in range(self.N)])

    def _expected(self, origins_by_group, step):
        prof = self._profiles()
        return np.concatenate([_normalized(prof[g][np.asarray(o) + step]) for g, o in origins_by_group.items()])

    def _survivors(self, origins, group, step):
        """Origins whose feature row and step target are both outside group b's NaN tail."""
        if group == "a":
            return list(origins)
        cutoff = self.N - self.TAIL
        return [i for i in origins if i < cutoff and i + step < cutoff]

    @pytest.mark.parametrize("stride", [1, 3])
    def test_training_slices(self, stride):
        with pytest.warns(UserWarning, match="dropped"):
            f = _matched(time_weighter=self._weighter(), nan_handling="drop", training_stride=stride).fit(
                self._y(), forecasting_horizon=H
            )
        n_inst = self.N - H
        origins = [i for i in range(n_inst) if i % stride == (n_inst - 1) % stride]
        for step, delivered in enumerate(_delivered_per_step(f), start=1):
            expected = self._expected({g: self._survivors(origins, g, step) for g in ("a", "b")}, step)
            np.testing.assert_allclose(delivered, expected)

    def test_holdout_training_and_evaluation_slices(self):
        val = 10
        with pytest.warns(UserWarning, match="dropped"):
            f = _matched(
                estimator=WeightRecordingRegressor(),
                time_weighter=self._weighter(),
                nan_handling="drop",
                validation_size=val,
            ).fit(self._y(), forecasting_horizon=H)
        head = self.N - val
        train_origins = list(range(head - H))
        eval_origins = [head - 1 + j for j in range(val + 1 - H)]
        for step, est in enumerate(f.estimator_, start=1):
            np.testing.assert_allclose(
                est.received_sample_weight_,
                self._expected({g: self._survivors(train_origins, g, step) for g in ("a", "b")}, step),
            )
            np.testing.assert_allclose(
                est.received_sample_weight_eval_set_[0],
                self._expected({g: self._survivors(eval_origins, g, step) for g in ("a", "b")}, step),
            )


class TestMatchedStepFamiliesAndWeighters:
    """Family smoke tests, vintage multiplication, and the geometric-profile property."""

    @pytest.mark.parametrize("family", FAMILIES, ids=lambda k: k.__name__)
    def test_families_fit_direct_matched_step(self, family, y_X_factory, class_proba_y_X_factory, monkeypatch):
        weighter = SeasonalEmphasisWeighter(seasonality=4, emphasis=3.0)
        if family is ClassProbaReductionForecaster:
            y, _ = class_proba_y_X_factory(length=60, n_targets=1, n_features=0, n_classes=3, seed=0)
            f = family(estimator=DecisionTreeClassifier(random_state=0), reduction_strategy="direct")
        elif family is IntervalReductionForecaster:
            y, _ = y_X_factory(length=60, n_targets=1, n_features=0, seed=0)
            f = family(estimator=QuantileRegressor(alpha=0.0), reduction_strategy="direct")
        else:
            y, _ = y_X_factory(length=60, n_targets=1, n_features=0, seed=0)
            f = family(estimator=LinearRegression(), reduction_strategy="direct")
        delivered = []
        original = BaseReductionForecaster._fit_single_estimator

        def spy(self, X_tab, y_tab, sample_weight, *args, **kwargs):
            delivered.append(np.asarray(sample_weight))
            return original(self, X_tab, y_tab, sample_weight, *args, **kwargs)

        monkeypatch.setattr(BaseReductionForecaster, "_fit_single_estimator", spy)
        f.set_params(time_weighter=weighter, sample_weight_alignment="matched_step").fit(y, forecasting_horizon=H)

        weights = weighter.compute_weights(y["time"]).to_numpy()
        rows = np.arange(len(y) - H)
        # Steps are fitted in order (n_jobs=None); the interval family repeats them once per bound.
        assert delivered and len(delivered) % H == 0
        for call, w in enumerate(delivered):
            step = call % H + 1
            np.testing.assert_allclose(w, _normalized(weights[rows + step]))
        assert not np.allclose(delivered[0], delivered[H - 1])

    def test_vintage_weights_multiply_in(self):
        times = _daily(30)
        vintage = np.array([1.0 + (i % 3) for i in range(30)])
        f = _matched(time_weighter=_table(times, _profile(30)), vintage_weighter=_table(times, vintage)).fit(
            _y(), forecasting_horizon=H
        )
        rows = np.arange(30 - H)
        for step, delivered in enumerate(_delivered_per_step(f), start=1):
            np.testing.assert_allclose(delivered, _normalized(_profile(30)[rows + step] * vintage[rows]))

    def test_exponential_decay_on_regular_grid_equals_first_step(self):
        weighter = ExponentialDecayWeighter(half_life=5)
        matched = _delivered_per_step(_matched(time_weighter=weighter).fit(_y(), forecasting_horizon=H))
        first = _delivered_per_step(
            PointReductionForecaster(
                estimator=SampleWeightRecorder(), reduction_strategy="direct", time_weighter=weighter
            ).fit(_y(), forecasting_horizon=H)
        )
        for a, b in zip(matched, first, strict=True):
            np.testing.assert_allclose(a, b)

    def test_calendar_months_break_the_equivalence(self):
        months = [datetime(2020 + (i // 12), i % 12 + 1, 1) for i in range(30)]
        y = pl.DataFrame({"time": months, "value": [float(i) for i in range(30)]})
        delivered = _delivered_per_step(
            _matched(time_weighter=ExponentialDecayWeighter(half_life=60)).fit(y, forecasting_horizon=H)
        )
        ratio = delivered[1] / delivered[0]
        assert not np.allclose(ratio, ratio[0])


_FALLBACK_MESSAGE = r'sample_weight_alignment="matched_step" has no effect with reduction_strategy="multi-output"'


def _fallback_warnings(record) -> list:
    return [w for w in record if issubclass(w.category, UserWarning) and "has no effect with" in str(w.message)]


class TestMatchedStepMultiOutputFallback:
    """matched_step under multi-output warns once per fit and computes first_step weights."""

    def _family_case(self, family, y_X_factory, class_proba_y_X_factory):
        if family is ClassProbaReductionForecaster:
            y, _ = class_proba_y_X_factory(length=60, n_targets=1, n_features=0, n_classes=3, seed=0)
            kwargs = {"estimator": DecisionTreeClassifier(random_state=0)}
            fit_kwargs, predict = {}, lambda f: f.predict()
        elif family is IntervalReductionForecaster:
            y, _ = y_X_factory(length=60, n_targets=1, n_features=0, seed=0)
            # The default estimator's QuantileRegressor accepts sample_weight only once requested.
            quantile = QuantileRegressor(alpha=0.0).set_fit_request(sample_weight=True)
            kwargs = {"estimator": MultiOutputRegressor(quantile)}
            fit_kwargs, predict = {"coverage_rates": [0.9]}, lambda f: f.predict_interval(coverage_rates=[0.9])
        else:
            y, _ = y_X_factory(length=60, n_targets=1, n_features=0, seed=0)
            kwargs = {"estimator": LinearRegression()}
            fit_kwargs, predict = {}, lambda f: f.predict()
        return y, kwargs, fit_kwargs, predict

    @pytest.mark.parametrize("family", FAMILIES, ids=lambda k: k.__name__)
    def test_warns_once_and_predicts_like_first_step(self, family, y_X_factory, class_proba_y_X_factory):
        import warnings

        y, kwargs, fit_kwargs, predict = self._family_case(family, y_X_factory, class_proba_y_X_factory)
        weighter = SeasonalEmphasisWeighter(seasonality=4, emphasis=3.0)
        matched = family(time_weighter=weighter, sample_weight_alignment="matched_step", **kwargs)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            matched.fit(y, forecasting_horizon=H, **fit_kwargs)
        assert len(_fallback_warnings(record)) == 1
        first = family(time_weighter=weighter, sample_weight_alignment="first_step", **kwargs)
        first.fit(y, forecasting_horizon=H, **fit_kwargs)
        pl.testing.assert_frame_equal(predict(matched), predict(first))

    @pytest.mark.parametrize(
        ("family", "strategy"),
        [
            (PointReductionForecaster, "direct"),
            (PointReductionForecaster, "dir-rec"),
            (ClassProbaReductionForecaster, "direct"),
            (IntervalReductionForecaster, "direct"),
            (IntervalReductionForecaster, "dir-rec"),
        ],
        ids=lambda v: v if isinstance(v, str) else v.__name__,
    )
    def test_per_step_strategies_never_warn(self, family, strategy, y_X_factory, class_proba_y_X_factory):
        import warnings

        y, kwargs, fit_kwargs, _ = self._family_case(family, y_X_factory, class_proba_y_X_factory)
        if family is IntervalReductionForecaster:
            kwargs = {"estimator": QuantileRegressor(alpha=0.0)}
        f = family(
            reduction_strategy=strategy,
            time_weighter=SeasonalEmphasisWeighter(seasonality=4, emphasis=3.0),
            sample_weight_alignment="matched_step",
            **kwargs,
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            f.fit(y, forecasting_horizon=H, **fit_kwargs)
        assert _fallback_warnings(record) == []

    def test_no_time_weighter_means_no_warning(self):
        import warnings

        f = PointReductionForecaster(estimator=LinearRegression(), sample_weight_alignment="matched_step")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            f.fit(_y(), forecasting_horizon=H)
        assert _fallback_warnings(record) == []

    def test_search_across_strategies_warns_per_multi_output_fit(self):
        import warnings

        n_splits = 2
        search = GridSearchCV(
            forecaster=PointReductionForecaster(
                estimator=LinearRegression(),
                time_weighter=SeasonalEmphasisWeighter(seasonality=4, emphasis=3.0),
                sample_weight_alignment="matched_step",
            ),
            param_grid={"reduction_strategy": ["multi-output", "direct"]},
            scoring=MeanAbsoluteError(),
            cv=ExpandingWindowSplitter(n_splits=n_splits, test_size=6),
            n_jobs=1,
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            search.fit(_y(60), forecasting_horizon=H)
        assert len(search.cv_results_["params"]) == 2
        refit = 1 if search.best_params_["reduction_strategy"] == "multi-output" else 0
        assert len(_fallback_warnings(record)) == n_splits + refit
        assert all(_FALLBACK_MESSAGE.replace("\\", "") in str(w.message) for w in _fallback_warnings(record))


@pytest.mark.parametrize("cls", [BaseReductionForecaster, *FAMILIES], ids=lambda k: k.__name__)
def test_parameter_docstrings_document_matched_step_and_decay(cls):
    """The Parameters section of the base and each family names the new value and parameter."""
    # cleandoc: Python 3.13+ already strips docstring indentation, earlier versions do not.
    doc = inspect.cleandoc(cls.__doc__)
    start = doc.index("Parameters\n")
    ends = [doc.find(f"\n{name}\n", start) for name in ("Attributes", "Notes", "Examples", "See Also")]
    section = doc[start : min(e for e in ends if e != -1)]
    assert "sample_weight_horizon_decay : float, default=0.5" in section
    assert '"matched_step"' in section
