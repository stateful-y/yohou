"""Tests for choosing one boosting round per estimator position from fold curves."""

from datetime import datetime, timedelta
from unittest import mock

import numpy as np
import polars as pl
import pytest

from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import ExpandingWindowSplitter
from yohou.model_selection import utils as ms_utils
from yohou.model_selection.utils import _evaluate_candidate_shared_rounds, _fit_and_score, _select_shared_rounds
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer

from .shared_round_stubs import CurveEarlyStoppingAdapter, CurveRegressor


def _valley(length: int, best: int, depth: float = 1.0) -> np.ndarray:
    """A loss curve over rounds 1..length with its minimum at round ``best``."""
    rounds = np.arange(1, length + 1)
    return depth * (rounds - best) ** 2 / length**2 + 0.5


class TestSelection:
    def test_mean_over_rounds_every_fold_reached(self):
        curves = [_valley(32, 12), _valley(50, 30), _valley(38, 18)]
        rounds, boundary = _select_shared_rounds({"step_1": [(c, False) for c in curves]})
        mean = np.mean([c[:32] for c in curves], axis=0)
        assert rounds == {"step_1": int(np.argmin(mean)) + 1}
        assert boundary == {"step_1": False}

    def test_higher_is_better(self):
        curves = [-_valley(40, 10), -_valley(40, 20)]
        rounds, _ = _select_shared_rounds({"multi_output": [(c, True) for c in curves]})
        assert rounds == {"multi_output": 15}

    def test_ties_take_the_smallest_round(self):
        curve = np.array([3.0, 1.0, 1.0, 1.0, 2.0])
        rounds, _ = _select_shared_rounds({"step_1": [(curve, False), (curve.copy(), False)]})
        assert rounds == {"step_1": 2}

    def test_round_at_the_shortest_curve_is_flagged(self):
        still_improving = np.linspace(2.0, 1.0, 20)
        rounds, boundary = _select_shared_rounds({
            "step_1": [(still_improving, False), (np.linspace(2.0, 0.5, 35), False)]
        })
        assert rounds == {"step_1": 20}
        assert boundary == {"step_1": True}

    def test_positions_are_chosen_independently(self):
        curves = {
            "coverage_rate_0.9_lower/step_1": [(_valley(30, 5), False), (_valley(30, 7), False)],
            "coverage_rate_0.9_lower/step_2": [(_valley(30, 20), False), (_valley(30, 22), False)],
            "coverage_rate_0.9_upper/step_1": [(_valley(30, 11), False), (_valley(30, 11), False)],
        }
        rounds, _ = _select_shared_rounds(curves)
        assert rounds == {
            "coverage_rate_0.9_lower/step_1": 6,
            "coverage_rate_0.9_lower/step_2": 21,
            "coverage_rate_0.9_upper/step_1": 11,
        }
        assert list(rounds) == list(curves)

    def test_boundary_is_flagged_per_position(self):
        curves = {
            "step_1": [(np.linspace(2.0, 1.0, 20), False), (np.linspace(2.0, 0.5, 35), False)],
            "step_2": [(_valley(30, 5), False), (_valley(30, 7), False)],
        }
        rounds, boundary = _select_shared_rounds(curves)
        assert rounds == {"step_1": 20, "step_2": 6}
        assert boundary == {"step_1": True, "step_2": False}

    def test_single_fold(self):
        rounds, _ = _select_shared_rounds({"step_1": [(_valley(25, 9), False)]})
        assert rounds == {"step_1": 9}


class TestRejections:
    def test_directions_must_agree(self):
        with pytest.raises(ValueError, match="direction"):
            _select_shared_rounds({"step_1": [(_valley(10, 3), False), (_valley(10, 3), True)]})

    def test_empty_curve(self):
        with pytest.raises(ValueError, match="empty"):
            _select_shared_rounds({"step_1": [(np.array([]), False)]})

    def test_no_folds_for_a_position(self):
        with pytest.raises(ValueError, match="no fold"):
            _select_shared_rounds({"step_1": []})


N_SPLITS = 3
TEST_SIZE = 12
HORIZON = 3


def _series(n: int = 120) -> pl.DataFrame:
    rng = np.random.default_rng(5)
    times = pl.datetime_range(datetime(2023, 1, 1), datetime(2023, 1, 1) + timedelta(hours=n - 1), "1h", eager=True)
    # A rising level, so later test windows need more rounds than earlier ones.
    return pl.DataFrame({"time": times, "value": 10.0 + np.arange(n) * 0.2 + rng.normal(0, 0.3, n)})


def _splits(y):
    return list(ExpandingWindowSplitter(n_splits=N_SPLITS, test_size=TEST_SIZE).split(y))


def _evaluate(forecaster, y, **overrides):
    kwargs = {
        "splits": _splits(y),
        "parameters": None,
        "early_stopping_adapter": CurveEarlyStoppingAdapter(),
        "scorer": MeanAbsoluteError(),
        "verbose": 0,
        "fit_params": {},
        "predict_func_params": {},
        "score_params": {},
        "return_train_score": False,
        "return_n_test_samples": True,
        "return_times": True,
        "error_score": np.nan,
    }
    kwargs.update(overrides)
    return _evaluate_candidate_shared_rounds(forecaster, y, None, HORIZON, **kwargs)


def _forecaster(**estimator_params):
    return PointReductionForecaster(
        estimator=CurveRegressor(**estimator_params),
        reduction_strategy="direct",
        actual_transformer=LagTransformer(lag=[1, 2]),
    )


class TestCandidateEvaluation:
    def test_each_fold_evaluates_on_its_own_test_window(self):
        y = _series()
        seen = []
        original = ms_utils._score_fold

        def capture(fold, **kwargs):
            seen.append(fold)
            return original(fold, **kwargs)

        with mock.patch.object(ms_utils, "_score_fold", capture):
            _evaluate(_forecaster(), y)
        assert len(seen) == N_SPLITS
        for fold, (_, test) in zip(seen, _splits(y), strict=True):
            window_values = set(y["value"].to_numpy()[test].tolist())
            for _, est in fold.forecaster._fitted_estimator_positions():
                assert set(est.received_eval_targets_.tolist()) <= window_values
            pl.testing.assert_frame_equal(fold.y_test, y[test])

    def test_scorer_receives_the_same_window(self):
        y = _series()
        scored_times = []
        original = ms_utils._score

        def capture(forecaster, y_train, y_test, y_pred, *args, **kwargs):
            scored_times.append(y_test["time"].to_list())
            return original(forecaster, y_train, y_test, y_pred, *args, **kwargs)

        with mock.patch.object(ms_utils, "_score", capture):
            _evaluate(_forecaster(), y)
        # Rolling prediction can forecast past the window's end; the scorer
        # scores against exactly the fold's test rows.
        assert len(scored_times) == N_SPLITS
        for times, (_, test) in zip(scored_times, _splits(y), strict=True):
            assert times == y["time"].to_numpy()[test].tolist()

    def test_every_fold_is_cut_to_the_shared_rounds(self):
        y = _series()
        seen = []
        original = ms_utils._score_fold

        def capture(fold, **kwargs):
            seen.append(fold)
            return original(fold, **kwargs)

        with mock.patch.object(ms_utils, "_score_fold", capture):
            _, record = _evaluate(_forecaster(), y)
        rounds = record["rounds"]
        assert list(rounds) == ["step_1", "step_2", "step_3"]
        trained = [
            {position: est.rounds_trained_ for position, est in fold.forecaster._fitted_estimator_positions()}
            for fold in seen
        ]
        assert len({t["step_1"] for t in trained}) > 1, (
            "folds must stop at different rounds for the check to mean anything"
        )
        for fold in seen:
            for position, est in fold.forecaster._fitted_estimator_positions():
                assert est.rounds_used_ == rounds[position]
                assert rounds[position] <= min(lengths[position] for lengths in record["curve_lengths"])

    def test_result_format_matches_fit_and_score(self):
        y = _series()
        results, _ = _evaluate(_forecaster(), y, return_train_score=True)
        forecaster = _forecaster()
        splits = _splits(y)
        reference = _fit_and_score(
            forecaster,
            y,
            None,
            HORIZON,
            scorer=MeanAbsoluteError(),
            train=splits[0][0],
            test=splits[0][1],
            verbose=0,
            parameters=None,
            fit_params={},
            predict_func_params={},
            score_params={},
            return_train_score=True,
            return_n_test_samples=True,
            return_times=True,
        )
        assert len(results) == N_SPLITS
        for result in results:
            assert set(result) == set(reference)
            assert result["fit_error"] is None
            assert np.isfinite(result["test_scores"])

    def test_verbose_prints_one_line_per_split(self, capsys):
        y = _series()
        _evaluate(_forecaster(), y, verbose=3, parameters={"estimator__patience": 5})
        lines = [line for line in capsys.readouterr().out.splitlines() if "END" in line]
        assert len(lines) == N_SPLITS
        assert all(line.startswith(f"[CV {i + 1}/{N_SPLITS}] END") for i, line in enumerate(lines))
        assert all("estimator__patience=5" in line for line in lines)

    def test_adapter_prepares_every_fold_and_validates_once(self):
        y = _series()
        adapter = CurveEarlyStoppingAdapter()
        _evaluate(_forecaster(), y, early_stopping_adapter=adapter)
        kinds = [call[0] for call in adapter.calls]
        assert kinds.count("validate") == 1
        assert kinds.count("prepare_fold_fit") == N_SPLITS
        assert kinds.count("truncate") == N_SPLITS * HORIZON

    def test_boundary_is_recorded(self):
        y = _series()
        _, record = _evaluate(_forecaster(n_rounds=8), y)
        assert record["rounds_at_boundary"] is True
        assert record["boundary_positions"] == ["step_1", "step_2", "step_3"]


class TestFailedFolds:
    def test_one_failing_fold(self):
        y = _series()
        splits = _splits(y)
        smallest_train = len(splits[0][0]) - HORIZON
        forecaster = _forecaster(fail_below_train_rows=smallest_train + 1)
        results, record = _evaluate(forecaster, y)
        assert results[0]["fit_error"] is not None and "refuses" in results[0]["fit_error"]
        assert np.isnan(results[0]["test_scores"])
        assert all(np.isfinite(r["test_scores"]) for r in results[1:])
        assert record["curve_lengths"][0] is None
        # Rounds come from the two successful folds alone.
        survivors = [lengths for lengths in record["curve_lengths"] if lengths is not None]
        assert len(survivors) == 2
        assert set(record["rounds"]) == {"step_1", "step_2", "step_3"}

    def test_every_fold_failing(self):
        y = _series()
        results, record = _evaluate(_forecaster(fail_below_train_rows=10_000), y)
        assert all(np.isnan(r["test_scores"]) for r in results)
        assert record["rounds"] == {}
        assert record["rounds_at_boundary"] is False

    def test_error_score_raise(self):
        y = _series()
        with pytest.raises(RuntimeError, match="refuses"):
            _evaluate(_forecaster(fail_below_train_rows=10_000), y, error_score="raise")


class TestCandidateChecks:
    def test_dir_rec_rejected_before_any_fit(self):
        y = _series()
        adapter = CurveEarlyStoppingAdapter()
        with pytest.raises(ValueError, match="dir-rec"):
            _evaluate(_forecaster(), y, parameters={"reduction_strategy": "dir-rec"}, early_stopping_adapter=adapter)
        assert adapter.calls == []

    def test_validation_size_rejected(self):
        y = _series()
        with pytest.raises(ValueError, match="validation_size=12"):
            _evaluate(_forecaster(), y, parameters={"validation_size": 12})


class TestHelpers:
    def test_merge_concatenates_lists(self):
        from yohou.model_selection.utils import _merge_fit_params

        assert _merge_fit_params({"callbacks": ["a"], "other": 1}, {"callbacks": ["b"], "new": 2}) == {
            "callbacks": ["a", "b"],
            "other": 1,
            "new": 2,
        }

    def test_merge_rejects_non_list_conflict(self):
        from yohou.model_selection.utils import _merge_fit_params

        with pytest.raises(ValueError, match="'marker' is set both by the caller and by the early-stopping adapter"):
            _merge_fit_params({"marker": 1}, {"marker": 2})

    def test_params_message(self):
        from yohou.model_selection.utils import _params_message

        assert _params_message(None) == ""
        assert _params_message({"b": 2, "a": 1}) == "a=1, b=2"
