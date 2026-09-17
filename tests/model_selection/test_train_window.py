"""Tests for the train-score recipe: which rows a train score covers."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from holdout_stub import _HoldoutStubForecaster
from yohou.interval import SplitConformalForecaster
from yohou.metrics import MeanAbsoluteError
from yohou.metrics.interval import EmpiricalCoverage, IntervalScore
from yohou.model_selection import ExpandingWindowSplitter, cross_validate
from yohou.model_selection import utils as ms_utils
from yohou.model_selection.utils import _MultimetricScorer, _score_train_window, _train_window_predictions
from yohou.point import SeasonalNaive

FH = 24


def _hourly(n, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return pl.DataFrame({
        "time": [datetime(2024, 1, 1) + timedelta(hours=int(i)) for i in t],
        "y_0": np.sin(2 * np.pi * t / 24) + 0.1 * rng.standard_normal(n),
    })


def _fitted(forecaster, y_train):
    return forecaster.fit(y_train, forecasting_horizon=FH)


class TestScoredRows:
    """The scored stretch ends before the forecaster's held-back rows."""

    def test_no_holdout_scores_end_of_training_window(self):
        y_train = _hourly(1000)
        forecaster = _fitted(_HoldoutStubForecaster(holdout=0, seasonality=24), y_train)
        window = _train_window_predictions(forecaster, y_train, None, n_rows=168, method="predict")
        np.testing.assert_array_equal(window.positions, np.arange(832, 1000))
        assert window.y_before.height == 832
        assert window.y_scored["time"].equals(y_train["time"][832:1000])

    def test_holdout_scores_rows_before_it(self):
        y_train = _hourly(1000)
        forecaster = _fitted(_HoldoutStubForecaster(holdout=168, seasonality=24), y_train)
        window = _train_window_predictions(forecaster, y_train, None, n_rows=168, method="predict")
        np.testing.assert_array_equal(window.positions, np.arange(664, 832))
        assert window.y_before.height == 664
        # Every forecast origin lies inside the learned-from rows, never in the held-back stretch.
        assert window.y_pred["vintage_time"].max() <= y_train["time"][831]

    def test_split_conformal_scores_before_its_calibration_stretch(self):
        y_train = _hourly(1000)
        forecaster = _fitted(
            SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=24), calibration_size=168), y_train
        )
        window = _train_window_predictions(
            forecaster, y_train, None, n_rows=168, method="predict_interval", coverage_rates=[0.9]
        )
        np.testing.assert_array_equal(window.positions, np.arange(664, 832))

    def test_sliding_window_uses_relative_positions(self):
        y = _hourly(2000)
        train = np.arange(500, 1500)
        y_train = y[train]
        forecaster = _fitted(_HoldoutStubForecaster(holdout=0, seasonality=24), y_train)
        window = _train_window_predictions(forecaster, y_train, None, n_rows=168, method="predict")
        np.testing.assert_array_equal(train[window.positions], np.arange(1332, 1500))
        assert window.y_scored["time"].equals(y["time"][1332:1500])

    def test_score_params_follow_scored_rows(self, monkeypatch):
        y = _hourly(2000)
        train = np.arange(500, 1500)
        y_train = y[train]
        forecaster = _fitted(_HoldoutStubForecaster(holdout=168, seasonality=24), y_train)
        window = _train_window_predictions(forecaster, y_train, None, n_rows=168, method="predict")
        captured = {}

        def fake_score(forecaster, y_train, y_test, y_pred, scorer, score_params, error_score):
            captured["params"] = score_params
            return 0.0

        monkeypatch.setattr(ms_utils, "_score", fake_score)
        row_ids = np.arange(len(y))
        _score_train_window(
            forecaster,
            window,
            MeanAbsoluteError(),
            y=y,
            score_params={"row_id": row_ids},
            train=train,
            error_score="raise",
        )
        np.testing.assert_array_equal(captured["params"]["row_id"], np.arange(1164, 1332))

    def test_train_predictions_use_the_given_stride(self):
        y_train = _hourly(1000)
        forecaster = _fitted(_HoldoutStubForecaster(holdout=0, seasonality=24), y_train)
        window = _train_window_predictions(forecaster, y_train, None, n_rows=168, method="predict", predict_stride=24)
        vintages = window.y_pred["vintage_time"].unique().sort()
        gaps = vintages.diff().drop_nulls().unique()
        assert gaps.to_list() == [timedelta(hours=24)]


class TestUnavailable:
    """Too few learned-from rows gives NaN and a warning, never other rows."""

    def test_holdout_leaves_no_room(self):
        y_train = _hourly(300)
        forecaster = _fitted(_HoldoutStubForecaster(holdout=168, seasonality=24), y_train)
        with pytest.warns(UserWarning, match=r"\(300 rows\).*\(168 rows\).*\(168\)"):
            window = _train_window_predictions(forecaster, y_train, None, n_rows=168, method="predict")
        assert window is None

    def test_short_training_window(self):
        y_train = _hourly(168)
        forecaster = _fitted(_HoldoutStubForecaster(holdout=0, seasonality=24), y_train)
        with pytest.warns(UserWarning, match="Train score is unavailable"):
            window = _train_window_predictions(forecaster, y_train, None, n_rows=168, method="predict")
        assert window is None

    def test_unavailable_window_scores_nan_per_scorer(self):
        scorer = _MultimetricScorer(scorers={"a": MeanAbsoluteError(), "b": MeanAbsoluteError()})
        y = _hourly(10)
        scores = _score_train_window(
            None, None, scorer, y=y, score_params=None, train=np.arange(10), error_score="raise"
        )
        assert set(scores) == {"a", "b"}
        assert all(np.isnan(v) for v in scores.values())


class TestCrossValidateAgreement:
    """Cross-validation computes its train score through the same recipe."""

    def _conformal(self):
        return SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=24), calibration_size=168)

    def test_cross_validate_matches_hand_computation(self):
        y = _hourly(1400)
        scorer = IntervalScore(coverage_rates=[0.9])
        cv = ExpandingWindowSplitter(n_splits=2, test_size=168)
        results = cross_validate(
            self._conformal(), y, forecasting_horizon=FH, scoring=scorer, cv=cv, return_train_score=True
        )
        for split, (train, _) in enumerate(cv.split(y)):
            y_train = y[train]
            forecaster = self._conformal().fit(y_train, forecasting_horizon=FH)
            n_before = len(train) - 168 - 168
            forecaster.rewind(y_train[:n_before])
            y_scored = y_train[n_before : n_before + 168]
            y_pred = forecaster.observe_predict_interval(y_scored, coverage_rates=[0.9])
            y_pred = y_pred.unique(subset=["vintage_time", "time"], keep="last").sort(["time", "vintage_time"])
            hand_scorer = IntervalScore(coverage_rates=[0.9]).fit(y_train[:n_before], forecaster=forecaster)
            # Cross-validation reports lower-is-better scores negated (sklearn's sign convention).
            expected = -hand_scorer(y_scored, y_pred)
            assert results["train_score"][split] == pytest.approx(expected)

    def test_two_scorers_share_one_walk_forward(self, monkeypatch):
        y = _hourly(1400)
        scoring = {"interval": IntervalScore(coverage_rates=[0.9]), "coverage": EmpiricalCoverage(coverage_rates=[0.9])}
        cv = ExpandingWindowSplitter(n_splits=2, test_size=168)
        results = cross_validate(
            self._conformal(), y, forecasting_horizon=FH, scoring=scoring, cv=cv, return_train_score=True
        )
        train, _ = next(iter(cv.split(y)))  # the first split
        y_train = y[train]
        forecaster = self._conformal().fit(y_train, forecasting_horizon=FH)

        calls = []
        original = forecaster.observe_predict_interval

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(forecaster, "observe_predict_interval", counting)
        window = _train_window_predictions(
            forecaster, y_train, None, n_rows=168, method="predict_interval", coverage_rates=[0.9]
        )
        for name, scorer in scoring.items():
            score = _score_train_window(
                forecaster, window, scorer, y=y, score_params=None, train=train, error_score="raise"
            )
            assert score == pytest.approx(results[f"train_{name}"][0])
        assert len(calls) == 1
