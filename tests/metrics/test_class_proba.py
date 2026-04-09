"""Tests for class-probability forecasting metrics."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from conftest import run_checks as _run_checks_base
from yohou.metrics import Accuracy, BrierScore, LogLoss
from yohou.testing import _yield_yohou_scorer_checks


@pytest.fixture
def class_proba_data():
    """Ground truth (string labels) and predicted probabilities for 3 classes."""
    dates = [datetime(2020, 1, i) for i in range(1, 6)]
    y_true = pl.DataFrame({
        "time": dates,
        "target": ["cat_a", "cat_b", "cat_c", "cat_a", "cat_b"],
    })
    y_pred = pl.DataFrame({
        "observed_time": [datetime(2019, 12, 31)] * 5,
        "time": dates,
        "target_proba_cat_a": [0.7, 0.1, 0.1, 0.8, 0.2],
        "target_proba_cat_b": [0.2, 0.8, 0.2, 0.1, 0.6],
        "target_proba_cat_c": [0.1, 0.1, 0.7, 0.1, 0.2],
    })
    return y_true, y_pred


@pytest.fixture
def perfect_proba_data():
    """Perfect predictions: probability 1.0 for the true class."""
    dates = [datetime(2020, 1, i) for i in range(1, 4)]
    y_true = pl.DataFrame({
        "time": dates,
        "target": ["cat_a", "cat_b", "cat_c"],
    })
    y_pred = pl.DataFrame({
        "observed_time": [datetime(2019, 12, 31)] * 3,
        "time": dates,
        "target_proba_cat_a": [1.0, 0.0, 0.0],
        "target_proba_cat_b": [0.0, 1.0, 0.0],
        "target_proba_cat_c": [0.0, 0.0, 1.0],
    })
    return y_true, y_pred


@pytest.fixture
def uniform_proba_data():
    """Uniform (worst non-degenerate) predictions: equal probability for all classes."""
    dates = [datetime(2020, 1, i) for i in range(1, 4)]
    y_true = pl.DataFrame({
        "time": dates,
        "target": ["cat_a", "cat_b", "cat_c"],
    })
    y_pred = pl.DataFrame({
        "observed_time": [datetime(2019, 12, 31)] * 3,
        "time": dates,
        "target_proba_cat_a": [1.0 / 3, 1.0 / 3, 1.0 / 3],
        "target_proba_cat_b": [1.0 / 3, 1.0 / 3, 1.0 / 3],
        "target_proba_cat_c": [1.0 / 3, 1.0 / 3, 1.0 / 3],
    })
    return y_true, y_pred


def run_checks(scorer, y_truth, y_pred):
    """Run all systematic checks for a scorer."""
    _run_checks_base(scorer, _yield_yohou_scorer_checks(scorer, y_truth, y_pred))


class TestSystematicChecks:
    """Systematic check-generator tests for all class-probability scorers."""

    def test_log_loss_systematic(self, class_proba_data):
        """Run systematic checks for LogLoss."""
        y_truth, y_pred = class_proba_data
        scorer = LogLoss()
        scorer.fit(y_truth)
        run_checks(scorer, y_truth, y_pred)

    def test_brier_score_systematic(self, class_proba_data):
        """Run systematic checks for BrierScore."""
        y_truth, y_pred = class_proba_data
        scorer = BrierScore()
        scorer.fit(y_truth)
        run_checks(scorer, y_truth, y_pred)

    def test_categorical_accuracy_systematic(self, class_proba_data):
        """Run systematic checks for Accuracy."""
        y_truth, y_pred = class_proba_data
        scorer = Accuracy()
        scorer.fit(y_truth)
        run_checks(scorer, y_truth, y_pred)


class TestLogLoss:
    """Tests for LogLoss scorer."""

    def test_perfect_predictions_near_zero(self, perfect_proba_data):
        """Perfect predictions should give near-zero log loss."""
        y_true, y_pred = perfect_proba_data
        scorer = LogLoss()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert np.isclose(score, 0.0, atol=1e-10)

    def test_uniform_predictions_above_zero(self, uniform_proba_data):
        """Uniform predictions should give positive log loss."""
        y_true, y_pred = uniform_proba_data
        scorer = LogLoss()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert score > 0

    def test_uniform_equals_log_n_classes(self, uniform_proba_data):
        """Uniform predictions should give -log(1/K) = log(K) for K classes."""
        y_true, y_pred = uniform_proba_data
        scorer = LogLoss()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert np.isclose(score, np.log(3), rtol=1e-5)

    def test_lower_is_better(self, class_proba_data, perfect_proba_data):
        """Better predictions should have lower log loss."""
        y_true, y_pred = class_proba_data
        y_true_perf, y_pred_perf = perfect_proba_data

        scorer = LogLoss()
        scorer.fit(y_true)
        score_imperfect = scorer.score(y_true, y_pred)

        scorer_perf = LogLoss()
        scorer_perf.fit(y_true_perf)
        score_perfect = scorer_perf.score(y_true_perf, y_pred_perf)

        assert score_perfect < score_imperfect

    def test_not_fitted_raises(self, class_proba_data):
        """Scoring before fit should raise NotFittedError."""
        y_true, y_pred = class_proba_data
        scorer = LogLoss()
        with pytest.raises(NotFittedError):
            scorer.score(y_true, y_pred)

    def test_tags(self):
        """Tags should report class_proba prediction type."""
        scorer = LogLoss()
        tags = scorer.__sklearn_tags__()
        assert tags.scorer_tags.prediction_type == "class_proba"
        assert tags.scorer_tags.lower_is_better is True


class TestBrierScore:
    """Tests for BrierScore scorer."""

    def test_perfect_predictions_zero(self, perfect_proba_data):
        """Perfect predictions should give zero Brier score."""
        y_true, y_pred = perfect_proba_data
        scorer = BrierScore()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert np.isclose(score, 0.0, atol=1e-10)

    def test_worst_predictions_positive(self, class_proba_data):
        """Imperfect predictions should give positive Brier score."""
        y_true, y_pred = class_proba_data
        scorer = BrierScore()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert score > 0

    def test_lower_is_better(self, class_proba_data, perfect_proba_data):
        """Better predictions should have lower Brier score."""
        y_true, y_pred = class_proba_data
        y_true_perf, y_pred_perf = perfect_proba_data

        scorer = BrierScore()
        scorer.fit(y_true)
        score_imperfect = scorer.score(y_true, y_pred)

        scorer_perf = BrierScore()
        scorer_perf.fit(y_true_perf)
        score_perfect = scorer_perf.score(y_true_perf, y_pred_perf)

        assert score_perfect < score_imperfect

    def test_not_fitted_raises(self, class_proba_data):
        """Scoring before fit should raise NotFittedError."""
        y_true, y_pred = class_proba_data
        scorer = BrierScore()
        with pytest.raises(NotFittedError):
            scorer.score(y_true, y_pred)

    def test_tags(self):
        """Tags should report class_proba prediction type."""
        scorer = BrierScore()
        tags = scorer.__sklearn_tags__()
        assert tags.scorer_tags.prediction_type == "class_proba"
        assert tags.scorer_tags.lower_is_better is True


class TestAccuracy:
    """Tests for Accuracy scorer."""

    def test_perfect_predictions_one(self, perfect_proba_data):
        """Perfect predictions should give accuracy of 1.0."""
        y_true, y_pred = perfect_proba_data
        scorer = Accuracy()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert np.isclose(score, 1.0)

    def test_good_predictions_above_zero(self, class_proba_data):
        """Good predictions should have positive accuracy."""
        y_true, y_pred = class_proba_data
        scorer = Accuracy()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert score > 0

    def test_higher_is_better(self, perfect_proba_data):
        """Better predictions should have higher accuracy."""
        y_true_perf, y_pred_perf = perfect_proba_data

        # Create bad predictions where argmax is wrong
        dates = [datetime(2020, 1, i) for i in range(1, 4)]
        y_true_bad = pl.DataFrame({
            "time": dates,
            "target": ["cat_a", "cat_b", "cat_c"],
        })
        y_pred_bad = pl.DataFrame({
            "observed_time": [datetime(2019, 12, 31)] * 3,
            "time": dates,
            "target_proba_cat_a": [0.1, 0.5, 0.5],  # wrong for cat_a
            "target_proba_cat_b": [0.5, 0.1, 0.1],  # wrong for cat_b
            "target_proba_cat_c": [0.4, 0.4, 0.4],  # wrong for cat_c
        })

        scorer_perf = Accuracy()
        scorer_perf.fit(y_true_perf)
        score_perfect = scorer_perf.score(y_true_perf, y_pred_perf)

        scorer_bad = Accuracy()
        scorer_bad.fit(y_true_bad)
        score_bad = scorer_bad.score(y_true_bad, y_pred_bad)

        assert score_perfect > score_bad

    def test_lower_is_better_false(self):
        """Accuracy has lower_is_better=False (higher accuracy is better)."""
        scorer = Accuracy()
        tags = scorer.__sklearn_tags__()
        assert tags.scorer_tags.lower_is_better is False

    def test_not_fitted_raises(self, class_proba_data):
        """Scoring before fit should raise NotFittedError."""
        y_true, y_pred = class_proba_data
        scorer = Accuracy()
        with pytest.raises(NotFittedError):
            scorer.score(y_true, y_pred)

    def test_tags(self):
        """Tags should report class_proba prediction type."""
        scorer = Accuracy()
        tags = scorer.__sklearn_tags__()
        assert tags.scorer_tags.prediction_type == "class_proba"
        assert tags.scorer_tags.lower_is_better is False


class TestMultiTarget:
    """Tests for multi-target class probability scoring."""

    @pytest.fixture
    def multi_target_data(self):
        """Multi-target data with two categorical columns."""
        dates = [datetime(2020, 1, i) for i in range(1, 4)]
        y_true = pl.DataFrame({
            "time": dates,
            "weather": ["sunny", "rainy", "cloudy"],
            "mood": ["happy", "sad", "happy"],
        })
        y_pred = pl.DataFrame({
            "observed_time": [datetime(2019, 12, 31)] * 3,
            "time": dates,
            "weather_proba_sunny": [0.7, 0.1, 0.2],
            "weather_proba_rainy": [0.2, 0.8, 0.1],
            "weather_proba_cloudy": [0.1, 0.1, 0.7],
            "mood_proba_happy": [0.8, 0.2, 0.9],
            "mood_proba_sad": [0.2, 0.8, 0.1],
        })
        return y_true, y_pred

    def test_log_loss_multi_target(self, multi_target_data):
        """LogLoss works with multiple categorical targets."""
        y_true, y_pred = multi_target_data
        scorer = LogLoss()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert isinstance(score, float)
        assert score > 0

    def test_brier_score_multi_target(self, multi_target_data):
        """BrierScore works with multiple categorical targets."""
        y_true, y_pred = multi_target_data
        scorer = BrierScore()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert isinstance(score, float)
        assert score > 0

    def test_categorical_accuracy_multi_target(self, multi_target_data):
        """Accuracy works with multiple categorical targets."""
        y_true, y_pred = multi_target_data
        scorer = Accuracy()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert isinstance(score, float)
        assert score > 0
