"""Tests for hard-label and ranking classification metrics."""

import warnings
from datetime import datetime

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from conftest import run_checks as _run_checks_base
from yohou.metrics import Accuracy, get_scorer, make_scorer
from yohou.metrics.classification import (
    FBetaScore,
    PRAuC,
    Precision,
    Recall,
    ROCAuC,
)
from yohou.testing import _yield_yohou_scorer_checks
from yohou.weighting import TableWeighter


@pytest.fixture
def class_data_5rows():
    """5-row data with 3 classes, 1 wrong prediction (row 3)."""
    dates = [datetime(2020, 1, i) for i in range(1, 6)]
    y_true = pl.DataFrame({
        "time": dates,
        "weather": ["sunny", "rainy", "cloudy", "sunny", "rainy"],
    })
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * 5,
        "time": dates,
        "weather_proba_sunny": [0.7, 0.1, 0.2, 0.2, 0.1],
        "weather_proba_rainy": [0.2, 0.8, 0.1, 0.1, 0.8],
        "weather_proba_cloudy": [0.1, 0.1, 0.7, 0.7, 0.1],
    })
    return y_true, y_pred


@pytest.fixture
def panel_data():
    """Panel data with 2 groups."""
    dates = [datetime(2020, 1, i) for i in range(1, 4)]
    y_true = pl.DataFrame({
        "time": dates,
        "grpA__weather": ["sunny", "rainy", "cloudy"],
        "grpB__weather": ["cloudy", "sunny", "rainy"],
    })
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * 3,
        "time": dates,
        "grpA__weather_proba_sunny": [0.7, 0.1, 0.2],
        "grpA__weather_proba_rainy": [0.2, 0.8, 0.1],
        "grpA__weather_proba_cloudy": [0.1, 0.1, 0.7],
        "grpB__weather_proba_sunny": [0.2, 0.7, 0.1],
        "grpB__weather_proba_rainy": [0.1, 0.2, 0.8],
        "grpB__weather_proba_cloudy": [0.7, 0.1, 0.1],
    })
    return y_true, y_pred


@pytest.fixture
def panel_data_differentiated():
    """Panel data where grpA predicts perfectly and grpB predicts inverted.

    This makes each group's ranking score distinct (grpA high, grpB low), so a
    panel subselection differs from the full-panel score.
    """
    dates = [datetime(2020, 1, i) for i in range(1, 5)]
    y_true = pl.DataFrame({
        "time": dates,
        "grpA__weather": ["sunny", "rainy", "sunny", "rainy"],
        "grpB__weather": ["sunny", "rainy", "sunny", "rainy"],
    })
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * 4,
        "time": dates,
        # grpA: confident and correct.
        "grpA__weather_proba_sunny": [0.9, 0.1, 0.8, 0.2],
        "grpA__weather_proba_rainy": [0.1, 0.9, 0.2, 0.8],
        # grpB: confident and inverted (wrong ranking).
        "grpB__weather_proba_sunny": [0.1, 0.9, 0.2, 0.8],
        "grpB__weather_proba_rainy": [0.9, 0.1, 0.8, 0.2],
    })
    return y_true, y_pred


def _sklearn_labels(y_true_df, y_pred_df, target="weather"):
    """Extract sklearn-format labels from yohou DataFrames."""
    true_labels = y_true_df[target].to_list()
    proba_cols = [c for c in y_pred_df.columns if c.startswith(f"{target}_proba_")]
    proba_arr = y_pred_df.select(proba_cols).to_numpy()
    class_labels = [c.split("_proba_", 1)[1] for c in proba_cols]
    pred_labels = [class_labels[i] for i in np.argmax(proba_arr, axis=1)]
    return true_labels, pred_labels


def run_checks(scorer, y_truth, y_pred):
    """Run all systematic checks for a scorer."""
    _run_checks_base(scorer, _yield_yohou_scorer_checks(scorer, y_truth, y_pred))


class TestPrecisionSystematic:
    def test_systematic_checks(self, class_data_5rows):
        y_true, y_pred = class_data_5rows
        run_checks(Precision(), y_true, y_pred)


class TestRecallSystematic:
    def test_systematic_checks(self, class_data_5rows):
        y_true, y_pred = class_data_5rows
        run_checks(Recall(), y_true, y_pred)


class TestFBetaScoreSystematic:
    def test_systematic_checks(self, class_data_5rows):
        y_true, y_pred = class_data_5rows
        run_checks(FBetaScore(beta=1.0), y_true, y_pred)


class TestROCAuCSystematic:
    def test_systematic_checks(self, class_data_5rows):
        y_true, y_pred = class_data_5rows
        run_checks(ROCAuC(), y_true, y_pred)


class TestPRAuCSystematic:
    def test_systematic_checks(self, class_data_5rows):
        y_true, y_pred = class_data_5rows
        run_checks(PRAuC(), y_true, y_pred)


class TestPrecision:
    def test_perfect_prediction(self):
        dates = [datetime(2020, 1, i) for i in range(1, 4)]
        y_true = pl.DataFrame({"time": dates, "t": ["a", "b", "c"]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": dates,
            "t_proba_a": [0.9, 0.05, 0.05],
            "t_proba_b": [0.05, 0.9, 0.05],
            "t_proba_c": [0.05, 0.05, 0.9],
        })
        scorer = Precision()
        scorer.fit(y_true)
        assert np.isclose(scorer.score(y_true, y_pred), 1.0)

    def test_matches_sklearn_macro(self, class_data_5rows):
        y_true, y_pred = class_data_5rows
        true_labels, pred_labels = _sklearn_labels(y_true, y_pred)
        expected = precision_score(true_labels, pred_labels, average="macro", zero_division=0.0)
        scorer = Precision(average="macro")
        scorer.fit(y_true)
        assert np.isclose(scorer.score(y_true, y_pred), expected, atol=1e-10)

    def test_matches_sklearn_micro(self, class_data_5rows):
        y_true, y_pred = class_data_5rows
        true_labels, pred_labels = _sklearn_labels(y_true, y_pred)
        expected = precision_score(true_labels, pred_labels, average="micro", zero_division=0.0)
        scorer = Precision(average="micro")
        scorer.fit(y_true)
        assert np.isclose(scorer.score(y_true, y_pred), expected, atol=1e-10)

    def test_matches_sklearn_weighted(self, class_data_5rows):
        y_true, y_pred = class_data_5rows
        true_labels, pred_labels = _sklearn_labels(y_true, y_pred)
        expected = precision_score(true_labels, pred_labels, average="weighted", zero_division=0.0)
        scorer = Precision(average="weighted")
        scorer.fit(y_true)
        assert np.isclose(scorer.score(y_true, y_pred), expected, atol=1e-10)

    def test_zero_division(self):
        dates = [datetime(2020, 1, i) for i in range(1, 3)]
        y_true = pl.DataFrame({"time": dates, "t": ["a", "a"]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 2,
            "time": dates,
            "t_proba_a": [0.9, 0.9],
            "t_proba_b": [0.1, 0.1],
        })
        # b is never predicted AND never true, so FP=0, TP=0 → zero division
        scorer = Precision(average="macro", zero_division=0.0)
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert isinstance(score, float)

    def test_panel_data(self, panel_data):
        y_true, y_pred = panel_data
        scorer = Precision()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestRecall:
    def test_matches_sklearn_macro(self, class_data_5rows):
        y_true, y_pred = class_data_5rows
        true_labels, pred_labels = _sklearn_labels(y_true, y_pred)
        expected = recall_score(true_labels, pred_labels, average="macro", zero_division=0.0)
        scorer = Recall(average="macro")
        scorer.fit(y_true)
        assert np.isclose(scorer.score(y_true, y_pred), expected, atol=1e-10)

    def test_matches_sklearn_micro(self, class_data_5rows):
        y_true, y_pred = class_data_5rows
        true_labels, pred_labels = _sklearn_labels(y_true, y_pred)
        expected = recall_score(true_labels, pred_labels, average="micro", zero_division=0.0)
        scorer = Recall(average="micro")
        scorer.fit(y_true)
        assert np.isclose(scorer.score(y_true, y_pred), expected, atol=1e-10)

    def test_matches_sklearn_weighted(self, class_data_5rows):
        y_true, y_pred = class_data_5rows
        true_labels, pred_labels = _sklearn_labels(y_true, y_pred)
        expected = recall_score(true_labels, pred_labels, average="weighted", zero_division=0.0)
        scorer = Recall(average="weighted")
        scorer.fit(y_true)
        assert np.isclose(scorer.score(y_true, y_pred), expected, atol=1e-10)


class TestFBetaScore:
    def test_f1_matches_sklearn(self, class_data_5rows):
        y_true, y_pred = class_data_5rows
        true_labels, pred_labels = _sklearn_labels(y_true, y_pred)
        expected = f1_score(true_labels, pred_labels, average="macro", zero_division=0.0)
        scorer = FBetaScore(beta=1.0, average="macro")
        scorer.fit(y_true)
        assert np.isclose(scorer.score(y_true, y_pred), expected, atol=1e-10)

    def test_f2_higher_recall_weight(self, class_data_5rows):
        y_true, y_pred = class_data_5rows
        scorer_f1 = FBetaScore(beta=1.0)
        scorer_f1.fit(y_true)
        scorer_f2 = FBetaScore(beta=2.0)
        scorer_f2.fit(y_true)
        # F2 weights recall more, result differs from F1
        assert not np.isclose(scorer_f1.score(y_true, y_pred), scorer_f2.score(y_true, y_pred))

    def test_metric_name_f1(self):
        assert FBetaScore(beta=1.0)._metric_name == "f1"

    def test_metric_name_f2(self):
        assert FBetaScore(beta=2.0)._metric_name == "f2"

    def test_metric_name_fbeta(self):
        assert FBetaScore(beta=0.5)._metric_name == "fbeta"

    def test_metric_name_in_dataframe_output(self):
        dates = [datetime(2020, 1, i) for i in range(1, 4)]
        y_true = pl.DataFrame({"time": dates, "t": ["a", "b", "c"]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": dates,
            "t_proba_a": [0.8, 0.1, 0.1],
            "t_proba_b": [0.1, 0.8, 0.1],
            "t_proba_c": [0.1, 0.1, 0.8],
        })
        scorer = FBetaScore(beta=1.0, aggregation_method=["stepwise", "componentwise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)
        assert "f1" in result.columns


class TestAccuracyMigration:
    def test_perfect_prediction(self):
        dates = [datetime(2020, 1, i) for i in range(1, 4)]
        y_true = pl.DataFrame({"time": dates, "t": ["a", "b", "c"]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": dates,
            "t_proba_a": [0.9, 0.05, 0.05],
            "t_proba_b": [0.05, 0.9, 0.05],
            "t_proba_c": [0.05, 0.05, 0.9],
        })
        scorer = Accuracy()
        scorer.fit(y_true)
        assert np.isclose(scorer.score(y_true, y_pred), 1.0)

    def test_all_wrong(self):
        dates = [datetime(2020, 1, i) for i in range(1, 4)]
        y_true = pl.DataFrame({"time": dates, "t": ["a", "b", "c"]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": dates,
            "t_proba_a": [0.1, 0.5, 0.5],
            "t_proba_b": [0.5, 0.1, 0.1],
            "t_proba_c": [0.4, 0.4, 0.4],
        })
        scorer = Accuracy()
        scorer.fit(y_true)
        assert np.isclose(scorer.score(y_true, y_pred), 0.0)

    def test_partial(self, class_data_5rows):
        y_true, y_pred = class_data_5rows
        scorer = Accuracy()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        # 4 out of 5 correct
        assert np.isclose(score, 0.8)

    def test_panel(self, panel_data):
        y_true, y_pred = panel_data
        scorer = Accuracy()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_no_average_param_exposed(self):
        """Accuracy should not expose average parameter."""
        with pytest.raises(TypeError):
            Accuracy(average="macro")

    def test_registry_lookup(self):
        scorer = get_scorer("accuracy")
        assert isinstance(scorer, Accuracy)


class TestROCAuC:
    def test_perfect_prediction(self):
        dates = [datetime(2020, 1, i) for i in range(1, 6)]
        y_true = pl.DataFrame({
            "time": dates,
            "t": ["a", "b", "c", "a", "b"],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": dates,
            "t_proba_a": [0.9, 0.05, 0.05, 0.9, 0.05],
            "t_proba_b": [0.05, 0.9, 0.05, 0.05, 0.9],
            "t_proba_c": [0.05, 0.05, 0.9, 0.05, 0.05],
        })
        scorer = ROCAuC()
        scorer.fit(y_true)
        assert np.isclose(scorer.score(y_true, y_pred), 1.0)

    def test_matches_sklearn(self, class_data_5rows):
        y_true, y_pred = class_data_5rows
        true_labels = y_true["weather"].to_list()
        proba_cols = [c for c in y_pred.columns if c.startswith("weather_proba_")]
        proba_arr = y_pred.select(proba_cols).to_numpy()
        class_labels = [c.split("_proba_", 1)[1] for c in proba_cols]

        # Compute per-class OvR AUC like our scorer does
        per_class_aucs = []
        for i, cls in enumerate(class_labels):
            y_binary = np.array([1.0 if label == cls else 0.0 for label in true_labels])
            if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
                continue
            per_class_aucs.append(roc_auc_score(y_binary, proba_arr[:, i]))
        expected = np.mean(per_class_aucs)

        scorer = ROCAuC(average="macro")
        scorer.fit(y_true)
        assert np.isclose(scorer.score(y_true, y_pred), expected, atol=1e-10)

    def test_panel_data(self, panel_data):
        y_true, y_pred = panel_data
        scorer = ROCAuC()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("group", ["grpA", "grpB"])
    def test_panel_subselection(self, panel_data_differentiated, group):
        """groups=[...] restricts the score to the named group for ranking scorers."""
        y_true, y_pred = panel_data_differentiated

        full = ROCAuC()
        full.fit(y_true)
        full_score = full.score(y_true, y_pred)

        subselected = ROCAuC(groups=[group])
        subselected.fit(y_true)
        sub_score = subselected.score(y_true, y_pred)

        assert isinstance(sub_score, float)
        assert 0.0 <= sub_score <= 1.0
        # grpA is perfect, grpB inverted; neither matches the full-panel average.
        assert sub_score != full_score
        expected = 1.0 if group == "grpA" else 0.0
        assert np.isclose(sub_score, expected)


class TestPRAuC:
    def test_matches_sklearn(self, class_data_5rows):
        y_true, y_pred = class_data_5rows
        true_labels = y_true["weather"].to_list()
        proba_cols = [c for c in y_pred.columns if c.startswith("weather_proba_")]
        proba_arr = y_pred.select(proba_cols).to_numpy()
        class_labels = [c.split("_proba_", 1)[1] for c in proba_cols]

        per_class = []
        for i, cls in enumerate(class_labels):
            y_binary = np.array([1.0 if label == cls else 0.0 for label in true_labels])
            if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
                continue
            per_class.append(average_precision_score(y_binary, proba_arr[:, i]))
        expected = np.mean(per_class)

        scorer = PRAuC(average="macro")
        scorer.fit(y_true)
        assert np.isclose(scorer.score(y_true, y_pred), expected, atol=1e-10)

    def test_panel_data(self, panel_data):
        y_true, y_pred = panel_data
        scorer = PRAuC()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("group", ["grpA", "grpB"])
    def test_panel_subselection(self, panel_data_differentiated, group):
        """groups=[...] restricts the score to the named group for ranking scorers."""
        y_true, y_pred = panel_data_differentiated

        full = PRAuC()
        full.fit(y_true)
        full_score = full.score(y_true, y_pred)

        subselected = PRAuC(groups=[group])
        subselected.fit(y_true)
        sub_score = subselected.score(y_true, y_pred)

        assert isinstance(sub_score, float)
        assert 0.0 <= sub_score <= 1.0
        # The two groups score differently, so neither matches the full-panel value.
        assert sub_score != full_score
        if group == "grpA":
            assert np.isclose(sub_score, 1.0)


class TestRegistry:
    @pytest.mark.parametrize(
        "name,cls",
        [
            ("precision", Precision),
            ("recall", Recall),
            ("fbeta", FBetaScore),
            ("f1", FBetaScore),
            ("roc_auc", ROCAuC),
            ("pr_auc", PRAuC),
        ],
    )
    def test_get_scorer(self, name, cls):
        scorer = get_scorer(name)
        assert isinstance(scorer, cls)

    def test_make_scorer_fbeta_with_beta(self):
        scorer = make_scorer("fbeta", beta=2.0)
        assert scorer.beta == 2.0

    def test_f1_alias_is_fbeta(self):
        scorer = get_scorer("f1")
        assert isinstance(scorer, FBetaScore)
        assert scorer.beta == 1.0


class TestPartialAggregation:
    def test_precision_stepwise(self, class_data_5rows):
        y_true, y_pred = class_data_5rows
        scorer = Precision(aggregation_method=["stepwise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)
        # Without componentwise, column keeps target name
        assert "weather" in result.columns

    def test_recall_componentwise(self):
        dates = [datetime(2020, 1, i) for i in range(1, 4)]
        y_true = pl.DataFrame({
            "time": dates,
            "mood": ["happy", "sad", "happy"],
            "color": ["red", "blue", "red"],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": dates,
            "mood_proba_happy": [0.8, 0.2, 0.9],
            "mood_proba_sad": [0.2, 0.8, 0.1],
            "color_proba_red": [0.9, 0.1, 0.8],
            "color_proba_blue": [0.1, 0.9, 0.2],
        })
        scorer = Recall(aggregation_method=["stepwise", "vintagewise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)
        assert "mood" in result.columns or "recall" in result.columns


class TestNaNInfProbabilities:
    def test_nan_probabilities_raise(self):
        """NaN in probability columns should raise ValueError."""
        dates = [datetime(2020, 1, i) for i in range(1, 4)]
        y_true = pl.DataFrame({"time": dates, "weather": ["sunny", "rainy", "cloudy"]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": dates,
            "weather_proba_sunny": [0.7, float("nan"), 0.2],
            "weather_proba_rainy": [0.2, 0.5, 0.1],
            "weather_proba_cloudy": [0.1, 0.5, 0.7],
        })
        scorer = Precision()
        scorer.fit(y_true)
        with pytest.raises(ValueError, match="NaN or infinite"):
            scorer.score(y_true, y_pred)

    def test_inf_probabilities_raise(self):
        """Inf in probability columns should raise ValueError."""
        dates = [datetime(2020, 1, i) for i in range(1, 4)]
        y_true = pl.DataFrame({"time": dates, "weather": ["sunny", "rainy", "cloudy"]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": dates,
            "weather_proba_sunny": [0.7, float("inf"), 0.2],
            "weather_proba_rainy": [0.2, 0.5, 0.1],
            "weather_proba_cloudy": [0.1, 0.5, 0.7],
        })
        scorer = ROCAuC()
        scorer.fit(y_true)
        with pytest.raises(ValueError, match="NaN or infinite"):
            scorer.score(y_true, y_pred)


class TestRankingWeightedAveraging:
    def test_roc_auc_weighted(self):
        """ROCAuC with average='weighted' should use support-weighted mean."""
        dates = [datetime(2020, 1, i) for i in range(1, 11)]
        # Imbalanced: 6 sunny, 3 rainy, 1 cloudy
        labels = ["sunny"] * 6 + ["rainy"] * 3 + ["cloudy"]
        y_true = pl.DataFrame({"time": dates, "weather": labels})

        np.random.seed(42)
        proba = np.random.dirichlet([1, 1, 1], size=10)
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 10,
            "time": dates,
            "weather_proba_sunny": proba[:, 0].tolist(),
            "weather_proba_rainy": proba[:, 1].tolist(),
            "weather_proba_cloudy": proba[:, 2].tolist(),
        })

        scorer_w = ROCAuC(average="weighted")
        scorer_w.fit(y_true)
        score_w = scorer_w.score(y_true, y_pred)

        scorer_m = ROCAuC(average="macro")
        scorer_m.fit(y_true)
        score_m = scorer_m.score(y_true, y_pred)

        # Weighted and macro should differ on imbalanced data
        assert isinstance(score_w, float)
        assert isinstance(score_m, float)
        assert score_w != score_m


class TestRankingCombinedWeights:
    def test_roc_auc_with_callable_time_weight(self):
        """ROCAuC with callable time_weight applies sample weights."""
        dates = [datetime(2020, 1, i) for i in range(1, 6)]
        y_true = pl.DataFrame({
            "time": dates,
            "weather": ["sunny", "rainy", "cloudy", "sunny", "rainy"],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": dates,
            "weather_proba_sunny": [0.7, 0.1, 0.2, 0.6, 0.1],
            "weather_proba_rainy": [0.2, 0.8, 0.1, 0.3, 0.8],
            "weather_proba_cloudy": [0.1, 0.1, 0.7, 0.1, 0.1],
        })
        weight_frame = pl.DataFrame({"time": dates, "weight": [1.0, 2.0, 1.0, 2.0, 1.0]})
        scorer = ROCAuC(time_weighter=TableWeighter(frame=weight_frame, on="time"))
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert isinstance(score, float)


class TestRankingSkipConstantClasses:
    def test_roc_auc_skips_absent_class(self):
        """ROCAuC should skip classes with no positive samples."""
        dates = [datetime(2020, 1, i) for i in range(1, 6)]
        # "cloudy" never appears in truth
        y_true = pl.DataFrame({
            "time": dates,
            "weather": ["sunny", "rainy", "sunny", "rainy", "sunny"],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": dates,
            "weather_proba_sunny": [0.7, 0.1, 0.6, 0.2, 0.8],
            "weather_proba_rainy": [0.2, 0.8, 0.3, 0.7, 0.1],
            "weather_proba_cloudy": [0.1, 0.1, 0.1, 0.1, 0.1],
        })
        scorer = ROCAuC()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        # Should compute without error, skipping the absent class
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_roc_auc_all_same_class_returns_zero(self):
        """ROCAuC should return 0.0 when all samples have the same label."""
        dates = [datetime(2020, 1, i) for i in range(1, 4)]
        y_true = pl.DataFrame({
            "time": dates,
            "weather": ["sunny", "sunny", "sunny"],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 3,
            "time": dates,
            "weather_proba_sunny": [0.7, 0.8, 0.9],
            "weather_proba_rainy": [0.2, 0.1, 0.05],
            "weather_proba_cloudy": [0.1, 0.1, 0.05],
        })
        scorer = ROCAuC()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        # All classes either have 0 or N positive samples → skip all → 0.0
        assert score == 0.0


class TestHardLabelPartialCollapse:
    def test_precision_stepwise_only(self, class_data_5rows):
        """Precision with stepwise only (no vintagewise) keeps a vintage axis."""
        y_true, y_pred = class_data_5rows
        scorer = Precision(aggregation_method=["stepwise", "componentwise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        # vintagewise was NOT collapsed, so vintage_time survives as the row axis;
        # componentwise renames the value column to the metric name.
        assert isinstance(result, pl.DataFrame)
        assert "vintage_time" in result.columns
        assert "precision" in result.columns
        assert "forecasting_step" not in result.columns
        # Single vintage in the fixture -> single output row.
        assert result.height == 1

    def test_recall_vintagewise_only(self, class_data_5rows):
        """Recall with vintagewise only keeps a forecasting_step axis."""
        y_true, y_pred = class_data_5rows
        scorer = Recall(aggregation_method=["vintagewise", "componentwise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        # stepwise was NOT collapsed, so forecasting_step survives as the row axis.
        assert isinstance(result, pl.DataFrame)
        assert "forecasting_step" in result.columns
        assert "recall" in result.columns

    def test_precision_stepwise_no_vintage(self):
        """Precision with stepwise collapse on data without vintage_time."""
        dates = [datetime(2020, 1, i) for i in range(1, 6)]
        y_true = pl.DataFrame({
            "time": dates,
            "weather": ["sunny", "rainy", "cloudy", "sunny", "rainy"],
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "weather_proba_sunny": [0.7, 0.1, 0.2, 0.6, 0.1],
            "weather_proba_rainy": [0.2, 0.8, 0.1, 0.3, 0.8],
            "weather_proba_cloudy": [0.1, 0.1, 0.7, 0.1, 0.1],
        })
        scorer = Precision(aggregation_method=["stepwise", "componentwise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        # No vintage_time column to retain, so stepwise collapse leaves a single
        # metric column with no axis columns.
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["precision"]
        assert result.height == 1
        # The single retained value is a valid precision in [0, 1].
        value = result["precision"].item()
        assert 0.0 <= value <= 1.0

    def test_precision_collapse_all_rows(self, class_data_5rows):
        """Precision with stepwise+vintagewise collapses to a single value column."""
        y_true, y_pred = class_data_5rows
        scorer = Precision(aggregation_method=["stepwise", "vintagewise"])
        scorer.fit(y_true)
        result = scorer.score(y_true, y_pred)
        # Both time axes collapsed but componentwise NOT requested, so the value
        # column keeps the target name and there is a single fully-collapsed row.
        assert isinstance(result, pl.DataFrame)
        assert "weather" in result.columns
        assert "vintage_time" not in result.columns
        assert "forecasting_step" not in result.columns
        assert result.height == 1
        value = result["weather"].item()
        assert 0.0 <= value <= 1.0


@pytest.fixture
def multi_vintage_class_data():
    """Multi-vintage classification data."""
    times = [datetime(2020, 1, i) for i in range(1, 6)]
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
        """_resolve_combined_weights warns when receiving dict weights."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ROCAuC._resolve_combined_weights(
                tw={"group_A": np.array([1.0, 2.0])},
                sw=np.array([1.0, 2.0]),
                n=2,
            )
        assert len(w) == 1
        assert "Panel-aware (dict) weights are not supported" in str(w[0].message)
        assert result is not None

    def test_roc_auc_multi_vintage(self, multi_vintage_class_data):
        """ROCAuC computes across multiple vintages."""
        y_true, y_pred = multi_vintage_class_data
        scorer = ROCAuC()
        scorer.fit(y_true)
        score = scorer.score(y_true, y_pred)
        assert isinstance(score, float)
        assert np.isfinite(score)
