"""Tests for robustness fixes: NaN validation, time weight errors, zero scale warnings."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from yohou.metrics import (
    Accuracy,
    BrierScore,
    LogLoss,
    MeanAbsoluteError,
    MeanAbsoluteScaledError,
    RootMeanSquaredScaledError,
)
from yohou.weighting import TableWeighter


@pytest.fixture()
def y_train():
    dates = [datetime(2024, 1, i) for i in range(1, 11)]
    return pl.DataFrame({"time": dates, "value": [float(i) for i in range(10)]})


@pytest.fixture()
def class_proba_train():
    dates = [datetime(2024, 1, i) for i in range(1, 11)]
    return pl.DataFrame({"time": dates, "target": ["a", "b"] * 5})


@pytest.fixture()
def class_proba_data():
    """Valid class proba test data."""
    y_true = pl.DataFrame({
        "time": [datetime(2024, 1, 11), datetime(2024, 1, 12)],
        "target": ["a", "b"],
    })
    y_pred = pl.DataFrame({
        "time": [datetime(2024, 1, 11), datetime(2024, 1, 12)],
        "target_proba_a": [0.8, 0.3],
        "target_proba_b": [0.2, 0.7],
    })
    return y_true, y_pred


class TestClassProbaNanValidation:
    """Probability columns must be finite and in [0, 1]."""

    @pytest.mark.parametrize("ScorerClass", [LogLoss, BrierScore, Accuracy])
    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf")])
    def test_non_finite_in_probabilities_raises(self, ScorerClass, bad_val, class_proba_train):
        """NaN and inf both fail the finiteness guard with the same message."""
        y_true = pl.DataFrame({
            "time": [datetime(2024, 1, 11), datetime(2024, 1, 12)],
            "target": ["a", "b"],
        })
        y_pred = pl.DataFrame({
            "time": [datetime(2024, 1, 11), datetime(2024, 1, 12)],
            "target_proba_a": [0.8, bad_val],
            "target_proba_b": [0.2, 0.7],
        })
        scorer = ScorerClass()
        scorer.fit(class_proba_train)
        with pytest.raises(ValueError, match="NaN or infinite"):
            scorer.score(y_true, y_pred)

    @pytest.mark.parametrize("ScorerClass", [LogLoss, BrierScore, Accuracy])
    def test_negative_probability_raises(self, ScorerClass, class_proba_train):
        y_true = pl.DataFrame({
            "time": [datetime(2024, 1, 11), datetime(2024, 1, 12)],
            "target": ["a", "b"],
        })
        y_pred = pl.DataFrame({
            "time": [datetime(2024, 1, 11), datetime(2024, 1, 12)],
            "target_proba_a": [-0.1, 0.3],
            "target_proba_b": [1.1, 0.7],
        })
        scorer = ScorerClass()
        scorer.fit(class_proba_train)
        with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
            scorer.score(y_true, y_pred)

    @pytest.mark.parametrize("ScorerClass", [LogLoss, BrierScore, Accuracy])
    def test_valid_probabilities_pass(self, ScorerClass, class_proba_train, class_proba_data):
        """Valid probabilities produce a score without error."""
        y_true, y_pred = class_proba_data
        scorer = ScorerClass()
        scorer.fit(class_proba_train)
        result = scorer.score(y_true, y_pred)
        assert isinstance(result, float)
        assert np.isfinite(result)


class TestTimeWeightMisalignment:
    """Time weight errors show which times are missing."""

    @pytest.mark.parametrize(
        "tw",
        [
            # Fully absent: time_weight covers wrong dates entirely.
            pl.DataFrame({
                "time": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "weight": [1.0, 2.0],
            }),
            # Partial: only one of the two scored dates is covered.
            pl.DataFrame({
                "time": [datetime(2024, 1, 11)],
                "weight": [1.0],
            }),
        ],
        ids=["fully-absent", "one-missing"],
    )
    def test_misaligned_time_weight_shows_missing_times(self, tw, y_train):
        """Times with no matching weight (all or some) report which are missing."""
        y_true = pl.DataFrame({
            "time": [datetime(2024, 1, 11), datetime(2024, 1, 12)],
            "value": [10.0, 20.0],
        })
        y_pred = pl.DataFrame({
            "time": [datetime(2024, 1, 11), datetime(2024, 1, 12)],
            "value": [12.0, 18.0],
        })
        mae = MeanAbsoluteError(time_weighter=TableWeighter(frame=tw, on="time"))
        mae.fit(y_train)
        with pytest.raises(ValueError, match="no values for times"):
            mae.score(y_true, y_pred)

    def test_zero_sum_weights_shows_message(self, y_train):
        y_true = pl.DataFrame({
            "time": [datetime(2024, 1, 11), datetime(2024, 1, 12)],
            "value": [10.0, 20.0],
        })
        y_pred = pl.DataFrame({
            "time": [datetime(2024, 1, 11), datetime(2024, 1, 12)],
            "value": [12.0, 18.0],
        })
        tw = pl.DataFrame({
            "time": [datetime(2024, 1, 11), datetime(2024, 1, 12)],
            "weight": [0.0, 0.0],
        })
        mae = MeanAbsoluteError(time_weighter=TableWeighter(frame=tw, on="time"))
        mae.fit(y_train)
        with pytest.raises(ValueError, match="All weights are zero"):
            mae.score(y_true, y_pred)


class TestZeroScaleWarning:
    """MASE and RMSSE warn when training data has zero scale."""

    def test_mase_warns_on_constant_training_data(self):
        dates = [datetime(2024, 1, i) for i in range(1, 21)]
        y_train = pl.DataFrame({"time": dates, "value": [5.0] * 20})

        scorer = MeanAbsoluteScaledError()
        with pytest.warns(UserWarning, match="zero scale"):
            scorer.fit(y_train)

        # Score should still work (clamped to 1e-10)
        y_true = pl.DataFrame({
            "time": [datetime(2024, 1, 21)],
            "value": [6.0],
        })
        y_pred = pl.DataFrame({
            "time": [datetime(2024, 1, 21)],
            "value": [7.0],
        })
        result = scorer.score(y_true, y_pred)
        assert np.isfinite(result)

    def test_rmsse_warns_on_constant_training_data(self):
        dates = [datetime(2024, 1, i) for i in range(1, 21)]
        y_train = pl.DataFrame({"time": dates, "value": [5.0] * 20})

        scorer = RootMeanSquaredScaledError()
        with pytest.warns(UserWarning, match="zero scale"):
            scorer.fit(y_train)

        result_scale = scorer.scales_["value"]
        assert result_scale == 1e-10

    def test_mase_no_warning_on_variable_data(self, y_train):
        scorer = MeanAbsoluteScaledError()
        scorer.fit(y_train)
        assert scorer.naive_errors_["value"] > 1e-10

    def test_rmsse_no_warning_on_variable_data(self, y_train):
        scorer = RootMeanSquaredScaledError()
        scorer.fit(y_train)
        assert scorer.scales_["value"] > 1e-10
