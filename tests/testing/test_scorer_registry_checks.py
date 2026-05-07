"""Systematic tests: run _yield_yohou_scorer_checks on every _SCORER_REGISTRY scorer."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from conftest import run_checks
from yohou.metrics import _SCORER_REGISTRY, get_scorer
from yohou.metrics.class_proba import BrierScore, LogLoss, RankedProbabilityScore
from yohou.metrics.classification import (
    Accuracy,
    FBetaScore,
    PRAuC,
    Precision,
    Recall,
    ROCAuC,
)
from yohou.metrics.interval import (
    CalibrationError,
    ContinuousRankedProbabilityScore,
    EmpiricalCoverage,
    IntervalScore,
    MeanIntervalWidth,
    PinballLoss,
)
from yohou.metrics.point import (
    MaxAbsoluteError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
    MeanDirectionalAccuracy,
    MeanSquaredError,
    MedianAbsoluteError,
    R2Score,
    RootMeanSquaredError,
    RootMeanSquaredScaledError,
    SymmetricMeanAbsolutePercentageError,
)
from yohou.testing import _yield_yohou_scorer_checks

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def point_data():
    """10-row point data. Values start at 1.0 to avoid MAPE zero-denominator."""
    n = 10
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    y_true = pl.DataFrame({"time": times, "value": [float(i + 1) for i in range(n)]})
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * n,
        "time": times,
        "value": [float(i + 1) + 0.5 for i in range(n)],
    })
    return y_true, y_pred


@pytest.fixture(scope="module")
def point_data_with_training():
    """100-row point data split 80/20 for RMSSE, MASE (need y_train)."""
    n = 100
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    values = [float(i % 7 + 1) for i in range(n)]
    y_full = pl.DataFrame({"time": times, "value": values})
    y_train = y_full[:80]
    y_test = y_full[80:]
    y_pred = pl.DataFrame({
        "vintage_time": [times[79]] * 20,
        "time": [times[i] for i in range(80, n)],
        "value": [v + 0.3 for v in values[80:]],
    })
    return y_train, y_test, y_pred


@pytest.fixture(scope="module")
def multivariate_point_data():
    """10-row, 2-component (temp, wind) point data."""
    n = 10
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    y_true = pl.DataFrame({
        "time": times,
        "temp": [float(i + 1) for i in range(n)],
        "wind": [float(i + 1) * 2 for i in range(n)],
    })
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * n,
        "time": times,
        "temp": [float(i + 1) + 0.5 for i in range(n)],
        "wind": [float(i + 1) * 2 + 0.5 for i in range(n)],
    })
    return y_true, y_pred


@pytest.fixture(scope="module")
def panel_point_data():
    """10-row, 2-group (A__value, B__value) point data."""
    n = 10
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    y_true = pl.DataFrame({
        "time": times,
        "A__value": [float(i + 1) for i in range(n)],
        "B__value": [float(i + 1) * 10 for i in range(n)],
    })
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * n,
        "time": times,
        "A__value": [float(i + 1) + 0.5 for i in range(n)],
        "B__value": [float(i + 1) * 10 + 0.5 for i in range(n)],
    })
    return y_true, y_pred


@pytest.fixture(scope="module")
def interval_single_rate_data():
    """10-row interval data with 0.9 rate. Includes coverage_rate column."""
    n = 10
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    y_true = pl.DataFrame({"time": times, "value": [float(i + 1) for i in range(n)]})
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * n,
        "time": times,
        "value_lower_0.9": [float(i + 1) - 2 for i in range(n)],
        "value_upper_0.9": [float(i + 1) + 2 for i in range(n)],
        "coverage_rate": [0.9] * n,
    })
    return y_true, y_pred


@pytest.fixture(scope="module")
def interval_multi_rate_data():
    """10-row interval data with 0.9 + 0.5 rates. Includes coverage_rate column."""
    n = 10
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    y_true = pl.DataFrame({"time": times, "value": [float(i + 1) for i in range(n)]})
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * n,
        "time": times,
        "value_lower_0.9": [float(i + 1) - 2 for i in range(n)],
        "value_upper_0.9": [float(i + 1) + 2 for i in range(n)],
        "value_lower_0.5": [float(i + 1) - 1 for i in range(n)],
        "value_upper_0.5": [float(i + 1) + 1 for i in range(n)],
        "coverage_rate": [0.9] * n,
    })
    return y_true, y_pred


@pytest.fixture(scope="module")
def class_proba_data():
    """10-row, 3-class probability data."""
    n = 10
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    labels = ["sunny", "rainy", "cloudy"] * 3 + ["sunny"]
    y_true = pl.DataFrame({"time": times, "weather": labels[:n]})
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * n,
        "time": times,
        "weather_proba_sunny": [0.7, 0.1, 0.2, 0.6, 0.1, 0.2, 0.8, 0.1, 0.1, 0.7],
        "weather_proba_rainy": [0.2, 0.8, 0.1, 0.3, 0.8, 0.1, 0.1, 0.8, 0.2, 0.2],
        "weather_proba_cloudy": [0.1, 0.1, 0.7, 0.1, 0.1, 0.7, 0.1, 0.1, 0.7, 0.1],
    })
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Registry instantiation
# ---------------------------------------------------------------------------


class TestRegistryInstantiation:
    """Verify get_scorer() instantiates every registry entry without error."""

    @pytest.mark.parametrize("name", sorted(_SCORER_REGISTRY.keys()))
    def test_get_scorer(self, name):
        scorer = get_scorer(name)
        assert scorer is not None
        assert hasattr(scorer, "__sklearn_tags__")


# ---------------------------------------------------------------------------
# Point scorers
# ---------------------------------------------------------------------------


class TestPointScorerChecks:
    """Run _yield_yohou_scorer_checks on all point scorers."""

    def _run(self, scorer_class, y_true, y_pred, **kwargs):
        scorer = scorer_class(**kwargs)
        scorer.fit(y_true)
        run_checks(scorer, _yield_yohou_scorer_checks(scorer, y_true, y_pred))

    def test_mae(self, point_data):
        self._run(MeanAbsoluteError, *point_data)

    def test_mse(self, point_data):
        self._run(MeanSquaredError, *point_data)

    def test_rmse(self, point_data):
        self._run(RootMeanSquaredError, *point_data)

    def test_rmsse(self, point_data_with_training):
        y_train, y_test, y_pred = point_data_with_training
        scorer = RootMeanSquaredScaledError(seasonality=7)
        scorer.fit(y_train)
        run_checks(scorer, _yield_yohou_scorer_checks(scorer, y_test, y_pred))

    def test_mape(self, point_data):
        self._run(MeanAbsolutePercentageError, *point_data)

    def test_smape(self, point_data):
        self._run(SymmetricMeanAbsolutePercentageError, *point_data)

    def test_mase(self, point_data_with_training):
        y_train, y_test, y_pred = point_data_with_training
        scorer = MeanAbsoluteScaledError(seasonality=7)
        scorer.fit(y_train)
        run_checks(scorer, _yield_yohou_scorer_checks(scorer, y_test, y_pred))

    def test_median_ae(self, point_data):
        self._run(MedianAbsoluteError, *point_data)

    def test_max_ae(self, point_data):
        self._run(MaxAbsoluteError, *point_data)

    def test_r2(self, point_data):
        self._run(R2Score, *point_data)

    def test_mda(self, point_data):
        self._run(MeanDirectionalAccuracy, *point_data)

    def test_mae_multivariate(self, multivariate_point_data):
        self._run(MeanAbsoluteError, *multivariate_point_data)

    def test_mae_panel(self, panel_point_data):
        self._run(MeanAbsoluteError, *panel_point_data)


# ---------------------------------------------------------------------------
# Interval scorers
# ---------------------------------------------------------------------------


class TestIntervalScorerChecks:
    """Run _yield_yohou_scorer_checks on all interval scorers."""

    def _run(self, scorer_class, y_true, y_pred, **kwargs):
        scorer = scorer_class(**kwargs)
        scorer.fit(y_true)
        run_checks(scorer, _yield_yohou_scorer_checks(scorer, y_true, y_pred))

    def test_empirical_coverage(self, interval_single_rate_data):
        self._run(EmpiricalCoverage, *interval_single_rate_data)

    def test_mean_interval_width(self, interval_single_rate_data):
        self._run(MeanIntervalWidth, *interval_single_rate_data)

    def test_interval_score(self, interval_single_rate_data):
        self._run(IntervalScore, *interval_single_rate_data)

    def test_pinball_loss(self, interval_single_rate_data):
        self._run(PinballLoss, *interval_single_rate_data)

    def test_calibration_error(self, interval_multi_rate_data):
        self._run(CalibrationError, *interval_multi_rate_data)

    def test_crps(self, interval_multi_rate_data):
        self._run(ContinuousRankedProbabilityScore, *interval_multi_rate_data)

    def test_crps_trapezoidal(self, interval_multi_rate_data):
        self._run(
            ContinuousRankedProbabilityScore,
            *interval_multi_rate_data,
            integration="trapezoidal",
        )


# ---------------------------------------------------------------------------
# Class-probability scorers
# ---------------------------------------------------------------------------


class TestClassProbaScorerChecks:
    """Run _yield_yohou_scorer_checks on class-probability scorers."""

    def _run(self, scorer_class, y_true, y_pred, **kwargs):
        scorer = scorer_class(**kwargs)
        scorer.fit(y_true)
        run_checks(scorer, _yield_yohou_scorer_checks(scorer, y_true, y_pred))

    def test_log_loss(self, class_proba_data):
        self._run(LogLoss, *class_proba_data)

    def test_brier_score(self, class_proba_data):
        self._run(BrierScore, *class_proba_data)

    def test_ranked_probability_score(self, class_proba_data):
        self._run(RankedProbabilityScore, *class_proba_data)


# ---------------------------------------------------------------------------
# Hard-label classification scorers
# ---------------------------------------------------------------------------


class TestHardLabelScorerChecks:
    """Run _yield_yohou_scorer_checks on hard-label classification scorers."""

    def _run(self, scorer_class, y_true, y_pred, **kwargs):
        scorer = scorer_class(**kwargs)
        scorer.fit(y_true)
        run_checks(scorer, _yield_yohou_scorer_checks(scorer, y_true, y_pred))

    def test_precision(self, class_proba_data):
        self._run(Precision, *class_proba_data)

    def test_recall(self, class_proba_data):
        self._run(Recall, *class_proba_data)

    def test_fbeta_1(self, class_proba_data):
        self._run(FBetaScore, *class_proba_data, beta=1.0)

    def test_accuracy(self, class_proba_data):
        self._run(Accuracy, *class_proba_data)

    def test_precision_micro(self, class_proba_data):
        self._run(Precision, *class_proba_data, average="micro")

    def test_fbeta_2(self, class_proba_data):
        self._run(FBetaScore, *class_proba_data, beta=2.0)


# ---------------------------------------------------------------------------
# Ranking scorers
# ---------------------------------------------------------------------------


class TestRankingScorerChecks:
    """Run _yield_yohou_scorer_checks on ranking classification scorers."""

    def _run(self, scorer_class, y_true, y_pred, **kwargs):
        scorer = scorer_class(**kwargs)
        scorer.fit(y_true)
        run_checks(scorer, _yield_yohou_scorer_checks(scorer, y_true, y_pred))

    def test_roc_auc(self, class_proba_data):
        self._run(ROCAuC, *class_proba_data)

    def test_pr_auc(self, class_proba_data):
        self._run(PRAuC, *class_proba_data)


# ---------------------------------------------------------------------------
# Registry completeness guard
# ---------------------------------------------------------------------------

# Collect all scorer classes tested above (unique set)
_TESTED_SCORER_CLASSES: set[type] = {
    MeanAbsoluteError,
    MeanSquaredError,
    RootMeanSquaredError,
    RootMeanSquaredScaledError,
    MeanAbsolutePercentageError,
    SymmetricMeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
    MedianAbsoluteError,
    MaxAbsoluteError,
    R2Score,
    MeanDirectionalAccuracy,
    EmpiricalCoverage,
    MeanIntervalWidth,
    IntervalScore,
    PinballLoss,
    CalibrationError,
    ContinuousRankedProbabilityScore,
    LogLoss,
    BrierScore,
    RankedProbabilityScore,
    Precision,
    Recall,
    FBetaScore,
    Accuracy,
    ROCAuC,
    PRAuC,
}


class TestRegistryCompleteness:
    """Guard: fails CI if a scorer is added to _SCORER_REGISTRY without a test here."""

    def test_all_registry_classes_tested(self):
        registry_classes = set(_SCORER_REGISTRY.values())
        untested = registry_classes - _TESTED_SCORER_CLASSES
        assert not untested, (
            f"Scorer classes in _SCORER_REGISTRY without systematic tests: {sorted(c.__name__ for c in untested)}"
        )
