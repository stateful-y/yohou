"""Tests for model_selection utility functions."""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from yohou.metrics.point import MeanAbsoluteError, MeanSquaredError
from yohou.model_selection.split import ExpandingWindowSplitter, check_cv
from yohou.model_selection.utils import (
    _check_scoring,
    _fit_and_score,
    _MultimetricScorer,
    _score,
)
from yohou.point.naive import SeasonalNaive


class TestCheckScoring:
    """Tests for _check_scoring validation."""

    def test_single_scorer(self):
        """Single BaseScorer instance passes through."""
        scorer = MeanAbsoluteError()
        result = _check_scoring(MagicMock(), scorer)
        assert result is scorer

    def test_dict_of_scorers(self):
        """Dict with string keys and BaseScorer values passes through."""
        scorers = {"mae": MeanAbsoluteError(), "mse": MeanSquaredError()}
        result = _check_scoring(MagicMock(), scorers)
        assert result == scorers

    def test_rejects_empty_dict(self):
        """Empty dict raises ValueError."""
        with pytest.raises(ValueError, match="empty dict"):
            _check_scoring(MagicMock(), {})

    def test_rejects_non_string_keys(self):
        """Dict with non-string keys raises ValueError."""
        with pytest.raises(ValueError, match="Non-string"):
            _check_scoring(MagicMock(), {1: MeanAbsoluteError()})

    def test_rejects_non_scorer_values(self):
        """Dict with non-scorer values raises ValueError."""
        with pytest.raises(ValueError, match="Non-scorer"):
            _check_scoring(MagicMock(), {"mae": "not_a_scorer"})

    def test_rejects_invalid_type(self):
        """Non-scorer, non-dict input raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scoring"):
            _check_scoring(MagicMock(), "mae")

    def test_rejects_list(self):
        """List input raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scoring"):
            _check_scoring(MagicMock(), [MeanAbsoluteError()])


class TestMultimetricScorer:
    """Tests for _MultimetricScorer."""

    def test_init(self):
        """Stores scorers and raise_exc flag."""
        scorers = {"mae": MeanAbsoluteError()}
        ms = _MultimetricScorer(scorers=scorers, raise_exc=False)
        assert ms._scorers is scorers
        assert ms._raise_exc is False

    def test_fit_calls_scorers_fit(self):
        """fit() calls fit on each scorer that has it."""
        scorer = MagicMock(spec=["fit", "score"])
        ms = _MultimetricScorer(scorers={"s": scorer})
        y = pl.DataFrame({"time": [1], "a": [1.0]})
        ms.fit(y)
        scorer.fit.assert_called_once_with(y, forecaster=None)

    def test_get_metadata_routing(self):
        """get_metadata_routing returns a MetadataRouter."""
        from sklearn.utils.metadata_routing import MetadataRouter

        scorers = {"mae": MeanAbsoluteError()}
        ms = _MultimetricScorer(scorers=scorers)
        router = ms.get_metadata_routing()
        assert isinstance(router, MetadataRouter)


class TestCheckCv:
    """Tests for check_cv input validation."""

    def test_none_returns_5_fold(self):
        """None defaults to 5-fold ExpandingWindowSplitter."""
        cv = check_cv(None)
        assert isinstance(cv, ExpandingWindowSplitter)
        assert cv.n_splits == 5

    def test_integer_creates_splitter(self):
        """Integer input creates ExpandingWindowSplitter with that many folds."""
        cv = check_cv(3, forecasting_horizon=5)
        assert isinstance(cv, ExpandingWindowSplitter)
        assert cv.n_splits == 3

    def test_splitter_passes_through(self):
        """BaseSplitter instance passes through unchanged."""
        splitter = ExpandingWindowSplitter(n_splits=4)
        result = check_cv(splitter)
        assert result is splitter

    def test_rejects_non_splitter_object(self):
        """Non-integer, non-splitter raises ValueError."""
        with pytest.raises(ValueError, match="Expected cv"):
            check_cv("invalid")

    def test_default_is_5(self):
        """No arguments defaults to 5 splits."""
        cv = check_cv()
        assert isinstance(cv, ExpandingWindowSplitter)
        assert cv.n_splits == 5


class TestGetResponseMethods:
    """Tests for _get_response_methods utility."""

    def test_single_point_scorer(self):
        """Single point scorer returns predict."""
        from yohou.model_selection.utils import _get_response_methods

        assert _get_response_methods(MeanAbsoluteError()) == {"predict"}

    def test_single_interval_scorer(self):
        """Single interval scorer returns predict_interval."""
        from yohou.metrics.interval import IntervalScore
        from yohou.model_selection.utils import _get_response_methods

        assert _get_response_methods(IntervalScore(coverage_rates=[0.9])) == {"predict_interval"}

    def test_multimetric_mixed(self):
        """Multimetric with mixed scorers returns both methods."""
        from yohou.metrics.interval import IntervalScore
        from yohou.model_selection.utils import _get_response_methods

        ms = _MultimetricScorer(scorers={"mae": MeanAbsoluteError(), "is": IntervalScore(coverage_rates=[0.9])})
        assert _get_response_methods(ms) == {"predict", "predict_interval"}

    def test_multimetric_only_interval_scorers(self):
        """Multimetric with only interval scorers returns predict_interval."""
        from yohou.metrics.interval import IntervalScore
        from yohou.model_selection.utils import _get_response_methods

        ms = _MultimetricScorer(scorers={"is": IntervalScore(coverage_rates=[0.9])})
        assert _get_response_methods(ms) == {"predict_interval"}


class TestResolveResponseMethod:
    """Tests for _resolve_response_method utility."""

    def test_single_point_scorer(self):
        """Single point scorer resolves to predict."""
        from yohou.model_selection.utils import _resolve_response_method

        assert _resolve_response_method(MeanAbsoluteError()) == "predict"

    def test_single_interval_scorer(self):
        """Single interval scorer resolves to predict_interval."""
        from yohou.metrics.interval import IntervalScore
        from yohou.model_selection.utils import _resolve_response_method

        assert _resolve_response_method(IntervalScore(coverage_rates=[0.9])) == "predict_interval"

    def test_multimetric_mixed_picks_richest(self):
        """Multimetric with mixed scorers resolves to the richest method."""
        from yohou.metrics.interval import IntervalScore
        from yohou.model_selection.utils import _resolve_response_method

        ms = _MultimetricScorer(scorers={"mae": MeanAbsoluteError(), "is": IntervalScore(coverage_rates=[0.9])})
        assert _resolve_response_method(ms) == "predict_interval"

    def test_multimetric_only_point_scorers(self):
        """Multimetric with only point scorers resolves to predict."""
        from yohou.model_selection.utils import _resolve_response_method

        ms = _MultimetricScorer(scorers={"mae": MeanAbsoluteError(), "mse": MeanSquaredError()})
        assert _resolve_response_method(ms) == "predict"


class TestValidateForecasterScorerCompatibility:
    """Tests for _validate_forecaster_scorer_compatibility."""

    def test_interval_scorer_with_point_forecaster_raises(self):
        """Interval scorer + point-only forecaster raises ValueError."""
        from yohou.metrics.interval import IntervalScore
        from yohou.model_selection.utils import _validate_forecaster_scorer_compatibility

        forecaster = SeasonalNaive()
        scorer = IntervalScore(coverage_rates=[0.9])
        with pytest.raises(ValueError, match="does not support predict_interval"):
            _validate_forecaster_scorer_compatibility(forecaster, scorer)

    def test_point_scorer_with_interval_only_forecaster_raises(self):
        """Point scorer + interval-only forecaster raises ValueError."""
        from yohou.model_selection.utils import _validate_forecaster_scorer_compatibility

        # Mock a pure interval-only forecaster
        forecaster = MagicMock()
        mock_tags = MagicMock()
        mock_tags.forecaster_tags.forecaster_type = frozenset({"interval"})
        forecaster.__sklearn_tags__ = MagicMock(return_value=mock_tags)

        scorer = MeanAbsoluteError()
        with pytest.raises(ValueError, match="does not support observe_predict"):
            _validate_forecaster_scorer_compatibility(forecaster, scorer)

    def test_point_scorer_with_both_type_forecaster_ok(self):
        """Point scorer + both-type forecaster passes."""
        from yohou.interval.split_conformal import SplitConformalForecaster
        from yohou.model_selection.utils import _validate_forecaster_scorer_compatibility

        forecaster = SplitConformalForecaster(point_forecaster=SeasonalNaive(), calibration_size=10)
        _validate_forecaster_scorer_compatibility(forecaster, MeanAbsoluteError())

    def test_interval_scorer_with_both_type_forecaster_ok(self):
        """Interval scorer + both-type forecaster passes."""
        from yohou.interval.split_conformal import SplitConformalForecaster
        from yohou.metrics.interval import IntervalScore
        from yohou.model_selection.utils import _validate_forecaster_scorer_compatibility

        forecaster = SplitConformalForecaster(point_forecaster=SeasonalNaive(), calibration_size=10)
        _validate_forecaster_scorer_compatibility(forecaster, IntervalScore(coverage_rates=[0.9]))

    def test_class_proba_scorer_with_point_forecaster_raises(self):
        """Class-proba scorer + point forecaster raises ValueError."""
        from yohou.metrics.class_proba import LogLoss
        from yohou.model_selection.utils import _validate_forecaster_scorer_compatibility

        forecaster = SeasonalNaive()
        scorer = LogLoss()
        with pytest.raises(ValueError, match="does not support predict_class_proba"):
            _validate_forecaster_scorer_compatibility(forecaster, scorer)

    def test_point_scorer_with_class_proba_forecaster_raises(self):
        """Point scorer + class-proba forecaster raises ValueError."""
        from yohou.model_selection.utils import _validate_forecaster_scorer_compatibility

        forecaster = MagicMock()
        mock_tags = MagicMock()
        mock_tags.forecaster_tags.forecaster_type = frozenset({"class_proba"})
        forecaster.__sklearn_tags__ = MagicMock(return_value=mock_tags)

        scorer = MeanAbsoluteError()
        with pytest.raises(ValueError, match="does not support observe_predict"):
            _validate_forecaster_scorer_compatibility(forecaster, scorer)


@pytest.fixture()
def fit_and_score_data():
    """Dataset with train/test indices for _fit_and_score tests."""
    length = 30
    fh = 5
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "target": [float(i) for i in range(length)]})
    train = np.arange(0, length - fh)
    test = np.arange(length - fh, length)
    return y, train, test, fh


@pytest.fixture()
def fitted_forecaster_data():
    """Fitted forecaster with train/test data for direct _score tests."""
    length = 30
    fh = 5
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "target": [float(i) for i in range(length)]})
    y_train = y.head(length - fh)
    y_test = y.tail(fh)

    forecaster = SeasonalNaive(seasonality=1)
    forecaster.fit(y_train, forecasting_horizon=fh)

    return forecaster, y_train, y_test


class _FailingForecaster(SeasonalNaive):
    """Forecaster whose fit() always raises."""

    def fit(self, y, X=None, forecasting_horizon=1, **kwargs):
        """Always raise RuntimeError."""
        raise RuntimeError("deliberate fit failure")


class _FittedFailingScorer(MeanAbsoluteError):
    """Scorer that passes the fitted check but always fails during scoring."""

    def score(self, y_truth, y_pred, /, **params):
        """Always raise RuntimeError after fitted check."""
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, ["_is_fitted"])
        raise RuntimeError("deliberate scoring failure")


class TestMultimetricScorerErrorSuppression:
    """Tests for _MultimetricScorer storing tracebacks when raise_exc=False."""

    def test_stores_traceback_when_raise_exc_false(self):
        """Exception traceback stored as string when raise_exc=False."""
        scorer = _FittedFailingScorer()
        y_truth = pl.DataFrame({
            "time": [datetime(2021, 1, 1), datetime(2021, 1, 2)],
            "target": [1.0, 2.0],
        })
        scorer.fit(y_truth)

        ms = _MultimetricScorer(scorers={"fail": scorer}, raise_exc=False)

        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2020, 12, 31)] * 2,
            "time": [datetime(2021, 1, 1), datetime(2021, 1, 2)],
            "target": [3.0, 4.0],
        })

        result = ms(y_truth, y_pred)
        assert isinstance(result["fail"], str)
        assert "deliberate scoring failure" in result["fail"]

    def test_raises_when_raise_exc_true(self):
        """Exception propagated when raise_exc=True (default)."""
        scorer = _FittedFailingScorer()
        y_truth = pl.DataFrame({
            "time": [datetime(2021, 1, 1), datetime(2021, 1, 2)],
            "target": [1.0, 2.0],
        })
        scorer.fit(y_truth)

        ms = _MultimetricScorer(scorers={"fail": scorer}, raise_exc=True)

        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2020, 12, 31)] * 2,
            "time": [datetime(2021, 1, 1), datetime(2021, 1, 2)],
            "target": [3.0, 4.0],
        })

        with pytest.raises(RuntimeError, match="deliberate scoring failure"):
            ms(y_truth, y_pred)


class TestFitAndScoreVerbose:
    """Tests for _fit_and_score verbose logging paths."""

    def test_verbose_3_with_split_progress(self, fit_and_score_data):
        """Verbose > 2 builds progress and parameter messages."""
        y, train, test, fh = fit_and_score_data
        result = _fit_and_score(
            SeasonalNaive(seasonality=1),
            y,
            None,
            fh,
            scorer=MeanAbsoluteError(),
            train=train,
            test=test,
            verbose=3,
            parameters={"seasonality": 1},
            fit_params=None,
            predict_func_params=None,
            score_params=None,
            split_progress=(0, 5),
            candidate_progress=(0, 3),
        )
        assert "test_scores" in result
        assert isinstance(result["test_scores"], float)

    def test_verbose_10_with_candidate_progress(self, fit_and_score_data):
        """Verbose > 9 includes candidate progress in message."""
        y, train, test, fh = fit_and_score_data
        result = _fit_and_score(
            SeasonalNaive(seasonality=1),
            y,
            None,
            fh,
            scorer=MeanAbsoluteError(),
            train=train,
            test=test,
            verbose=10,
            parameters={"seasonality": 1},
            fit_params=None,
            predict_func_params=None,
            score_params=None,
            split_progress=(2, 5),
            candidate_progress=(1, 3),
        )
        assert "test_scores" in result

    def test_verbose_3_multimetric_formats_dict_scores(self, fit_and_score_data):
        """Verbose > 2 with multimetric scorer formats dict test_scores."""
        y, train, test, fh = fit_and_score_data
        ms = _MultimetricScorer(
            scorers={"mae": MeanAbsoluteError(), "mse": MeanSquaredError()},
        )
        result = _fit_and_score(
            SeasonalNaive(seasonality=1),
            y,
            None,
            fh,
            scorer=ms,
            train=train,
            test=test,
            verbose=3,
            parameters={"seasonality": 1},
            fit_params=None,
            predict_func_params=None,
            score_params=None,
            split_progress=(0, 5),
        )
        assert isinstance(result["test_scores"], dict)
        assert "mae" in result["test_scores"]
        assert "mse" in result["test_scores"]

    def test_verbose_2_without_parameters(self, fit_and_score_data):
        """Verbose > 1 with parameters=None produces empty params_msg."""
        y, train, test, fh = fit_and_score_data
        result = _fit_and_score(
            SeasonalNaive(seasonality=1),
            y,
            None,
            fh,
            scorer=MeanAbsoluteError(),
            train=train,
            test=test,
            verbose=3,
            parameters=None,
            fit_params=None,
            predict_func_params=None,
            score_params=None,
        )
        assert "test_scores" in result


class TestFitAndScoreFitError:
    """Tests for _fit_and_score when fitting fails."""

    def test_error_score_numeric_single_scorer(self, fit_and_score_data):
        """Numeric error_score used as test_scores when fit() raises."""
        y, train, test, fh = fit_and_score_data
        result = _fit_and_score(
            _FailingForecaster(seasonality=1),
            y,
            None,
            fh,
            scorer=MeanAbsoluteError(),
            train=train,
            test=test,
            verbose=0,
            parameters=None,
            fit_params=None,
            predict_func_params=None,
            score_params=None,
            error_score=0.0,
        )
        assert result["test_scores"] == 0.0
        assert result["fit_error"] is not None
        assert "deliberate fit failure" in result["fit_error"]

    def test_error_score_multimetric(self, fit_and_score_data):
        """Multimetric scorer gets error_score for each metric on fit failure."""
        y, train, test, fh = fit_and_score_data
        ms = _MultimetricScorer(
            scorers={"mae": MeanAbsoluteError(), "mse": MeanSquaredError()},
        )
        result = _fit_and_score(
            _FailingForecaster(seasonality=1),
            y,
            None,
            fh,
            scorer=ms,
            train=train,
            test=test,
            verbose=0,
            parameters=None,
            fit_params=None,
            predict_func_params=None,
            score_params=None,
            error_score=0.0,
        )
        assert result["test_scores"] == {"mae": 0.0, "mse": 0.0}
        assert result["fit_error"] is not None

    def test_error_score_with_return_train_score_single(self, fit_and_score_data):
        """Train scores also set to error_score when fit fails (single scorer)."""
        y, train, test, fh = fit_and_score_data
        result = _fit_and_score(
            _FailingForecaster(seasonality=1),
            y,
            None,
            fh,
            scorer=MeanAbsoluteError(),
            train=train,
            test=test,
            verbose=0,
            parameters=None,
            fit_params=None,
            predict_func_params=None,
            score_params=None,
            error_score=0.0,
            return_train_score=True,
        )
        assert result["test_scores"] == 0.0
        assert result["train_scores"] == 0.0

    def test_error_score_with_return_train_score_multimetric(self, fit_and_score_data):
        """Train scores also set to error_score for multimetric on fit failure."""
        y, train, test, fh = fit_and_score_data
        ms = _MultimetricScorer(
            scorers={"mae": MeanAbsoluteError(), "mse": MeanSquaredError()},
        )
        result = _fit_and_score(
            _FailingForecaster(seasonality=1),
            y,
            None,
            fh,
            scorer=ms,
            train=train,
            test=test,
            verbose=0,
            parameters=None,
            fit_params=None,
            predict_func_params=None,
            score_params=None,
            error_score=0.0,
            return_train_score=True,
        )
        assert result["test_scores"] == {"mae": 0.0, "mse": 0.0}
        assert result["train_scores"] == {"mae": 0.0, "mse": 0.0}

    def test_error_score_raise_propagates(self, fit_and_score_data):
        """error_score='raise' propagates the fit exception."""
        y, train, test, fh = fit_and_score_data
        with pytest.raises(RuntimeError, match="deliberate fit failure"):
            _fit_and_score(
                _FailingForecaster(seasonality=1),
                y,
                None,
                fh,
                scorer=MeanAbsoluteError(),
                train=train,
                test=test,
                verbose=0,
                parameters=None,
                fit_params=None,
                predict_func_params=None,
                score_params=None,
                error_score="raise",
            )


class TestFitAndScoreReturnTrainScore:
    """Tests for _fit_and_score with return_train_score=True on successful fit."""

    def test_train_score_computed_after_rewind(self, fit_and_score_data):
        """Training scores computed by rewinding and scoring on train data."""
        y, train, test, fh = fit_and_score_data
        result = _fit_and_score(
            SeasonalNaive(seasonality=1),
            y,
            None,
            fh,
            scorer=MeanAbsoluteError(),
            train=train,
            test=test,
            verbose=0,
            parameters=None,
            fit_params=None,
            predict_func_params=None,
            score_params=None,
            return_train_score=True,
        )
        assert "train_scores" in result
        assert isinstance(result["train_scores"], float)
        assert result["train_scores"] <= 0.0  # negated score (lower_is_better defaults)
        assert "test_scores" in result


class TestFitAndScoreOptionalReturns:
    """Tests for _fit_and_score optional return flags."""

    def test_return_times(self, fit_and_score_data):
        """Fit and score times returned when return_times=True."""
        y, train, test, fh = fit_and_score_data
        result = _fit_and_score(
            SeasonalNaive(seasonality=1),
            y,
            None,
            fh,
            scorer=MeanAbsoluteError(),
            train=train,
            test=test,
            verbose=0,
            parameters=None,
            fit_params=None,
            predict_func_params=None,
            score_params=None,
            return_times=True,
        )
        assert "fit_time" in result
        assert "score_time" in result
        assert result["fit_time"] >= 0.0
        assert result["score_time"] >= 0.0

    def test_return_n_test_samples(self, fit_and_score_data):
        """Number of test samples returned when flag is set."""
        y, train, test, fh = fit_and_score_data
        result = _fit_and_score(
            SeasonalNaive(seasonality=1),
            y,
            None,
            fh,
            scorer=MeanAbsoluteError(),
            train=train,
            test=test,
            verbose=0,
            parameters=None,
            fit_params=None,
            predict_func_params=None,
            score_params=None,
            return_n_test_samples=True,
        )
        assert result["n_test_samples"] == len(test)

    def test_return_parameters(self, fit_and_score_data):
        """Evaluated parameters returned when flag is set."""
        y, train, test, fh = fit_and_score_data
        params = {"seasonality": 1}
        result = _fit_and_score(
            SeasonalNaive(seasonality=1),
            y,
            None,
            fh,
            scorer=MeanAbsoluteError(),
            train=train,
            test=test,
            verbose=0,
            parameters=params,
            fit_params=None,
            predict_func_params=None,
            score_params=None,
            return_parameters=True,
        )
        assert result["parameters"] == params

    def test_return_forecaster(self, fit_and_score_data):
        """Fitted forecaster returned when flag is set."""
        y, train, test, fh = fit_and_score_data
        result = _fit_and_score(
            SeasonalNaive(seasonality=1),
            y,
            None,
            fh,
            scorer=MeanAbsoluteError(),
            train=train,
            test=test,
            verbose=0,
            parameters=None,
            fit_params=None,
            predict_func_params=None,
            score_params=None,
            return_forecaster=True,
        )
        assert "forecaster" in result
        assert isinstance(result["forecaster"], SeasonalNaive)


class TestScoreSingleScorerError:
    """Tests for _score error handling with a single scorer."""

    def test_error_score_returned_on_scoring_failure(self, fitted_forecaster_data):
        """Numeric error_score returned and UserWarning raised when scorer fails."""
        forecaster, y_train, y_test = fitted_forecaster_data

        failing_scorer = MagicMock()
        failing_scorer.side_effect = ValueError("scoring failed")
        mock_tags = MagicMock()
        mock_tags.scorer_tags = None
        failing_scorer.__sklearn_tags__ = MagicMock(return_value=mock_tags)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _score(
                forecaster,
                y_train,
                y_test,
                None,
                None,
                failing_scorer,
                None,
                error_score=0.0,
            )

        assert result == 0.0
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "Scoring failed" in str(w[0].message)


class TestScoreDataFrameRejection:
    """Tests for _score rejecting DataFrame scores from non-aggregated scorers."""

    def test_raises_for_scalar_scorer_returning_dataframe(self, fitted_forecaster_data):
        """ValueError raised when a scalar scorer returns a DataFrame."""
        forecaster, y_train, y_test = fitted_forecaster_data

        df_scorer = MagicMock()
        df_scorer.return_value = pl.DataFrame({"score": [1.0, 2.0]})
        df_scorer.fit = MagicMock()
        df_scorer._response_method = "predict"

        with pytest.raises(ValueError, match="aggregation_method"):
            _score(
                forecaster,
                y_train,
                y_test,
                None,
                None,
                df_scorer,
                None,
                error_score="raise",
            )

    def test_raises_for_multimetric_scorer_returning_dataframe_value(self, fitted_forecaster_data):
        """ValueError raised when a multimetric scorer value is a DataFrame."""
        forecaster, y_train, y_test = fitted_forecaster_data

        ms = MagicMock(spec=_MultimetricScorer)
        ms._scorers = {"a": MeanAbsoluteError()}
        ms.return_value = {"a": pl.DataFrame({"score": [1.0]})}
        ms.fit = MagicMock()

        with pytest.raises(ValueError, match="aggregation_method"):
            _score(
                forecaster,
                y_train,
                y_test,
                None,
                None,
                ms,
                None,
                error_score="raise",
            )


class TestScoreNegation:
    """Tests for _score negating lower_is_better scores."""

    def test_negation_single_lower_is_better(self, fitted_forecaster_data):
        """Single scorer with lower_is_better=True returns negated score."""
        forecaster, y_train, y_test = fitted_forecaster_data

        mock_scorer = MagicMock()
        mock_scorer.return_value = 5.0
        mock_scorer.fit = MagicMock()
        mock_scorer._response_method = "predict"
        mock_tags = MagicMock()
        mock_tags.scorer_tags.lower_is_better = True
        mock_scorer.__sklearn_tags__ = MagicMock(return_value=mock_tags)

        result = _score(
            forecaster,
            y_train,
            y_test,
            None,
            None,
            mock_scorer,
            None,
            error_score="raise",
        )
        assert result == -5.0

    def test_no_negation_single_higher_is_better(self, fitted_forecaster_data):
        """Single scorer with lower_is_better=False returns positive score."""
        forecaster, y_train, y_test = fitted_forecaster_data

        mock_scorer = MagicMock()
        mock_scorer.return_value = 5.0
        mock_scorer.fit = MagicMock()
        mock_scorer._response_method = "predict"
        mock_tags = MagicMock()
        mock_tags.scorer_tags.lower_is_better = False
        mock_scorer.__sklearn_tags__ = MagicMock(return_value=mock_tags)

        result = _score(
            forecaster,
            y_train,
            y_test,
            None,
            None,
            mock_scorer,
            None,
            error_score="raise",
        )
        assert result == 5.0

    def test_negation_multimetric_lower_is_better(self, fitted_forecaster_data):
        """MultimetricScorer with lower_is_better=True negates dict scores."""
        forecaster, y_train, y_test = fitted_forecaster_data

        scorer = MeanAbsoluteError()
        original_tags_fn = scorer.__sklearn_tags__

        def _patched_tags():
            tags = original_tags_fn()
            tags.scorer_tags.lower_is_better = True
            return tags

        scorer.__sklearn_tags__ = _patched_tags

        ms = _MultimetricScorer(scorers={"mae": scorer})

        result = _score(
            forecaster,
            y_train,
            y_test,
            None,
            None,
            ms,
            {},
            error_score="raise",
        )
        assert isinstance(result, dict)
        assert result["mae"] < 0.0

    def test_score_string_passthrough(self, fitted_forecaster_data):
        """String error scores are returned without negation."""
        forecaster, y_train, y_test = fitted_forecaster_data

        mock_scorer = MagicMock()
        mock_scorer.return_value = "error string"
        mock_scorer.fit = MagicMock()
        mock_scorer._response_method = "predict"

        result = _score(
            forecaster,
            y_train,
            y_test,
            None,
            None,
            mock_scorer,
            None,
            error_score="raise",
        )
        assert result == "error string"


class TestScoreMultimetricStringError:
    """Tests for _score handling string errors in multimetric dict."""

    def test_error_string_replaced_with_error_score(self, fitted_forecaster_data):
        """String error values in dict are replaced with numeric error_score."""
        forecaster, y_train, y_test = fitted_forecaster_data

        ms = MagicMock(spec=_MultimetricScorer)
        ms._scorers = {"a": MeanAbsoluteError()}
        ms.return_value = {"a": "Traceback (most recent call last):\n..."}
        ms.fit = MagicMock()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _score(
                forecaster,
                y_train,
                y_test,
                None,
                None,
                ms,
                None,
                error_score=0.0,
            )

        assert result["a"] == 0.0
        assert len(w) == 1
        assert "Scoring failed" in str(w[0].message)
