"""Tests for model_selection utility functions."""

from unittest.mock import MagicMock

import polars as pl
import pytest

from yohou.metrics.point import MeanAbsoluteError, MeanSquaredError
from yohou.model_selection.split import ExpandingWindowSplitter, check_cv
from yohou.model_selection.utils import _check_scoring, _MultimetricScorer


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
        scorer.fit.assert_called_once_with(y)

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


class TestNeedsPointPredictions:
    """Tests for _needs_point_predictions utility."""

    def test_single_point_scorer(self):
        """Single point scorer returns True."""
        from yohou.model_selection.utils import _needs_point_predictions

        assert _needs_point_predictions(MeanAbsoluteError()) is True

    def test_single_interval_scorer(self):
        """Single interval scorer returns False."""
        from yohou.metrics.interval import IntervalScore
        from yohou.model_selection.utils import _needs_point_predictions

        assert _needs_point_predictions(IntervalScore(coverage_rates=[0.9])) is False

    def test_multimetric_with_point_scorer(self):
        """Multimetric containing a point scorer returns True."""
        from yohou.metrics.interval import IntervalScore
        from yohou.model_selection.utils import _needs_point_predictions

        ms = _MultimetricScorer(scorers={"mae": MeanAbsoluteError(), "is": IntervalScore(coverage_rates=[0.9])})
        assert _needs_point_predictions(ms) is True

    def test_multimetric_only_interval_scorers(self):
        """Multimetric with only interval scorers returns False."""
        from yohou.metrics.interval import IntervalScore
        from yohou.model_selection.utils import _needs_point_predictions

        ms = _MultimetricScorer(scorers={"is": IntervalScore(coverage_rates=[0.9])})
        assert _needs_point_predictions(ms) is False


class TestNeedsIntervalPredictions:
    """Tests for _needs_interval_predictions utility."""

    def test_single_point_scorer(self):
        """Single point scorer returns False."""
        from yohou.model_selection.utils import _needs_interval_predictions

        assert _needs_interval_predictions(MeanAbsoluteError()) is False

    def test_single_interval_scorer(self):
        """Single interval scorer returns True."""
        from yohou.metrics.interval import IntervalScore
        from yohou.model_selection.utils import _needs_interval_predictions

        assert _needs_interval_predictions(IntervalScore(coverage_rates=[0.9])) is True

    def test_multimetric_with_interval_scorer(self):
        """Multimetric containing an interval scorer returns True."""
        from yohou.metrics.interval import IntervalScore
        from yohou.model_selection.utils import _needs_interval_predictions

        ms = _MultimetricScorer(scorers={"mae": MeanAbsoluteError(), "is": IntervalScore(coverage_rates=[0.9])})
        assert _needs_interval_predictions(ms) is True

    def test_multimetric_only_point_scorers(self):
        """Multimetric with only point scorers returns False."""
        from yohou.model_selection.utils import _needs_interval_predictions

        ms = _MultimetricScorer(scorers={"mae": MeanAbsoluteError(), "mse": MeanSquaredError()})
        assert _needs_interval_predictions(ms) is False


class TestValidateForecasterScorerCompatibility:
    """Tests for _validate_forecaster_scorer_compatibility."""

    def test_interval_scorer_with_point_forecaster_raises(self):
        """Interval scorer + point-only forecaster raises ValueError."""
        from yohou.metrics.interval import IntervalScore
        from yohou.model_selection.utils import _validate_forecaster_scorer_compatibility
        from yohou.point.naive import SeasonalNaive

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
        mock_tags.forecaster_tags.forecaster_type = "interval"
        forecaster.__sklearn_tags__ = MagicMock(return_value=mock_tags)

        scorer = MeanAbsoluteError()
        with pytest.raises(ValueError, match="does not support observe_predict"):
            _validate_forecaster_scorer_compatibility(forecaster, scorer)

    def test_point_scorer_with_both_type_forecaster_ok(self):
        """Point scorer + both-type forecaster passes."""
        from yohou.interval.split_conformal import SplitConformalForecaster
        from yohou.model_selection.utils import _validate_forecaster_scorer_compatibility
        from yohou.point.naive import SeasonalNaive

        forecaster = SplitConformalForecaster(point_forecaster=SeasonalNaive(), calibration_size=10)
        _validate_forecaster_scorer_compatibility(forecaster, MeanAbsoluteError())

    def test_interval_scorer_with_both_type_forecaster_ok(self):
        """Interval scorer + both-type forecaster passes."""
        from yohou.interval.split_conformal import SplitConformalForecaster
        from yohou.metrics.interval import IntervalScore
        from yohou.model_selection.utils import _validate_forecaster_scorer_compatibility
        from yohou.point.naive import SeasonalNaive

        forecaster = SplitConformalForecaster(point_forecaster=SeasonalNaive(), calibration_size=10)
        _validate_forecaster_scorer_compatibility(forecaster, IntervalScore(coverage_rates=[0.9]))
