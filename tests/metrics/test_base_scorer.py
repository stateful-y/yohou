"""Tests for base scorer classes: abstract enforcement, time_weight, and panel weights."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from yohou.metrics import MeanAbsoluteError
from yohou.metrics.base import BaseIntervalScorer, BasePointScorer, BaseScorer


@pytest.fixture
def y_true_pred_pair():
    """Simple ground truth / prediction pair (5 timesteps, 1 column)."""
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]
    y_true = pl.DataFrame({"time": times, "value": [10.0, 20.0, 30.0, 40.0, 50.0]})
    y_pred = pl.DataFrame({
        "observed_time": [datetime(2019, 12, 31)] * 5,
        "time": times,
        "value": [12.0, 18.0, 33.0, 38.0, 55.0],
    })
    return y_true, y_pred


@pytest.fixture
def panel_y_true_pred_pair():
    """Panel data ground truth / prediction pair (5 timesteps, 2 groups)."""
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]
    y_true = pl.DataFrame({
        "time": times,
        "A__value": [10.0, 20.0, 30.0, 40.0, 50.0],
        "B__value": [100.0, 200.0, 300.0, 400.0, 500.0],
    })
    y_pred = pl.DataFrame({
        "observed_time": [datetime(2019, 12, 31)] * 5,
        "time": times,
        "A__value": [12.0, 18.0, 33.0, 38.0, 55.0],
        "B__value": [110.0, 190.0, 310.0, 390.0, 510.0],
    })
    return y_true, y_pred


class TestAbstractEnforcement:
    """Verify abstract base classes cannot be instantiated directly."""

    def test_base_scorer_cannot_be_instantiated(self):
        """BaseScorer is abstract and should raise TypeError on instantiation."""
        with pytest.raises(TypeError, match="abstract method"):
            BaseScorer()

    def test_base_point_scorer_cannot_be_instantiated(self):
        """BasePointScorer is abstract and should raise TypeError on instantiation."""
        with pytest.raises(TypeError, match="abstract method"):
            BasePointScorer()

    def test_base_interval_scorer_cannot_be_instantiated(self):
        """BaseIntervalScorer is abstract and should raise TypeError on instantiation."""
        with pytest.raises(TypeError, match="abstract method"):
            BaseIntervalScorer()


class TestTimeWeightCallable:
    """Tests for time_weight as a callable passed to score()."""

    def test_callable_changes_score(self, y_true_pred_pair):
        """Score with time_weight callable should differ from score without."""
        y_true, y_pred = y_true_pred_pair
        scorer = MeanAbsoluteError()
        scorer.fit(y_true)

        score_plain = scorer.score(y_true, y_pred)

        def recent_emphasis(time: pl.Series) -> pl.Series:
            n = len(time)
            return pl.Series("w", list(range(1, n + 1)), dtype=pl.Float64)

        scorer.set_score_request(time_weight=True)
        score_weighted = scorer.score(y_true, y_pred, time_weight=recent_emphasis)

        assert isinstance(score_plain, float)
        assert isinstance(score_weighted, float)
        assert score_plain != score_weighted

    def test_uniform_callable_preserves_score(self, y_true_pred_pair):
        """Uniform weights should give the same score as no time_weight."""
        y_true, y_pred = y_true_pred_pair
        scorer = MeanAbsoluteError()
        scorer.fit(y_true)

        score_plain = scorer.score(y_true, y_pred)

        def uniform(time: pl.Series) -> pl.Series:
            return pl.Series("w", [1.0] * len(time), dtype=pl.Float64)

        scorer.set_score_request(time_weight=True)
        score_uniform = scorer.score(y_true, y_pred, time_weight=uniform)

        assert score_plain == pytest.approx(score_uniform)

    def test_two_param_callable(self, y_true_pred_pair):
        """2-parameter callable receives group_name (None for global data)."""
        y_true, y_pred = y_true_pred_pair
        received_groups = []

        def weight_fn(time: pl.Series, group_name: str | None) -> pl.Series:
            received_groups.append(group_name)
            return pl.Series("w", [1.0] * len(time), dtype=pl.Float64)

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        scorer.set_score_request(time_weight=True)
        scorer.score(y_true, y_pred, time_weight=weight_fn)

        # For global data, group_name should be None
        assert received_groups == [None]

    def test_two_param_callable_panel(self, panel_y_true_pred_pair):
        """2-parameter callable receives group_name for each panel group."""
        y_true, y_pred = panel_y_true_pred_pair
        received_groups = []

        def weight_fn(time: pl.Series, group_name: str | None) -> pl.Series:
            received_groups.append(group_name)
            return pl.Series("w", [1.0] * len(time), dtype=pl.Float64)

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        scorer.set_score_request(time_weight=True)
        scorer.score(y_true, y_pred, time_weight=weight_fn)

        assert set(received_groups) == {"A", "B"}


class TestTimeWeightDataFrame:
    """Tests for time_weight as a pl.DataFrame passed to score()."""

    def test_dataframe_weight_changes_score(self, y_true_pred_pair):
        """Score with DataFrame time_weight should differ from plain score."""
        y_true, y_pred = y_true_pred_pair
        scorer = MeanAbsoluteError()
        scorer.fit(y_true)

        score_plain = scorer.score(y_true, y_pred)

        times = y_true["time"].to_list()
        tw_df = pl.DataFrame({
            "time": times,
            "weight": [0.1, 0.1, 0.1, 0.1, 10.0],
        })

        scorer.set_score_request(time_weight=True)
        score_weighted = scorer.score(y_true, y_pred, time_weight=tw_df)

        assert isinstance(score_weighted, float)
        assert score_plain != score_weighted

    def test_dataframe_uniform_preserves_score(self, y_true_pred_pair):
        """Uniform DataFrame weights should preserve the score."""
        y_true, y_pred = y_true_pred_pair
        scorer = MeanAbsoluteError()
        scorer.fit(y_true)

        score_plain = scorer.score(y_true, y_pred)

        times = y_true["time"].to_list()
        tw_df = pl.DataFrame({
            "time": times,
            "weight": [1.0] * 5,
        })

        scorer.set_score_request(time_weight=True)
        score_uniform = scorer.score(y_true, y_pred, time_weight=tw_df)

        assert score_plain == pytest.approx(score_uniform)


class TestTimeWeightValidation:
    """Tests for error paths in _process_time_weights."""

    def test_nan_weights_raises(self, y_true_pred_pair):
        """NaN values in weights should raise ValueError."""
        y_true, y_pred = y_true_pred_pair

        def bad_weights(time: pl.Series) -> pl.Series:
            return pl.Series("w", [1.0, float("nan"), 1.0, 1.0, 1.0])

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        scorer.set_score_request(time_weight=True)

        with pytest.raises(ValueError, match="NaN"):
            scorer.score(y_true, y_pred, time_weight=bad_weights)

    def test_negative_weights_raises(self, y_true_pred_pair):
        """Negative weight values should raise ValueError."""
        y_true, y_pred = y_true_pred_pair

        def bad_weights(time: pl.Series) -> pl.Series:
            return pl.Series("w", [1.0, -1.0, 1.0, 1.0, 1.0])

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        scorer.set_score_request(time_weight=True)

        with pytest.raises(ValueError, match="negative"):
            scorer.score(y_true, y_pred, time_weight=bad_weights)

    def test_infinite_weights_raises(self, y_true_pred_pair):
        """Infinite weight values should raise ValueError."""
        y_true, y_pred = y_true_pred_pair

        def bad_weights(time: pl.Series) -> pl.Series:
            return pl.Series("w", [1.0, float("inf"), 1.0, 1.0, 1.0])

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        scorer.set_score_request(time_weight=True)

        with pytest.raises(ValueError, match="infinite"):
            scorer.score(y_true, y_pred, time_weight=bad_weights)

    def test_zero_sum_weights_raises(self, y_true_pred_pair):
        """Weights that sum to zero should raise ValueError."""
        y_true, y_pred = y_true_pred_pair

        def bad_weights(time: pl.Series) -> pl.Series:
            return pl.Series("w", [0.0, 0.0, 0.0, 0.0, 0.0])

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        scorer.set_score_request(time_weight=True)

        with pytest.raises(ValueError, match="zero"):
            scorer.score(y_true, y_pred, time_weight=bad_weights)

    def test_wrong_length_callable_raises(self, y_true_pred_pair):
        """Callable returning wrong-length Series should raise ValueError."""
        y_true, y_pred = y_true_pred_pair

        def bad_weights(time: pl.Series) -> pl.Series:
            return pl.Series("w", [1.0, 1.0])  # Too short

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        scorer.set_score_request(time_weight=True)

        with pytest.raises(ValueError, match="weights"):
            scorer.score(y_true, y_pred, time_weight=bad_weights)

    def test_non_series_return_raises(self, y_true_pred_pair):
        """Callable not returning pl.Series should raise ValueError."""
        y_true, y_pred = y_true_pred_pair

        def bad_weights(time: pl.Series) -> list:
            return [1.0] * len(time)

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        scorer.set_score_request(time_weight=True)

        with pytest.raises(ValueError, match="pl.Series"):
            scorer.score(y_true, y_pred, time_weight=bad_weights)

    def test_invalid_type_raises(self, y_true_pred_pair):
        """Non-callable, non-DataFrame time_weight should raise ValueError."""
        y_true, y_pred = y_true_pred_pair

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        scorer.set_score_request(time_weight=True)

        with pytest.raises(ValueError, match="callable.*DataFrame.*None"):
            scorer.score(y_true, y_pred, time_weight="invalid")

    def test_three_param_callable_raises(self, y_true_pred_pair):
        """Callable with 3 parameters should raise ValueError."""
        y_true, y_pred = y_true_pred_pair

        def bad_weights(time: pl.Series, group: str, extra: int) -> pl.Series:
            return pl.Series("w", [1.0] * len(time))

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        scorer.set_score_request(time_weight=True)

        with pytest.raises(ValueError, match="1 parameter.*2 parameters.*got 3"):
            scorer.score(y_true, y_pred, time_weight=bad_weights)

    def test_dataframe_missing_weight_column_raises(self, y_true_pred_pair):
        """DataFrame missing 'weight' column should raise ValueError."""
        y_true, y_pred = y_true_pred_pair

        times = y_true["time"].to_list()
        tw_df = pl.DataFrame({"time": times, "wrong_col": [1.0] * 5})

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        scorer.set_score_request(time_weight=True)

        with pytest.raises(ValueError, match="weight"):
            scorer.score(y_true, y_pred, time_weight=tw_df)


class TestPanelGroupWeights:
    """Tests for panel_group_weight parameter on BaseScorer."""

    def test_panel_group_weight_affects_score(self, panel_y_true_pred_pair):
        """Different panel group weights should change the score."""
        y_true, y_pred = panel_y_true_pred_pair

        scorer_equal = MeanAbsoluteError()
        scorer_equal.fit(y_true)
        score_equal = scorer_equal.score(y_true, y_pred)

        scorer_weighted = MeanAbsoluteError(panel_group_weight={"A": 10.0, "B": 0.1})
        scorer_weighted.fit(y_true)
        score_weighted = scorer_weighted.score(y_true, y_pred)

        assert score_equal != score_weighted

    def test_equal_explicit_weights_match_default(self, panel_y_true_pred_pair):
        """Explicitly equal panel group weights should give the same result as default."""
        y_true, y_pred = panel_y_true_pred_pair

        scorer_default = MeanAbsoluteError()
        scorer_default.fit(y_true)
        score_default = scorer_default.score(y_true, y_pred)

        scorer_explicit = MeanAbsoluteError(panel_group_weight={"A": 1.0, "B": 1.0})
        scorer_explicit.fit(y_true)
        score_explicit = scorer_explicit.score(y_true, y_pred)

        assert score_default == pytest.approx(score_explicit)


class TestScorerTags:
    """Verify scorer tags are set correctly."""

    def test_point_scorer_prediction_type(self):
        """MeanAbsoluteError should have prediction_type='point'."""
        scorer = MeanAbsoluteError()
        tags = scorer.__sklearn_tags__()
        assert tags.scorer_tags is not None
        assert tags.scorer_tags.prediction_type == "point"

    def test_scorer_estimator_type(self):
        """All scorers should have estimator_type='scorer'."""
        scorer = MeanAbsoluteError()
        tags = scorer.__sklearn_tags__()
        assert tags.estimator_type == "scorer"

    def test_scorer_requires_calibration_false(self):
        """Simple scorers should not require calibration."""
        scorer = MeanAbsoluteError()
        tags = scorer.__sklearn_tags__()
        assert tags.scorer_tags is not None
        assert tags.scorer_tags.requires_calibration is False
