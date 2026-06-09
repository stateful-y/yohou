"""Tests for base scorer classes: abstract enforcement, time_weight, and panel weights."""

from datetime import datetime, timedelta

import polars as pl
import pytest
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from yohou.metrics import MeanAbsoluteError
from yohou.metrics.base import BaseIntervalScorer, BasePointScorer, BaseScorer
from yohou.utils.weighting import BaseWeighter, LookupWeighter, TableWeighter


class _CallableWeighter(BaseWeighter):
    """Test helper: wrap a weight callable as a BaseWeighter."""

    _parameter_constraints = {"fn": [callable]}

    def __init__(self, fn):
        self.fn = fn

    def compute_weights(self, key, group_name=None):
        import inspect

        n_params = len(inspect.signature(self.fn).parameters)
        result = self.fn(key) if n_params == 1 else self.fn(key, group_name)
        return result if isinstance(result, pl.Series) else pl.Series(result, dtype=pl.Float64)


def _weighter(w, on="time"):
    """Test helper: build a weighter from a callable, DataFrame, or dict."""
    if isinstance(w, pl.DataFrame):
        return TableWeighter(frame=w, on=on)
    if isinstance(w, dict):
        return LookupWeighter(mapping={k: v for k, v in w.items() if k != "*"}, default=w.get("*", 1.0))
    return _CallableWeighter(w)


class _DummyForecaster(BaseEstimator):
    """Minimal estimator stub to satisfy check_is_fitted."""

    def fit(self, X=None, y=None):
        return self


@pytest.fixture
def y_true_pred_pair():
    """Simple ground truth / prediction pair (5 timesteps, 1 column)."""
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]
    y_true = pl.DataFrame({"time": times, "value": [10.0, 20.0, 30.0, 40.0, 50.0]})
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2019, 12, 31)] * 5,
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
        "vintage_time": [datetime(2019, 12, 31)] * 5,
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

        score_weighted = scorer.set_params(time_weighter=_weighter(recent_emphasis, on="time")).score(y_true, y_pred)

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

        score_uniform = scorer.set_params(time_weighter=_weighter(uniform, on="time")).score(y_true, y_pred)

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
        scorer.set_params(time_weighter=_weighter(weight_fn, on="time")).score(y_true, y_pred)

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
        scorer.set_params(time_weighter=_weighter(weight_fn, on="time")).score(y_true, y_pred)

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

        score_weighted = scorer.set_params(time_weighter=_weighter(tw_df, on="time")).score(y_true, y_pred)

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

        score_uniform = scorer.set_params(time_weighter=_weighter(tw_df, on="time")).score(y_true, y_pred)

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

        with pytest.raises(ValueError, match="NaN"):
            scorer.set_params(time_weighter=_weighter(bad_weights, on="time")).score(y_true, y_pred)

    def test_negative_weights_raises(self, y_true_pred_pair):
        """Negative weight values should raise ValueError."""
        y_true, y_pred = y_true_pred_pair

        def bad_weights(time: pl.Series) -> pl.Series:
            return pl.Series("w", [1.0, -1.0, 1.0, 1.0, 1.0])

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)

        with pytest.raises(ValueError, match="negative"):
            scorer.set_params(time_weighter=_weighter(bad_weights, on="time")).score(y_true, y_pred)

    def test_infinite_weights_raises(self, y_true_pred_pair):
        """Infinite weight values should raise ValueError."""
        y_true, y_pred = y_true_pred_pair

        def bad_weights(time: pl.Series) -> pl.Series:
            return pl.Series("w", [1.0, float("inf"), 1.0, 1.0, 1.0])

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)

        with pytest.raises(ValueError, match="infinite"):
            scorer.set_params(time_weighter=_weighter(bad_weights, on="time")).score(y_true, y_pred)

    def test_zero_sum_weights_raises(self, y_true_pred_pair):
        """Weights that sum to zero should raise ValueError."""
        y_true, y_pred = y_true_pred_pair

        def bad_weights(time: pl.Series) -> pl.Series:
            return pl.Series("w", [0.0, 0.0, 0.0, 0.0, 0.0])

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)

        with pytest.raises(ValueError, match="zero"):
            scorer.set_params(time_weighter=_weighter(bad_weights, on="time")).score(y_true, y_pred)

    def test_wrong_length_callable_raises(self, y_true_pred_pair):
        """Callable returning wrong-length Series should raise ValueError."""
        y_true, y_pred = y_true_pred_pair

        def bad_weights(time: pl.Series) -> pl.Series:
            return pl.Series("w", [1.0, 1.0])  # Too short

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)

        with pytest.raises(ValueError, match="weights"):
            scorer.set_params(time_weighter=_weighter(bad_weights, on="time")).score(y_true, y_pred)

    def test_dataframe_missing_weight_column_raises(self, y_true_pred_pair):
        """DataFrame missing 'weight' column should raise ValueError."""
        y_true, y_pred = y_true_pred_pair

        times = y_true["time"].to_list()
        tw_df = pl.DataFrame({"time": times, "wrong_col": [1.0] * 5})

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)

        with pytest.raises(ValueError, match="weight"):
            scorer.set_params(time_weighter=_weighter(tw_df, on="time")).score(y_true, y_pred)


class TestPanelGroupWeights:
    """Tests for group_weight parameter on BaseScorer."""

    def test_group_weight_affects_score(self, panel_y_true_pred_pair):
        """Different panel group weights should change the score."""
        y_true, y_pred = panel_y_true_pred_pair

        scorer_equal = MeanAbsoluteError()
        scorer_equal.fit(y_true)
        score_equal = scorer_equal.score(y_true, y_pred)

        scorer_weighted = MeanAbsoluteError(groups={"A": 10.0, "B": 0.1})
        scorer_weighted.fit(y_true)
        score_weighted = scorer_weighted.score(y_true, y_pred)

        assert score_equal != score_weighted

    def test_equal_explicit_weights_match_default(self, panel_y_true_pred_pair):
        """Explicitly equal panel group weights should give the same result as default."""
        y_true, y_pred = panel_y_true_pred_pair

        scorer_default = MeanAbsoluteError()
        scorer_default.fit(y_true)
        score_default = scorer_default.score(y_true, y_pred)

        scorer_explicit = MeanAbsoluteError(groups={"A": 1.0, "B": 1.0})
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


class TestScorerCallableInterface:
    """Tests for scorer __call__ method."""

    def test_call_returns_same_as_score(self, y_true_pred_pair):
        """__call__ delegates to score and returns the same result."""
        y_true, y_pred = y_true_pred_pair
        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        score_via_score = scorer.score(y_true, y_pred)
        score_via_call = scorer(y_true, y_pred)
        assert score_via_score == pytest.approx(score_via_call)


class TestPanelGroupWeightsZero:
    """Tests for zero panel group weight edge case."""

    def test_zero_total_weight_raises(self):
        """Panel group weights summing to zero raise ValueError."""
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]
        y_true = pl.DataFrame({
            "time": times,
            "A__value": [10.0, 20.0, 30.0, 40.0, 50.0],
            "B__value": [100.0, 200.0, 300.0, 400.0, 500.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2019, 12, 31)] * 5,
            "time": times,
            "A__value": [12.0, 18.0, 33.0, 38.0, 55.0],
            "B__value": [110.0, 190.0, 310.0, 390.0, 510.0],
        })
        scorer = MeanAbsoluteError(
            groups={"A": 0.0, "B": 0.0},
        )
        scorer.fit(y_true)
        with pytest.raises(ValueError, match="Total panel group weight is zero"):
            scorer.score(y_true, y_pred)


class TestTimeWeightTwoParam:
    """Tests for 2-parameter time_weight callable (panel-aware)."""

    def test_two_param_callable_receives_group_name(self, panel_y_true_pred_pair):
        """2-parameter callable receives group name as second argument."""
        y_true, y_pred = panel_y_true_pred_pair
        received_groups = []

        def panel_weight(time: pl.Series, group: str) -> pl.Series:
            received_groups.append(group)
            return pl.Series("w", [1.0] * len(time), dtype=pl.Float64)

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        scorer.set_params(time_weighter=_weighter(panel_weight, on="time")).score(y_true, y_pred)
        assert len(received_groups) > 0


class TestProcessTimeWeightsEdgeCases:
    """Tests for _process_time_weights edge cases."""

    def test_callable_wrong_length_raises(self, y_true_pred_pair):
        """Callable returning wrong length Series raises ValueError."""
        y_true, y_pred = y_true_pred_pair

        def wrong_len_weight(time: pl.Series) -> pl.Series:
            return pl.Series("w", [1.0, 2.0], dtype=pl.Float64)

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)
        with pytest.raises(ValueError, match="weights.*rows"):
            scorer.set_params(time_weighter=_weighter(wrong_len_weight, on="time")).score(y_true, y_pred)


class TestValidateParametersEdgeCases:
    """Tests for _validate_parameters error branches."""

    def test_aggregation_no_valid_methods_raises(self):
        """Providing aggregation_method without valid_aggregation_methods raises."""
        scorer = MeanAbsoluteError()
        with pytest.raises(ValueError, match="valid_aggregation_methods must be provided"):
            scorer._validate_parameters(aggregation_method="stepwise", valid_aggregation_methods=None)

    def test_aggregation_invalid_string_raises(self):
        """Invalid string aggregation_method raises ValueError."""
        scorer = MeanAbsoluteError()
        with pytest.raises(ValueError, match="Invalid aggregation_method"):
            scorer._validate_parameters(aggregation_method="bad", valid_aggregation_methods={"stepwise"})

    def test_aggregation_empty_list_raises(self):
        """Empty list aggregation_method raises ValueError."""
        scorer = MeanAbsoluteError()
        with pytest.raises(ValueError, match="cannot be empty"):
            scorer._validate_parameters(aggregation_method=[], valid_aggregation_methods={"stepwise"})

    def test_aggregation_invalid_list_element_raises(self):
        """Invalid element in aggregation_method list raises ValueError."""
        scorer = MeanAbsoluteError()
        with pytest.raises(ValueError, match="Invalid aggregation_method"):
            scorer._validate_parameters(aggregation_method=["bad"], valid_aggregation_methods={"stepwise"})

    def test_aggregation_non_string_type_raises(self):
        """Non-string non-list aggregation_method raises ValueError."""
        scorer = MeanAbsoluteError()
        with pytest.raises(ValueError, match="must be a string or list"):
            scorer._validate_parameters(aggregation_method=42, valid_aggregation_methods={"stepwise"})

    def test_groups_non_list_raises(self):
        """Non-list/dict groups raises ValueError."""
        scorer = MeanAbsoluteError(groups="not_a_list")
        with pytest.raises(ValueError, match="groups.*must be an instance of 'list'"):
            scorer.fit(None)

    def test_component_names_non_list_raises(self):
        """Non-list/dict components raises ValueError."""
        scorer = MeanAbsoluteError(components="not_a_list")
        with pytest.raises(ValueError, match="components.*must be an instance of 'list'"):
            scorer.fit(None)


class TestAggregationMethodCombinations:
    """Tests for various aggregation method combinations in BasePointScorer."""

    @pytest.fixture
    def multivariate_data(self):
        """Multi-column truth and prediction DataFrames."""
        from datetime import datetime, timedelta

        n = 10
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        y_truth = pl.DataFrame({
            "time": times,
            "a": [float(i) for i in range(n)],
            "b": [float(i) * 2 for i in range(n)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [times[0]] * n,
            "time": times,
            "a": [float(i) + 0.5 for i in range(n)],
            "b": [float(i) * 2 + 0.5 for i in range(n)],
        })
        return y_truth, y_pred

    @pytest.fixture
    def panel_data(self):
        """Panel truth and prediction DataFrames with 2 groups."""
        from datetime import datetime, timedelta

        n = 10
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        y_truth = pl.DataFrame({
            "time": times,
            "g1__val": [float(i) for i in range(n)],
            "g2__val": [float(i) * 2 for i in range(n)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [times[0]] * n,
            "time": times,
            "g1__val": [float(i) + 0.5 for i in range(n)],
            "g2__val": [float(i) * 2 + 0.5 for i in range(n)],
        })
        return y_truth, y_pred

    def test_stepwise_vintagewise_only_returns_dataframe(self, multivariate_data):
        """aggregation_method=['stepwise', 'vintagewise'] returns DataFrame."""
        from yohou.metrics.point import MeanAbsoluteError

        y_truth, y_pred = multivariate_data
        scorer = MeanAbsoluteError(aggregation_method=["stepwise", "vintagewise"])
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        assert isinstance(result, pl.DataFrame)

    def test_componentwise_only_returns_dataframe(self, multivariate_data):
        """aggregation_method=['componentwise'] returns DataFrame."""
        from yohou.metrics.point import MeanAbsoluteError

        y_truth, y_pred = multivariate_data
        scorer = MeanAbsoluteError(aggregation_method=["componentwise"])
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        assert isinstance(result, pl.DataFrame)

    def test_groupwise_only_returns_dataframe(self, panel_data):
        """aggregation_method=['groupwise'] returns DataFrame for panel data."""
        from yohou.metrics.point import MeanAbsoluteError

        y_truth, y_pred = panel_data
        scorer = MeanAbsoluteError(aggregation_method=["groupwise"])
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        assert isinstance(result, pl.DataFrame)

    def test_stepwise_vintagewise_componentwise_returns_scalar(self, multivariate_data):
        """aggregation_method=['stepwise', 'vintagewise', 'componentwise'] returns scalar."""
        from yohou.metrics.point import MeanAbsoluteError

        y_truth, y_pred = multivariate_data
        scorer = MeanAbsoluteError(aggregation_method=["stepwise", "vintagewise", "componentwise"])
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        assert isinstance(result, float)

    def test_group_weight_in_collapse(self, panel_data):
        """group_weight dict is used by _collapse_panel_groups."""
        from yohou.metrics.point import MeanAbsoluteError

        y_truth, y_pred = panel_data
        scorer = MeanAbsoluteError(
            aggregation_method="all",
            groups={"g1": 2.0, "g2": 1.0},
        )
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        assert isinstance(result, float)


class TestGroupwiseAggregation:
    """Tests for groupwise-only aggregation on point scorer."""

    @pytest.fixture
    def panel_scorer_data(self):
        """Panel truth and prediction DataFrames with 2 groups."""
        from datetime import datetime, timedelta

        n = 10
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        y_truth = pl.DataFrame({
            "time": times,
            "g1__val": [float(i) for i in range(n)],
            "g2__val": [float(i) * 2 for i in range(n)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [times[0]] * n,
            "time": times,
            "g1__val": [float(i) + 0.5 for i in range(n)],
            "g2__val": [float(i) * 2 + 0.5 for i in range(n)],
        })
        return y_truth, y_pred

    def test_groupwise_aggregation_on_panel(self, panel_scorer_data):
        """aggregation_method='groupwise' on panel data aggregates across groups."""
        from yohou.metrics.point import MeanAbsoluteError

        y_truth, y_pred = panel_scorer_data
        scorer = MeanAbsoluteError(aggregation_method=["groupwise"])
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        assert isinstance(result, pl.DataFrame)

    def test_panel_stepwise_vintagewise_componentwise_groupwise(self, panel_scorer_data):
        """All three aggregation dimensions yield scalar on panel data."""
        from yohou.metrics.point import MeanAbsoluteError

        y_truth, y_pred = panel_scorer_data
        scorer = MeanAbsoluteError(
            aggregation_method=["stepwise", "vintagewise", "componentwise", "groupwise"],
        )
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        assert isinstance(result, float)


class TestPanelComponentwiseRename:
    """Tests for panel componentwise scoring with __score column rename."""

    @pytest.fixture
    def panel_scorer_data(self):
        """Panel truth and prediction DataFrames with 2 groups."""
        from datetime import datetime, timedelta

        n = 10
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        y_truth = pl.DataFrame({
            "time": times,
            "g1__val": [float(i) for i in range(n)],
            "g2__val": [float(i) * 2 for i in range(n)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [times[0]] * n,
            "time": times,
            "g1__val": [float(i) + 0.5 for i in range(n)],
            "g2__val": [float(i) * 2 + 0.5 for i in range(n)],
        })
        return y_truth, y_pred

    def test_mae_panel_componentwise_renames_score_columns(self, panel_scorer_data):
        """MAE componentwise on panel data renames group__score to group__mae."""
        from yohou.metrics.point import MeanAbsoluteError

        y_truth, y_pred = panel_scorer_data
        scorer = MeanAbsoluteError(aggregation_method=["componentwise"])
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        assert isinstance(result, pl.DataFrame)
        assert "time" in result.columns
        mae_cols = [c for c in result.columns if "mae" in c]
        assert len(mae_cols) > 0
        score_cols = [c for c in result.columns if c.endswith("__score")]
        assert len(score_cols) == 0

    def test_rmse_panel_componentwise_renames_score_columns(self, panel_scorer_data):
        """RMSE componentwise on panel data renames group__score to group__rmse."""
        from yohou.metrics.point import RootMeanSquaredError

        y_truth, y_pred = panel_scorer_data
        scorer = RootMeanSquaredError(aggregation_method=["componentwise"])
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        assert isinstance(result, pl.DataFrame)
        assert "time" in result.columns
        rmse_cols = [c for c in result.columns if "rmse" in c]
        assert len(rmse_cols) > 0
        score_cols = [c for c in result.columns if c.endswith("__score")]
        assert len(score_cols) == 0


class TestPanelGroupWeight:
    """Tests for _apply_panel_weights with custom group_weight."""

    def test_custom_group_weight(self):
        """Custom group_weight applies weighted average to group scores."""
        from datetime import datetime, timedelta

        from yohou.metrics.point import MeanAbsoluteError

        n = 10
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        y_truth = pl.DataFrame({
            "time": times,
            "g1__val": [float(i) for i in range(n)],
            "g2__val": [float(i) * 2 for i in range(n)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [times[0]] * n,
            "time": times,
            "g1__val": [float(i) + 1.0 for i in range(n)],
            "g2__val": [float(i) * 2 + 2.0 for i in range(n)],
        })
        scorer_equal = MeanAbsoluteError(aggregation_method="all")
        scorer_equal.fit(y_truth)
        score_equal = scorer_equal.score(y_truth, y_pred)

        scorer_weighted = MeanAbsoluteError(
            aggregation_method="all",
            groups={"g1": 10.0, "g2": 0.0001},
        )
        scorer_weighted.fit(y_truth)
        score_weighted = scorer_weighted.score(y_truth, y_pred)
        assert isinstance(score_equal, float)
        assert isinstance(score_weighted, float)
        assert score_weighted != pytest.approx(score_equal, abs=0.01)


class TestIntervalScorerAggregation:
    """Tests for interval scorer aggregation method branches."""

    @pytest.fixture
    def interval_data(self):
        """Interval truth and prediction DataFrames."""
        from datetime import datetime, timedelta

        n = 10
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        y_truth = pl.DataFrame({
            "time": times,
            "value": [10.0 + i for i in range(n)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [times[0]] * n,
            "time": times,
            "value_lower_0.9": [8.0 + i for i in range(n)],
            "value_upper_0.9": [12.0 + i for i in range(n)],
            "value_lower_0.5": [9.0 + i for i in range(n)],
            "value_upper_0.5": [11.0 + i for i in range(n)],
        })
        return y_truth, y_pred

    def test_interval_score_componentwise(self, interval_data):
        """IntervalScore with componentwise aggregation returns per-timestep DataFrame."""
        from yohou.metrics import IntervalScore

        y_truth, y_pred = interval_data
        scorer = IntervalScore(aggregation_method="componentwise")
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        assert isinstance(result, pl.DataFrame | dict)

    def test_interval_score_coveragewise(self, interval_data):
        """IntervalScore with coveragewise aggregation averages across rates."""
        from yohou.metrics import IntervalScore

        y_truth, y_pred = interval_data
        scorer = IntervalScore(aggregation_method=["stepwise", "vintagewise", "componentwise", "coveragewise"])
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        assert isinstance(result, float)

    def test_interval_score_all(self, interval_data):
        """IntervalScore with 'all' returns scalar float."""
        from yohou.metrics import IntervalScore

        y_truth, y_pred = interval_data
        scorer = IntervalScore(aggregation_method="all")
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        assert isinstance(result, float)
        assert result > 0

    def test_interval_score_invalid_coverage_rates_type(self, interval_data):
        """IntervalScore with non-list coverage_rates raises ValueError."""
        from yohou.metrics import IntervalScore

        y_truth, _ = interval_data
        scorer = IntervalScore(coverage_rates=0.9)
        with pytest.raises((ValueError, TypeError), match="coverage"):
            scorer.fit(y_truth)


class TestIntervalScorerGroupwise:
    """Tests for interval scorer groupwise aggregation on panel data."""

    @pytest.fixture
    def panel_interval_data(self):
        """Panel interval truth and prediction DataFrames."""
        from datetime import datetime, timedelta

        n = 10
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        y_truth = pl.DataFrame({
            "time": times,
            "sales__store_1": [10.0 + i for i in range(n)],
            "sales__store_2": [20.0 + i for i in range(n)],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [times[0]] * n,
            "time": times,
            "sales__store_1_lower_0.9": [8.0 + i for i in range(n)],
            "sales__store_1_upper_0.9": [12.0 + i for i in range(n)],
            "sales__store_2_lower_0.9": [18.0 + i for i in range(n)],
            "sales__store_2_upper_0.9": [22.0 + i for i in range(n)],
        })
        return y_truth, y_pred

    def test_groupwise_interval_scorer(self, panel_interval_data):
        """IntervalScore with groupwise on panel returns per-rate DataFrame."""
        from yohou.metrics import IntervalScore

        y_truth, y_pred = panel_interval_data
        scorer = IntervalScore(aggregation_method=["groupwise"])
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        assert isinstance(result, pl.DataFrame | dict)

    def test_groupwise_coveragewise_interval_scorer(self, panel_interval_data):
        """IntervalScore with groupwise+coveragewise averages across rates."""
        from yohou.metrics import IntervalScore

        y_truth, y_pred = panel_interval_data
        scorer = IntervalScore(aggregation_method=["groupwise", "coveragewise"])
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        assert isinstance(result, pl.DataFrame | float)

    def test_all_dimensions_interval_panel(self, panel_interval_data):
        """IntervalScore with 'all' on panel returns scalar float."""
        from yohou.metrics import IntervalScore

        y_truth, y_pred = panel_interval_data
        scorer = IntervalScore(aggregation_method="all")
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        assert isinstance(result, float)
        assert result > 0


class TestFitForecasterParam:
    """Tests for the optional forecaster parameter on scorer.fit()."""

    @pytest.fixture
    def daily_y_train(self):
        """Daily training series with 10 timesteps."""
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)]
        return pl.DataFrame({"time": times, "value": list(range(10))})

    @pytest.fixture
    def panel_y_train(self):
        """Panel training series with 2 groups and 10 timesteps."""
        times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)]
        return pl.DataFrame({
            "time": times,
            "A__value": list(range(10)),
            "B__value": list(range(10, 20)),
        })

    @staticmethod
    def _make_fitted_forecaster(interval="1d", horizon=5):
        """Create a minimal fitted forecaster-like estimator."""
        fc = _DummyForecaster()
        fc.interval_ = interval
        fc.fit_forecasting_horizon_ = horizon
        return fc

    def test_fit_without_forecaster_infers_interval(self, daily_y_train):
        """fit(y_train) without forecaster infers interval_ from data."""
        scorer = MeanAbsoluteError()
        scorer.fit(daily_y_train)
        assert scorer.interval_ == "1d"

    def test_fit_with_forecaster_uses_its_interval(self, daily_y_train):
        """fit(y_train, forecaster=fc) uses forecaster.interval_."""
        fc = self._make_fitted_forecaster(interval="2h")
        scorer = MeanAbsoluteError()
        scorer.fit(daily_y_train, forecaster=fc)
        assert scorer.interval_ == "2h"

    def test_fit_with_forecaster_stores_horizon(self, daily_y_train):
        """fit() stores forecaster_horizon_ from the forecaster."""
        fc = self._make_fitted_forecaster(horizon=7)
        scorer = MeanAbsoluteError()
        scorer.fit(daily_y_train, forecaster=fc)
        assert scorer.forecaster_horizon_ == 7

    def test_fit_without_forecaster_has_no_horizon(self, daily_y_train):
        """fit() without forecaster does not set forecaster_horizon_."""
        scorer = MeanAbsoluteError()
        scorer.fit(daily_y_train)
        assert not hasattr(scorer, "forecaster_horizon_")

    def test_fit_infers_groups_from_y_train(self, panel_y_train):
        """groups_ is always inferred from y_train, not from forecaster."""
        fc = self._make_fitted_forecaster()
        scorer = MeanAbsoluteError()
        scorer.fit(panel_y_train, forecaster=fc)
        assert set(scorer.groups_) == {"A", "B"}

    def test_fit_non_panel_has_no_groups(self, daily_y_train):
        """Non-panel data results in groups_=None."""
        scorer = MeanAbsoluteError()
        scorer.fit(daily_y_train)
        assert scorer.groups_ is None

    def test_unfitted_forecaster_raises(self, daily_y_train):
        """Passing an unfitted forecaster raises NotFittedError."""
        fc = _DummyForecaster()  # no fitted attributes (nothing ending with _)
        scorer = MeanAbsoluteError()
        with pytest.raises(NotFittedError):
            scorer.fit(daily_y_train, forecaster=fc)

    def test_backward_compat_positional_y_train(self, daily_y_train):
        """Existing code calling fit(y_train) still works."""
        scorer = MeanAbsoluteError()
        result = scorer.fit(daily_y_train)
        assert result is scorer
        assert scorer._is_fitted

    def test_interval_none_for_single_row(self):
        """Fewer than 2 rows results in interval_=None."""
        y = pl.DataFrame({"time": [datetime(2020, 1, 1)], "value": [1.0]})
        scorer = MeanAbsoluteError()
        scorer.fit(y)
        assert scorer.interval_ is None


class TestPreFilterZeroWeights:
    """Tests for _pre_filter_zero_weights error paths and panel-aware branches."""

    def test_all_zero_weights_raises(self):
        """Combined weights zeroing all rows raises ValueError."""
        base = datetime(2024, 1, 10)
        y_true = pl.DataFrame({
            "time": [
                base + timedelta(days=1),
                base + timedelta(days=2),
                base + timedelta(days=3),
            ],
            "value": [10.0, 20.0, 30.0],
        })
        # 2 vintages, 2 steps each:
        # Row 0: vintage=base, step=1 → step_weight=0
        # Row 1: vintage=base, step=2 → step_weight=1, vintage_weight=1
        # Row 2: vintage=base+1, step=1 → step_weight=0
        # Row 3: vintage=base+1, step=2 → step_weight=1, vintage_weight=0
        y_pred = pl.DataFrame({
            "vintage_time": [base, base, base + timedelta(days=1), base + timedelta(days=1)],
            "time": [
                base + timedelta(days=1),
                base + timedelta(days=2),
                base + timedelta(days=2),
                base + timedelta(days=3),
            ],
            "value": [11.0, 22.0, 19.0, 28.0],
        })
        y_train = pl.DataFrame({
            "time": [base - timedelta(days=i) for i in range(9, -1, -1)],
            "value": [float(i) for i in range(10)],
        })

        scorer = MeanAbsoluteError()
        scorer.fit(y_train)

        # step_weight zeros step 1 (rows 0, 2). vintage_weight zeros 2nd vintage (rows 2, 3).
        # zero_mask: [True, False, True, True]. Row 1 survives. NOT all zero.
        # To make ALL zero: also zero the remaining vintage=base row at step 2.
        # vintage_weight: {base: 0.0, base+1: 0.0, "*": 1.0} → all vintages zero → caught early.
        # Alternative approach: step_weight zeros step 1, time_weight zeros the specific times at step 2.
        # time_weight zeros base+2 and base+3 (step-2 times), step_weight zeros step 1.
        scorer2 = MeanAbsoluteError()
        scorer2.fit(y_train)

        with pytest.raises(ValueError, match="All rows have zero weight"):
            scorer2.set_params(
                step_weighter=_weighter({1: 0.0, 2: 1.0}, on="forecasting_step"),
                time_weighter=_weighter(
                    {
                        base + timedelta(days=1): 1.0,
                        base + timedelta(days=2): 0.0,
                        base + timedelta(days=3): 0.0,
                    },
                    on="time",
                ),
            ).score(y_true, y_pred)

    def test_panel_aware_callable_resolves_per_group(self, panel_y_true_pred_pair):
        """2-param callable produces a dict of per-group weight arrays."""
        y_true, y_pred = panel_y_true_pred_pair
        scorer = MeanAbsoluteError()
        scorer.fit(y_true)

        def group_weight(time: pl.Series, group_name: str | None) -> pl.Series:
            val = 2.0 if group_name == "A" else 1.0
            return pl.Series("w", [val] * len(time), dtype=pl.Float64)

        result = scorer.set_params(time_weighter=_weighter(group_weight, on="time")).score(y_true, y_pred)
        assert isinstance(result, float)

    def test_panel_aware_dataframe_with_group_weight_cols(self, panel_y_true_pred_pair):
        """DataFrame with group-specific weight columns is resolved per-group."""
        y_true, y_pred = panel_y_true_pred_pair
        times = y_true["time"].to_list()
        tw_df = pl.DataFrame({
            "time": times,
            "A_weight": [2.0, 2.0, 2.0, 2.0, 2.0],
            "B_weight": [1.0, 1.0, 1.0, 1.0, 1.0],
        })
        scorer = MeanAbsoluteError()
        scorer.fit(y_true)

        result = scorer.set_params(time_weighter=_weighter(tw_df, on="time")).score(y_true, y_pred)
        assert isinstance(result, float)


class TestApplyWeightsDict:
    """Tests for _apply_weights with dict (panel-aware) weight resolution."""

    @pytest.fixture
    def panel_vintage_data(self):
        """Panel data with vintage_time and forecasting_step for scorer tests."""
        base = datetime(2024, 1, 10)
        y_train = pl.DataFrame({
            "time": [base - timedelta(days=i) for i in range(9, -1, -1)],
            "A__value": [float(i) for i in range(10)],
            "B__value": [float(i) * 2 for i in range(10)],
        })
        y_true = pl.DataFrame({
            "time": [
                base + timedelta(days=1),
                base + timedelta(days=2),
                base + timedelta(days=3),
            ],
            "A__value": [10.0, 20.0, 30.0],
            "B__value": [100.0, 200.0, 300.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [base, base, base + timedelta(days=1)],
            "time": [
                base + timedelta(days=1),
                base + timedelta(days=2),
                base + timedelta(days=2),
            ],
            "A__value": [11.0, 22.0, 19.0],
            "B__value": [101.0, 202.0, 199.0],
        })
        return y_train, y_true, y_pred

    def test_panel_step_weight_dict_applied(self, panel_vintage_data):
        """Panel-aware step_weight dict combines step weights per group."""
        y_train, y_true, y_pred = panel_vintage_data

        def step_wt(step_series: pl.Series, group_name: str | None) -> pl.Series:
            return pl.Series("w", [1.0] * len(step_series), dtype=pl.Float64)

        scorer = MeanAbsoluteError()
        scorer.fit(y_train)
        result = scorer.set_params(step_weighter=_weighter(step_wt, on="forecasting_step")).score(y_true, y_pred)
        assert isinstance(result, float)

    def test_panel_vintage_weight_dict_applied(self, panel_vintage_data):
        """Panel-aware vintage_weight dict combines vintage weights per group."""
        y_train, y_true, y_pred = panel_vintage_data

        def vintage_wt(vintage_series: pl.Series, group_name: str | None) -> pl.Series:
            return pl.Series("w", [1.0] * len(vintage_series), dtype=pl.Float64)

        scorer = MeanAbsoluteError()
        scorer.fit(y_train)
        result = scorer.set_params(vintage_weighter=_weighter(vintage_wt, on="vintage_time")).score(y_true, y_pred)
        assert isinstance(result, float)

    def test_panel_step_and_vintage_weight_combined(self, panel_vintage_data):
        """Panel-aware step_weight + vintage_weight dicts are combined per group."""
        y_train, y_true, y_pred = panel_vintage_data

        def step_wt(step_series: pl.Series, group_name: str | None) -> pl.Series:
            return pl.Series("w", [2.0] * len(step_series), dtype=pl.Float64)

        def vintage_wt(vintage_series: pl.Series, group_name: str | None) -> pl.Series:
            return pl.Series("w", [3.0] * len(vintage_series), dtype=pl.Float64)

        scorer = MeanAbsoluteError()
        scorer.fit(y_train)
        result = scorer.set_params(
            step_weighter=_weighter(step_wt, on="forecasting_step"),
            vintage_weighter=_weighter(vintage_wt, on="vintage_time"),
        ).score(y_true, y_pred)
        assert isinstance(result, float)

    def test_panel_zero_time_weight_dict_filters(self, panel_y_true_pred_pair):
        """Panel-aware time_weight dict with zeros filters rows and slices dict arrays."""
        y_true, y_pred = panel_y_true_pred_pair

        # 2-param callable producing different weights per group, with zeros
        def tw_fn(time: pl.Series, group_name: str | None) -> pl.Series:
            n = len(time)
            weights = [1.0] * n
            weights[0] = 0.0  # zero first row
            return pl.Series("w", weights, dtype=pl.Float64)

        scorer = MeanAbsoluteError()
        scorer.fit(y_true)

        # This should work (the 2-param callable is panel-aware → returns dict
        # internally, so zero-mask filtering must slice the dict)
        result = scorer.set_params(time_weighter=_weighter(tw_fn, on="time")).score(y_true, y_pred)
        assert isinstance(result, float)

    def test_step_weight_zero_triggers_zero_mask(self, panel_vintage_data):
        """Step weight dict with zero value triggers zero_mask filtering."""
        y_train, y_true, y_pred = panel_vintage_data
        scorer = MeanAbsoluteError()
        scorer.fit(y_train)
        result = scorer.set_params(step_weighter=_weighter({1: 0.0, 2: 1.0}, on="forecasting_step")).score(
            y_true, y_pred
        )
        assert isinstance(result, float)

    def test_zero_step_weight_slices_tw_dict(self, panel_vintage_data):
        """Zero step_weight (array) triggers zero_mask; tw dict is sliced."""
        y_train, y_true, y_pred = panel_vintage_data

        def tw_fn(time: pl.Series, group_name: str | None) -> pl.Series:
            return pl.Series("w", [1.0] * len(time), dtype=pl.Float64)

        scorer = MeanAbsoluteError()
        scorer.fit(y_train)
        result = scorer.set_params(
            time_weighter=_weighter(tw_fn, on="time"), step_weighter=_weighter({1: 0.0, 2: 1.0}, on="forecasting_step")
        ).score(y_true, y_pred)
        assert isinstance(result, float)

    def test_zero_tw_slices_sw_and_vw_dicts(self, panel_vintage_data):
        """Zero time_weight (array) triggers zero_mask; sw and vw dicts are sliced."""
        y_train, y_true, y_pred = panel_vintage_data

        def tw_fn(time: pl.Series) -> pl.Series:
            w = [1.0] * len(time)
            w[0] = 0.0
            return pl.Series("w", w, dtype=pl.Float64)

        def sw_fn(step: pl.Series, group_name: str | None) -> pl.Series:
            return pl.Series("w", [1.0] * len(step), dtype=pl.Float64)

        def vw_fn(vintage: pl.Series, group_name: str | None) -> pl.Series:
            return pl.Series("w", [1.0] * len(vintage), dtype=pl.Float64)

        scorer = MeanAbsoluteError()
        scorer.fit(y_train)
        result = scorer.set_params(
            time_weighter=_weighter(tw_fn, on="time"),
            step_weighter=_weighter(sw_fn, on="forecasting_step"),
            vintage_weighter=_weighter(vw_fn, on="vintage_time"),
        ).score(y_true, y_pred)
        assert isinstance(result, float)
