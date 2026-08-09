"""Tests for the concrete step-kind transformers."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer as SklearnFunctionTransformer
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from yohou.compose import FeaturePipeline
from yohou.preprocessing import StepAggregator, StepColumnReducer, StepFrameReducer
from yohou.testing.step_transformer import STEP_TRANSFORMER_CHECKS


def _frame(with_nulls: bool = False) -> pl.DataFrame:
    """A step frame with two base columns over a horizon of three."""
    times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(6)]
    temp_3 = [9.0, 1.0, 4.0, 2.0, 8.0, 3.0]
    if with_nulls:
        temp_3 = [None, None, 4.0, 2.0, 8.0, 3.0]
    return pl.DataFrame({
        "time": times,
        "temp_step_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "temp_step_2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "temp_step_3": temp_3,
        "rain_step_1": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "rain_step_2": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "rain_step_3": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    })


class _AlwaysFails(TransformerMixin, BaseEstimator):
    """An inner estimator that fails regardless of coverage."""

    def fit(self, X, y=None):
        raise RuntimeError("inner estimator exploded")

    def transform(self, X):  # pragma: no cover - never reached
        return X


TRANSFORMERS = [
    StepAggregator(aggregations=("mean", "min")),
    StepColumnReducer(reducer=SklearnStandardScaler()),
    StepFrameReducer(reducer=SklearnStandardScaler(), prefix="wx"),
]


@pytest.mark.parametrize("check", STEP_TRANSFORMER_CHECKS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("transformer", TRANSFORMERS, ids=lambda t: type(t).__name__)
def test_systematic_step_transformer_checks(check, transformer):
    """Every concrete step transformer satisfies the shared contract."""
    check(clone(transformer), _frame())


class TestStepAggregator:
    """Closed-vocabulary reduction along the horizon."""

    def test_emits_one_column_per_aggregation(self):
        """Each base block collapses to one column per configured summary."""
        out = StepAggregator(aggregations=("min", "max", "mean", "std", "sum")).fit_transform(_frame())
        assert out.columns == [
            "time",
            "temp_step_min",
            "temp_step_max",
            "temp_step_mean",
            "temp_step_std",
            "temp_step_sum",
            "rain_step_min",
            "rain_step_max",
            "rain_step_mean",
            "rain_step_std",
            "rain_step_sum",
        ]

    def test_values_are_correct(self):
        """Aggregates match a hand-computed row."""
        out = StepAggregator(aggregations=("min", "max", "mean", "sum")).fit_transform(_frame())
        # Row 0 of temp is (1, 2, 9).
        assert out["temp_step_min"][0] == 1.0
        assert out["temp_step_max"][0] == 9.0
        assert out["temp_step_mean"][0] == pytest.approx(4.0)
        assert out["temp_step_sum"][0] == 12.0

    def test_raw_step_columns_are_gone(self):
        """The block is replaced, not augmented; augmentation is a FeatureUnion's job."""
        out = StepAggregator().fit_transform(_frame())
        assert not [c for c in out.columns if c.endswith(("_step_1", "_step_2", "_step_3"))]

    def test_null_policy_ignore(self):
        """ "ignore" summarises over the steps that carry a value."""
        out = StepAggregator(aggregations=("mean",), null_policy="ignore").fit_transform(_frame(with_nulls=True))
        # Row 0 covers only (1, 2).
        assert out["temp_step_mean"][0] == pytest.approx(1.5)

    def test_null_policy_propagate(self):
        """ "propagate" yields null for any row with a missing step."""
        out = StepAggregator(aggregations=("mean",), null_policy="propagate").fit_transform(_frame(with_nulls=True))
        assert out["temp_step_mean"][0] is None
        assert out["temp_step_mean"][2] is not None

    def test_coverage_companion_off_by_default(self):
        """Under full coverage the companion would be constant, so it is opt-in."""
        out = StepAggregator().fit_transform(_frame())
        assert not [c for c in out.columns if c.endswith("_n_covered")]

    def test_coverage_companion_counts_contributing_steps(self):
        """The companion distinguishes a mean over 2 steps from one over 3."""
        out = StepAggregator(emit_coverage=True).fit_transform(_frame(with_nulls=True))
        assert out["temp_step_n_covered"].to_list() == [2, 2, 3, 3, 3, 3]

    def test_unknown_aggregation_refused(self):
        """The vocabulary is closed."""
        with pytest.raises(ValueError, match="Unknown aggregation"):
            StepAggregator(aggregations=("median",)).fit(_frame())

    def test_numeric_name_refused(self):
        """A bare-integer name would be indistinguishable from a step index."""
        with pytest.raises(ValueError, match="Unknown aggregation"):
            StepAggregator(aggregations=("3",)).fit(_frame())

    def test_empty_aggregations_refused(self):
        """At least one summary must be named."""
        with pytest.raises(ValueError, match="at least one summary"):
            StepAggregator(aggregations=()).fit(_frame())

    def test_feature_names_out(self):
        """get_feature_names_out matches the transform output."""
        t = StepAggregator(aggregations=("mean", "max"), emit_coverage=True).fit(_frame())
        out = t.transform(_frame())
        assert t.get_feature_names_out() == [c for c in out.columns if c != "time"]


class TestNoStepBlocks:
    """A frame with nothing to reduce is an error, not an empty result."""

    @pytest.fixture
    def blockless(self) -> pl.DataFrame:
        """A frame whose columns are all horizon-agnostic."""
        return pl.DataFrame({
            "time": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(4)],
            "temp_step_mean": [1.0, 2.0, 3.0, 4.0],
            "temp_step_c0": [0.5, 1.5, 2.5, 3.5],
        })

    @pytest.mark.parametrize("transformer", TRANSFORMERS, ids=lambda t: type(t).__name__)
    def test_refused_with_a_diagnosis(self, blockless, transformer):
        """Every step transformer refuses rather than emitting an empty frame."""
        with pytest.raises(ValueError, match="no step blocks to reduce"):
            clone(transformer).fit(blockless)

    def test_message_names_what_it_received(self, blockless):
        """The message shows the columns present, so the mistake is visible."""
        with pytest.raises(ValueError) as excinfo:
            StepAggregator().fit(blockless)
        assert "temp_step_mean" in str(excinfo.value)

    def test_chaining_two_step_transformers_is_caught(self):
        """The realistic cause: a pipeline whose first stage leaves nothing indexed.

        Silently returning an empty frame here would drop every step feature from
        the design matrix with nothing to say so.
        """
        pipeline = FeaturePipeline([
            ("reduce", StepColumnReducer(reducer=SklearnStandardScaler())),
            ("aggregate", StepAggregator()),
        ])
        with pytest.raises(ValueError, match="chaining two step transformers"):
            pipeline.fit(_frame())


class TestStepColumnReducer:
    """One inner estimator per base column."""

    def test_scaler_preserves_width_per_base(self):
        """A width-preserving inner estimator yields one column per input step."""
        out = StepColumnReducer(reducer=SklearnStandardScaler()).fit_transform(_frame())
        assert out.columns == [
            "time",
            "temp_step_c0",
            "temp_step_c1",
            "temp_step_c2",
            "rain_step_c0",
            "rain_step_c1",
            "rain_step_c2",
        ]

    def test_scaler_standardises_per_step_position(self):
        """Each step position is standardised independently across rows."""
        out = StepColumnReducer(reducer=SklearnStandardScaler()).fit_transform(_frame())
        column = out["temp_step_c0"].to_numpy()
        assert column.mean() == pytest.approx(0.0, abs=1e-12)
        assert column.std() == pytest.approx(1.0)

    def test_reducer_narrows_each_base_block(self):
        """A reducer emits n_components columns per base column."""
        out = StepColumnReducer(reducer=PCA(n_components=2)).fit_transform(_frame())
        assert out.columns == ["time", "temp_step_c0", "temp_step_c1", "rain_step_c0", "rain_step_c1"]

    def test_base_columns_stay_unmixed(self):
        """Dropping one base column leaves the other's output unchanged.

        This is the property that distinguishes per-column from whole-frame
        reduction, and the reason per-variable provenance survives.
        """
        both = StepColumnReducer(reducer=PCA(n_components=2)).fit_transform(_frame())
        temp_only_frame = _frame().drop("rain_step_1", "rain_step_2", "rain_step_3")
        temp_only = StepColumnReducer(reducer=PCA(n_components=2)).fit_transform(temp_only_frame)
        np.testing.assert_allclose(
            np.abs(both.select("temp_step_c0", "temp_step_c1").to_numpy()),
            np.abs(temp_only.select("temp_step_c0", "temp_step_c1").to_numpy()),
        )

    def test_function_transformer_is_the_escape_hatch(self):
        """An arbitrary reduction is reachable without opening StepAggregator's vocabulary."""
        p90 = SklearnFunctionTransformer(lambda a: np.percentile(a, 90, axis=1, keepdims=True))
        out = StepColumnReducer(reducer=p90).fit_transform(_frame())
        assert out.columns == ["time", "temp_step_c0", "rain_step_c0"]


class TestStepFrameReducer:
    """One inner estimator over the whole step frame."""

    def test_emits_prefixed_columns(self):
        """Output is named from the prefix, keeping the {base}_step_{name} convention."""
        out = StepFrameReducer(reducer=PCA(n_components=2), prefix="wx").fit_transform(_frame())
        assert out.columns == ["time", "wx_step_c0", "wx_step_c1"]

    def test_mixes_base_columns(self):
        """Whole-frame reduction blends variables, unlike the per-column wrapper."""
        both = StepFrameReducer(reducer=PCA(n_components=2), prefix="wx").fit_transform(_frame())
        temp_only_frame = _frame().drop("rain_step_1", "rain_step_2", "rain_step_3")
        temp_only = StepFrameReducer(reducer=PCA(n_components=2), prefix="wx").fit_transform(temp_only_frame)
        assert not np.allclose(np.abs(both.to_numpy()[:, 1:]), np.abs(temp_only.to_numpy()[:, 1:]))

    def test_empty_prefix_refused(self):
        """The prefix names the output block, so it cannot be empty."""
        with pytest.raises(ValueError, match="non-empty string"):
            StepFrameReducer(reducer=PCA(n_components=2), prefix="").fit(_frame())

    def test_numeric_prefix_refused(self):
        """A bare-integer prefix would collide with a horizon step index."""
        with pytest.raises(ValueError, match="bare integer"):
            StepFrameReducer(reducer=PCA(n_components=2), prefix="3").fit(_frame())


class TestFixedWidthGuard:
    """Output width must be knowable before fit, for the panel schema contract."""

    @pytest.mark.parametrize("n_components", [0.95, None, "mle"], ids=["float", "none", "mle"])
    def test_data_determined_width_refused(self, n_components):
        """A width the data decides cannot satisfy the per-group schema."""
        with pytest.raises(ValueError, match="fixed before fit"):
            StepColumnReducer(reducer=PCA(n_components=n_components)).fit(_frame())

    def test_integer_width_accepted(self):
        """A positive integer is fine."""
        StepColumnReducer(reducer=TruncatedSVD(n_components=2)).fit(_frame())

    def test_width_preserving_estimator_accepted(self):
        """An estimator carrying no n_components at all is not refused."""
        StepColumnReducer(reducer=SklearnStandardScaler()).fit(_frame())


class TestMinSteps:
    """The wrapper's declared minimum block width."""

    def test_read_from_n_components(self):
        """A projection onto k components needs at least k step columns."""
        assert StepColumnReducer(reducer=PCA(n_components=2)).min_steps == 2

    def test_defaults_to_one_without_n_components(self):
        """No stated requirement means no requirement."""
        assert StepColumnReducer(reducer=SklearnStandardScaler()).min_steps == 1

    def test_readable_before_fit(self):
        """The forecaster reads this before fitting the slot, so it must not need one."""
        assert StepFrameReducer(reducer=PCA(n_components=3), prefix="wx").min_steps == 3


class TestNullGuard:
    """Partial coverage is diagnosed by trying, not by reading a tag."""

    def test_failing_estimator_gets_a_coverage_diagnosis(self):
        """The message names the column, its coverage, and both remedies."""
        with pytest.raises(ValueError) as excinfo:
            StepColumnReducer(reducer=PCA(n_components=2)).fit(_frame(with_nulls=True))
        message = str(excinfo.value)
        assert "'temp'" in message
        assert "2 of 3 steps" in message
        assert "SimpleImputer" in message
        assert "StepAggregator" in message

    def test_inner_exception_is_chained(self):
        """sklearn's own error stays visible underneath ours."""
        with pytest.raises(ValueError) as excinfo:
            StepColumnReducer(reducer=PCA(n_components=2)).fit(_frame(with_nulls=True))
        assert excinfo.value.__cause__ is not None

    def test_imputing_pipeline_is_accepted(self):
        """The recommended remedy works, despite its allow_nan tag reporting False.

        scikit-learn takes a pipeline's input tag from its last step, so a tag gate
        would reject this composition. Trying does not.
        """
        inner = Pipeline([("impute", SimpleImputer()), ("reduce", PCA(n_components=2))])
        assert inner.__sklearn_tags__().input_tags.allow_nan is False
        out = StepColumnReducer(reducer=inner).fit_transform(_frame(with_nulls=True))
        assert out.columns == ["time", "temp_step_c0", "temp_step_c1", "rain_step_c0", "rain_step_c1"]

    def test_natively_tolerant_estimator_is_accepted(self):
        """An estimator that handles nulls itself passes through."""
        out = StepColumnReducer(reducer=SklearnStandardScaler()).fit_transform(_frame(with_nulls=True))
        assert "temp_step_c0" in out.columns

    def test_unrelated_failure_is_not_relabelled(self):
        """A failure on a fully covered block propagates unchanged."""
        with pytest.raises(RuntimeError, match="inner estimator exploded"):
            StepColumnReducer(reducer=_AlwaysFails()).fit(_frame())

    def test_guard_silent_under_full_coverage(self):
        """No diagnosis is produced when nothing is missing."""
        StepColumnReducer(reducer=PCA(n_components=2)).fit(_frame())

    def test_frame_reducer_guard_too(self):
        """The whole-frame wrapper shares the guard."""
        with pytest.raises(ValueError, match="only partially covered"):
            StepFrameReducer(reducer=PCA(n_components=2), prefix="wx").fit(_frame(with_nulls=True))
