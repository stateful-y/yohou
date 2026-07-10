"""Dedicated tests for FeaturePipeline composition class.

Tests lifecycle, validation, error paths, observe/rewind/transform,
inverse_transform, parameter routing, and systematic checks.
"""

import sys
from pathlib import Path

import polars as pl
import polars.selectors as cs
import pytest
from polars.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import InvertibleTransformer, SimpleTransformer, StatelessTransformer
from yohou.compose import FeaturePipeline


class TestFeaturePipelineFitTransform:
    """Tests for FeaturePipeline fit/transform lifecycle."""

    def test_fit_returns_self(self, time_series_factory):
        """Fit returns the pipeline instance for chaining."""
        X = time_series_factory(length=50)
        pipe = FeaturePipeline([("s1", SimpleTransformer(observation_horizon=1))])
        result = pipe.fit(X)
        assert result is pipe

    def test_transform_preserves_time_column(self, time_series_factory):
        """Transform output includes the time column."""
        X = time_series_factory(length=50)
        pipe = FeaturePipeline([("s1", SimpleTransformer(observation_horizon=0, add_constant=1.0))])
        pipe.fit(X)
        X_t = pipe.transform(X)
        assert "time" in X_t.columns

    def test_fit_transform_equals_fit_then_transform(self, time_series_factory):
        """fit_transform produces same output as fit then transform."""
        X = time_series_factory(length=50)
        pipe1 = FeaturePipeline([("s1", SimpleTransformer(observation_horizon=0, add_constant=5.0))])
        pipe2 = clone(pipe1)

        X_ft = pipe1.fit_transform(X)
        pipe2.fit(X)
        X_t = pipe2.transform(X)

        assert_frame_equal(X_ft, X_t)

    def test_sequential_addition(self, time_series_factory):
        """Two additive steps produce cumulative result."""
        X = time_series_factory(length=50)
        pipe = FeaturePipeline([
            ("add3", SimpleTransformer(observation_horizon=0, add_constant=3.0)),
            ("add7", SimpleTransformer(observation_horizon=0, add_constant=7.0)),
        ])
        pipe.fit(X)
        X_t = pipe.transform(X)
        X_expected = X.select([pl.col("time"), (cs.numeric() & ~cs.by_name("time")) + 10.0])
        assert_frame_equal(X_t, X_expected)

    def test_single_step_passthrough(self, time_series_factory):
        """Pipeline with passthrough step passes data unchanged."""
        X = time_series_factory(length=50)
        pipe = FeaturePipeline([("pass", "passthrough")])
        pipe.fit(X)
        X_t = pipe.transform(X)
        assert_frame_equal(X_t, X)

    def test_is_fitted_after_fit(self, time_series_factory):
        """Pipeline is fitted after calling fit."""
        X = time_series_factory(length=50)
        pipe = FeaturePipeline([("s1", SimpleTransformer(observation_horizon=0))])
        pipe.fit(X)
        check_is_fitted(pipe)


class TestFeaturePipelineObservationHorizon:
    """Tests for observation_horizon property."""

    def test_sum_of_steps(self, time_series_factory):
        """Observation horizon is the sum across all steps."""
        X = time_series_factory(length=50)
        pipe = FeaturePipeline([
            ("s1", SimpleTransformer(observation_horizon=2)),
            ("s2", SimpleTransformer(observation_horizon=3)),
        ])
        pipe.fit(X)
        assert pipe.observation_horizon == 5

    def test_zero_horizon_steps(self, time_series_factory):
        """Pipeline of stateless transformers has zero observation horizon."""
        X = time_series_factory(length=50)
        pipe = FeaturePipeline([
            ("s1", StatelessTransformer()),
            ("s2", StatelessTransformer()),
        ])
        pipe.fit(X)
        assert pipe.observation_horizon == 0

    def test_mixed_horizons(self, time_series_factory):
        """Pipeline with mixed stateful/stateless has correct horizon."""
        X = time_series_factory(length=50)
        pipe = FeaturePipeline([
            ("stateless", StatelessTransformer()),
            ("stateful", SimpleTransformer(observation_horizon=3)),
        ])
        pipe.fit(X)
        assert pipe.observation_horizon == 3


class TestFeaturePipelineObserveRewind:
    """Tests for observe and rewind temporal methods."""

    def test_observe_returns_self(self, time_series_factory):
        """Observe returns the pipeline instance."""
        X_all = time_series_factory(length=60)
        X_fit = X_all[:50]
        X_new = X_all[50:]
        pipe = FeaturePipeline([("s1", SimpleTransformer(observation_horizon=0))])
        pipe.fit(X_fit)
        result = pipe.observe(X_new)
        assert result is pipe

    def test_rewind_returns_self(self, time_series_factory):
        """Rewind returns the pipeline instance."""
        X = time_series_factory(length=50)
        pipe = FeaturePipeline([("s1", SimpleTransformer(observation_horizon=0))])
        pipe.fit(X)
        result = pipe.rewind(X)
        assert result is pipe

    def test_observe_transform_produces_output(self, time_series_factory):
        """observe_transform produces transformed output."""
        X_all = time_series_factory(length=60)
        X_fit = X_all[:50]
        X_new = X_all[50:]
        pipe = FeaturePipeline([("s1", SimpleTransformer(observation_horizon=0, add_constant=1.0))])
        pipe.fit(X_fit)
        X_ot = pipe.observe_transform(X_new)
        assert isinstance(X_ot, pl.DataFrame)
        assert "time" in X_ot.columns
        assert len(X_ot) > 0

    def test_rewind_transform_produces_output(self, time_series_factory):
        """rewind_transform produces transformed output."""
        X_all = time_series_factory(length=60)
        X_fit = X_all[:50]
        X_new = X_all[50:]
        pipe = FeaturePipeline([("s1", SimpleTransformer(observation_horizon=0, add_constant=2.0))])
        pipe.fit(X_fit)
        pipe.observe(X_new)
        X_rt = pipe.rewind_transform(X_fit)
        assert isinstance(X_rt, pl.DataFrame)
        assert "time" in X_rt.columns


class TestFeaturePipelineInverseTransform:
    """Tests for inverse_transform capability."""

    def test_inverse_tags_all_invertible(self):
        """Pipeline reports invertible when all steps have inverse_transform."""
        pipe = FeaturePipeline([
            ("inv", InvertibleTransformer(observation_horizon=0, offset=5.0)),
        ])
        tags = pipe.__sklearn_tags__()
        assert tags.transformer_tags.invertible is True

    def test_inverse_tags_not_invertible_without_method(self):
        """Pipeline reports not-invertible when steps lack inverse_transform."""
        pipe = FeaturePipeline([("s1", StatelessTransformer())])
        tags = pipe.__sklearn_tags__()
        assert tags.transformer_tags.invertible is False

    def test_three_step_stateful_inverse_round_trips(self):
        """Three stateful steps with distinct horizons must invert correctly.

        Regression: the inverse_transform X_p context for each step must be
        paired with that step's input space. With the wrong ordering, the most-
        transformed context was paired with the last step's inverse, corrupting
        the round trip for pipelines of three or more stateful transformers.
        """
        from datetime import datetime, timedelta

        from yohou.stationarity import SeasonalDifferencing

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=39),
            interval="1d",
            eager=True,
        )
        X = pl.DataFrame({"time": time, "v": [float(i * i % 17 + i) for i in range(40)]})

        pipe = FeaturePipeline([
            ("d1", SeasonalDifferencing(seasonality=1)),
            ("d2", SeasonalDifferencing(seasonality=2)),
            ("d3", SeasonalDifferencing(seasonality=3)),
        ])
        pipe.fit(X)
        X_t = pipe.transform(X)

        recovered = pipe.inverse_transform(X_t=X_t, X_p=X)

        # Recovered values must match the original on every overlapping timestamp.
        merged = recovered.join(X, on="time", suffix="_orig")
        max_abs_diff = merged.select((pl.col("v") - pl.col("v_orig")).abs().max()).item()
        assert max_abs_diff == pytest.approx(0.0, abs=1e-9)


class TestFeaturePipelineParams:
    """Tests for get_params and set_params."""

    def test_get_params_contains_steps(self):
        """get_params returns steps in shallow mode."""
        pipe = FeaturePipeline([("s1", SimpleTransformer(observation_horizon=1))])
        params = pipe.get_params(deep=False)
        assert "steps" in params
        assert "memory" in params
        assert "verbose" in params

    def test_get_params_deep(self):
        """get_params(deep=True) exposes step parameters."""
        pipe = FeaturePipeline([
            ("s1", SimpleTransformer(observation_horizon=1, add_constant=5.0)),
        ])
        params = pipe.get_params(deep=True)
        assert "s1__add_constant" in params
        assert params["s1__add_constant"] == 5.0

    def test_set_params_step_parameter(self, time_series_factory):
        """set_params changes a nested step parameter."""
        pipe = FeaturePipeline([
            ("s1", SimpleTransformer(observation_horizon=0, add_constant=1.0)),
        ])
        pipe.set_params(s1__add_constant=99.0)
        assert pipe.get_params()["s1__add_constant"] == 99.0

    def test_clone_produces_equal_output(self, time_series_factory):
        """Cloned pipeline produces identical output."""
        X = time_series_factory(length=50)
        pipe = FeaturePipeline([("s1", SimpleTransformer(observation_horizon=0, add_constant=3.0))])
        pipe_c = clone(pipe)
        pipe.fit(X)
        pipe_c.fit(X)
        assert_frame_equal(pipe.transform(X), pipe_c.transform(X))


class TestFeaturePipelineAccessors:
    """Tests for named_steps, __len__, __getitem__."""

    def test_len(self):
        """Pipeline length matches number of steps."""
        pipe = FeaturePipeline([
            ("a", SimpleTransformer(observation_horizon=0)),
            ("b", SimpleTransformer(observation_horizon=0)),
        ])
        assert len(pipe) == 2

    def test_getitem_by_index(self):
        """Accessing pipeline by integer index returns the step estimator."""
        t = SimpleTransformer(observation_horizon=0)
        pipe = FeaturePipeline([("a", t)])
        assert pipe[0] is t

    def test_getitem_by_name(self, time_series_factory):
        """Accessing pipeline by name returns the step estimator."""
        X = time_series_factory(length=50)
        t = SimpleTransformer(observation_horizon=0)
        pipe = FeaturePipeline([("mystep", t)])
        pipe.fit(X)
        assert pipe["mystep"] is t

    def test_getitem_slice(self):
        """Slicing pipeline returns a sub-pipeline."""
        pipe = FeaturePipeline([
            ("a", SimpleTransformer(observation_horizon=0)),
            ("b", SimpleTransformer(observation_horizon=0)),
            ("c", SimpleTransformer(observation_horizon=0)),
        ])
        sub = pipe[:2]
        assert isinstance(sub, FeaturePipeline)
        assert len(sub) == 2

    def test_named_steps_access(self, time_series_factory):
        """named_steps provides attribute access to steps."""
        X = time_series_factory(length=50)
        pipe = FeaturePipeline([
            ("first", SimpleTransformer(observation_horizon=0)),
            ("second", SimpleTransformer(observation_horizon=0)),
        ])
        pipe.fit(X)
        assert "first" in pipe.named_steps
        assert "second" in pipe.named_steps

    def test_n_features_in(self, time_series_factory):
        """n_features_in_ returns count of non-time columns after fit."""
        X = time_series_factory(length=50, n_components=3)
        pipe = FeaturePipeline([("s1", SimpleTransformer(observation_horizon=0))])
        pipe.fit(X)
        assert pipe.n_features_in_ == 3

    def test_feature_names_in(self, time_series_factory):
        """feature_names_in_ returns column names after fit."""
        X = time_series_factory(length=50, n_components=2)
        pipe = FeaturePipeline([("s1", SimpleTransformer(observation_horizon=0))])
        pipe.fit(X)
        assert hasattr(pipe, "feature_names_in_")


class TestFeaturePipelineValidation:
    """Tests for validation and error paths."""

    def test_invalid_step_type_raises(self, time_series_factory):
        """Non-transformer step raises TypeError during fit."""
        X = time_series_factory(length=50)
        pipe = FeaturePipeline([("bad", 42)])
        with pytest.raises(TypeError, match="BaseActualTransformer|doesn't"):
            pipe.fit(X)

    def test_not_fitted_observe_raises(self, time_series_factory):
        """Calling observe before fit raises NotFittedError."""
        from sklearn.exceptions import NotFittedError

        X = time_series_factory(length=50)
        pipe = FeaturePipeline([("s1", SimpleTransformer(observation_horizon=0))])
        with pytest.raises(NotFittedError):
            pipe.observe(X)

    def test_observe_with_temporal_gap_raises(self, time_series_factory):
        """observe with a non-contiguous X raises a gap ValueError.

        FeaturePipeline.observe validates temporal continuity against the last
        observed timestamp; a batch that does not begin one interval after the
        previously observed data must be rejected.
        """
        X_all = time_series_factory(length=60)
        pipe = FeaturePipeline([("s1", SimpleTransformer(observation_horizon=0))])
        pipe.fit(X_all[:50])
        # Skip rows 50-51 so the next batch starts two intervals late.
        X_gapped = X_all[52:]
        with pytest.raises(ValueError, match="[Gg]ap"):
            pipe.observe(X_gapped)

    def test_slice_with_step_raises(self):
        """Slicing with step != 1 raises ValueError."""
        pipe = FeaturePipeline([
            ("a", SimpleTransformer(observation_horizon=0)),
            ("b", SimpleTransformer(observation_horizon=0)),
            ("c", SimpleTransformer(observation_horizon=0)),
        ])
        with pytest.raises(ValueError, match="step"):
            pipe[::2]


class TestFeaturePipelineTags:
    """Tests for __sklearn_tags__ aggregation."""

    def test_stateless_steps_produce_stateless_tag(self):
        """Pipeline of stateless transformers has stateful=False."""
        pipe = FeaturePipeline([
            ("s1", StatelessTransformer()),
            ("s2", StatelessTransformer()),
        ])
        tags = pipe.__sklearn_tags__()
        assert tags.transformer_tags.stateful is False

    def test_passthrough_only_pipeline(self):
        """Pipeline with only passthrough steps is stateless and non-invertible."""
        pipe = FeaturePipeline([("p", "passthrough")])
        tags = pipe.__sklearn_tags__()
        assert tags.transformer_tags.stateful is False
        assert tags.transformer_tags.invertible is False

    def test_stateful_step_produces_stateful_tag(self):
        """Pipeline with at least one stateful transformer has stateful=True."""
        from yohou.preprocessing.window import LagTransformer

        pipe = FeaturePipeline([
            ("s1", StatelessTransformer()),
            ("s2", LagTransformer(lag=[1])),
        ])
        tags = pipe.__sklearn_tags__()
        assert tags.transformer_tags.stateful is True

    def test_non_invertible_step_produces_non_invertible_tag(self):
        """Pipeline with non-invertible step has invertible=False."""
        pipe = FeaturePipeline([
            ("s1", StatelessTransformer()),
        ])
        tags = pipe.__sklearn_tags__()
        assert tags.transformer_tags.invertible is False

    def test_invertible_steps_produce_invertible_tag(self):
        """Pipeline of all invertible steps has invertible=True."""
        pipe = FeaturePipeline([
            ("s1", InvertibleTransformer()),
            ("s2", InvertibleTransformer()),
        ])
        tags = pipe.__sklearn_tags__()
        assert tags.transformer_tags.invertible is True

    def test_min_value_from_first_step(self):
        """min_value comes from the first transformer in the pipeline."""
        from yohou.preprocessing.window import LagTransformer

        pipe = FeaturePipeline([
            ("s1", StatelessTransformer()),
            ("s2", LagTransformer(lag=[1])),
        ])
        tags = pipe.__sklearn_tags__()
        first_min = StatelessTransformer().__sklearn_tags__().input_tags.min_value
        assert tags.input_tags.min_value == first_min
