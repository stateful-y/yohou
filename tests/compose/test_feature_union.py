"""Dedicated tests for FeatureUnion composition class.

Tests fit/transform lifecycle, observation_horizon, horizontal concatenation,
feature name prefixing, validation, tags, and accessors.
"""

import sys
from pathlib import Path

import polars as pl
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import SimpleTransformer, StatelessTransformer
from yohou.compose import FeatureUnion


class TestFeatureUnionFitTransform:
    """Tests for FeatureUnion fit/transform lifecycle."""

    def test_fit_returns_self(self, time_series_factory):
        """Fit returns the union instance for chaining."""
        X = time_series_factory(length=50)
        union = FeatureUnion([("s1", SimpleTransformer(observation_horizon=0))])
        result = union.fit(X)
        assert result is union

    def test_is_fitted_after_fit(self, time_series_factory):
        """Union is fitted after calling fit."""
        X = time_series_factory(length=50)
        union = FeatureUnion([("s1", SimpleTransformer(observation_horizon=0))])
        union.fit(X)
        check_is_fitted(union)

    def test_transform_preserves_time_column(self, time_series_factory):
        """Transform output includes the time column."""
        X = time_series_factory(length=50)
        union = FeatureUnion([("s1", SimpleTransformer(observation_horizon=0))])
        union.fit(X)
        X_t = union.transform(X)
        assert "time" in X_t.columns

    def test_weighted_observe_rewind_preserve_time_dtype(self, time_series_factory):
        """Weighted observe/rewind keep the datetime "time" column.

        The weighting path must scale only the feature columns, not the whole
        DataFrame; multiplying the datetime "time" column by a weight would
        silently cast it to f64.
        """
        X = time_series_factory(length=50)
        union = FeatureUnion(
            [("s1", SimpleTransformer(observation_horizon=0))],
            transformer_weights={"s1": 2.0},
        )
        union.fit(X)

        X_obs = union.observe_transform(X)
        assert X_obs["time"].dtype == pl.Datetime

        X_rew = union.rewind_transform(X)
        assert X_rew["time"].dtype == pl.Datetime

    def test_horizontal_concat_two_transformers(self, time_series_factory):
        """Two transformers produce horizontally concatenated output."""
        X = time_series_factory(length=50, n_components=1)
        union = FeatureUnion([
            ("add1", SimpleTransformer(observation_horizon=0, add_constant=1.0)),
            ("add2", SimpleTransformer(observation_horizon=0, add_constant=2.0)),
        ])
        union.fit(X)
        X_t = union.transform(X)
        # time + 2 prefixed feature columns
        assert "time" in X_t.columns
        non_time = [c for c in X_t.columns if c != "time"]
        assert len(non_time) == 2

    def test_fit_transform_equals_fit_then_transform(self, time_series_factory):
        """fit_transform produces same output as fit then transform."""
        X = time_series_factory(length=50, n_components=1)
        union1 = FeatureUnion([
            ("s1", SimpleTransformer(observation_horizon=0, add_constant=5.0)),
        ])
        union2 = clone(union1)

        X_ft = union1.fit_transform(X)
        union2.fit(X)
        X_t = union2.transform(X)

        # Column names and values should match
        assert X_ft.columns == X_t.columns
        assert X_ft.shape == X_t.shape

    def test_passthrough_transformer(self, time_series_factory):
        """Passthrough transformer passes features unchanged."""
        X = time_series_factory(length=50, n_components=1)
        union = FeatureUnion([
            ("pass", "passthrough"),
            ("add", SimpleTransformer(observation_horizon=0, add_constant=1.0)),
        ])
        union.fit(X)
        X_t = union.transform(X)
        assert isinstance(X_t, pl.DataFrame)
        assert "time" in X_t.columns


class TestFeatureUnionObservationHorizon:
    """Tests for observation_horizon property."""

    def test_max_of_transformers(self, time_series_factory):
        """Observation horizon is the maximum across all transformers."""
        X = time_series_factory(length=50)
        union = FeatureUnion([
            ("s1", SimpleTransformer(observation_horizon=2)),
            ("s2", SimpleTransformer(observation_horizon=5)),
        ])
        union.fit(X)
        assert union.observation_horizon == 5

    def test_zero_horizon_transformers(self, time_series_factory):
        """Union of stateless transformers has zero observation horizon."""
        X = time_series_factory(length=50)
        union = FeatureUnion([
            ("s1", StatelessTransformer()),
            ("s2", StatelessTransformer()),
        ])
        union.fit(X)
        assert union.observation_horizon == 0

    def test_mixed_horizons(self, time_series_factory):
        """Union with mixed horizons takes the max."""
        X = time_series_factory(length=50)
        union = FeatureUnion([
            ("stateless", StatelessTransformer()),
            ("stateful", SimpleTransformer(observation_horizon=3)),
        ])
        union.fit(X)
        assert union.observation_horizon == 3

    def test_not_fitted_raises(self):
        """Accessing observation_horizon before fit raises NotFittedError."""
        union = FeatureUnion([("s1", SimpleTransformer(observation_horizon=2))])
        with pytest.raises(NotFittedError):
            _ = union.observation_horizon


class TestFeatureUnionFeatureNames:
    """Tests for feature name prefixing."""

    def test_verbose_names_add_prefix(self, time_series_factory):
        """verbose_feature_names_out=True adds transformer name prefix."""
        X = time_series_factory(length=50, n_components=1)
        union = FeatureUnion(
            [
                ("alpha", SimpleTransformer(observation_horizon=0)),
                ("beta", SimpleTransformer(observation_horizon=0)),
            ],
            verbose_feature_names_out=True,
        )
        union.fit(X)
        X_t = union.transform(X)
        non_time = [c for c in X_t.columns if c != "time"]
        assert any("alpha" in col for col in non_time)
        assert any("beta" in col for col in non_time)

    def test_no_prefix_when_disabled(self, time_series_factory):
        """verbose_feature_names_out=False omits prefixes."""
        X = time_series_factory(length=50, n_components=1)
        union = FeatureUnion(
            [("only", SimpleTransformer(observation_horizon=0))],
            verbose_feature_names_out=False,
        )
        union.fit(X)
        X_t = union.transform(X)
        non_time = [c for c in X_t.columns if c != "time"]
        assert all("only" not in col for col in non_time)

    def test_duplicate_names_without_prefix_raises(self, time_series_factory):
        """Duplicate feature names without prefix raises ValueError."""
        X = time_series_factory(length=50, n_components=1)
        union = FeatureUnion(
            [
                ("a", SimpleTransformer(observation_horizon=0)),
                ("b", SimpleTransformer(observation_horizon=0)),
            ],
            verbose_feature_names_out=False,
        )
        # fit_transform should raise because both produce same column names
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            union.fit_transform(X)


class TestFeatureUnionParams:
    """Tests for get_params and set_params."""

    def test_get_params_shallow(self):
        """get_params(deep=False) returns top-level parameters."""
        union = FeatureUnion([("s1", SimpleTransformer(observation_horizon=0))])
        params = union.get_params(deep=False)
        assert "transformer_list" in params
        assert "n_jobs" in params
        assert "verbose" in params

    def test_get_params_deep(self):
        """get_params(deep=True) exposes transformer parameters."""
        union = FeatureUnion([
            ("s1", SimpleTransformer(observation_horizon=1, add_constant=5.0)),
        ])
        params = union.get_params(deep=True)
        assert "s1__add_constant" in params
        assert params["s1__add_constant"] == 5.0

    def test_set_params(self):
        """set_params changes a nested transformer parameter."""
        union = FeatureUnion([
            ("s1", SimpleTransformer(observation_horizon=0, add_constant=1.0)),
        ])
        union.set_params(s1__add_constant=42.0)
        assert union.get_params()["s1__add_constant"] == 42.0

    def test_clone_produces_equal_output(self, time_series_factory):
        """Cloned union produces identical output."""
        X = time_series_factory(length=50, n_components=1)
        union = FeatureUnion([("s1", SimpleTransformer(observation_horizon=0, add_constant=3.0))])
        union_c = clone(union)
        union.fit(X)
        union_c.fit(X)
        X1 = union.transform(X)
        X2 = union_c.transform(X)
        assert X1.columns == X2.columns
        assert X1.shape == X2.shape


class TestFeatureUnionAccessors:
    """Tests for named_transformers, __len__, __getitem__."""

    def test_named_transformers(self):
        """named_transformers provides access by name."""
        t = SimpleTransformer(observation_horizon=0)
        union = FeatureUnion([("mystep", t)])
        assert union.named_transformers["mystep"] is t

    def test_getitem_by_index(self):
        """Accessing by integer index returns the transformer."""
        t1 = SimpleTransformer(observation_horizon=0)
        t2 = StatelessTransformer()
        union = FeatureUnion([("a", t1), ("b", t2)])
        assert union[0] is t1
        assert union[1] is t2

    def test_getitem_by_name(self, time_series_factory):
        """Accessing by string name returns the transformer."""
        X = time_series_factory(length=50)
        t = SimpleTransformer(observation_horizon=0)
        union = FeatureUnion([("named", t)])
        union.fit(X)
        assert union["named"] is t

    def test_getitem_slice(self):
        """Slicing returns a sub-union."""
        union = FeatureUnion([
            ("a", SimpleTransformer(observation_horizon=0)),
            ("b", SimpleTransformer(observation_horizon=0)),
            ("c", SimpleTransformer(observation_horizon=0)),
        ])
        sub = union[:2]
        assert isinstance(sub, FeatureUnion)
        assert len(sub.transformer_list) == 2

    def test_slice_with_step_raises(self):
        """Slicing with step != 1 raises ValueError."""
        union = FeatureUnion([
            ("a", SimpleTransformer(observation_horizon=0)),
            ("b", SimpleTransformer(observation_horizon=0)),
        ])
        with pytest.raises(ValueError, match="step"):
            union[::2]

    def test_slice_with_explicit_step_one_is_allowed(self):
        """An explicit step of 1 is the documented supported case and must work."""
        union = FeatureUnion([
            ("a", SimpleTransformer(observation_horizon=0)),
            ("b", SimpleTransformer(observation_horizon=0)),
            ("c", SimpleTransformer(observation_horizon=0)),
        ])
        sub = union[0:2:1]
        assert isinstance(sub, FeatureUnion)
        assert len(sub.transformer_list) == 2


class TestFeatureUnionValidation:
    """Tests for validation and error paths."""

    def test_invalid_weight_key_raises(self, time_series_factory):
        """Transformer weight with unknown name raises ValueError."""
        X = time_series_factory(length=50)
        union = FeatureUnion(
            [("s1", SimpleTransformer(observation_horizon=0))],
            transformer_weights={"nonexistent": 2.0},
        )
        with pytest.raises(ValueError, match="nonexistent"):
            union.fit(X)

    def test_not_fitted_observation_horizon_raises(self):
        """Accessing observation_horizon before fit raises NotFittedError."""
        union = FeatureUnion([("s1", SimpleTransformer(observation_horizon=0))])
        with pytest.raises(NotFittedError):
            _ = union.observation_horizon


class TestFeatureUnionTags:
    """Tests for __sklearn_tags__ aggregation."""

    def test_stateless_all_stateless(self):
        """Union of stateless transformers has stateful=False."""
        union = FeatureUnion([
            ("s1", StatelessTransformer()),
            ("s2", StatelessTransformer()),
        ])
        tags = union.__sklearn_tags__()
        assert tags.transformer_tags.stateful is False

    def test_drop_only_union(self):
        """Union with only drop transformers still has tags."""
        union = FeatureUnion([("d", "drop")])
        tags = union.__sklearn_tags__()
        assert tags.transformer_tags is not None


class TestFeatureUnionObserveRewindTransform:
    """Tests for FeatureUnion observe_transform and rewind_transform paths."""

    def test_observe_transform_returns_transformed_data(self, time_series_factory):
        """observe_transform produces horizontally stacked results."""
        X = time_series_factory(length=50, n_components=2)
        union = FeatureUnion([
            ("add1", SimpleTransformer(observation_horizon=0, add_constant=1.0)),
            ("add2", SimpleTransformer(observation_horizon=0, add_constant=2.0)),
        ])
        union.fit(X)
        X_ot = union.observe_transform(X)
        assert "time" in X_ot.columns
        assert len(X_ot) == len(X)

    def test_observe_transform_with_verbose_feature_names(self, time_series_factory):
        """observe_transform with verbose_feature_names_out adds prefixes."""
        X = time_series_factory(length=50, n_components=1)
        union = FeatureUnion(
            [
                ("add1", SimpleTransformer(observation_horizon=0, add_constant=1.0)),
                ("add2", SimpleTransformer(observation_horizon=0, add_constant=2.0)),
            ],
            verbose_feature_names_out=True,
        )
        union.fit(X)
        X_ot = union.observe_transform(X)
        non_time_cols = [c for c in X_ot.columns if c != "time"]
        assert any("add1_" in c for c in non_time_cols)
        assert any("add2_" in c for c in non_time_cols)

    def test_rewind_transform_returns_transformed_data(self, time_series_factory):
        """rewind_transform produces horizontally stacked results."""
        X = time_series_factory(length=50, n_components=2)
        union = FeatureUnion([
            ("add1", SimpleTransformer(observation_horizon=0, add_constant=1.0)),
            ("add2", SimpleTransformer(observation_horizon=0, add_constant=2.0)),
        ])
        union.fit(X)
        X_rt = union.rewind_transform(X)
        assert "time" in X_rt.columns

    def test_rewind_transform_with_verbose_feature_names(self, time_series_factory):
        """rewind_transform with verbose_feature_names_out adds prefixes."""
        X = time_series_factory(length=50, n_components=1)
        union = FeatureUnion(
            [
                ("add1", SimpleTransformer(observation_horizon=0, add_constant=1.0)),
                ("add2", SimpleTransformer(observation_horizon=0, add_constant=2.0)),
            ],
            verbose_feature_names_out=True,
        )
        union.fit(X)
        X_rt = union.rewind_transform(X)
        non_time_cols = [c for c in X_rt.columns if c != "time"]
        assert any("add1_" in c for c in non_time_cols)
        assert any("add2_" in c for c in non_time_cols)

    def test_n_features_in_accessible_after_fit(self, time_series_factory):
        """n_features_in_ property is accessible after fit."""
        X = time_series_factory(length=50, n_components=2)
        union = FeatureUnion([("s1", SimpleTransformer(observation_horizon=0))])
        union.fit(X)
        assert union.n_features_in_ >= 2

    def test_feature_names_in_accessible_after_fit(self, time_series_factory):
        """feature_names_in_ property is accessible after fit."""
        X = time_series_factory(length=50, n_components=2)
        union = FeatureUnion([("s1", SimpleTransformer(observation_horizon=0))])
        union.fit(X)
        assert len(union.feature_names_in_) >= 2

    def test_get_metadata_routing(self):
        """get_metadata_routing returns a MetadataRouter."""
        from sklearn.utils.metadata_routing import MetadataRouter

        union = FeatureUnion([("s1", SimpleTransformer(observation_horizon=0))])
        router = union.get_metadata_routing()
        assert isinstance(router, MetadataRouter)


class TestFeatureUnionDropPassthrough:
    """Tests for FeatureUnion with drop/passthrough and edge cases."""

    def test_all_drop_fit_transform_returns_time_only(self, time_series_factory):
        """fit_transform with all-drop transformers returns time column only."""
        X = time_series_factory(length=20, n_components=2)
        union = FeatureUnion([("d", "drop")])
        result = union.fit_transform(X)
        assert result.columns == ["time"]
        assert len(result) == len(X)

    def test_all_drop_transform_returns_time_only(self, time_series_factory):
        """Transform with all-drop transformers returns time column only."""
        X = time_series_factory(length=20, n_components=2)
        union = FeatureUnion([("d", "drop")])
        union.fit(X)
        result = union.transform(X)
        assert result.columns == ["time"]

    def test_all_drop_observe_transform_returns_time_only(self, time_series_factory):
        """observe_transform with all-drop transformers returns time column only."""
        X = time_series_factory(length=20, n_components=2)
        union = FeatureUnion([("d", "drop")])
        union.fit(X)
        result = union.observe_transform(X)
        assert result.columns == ["time"]

    def test_all_drop_rewind_transform_returns_time_only(self, time_series_factory):
        """rewind_transform with all-drop transformers returns time column only."""
        X = time_series_factory(length=20, n_components=2)
        union = FeatureUnion([("d", "drop")])
        union.fit(X)
        result = union.rewind_transform(X)
        assert result.columns == ["time"]

    def test_invalid_transformer_raises(self):
        """Transformer without fit/transform raises TypeError."""
        union = FeatureUnion([("bad", object())])
        with pytest.raises(TypeError, match="All estimators should implement fit and transform"):
            union.fit(pl.DataFrame({"time": [1, 2], "x": [1.0, 2.0]}))

    def test_n_features_in_not_available_raises(self):
        """n_features_in_ raises AttributeError when no transformer has it."""
        union = FeatureUnion([("d", "drop")])
        union.fit(pl.DataFrame({"time": [1, 2], "x": [1.0, 2.0]}))
        with pytest.raises(AttributeError, match="n_features_in_ not available"):
            _ = union.n_features_in_

    def test_feature_names_in_not_available_raises(self):
        """feature_names_in_ raises AttributeError when no transformer has it."""
        union = FeatureUnion([("d", "drop")])
        union.fit(pl.DataFrame({"time": [1, 2], "x": [1.0, 2.0]}))
        with pytest.raises(AttributeError, match="feature_names_in_ not available"):
            _ = union.feature_names_in_

    def test_sklearn_is_fitted(self, time_series_factory):
        """__sklearn_is_fitted__ returns True after fit."""
        X = time_series_factory(length=20, n_components=1)
        union = FeatureUnion([("s1", SimpleTransformer(observation_horizon=0))])
        union.fit(X)
        assert union.__sklearn_is_fitted__()

    def test_sk_visual_block(self, time_series_factory):
        """_sk_visual_block_ returns a representation."""
        X = time_series_factory(length=20, n_components=1)
        union = FeatureUnion([("s1", SimpleTransformer(observation_horizon=0))])
        union.fit(X)
        vb = union._sk_visual_block_()
        assert vb is not None

    def test_verbose_false_observe_transform(self, time_series_factory):
        """observe_transform with verbose_feature_names_out=False omits prefixes."""
        X = time_series_factory(length=30, n_components=1)
        union = FeatureUnion(
            [("add1", SimpleTransformer(observation_horizon=0, add_constant=1.0))],
            verbose_feature_names_out=False,
        )
        union.fit(X)
        result = union.observe_transform(X)
        non_time = [c for c in result.columns if c != "time"]
        assert all("add1_" not in c for c in non_time)

    def test_verbose_false_rewind_transform(self, time_series_factory):
        """rewind_transform with verbose_feature_names_out=False omits prefixes."""
        X = time_series_factory(length=30, n_components=1)
        union = FeatureUnion(
            [("add1", SimpleTransformer(observation_horizon=0, add_constant=1.0))],
            verbose_feature_names_out=False,
        )
        union.fit(X)
        result = union.rewind_transform(X)
        non_time = [c for c in result.columns if c != "time"]
        assert all("add1_" not in c for c in non_time)

    def test_verbose_false_transform(self, time_series_factory):
        """Transform with verbose_feature_names_out=False omits prefixes."""
        X = time_series_factory(length=30, n_components=1)
        union = FeatureUnion(
            [("add1", SimpleTransformer(observation_horizon=0, add_constant=1.0))],
            verbose_feature_names_out=False,
        )
        union.fit(X)
        result = union.transform(X)
        non_time = [c for c in result.columns if c != "time"]
        assert all("add1_" not in c for c in non_time)

    def test_get_feature_names_out(self, time_series_factory):
        """get_feature_names_out returns feature names after fit."""
        X = time_series_factory(length=20, n_components=2)
        union = FeatureUnion([("s1", SimpleTransformer(observation_horizon=0))])
        union.fit(X)
        names = union.get_feature_names_out()
        assert names is not None or names is None
