"""Tests for composition classes (FeaturePipeline, FeatureUnion, ColumnTransformer).

Tests both the existing specific behavior and systematic validation
using dummy transformers.
"""

import sys
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).parent))
from conftest import SimpleTransformer
from yohou.compose.feature_pipeline import ColumnTransformer, FeaturePipeline, FeatureUnion


class TestFeaturePipeline:
    """Tests for FeaturePipeline composition behavior."""

    def test_observation_horizon_sum(self, dummy_transformers, time_series_factory):
        """Test observation_horizon is sum of component horizons."""
        X = time_series_factory(length=50)

        step1 = SimpleTransformer(observation_horizon=1)
        step2 = SimpleTransformer(observation_horizon=2)

        pipeline = FeaturePipeline([("step1", step1), ("step2", step2)])
        pipeline.fit(X)

        expected_horizon = 1 + 2
        assert pipeline.observation_horizon == expected_horizon, (
            f"Expected horizon {expected_horizon}, got {pipeline.observation_horizon}"
        )

    def test_sequential_execution(self, dummy_transformers, time_series_factory):
        """Test transformers execute sequentially."""
        X = time_series_factory(length=50)

        step1 = SimpleTransformer(observation_horizon=1, add_constant=10.0)
        step2 = SimpleTransformer(observation_horizon=1, add_constant=5.0)

        pipeline = FeaturePipeline([("step1", step1), ("step2", step2)])
        pipeline.fit(X)

        X_trans = pipeline.transform(X)

        X_expected = X.select([pl.col("time"), (pl.all().exclude("time") + 15.0)])
        assert_frame_equal(X_trans, X_expected)

    def test_named_access(self, dummy_transformers, time_series_factory):
        """Test named steps are accessible after fitting."""
        X = time_series_factory(length=50)

        step1 = SimpleTransformer(observation_horizon=1)
        step2 = SimpleTransformer(observation_horizon=2)

        pipeline = FeaturePipeline([("first", step1), ("second", step2)])
        pipeline.fit(X)

        assert hasattr(pipeline, "named_steps")
        assert "first" in pipeline.named_steps
        assert "second" in pipeline.named_steps

        params = pipeline.get_params(deep=True)
        assert "first" in params or "steps" in params, (
            f"Expected 'first' or 'steps' in params, got: {list(params.keys())}"
        )

    def test_with_clone(self, dummy_transformers, time_series_factory):
        """Test cloned pipeline produces identical output."""
        X = time_series_factory(length=50)

        pipeline = FeaturePipeline([
            ("step1", dummy_transformers["simple"]),
            ("step2", dummy_transformers["invertible"]),
        ])
        pipeline_clone = clone(pipeline)

        pipeline.fit(X)
        pipeline_clone.fit(X)

        X_trans1 = pipeline.transform(X)
        X_trans2 = pipeline_clone.transform(X)

        assert_frame_equal(X_trans1, X_trans2)

    def test_get_set_params(self, dummy_transformers, time_series_factory):
        """Test get_params and set_params work correctly."""
        X = time_series_factory(length=50)

        pipeline = FeaturePipeline([
            ("step1", SimpleTransformer(observation_horizon=1, add_constant=5.0)),
        ])

        params = pipeline.get_params(deep=True)
        assert "steps" in params, f"Expected 'steps' in params, got: {list(params.keys())}"
        assert "memory" in params

        pipeline.fit(X)
        assert "step1" in pipeline.named_steps
        assert pipeline.named_steps["step1"].add_constant == 5.0


class TestFeatureUnion:
    """Tests for FeatureUnion composition behavior."""

    def test_observation_horizon_max(self, dummy_transformers, time_series_factory):
        """Test observation_horizon is max of component horizons."""
        X = time_series_factory(length=50)

        trans1 = SimpleTransformer(observation_horizon=1, add_constant=10.0)
        trans2 = SimpleTransformer(observation_horizon=2, add_constant=20.0)

        union = FeatureUnion([("trans1", trans1), ("trans2", trans2)])
        union.fit(X)

        expected_horizon = 2
        assert union.observation_horizon == expected_horizon, (
            f"Expected horizon {expected_horizon}, got {union.observation_horizon}"
        )

    def test_horizontal_concat(self, dummy_transformers, time_series_factory):
        """Test outputs are concatenated horizontally with prefixes."""
        X = time_series_factory(length=50, n_components=1)

        trans1 = SimpleTransformer(observation_horizon=1, add_constant=10.0)
        trans2 = SimpleTransformer(observation_horizon=1, add_constant=20.0)

        union = FeatureUnion([("trans1", trans1), ("trans2", trans2)])
        union.fit(X)

        X_trans = union.transform(X)

        assert "time" in X_trans.columns
        feature_cols = [col for col in X_trans.columns if col != "time"]
        assert len(feature_cols) == 2, (
            f"Expected 2 feature columns with prefixes, got {len(feature_cols)}: {feature_cols}"
        )


class TestColumnTransformer:
    """Tests for ColumnTransformer column-specific application."""

    def test_column_selection(self, time_series_factory):
        """Test transformers apply to specific columns."""
        X = time_series_factory(length=50, n_components=3)

        trans1 = SimpleTransformer(observation_horizon=0, add_constant=10.0)
        trans2 = SimpleTransformer(observation_horizon=0, add_constant=20.0)

        ct = ColumnTransformer(
            [("trans1", trans1, ["feature_0"]), ("trans2", trans2, ["feature_1"])],
            remainder="passthrough",
        )
        ct.fit(X)
        X_trans = ct.transform(X)

        assert "time" in X_trans.columns
        feature_cols = [col for col in X_trans.columns if col != "time"]
        assert len(feature_cols) >= 2, f"Expected at least 2 feature columns, got {len(feature_cols)}: {feature_cols}"
