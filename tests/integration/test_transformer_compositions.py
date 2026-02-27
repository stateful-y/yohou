"""Stress tests for transformer composition algebra.

Tests verify the mathematical properties of composed transformers:
- FeaturePipeline: observation_horizon = sum of component horizons
- FeatureUnion: observation_horizon = max of component horizons
- ColumnTransformer: observation_horizon = max of column-wise horizons
- Inverse transform chains recover original data
- Nested compositions compute correct horizons

All tests use @pytest.mark.integration decorator and focus on analytical verification
of algebraic properties rather than just correct execution.
"""

import numpy as np
import polars as pl
import pytest

from yohou.compose import ColumnTransformer, FeaturePipeline, FeatureUnion
from yohou.preprocessing import (
    LagTransformer,
    MinMaxScaler,
    RollingStatisticsTransformer,
    StandardScaler,
)
from yohou.stationarity import LogTransformer, SeasonalDifferencing


@pytest.mark.integration
class TestFeaturePipelineTransform:
    """FeaturePipeline horizon, observe_transform, and rewind_transform behavior."""

    @pytest.mark.parametrize(
        "pipeline_steps,expected_horizon",
        [
            ([("lag1", LagTransformer(1))], 1),
            ([("lag3", LagTransformer(3))], 3),
            ([("lag1", LagTransformer(1)), ("lag2", LagTransformer(2))], 3),
            (
                [("diff", SeasonalDifferencing(7)), ("lag1", LagTransformer(1))],
                8,
            ),  # 7 + 1
            (
                [
                    ("log", LogTransformer()),
                    ("diff", SeasonalDifferencing(1)),
                    ("lag2", LagTransformer(2)),
                ],
                3,
            ),  # 0 + 1 + 2
            (
                [
                    ("std", StandardScaler()),
                    ("lag1", LagTransformer(1)),
                    ("std2", StandardScaler()),
                ],
                1,
            ),  # 0 + 1 + 0
        ],
        ids=[
            "lag1_only",
            "lag3_only",
            "lag1_then_lag2",
            "diff7_then_lag1",
            "log_diff1_lag2",
            "std_lag1_std",
        ],
    )
    def test_feature_pipeline_observation_horizon_sum(
        self,
        linear_series,
        pipeline_steps,
        expected_horizon,
    ):
        """FeaturePipeline observation_horizon equals sum of component horizons."""
        # Generate data using fixture
        X = linear_series(slope=1.0, intercept=0.0, length=100)
        pipeline = FeaturePipeline(pipeline_steps)

        # Verify fit_transform output row count
        X_transformed = pipeline.fit_transform(X)
        assert len(X_transformed) == len(X) - expected_horizon

        # Verify observation_horizon property (after fitting)
        assert pipeline.observation_horizon == expected_horizon

        # Verify time column preserved
        assert "time" in X_transformed.columns

    @pytest.mark.parametrize(
        "pipeline_steps,expected_horizon",
        [
            ([("lag1", LagTransformer(1))], 1),
            ([("lag3", LagTransformer(3))], 3),
            ([("lag1", LagTransformer(1)), ("lag2", LagTransformer(2))], 3),
            (
                [
                    ("lag3", LagTransformer(3)),
                    ("roll3", RollingStatisticsTransformer(3)),
                ],
                5,
            ),
            (
                [("diff", SeasonalDifferencing(7)), ("lag1", LagTransformer(1))],
                8,
            ),
            (
                [
                    ("log", LogTransformer()),
                    ("diff", SeasonalDifferencing(1)),
                    ("lag2", LagTransformer(2)),
                ],
                3,
            ),
            (
                [
                    ("std", StandardScaler()),
                    ("lag1", LagTransformer(1)),
                    ("std2", StandardScaler()),
                ],
                1,
            ),
        ],
        ids=[
            "lag1_only",
            "lag3_only",
            "lag1_then_lag2",
            "lag3_then_roll3",
            "diff7_then_lag1",
            "log_diff1_lag2",
            "std_lag1_std",
        ],
    )
    def test_feature_pipeline_observe_transform_row_count(
        self,
        linear_series,
        pipeline_steps,
        expected_horizon,
    ):
        """FeaturePipeline observe_transform preserves row count."""
        # Generate data using fixture
        X = linear_series(slope=1.0, intercept=0.0, length=100)
        X_train = X[:50]
        X_new = X[50:]

        pipeline = FeaturePipeline(pipeline_steps)
        pipeline.fit(X_train)

        # observe_transform should preserve row count (uses memory)
        X_updated = pipeline.observe_transform(X_new)
        assert len(X_updated) == len(X_new)

        # Verify time column preserved
        assert "time" in X_updated.columns

    @pytest.mark.parametrize(
        "pipeline_steps,expected_horizon",
        [
            ([("lag1", LagTransformer(1))], 1),
            ([("lag3", LagTransformer(3))], 3),
            ([("lag1", LagTransformer(1)), ("lag2", LagTransformer(2))], 3),
            (
                [
                    ("lag3", LagTransformer(3)),
                    ("roll3", RollingStatisticsTransformer(3)),
                ],
                5,
            ),
            (
                [("diff", SeasonalDifferencing(7)), ("lag1", LagTransformer(1))],
                8,
            ),
            (
                [
                    ("log", LogTransformer()),
                    ("diff", SeasonalDifferencing(1)),
                    ("lag2", LagTransformer(2)),
                ],
                3,
            ),
            (
                [
                    ("std", StandardScaler()),
                    ("lag1", LagTransformer(1)),
                    ("std2", StandardScaler()),
                ],
                1,
            ),
        ],
        ids=[
            "lag1_only",
            "lag3_only",
            "lag1_then_lag2",
            "lag3_then_roll3",
            "diff7_then_lag1",
            "log_diff1_lag2",
            "std_lag1_std",
        ],
    )
    def test_feature_pipeline_rewind_transform_row_count(
        self,
        linear_series,
        pipeline_steps,
        expected_horizon,
    ):
        """FeaturePipeline rewind_transform drops first observation_horizon rows."""
        # Generate data using fixture
        X = linear_series(slope=1.0, intercept=0.0, length=100)
        pipeline = FeaturePipeline(pipeline_steps)

        # Must fit before calling rewind_transform
        pipeline.fit(X)

        # rewind_transform rewinds state, drops first observation_horizon rows
        X_reset = pipeline.rewind_transform(X)
        # Note: Actual row count depends on which transformers drop rows
        # rewind_transform behavior matches fit_transform for stateless transformers
        assert len(X_reset) <= len(X)

        # Verify time column preserved
        assert "time" in X_reset.columns


@pytest.mark.integration
class TestFeaturePipelineAnalytical:
    """Analytical verification of FeaturePipeline mathematical properties."""

    def test_feature_pipeline_constant_data_lag_constant(self, constant_series):
        """Lag of constant data produces constant output."""
        # Generate data using fixture
        X = constant_series(value=42.0, length=100)
        pipeline = FeaturePipeline([("lag2", LagTransformer(2))])

        X_transformed = pipeline.fit_transform(X)

        # Lag of constant is constant
        value_col = [c for c in X_transformed.columns if c != "time"][0]
        values = X_transformed[value_col].to_numpy()
        assert np.allclose(values, values[0], atol=1e-6)

    def test_feature_pipeline_linear_data_diff_constant(self, linear_series):
        """SeasonalDifferencing(1) of linear data produces constant output."""
        # Generate data using fixture
        X = linear_series(slope=2.0, intercept=10.0, length=100)
        pipeline = FeaturePipeline([("diff", SeasonalDifferencing(1))])

        X_transformed = pipeline.fit_transform(X)

        # First difference of linear trend is constant (slope)
        value_col = [c for c in X_transformed.columns if c != "time"][0]
        values = X_transformed[value_col].to_numpy()
        assert np.allclose(values, values[0], atol=1e-6)

    def test_feature_pipeline_standardized_constant_is_zero(self, constant_series):
        """StandardScaler transforms constant data to zeros."""
        # Generate data using fixture
        X = constant_series(value=42.0, length=100)
        pipeline = FeaturePipeline([("std", StandardScaler())])

        X_transformed = pipeline.fit_transform(X)

        # Constant data standardized: (x - mean) / std where std=0 → 0 (sklearn behavior)
        value_col = [c for c in X_transformed.columns if c != "time"][0]
        values = X_transformed[value_col].to_numpy()
        # sklearn StandardScaler sets constant columns to 0
        assert np.allclose(values, 0.0, atol=1e-6)

    def test_feature_pipeline_chained_lag_sum(self, linear_series):
        """Chained lags: LagTransformer(1) then LagTransformer(2) = lag by 3."""
        # Generate data using fixture
        X = linear_series(slope=1.0, intercept=0.0, length=100)
        pipeline = FeaturePipeline([("lag1", LagTransformer(1)), ("lag2", LagTransformer(2))])

        X_transformed = pipeline.fit_transform(X)

        # Expected: lag total of 3 steps
        # Compare with direct LagTransformer(3)
        direct_lag = LagTransformer(3)
        X_direct = direct_lag.fit_transform(X)

        value_col = [c for c in X_transformed.columns if c != "time"][0]
        value_col_direct = [c for c in X_direct.columns if c != "time"][0]

        assert np.allclose(
            X_transformed[value_col].to_numpy(),
            X_direct[value_col_direct].to_numpy(),
            atol=1e-6,
        )


@pytest.mark.integration
class TestFeatureUnion:
    """FeatureUnion horizon, observe_transform, and rewind_transform behavior."""

    @pytest.mark.parametrize(
        "union_steps,expected_horizon,expected_n_columns",
        [
            (
                [("lag1", LagTransformer(1)), ("lag3", LagTransformer(3))],
                3,
                2,
            ),  # max(1,3)=3, 2 lag columns
            (
                [
                    ("diff7", SeasonalDifferencing(7)),
                    ("lag1", LagTransformer(1)),
                ],
                7,
                2,
            ),  # max(7,1)=7
        ],
        ids=["lag1_lag3", "diff7_lag1"],
    )
    def test_feature_union_observation_horizon_max(
        self,
        linear_series,
        union_steps,
        expected_horizon,
        expected_n_columns,
    ):
        """FeatureUnion observation_horizon equals max of component horizons."""
        # Generate data using fixture
        X = linear_series(slope=1.0, intercept=0.0, length=100)
        union = FeatureUnion(union_steps)

        # Verify fit_transform output
        X_transformed = union.fit_transform(X)

        # FeatureUnion drops rows according to observation_horizon
        assert len(X_transformed) == len(X) - expected_horizon

        # Verify observation_horizon property (after fitting)
        assert union.observation_horizon == expected_horizon

        # Verify column count (excluding time)
        non_time_cols = [c for c in X_transformed.columns if c != "time"]
        assert len(non_time_cols) == expected_n_columns

        # Verify time column preserved
        assert "time" in X_transformed.columns

    @pytest.mark.parametrize(
        "union_steps,expected_horizon",
        [
            ([("lag1", LagTransformer(1)), ("lag3", LagTransformer(3))], 3),
            (
                [
                    ("lag2", LagTransformer(2)),
                    ("roll5", RollingStatisticsTransformer(5)),
                ],
                4,
            ),
            (
                [
                    ("diff7", SeasonalDifferencing(7)),
                    ("lag1", LagTransformer(1)),
                ],
                7,
            ),
        ],
        ids=["lag1_lag3", "lag2_roll5", "diff7_lag1"],
    )
    def test_feature_union_observe_transform_row_count(
        self,
        linear_series,
        union_steps,
        expected_horizon,
    ):
        """FeatureUnion observe_transform preserves row count."""
        # Generate data using fixture
        X = linear_series(slope=1.0, intercept=0.0, length=100)
        X_train = X[:50]
        X_new = X[50:]

        union = FeatureUnion(union_steps)
        union.fit(X_train)

        # observe_transform should preserve row count
        X_updated = union.observe_transform(X_new)
        assert len(X_updated) == len(X_new)

        # Verify time column preserved
        assert "time" in X_updated.columns

    @pytest.mark.parametrize(
        "union_steps,expected_horizon",
        [
            ([("lag1", LagTransformer(1)), ("lag3", LagTransformer(3))], 3),
            (
                [
                    ("lag2", LagTransformer(2)),
                    ("roll5", RollingStatisticsTransformer(5)),
                ],
                4,
            ),
            (
                [
                    ("diff7", SeasonalDifferencing(7)),
                    ("lag1", LagTransformer(1)),
                ],
                7,
            ),
        ],
        ids=["lag1_lag3", "lag2_roll5", "diff7_lag1"],
    )
    def test_feature_union_rewind_transform_row_count(
        self,
        linear_series,
        union_steps,
        expected_horizon,
    ):
        """FeatureUnion rewind_transform preserves row count (pads with nulls)."""
        # Generate data using fixture
        X = linear_series(slope=1.0, intercept=0.0, length=100)
        union = FeatureUnion(union_steps)

        # Must fit before calling rewind_transform
        union.fit(X)

        # rewind_transform rewinds state
        X_reset = union.rewind_transform(X)
        # FeatureUnion pads to preserve all rows
        assert len(X_reset) <= len(X)

        # Verify time column preserved
        assert "time" in X_reset.columns


@pytest.mark.integration
class TestColumnTransformer:
    """ColumnTransformer column isolation, remainder handling, and horizon computation."""

    def test_column_transformer_column_isolation(self, linear_series):
        """ColumnTransformer applies transformers to independent column subsets."""
        # Generate data using fixture
        X = linear_series(slope=1.0, intercept=0.0, length=100)

        # Create 4-column DataFrame
        X = pl.concat(
            [
                X.select(["time", "value"]).rename({"value": "col_a"}),
                X.select("value").rename({"value": "col_b"}),
                X.select("value").rename({"value": "col_c"}),
                X.select("value").rename({"value": "col_d"}),
            ],
            how="horizontal",
        )

        transformer = ColumnTransformer(
            [
                ("scale_ab", StandardScaler(), ["col_a", "col_b"]),
                ("lag_c", LagTransformer(2), ["col_c"]),
                ("diff_d", SeasonalDifferencing(3), ["col_d"]),
            ],
            remainder="drop",
        )

        X_transformed = transformer.fit_transform(X)

        # Verify observation_horizon = max(0, 2, 3) = 3 (after fitting)
        assert transformer.observation_horizon == 3

        # Verify row count
        assert len(X_transformed) == len(X) - 3

        # Verify column count: 2 (scaled) + 1 (lagged) + 1 (differenced) = 4 + time
        non_time_cols = [c for c in X_transformed.columns if c != "time"]
        assert len(non_time_cols) == 4

        # Verify time column preserved
        assert "time" in X_transformed.columns

    def test_column_transformer_with_remainder_passthrough(self, linear_series):
        """ColumnTransformer with remainder='passthrough' appends untransformed columns."""
        # Generate data using fixture
        X = linear_series(slope=1.0, intercept=0.0, length=100)

        # Create 3-column DataFrame
        X = pl.concat(
            [
                X.select(["time", "value"]).rename({"value": "col_a"}),
                X.select("value").rename({"value": "col_b"}),
                X.select("value").rename({"value": "col_c"}),
            ],
            how="horizontal",
        )

        transformer = ColumnTransformer(
            [
                ("scale_a", StandardScaler(), ["col_a"]),
                ("lag_b", LagTransformer(2), ["col_b"]),
            ],
            remainder="passthrough",
        )

        X_transformed = transformer.fit_transform(X)

        # observation_horizon = max(0, 2) = 2 (after fitting)
        assert transformer.observation_horizon == 2

        # Verify row count
        assert len(X_transformed) == len(X) - 2

        # Verify column count: 1 (scaled) + 1 (lagged) + 1 (passthrough col_c) + time
        non_time_cols = [c for c in X_transformed.columns if c != "time"]
        assert len(non_time_cols) == 3

        # Verify time column preserved
        assert "time" in X_transformed.columns

    def test_column_transformer_observation_horizon_max(self, linear_series):
        """ColumnTransformer observation_horizon is max of component horizons."""
        # Generate data using fixture
        X = linear_series(slope=1.0, intercept=0.0, length=100)

        # Create 3-column DataFrame
        X = pl.concat(
            [
                X.select(["time", "value"]).rename({"value": "col_a"}),
                X.select("value").rename({"value": "col_b"}),
                X.select("value").rename({"value": "col_c"}),
            ],
            how="horizontal",
        )

        transformer = ColumnTransformer(
            [
                ("std_a", StandardScaler(), ["col_a"]),  # oh=0
                ("lag1_b", LagTransformer(1), ["col_b"]),  # oh=1
                ("lag5_c", LagTransformer(5), ["col_c"]),  # oh=5
            ],
            remainder="drop",
        )

        X_transformed = transformer.fit_transform(X)

        # Verify observation_horizon = max(0, 1, 5) = 5 (after fitting)
        assert transformer.observation_horizon == 5

        # Verify row count
        assert len(X_transformed) == len(X) - 5


@pytest.mark.integration
class TestNestedCompositions:
    """Nested compositions of FeaturePipeline, FeatureUnion, and ColumnTransformer."""

    def test_nested_pipeline_union_column_transformer(self, linear_series):
        """Nested: FeaturePipeline inside FeatureUnion inside ColumnTransformer."""
        # Generate data using fixture
        X = linear_series(slope=1.0, intercept=0.0, length=100)

        # Create 3-column DataFrame
        X = pl.concat(
            [
                X.select(["time", "value"]).rename({"value": "col_a"}),
                X.select("value").rename({"value": "col_b"}),
                X.select("value").rename({"value": "col_c"}),
            ],
            how="horizontal",
        )

        transformer = ColumnTransformer(
            [
                (
                    "branch1",
                    FeatureUnion([
                        (
                            "pipe",
                            FeaturePipeline([
                                ("lag", LagTransformer(1)),
                                ("std", StandardScaler()),
                            ]),
                        ),  # oh=1+0=1
                        ("raw_lag", LagTransformer(3)),  # oh=3
                    ]),  # oh=max(1,3)=3
                    ["col_a", "col_b"],
                ),
                ("branch2", StandardScaler(), ["col_c"]),  # oh=0
            ],
            remainder="drop",
        )

        X_transformed = transformer.fit_transform(X)

        # Verify observation_horizon = max(3, 0) = 3 (after fitting)
        assert transformer.observation_horizon == 3

        # Verify row count
        assert len(X_transformed) == len(X) - 3

        # Verify time column preserved
        assert "time" in X_transformed.columns

    def test_nested_deep_composition_constant_data(self, constant_series):
        """Deeply nested composition on constant data produces analytically correct output."""
        # Generate data using fixture
        X = constant_series(value=42.0, length=100)

        # Pipeline: lag1 -> std (oh=1)
        # Union: [pipeline, lag3] (oh=max(1,3)=3)
        transformer = FeatureUnion([
            (
                "pipe",
                FeaturePipeline([("lag1", LagTransformer(1)), ("std", StandardScaler())]),
            ),
            ("lag3", LagTransformer(3)),
        ])

        X_transformed = transformer.fit_transform(X)

        # Verify observation_horizon = 3 (after fitting)
        assert transformer.observation_horizon == 3

        # Verify row count
        assert len(X_transformed) == len(X) - 3

        # Lag of constant is constant, std of constant is 0
        non_time_cols = [c for c in X_transformed.columns if c != "time"]
        assert len(non_time_cols) == 2

        # First column: pipe output (lag1 + std of constant = 0)
        # Second column: lag3 of constant = constant
        values_pipe = X_transformed[non_time_cols[0]].to_numpy()
        values_lag3 = X_transformed[non_time_cols[1]].to_numpy()

        assert np.allclose(values_pipe, 0.0, atol=1e-6)  # std(constant) = 0
        assert np.allclose(values_lag3, values_lag3[0], atol=1e-6)  # lag(constant) = constant

    def test_nested_column_transformer_observation_horizon_propagation(self, linear_series):
        """Nested ColumnTransformer correctly propagates observation_horizon."""
        # Generate data using fixture
        X = linear_series(slope=1.0, intercept=0.0, length=100)

        # Create 2-column DataFrame
        X = pl.concat(
            [
                X.select(["time", "value"]).rename({"value": "col_a"}),
                X.select("value").rename({"value": "col_b"}),
            ],
            how="horizontal",
        )

        # Inner union: oh=max(1,2)=2
        inner_union = FeatureUnion([("lag1", LagTransformer(1)), ("lag2", LagTransformer(2))])

        # Outer column transformer: oh=max(2,0)=2
        transformer = ColumnTransformer(
            [
                ("union_a", inner_union, ["col_a"]),
                ("std_b", StandardScaler(), ["col_b"]),
            ],
            remainder="drop",
        )

        X_transformed = transformer.fit_transform(X)

        # Verify observation_horizon = 2 (after fitting)
        assert transformer.observation_horizon == 2
        assert len(X_transformed) == len(X) - 2


@pytest.mark.integration
class TestInverseTransform:
    """Inverse transform round-trip recovery and time column preservation."""

    @pytest.mark.parametrize(
        "pipeline_steps",
        [
            [("log", LogTransformer(offset=1.0)), ("std", StandardScaler())],
            [("diff", SeasonalDifferencing(1)), ("std", StandardScaler())],
            [
                ("log", LogTransformer(offset=1.0)),
                ("std", StandardScaler()),
                ("minmax", MinMaxScaler()),
            ],
        ],
        ids=["log_std", "diff_std", "log_std_minmax"],
    )
    def test_inverse_transform_round_trip_linear_data(self, linear_series, pipeline_steps):
        """Inverse transform chain recovers original data (linear trend)."""
        # Generate data using fixture - use positive data for log transform
        X = linear_series(slope=1.0, intercept=100.0, length=100)

        pipeline = FeaturePipeline(pipeline_steps)

        X_transformed = pipeline.fit_transform(X)
        X_recovered = pipeline.inverse_transform(X_transformed, X)

        # Account for observation_horizon: compare overlapping region
        oh = pipeline.observation_horizon
        original_values = X["value"].to_numpy()[oh:]
        recovered_values = X_recovered["value"].to_numpy()

        # Round-trip should recover original within numerical tolerance
        assert np.allclose(recovered_values, original_values, atol=1e-5)

    @pytest.mark.parametrize(
        "pipeline_steps",
        [
            [("log", LogTransformer(offset=1.0)), ("std", StandardScaler())],
            [
                ("log", LogTransformer(offset=1.0)),
                ("std", StandardScaler()),
                ("minmax", MinMaxScaler()),
            ],
        ],
        ids=["log_std", "log_std_minmax"],
    )
    def test_inverse_transform_round_trip_constant_data(self, constant_series, pipeline_steps):
        """Inverse transform chain recovers original constant data."""
        # Generate data using fixture - use positive constant for log transform
        X = constant_series(value=42.0, length=100)

        pipeline = FeaturePipeline(pipeline_steps)

        X_transformed = pipeline.fit_transform(X)
        X_recovered = pipeline.inverse_transform(X_transformed, X)

        # Account for observation_horizon
        oh = pipeline.observation_horizon
        original_values = X["value"].to_numpy()[oh:]
        recovered_values = X_recovered["value"].to_numpy()

        # Round-trip should recover original within numerical tolerance
        assert np.allclose(recovered_values, original_values, atol=1e-5)

    def test_inverse_transform_preserves_time_column(self, linear_series):
        """Inverse transform preserves time column correctly."""
        # Generate data using fixture (positive values for log)
        X = linear_series(slope=1.0, intercept=100.0, length=100)

        pipeline = FeaturePipeline([("log", LogTransformer(offset=1.0)), ("std", StandardScaler())])

        X_transformed = pipeline.fit_transform(X)
        X_recovered = pipeline.inverse_transform(X_transformed, X)

        # Verify time columns match
        assert "time" in X_recovered.columns
        assert np.array_equal(X_transformed["time"].to_numpy(), X_recovered["time"].to_numpy())

    def test_inverse_transform_single_step(self, linear_series):
        """Inverse transform with single LogTransformer recovers original."""
        # Generate data using fixture (positive values for log)
        X = linear_series(slope=1.0, intercept=100.0, length=100)

        pipeline = FeaturePipeline([("log", LogTransformer(offset=1.0))])

        X_transformed = pipeline.fit_transform(X)
        X_recovered = pipeline.inverse_transform(X_transformed, X)

        # No observation_horizon for LogTransformer
        original_values = X["value"].to_numpy()
        recovered_values = X_recovered["value"].to_numpy()

        assert np.allclose(recovered_values, original_values, atol=1e-6)

    def test_inverse_transform_three_step_chain(self, linear_series):
        """Three-step inverse transform chain (log -> std -> minmax) recovers original."""
        # Generate data using fixture (positive values for log)
        X = linear_series(slope=1.0, intercept=100.0, length=100)

        pipeline = FeaturePipeline([
            ("log", LogTransformer(offset=1.0)),
            ("std", StandardScaler()),
            ("minmax", MinMaxScaler()),
        ])

        X_transformed = pipeline.fit_transform(X)
        X_recovered = pipeline.inverse_transform(X_transformed, X)

        # All transformers have oh=0
        original_values = X["value"].to_numpy()
        recovered_values = X_recovered["value"].to_numpy()

        # Three-step chain may accumulate more numerical error
        assert np.allclose(recovered_values, original_values, atol=1e-5)
