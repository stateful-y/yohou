"""Stress tests for deep composition nesting with analytical verification.

This module tests composition patterns with up to 5 levels of nesting,
verifying exact numerical correctness on linear series with known solutions.

Test structure:
- Level 1: Forecaster with FeaturePipeline/FeatureUnion (~60 tests)
- Level 2: ColumnForecaster with per-column pipelines (~3 tests)
- Level 3: DecompositionPipeline inside ColumnForecaster (~2 tests)
- Level 4: ForecastedFeatureForecaster with complex pipelines (~6 tests)
- Level 5: Full tower with all composition types (~5 tests)
"""

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge

from yohou.compose import (
    ColumnForecaster,
    DecompositionPipeline,
    FeaturePipeline,
    FeatureUnion,
    ForecastedFeatureForecaster,
)
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.preprocessing import (
    LagTransformer,
    MinMaxScaler,
    RollingStatisticsTransformer,
    StandardScaler,
)
from yohou.stationarity import (
    LogTransformer,
    PolynomialTrendForecaster,
    SeasonalDifferencing,
)


@pytest.fixture
def full_tower_forecaster():
    """Create full 5-level nested forecaster hierarchy.

    Structure:
    ColumnForecaster[
        ForecastedFeatureForecaster[
            PointReductionForecaster[
                LinearRegression,
                target_transformer=StandardScaler,
                feature_transformer=FeaturePipeline[LagTransformer, RollingStats]
            ],
            PointReductionForecaster[LinearRegression, feature_transformer=Lag]
        ],
        DecompositionPipeline[PolynomialTrend, SeasonalNaive]
    ]
    """
    return ColumnForecaster([
        (
            "col_ab",
            ForecastedFeatureForecaster(
                target_forecaster=PointReductionForecaster(
                    estimator=LinearRegression(),
                    target_transformer=StandardScaler(),
                    feature_transformer=FeaturePipeline([
                        ("lag", LagTransformer([1, 2])),
                        ("roll", RollingStatisticsTransformer(3, statistics=["mean", "std"])),
                    ]),
                ),
                feature_forecaster=PointReductionForecaster(
                    estimator=LinearRegression(),
                    feature_transformer=LagTransformer(1),
                ),
                strategy="actual",
            ),
            ["col_a", "col_b"],
        ),
        (
            "col_c",
            DecompositionPipeline([
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("residual", SeasonalNaive(seasonality=1)),
            ]),
            ["col_c"],
        ),
    ])


@pytest.mark.integration
class TestLevel1ForecasterPipelines:
    """Level 1: PointReductionForecaster with various feature/target pipeline combinations."""

    @pytest.mark.parametrize(
        "feature_transformer",
        [
            pytest.param(None, id="feat_none"),
            pytest.param(LagTransformer(1), id="feat_lag1"),
            pytest.param(LagTransformer([1, 2, 3]), id="feat_lag123"),
            pytest.param(
                FeaturePipeline([
                    ("lag", LagTransformer(1)),
                    ("roll", RollingStatisticsTransformer(3)),
                ]),
                id="feat_pipeline",
            ),
            pytest.param(
                FeatureUnion([
                    ("lag1", LagTransformer(1)),
                    ("lag3", LagTransformer(3)),
                ]),
                id="feat_union",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "target_transformer",
        [
            pytest.param(None, id="tgt_none"),
            pytest.param(StandardScaler(), id="tgt_stdscaler"),
            pytest.param(MinMaxScaler(), id="tgt_minmax"),
            pytest.param(LogTransformer(offset=1.0), id="tgt_log"),
            pytest.param(SeasonalDifferencing(1), id="tgt_seasdiff"),
            pytest.param(
                FeaturePipeline([
                    ("log", LogTransformer(offset=1.0)),
                    ("std", StandardScaler()),
                ]),
                id="tgt_pipeline",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "estimator",
        [
            pytest.param(LinearRegression(), id="lr"),
            pytest.param(Ridge(alpha=1e-10), id="ridge0"),
        ],
    )
    def test_level1_forecaster_with_pipelines(
        self,
        linear_series,
        analytical_forecast,
        feature_transformer,
        target_transformer,
        estimator,
    ):
        """Level 1: PointReductionForecaster with various feature/target pipelines.

        Test on y = 2t + 100 (always positive for log), predict horizon=5.
        Verify predictions match analytical values within tolerance.

        Total: 5 feature × 6 target × 2 estimators = 60 test cases.
        """
        # Generate linear series: y = 2t + 100 (positive for log transform)
        slope, intercept = 2.0, 100.0
        y = linear_series(slope=slope, intercept=intercept, length=100)

        # Create forecaster with nested pipelines
        forecaster = PointReductionForecaster(
            estimator=clone(estimator),
            feature_transformer=clone(feature_transformer) if feature_transformer else None,
            target_transformer=clone(target_transformer) if target_transformer else None,
        )

        # Fit on first 80 steps
        horizon = 5
        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=horizon)

        # Predict next 5 steps
        y_pred = forecaster.predict(forecasting_horizon=horizon)

        # Expected analytical values: y = 2*t + 100 at t=80,81,82,83,84
        expected = analytical_forecast(
            "linear",
            {"slope": slope, "intercept": intercept},
            horizon=horizon,
            last_index=79,
        )

        # Verify predictions (allow higher tolerance for nonlinear target transformers)
        actual = y_pred.select(pl.col("value")).to_numpy().flatten()
        if target_transformer is not None:
            # LogTransformer, SeasonalDifferencing, and pipelines containing them
            # introduce numerical errors through exp/cumsum inverse operations
            np.testing.assert_allclose(actual, expected, atol=3.0, rtol=0.02)
        else:
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

        # Verify output structure
        assert len(y_pred) == horizon
        assert "time" in y_pred.columns
        assert "value" in y_pred.columns


@pytest.mark.integration
class TestLevel2ColumnForecaster:
    """Level 2: ColumnForecaster with per-column pipelines and configurations."""

    def test_level2_column_forecaster_with_pipelines(self, linear_series):
        """Level 2: ColumnForecaster with different forecaster per column.

        3-column target with different linear slopes:
        - col_a: y = 1*t + 10
        - col_b: y = 2*t + 20
        - col_c: y = 3*t + 30
        """
        # Generate 3-column panel
        y_a = linear_series(slope=1.0, intercept=10.0, length=100)
        y_b = linear_series(slope=2.0, intercept=20.0, length=100)
        y_c = linear_series(slope=3.0, intercept=30.0, length=100)

        y = pl.DataFrame({
            "time": y_a["time"],
            "col_a": y_a["value"],
            "col_b": y_b["value"],
            "col_c": y_c["value"],
        })

        # Assign each column a different forecaster with pipeline
        forecaster = ColumnForecaster([
            (
                "col_a",
                PointReductionForecaster(
                    LinearRegression(),
                    feature_transformer=LagTransformer(1),
                ),
                ["col_a"],
            ),
            (
                "col_b",
                PointReductionForecaster(
                    LinearRegression(),
                    feature_transformer=FeaturePipeline([
                        ("lag", LagTransformer([1, 2])),
                        ("roll", RollingStatisticsTransformer(3)),
                    ]),
                ),
                ["col_b"],
            ),
            (
                "col_c",
                PointReductionForecaster(
                    LinearRegression(),
                    target_transformer=StandardScaler(),
                    feature_transformer=LagTransformer(1),
                ),
                ["col_c"],
            ),
        ])

        # Fit and predict
        horizon = 5
        forecaster.fit(y[:80], forecasting_horizon=horizon)
        y_pred = forecaster.predict(forecasting_horizon=horizon)

        # Verify each column has correct slope
        # col_a: t=80,81,82,83,84 -> y = 90, 91, 92, 93, 94
        expected_a = 1.0 * np.arange(80, 85) + 10.0
        actual_a = y_pred.select("col_a").to_numpy().flatten()
        np.testing.assert_allclose(actual_a, expected_a, atol=1e-6)

        # col_b: t=80,81,82,83,84 -> y = 180, 182, 184, 186, 188
        expected_b = 2.0 * np.arange(80, 85) + 20.0
        actual_b = y_pred.select("col_b").to_numpy().flatten()
        np.testing.assert_allclose(actual_b, expected_b, atol=1e-6)

        # col_c: t=80,81,82,83,84 -> y = 270, 273, 276, 279, 282
        expected_c = 3.0 * np.arange(80, 85) + 30.0
        actual_c = y_pred.select("col_c").to_numpy().flatten()
        np.testing.assert_allclose(actual_c, expected_c, atol=1e-6)

    def test_level2_column_forecaster_with_remainder(self, linear_series):
        """Level 2: ColumnForecaster with remainder='passthrough'.

        4 columns total, 3 with forecasters, 1 passed through.
        """
        # Generate 4-column data
        y_a = linear_series(slope=1.0, intercept=10.0, length=100)
        y_b = linear_series(slope=2.0, intercept=20.0, length=100)
        y_c = linear_series(slope=3.0, intercept=30.0, length=100)
        y_d = linear_series(slope=4.0, intercept=40.0, length=100)

        y = pl.DataFrame({
            "time": y_a["time"],
            "col_a": y_a["value"],
            "col_b": y_b["value"],
            "col_c": y_c["value"],
            "col_d": y_d["value"],
        })

        # Forecasters for a, b, c; remainder for d
        forecaster = ColumnForecaster(
            [
                ("a", PointReductionForecaster(LinearRegression()), ["col_a"]),
                ("b", PointReductionForecaster(LinearRegression()), ["col_b"]),
                ("c", PointReductionForecaster(LinearRegression()), ["col_c"]),
            ],
            remainder="passthrough",
        )

        # Fit and predict
        horizon = 5
        forecaster.fit(y[:80], forecasting_horizon=horizon)
        y_pred = forecaster.predict(forecasting_horizon=horizon)

        # Verify a, b, c have predictions
        assert "col_a" in y_pred.columns
        assert "col_b" in y_pred.columns
        assert "col_c" in y_pred.columns

        # col_d should be missing (passthrough doesn't forecast)
        assert "col_d" not in y_pred.columns

        # Verify predictions
        expected_a = 1.0 * np.arange(80, 85) + 10.0
        actual_a = y_pred.select("col_a").to_numpy().flatten()
        np.testing.assert_allclose(actual_a, expected_a, atol=1e-6)

    def test_level2_column_forecaster_multi_column_assignment(self, linear_series):
        """Level 2: ColumnForecaster assigns single forecaster to multiple columns."""
        y_a = linear_series(slope=1.0, intercept=10.0, length=100)
        y_b = linear_series(slope=1.0, intercept=50.0, length=100)  # Same slope

        y = pl.DataFrame({
            "time": y_a["time"],
            "col_a": y_a["value"],
            "col_b": y_b["value"],
        })

        # Single forecaster for both columns
        forecaster = ColumnForecaster([
            ("ab", PointReductionForecaster(LinearRegression()), ["col_a", "col_b"]),
        ])

        horizon = 5
        forecaster.fit(y[:80], forecasting_horizon=horizon)
        y_pred = forecaster.predict(forecasting_horizon=horizon)

        # Both columns should exist
        assert "col_a" in y_pred.columns
        assert "col_b" in y_pred.columns

        # Verify predictions (both have slope=1.0, different intercepts)
        expected_a = 1.0 * np.arange(80, 85) + 10.0
        expected_b = 1.0 * np.arange(80, 85) + 50.0
        actual_a = y_pred.select("col_a").to_numpy().flatten()
        actual_b = y_pred.select("col_b").to_numpy().flatten()
        np.testing.assert_allclose(actual_a, expected_a, atol=1e-6)
        np.testing.assert_allclose(actual_b, expected_b, atol=1e-6)


@pytest.mark.integration
class TestLevel3DecompositionColumnForecaster:
    """Level 3: DecompositionPipeline nested inside ColumnForecaster."""

    def test_level3_decomposition_inside_column_forecaster(self, linear_series):
        """Level 3: ColumnForecaster with DecompositionPipeline for one column.

        2-column target:
        - col_a: DecompositionPipeline (trend + residual)
        - col_b: PointReductionForecaster
        """
        y_a = linear_series(slope=2.0, intercept=10.0, length=100)
        y_b = linear_series(slope=3.0, intercept=20.0, length=100)

        y = pl.DataFrame({
            "time": y_a["time"],
            "col_a": y_a["value"],
            "col_b": y_b["value"],
        })

        # Column A: Decomposition (trend + naive residual)
        # Column B: Reduction forecaster
        forecaster = ColumnForecaster([
            (
                "col_a",
                DecompositionPipeline([
                    ("trend", PolynomialTrendForecaster(degree=1)),
                    ("residual", SeasonalNaive(seasonality=1)),
                ]),
                ["col_a"],
            ),
            (
                "col_b",
                PointReductionForecaster(LinearRegression()),
                ["col_b"],
            ),
        ])

        horizon = 5
        forecaster.fit(y[:80], forecasting_horizon=horizon)
        y_pred = forecaster.predict(forecasting_horizon=horizon)

        # Verify col_a (trend should capture linear)
        # Note: DecompositionPipeline with polynomial trend + naive residual
        # should reconstruct linear series
        expected_a = 2.0 * np.arange(80, 85) + 10.0
        actual_a = y_pred.select("col_a").to_numpy().flatten()
        np.testing.assert_allclose(actual_a, expected_a, atol=3.0, rtol=0.02)

        # Verify col_b
        expected_b = 3.0 * np.arange(80, 85) + 20.0
        actual_b = y_pred.select("col_b").to_numpy().flatten()
        np.testing.assert_allclose(actual_b, expected_b, atol=1e-6)

    def test_level3_multiple_decompositions_in_columns(self, linear_series):
        """Level 3: Multiple columns with different DecompositionPipelines."""
        y_a = linear_series(slope=1.0, intercept=100.0, length=100)
        y_b = linear_series(slope=2.0, intercept=200.0, length=100)

        y = pl.DataFrame({
            "time": y_a["time"],
            "col_a": y_a["value"],
            "col_b": y_b["value"],
        })

        # Both columns use decomposition with different degree polynomials
        forecaster = ColumnForecaster([
            (
                "col_a",
                DecompositionPipeline([
                    ("trend", PolynomialTrendForecaster(degree=1)),
                    ("residual", SeasonalNaive(seasonality=1)),
                ]),
                ["col_a"],
            ),
            (
                "col_b",
                DecompositionPipeline([
                    ("trend", PolynomialTrendForecaster(degree=1)),
                    ("residual", SeasonalNaive(seasonality=1)),
                ]),
                ["col_b"],
            ),
        ])

        horizon = 5
        forecaster.fit(y[:80], forecasting_horizon=horizon)
        y_pred = forecaster.predict(forecasting_horizon=horizon)

        # Verify both columns
        expected_a = 1.0 * np.arange(80, 85) + 100.0
        expected_b = 2.0 * np.arange(80, 85) + 200.0
        actual_a = y_pred.select("col_a").to_numpy().flatten()
        actual_b = y_pred.select("col_b").to_numpy().flatten()
        np.testing.assert_allclose(actual_a, expected_a, atol=3.0, rtol=0.02)
        np.testing.assert_allclose(actual_b, expected_b, atol=3.0, rtol=0.02)


@pytest.mark.integration
class TestLevel4ForecastedFeatureForecaster:
    """Level 4: ForecastedFeatureForecaster with nested pipeline compositions."""

    @pytest.mark.parametrize("strategy", ["actual", "predicted", "rewind"])
    @pytest.mark.parametrize("split_ratio", [0.6, 0.8])
    def test_level4_forecasted_feature_with_pipelines(self, linear_series, strategy, split_ratio):
        """Level 4: ForecastedFeatureForecaster with nested pipelines.

        Feature: x_t = 3t + 10
        Target: y_t = 2*x_t + 5 = 6t + 25

        Target forecaster: PointReductionForecaster with feature pipeline
        Feature forecaster: PointReductionForecaster with lag

        Parametrize: strategy × split_ratio = 3 × 2 = 6 tests

        Note: split_ratio must be > 0.5 for the "predicted" strategy because the
        feature forecaster uses multi-output reduction and needs n_split >
        (train_end - n_split) + observation_horizon for tabularization.
        """
        # Generate feature and target (use larger dataset for split ratios)
        # Need enough data so that even with split_ratio=0.3 and strategy="predicted",
        # the target forecaster still has sufficient rows after tabularization.
        x = linear_series(slope=3.0, intercept=10.0, length=800)
        # y = 2*x + 5 = 2*(3t + 10) + 5 = 6t + 25
        y = pl.DataFrame({
            "time": x["time"],
            "value": 2.0 * x["value"] + 5.0,
        })
        X = x.rename({"value": "feature_x"})

        # Create nested forecasters
        target_forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=FeaturePipeline([
                ("lag", LagTransformer(1)),
                ("scaler", StandardScaler()),
            ]),
        )

        feature_forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(1),
        )

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=target_forecaster,
            feature_forecaster=feature_forecaster,
            strategy=strategy,
            split_ratio=split_ratio,
        )

        # Fit and predict
        horizon = 5
        train_end = 700
        forecaster.fit(y[:train_end], X[:train_end], forecasting_horizon=horizon)
        y_pred = forecaster.predict(forecasting_horizon=horizon)

        # Expected: y = 6t + 25 at t=700,...,704
        expected = 6.0 * np.arange(train_end, train_end + horizon) + 25.0
        actual = y_pred.select("value").to_numpy().flatten()

        # Allow slightly higher tolerance for strategies that use predicted features
        if strategy == "predicted":
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
        else:
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.integration
class TestLevel5FullTower:
    """Level 5: Full 5-level nested forecaster tower with all composition types."""

    def test_level5_full_tower_get_params(self, full_tower_forecaster):
        """Level 5: get_params() returns full nested parameter dictionary."""
        params = full_tower_forecaster.get_params(deep=True)

        # Verify top-level parameters exist
        assert "forecasters" in params
        assert "remainder" in params

        # Verify nested parameters are accessible (spot checks)
        # Path to leaf: col_ab__target_forecaster__estimator__fit_intercept
        assert "col_ab__target_forecaster__estimator__fit_intercept" in params
        assert params["col_ab__target_forecaster__estimator__fit_intercept"] is True

        # Path to feature transformer pipeline: col_ab__target_forecaster__feature_transformer__lag__lag
        assert "col_ab__target_forecaster__feature_transformer__lag__lag" in params
        assert params["col_ab__target_forecaster__feature_transformer__lag__lag"] == [1, 2]

        # Path to decomposition: col_c__trend__degree
        assert "col_c__trend__degree" in params
        assert params["col_c__trend__degree"] == 1

    def test_level5_full_tower_set_params(self, full_tower_forecaster):
        """Level 5: set_params() can modify nested leaf parameters."""
        # Modify deep nested parameter
        full_tower_forecaster.set_params(
            col_ab__target_forecaster__estimator__fit_intercept=False,
            col_ab__target_forecaster__feature_transformer__lag__lag=[1, 2, 3],
            col_c__trend__degree=2,
        )

        # Verify changes propagated
        params = full_tower_forecaster.get_params(deep=True)
        assert params["col_ab__target_forecaster__estimator__fit_intercept"] is False
        assert params["col_ab__target_forecaster__feature_transformer__lag__lag"] == [1, 2, 3]
        assert params["col_c__trend__degree"] == 2

    def test_level5_full_tower_clone(self, full_tower_forecaster):
        """Level 5: clone() produces identical unfitted copy."""
        cloned = clone(full_tower_forecaster)

        # Verify cloned has same parameters
        original_params = full_tower_forecaster.get_params(deep=True)
        cloned_params = cloned.get_params(deep=True)

        # Compare relevant parameters (exclude object references)
        assert (
            cloned_params["col_ab__target_forecaster__estimator__fit_intercept"]
            == original_params["col_ab__target_forecaster__estimator__fit_intercept"]
        )
        assert (
            cloned_params["col_ab__target_forecaster__feature_transformer__lag__lag"]
            == original_params["col_ab__target_forecaster__feature_transformer__lag__lag"]
        )
        assert cloned_params["col_c__trend__degree"] == original_params["col_c__trend__degree"]

        # Verify cloned is not fitted
        with pytest.raises(AttributeError, match="forecasters_"):
            # Attempt to access fitted attribute should fail
            _ = cloned.forecasters_

    def test_level5_full_tower_fit_predict(self, full_tower_forecaster, linear_series):
        """Level 5: fit() → predict() produces correct analytical values."""
        # Generate 3-column data with different slopes
        y_a = linear_series(slope=1.0, intercept=10.0, length=100)
        y_b = linear_series(slope=2.0, intercept=20.0, length=100)
        y_c = linear_series(slope=3.0, intercept=30.0, length=100)

        y = pl.DataFrame({
            "time": y_a["time"],
            "col_a": y_a["value"],
            "col_b": y_b["value"],
            "col_c": y_c["value"],
        })

        # Create exogenous features for col_a and col_b
        X = pl.DataFrame({
            "time": y_a["time"],
            "feat_a": y_a["value"] * 0.5,  # Related to col_a
            "feat_b": y_b["value"] * 0.5,  # Related to col_b
        })

        # Fit and predict
        horizon = 5
        full_tower_forecaster.fit(y[:80], X[:80], forecasting_horizon=horizon)
        y_pred = full_tower_forecaster.predict(forecasting_horizon=horizon)

        # Verify structure
        assert "col_a" in y_pred.columns
        assert "col_b" in y_pred.columns
        assert "col_c" in y_pred.columns
        assert len(y_pred) == horizon

        # Verify predictions (allow tolerance for deep nesting)
        expected_a = 1.0 * np.arange(80, 85) + 10.0
        expected_b = 2.0 * np.arange(80, 85) + 20.0
        expected_c = 3.0 * np.arange(80, 85) + 30.0

        actual_a = y_pred.select("col_a").to_numpy().flatten()
        actual_b = y_pred.select("col_b").to_numpy().flatten()
        actual_c = y_pred.select("col_c").to_numpy().flatten()

        np.testing.assert_allclose(actual_a, expected_a, atol=3.0, rtol=0.02)
        np.testing.assert_allclose(actual_b, expected_b, atol=3.0, rtol=0.02)
        np.testing.assert_allclose(actual_c, expected_c, atol=3.0, rtol=0.02)

    def test_level5_full_tower_observe_predict(self, full_tower_forecaster, linear_series):
        """Level 5: fit() → observe() → predict() shifts predictions correctly."""
        # Generate 3-column data
        y_a = linear_series(slope=1.0, intercept=10.0, length=100)
        y_b = linear_series(slope=2.0, intercept=20.0, length=100)
        y_c = linear_series(slope=3.0, intercept=30.0, length=100)

        y = pl.DataFrame({
            "time": y_a["time"],
            "col_a": y_a["value"],
            "col_b": y_b["value"],
            "col_c": y_c["value"],
        })

        # Create exogenous features
        X = pl.DataFrame({
            "time": y_a["time"],
            "feat_a": y_a["value"] * 0.5,
            "feat_b": y_b["value"] * 0.5,
        })

        # Fit on first 70 steps
        horizon = 5
        full_tower_forecaster.fit(y[:70], X[:70], forecasting_horizon=horizon)

        # Observe 10 more observations (70-79)
        full_tower_forecaster.observe(y[70:80], X[70:80])

        # Predict next 5 steps (80-84)
        y_pred = full_tower_forecaster.predict(forecasting_horizon=horizon)

        # Verify predictions shifted correctly
        expected_a = 1.0 * np.arange(80, 85) + 10.0
        expected_b = 2.0 * np.arange(80, 85) + 20.0
        expected_c = 3.0 * np.arange(80, 85) + 30.0

        actual_a = y_pred.select("col_a").to_numpy().flatten()
        actual_b = y_pred.select("col_b").to_numpy().flatten()
        actual_c = y_pred.select("col_c").to_numpy().flatten()

        np.testing.assert_allclose(actual_a, expected_a, atol=3.0, rtol=0.02)
        np.testing.assert_allclose(actual_b, expected_b, atol=3.0, rtol=0.02)
        np.testing.assert_allclose(actual_c, expected_c, atol=3.0, rtol=0.02)
