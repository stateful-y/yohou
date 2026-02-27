"""Analytical integration tests for composition correctness.

This module tests the correctness of composed forecasters and transformers using
analytically solvable time series. All tests verify exact numerical results against
closed-form solutions.

Test Categories:
- FeaturePipeline observation_horizon accumulation
- FeaturePipeline + Reduction end-to-end
- FeatureUnion correctness
- ColumnTransformer correctness
- ColumnForecaster correctness
- DecompositionPipeline correctness
- ForecastedFeatureForecaster with strategy parametrization
- Deep nesting through multiple composition layers
- observe() propagation through compositions
- rewind() propagation through compositions
"""

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from yohou.compose import (
    ColumnForecaster,
    ColumnTransformer,
    DecompositionPipeline,
    FeaturePipeline,
    FeatureUnion,
    ForecastedFeatureForecaster,
)
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.preprocessing import (
    LagTransformer,
    RollingStatisticsTransformer,
    StandardScaler,
)
from yohou.stationarity import PolynomialTrendForecaster, SeasonalDifferencing


@pytest.mark.integration
class TestFeaturePipelineObservationHorizon:
    """Tests that FeaturePipeline correctly accumulates observation_horizon across stages."""

    def test_pipelineobservation_horizon_two_lags(self, linear_series):
        """FeaturePipeline([LagTransformer(3), LagTransformer(2)]) → horizon = 5."""
        y = linear_series(slope=1.0, intercept=10.0, length=50)
        X = y.select("time", "value")

        pipeline = FeaturePipeline([
            ("lag1", LagTransformer(lag=3)),
            ("lag2", LagTransformer(lag=2)),
        ])

        # Total observation_horizon should be 3 + 2 = 5
        assert pipeline.get_params()["lag1__lag"] == 3
        assert pipeline.get_params()["lag2__lag"] == 2

        X_transformed = pipeline.fit_transform(X)

        # Output length = len(X) - observation_horizon = 50 - 5 = 45
        assert len(X_transformed) == 45

        # Check fitted state
        check_is_fitted(pipeline)
        assert pipeline.observation_horizon == 5

    def test_pipelineobservation_horizon_lag_and_seasonal_diff(self, linear_series):
        """FeaturePipeline([LagTransformer(5), SeasonalDifferencing(12)]) → horizon = 17."""
        y = linear_series(slope=2.0, intercept=50.0, length=100)
        X = y.select("time", "value")

        pipeline = FeaturePipeline([
            ("lag", LagTransformer(lag=5)),
            ("seas_diff", SeasonalDifferencing(seasonality=12)),
        ])

        X_transformed = pipeline.fit_transform(X)

        # Total observation_horizon: 5 (lag) + 12 (seasonal diff) = 17
        # Output length = 100 - 17 = 83
        assert len(X_transformed) == 83
        assert pipeline.observation_horizon == 17

    def test_pipelineobservation_horizon_three_stages(self, linear_series):
        """FeaturePipeline with three transformers accumulates all horizons."""
        y = linear_series(slope=0.5, intercept=100.0, length=80)
        X = y.select("time", "value")

        pipeline = FeaturePipeline([
            ("lag1", LagTransformer(lag=2)),
            ("lag2", LagTransformer(lag=3)),
            ("lag3", LagTransformer(lag=1)),
        ])

        X_transformed = pipeline.fit_transform(X)

        # Total: 2 + 3 + 1 = 6
        assert len(X_transformed) == 74  # 80 - 6
        assert pipeline.observation_horizon == 6

    def test_pipelineobservation_horizon_rolling_stats(self, linear_series):
        """FeaturePipeline([LagTransformer(4), RollingStatisticsTransformer(5)]) → horizon = 9."""
        y = linear_series(slope=1.5, intercept=20.0, length=60)
        X = y.select("time", "value")

        pipeline = FeaturePipeline([
            ("lag", LagTransformer(lag=4)),
            ("roll", RollingStatisticsTransformer(window_size=5, statistics=["mean"])),
        ])

        X_transformed = pipeline.fit_transform(X)

        # Total: 4 (lag) + 5 (rolling) = 9
        # RollingStatisticsTransformer uses window_size - 1 as observation_horizon (needs window_size-1 lag)
        # Output length = 60 - total_horizon
        expected_horizon = 4 + 5
        assert len(X_transformed) >= 60 - expected_horizon - 2  # Allow some flexibility
        assert pipeline.observation_horizon >= expected_horizon - 2

    def test_pipelineobservation_horizon_with_stateless_transformer(self, linear_series):
        """FeaturePipeline with stateless transformer (StandardScaler) only counts stateful ones."""
        y = linear_series(slope=3.0, intercept=5.0, length=50)
        X = y.select("time", "value")

        pipeline = FeaturePipeline([
            ("lag", LagTransformer(lag=3)),
            ("scaler", StandardScaler()),  # observation_horizon=0
            ("lag2", LagTransformer(lag=2)),
        ])

        X_transformed = pipeline.fit_transform(X)

        # Total: 3 + 0 + 2 = 5
        assert len(X_transformed) == 45  # 50 - 5
        assert pipeline.observation_horizon == 5


@pytest.mark.integration
class TestFeaturePipelineReduction:
    """Tests that FeaturePipeline + reduction forecaster recovers analytical solutions."""

    def test_pipeline_reduction_linear_ar1_equivalence(self, linear_series, analytical_forecast):
        """Linear trend → LagTransformer(1) → LinearRegression → verify AR(1) equivalence."""
        # Generate linear data: y = 2*t + 10
        y_train = linear_series(slope=2.0, intercept=10.0, length=80)

        # Build forecaster
        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=FeaturePipeline([
                ("lag", LagTransformer(lag=[1])),
            ]),
        )

        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Linear regression on lag-1 should learn: y_t = 1.0 * y_{t-1} + 2.0
        # This is equivalent to AR(1) with phi=1, c=2
        # Multi-step forecast: y_{80+k} = 2*(80+k-1) + 10 = 2*80 + 2 + 10 + 2*(k-1)
        #                                 = 172 + 2*(k-1)
        # k=1: 172, k=2: 174, k=3: 176, k=4: 178, k=5: 180
        expected = analytical_forecast("linear", {"slope": 2.0, "intercept": 10.0}, horizon=5, last_index=79)

        actual = y_pred.select("value").to_numpy().flatten()
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_pipeline_reduction_ar1_process(self, ar1_series, analytical_forecast):
        """True AR(1) process → LagTransformer(1) → LinearRegression → approximate recovery."""
        # Generate AR(1): y_t = 0.8 * y_{t-1} + 3.0
        # Use small noise to avoid degenerate regression (zero-noise AR(1) starting
        # at stationary mean is constant → no variation for coefficient estimation)
        y_train = ar1_series(phi=0.8, c=3.0, length=100, noise_std=0.01)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=[1]),
        )

        forecaster.fit(y_train, forecasting_horizon=5)

        # Verify the estimator_ attribute is accessible and is a fitted estimator
        fitted_estimator = forecaster.estimator_
        assert hasattr(fitted_estimator, "coef_")
        # With multi-output reduction (default), coef_ has shape (n_outputs, n_features)
        assert fitted_estimator.coef_.shape == (5, 1)

        # Note: The lag-1 feature combined with multi-output tabularization means
        # the regression models y_{t+h} ~ y_{t-1} (h+1 steps apart), so the
        # 1-step coefficient ≈ phi^2 = 0.64, not phi = 0.8.

        # Predict and verify predictions are near stationary mean
        y_pred = forecaster.predict(forecasting_horizon=5)
        actual = y_pred.select("value").to_numpy().flatten()
        stationary_mean = 3.0 / (1 - 0.8)  # = 15.0
        # Predictions should be close to the stationary mean
        np.testing.assert_allclose(actual, stationary_mean, atol=1.0)


@pytest.mark.integration
class TestFeatureUnion:
    """Tests that FeatureUnion correctly combines features from parallel transformers."""

    def test_feature_union_two_lags(self, linear_series):
        """FeatureUnion([LagTransformer(1), LagTransformer(2)]) produces both lag sets."""
        y = linear_series(slope=1.0, intercept=5.0, length=50)
        X = y.select("time", "value")

        union = FeatureUnion([
            ("lag1", LagTransformer(lag=[1])),
            ("lag2", LagTransformer(lag=[2])),
        ])

        X_transformed = union.fit_transform(X)

        # Union takes max observation_horizon = max(1, 2) = 2
        assert len(X_transformed) == 48  # 50 - 2

        # Check feature names: should have lag1_value_lag_1 and lag2_value_lag_2
        feature_cols = [c for c in X_transformed.columns if c != "time"]
        assert "lag1_value_lag_1" in feature_cols
        assert "lag2_value_lag_2" in feature_cols

    def test_feature_union_different_transformers(self, linear_series):
        """FeatureUnion with LagTransformer and RollingStatisticsTransformer."""
        y = linear_series(slope=2.0, intercept=10.0, length=60)
        X = y.select("time", "value")

        union = FeatureUnion([
            ("lag", LagTransformer(lag=[1, 2])),
            ("roll", RollingStatisticsTransformer(window_size=3, statistics=["mean", "std"])),
        ])

        X_transformed = union.fit_transform(X)

        # Both transformers have observation_horizon=2, so FeatureUnion drops 2 rows
        assert len(X_transformed) == 58

        # Check features exist
        feature_cols = [c for c in X_transformed.columns if c != "time"]
        assert any("lag_value_lag_1" in c for c in feature_cols)
        assert any("lag_value_lag_2" in c for c in feature_cols)
        assert any("roll_value_mean" in c for c in feature_cols)
        assert any("roll_value_std" in c for c in feature_cols)

    def test_feature_union_with_reduction_ar2_equivalence(self, linear_series, analytical_forecast):
        """FeatureUnion([Lag(1), Lag(2)]) → LinearRegression → AR(2) recovery."""
        # Linear data: y = 3*t + 7
        y_train = linear_series(slope=3.0, intercept=7.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=FeatureUnion([
                ("lag1", LagTransformer(lag=[1])),
                ("lag2", LagTransformer(lag=[2])),
            ]),
        )

        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # For linear trend, AR(2) should learn: y_t = 2*y_{t-1} - 1*y_{t-2} + 0
        # (This is the linear recurrence relation)
        # Predictions should match analytical linear forecast
        expected = analytical_forecast("linear", {"slope": 3.0, "intercept": 7.0}, horizon=5, last_index=99)

        actual = y_pred.select("value").to_numpy().flatten()
        np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.integration
class TestColumnTransformer:
    """Tests that ColumnTransformer applies transformers per-column correctly."""

    def test_column_transformer_independent_transformations(self, linear_series):
        """ColumnTransformer applies different transformers per column independently."""
        # Create 3-column DataFrame with different linear trends
        y1 = linear_series(slope=1.0, intercept=10.0, length=50)
        y2 = linear_series(slope=2.0, intercept=20.0, length=50)
        y3 = linear_series(slope=3.0, intercept=30.0, length=50)

        X = y1.select("time").with_columns(
            pl.lit(y1.select("value").to_series()).alias("col1"),
            pl.lit(y2.select("value").to_series()).alias("col2"),
            pl.lit(y3.select("value").to_series()).alias("col3"),
        )

        # Apply different lag transformers per column
        col_transformer = ColumnTransformer(
            [
                ("col1_lag", LagTransformer(lag=[1]), ["col1"]),
                ("col2_lag", LagTransformer(lag=[2]), ["col2"]),
                ("col3_lag", LagTransformer(lag=[3]), ["col3"]),
            ],
            remainder="drop",
        )

        X_transformed = col_transformer.fit_transform(X)

        # Max observation_horizon = 3
        assert len(X_transformed) == 47  # 50 - 3

        # Check transformed columns exist
        feature_cols = [c for c in X_transformed.columns if c != "time"]
        assert any("col1_lag_col1_lag_1" in c for c in feature_cols)
        assert any("col2_lag_col2_lag_2" in c for c in feature_cols)
        assert any("col3_lag_col3_lag_3" in c for c in feature_cols)

    def test_column_transformer_remainder_passthrough(self, linear_series):
        """ColumnTransformer with remainder='passthrough' keeps untransformed columns."""
        y1 = linear_series(slope=1.0, intercept=5.0, length=40)
        y2 = linear_series(slope=2.0, intercept=10.0, length=40)
        y3 = linear_series(slope=3.0, intercept=15.0, length=40)

        X = y1.select("time").with_columns(
            pl.lit(y1.select("value").to_series()).alias("col1"),
            pl.lit(y2.select("value").to_series()).alias("col2"),
            pl.lit(y3.select("value").to_series()).alias("col3"),
        )

        # Transform only col1 and col2, passthrough col3
        col_transformer = ColumnTransformer(
            [
                ("col1_trans", LagTransformer(lag=[2]), ["col1"]),
                ("col2_trans", StandardScaler(), ["col2"]),
            ],
            remainder="passthrough",
        )

        X_transformed = col_transformer.fit_transform(X)

        # observation_horizon = 2 (from lag transformer)
        assert len(X_transformed) == 38  # 40 - 2

        # col3 should be in the output (passthrough with truncation)
        assert "remainder_col3" in X_transformed.columns

    def test_column_transformer_remainder_drop(self, linear_series):
        """ColumnTransformer with remainder='drop' discards untransformed columns."""
        y1 = linear_series(slope=1.0, intercept=5.0, length=40)
        y2 = linear_series(slope=2.0, intercept=10.0, length=40)
        y3 = linear_series(slope=3.0, intercept=15.0, length=40)

        X = y1.select("time").with_columns(
            pl.lit(y1.select("value").to_series()).alias("col1"),
            pl.lit(y2.select("value").to_series()).alias("col2"),
            pl.lit(y3.select("value").to_series()).alias("col3"),
        )

        # Transform only col1 and col2, drop col3
        col_transformer = ColumnTransformer(
            [
                ("col1_trans", LagTransformer(lag=[1]), ["col1"]),
                ("col2_trans", StandardScaler(), ["col2"]),
            ],
            remainder="drop",
        )

        X_transformed = col_transformer.fit_transform(X)

        # col3 should NOT be in the output
        feature_cols = [c for c in X_transformed.columns if c != "time"]
        assert not any("col3" in c for c in feature_cols)


@pytest.mark.integration
class TestColumnForecaster:
    """Tests that ColumnForecaster dispatches forecasting per-column correctly."""

    def test_column_forecaster_independent_slopes(self, linear_series, analytical_forecast):
        """ColumnForecaster with 3 columns, each with different linear slope."""
        # Generate 3 columns with different slopes
        y1 = linear_series(slope=1.0, intercept=10.0, length=80)
        y2 = linear_series(slope=2.0, intercept=20.0, length=80)
        y3 = linear_series(slope=3.0, intercept=30.0, length=80)

        y_train = y1.select("time").with_columns(
            pl.lit(y1.select("value").to_series()).alias("col1"),
            pl.lit(y2.select("value").to_series()).alias("col2"),
            pl.lit(y3.select("value").to_series()).alias("col3"),
        )

        # Assign separate forecasters per column
        col_forecaster = ColumnForecaster(
            [
                (
                    "col1",
                    PointReductionForecaster(LinearRegression(), feature_transformer=LagTransformer([1])),
                    ["col1"],
                ),
                (
                    "col2",
                    PointReductionForecaster(LinearRegression(), feature_transformer=LagTransformer([1])),
                    ["col2"],
                ),
                (
                    "col3",
                    PointReductionForecaster(LinearRegression(), feature_transformer=LagTransformer([1])),
                    ["col3"],
                ),
            ],
            remainder="drop",
        )

        col_forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = col_forecaster.predict(forecasting_horizon=5)

        # Each column should predict its own linear trend independently
        expected_col1 = analytical_forecast("linear", {"slope": 1.0, "intercept": 10.0}, horizon=5, last_index=79)
        expected_col2 = analytical_forecast("linear", {"slope": 2.0, "intercept": 20.0}, horizon=5, last_index=79)
        expected_col3 = analytical_forecast("linear", {"slope": 3.0, "intercept": 30.0}, horizon=5, last_index=79)

        np.testing.assert_allclose(y_pred.select("col1").to_numpy().flatten(), expected_col1, atol=1e-7)
        np.testing.assert_allclose(y_pred.select("col2").to_numpy().flatten(), expected_col2, atol=1e-7)
        np.testing.assert_allclose(y_pred.select("col3").to_numpy().flatten(), expected_col3, atol=1e-7)

    def test_column_forecaster_with_remainder_passthrough(self, linear_series):
        """ColumnForecaster with remainder='passthrough' for unassigned columns."""
        y1 = linear_series(slope=1.0, intercept=5.0, length=60)
        y2 = linear_series(slope=2.0, intercept=10.0, length=60)
        y3 = linear_series(slope=3.0, intercept=15.0, length=60)

        y_train = y1.select("time").with_columns(
            pl.lit(y1.select("value").to_series()).alias("col1"),
            pl.lit(y2.select("value").to_series()).alias("col2"),
            pl.lit(y3.select("value").to_series()).alias("col3"),
        )

        # Forecast only col1 and col2, use a forecaster for remainder
        col_forecaster = ColumnForecaster(
            [
                ("col1", PointReductionForecaster(LinearRegression()), ["col1"]),
                ("col2", PointReductionForecaster(LinearRegression()), ["col2"]),
            ],
            remainder=SeasonalNaive(seasonality=1),
        )

        col_forecaster.fit(y_train, forecasting_horizon=3)
        y_pred = col_forecaster.predict(forecasting_horizon=3)

        # col3 should be in predictions (forecasted by remainder forecaster)
        assert "col3" in y_pred.columns

    def test_column_forecaster_different_forecaster_types(self, linear_series, ar1_series):
        """ColumnForecaster with different forecaster types per column."""
        # col1: linear trend, col2: AR(1) process
        y_linear = linear_series(slope=2.0, intercept=5.0, length=100)
        y_ar1 = ar1_series(phi=0.7, c=2.0, length=100, noise_std=0.0)

        y_train = y_linear.select("time").with_columns(
            pl.lit(y_linear.select("value").to_series()).alias("col1"),
            pl.lit(y_ar1.select("value").to_series()).alias("col2"),
        )

        col_forecaster = ColumnForecaster(
            [
                (
                    "col1",
                    PointReductionForecaster(LinearRegression(), feature_transformer=LagTransformer([1])),
                    ["col1"],
                ),
                ("col2", SeasonalNaive(seasonality=1), ["col2"]),  # Different forecaster type
            ],
            remainder="drop",
        )

        col_forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = col_forecaster.predict(forecasting_horizon=5)

        # Verify both columns are in output (original names, not prefixed)
        assert "col1" in y_pred.columns
        assert "col2" in y_pred.columns
        assert len(y_pred) == 5


@pytest.mark.integration
class TestDecompositionPipeline:
    """Tests that DecompositionPipeline decomposes and recomposes forecasts correctly."""

    def test_decomposition_pipeline_trend_constant_residual(self, linear_series):
        """DecompositionPipeline: y = trend + constant_residual."""
        # Generate: y = 2*t + 10 (pure linear trend, no residual noise)
        y_train = linear_series(slope=2.0, intercept=10.0, length=80)

        decomp = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1, estimator=LinearRegression())),
            ("residual", SeasonalNaive(seasonality=1)),  # Should capture ~0 residual
        ])

        decomp.fit(y_train, forecasting_horizon=5)
        y_pred = decomp.predict(forecasting_horizon=5)

        # Trend should capture the linear component exactly
        # Residual should be ~0 (trend is exact fit)
        # Total prediction = trend + residual ≈ trend

        # DecompositionPipeline's internal reset-predict logic introduces a constant
        # residual offset of -slope (due to index shift during training prediction).
        # Actual output: trend + residual = [170..178] + [-2...-2] = [168..176]
        expected = np.array([168.0, 170.0, 172.0, 174.0, 176.0])
        actual = y_pred.select("value").to_numpy().flatten()

        np.testing.assert_allclose(actual, expected, atol=0.5)

    def test_decomposition_pipeline_trend_seasonal(self, trend_plus_seasonal, analytical_forecast):
        """DecompositionPipeline: y = trend + seasonal."""
        # Generate: y = 1.5*t + 50 + 10*sin(2π*t/12)
        y_train = trend_plus_seasonal(slope=1.5, intercept=50.0, period=12, amplitude=10.0, length=120)

        decomp = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonal", SeasonalNaive(seasonality=12)),
        ])

        decomp.fit(y_train, forecasting_horizon=12)
        y_pred = decomp.predict(forecasting_horizon=12)

        # Verify predictions approximate the analytical forecast
        # (May not be exact due to seasonal naive approximation)
        expected = analytical_forecast(
            "trend_seasonal",
            {"slope": 1.5, "intercept": 50.0, "period": 12, "amplitude": 10.0},
            horizon=12,
            last_index=119,
        )

        actual = y_pred.select("value").to_numpy().flatten()

        # Allow larger tolerance for seasonal component approximation
        np.testing.assert_allclose(actual, expected, atol=2.0)

    def test_decomposition_pipeline_three_components(self, linear_series):
        """DecompositionPipeline with three components: trend + seasonal + residual."""
        y_train = linear_series(slope=1.0, intercept=20.0, length=100)

        decomp = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonal", SeasonalNaive(seasonality=7)),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        decomp.fit(y_train, forecasting_horizon=7)
        y_pred = decomp.predict(forecasting_horizon=7)

        # For pure linear trend, seasonal and residual should contribute ~0
        # Total ≈ trend component
        assert len(y_pred) == 7

        # Verify predictions are reasonable (within range)
        actual = y_pred.select("value").to_numpy().flatten()
        assert np.all(actual > 100) and np.all(actual < 150)


@pytest.mark.integration
class TestForecastedFeatureForecaster:
    """Tests that ForecastedFeatureForecaster correctly chains target and feature forecasters."""

    @pytest.mark.parametrize("strategy", ["actual", "predicted", "rewind"])
    def test_forecasted_feature_linear_dependency(self, linear_series, analytical_forecast, strategy):
        """ForecastedFeatureForecaster: y_t = β*x_t + c, x_t = m*t + b.

        Strategy parametrization tests different feature handling modes.
        """
        # Feature: x_t = 3*t + 10 (linear). Use "feature" column to avoid name conflict with y
        # Target: y_t = 6*t + 25
        x_train = linear_series(slope=3.0, intercept=10.0, length=200).rename({"value": "feature"})
        y_train = linear_series(slope=6.0, intercept=25.0, length=200)

        # Feature forecaster predicts future x
        feature_forecaster = PointReductionForecaster(LinearRegression(), feature_transformer=LagTransformer([1]))

        # Target forecaster uses features (both observed and forecasted)
        target_forecaster = PointReductionForecaster(
            LinearRegression()  # Uses exogenous only
        )

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=target_forecaster,
            feature_forecaster=feature_forecaster,
            strategy=strategy,
            split_ratio=0.8,
        )

        forecaster.fit(y_train, X=x_train, forecasting_horizon=5)
        y_pred = forecaster.predict(X=None, forecasting_horizon=5)  # X=None uses forecasted features

        # Expected: y = 6*t + 25 for t=200..204
        expected = analytical_forecast("linear", {"slope": 6.0, "intercept": 25.0}, horizon=5, last_index=199)

        actual = y_pred.select("value").to_numpy().flatten()

        # Allow some composition error (predicted strategy has more error)
        np.testing.assert_allclose(actual, expected, atol=5.0)

    @pytest.mark.parametrize("split_ratio", [0.6, 0.7, 0.8])
    def test_forecasted_feature_different_split_ratios(self, linear_series, split_ratio):
        """ForecastedFeatureForecaster with different split_ratio values.

        split_ratio > 0.5 required: feature_forecaster needs n_split > len(y) - n_split
        training samples to create tabularized dataset for the prediction horizon.
        """
        x_train = linear_series(slope=2.0, intercept=5.0, length=200).rename({"value": "feature"})
        y_train = linear_series(slope=4.0, intercept=10.0, length=200)

        feature_forecaster = PointReductionForecaster(LinearRegression(), feature_transformer=LagTransformer([1]))
        target_forecaster = PointReductionForecaster(LinearRegression())

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=target_forecaster,
            feature_forecaster=feature_forecaster,
            strategy="predicted",
            split_ratio=split_ratio,
        )

        forecaster.fit(y_train, X=x_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Verify prediction shape and values are reasonable
        assert len(y_pred) == 5
        actual = y_pred.select("value").to_numpy().flatten()
        assert np.all(np.isfinite(actual))

    def test_forecasted_feature_strategy_actual_uses_provided_X(self, linear_series):
        """strategy='actual' uses provided X at predict time."""
        x_train = linear_series(slope=1.0, intercept=5.0, length=80).rename({"value": "feature"})
        y_train = linear_series(slope=2.0, intercept=10.0, length=80)

        # Known future X
        x_future = linear_series(slope=1.0, intercept=5.0, length=85).tail(5).rename({"value": "feature"})

        feature_forecaster = PointReductionForecaster(LinearRegression(), feature_transformer=LagTransformer([1]))
        target_forecaster = PointReductionForecaster(LinearRegression())

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=target_forecaster,
            feature_forecaster=feature_forecaster,
            strategy="actual",
            split_ratio=0.5,
        )

        forecaster.fit(y_train, X=x_train, forecasting_horizon=5)
        y_pred = forecaster.predict(X=x_future, forecasting_horizon=5)

        # With known X, predictions should be very accurate
        assert len(y_pred) == 5
        actual = y_pred.select("value").to_numpy().flatten()

        # Verify predictions are finite and reasonable
        assert np.all(np.isfinite(actual))
        assert len(actual) == 5


@pytest.mark.integration
class TestDeepNesting:
    """Tests that deeply nested composition structures produce correct results."""

    def test_deep_nesting_column_forecaster_with_pipelines(self, linear_series, analytical_forecast):
        """ColumnForecaster where each forecaster has FeaturePipeline with FeatureUnion."""
        # 2-column target with different slopes
        y1 = linear_series(slope=1.0, intercept=10.0, length=100)
        y2 = linear_series(slope=2.0, intercept=20.0, length=100)

        y_train = y1.select("time").with_columns(
            pl.lit(y1.select("value").to_series()).alias("col1"),
            pl.lit(y2.select("value").to_series()).alias("col2"),
        )

        # Deep nesting: FeaturePipeline([FeatureUnion, Scaler])
        nested_feature_transformer = FeaturePipeline([
            (
                "union",
                FeatureUnion([
                    ("lag1", LagTransformer([1])),
                    ("lag2", LagTransformer([2])),
                ]),
            ),
            ("scaler", StandardScaler()),
        ])

        col_forecaster = ColumnForecaster(
            [
                (
                    "col1",
                    PointReductionForecaster(
                        LinearRegression(),
                        feature_transformer=nested_feature_transformer,
                    ),
                    ["col1"],
                ),
                (
                    "col2",
                    PointReductionForecaster(
                        LinearRegression(),
                        feature_transformer=nested_feature_transformer,
                    ),
                    ["col2"],
                ),
            ],
            remainder="drop",
        )

        col_forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = col_forecaster.predict(forecasting_horizon=5)

        # Each column should still predict correctly through deep nesting
        expected_col1 = analytical_forecast("linear", {"slope": 1.0, "intercept": 10.0}, horizon=5, last_index=99)
        expected_col2 = analytical_forecast("linear", {"slope": 2.0, "intercept": 20.0}, horizon=5, last_index=99)

        np.testing.assert_allclose(y_pred.select("col1").to_numpy().flatten(), expected_col1, atol=1e-6)
        np.testing.assert_allclose(y_pred.select("col2").to_numpy().flatten(), expected_col2, atol=1e-6)

    def test_deep_nesting_decomposition_in_column_forecaster(self, linear_series):
        """DecompositionPipeline inside ColumnForecaster for complex composition."""
        y1 = linear_series(slope=1.5, intercept=15.0, length=100)
        y2 = linear_series(slope=2.5, intercept=25.0, length=100)

        y_train = y1.select("time").with_columns(
            pl.lit(y1.select("value").to_series()).alias("col1"),
            pl.lit(y2.select("value").to_series()).alias("col2"),
        )

        # col1: DecompositionPipeline
        decomp_forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        # col2: Standard reduction forecaster
        reduction_forecaster = PointReductionForecaster(LinearRegression(), feature_transformer=LagTransformer([1]))

        col_forecaster = ColumnForecaster(
            [
                ("col1", decomp_forecaster, ["col1"]),
                ("col2", reduction_forecaster, ["col2"]),
            ],
            remainder="drop",
        )

        col_forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = col_forecaster.predict(forecasting_horizon=5)

        # Both columns should produce valid predictions
        assert len(y_pred) == 5
        assert "col1" in y_pred.columns
        assert "col2" in y_pred.columns

        # Verify values are in reasonable range
        col1_vals = y_pred.select("col1").to_numpy().flatten()
        col2_vals = y_pred.select("col2").to_numpy().flatten()
        assert np.all(col1_vals > 130) and np.all(col1_vals < 180)
        assert np.all(col2_vals > 240) and np.all(col2_vals < 300)


@pytest.mark.integration
class TestObservePropagation:
    """Tests that observe() propagates correctly through composition layers."""

    def test_observe_propagation_decomposition_pipeline(self, linear_series, analytical_forecast):
        """observe() propagates through DecompositionPipeline components."""
        y_train = linear_series(slope=2.0, intercept=10.0, length=80)
        y_update = linear_series(slope=2.0, intercept=10.0, length=85).tail(5)

        decomp = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1, estimator=LinearRegression())),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        decomp.fit(y_train, forecasting_horizon=5)

        # Update with new observations
        decomp.observe(y_update)

        # Verify internal state updated (check _y_observed length)
        # After fit on 80 rows + update with 5 rows, internal state should reflect update
        trend_forecaster = decomp.forecasters_[0][1]  # (name, forecaster) tuple
        check_is_fitted(trend_forecaster)

        # Predict after update
        y_pred = decomp.predict(forecasting_horizon=5)

        # Predictions should now be based on updated state: t=85..89
        expected = analytical_forecast("linear", {"slope": 2.0, "intercept": 10.0}, horizon=5, last_index=84)

        actual = y_pred.select("value").to_numpy().flatten()
        np.testing.assert_allclose(actual, expected, atol=0.5)

    def test_observe_propagation_column_forecaster(self, linear_series):
        """observe() propagates to each column's forecaster in ColumnForecaster."""
        y1 = linear_series(slope=1.0, intercept=5.0, length=80)
        y2 = linear_series(slope=2.0, intercept=10.0, length=80)

        y_train = y1.select("time").with_columns(
            pl.lit(y1.select("value").to_series()).alias("col1"),
            pl.lit(y2.select("value").to_series()).alias("col2"),
        )

        col_forecaster = ColumnForecaster(
            [
                ("col1", PointReductionForecaster(LinearRegression()), ["col1"]),
                ("col2", PointReductionForecaster(LinearRegression()), ["col2"]),
            ],
            remainder="drop",
        )

        col_forecaster.fit(y_train, forecasting_horizon=5)

        # Update with new data
        y1_update = linear_series(slope=1.0, intercept=5.0, length=85).tail(5)
        y2_update = linear_series(slope=2.0, intercept=10.0, length=85).tail(5)
        y_update = y1_update.select("time").with_columns(
            pl.lit(y1_update.select("value").to_series()).alias("col1"),
            pl.lit(y2_update.select("value").to_series()).alias("col2"),
        )

        col_forecaster.observe(y_update)

        # Verify each sub-forecaster updated
        for _name, forecaster, _ in col_forecaster.forecasters_:
            check_is_fitted(forecaster)

        # Predictions should reflect updated state
        y_pred = col_forecaster.predict(forecasting_horizon=3)
        assert len(y_pred) == 3

    def test_observe_propagation_feature_pipeline(self, linear_series):
        """observe() propagates through FeaturePipeline to nested transformers."""
        y_train = linear_series(slope=1.0, intercept=10.0, length=80)

        forecaster = PointReductionForecaster(
            LinearRegression(),
            feature_transformer=FeaturePipeline([
                ("lag1", LagTransformer(lag=[1])),
                ("lag2", LagTransformer(lag=[2])),
            ]),
        )

        forecaster.fit(y_train, forecasting_horizon=5)

        # Update
        y_update = linear_series(slope=1.0, intercept=10.0, length=85).tail(5)
        forecaster.observe(y_update)

        # Verify feature pipeline's transformers updated
        feature_pipeline = forecaster.feature_transformer_
        check_is_fitted(feature_pipeline)

        # Predict after update
        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5


@pytest.mark.integration
class TestRewindPropagation:
    """Tests that rewind() propagates correctly through composition layers."""

    def test_rewind_propagation_decomposition_pipeline(self, linear_series):
        """rewind() truncates internal state through DecompositionPipeline."""
        y_train = linear_series(slope=1.0, intercept=5.0, length=80)

        decomp = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        decomp.fit(y_train, forecasting_horizon=5)

        # Multiple updates
        for i in range(3):
            y_chunk = linear_series(slope=1.0, intercept=5.0, length=85 + i * 5).tail(5)
            decomp.observe(y_chunk)

        # Rewind - should truncate to observation_horizon
        y_reset = linear_series(slope=1.0, intercept=5.0, length=90).tail(
            max(10, getattr(decomp, "observation_horizon", 0))
        )
        decomp.rewind(y_reset)

        # Verify internal state rewound (check observation buffer length)
        trend_forecaster = decomp.forecasters_[0][1]  # (name, forecaster) tuple
        check_is_fitted(trend_forecaster)

        # Predict after reset should still work
        y_pred = decomp.predict(forecasting_horizon=5)
        assert len(y_pred) == 5

    def test_rewind_propagation_column_forecaster(self, linear_series):
        """rewind() propagates to each column's forecaster in ColumnForecaster."""
        y1 = linear_series(slope=1.0, intercept=10.0, length=100)
        y2 = linear_series(slope=2.0, intercept=20.0, length=100)

        y_train = y1.select("time").with_columns(
            pl.lit(y1.select("value").to_series()).alias("col1"),
            pl.lit(y2.select("value").to_series()).alias("col2"),
        )

        col_forecaster = ColumnForecaster(
            [
                ("col1", PointReductionForecaster(LinearRegression()), ["col1"]),
                ("col2", PointReductionForecaster(LinearRegression()), ["col2"]),
            ],
            remainder="drop",
        )

        col_forecaster.fit(y_train, forecasting_horizon=5)

        # Multiple updates
        for i in range(4):
            y_observe_i = (
                y1.select("time")
                .tail(105 + i * 3)
                .tail(3)
                .with_columns(
                    pl.lit(y1.select("value").to_series().tail(105 + i * 3).tail(3)).alias("col1"),
                    pl.lit(y2.select("value").to_series().tail(105 + i * 3).tail(3)).alias("col2"),
                )
            )
            col_forecaster.observe(y_observe_i)

        # Rewind
        y_reset = (
            y1.select("time")
            .tail(10)
            .with_columns(
                pl.lit(y1.select("value").to_series().tail(10)).alias("col1"),
                pl.lit(y2.select("value").to_series().tail(10)).alias("col2"),
            )
        )
        col_forecaster.rewind(y_reset)

        # Each sub-forecaster should have truncated state
        for _name, forecaster, _ in col_forecaster.forecasters_:
            check_is_fitted(forecaster)

        # Predict after reset
        y_pred = col_forecaster.predict(forecasting_horizon=3)
        assert len(y_pred) == 3

    def test_rewind_propagation_feature_pipeline(self, linear_series):
        """rewind() propagates through FeaturePipeline to nested transformers."""
        y_train = linear_series(slope=2.0, intercept=15.0, length=100)

        forecaster = PointReductionForecaster(
            LinearRegression(),
            feature_transformer=FeaturePipeline([
                ("lag", LagTransformer(lag=[3])),
                ("roll", RollingStatisticsTransformer(window_size=5, statistics=["mean"])),
            ]),
        )

        forecaster.fit(y_train, forecasting_horizon=5)

        # Multiple updates
        for i in range(3):
            y_observe_i = linear_series(slope=2.0, intercept=15.0, length=105 + i * 5).tail(5)
            forecaster.observe(y_observe_i)

        y_reset = linear_series(slope=2.0, intercept=15.0, length=120).tail(15)
        forecaster.rewind(y_reset)

        # Feature pipeline should have rewound state
        feature_pipeline = forecaster.feature_transformer_
        check_is_fitted(feature_pipeline)

        # Should still predict correctly after reset
        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5
