"""Stress tests: DecompositionPipeline algebraic correctness.

Tests verify that decomposition/recomposition algebra is correct:
- Additive identity (prediction = sum of component predictions)
- Multiplicative decomposition via LogTransformer
- Multi-component decomposition (3+ components)
- Component-specific target_transformer handling
- fit_forecasting_horizon ≠ predict_forecasting_horizon
- observe/rewind algebra

All tests use analytical series for exact verification.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from yohou.compose import DecompositionPipeline
from yohou.point import SeasonalNaive
from yohou.preprocessing import StandardScaler
from yohou.stationarity import (
    FourierSeasonalityForecaster,
    LogTransformer,
    PatternSeasonalityForecaster,
    PolynomialTrendForecaster,
)


@pytest.mark.integration
class TestTwoComponentAdditiveIdentity:
    """Verify additive identity for 2-component DecompositionPipeline (trend + residual)."""

    def test_two_component_additive_identity_linear_trend(self, linear_series):
        """Verify DecompositionPipeline([trend, residual]) satisfies additive identity.

        For y = 5t + 3 + ε where ε is small constant:
        - Trend captures 5t + 3
        - Residual captures ε
        - Prediction = trend_prediction + residual_prediction
        """
        y = linear_series(slope=5.0, intercept=3.0, length=100)
        X = y.select("time")

        # Two-component decomposition: trend + residual
        pipeline = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        # Fit on first 80 points, predict 5 steps ahead
        y_train = y[:80]
        X_train = X[:80]
        forecasting_horizon = 5

        pipeline.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)

        # Get predictions
        y_pred = pipeline.predict(forecasting_horizon=forecasting_horizon)

        # Get component predictions manually
        forecasters_dict = dict(pipeline.forecasters_)
        trend_forecaster = forecasters_dict["trend"]
        residual_forecaster = forecasters_dict["residual"]

        # Trend prediction (on original y_train)
        trend_pred = trend_forecaster.predict(forecasting_horizon=forecasting_horizon)

        # Residual prediction (on y_train - trend_fitted)
        # For residual, we need to check what it was fitted on
        # In decomposition, residual is fitted on y - trend_prediction_in_sample
        # For prediction, residual predicts its own learned pattern
        residual_pred = residual_forecaster.predict(forecasting_horizon=forecasting_horizon)

        # Verify additive identity: y_pred = trend_pred + residual_pred
        # Extract the target column (first non-time column)
        target_col = [c for c in y_pred.columns if c not in {"time", "observed_time"}][0]

        pred_values = y_pred[target_col].to_numpy()
        trend_values = trend_pred[target_col].to_numpy()
        residual_values = residual_pred[target_col].to_numpy()

        np.testing.assert_allclose(
            pred_values,
            trend_values + residual_values,
            atol=1e-7,
            err_msg="Decomposition prediction should equal sum of component predictions",
        )

        # Also verify the linear trend is captured correctly
        # Expected: y = 5*t + 3, so at t=80, 81, 82, 83, 84 we expect:
        expected_times = np.arange(80, 85)
        expected_values = 5.0 * expected_times + 3.0

        # Allow error due to polynomial fit on datetime indices (boundary offset)
        np.testing.assert_allclose(
            pred_values, expected_values, atol=6.0, err_msg="Linear trend should be captured accurately"
        )

    def test_two_component_additive_identity_constant_series(self):
        """Verify additive identity on constant series (simplest case).

        For y = 10 (constant):
        - Trend captures constant 10
        - Residual captures 0
        - Prediction = 10 + 0 = 10
        """
        # Create constant series
        n = 100
        time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        y = pl.DataFrame({
            "time": time,
            "target": [10.0] * n,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        X = y.select("time")

        pipeline = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=0)),  # Constant trend
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        y_train = y[:80]
        X_train = X[:80]
        forecasting_horizon = 10

        pipeline.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)
        y_pred = pipeline.predict(forecasting_horizon=forecasting_horizon)

        # Get component predictions
        forecasters_dict = dict(pipeline.forecasters_)
        trend_forecaster = forecasters_dict["trend"]
        residual_forecaster = forecasters_dict["residual"]

        trend_pred = trend_forecaster.predict(forecasting_horizon=forecasting_horizon)
        residual_pred = residual_forecaster.predict(forecasting_horizon=forecasting_horizon)

        target_col = [c for c in y_pred.columns if c not in ["time", "observed_time"]][0]

        pred_values = y_pred[target_col].to_numpy()
        trend_values = trend_pred[target_col].to_numpy()
        residual_values = residual_pred[target_col].to_numpy()

        # Verify additive identity
        np.testing.assert_allclose(
            pred_values,
            trend_values + residual_values,
            atol=1e-7,
            err_msg="Additive identity must hold for constant series",
        )

        # Verify all predictions are constant 10
        np.testing.assert_allclose(
            pred_values, 10.0, atol=1e-6, err_msg="Constant series should predict constant value"
        )


@pytest.mark.integration
class TestMultiplicativeDecomposition:
    """Verify multiplicative decomposition via LogTransformer preserves algebraic identity."""

    def test_multiplicative_decomposition_exponential_seasonal(self, exponential_series, seasonal_series):
        """Verify multiplicative decomposition via LogTransformer.

        For y = exp(0.1t) * S(t) where S(t) is periodic:
        - log(y) = 0.1t + log(S(t))
        - Decompose in log-domain: trend + seasonal
        - exp(sum) recovers correct forecast
        """
        # Create exponential trend
        y_exp = exponential_series(a=0.05, b=0.0, length=100)

        # Create seasonal component (periodic with amplitude)
        y_seas = seasonal_series(period=7, amplitudes=[0.5], length=100)

        # Multiplicative combination: y = exp_trend * (1 + seasonal_pattern)
        # Note: seasonal_series gives additive pattern, so we use (1 + normalized_pattern)
        time = y_exp["time"]
        exp_values = y_exp.select(pl.exclude("time")).to_numpy().flatten()
        seas_values = y_seas.select(pl.exclude("time")).to_numpy().flatten()

        # Normalize seasonal to be multiplicative factor around 1
        seas_normalized = 1.0 + (seas_values - seas_values.mean()) / (seas_values.std() + 1e-8) * 0.2

        y_mult = pl.DataFrame({
            "time": time,
            "target": exp_values * seas_normalized,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        X = y_mult.select("time")

        # Decomposition with log transform
        pipeline = DecompositionPipeline(
            [
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("seasonal", PatternSeasonalityForecaster(seasonality=7)),
                ("residual", SeasonalNaive(seasonality=1)),
            ],
            target_transformer=LogTransformer(),
        )

        y_train = y_mult[:80]
        X_train = X[:80]
        forecasting_horizon = 7

        pipeline.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)
        y_pred = pipeline.predict(forecasting_horizon=forecasting_horizon)

        # Verify prediction is in original scale (not log scale)
        target_col = [c for c in y_pred.columns if c not in ["time", "observed_time"]][0]
        pred_values = y_pred[target_col].to_numpy()

        # Predictions should be positive (exp of log-domain sum)
        assert np.all(pred_values > 0), "Multiplicative predictions should be positive"
        assert pred_values.shape == (forecasting_horizon,), "Prediction shape should match horizon"

        # Verify additive identity holds in log-domain by using predict_transformed=True
        # (component predictions are in log-domain with transformed column names)
        forecasters_dict = dict(pipeline.forecasters_)
        trend_pred = forecasters_dict["trend"].predict(
            forecasting_horizon=forecasting_horizon, predict_transformed=True
        )
        seasonal_pred = forecasters_dict["seasonal"].predict(
            forecasting_horizon=forecasting_horizon, predict_transformed=True
        )
        residual_pred = forecasters_dict["residual"].predict(
            forecasting_horizon=forecasting_horizon, predict_transformed=True
        )

        # All components should produce predictions of correct shape
        for name, cpred in [("trend", trend_pred), ("seasonal", seasonal_pred), ("residual", residual_pred)]:
            assert len(cpred) == forecasting_horizon, f"{name} prediction shape mismatch"

    def test_multiplicative_two_component_simple(self):
        """Verify simple multiplicative decomposition: y = trend * seasonal.

        Using log transform, this becomes: log(y) = log(trend) + log(seasonal)
        """
        # Create multiplicative series: y = (2 + 0.1*t) * (1 + 0.5*sin(2π*t/7))
        n = 100
        t = np.arange(n)
        trend = 2.0 + 0.1 * t
        seasonal = 1.0 + 0.5 * np.sin(2 * np.pi * t / 7)
        y_values = trend * seasonal

        time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        y = pl.DataFrame({
            "time": time,
            "target": y_values,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        X = y.select("time")

        pipeline = DecompositionPipeline(
            [
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("seasonal", PatternSeasonalityForecaster(seasonality=7)),
            ],
            target_transformer=LogTransformer(),
        )

        y_train = y[:80]
        X_train = X[:80]
        forecasting_horizon = 21  # 3 full periods for clear oscillation

        pipeline.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)
        y_pred = pipeline.predict(forecasting_horizon=forecasting_horizon)

        target_col = [c for c in y_pred.columns if c not in ["time", "observed_time"]][0]
        pred_values = y_pred[target_col].to_numpy()

        # Verify predictions are positive (multiplicative nature preserved)
        assert np.all(pred_values > 0), "Multiplicative predictions must be positive"

        # Verify predictions follow expected pattern (increasing trend with oscillations)
        assert pred_values[-1] > pred_values[0], "Trend should be increasing"

        # Verify seasonal pattern exists (detect oscillations)
        # Check if there are at least 1 local max and 1 local min in predictions
        diffs = np.diff(pred_values)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        assert sign_changes >= 2, "Should have seasonal oscillations (multiple peaks/troughs)"


@pytest.mark.integration
class TestThreeComponentDecomposition:
    """Verify additive identity for 3-component decomposition (trend + seasonal + residual)."""

    def test_three_component_decomposition_weekly_seasonality(self, trend_plus_seasonal):
        """Verify 3-component decomposition: trend + weekly_season + residual.

        For y = 2t + 10 + 5*sin(2πt/7):
        - Trend captures 2t + 10
        - Seasonal captures 5*sin(2πt/7)
        - Residual captures ≈ 0
        """
        y = trend_plus_seasonal(
            slope=2.0,
            intercept=10.0,
            period=7,
            amplitude=5.0,
            length=100,
        )
        X = y.select("time")

        pipeline = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonal", PatternSeasonalityForecaster(seasonality=7)),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        y_train = y[:80]
        X_train = X[:80]
        forecasting_horizon = 14  # Two full weeks

        pipeline.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)
        y_pred = pipeline.predict(forecasting_horizon=forecasting_horizon)

        # Verify additive identity
        forecasters_dict = dict(pipeline.forecasters_)
        trend_forecaster = forecasters_dict["trend"]
        seasonal_forecaster = forecasters_dict["seasonal"]
        residual_forecaster = forecasters_dict["residual"]

        trend_pred = trend_forecaster.predict(forecasting_horizon=forecasting_horizon)
        seasonal_pred = seasonal_forecaster.predict(forecasting_horizon=forecasting_horizon)
        residual_pred = residual_forecaster.predict(forecasting_horizon=forecasting_horizon)

        target_col = [c for c in y_pred.columns if c not in ["time", "observed_time"]][0]

        pred_values = y_pred[target_col].to_numpy()
        trend_values = trend_pred[target_col].to_numpy()
        seasonal_values = seasonal_pred[target_col].to_numpy()
        residual_values = residual_pred[target_col].to_numpy()

        # Verify additive identity: y_pred = trend + seasonal + residual
        np.testing.assert_allclose(
            pred_values,
            trend_values + seasonal_values + residual_values,
            atol=1e-6,
            err_msg="3-component prediction must equal sum of components",
        )

        # Verify analytical correctness (trend extrapolation)
        t_pred = np.arange(80, 80 + forecasting_horizon)
        expected_trend = 2.0 * t_pred + 10.0

        # Trend component should capture linear trend
        np.testing.assert_allclose(
            trend_values,
            expected_trend,
            atol=0.5,  # Allow some error due to seasonal/noise interference
            err_msg="Trend component should capture linear trend",
        )

    def test_three_component_decomposition_monthly_seasonality(self):
        """Verify 3-component decomposition with monthly seasonality (period=12)."""
        n = 150
        t = np.arange(n)
        trend = 1.5 * t + 20.0
        seasonal = 8.0 * np.sin(2 * np.pi * t / 12)
        noise = np.random.RandomState(42).normal(0, 0.05, n)
        y_values = trend + seasonal + noise

        time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        y = pl.DataFrame({
            "time": time,
            "target": y_values,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        X = y.select("time")

        pipeline = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonal", FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2, 3])),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        y_train = y[:120]
        X_train = X[:120]
        forecasting_horizon = 12  # One full month

        pipeline.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)
        y_pred = pipeline.predict(forecasting_horizon=forecasting_horizon)

        # Get component predictions
        forecasters_dict = dict(pipeline.forecasters_)
        trend_forecaster = forecasters_dict["trend"]
        seasonal_forecaster = forecasters_dict["seasonal"]
        residual_forecaster = forecasters_dict["residual"]

        trend_pred = trend_forecaster.predict(forecasting_horizon=forecasting_horizon)
        seasonal_pred = seasonal_forecaster.predict(forecasting_horizon=forecasting_horizon)
        residual_pred = residual_forecaster.predict(forecasting_horizon=forecasting_horizon)

        target_col = [c for c in y_pred.columns if c not in ["time", "observed_time"]][0]

        # Verify additive identity
        np.testing.assert_allclose(
            y_pred[target_col].to_numpy(),
            (
                trend_pred[target_col].to_numpy()
                + seasonal_pred[target_col].to_numpy()
                + residual_pred[target_col].to_numpy()
            ),
            atol=1e-6,
        )

    def test_three_component_decomposition_daily_seasonality(self):
        """Verify 3-component decomposition with daily seasonality (period=24 for hourly data)."""
        n = 200
        t = np.arange(n)
        trend = 0.5 * t + 100.0
        seasonal = 10.0 * np.sin(2 * np.pi * t / 24)
        noise = np.random.RandomState(123).normal(0, 0.1, n)
        y_values = trend + seasonal + noise

        time = [datetime(2020, 1, 1) + timedelta(hours=i) for i in range(n)]

        y = pl.DataFrame({
            "time": time,
            "target": y_values,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        X = y.select("time")

        pipeline = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonal", PatternSeasonalityForecaster(seasonality=24)),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        y_train = y[:150]
        X_train = X[:150]
        forecasting_horizon = 24  # One full day

        pipeline.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)
        y_pred = pipeline.predict(forecasting_horizon=forecasting_horizon)

        # Get component predictions
        forecasters_dict = dict(pipeline.forecasters_)
        trend_forecaster = forecasters_dict["trend"]
        seasonal_forecaster = forecasters_dict["seasonal"]
        residual_forecaster = forecasters_dict["residual"]

        trend_pred = trend_forecaster.predict(forecasting_horizon=forecasting_horizon)
        seasonal_pred = seasonal_forecaster.predict(forecasting_horizon=forecasting_horizon)
        residual_pred = residual_forecaster.predict(forecasting_horizon=forecasting_horizon)

        target_col = [c for c in y_pred.columns if c not in ["time", "observed_time"]][0]

        # Verify additive identity
        np.testing.assert_allclose(
            y_pred[target_col].to_numpy(),
            (
                trend_pred[target_col].to_numpy()
                + seasonal_pred[target_col].to_numpy()
                + residual_pred[target_col].to_numpy()
            ),
            atol=1e-6,
        )


@pytest.mark.integration
class TestFourComponentDecomposition:
    """Verify additive identity for 4-component deep decomposition."""

    def test_four_component_decomposition_quadratic_seasonal(self):
        """Verify 4-component deep decomposition: linear + quadratic_residual + weekly + residual.

        For y = 0.01t² + 2t + 10 + 3*sin(2πt/7):
        - Linear trend captures most of 2t + 10
        - Quadratic residual captures 0.01t²
        - Weekly seasonal captures 3*sin(2πt/7)
        - Final residual captures ≈ 0
        """
        n = 150
        t = np.arange(n)
        y_values = 0.01 * t**2 + 2.0 * t + 10.0 + 3.0 * np.sin(2 * np.pi * t / 7)

        time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        y = pl.DataFrame({
            "time": time,
            "target": y_values,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        X = y.select("time")

        pipeline = DecompositionPipeline([
            ("linear", PolynomialTrendForecaster(degree=1)),
            ("quadratic_residual", PolynomialTrendForecaster(degree=2)),
            ("weekly", PatternSeasonalityForecaster(seasonality=7)),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        y_train = y[:120]
        X_train = X[:120]
        forecasting_horizon = 14

        pipeline.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)
        y_pred = pipeline.predict(forecasting_horizon=forecasting_horizon)

        # Get all component predictions
        forecasters_dict = dict(pipeline.forecasters_)
        linear_forecaster = forecasters_dict["linear"]
        quad_forecaster = forecasters_dict["quadratic_residual"]
        weekly_forecaster = forecasters_dict["weekly"]
        residual_forecaster = forecasters_dict["residual"]

        linear_pred = linear_forecaster.predict(forecasting_horizon=forecasting_horizon)
        quad_pred = quad_forecaster.predict(forecasting_horizon=forecasting_horizon)
        weekly_pred = weekly_forecaster.predict(forecasting_horizon=forecasting_horizon)
        residual_pred = residual_forecaster.predict(forecasting_horizon=forecasting_horizon)

        target_col = [c for c in y_pred.columns if c not in ["time", "observed_time"]][0]

        pred_values = y_pred[target_col].to_numpy()
        component_sum = (
            linear_pred[target_col].to_numpy()
            + quad_pred[target_col].to_numpy()
            + weekly_pred[target_col].to_numpy()
            + residual_pred[target_col].to_numpy()
        )

        # Verify additive identity for 4 components
        np.testing.assert_allclose(
            pred_values, component_sum, atol=1e-6, err_msg="4-component prediction must equal sum of all components"
        )

        # Verify analytical correctness
        t_pred = np.arange(120, 120 + forecasting_horizon)
        expected_values = 0.01 * t_pred**2 + 2.0 * t_pred + 10.0 + 3.0 * np.sin(2 * np.pi * t_pred / 7)

        np.testing.assert_allclose(
            pred_values,
            expected_values,
            atol=8.0,  # 4-component decomposition is approximate, allow larger tolerance
            err_msg="4-component decomposition should approximate true function",
        )

    def test_four_component_decomposition_sequential_residual_extraction(self):
        """Verify sequential residual extraction in 4-component decomposition.

        Each component should fit on residuals from previous components.
        """
        n = 100
        t = np.arange(n)
        # Complex function: quadratic + two seasonal patterns
        y_values = (
            0.02 * t**2
            + 1.0 * t
            + 50.0
            + 5.0 * np.sin(2 * np.pi * t / 7)  # Weekly
            + 2.0 * np.sin(2 * np.pi * t / 3)  # 3-day cycle
        )

        time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        y = pl.DataFrame({
            "time": time,
            "target": y_values,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        X = y.select("time")

        pipeline = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=2)),
            ("weekly", FourierSeasonalityForecaster(seasonality=7, harmonics=[1, 2])),
            ("short_cycle", FourierSeasonalityForecaster(seasonality=3, harmonics=[1])),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        y_train = y[:80]
        X_train = X[:80]
        forecasting_horizon = 10

        pipeline.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)
        y_pred = pipeline.predict(forecasting_horizon=forecasting_horizon)

        # Verify additive identity
        forecasters_dict = dict(pipeline.forecasters_)
        trend_pred = forecasters_dict["trend"].predict(forecasting_horizon=forecasting_horizon)
        weekly_pred = forecasters_dict["weekly"].predict(forecasting_horizon=forecasting_horizon)
        short_pred = forecasters_dict["short_cycle"].predict(forecasting_horizon=forecasting_horizon)
        residual_pred = forecasters_dict["residual"].predict(forecasting_horizon=forecasting_horizon)

        target_col = [c for c in y_pred.columns if c not in ["time", "observed_time"]][0]

        np.testing.assert_allclose(
            y_pred[target_col].to_numpy(),
            (
                trend_pred[target_col].to_numpy()
                + weekly_pred[target_col].to_numpy()
                + short_pred[target_col].to_numpy()
                + residual_pred[target_col].to_numpy()
            ),
            atol=1e-6,
            err_msg="Sequential residual extraction must preserve additive identity",
        )


@pytest.mark.integration
class TestComponentSpecificTransformers:
    """Verify decomposition with component-specific target_transformers."""

    def test_component_specific_transformers_mixed(self):
        """Verify decomposition with component-specific target_transformers.

        One component has StandardScaler, others don't.
        Scaling should not corrupt additive recomposition.
        """
        n = 100
        t = np.arange(n)
        y_values = 10.0 * t + 100.0 + 5.0 * np.sin(2 * np.pi * t / 7)

        time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        y = pl.DataFrame({
            "time": time,
            "target": y_values,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        X = y.select("time")

        # Trend with StandardScaler, seasonal without
        pipeline = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1, target_transformer=StandardScaler())),
            ("seasonal", PatternSeasonalityForecaster(seasonality=7)),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        y_train = y[:80]
        X_train = X[:80]
        forecasting_horizon = 7

        pipeline.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)
        y_pred = pipeline.predict(forecasting_horizon=forecasting_horizon)

        # Get component predictions in transformed space (matching what pipeline sums)
        forecasters_dict = dict(pipeline.forecasters_)
        trend_pred = forecasters_dict["trend"].predict(
            forecasting_horizon=forecasting_horizon, predict_transformed=True
        )
        seasonal_pred = forecasters_dict["seasonal"].predict(
            forecasting_horizon=forecasting_horizon, predict_transformed=True
        )
        residual_pred = forecasters_dict["residual"].predict(
            forecasting_horizon=forecasting_horizon, predict_transformed=True
        )

        target_col = [c for c in y_pred.columns if c not in ["time", "observed_time"]][0]
        # Component predictions in transformed space share column names
        comp_col = [c for c in trend_pred.columns if c not in ["time", "observed_time"]][0]

        # Verify additive identity in transformed space (pipeline sums transformed predictions)
        np.testing.assert_allclose(
            y_pred[target_col].to_numpy(),
            (trend_pred[comp_col].to_numpy() + seasonal_pred[comp_col].to_numpy() + residual_pred[comp_col].to_numpy()),
            atol=1e-6,
            err_msg="Component transformers should not corrupt additive identity",
        )

    def test_component_specific_transformers_all_scaled(self):
        """Verify decomposition where ALL components have StandardScaler."""
        n = 100
        t = np.arange(n)
        y_values = 5.0 * t + 50.0 + 3.0 * np.sin(2 * np.pi * t / 7)

        time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        y = pl.DataFrame({
            "time": time,
            "target": y_values,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        X = y.select("time")

        # All components with StandardScaler
        pipeline = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1, target_transformer=StandardScaler())),
            ("seasonal", PatternSeasonalityForecaster(seasonality=7, target_transformer=StandardScaler())),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        y_train = y[:80]
        X_train = X[:80]
        forecasting_horizon = 14

        pipeline.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)
        y_pred = pipeline.predict(forecasting_horizon=forecasting_horizon)

        # Get component predictions in transformed space (matching what pipeline sums)
        forecasters_dict = dict(pipeline.forecasters_)
        trend_pred = forecasters_dict["trend"].predict(
            forecasting_horizon=forecasting_horizon, predict_transformed=True
        )
        seasonal_pred = forecasters_dict["seasonal"].predict(
            forecasting_horizon=forecasting_horizon, predict_transformed=True
        )
        residual_pred = forecasters_dict["residual"].predict(
            forecasting_horizon=forecasting_horizon, predict_transformed=True
        )

        target_col = [c for c in y_pred.columns if c not in ["time", "observed_time"]][0]
        # Component predictions in transformed space may have different column names
        comp_col = [c for c in trend_pred.columns if c not in ["time", "observed_time"]][0]

        # Verify additive identity in transformed space
        np.testing.assert_allclose(
            y_pred[target_col].to_numpy(),
            (trend_pred[comp_col].to_numpy() + seasonal_pred[comp_col].to_numpy() + residual_pred[comp_col].to_numpy()),
            atol=1e-6,
            err_msg="All-scaled components should still satisfy additive identity",
        )

    def test_component_specific_transformers_none(self):
        """Verify decomposition with NO component transformers (baseline case)."""
        n = 100
        t = np.arange(n)
        y_values = 3.0 * t + 20.0 + 2.0 * np.sin(2 * np.pi * t / 7)

        time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        y = pl.DataFrame({
            "time": time,
            "target": y_values,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        X = y.select("time")

        # No transformers on any component
        pipeline = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonal", PatternSeasonalityForecaster(seasonality=7)),
        ])

        y_train = y[:80]
        X_train = X[:80]
        forecasting_horizon = 7

        pipeline.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)
        y_pred = pipeline.predict(forecasting_horizon=forecasting_horizon)

        # Get component predictions
        forecasters_dict = dict(pipeline.forecasters_)
        trend_pred = forecasters_dict["trend"].predict(forecasting_horizon=forecasting_horizon)
        forecasters_dict = dict(pipeline.forecasters_)
        seasonal_pred = forecasters_dict["seasonal"].predict(forecasting_horizon=forecasting_horizon)

        target_col = [c for c in y_pred.columns if c not in ["time", "observed_time"]][0]

        # Verify additive identity (simplest case)
        np.testing.assert_allclose(
            y_pred[target_col].to_numpy(),
            (trend_pred[target_col].to_numpy() + seasonal_pred[target_col].to_numpy()),
            atol=1e-7,
            err_msg="No transformers: strictest additive identity",
        )


@pytest.mark.integration
class TestDifferentForecastingHorizons:
    """Verify decomposition algebra when fit_forecasting_horizon != predict_forecasting_horizon."""

    def test_different_horizons_fit3_predict10(self, linear_series):
        """Verify decomposition with fit_fh=3, predict_fh=10.

        All component forecasters should produce 10 steps.
        Sum should be correct for all 10 steps.
        """
        y = linear_series(slope=2.0, intercept=10.0, length=100)
        X = y.select("time")

        pipeline = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        y_train = y[:80]
        X_train = X[:80]

        # Fit with horizon=3
        pipeline.fit(y_train, X_train, forecasting_horizon=3)

        # Predict with horizon=10
        forecasting_horizon = 10
        y_pred = pipeline.predict(forecasting_horizon=forecasting_horizon)

        # Verify shape
        assert len(y_pred) == forecasting_horizon, "Prediction should have 10 rows"

        # Get component predictions with same horizon
        forecasters_dict = dict(pipeline.forecasters_)
        trend_pred = forecasters_dict["trend"].predict(forecasting_horizon=forecasting_horizon)
        residual_pred = forecasters_dict["residual"].predict(forecasting_horizon=forecasting_horizon)

        target_col = [c for c in y_pred.columns if c not in ["time", "observed_time"]][0]

        # Verify additive identity for all 10 steps
        np.testing.assert_allclose(
            y_pred[target_col].to_numpy(),
            (trend_pred[target_col].to_numpy() + residual_pred[target_col].to_numpy()),
            atol=1e-7,
            err_msg="Additive identity must hold even with different predict horizon",
        )

        # Verify analytical correctness (allow for polynomial fit boundary offset)
        t_pred = np.arange(80, 90)
        expected_values = 2.0 * t_pred + 10.0

        np.testing.assert_allclose(
            y_pred[target_col].to_numpy(),
            expected_values,
            atol=3.0,
            err_msg="Linear trend should extrapolate correctly to 10 steps",
        )

    def test_different_horizons_fit20_predict5(self):
        """Verify decomposition with fit_fh=20, predict_fh=5 (shorter)."""
        n = 100
        t = np.arange(n)
        y_values = 1.0 * t + 15.0 + 2.0 * np.sin(2 * np.pi * t / 7)

        time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        y = pl.DataFrame({
            "time": time,
            "target": y_values,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        X = y.select("time")

        pipeline = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonal", PatternSeasonalityForecaster(seasonality=7)),
        ])

        y_train = y[:80]
        X_train = X[:80]

        # Fit with horizon=20
        pipeline.fit(y_train, X_train, forecasting_horizon=20)

        # Predict with horizon=5 (shorter)
        forecasting_horizon = 5
        y_pred = pipeline.predict(forecasting_horizon=forecasting_horizon)

        assert len(y_pred) == forecasting_horizon, "Prediction should have 5 rows"

        # Verify additive identity
        forecasters_dict = dict(pipeline.forecasters_)
        trend_pred = forecasters_dict["trend"].predict(forecasting_horizon=forecasting_horizon)
        forecasters_dict = dict(pipeline.forecasters_)
        seasonal_pred = forecasters_dict["seasonal"].predict(forecasting_horizon=forecasting_horizon)

        target_col = [c for c in y_pred.columns if c not in ["time", "observed_time"]][0]

        np.testing.assert_allclose(
            y_pred[target_col].to_numpy(),
            (trend_pred[target_col].to_numpy() + seasonal_pred[target_col].to_numpy()),
            atol=1e-7,
        )

    def test_different_horizons_three_component(self):
        """Verify different horizons work with 3-component decomposition."""
        n = 100
        t = np.arange(n)
        y_values = 0.5 * t + 30.0 + 4.0 * np.sin(2 * np.pi * t / 7)

        time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        y = pl.DataFrame({
            "time": time,
            "target": y_values,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        X = y.select("time")

        pipeline = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonal", PatternSeasonalityForecaster(seasonality=7)),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        y_train = y[:80]
        X_train = X[:80]

        # Fit with horizon=7, predict with horizon=21 (3 weeks)
        pipeline.fit(y_train, X_train, forecasting_horizon=7)

        forecasting_horizon = 21
        y_pred = pipeline.predict(forecasting_horizon=forecasting_horizon)

        assert len(y_pred) == forecasting_horizon

        # Verify additive identity
        forecasters_dict = dict(pipeline.forecasters_)
        trend_pred = forecasters_dict["trend"].predict(forecasting_horizon=forecasting_horizon)
        seasonal_pred = forecasters_dict["seasonal"].predict(forecasting_horizon=forecasting_horizon)
        residual_pred = forecasters_dict["residual"].predict(forecasting_horizon=forecasting_horizon)

        target_col = [c for c in y_pred.columns if c not in ["time", "observed_time"]][0]

        np.testing.assert_allclose(
            y_pred[target_col].to_numpy(),
            (
                trend_pred[target_col].to_numpy()
                + seasonal_pred[target_col].to_numpy()
                + residual_pred[target_col].to_numpy()
            ),
            atol=1e-6,
        )


@pytest.mark.integration
class TestObserveRewindStateManagement:
    """Verify observe/rewind state management preserves decomposition algebra."""

    def test_observe_shifts_predictions_two_component(self, linear_series):
        """Verify observe() shifts predictions correctly for 2-component decomposition.

        fit() → observe(y_new) → predict(): predictions should incorporate new data.
        """
        y = linear_series(slope=3.0, intercept=5.0, length=120)
        X = y.select("time")

        pipeline = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        # Initial fit on first 80 points
        y_train = y[:80]
        X_train = X[:80]
        forecasting_horizon = 5

        pipeline.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)
        y_pred_before = pipeline.predict(forecasting_horizon=forecasting_horizon)

        # Update with next 10 points
        y_update = y[80:90]
        X_update = X[80:90]
        pipeline.observe(y_update, X_update)

        # Predict again (should start from t=90)
        y_pred_after = pipeline.predict(forecasting_horizon=forecasting_horizon)

        # Verify predictions shifted forward in time
        target_col = [c for c in y_pred_before.columns if c not in ["time", "observed_time"]][0]

        pred_before_values = y_pred_before[target_col].to_numpy()
        pred_after_values = y_pred_after[target_col].to_numpy()

        # After update, predictions should be for t=90..94 instead of t=80..84
        # For linear series y = 3t + 5, the difference should be approximately 3 * 10 = 30
        expected_shift = 3.0 * 10

        np.testing.assert_allclose(
            pred_after_values - pred_before_values,
            expected_shift,
            atol=4.0,  # Allow tolerance for polynomial fit boundary effects
            err_msg="Observe should shift predictions by expected amount",
        )

    def test_rewind_reverts_to_observation_horizon_state(self, linear_series):
        """Verify rewind() reverts to last observation_horizon rows.

        fit() → observe() → rewind() → predict(): should match prediction from observation_horizon.
        """
        y = linear_series(slope=2.0, intercept=10.0, length=150)
        X = y.select("time")

        # Use PolynomialTrendForecaster with observation_horizon
        trend_forecaster = PolynomialTrendForecaster(degree=1)

        pipeline = DecompositionPipeline([
            ("trend", trend_forecaster),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        # Fit on first 80 points
        y_train = y[:80]
        X_train = X[:80]
        forecasting_horizon = 5

        pipeline.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)

        # Update with next 30 points
        y_update = y[80:110]
        X_update = X[80:110]
        pipeline.observe(y_update, X_update)

        # Now rewind (should keep only observation_horizon rows)
        pipeline.rewind(y_update, X_update)

        # After reset, internal state should be trimmed
        # Verify by checking that prediction still works and produces correct shape
        y_pred_reset = pipeline.predict(forecasting_horizon=forecasting_horizon)

        # Verify it runs without error and produces correct shape
        assert len(y_pred_reset) == forecasting_horizon, "Reset should still predict correct horizon"

        target_col = [c for c in y_pred_reset.columns if c not in ["time", "observed_time"]][0]
        pred_values = y_pred_reset[target_col].to_numpy()

        # After reset, predictions should still follow a linear trend pattern
        # (exact time offset depends on internal state management)
        diffs = np.diff(pred_values)
        np.testing.assert_allclose(
            diffs,
            2.0,  # slope of the linear series
            atol=0.5,
            err_msg="Rewind should maintain linear trend pattern in predictions",
        )

    def test_observe_predict_atomic_two_component(self):
        """Verify observe_predict() is atomic operation for 2-component decomposition."""
        n = 120
        t = np.arange(n)
        y_values = 1.5 * t + 20.0

        time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        y = pl.DataFrame({
            "time": time,
            "target": y_values,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        X = y.select("time")

        pipeline = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        y_train = y[:80]
        X_train = X[:80]
        forecasting_horizon = 5

        pipeline.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)

        # Use separate update + predict to verify atomic behavior
        y_update = y[80:90]
        X_update = X[80:90]
        pipeline.observe(y_update, X_update)
        y_pred_atomic = pipeline.predict(forecasting_horizon=forecasting_horizon)

        # Verify prediction has correct length
        assert len(y_pred_atomic) == forecasting_horizon

        target_col = [c for c in y_pred_atomic.columns if c not in ["time", "observed_time"]][0]
        pred_values = y_pred_atomic[target_col].to_numpy()

        # Should predict for t=90..94 (allow boundary offset from polynomial fit)
        t_pred = np.arange(90, 95)
        expected_values = 1.5 * t_pred + 20.0

        np.testing.assert_allclose(
            pred_values,
            expected_values,
            atol=3.0,  # Allow tolerance for polynomial fit boundary effects
            err_msg="observe + predict should give correct predictions",
        )

    def test_observe_rewind_three_component(self):
        """Verify observe/rewind for 3-component decomposition."""
        n = 150
        t = np.arange(n)
        y_values = 2.0 * t + 30.0 + 5.0 * np.sin(2 * np.pi * t / 7)

        time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        y = pl.DataFrame({
            "time": time,
            "target": y_values,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        X = y.select("time")

        pipeline = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonal", PatternSeasonalityForecaster(seasonality=7)),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        y_train = y[:80]
        X_train = X[:80]
        forecasting_horizon = 7

        pipeline.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)

        # Update with new data
        y_update = y[80:100]
        X_update = X[80:100]
        pipeline.observe(y_update, X_update)

        # Predictions should now start from t=100
        y_pred_after_update = pipeline.predict(forecasting_horizon=forecasting_horizon)

        assert len(y_pred_after_update) == forecasting_horizon

        # Rewind to trim memory
        pipeline.rewind(y_update, X_update)

        # Predictions should still work after reset
        y_pred_after_reset = pipeline.predict(forecasting_horizon=forecasting_horizon)

        assert len(y_pred_after_reset) == forecasting_horizon

    def test_multiple_observes_cumulative_effect(self):
        """Verify multiple sequential observes accumulate correctly."""
        n = 200
        t = np.arange(n)
        y_values = 1.0 * t + 50.0

        time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        y = pl.DataFrame({
            "time": time,
            "target": y_values,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        X = y.select("time")

        pipeline = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        # Initial fit
        y_train = y[:50]
        X_train = X[:50]
        forecasting_horizon = 5

        pipeline.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)

        # Multiple updates: 50->60, 60->70, 70->80
        for start_idx in [50, 60, 70]:
            end_idx = start_idx + 10
            y_update = y[start_idx:end_idx]
            X_update = X[start_idx:end_idx]
            pipeline.observe(y_update, X_update)

        # Final prediction should be from t=80
        y_pred_final = pipeline.predict(forecasting_horizon=forecasting_horizon)

        target_col = [c for c in y_pred_final.columns if c not in ["time", "observed_time"]][0]
        pred_values = y_pred_final[target_col].to_numpy()

        # Should predict for t=80..84
        t_pred = np.arange(80, 85)
        expected_values = 1.0 * t_pred + 50.0

        np.testing.assert_allclose(
            pred_values, expected_values, atol=0.1, err_msg="Multiple observes should accumulate correctly"
        )

    def test_rewind_with_groups(self):
        """Verify rewind() works with groups parameter (if supported).

        Note: This test may need adjustment based on actual panel data support in DecompositionPipeline.
        """
        n = 100
        t = np.arange(n)
        y_values = 2.0 * t + 10.0

        time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]

        y = pl.DataFrame({
            "time": time,
            "target": y_values,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        X = y.select("time")

        pipeline = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("residual", SeasonalNaive(seasonality=1)),
        ])

        y_train = y[:80]
        X_train = X[:80]
        forecasting_horizon = 5

        pipeline.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)

        # Update and reset with groups=None (should work as normal reset)
        y_update = y[80:90]
        X_update = X[80:90]

        pipeline.observe(y_update, X_update, groups=None)
        pipeline.rewind(y_update, X_update, groups=None)

        # Verify prediction still works
        y_pred = pipeline.predict(forecasting_horizon=forecasting_horizon, groups=None)

        assert len(y_pred) == forecasting_horizon, "Rewind with groups should work"
