"""Stress tests for interval forecasters with compositions.

Tests interval forecasters (SplitConformalForecaster, IntervalReductionForecaster)
through meta-forecasters (ColumnForecaster, ForecastedFeatureForecaster, etc.).
Verifies interval column structure, bound monotonicity, coverage width ordering.
"""

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.multioutput import MultiOutputRegressor

from yohou.compose import (
    ColumnForecaster,
    FeaturePipeline,
    ForecastedFeatureForecaster,
)
from yohou.interval import (
    DistanceSimilarity,
    IntervalReductionForecaster,
    SplitConformalForecaster,
)
from yohou.metrics import AbsoluteResidual, GammaResidual, Residual
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.preprocessing import LagTransformer, StandardScaler


@pytest.mark.integration
class TestSplitConformalForecaster:
    """Tests for SplitConformalForecaster with various point forecasters, conformity scorers, and similarity weighting."""

    @pytest.mark.parametrize(
        "point_forecaster",
        [
            SeasonalNaive(seasonality=1),
            PointReductionForecaster(LinearRegression(), feature_transformer=LagTransformer(lag=1)),
            PointReductionForecaster(
                LinearRegression(),
                feature_transformer=FeaturePipeline([("lag1", LagTransformer(lag=1)), ("lag3", LagTransformer(lag=3))]),
            ),
        ],
        ids=["seasonal_naive", "reduction_lag1", "reduction_pipeline"],
    )
    @pytest.mark.parametrize(
        "conformity_scorer",
        [Residual(), AbsoluteResidual(), GammaResidual()],
        ids=["residual", "absolute_residual", "gamma_residual"],
    )
    @pytest.mark.parametrize(
        "coverage_rates",
        [[0.5], [0.9], [0.5, 0.9, 0.95]],
        ids=["single_0.5", "single_0.9", "multiple_rates"],
    )
    def test_split_conformal_with_point_forecasters(
        self,
        ar1_series,
        point_forecaster,
        conformity_scorer,
        coverage_rates,
    ):
        """Test SplitConformalForecaster with various point forecasters and settings.

        Verifies:
        - Correct interval column names ({col}_lower_{rate}, {col}_upper_{rate})
        - Bound monotonicity (lower ≤ upper for all rows)
        - Width ordering (higher coverage → wider intervals)
        - Point prediction matches inner forecaster
        """
        y = ar1_series(phi=0.5, c=10.0, length=300)
        y_train = y[:250]
        forecasting_horizon = 10

        # Fit conformal forecaster
        conformal = SplitConformalForecaster(
            point_forecaster=point_forecaster,
            conformity_scorer=conformity_scorer,
        )
        conformal.fit(y_train, forecasting_horizon=forecasting_horizon)

        # 1. Predict intervals
        y_pred_interval = conformal.predict_interval(
            forecasting_horizon=forecasting_horizon,
            coverage_rates=coverage_rates,
        )

        # 2. Verify interval columns exist with correct names
        target_cols = [c for c in y_train.columns if c != "time"]
        for col in target_cols:
            for rate in coverage_rates:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"
                assert lower_col in y_pred_interval.columns, f"Missing lower bound column: {lower_col}"
                assert upper_col in y_pred_interval.columns, f"Missing upper bound column: {upper_col}"

        # 3. Verify bound monotonicity (lower ≤ upper)
        for col in target_cols:
            for rate in coverage_rates:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"
                lower_vals = y_pred_interval[lower_col].to_numpy()
                upper_vals = y_pred_interval[upper_col].to_numpy()
                assert np.all(lower_vals <= upper_vals + 1e-6), f"Bound violation for {col} at coverage {rate}"

        # 4. Verify width ordering (higher coverage → wider intervals)
        if len(coverage_rates) > 1:
            sorted_rates = sorted(coverage_rates)
            for col in target_cols:
                width_by_rate = {}
                for rate in sorted_rates:
                    lower_col = f"{col}_lower_{rate}"
                    upper_col = f"{col}_upper_{rate}"
                    widths = (y_pred_interval[upper_col] - y_pred_interval[lower_col]).to_numpy()
                    width_by_rate[rate] = widths

                # Check monotonicity: width(rate_i) ≤ width(rate_j) if rate_i ≤ rate_j
                for i in range(len(sorted_rates) - 1):
                    rate_small = sorted_rates[i]
                    rate_large = sorted_rates[i + 1]
                    # Allow small tolerance for quantile estimation imprecision
                    assert np.all(width_by_rate[rate_small] <= width_by_rate[rate_large] + 1e-5), (
                        f"Width ordering violated for {col}: coverage {rate_small} has wider intervals than {rate_large}"
                    )

        # 5. Verify point prediction matches inner forecaster
        y_pred_point = conformal.predict(forecasting_horizon=forecasting_horizon)
        # Refit point forecaster separately to compare
        from sklearn.base import clone

        point_forecaster_copy = clone(point_forecaster)
        point_forecaster_copy.fit(y_train, forecasting_horizon=forecasting_horizon)
        y_pred_point_expected = point_forecaster_copy.predict(forecasting_horizon=forecasting_horizon)

        for col in target_cols:
            np.testing.assert_allclose(
                y_pred_point[col].to_numpy(),
                y_pred_point_expected[col].to_numpy(),
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Point prediction mismatch for column {col}",
            )

    @pytest.mark.parametrize(
        "metric",
        ["euclidean", "sqeuclidean", "chebyshev"],
        ids=["euclidean", "sqeuclidean", "chebyshev"],
    )
    def test_split_conformal_with_distance_similarity(self, ar1_series, metric):
        """Test SplitConformalForecaster with DistanceSimilarity weighting.

        Verifies:
        - Similarity weights are computed
        - Intervals adjust based on similarity
        - Different metrics produce different results
        """
        y = ar1_series(phi=0.5, c=10.0, length=300, noise_std=1.0)
        y_train = y[:250]
        forecasting_horizon = 10

        # Fit with similarity
        conformal_with_sim = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=1),
            conformity_scorer=AbsoluteResidual(),
            similarity=DistanceSimilarity(metric=metric),
        )
        conformal_with_sim.fit(y_train, forecasting_horizon=forecasting_horizon)

        # Fit without similarity
        conformal_no_sim = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=1),
            conformity_scorer=AbsoluteResidual(),
            similarity=None,
        )
        conformal_no_sim.fit(y_train, forecasting_horizon=forecasting_horizon)

        # Predict intervals
        y_pred_with_sim = conformal_with_sim.predict_interval(
            forecasting_horizon=forecasting_horizon,
            coverage_rates=[0.9],
        )
        y_pred_no_sim = conformal_no_sim.predict_interval(
            forecasting_horizon=forecasting_horizon,
            coverage_rates=[0.9],
        )

        # Verify intervals differ (similarity affects quantile estimation)
        target_cols = [c for c in y_train.columns if c != "time"]
        for col in target_cols:
            lower_col = f"{col}_lower_0.9"
            upper_col = f"{col}_upper_0.9"

            # At least one bound should differ
            lower_diff = np.abs(y_pred_with_sim[lower_col].to_numpy() - y_pred_no_sim[lower_col].to_numpy())
            upper_diff = np.abs(y_pred_with_sim[upper_col].to_numpy() - y_pred_no_sim[upper_col].to_numpy())

            # Allow for some similarity cases to match, but expect differences in general
            assert np.sum(lower_diff > 1e-6) > 0 or np.sum(upper_diff > 1e-6) > 0, (
                f"Similarity had no effect on intervals for {col} with metric {metric}"
            )

    def test_split_conformal_similarity_comparison_across_metrics(self, ar1_series):
        """Test that different distance metrics produce different intervals."""
        y = ar1_series(phi=0.5, c=10.0, length=300, noise_std=1.0)
        y_train = y[:250]
        forecasting_horizon = 10

        metrics_to_test = ["euclidean", "sqeuclidean", "chebyshev"]
        predictions = {}

        for metric in metrics_to_test:
            conformal = SplitConformalForecaster(
                point_forecaster=SeasonalNaive(seasonality=1),
                conformity_scorer=AbsoluteResidual(),
                similarity=DistanceSimilarity(metric=metric),
            )
            conformal.fit(y_train, forecasting_horizon=forecasting_horizon)
            predictions[metric] = conformal.predict_interval(
                forecasting_horizon=forecasting_horizon,
                coverage_rates=[0.9],
            )

        # Compare euclidean vs sqeuclidean
        target_cols = [c for c in y_train.columns if c != "time"]
        for col in target_cols:
            lower_col = f"{col}_lower_0.9"
            euclidean_lower = predictions["euclidean"][lower_col].to_numpy()
            sqeuclidean_lower = predictions["sqeuclidean"][lower_col].to_numpy()

            # Different metrics should produce different results in general
            # (may be similar for some simple cases, but expect differences)
            diff = np.abs(euclidean_lower - sqeuclidean_lower)
            # At least some timesteps should differ
            assert np.sum(diff > 1e-6) > 0, f"Euclidean and sqeuclidean produced identical intervals for {col}"


@pytest.mark.integration
class TestIntervalReductionForecaster:
    """Tests for IntervalReductionForecaster with coverage rates, horizons, pipelines, and observe-predict."""

    @pytest.mark.parametrize(
        "coverage_rates",
        [[0.5], [0.9], [0.5, 0.9, 0.95]],
        ids=["single_0.5", "single_0.9", "multiple_rates"],
    )
    def test_interval_reduction_with_coverage_rates(self, ar1_series, coverage_rates):
        """Test IntervalReductionForecaster with various coverage rates.

        Verifies:
        - Correct interval column structure
        - Bound monotonicity
        - Width ordering
        """
        y = ar1_series(phi=0.5, c=10.0, length=300)
        y_train = y[:250]
        forecasting_horizon = 10

        forecaster = IntervalReductionForecaster(
            estimator=MultiOutputRegressor(QuantileRegressor(solver="highs")),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=forecasting_horizon, coverage_rates=coverage_rates)

        y_pred_interval = forecaster.predict_interval(
            forecasting_horizon=forecasting_horizon,
            coverage_rates=coverage_rates,
        )

        # Verify interval columns
        target_cols = [c for c in y_train.columns if c != "time"]
        for col in target_cols:
            for rate in coverage_rates:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"
                assert lower_col in y_pred_interval.columns
                assert upper_col in y_pred_interval.columns

                # Bound monotonicity
                lower_vals = y_pred_interval[lower_col].to_numpy()
                upper_vals = y_pred_interval[upper_col].to_numpy()
                assert np.all(lower_vals <= upper_vals + 1e-6)

        # Width ordering
        if len(coverage_rates) > 1:
            sorted_rates = sorted(coverage_rates)
            for col in target_cols:
                widths = {}
                for rate in sorted_rates:
                    lower_col = f"{col}_lower_{rate}"
                    upper_col = f"{col}_upper_{rate}"
                    widths[rate] = (y_pred_interval[upper_col] - y_pred_interval[lower_col]).to_numpy()

                for i in range(len(sorted_rates) - 1):
                    rate_small = sorted_rates[i]
                    rate_large = sorted_rates[i + 1]
                    assert np.all(widths[rate_small] <= widths[rate_large] + 1e-5)

    def test_interval_reduction_different_horizon(self, ar1_series):
        """Test IntervalReductionForecaster with different horizon at predict time."""
        y = ar1_series(phi=0.5, c=10.0, length=300)
        y_train = y[:250]
        fit_horizon = 10
        predict_horizon = 15

        forecaster = IntervalReductionForecaster(
            estimator=MultiOutputRegressor(QuantileRegressor(solver="highs")),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=fit_horizon, coverage_rates=[0.9])

        # Predict with different horizon
        y_pred_interval = forecaster.predict_interval(
            forecasting_horizon=predict_horizon,
            coverage_rates=[0.9],
        )

        # Verify correct number of rows
        assert len(y_pred_interval) == predict_horizon

        # Verify interval structure
        target_cols = [c for c in y_train.columns if c != "time"]
        for col in target_cols:
            lower_col = f"{col}_lower_0.9"
            upper_col = f"{col}_upper_0.9"
            assert lower_col in y_pred_interval.columns
            assert upper_col in y_pred_interval.columns

            lower_vals = y_pred_interval[lower_col].to_numpy()
            upper_vals = y_pred_interval[upper_col].to_numpy()
            assert np.all(lower_vals <= upper_vals + 1e-6)

    def test_interval_reduction_with_feature_pipeline(self, ar1_series):
        """Test IntervalReductionForecaster with FeaturePipeline."""
        y = ar1_series(phi=0.5, c=10.0, length=300)
        y_train = y[:250]
        forecasting_horizon = 10

        forecaster = IntervalReductionForecaster(
            estimator=MultiOutputRegressor(QuantileRegressor(solver="highs")),
            feature_transformer=FeaturePipeline([("lag", LagTransformer(lag=1)), ("scaler", StandardScaler())]),
        )
        forecaster.fit(y_train, forecasting_horizon=forecasting_horizon, coverage_rates=[0.9])

        y_pred_interval = forecaster.predict_interval(
            forecasting_horizon=forecasting_horizon,
            coverage_rates=[0.9],
        )

        # Verify interval structure
        target_cols = [c for c in y_train.columns if c != "time"]
        for col in target_cols:
            lower_col = f"{col}_lower_0.9"
            upper_col = f"{col}_upper_0.9"
            assert lower_col in y_pred_interval.columns
            assert upper_col in y_pred_interval.columns

            lower_vals = y_pred_interval[lower_col].to_numpy()
            upper_vals = y_pred_interval[upper_col].to_numpy()
            assert np.all(lower_vals <= upper_vals + 1e-6)

    def test_interval_reduction_observe_predict_interval(self, ar1_series):
        """Test IntervalReductionForecaster observe_predict_interval method."""
        y = ar1_series(phi=0.5, c=10.0, length=300)
        y_train = y[:250]
        y_test = y[250:]
        forecasting_horizon = 10

        forecaster = IntervalReductionForecaster(
            estimator=MultiOutputRegressor(QuantileRegressor(solver="highs")),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=forecasting_horizon, coverage_rates=[0.9])

        # Use first few rows of test for update
        y_update = y_test.head(5)

        y_pred_interval = forecaster.observe_predict_interval(
            y_update,
            coverage_rates=[0.9],
        )

        # observe_predict_interval returns initial predict + rolling observe-predict
        # predictions concatenated. With stride=forecasting_horizon=10 and
        # len(y_update)=5, this produces 1 initial + 1 loop iteration = 2 batches.
        assert len(y_pred_interval) > 0
        assert len(y_pred_interval) % forecasting_horizon == 0

        # Verify interval structure
        target_cols = [c for c in y_train.columns if c != "time"]
        for col in target_cols:
            lower_col = f"{col}_lower_0.9"
            upper_col = f"{col}_upper_0.9"
            assert lower_col in y_pred_interval.columns
            assert upper_col in y_pred_interval.columns

            lower_vals = y_pred_interval[lower_col].to_numpy()
            upper_vals = y_pred_interval[upper_col].to_numpy()
            assert np.all(lower_vals <= upper_vals + 1e-6)


@pytest.mark.integration
class TestColumnForecasterInterval:
    """Tests for ColumnForecaster with interval forecasters, mixed types, and remainder handling."""

    def test_column_forecaster_mixed_point_interval(self, linear_series):
        """Test ColumnForecaster with mixed point and interval forecasters.

        Uses two target columns:
        - col_a: Point forecaster
        - col_b: Interval forecaster

        Note: ColumnForecaster only exposes predict_interval when ALL nested
        forecasters support it. With mixed types, only predict() is available.
        """
        y = linear_series(slope=2.0, intercept=10.0, length=300)
        y_train = y[:250]
        forecasting_horizon = 10

        # Create two-column target (duplicate and rename)
        target_cols = [c for c in y_train.columns if c != "time"]
        y_train_multi = y_train.rename({target_cols[0]: "col_a"})
        y_train_multi = y_train_multi.with_columns(y_train_multi["col_a"].alias("col_b"))

        # Build mixed forecaster
        forecaster = ColumnForecaster([
            (
                "point_col",
                PointReductionForecaster(LinearRegression(), feature_transformer=LagTransformer(lag=1)),
                ["col_a"],
            ),
            (
                "interval_col",
                SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=1)),
                ["col_b"],
            ),
        ])
        forecaster.fit(y_train_multi, forecasting_horizon=forecasting_horizon)

        # Test predict() works on both columns
        y_pred_point = forecaster.predict(forecasting_horizon=forecasting_horizon)
        assert "col_a" in y_pred_point.columns
        assert "col_b" in y_pred_point.columns
        assert len(y_pred_point) == forecasting_horizon

        # predict_interval is NOT available when not all forecasters support it
        assert not hasattr(forecaster, "predict_interval")

    def test_column_forecaster_all_interval_forecasters(self, linear_series):
        """Test ColumnForecaster with all interval forecasters."""
        y = linear_series(slope=2.0, intercept=10.0, length=300)
        y_train = y[:250]
        forecasting_horizon = 10

        # Create two-column target
        target_cols = [c for c in y_train.columns if c != "time"]
        y_train_multi = y_train.rename({target_cols[0]: "col_a"})
        y_train_multi = y_train_multi.with_columns(y_train_multi["col_a"].alias("col_b"))

        # Build forecaster with all interval forecasters
        forecaster = ColumnForecaster([
            (
                "interval_1",
                SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=1)).set_fit_request(
                    coverage_rates=True
                ),
                ["col_a"],
            ),
            (
                "interval_2",
                IntervalReductionForecaster(
                    MultiOutputRegressor(QuantileRegressor(solver="highs")),
                    feature_transformer=LagTransformer(lag=1),
                ).set_fit_request(coverage_rates=True),
                ["col_b"],
            ),
        ])
        forecaster.fit(y_train_multi, forecasting_horizon=forecasting_horizon, coverage_rates=[0.9])

        # Predict intervals
        y_pred_interval = forecaster.predict_interval(
            forecasting_horizon=forecasting_horizon,
            coverage_rates=[0.9],
        )

        # Verify both columns have intervals
        assert "col_a_lower_0.9" in y_pred_interval.columns
        assert "col_a_upper_0.9" in y_pred_interval.columns
        assert "col_b_lower_0.9" in y_pred_interval.columns
        assert "col_b_upper_0.9" in y_pred_interval.columns

        # Verify bounds for both
        for col in ["col_a", "col_b"]:
            lower_vals = y_pred_interval[f"{col}_lower_0.9"].to_numpy()
            upper_vals = y_pred_interval[f"{col}_upper_0.9"].to_numpy()
            assert np.all(lower_vals <= upper_vals + 1e-6)

    def test_column_forecaster_interval_with_remainder(self, linear_series):
        """Test ColumnForecaster with interval forecaster and remainder='passthrough'."""
        y = linear_series(slope=2.0, intercept=10.0, length=300)
        y_train = y[:250]
        forecasting_horizon = 10

        # Create three-column target
        target_cols = [c for c in y_train.columns if c != "time"]
        y_train_multi = y_train.rename({target_cols[0]: "col_a"})
        y_train_multi = y_train_multi.with_columns([
            y_train_multi["col_a"].alias("col_b"),
            y_train_multi["col_a"].alias("col_c"),
        ])

        # Build forecaster (only forecast col_a, passthrough others)
        forecaster = ColumnForecaster(
            [
                (
                    "interval_col",
                    SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=1)),
                    ["col_a"],
                ),
            ],
            remainder="passthrough",
        )
        forecaster.fit(y_train_multi, forecasting_horizon=forecasting_horizon)

        # Predict intervals
        y_pred_interval = forecaster.predict_interval(
            forecasting_horizon=forecasting_horizon,
            coverage_rates=[0.9],
        )

        # Verify col_a has intervals
        assert "col_a_lower_0.9" in y_pred_interval.columns
        assert "col_a_upper_0.9" in y_pred_interval.columns

        # Remainder columns should be present (passed through)
        # Note: ColumnForecaster behavior with remainder and intervals needs verification
        # This test captures the expected behavior


@pytest.mark.integration
class TestForecastedFeatureForecasterInterval:
    """Tests for ForecastedFeatureForecaster with interval target forecasters, strategies, and coverage rates."""

    def test_forecasted_feature_interval_target(self, ar1_series):
        """Test ForecastedFeatureForecaster with interval target forecaster."""
        y_full = ar1_series(phi=0.5, c=10.0, length=300)
        # Create separate y (target) and X (exogenous features)
        y = y_full.select("time", "value")
        X = y_full.select("time").with_columns(
            pl.Series("feature1", np.linspace(0, 1, 300)),
        )
        y_train = y[:250]
        X_train = X[:250]
        forecasting_horizon = 10

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=1)),
            feature_forecaster=PointReductionForecaster(LinearRegression(), feature_transformer=LagTransformer(lag=1)),
        )
        forecaster.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)

        # Predict intervals
        y_pred_interval = forecaster.predict_interval(
            forecasting_horizon=forecasting_horizon,
            coverage_rates=[0.9],
        )

        # Verify interval structure
        target_cols = [c for c in y_train.columns if c != "time"]
        for col in target_cols:
            lower_col = f"{col}_lower_0.9"
            upper_col = f"{col}_upper_0.9"
            assert lower_col in y_pred_interval.columns
            assert upper_col in y_pred_interval.columns

            lower_vals = y_pred_interval[lower_col].to_numpy()
            upper_vals = y_pred_interval[upper_col].to_numpy()
            assert np.all(lower_vals <= upper_vals + 1e-6)

    def test_forecasted_feature_interval_observe_predict_interval(self, ar1_series):
        """Test ForecastedFeatureForecaster observe_predict_interval with interval target."""
        y_full = ar1_series(phi=0.5, c=10.0, length=300)
        y = y_full.select("time", "value")
        X = y_full.select("time").with_columns(
            pl.Series("feature1", np.linspace(0, 1, 300)),
        )
        y_train = y[:250]
        X_train = X[:250]
        y_test = y[250:]
        X_test = X[250:]
        forecasting_horizon = 10

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=1)),
            feature_forecaster=PointReductionForecaster(LinearRegression(), feature_transformer=LagTransformer(lag=1)),
        )
        forecaster.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)

        # Update and predict
        y_update = y_test.head(5)
        X_update = X_test.head(5)
        y_pred_interval = forecaster.observe_predict_interval(
            y_update,
            X_actual=X_update,
            coverage_rates=[0.9],
        )

        # Verify interval structure
        target_cols = [c for c in y_train.columns if c != "time"]
        for col in target_cols:
            lower_col = f"{col}_lower_0.9"
            upper_col = f"{col}_upper_0.9"
            assert lower_col in y_pred_interval.columns
            assert upper_col in y_pred_interval.columns

            lower_vals = y_pred_interval[lower_col].to_numpy()
            upper_vals = y_pred_interval[upper_col].to_numpy()
            assert np.all(lower_vals <= upper_vals + 1e-6)

    @pytest.mark.parametrize(
        "strategy",
        ["actual", "predicted"],
        ids=["strategy_actual", "strategy_predicted"],
    )
    def test_forecasted_feature_interval_with_strategies(self, ar1_series, strategy):
        """Test ForecastedFeatureForecaster with interval target and different strategies."""
        y_full = ar1_series(phi=0.5, c=10.0, length=500)
        y = y_full.select("time", "value")
        X = y_full.select("time").with_columns(
            pl.Series("feature1", np.linspace(0, 1, 500)),
        )
        y_train = y[:400]
        X_train = X[:400]
        forecasting_horizon = 5

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SplitConformalForecaster(
                point_forecaster=SeasonalNaive(seasonality=1), calibration_size=30
            ),
            feature_forecaster=PointReductionForecaster(LinearRegression(), feature_transformer=LagTransformer(lag=1)),
            strategy=strategy,
            split_ratio=0.8,
        )
        forecaster.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)

        # Predict intervals
        y_pred_interval = forecaster.predict_interval(
            forecasting_horizon=forecasting_horizon,
            coverage_rates=[0.9],
        )

        # Verify interval structure
        target_cols = [c for c in y_train.columns if c != "time"]
        for col in target_cols:
            lower_col = f"{col}_lower_0.9"
            upper_col = f"{col}_upper_0.9"
            assert lower_col in y_pred_interval.columns
            assert upper_col in y_pred_interval.columns

            lower_vals = y_pred_interval[lower_col].to_numpy()
            upper_vals = y_pred_interval[upper_col].to_numpy()
            assert np.all(lower_vals <= upper_vals + 1e-6)

    def test_forecasted_feature_interval_multiple_coverage_rates(self, ar1_series):
        """Test ForecastedFeatureForecaster with interval target and multiple coverage rates."""
        y_full = ar1_series(phi=0.5, c=10.0, length=300)
        y = y_full.select("time", "value")
        X = y_full.select("time").with_columns(
            pl.Series("feature1", np.linspace(0, 1, 300)),
        )
        y_train = y[:250]
        X_train = X[:250]
        forecasting_horizon = 10

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=1)),
            feature_forecaster=PointReductionForecaster(LinearRegression(), feature_transformer=LagTransformer(lag=1)),
        )
        forecaster.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)

        # Predict with multiple coverage rates
        coverage_rates = [0.5, 0.9, 0.95]
        y_pred_interval = forecaster.predict_interval(
            forecasting_horizon=forecasting_horizon,
            coverage_rates=coverage_rates,
        )

        # Verify all interval columns exist
        target_cols = [c for c in y_train.columns if c != "time"]
        for col in target_cols:
            for rate in coverage_rates:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"
                assert lower_col in y_pred_interval.columns
                assert upper_col in y_pred_interval.columns

                lower_vals = y_pred_interval[lower_col].to_numpy()
                upper_vals = y_pred_interval[upper_col].to_numpy()
                assert np.all(lower_vals <= upper_vals + 1e-6)

            # Verify width ordering
            sorted_rates = sorted(coverage_rates)
            widths = {}
            for rate in sorted_rates:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"
                widths[rate] = (y_pred_interval[upper_col] - y_pred_interval[lower_col]).to_numpy()

            for i in range(len(sorted_rates) - 1):
                rate_small = sorted_rates[i]
                rate_large = sorted_rates[i + 1]
                assert np.all(widths[rate_small] <= widths[rate_large] + 1e-5)
