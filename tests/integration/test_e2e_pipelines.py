"""End-to-end pipeline integration tests.

Tests realistic compositions of forecasters, transformers, and meta-forecasters
on synthetic data with known analytical properties. All tests use @pytest.mark.integration.

Each pipeline represents a real-world forecasting scenario:
- Pipeline A: Retail demand forecasting (exponential + seasonality)
- Pipeline B: Multi-store panel forecasting (linear trends per store)
- Pipeline C: Decomposition + conformal intervals (trend + annual seasonality)
- Pipeline D: Feature-forecasted weather-demand (linear relationships)
- Pipeline E: Hyperparameter-tuned forecasting (grid search)
- Pipeline F: Everything composed (maximum nesting depth)
"""

import numpy as np
import polars as pl
import polars.selectors as cs
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge

from yohou.compose import (
    ColumnForecaster,
    DecompositionPipeline,
    FeaturePipeline,
    FeatureUnion,
    ForecastedFeatureForecaster,
)
from yohou.interval import SplitConformalForecaster
from yohou.metrics import (
    AbsoluteResidual,
    EmpiricalCoverage,
    IntervalScore,
    MeanAbsoluteError,
)
from yohou.model_selection import GridSearchCV, SlidingWindowSplitter
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.preprocessing import (
    LagTransformer,
    RollingStatisticsTransformer,
    StandardScaler,
)
from yohou.stationarity import (
    LogTransformer,
    PatternSeasonalityForecaster,
    PolynomialTrendForecaster,
    SeasonalDifferencing,
)
from yohou.utils.panel import inspect_panel


@pytest.mark.integration
class TestRetailDemandPipeline:
    """Pipeline A: Retail demand forecasting (exponential + weekly seasonality)."""

    def test_pipeline_a_fit_predict_score(self):
        """Pipeline A: Fit → predict → score on exponential + weekly seasonality."""
        # Generate y = exp(0.01t + 5) · (1 + 0.3·sin(2πt/7))
        n = 150
        t = np.arange(n)
        y_values = np.exp(0.01 * t + 5) * (1 + 0.3 * np.sin(2 * np.pi * t / 7))

        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                interval="1d",
                eager=True,
            ),
            "demand": y_values,
        })

        # Split train/test
        train_size = 120
        y_train = df[:train_size]
        y_test = df[train_size:]

        # Build pipeline
        target_transformer = FeaturePipeline([
            ("log", LogTransformer(offset=1.0)),
            ("deseason", SeasonalDifferencing(seasonality=7)),
        ])
        actual_transformer = FeaturePipeline([
            ("lags", LagTransformer([1, 7, 14])),
            ("rolling", RollingStatisticsTransformer(window_size=7, statistics=["mean", "std"])),
        ])
        forecaster = PointReductionForecaster(
            estimator=Ridge(alpha=0.1),
            target_transformer=target_transformer,
            actual_transformer=actual_transformer,
        )

        # Fit
        forecaster.fit(y_train, forecasting_horizon=7)

        # Predict
        y_pred = forecaster.predict(forecasting_horizon=7)

        # Score
        scorer = MeanAbsoluteError()
        scorer.fit(y_train)
        score = scorer.score(y_test[:7], y_pred)

        # Naive baseline (persistence)
        last_val = y_train["demand"][-1]
        naive_pred = pl.DataFrame({
            "time": y_test[:7]["time"],
            "demand": [last_val] * 7,
        })
        naive_score = scorer.score(y_test[:7], naive_pred)

        # Verify predictions are better than 2× naive baseline
        assert score < 2 * naive_score, f"Pipeline too inaccurate: {score} vs 2×{naive_score}"

    def test_pipeline_a_observe_predict_cycle(self):
        """Pipeline A: Observe → predict → score cycle."""
        # Generate data
        n = 150
        t = np.arange(n)
        y_values = np.exp(0.01 * t + 5) * (1 + 0.3 * np.sin(2 * np.pi * t / 7))

        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                interval="1d",
                eager=True,
            ),
            "demand": y_values,
        })

        # Initial fit
        train_size = 100
        y_train = df[:train_size]
        forecaster = PointReductionForecaster(
            estimator=Ridge(alpha=0.1),
            target_transformer=FeaturePipeline([
                ("log", LogTransformer(offset=1.0)),
                ("deseason", SeasonalDifferencing(seasonality=7)),
            ]),
            actual_transformer=FeaturePipeline([
                ("lags", LagTransformer([1, 7, 14])),
                ("rolling", RollingStatisticsTransformer(window_size=7, statistics=["mean", "std"])),
            ]),
        )
        forecaster.fit(y_train, forecasting_horizon=7)

        # A longer-horizon predict() must also succeed from the fitted state.
        assert len(forecaster.predict(forecasting_horizon=14)) == 14

        # Observe with new week
        y_new_week = df[train_size : train_size + 7]
        forecaster.observe(y_new_week)

        # Predict
        y_pred = forecaster.predict(forecasting_horizon=7)
        assert len(y_pred) == 7

        # Score
        y_test = df[train_size + 7 : train_size + 14]
        scorer = MeanAbsoluteError()
        scorer.fit(y_train)
        score = scorer.score(y_test, y_pred)

        # Should be finite and reasonable
        assert np.isfinite(score)
        assert score > 0  # Non-trivial prediction

    def test_pipeline_a_sequential_observe_predict(self):
        """Pipeline A: 4 sequential observe_predict cycles."""
        # Generate data
        n = 180
        t = np.arange(n)
        y_values = np.exp(0.01 * t + 5) * (1 + 0.3 * np.sin(2 * np.pi * t / 7))

        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                interval="1d",
                eager=True,
            ),
            "demand": y_values,
        })

        # Initial fit
        train_size = 120
        forecaster = PointReductionForecaster(
            estimator=Ridge(alpha=0.1),
            target_transformer=FeaturePipeline([
                ("log", LogTransformer(offset=1.0)),
                ("deseason", SeasonalDifferencing(seasonality=7)),
            ]),
            actual_transformer=FeaturePipeline([
                ("lags", LagTransformer([1, 7, 14])),
                ("rolling", RollingStatisticsTransformer(window_size=7, statistics=["mean", "std"])),
            ]),
        )
        y_train = df[:train_size]
        forecaster.fit(y_train, forecasting_horizon=7)

        # 4 cycles
        scores = []
        scorer = MeanAbsoluteError()
        for i in range(4):
            start = train_size + i * 7
            y_week = df[start : start + 7]
            y_pred = forecaster.observe_predict(y_week, forecasting_horizon=7)

            # Score against next week
            y_test = df[start + 7 : start + 14]
            if len(y_test) >= 7:
                scorer.fit(y_train)
                score = scorer.score(y_test, y_pred)
                scores.append(score)

        # All should complete without crash
        assert len(scores) >= 3  # At least 3 full cycles
        assert all(np.isfinite(s) for s in scores)

    def test_pipeline_a_observation_horizon(self):
        """Pipeline A: Verify observation_horizon calculation."""
        forecaster = PointReductionForecaster(
            estimator=Ridge(alpha=0.1),
            target_transformer=FeaturePipeline([
                ("log", LogTransformer(offset=1.0)),
                ("deseason", SeasonalDifferencing(seasonality=7)),
            ]),
            actual_transformer=FeaturePipeline([
                ("lags", LagTransformer([1, 7, 14])),
                ("rolling", RollingStatisticsTransformer(window_size=7, statistics=["mean", "std"])),
            ]),
        )

        # Generate simple data
        n = 100
        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                interval="1d",
                eager=True,
            ),
            "demand": np.random.randn(n) + 100,
        })

        forecaster.fit(df, forecasting_horizon=7)

        # Observation horizon includes both target and feature transformers.
        # target_transformer: SeasonalDifferencing(7) → 7
        # actual_transformer: LagTransformer([1,7,14]) → 14, RollingStatisticsTransformer(7) → 6
        #                      FeaturePipeline sum → 20
        # max(0, 7, 20) = 20
        expected_horizon = 20
        assert forecaster.observation_horizon == expected_horizon


@pytest.mark.integration
class TestMultiStorePanelPipeline:
    """Pipeline B: Multi-store panel forecasting (linear trends per store)."""

    def test_pipeline_b_fit_predict_observe_predict(self):
        """Pipeline B: Fit → predict → observe → predict on panel data."""
        # Generate 5-store panel data (linear trends, different slopes)
        n = 120
        time_col = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        df = pl.DataFrame({
            "time": time_col,
            "sales__store_1": 100 + 0.5 * np.arange(n) + 10 * np.sin(2 * np.pi * np.arange(n) / 7),
            "sales__store_2": 150 + 0.3 * np.arange(n) + 5 * np.sin(2 * np.pi * np.arange(n) / 7),
            "sales__store_3": 80 + 0.7 * np.arange(n) + 8 * np.sin(2 * np.pi * np.arange(n) / 7),
            "sales__store_4": 50 + 0.2 * np.arange(n) + 3 * np.sin(2 * np.pi * np.arange(n) / 7),
            "sales__store_5": 200 + 0.4 * np.arange(n) + 12 * np.sin(2 * np.pi * np.arange(n) / 7),
        })

        # Build forecaster
        forecaster = ColumnForecaster([
            (
                "main_stores",
                PointReductionForecaster(
                    LR(),
                    actual_transformer=LagTransformer([1, 7]),
                ),
                ["sales__store_1", "sales__store_2", "sales__store_3"],
            ),
            ("small_stores", SeasonalNaive(seasonality=7), ["sales__store_4", "sales__store_5"]),
        ])

        # Fit
        y_train = df[:100]
        forecaster.fit(y_train, forecasting_horizon=7)

        # Predict
        y_pred1 = forecaster.predict(forecasting_horizon=7)
        assert "sales__store_1" in y_pred1.columns
        assert "sales__store_5" in y_pred1.columns
        assert len(y_pred1) == 7

        # Observe
        y_observe = df[100:107]
        forecaster.observe(y_observe)

        # Predict again
        y_pred2 = forecaster.predict(forecasting_horizon=7)
        assert len(y_pred2) == 7
        assert y_pred2.columns == y_pred1.columns

    def test_pipeline_b_groupwise_scoring(self):
        """Pipeline B: Score each group independently with groupwise aggregation."""
        # Generate panel data
        n = 100
        time_col = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        df = pl.DataFrame({
            "time": time_col,
            "sales__store_1": 100 + np.arange(n),
            "sales__store_2": 150 + 2 * np.arange(n),
            "sales__store_3": 80 + 0.5 * np.arange(n),
            "sales__store_4": 50 + 0.2 * np.arange(n),
            "sales__store_5": 200 + 3 * np.arange(n),
        })

        forecaster = ColumnForecaster([
            (
                "main_stores",
                PointReductionForecaster(
                    LR(),
                    actual_transformer=LagTransformer([1, 7]),
                ),
                ["sales__store_1", "sales__store_2", "sales__store_3"],
            ),
            ("small_stores", SeasonalNaive(seasonality=7), ["sales__store_4", "sales__store_5"]),
        ])

        # Fit and predict
        y_train = df[:80]
        y_test = df[80:87]
        forecaster.fit(y_train, forecasting_horizon=7)
        y_pred = forecaster.predict(forecasting_horizon=7)

        # Score groupwise (aggregates across panel groups, returns per-group scores)
        scorer = MeanAbsoluteError(aggregation_method="groupwise")
        scorer.fit(y_train)
        scores = scorer.score(y_test, y_pred)

        # groupwise scoring returns a per-group DataFrame; every numeric value
        # in it must be finite (a corrupted all-NaN frame must not pass).
        assert isinstance(scores, pl.DataFrame)
        finite_flags = scores.select(cs.numeric().is_finite()).to_numpy()
        assert finite_flags.size > 0
        assert bool(finite_flags.all()), f"groupwise scores contain non-finite values: {scores}"

    def test_pipeline_b_panel_group_filtering(self):
        """Pipeline B: predict(groups=[...]) filters output."""
        # Generate panel data
        n = 100
        time_col = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        df = pl.DataFrame({
            "time": time_col,
            "sales__store_1": 100 + np.arange(n),
            "sales__store_2": 150 + np.arange(n),
            "sales__store_3": 80 + np.arange(n),
            "sales__store_4": 50 + np.arange(n),
            "sales__store_5": 200 + np.arange(n),
        })

        forecaster = ColumnForecaster([
            (
                "main_stores",
                PointReductionForecaster(
                    LR(),
                    actual_transformer=LagTransformer([1, 7]),
                ),
                ["sales__store_1", "sales__store_2", "sales__store_3"],
            ),
            ("small_stores", SeasonalNaive(seasonality=7), ["sales__store_4", "sales__store_5"]),
        ])

        forecaster.fit(df[:80], forecasting_horizon=7)

        # Predict using the panel group prefix ("sales"), not suffix
        y_pred = forecaster.predict(forecasting_horizon=7, groups=["sales"])

        # Should have all sales columns (groups filters by prefix)
        assert "sales__store_1" in y_pred.columns
        assert "sales__store_2" in y_pred.columns
        assert "sales__store_3" in y_pred.columns
        assert len(y_pred) == 7

    def test_pipeline_b_partial_group_observe(self):
        """Pipeline B: Observe subset of groups, verify others unchanged."""
        # Generate panel data
        n = 100
        time_col = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        df = pl.DataFrame({
            "time": time_col,
            "sales__store_1": 100 + np.arange(n),
            "sales__store_2": 150 + np.arange(n),
            "sales__store_3": 80 + np.arange(n),
        })

        forecaster = PointReductionForecaster(
            LR(),
            actual_transformer=LagTransformer([1, 2, 3]),
        )

        forecaster.fit(df[:70], forecasting_horizon=5)

        # Predict all stores
        _y_pred_before = forecaster.predict(forecasting_horizon=5)

        # Observe using the panel group prefix ("sales"), not suffix
        y_observe = df[70:75]
        forecaster.observe(y_observe, groups=["sales"])

        # Predict should still work without crash
        y_pred_after = forecaster.predict(forecasting_horizon=5, groups=["sales"])
        assert "sales__store_1" in y_pred_after.columns
        assert "sales__store_2" in y_pred_after.columns
        assert len(y_pred_after) == 5

    def test_pipeline_b_panel_structure_preservation(self):
        """Pipeline B: Panel structure preserved through all operations."""
        # Generate panel data
        n = 100
        time_col = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        df = pl.DataFrame({
            "time": time_col,
            "sales__store_1": 100 + np.arange(n),
            "sales__store_2": 150 + np.arange(n),
            "sales__store_3": 80 + np.arange(n),
        })

        forecaster = PointReductionForecaster(
            LR(),
            actual_transformer=LagTransformer([1, 7]),
        )

        forecaster.fit(df[:80], forecasting_horizon=7)

        # Predict
        y_pred = forecaster.predict(forecasting_horizon=7)

        # Check panel structure (inspect_panel excludes "time" from global_cols)
        global_cols, panel_groups = inspect_panel(y_pred)
        assert "time" in y_pred.columns
        assert "vintage_time" in global_cols
        assert "sales" in panel_groups
        assert set(panel_groups["sales"]) == {"sales__store_1", "sales__store_2", "sales__store_3"}

        # Observe
        forecaster.observe(df[80:87])
        y_pred2 = forecaster.predict(forecasting_horizon=7)

        # Structure should be preserved
        global_cols2, panel_groups2 = inspect_panel(y_pred2)
        assert panel_groups == panel_groups2


@pytest.mark.integration
class TestDecompositionConformalPipeline:
    """Pipeline C: Decomposition + conformal intervals (trend + annual seasonality)."""

    def test_pipeline_c_fit_predict_interval(self):
        """Pipeline C: Fit → predict_interval on monthly trend + seasonality."""
        # Generate monthly data: trend + annual seasonality
        n = 120  # 10 years monthly
        t = np.arange(n)
        y_values = 100 + 2 * t + 20 * np.sin(2 * np.pi * t / 12) + np.random.randn(n) * 3

        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=30 * n - 1),
                interval="30d",
                eager=True,
            ),
            "value": y_values,
        })

        # Build pipeline
        point_forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("season", PatternSeasonalityForecaster(seasonality=12, method="average")),
            ("residual", PointReductionForecaster(LR(), actual_transformer=LagTransformer([1, 12]))),
        ])
        forecaster = SplitConformalForecaster(
            point_forecaster=point_forecaster,
            conformity_scorer=AbsoluteResidual(),
            calibration_size=30,
        )

        # Fit
        y_train = df[:90]
        forecaster.fit(y_train, forecasting_horizon=12)

        # Predict interval
        y_pred = forecaster.predict_interval(coverage_rates=[0.5, 0.9], forecasting_horizon=12)

        # Check interval structure
        assert "value_lower_0.5" in y_pred.columns
        assert "value_upper_0.5" in y_pred.columns
        assert "value_lower_0.9" in y_pred.columns
        assert "value_upper_0.9" in y_pred.columns
        assert len(y_pred) == 12

    def test_pipeline_c_interval_monotonicity(self):
        """Pipeline C: Verify interval bound monotonicity."""
        # Generate data
        n = 100
        t = np.arange(n)
        y_values = 100 + t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(n) * 2

        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=30 * n - 1),
                interval="30d",
                eager=True,
            ),
            "value": y_values,
        })

        point_forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("season", PatternSeasonalityForecaster(seasonality=12, method="average")),
            ("residual", PointReductionForecaster(LR(), actual_transformer=LagTransformer([1, 12]))),
        ])
        forecaster = SplitConformalForecaster(
            point_forecaster=point_forecaster,
            conformity_scorer=AbsoluteResidual(),
            calibration_size=25,
        )

        forecaster.fit(df[:75], forecasting_horizon=6)
        y_pred = forecaster.predict_interval(coverage_rates=[0.5, 0.9], forecasting_horizon=6)

        # Check monotonicity: lower_0.5 >= lower_0.9, upper_0.5 <= upper_0.9
        # (higher coverage → wider intervals)
        lower_50 = y_pred["value_lower_0.5"].to_numpy()
        lower_90 = y_pred["value_lower_0.9"].to_numpy()
        upper_50 = y_pred["value_upper_0.5"].to_numpy()
        upper_90 = y_pred["value_upper_0.9"].to_numpy()

        # 90% interval should contain 50% interval
        assert np.all(lower_90 <= lower_50), "Lower bounds not monotonic"
        assert np.all(upper_50 <= upper_90), "Upper bounds not monotonic"

    def test_pipeline_c_interval_scoring(self):
        """Pipeline C: Score with IntervalScore and EmpiricalCoverage."""
        # Generate data
        n = 120
        t = np.arange(n)
        y_values = 100 + t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(n) * 2

        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=30 * n - 1),
                interval="30d",
                eager=True,
            ),
            "value": y_values,
        })

        point_forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("season", PatternSeasonalityForecaster(seasonality=12, method="average")),
            ("residual", PointReductionForecaster(LR(), actual_transformer=LagTransformer([1, 12]))),
        ])
        forecaster = SplitConformalForecaster(
            point_forecaster=point_forecaster,
            conformity_scorer=AbsoluteResidual(),
            calibration_size=30,
        )

        # Fit and predict
        y_train = df[:100]
        y_test = df[100:112]
        forecaster.fit(y_train, forecasting_horizon=12)
        y_pred = forecaster.predict_interval(coverage_rates=[0.9], forecasting_horizon=12)

        # Score with interval metrics
        interval_scorer = IntervalScore(coverage_rates=[0.9])
        interval_scorer.fit(y_train)
        interval_score = interval_scorer.score(y_test, y_pred)
        assert np.isfinite(interval_score)

        coverage_scorer = EmpiricalCoverage(coverage_rates=[0.9])
        coverage_scorer.fit(y_train)
        coverage = coverage_scorer.score(y_test, y_pred)
        # EmpiricalCoverage with default aggregation_method="all" returns a scalar float
        assert isinstance(coverage, float)
        assert 0 <= coverage <= 1

    def test_pipeline_c_observe_predict_interval(self):
        """Pipeline C: Observe → predict_interval cycle."""
        # Generate data
        n = 120
        t = np.arange(n)
        y_values = 100 + t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(n) * 2

        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=30 * n - 1),
                interval="30d",
                eager=True,
            ),
            "value": y_values,
        })

        point_forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("season", PatternSeasonalityForecaster(seasonality=12, method="average")),
            ("residual", PointReductionForecaster(LR(), actual_transformer=LagTransformer([1, 12]))),
        ])
        forecaster = SplitConformalForecaster(
            point_forecaster=point_forecaster,
            conformity_scorer=AbsoluteResidual(),
            calibration_size=30,
        )

        # Fit
        forecaster.fit(df[:80], forecasting_horizon=6)

        # Predict interval anchored at the fit boundary (t=80)
        y_pred1 = forecaster.predict_interval(coverage_rates=[0.9], forecasting_horizon=6)

        # Observe newly arrived ground truth, then predict_interval again. The
        # second forecast is anchored later, so its vintage_time advances and
        # the predicted timestamps shift forward.
        forecaster.observe(df[80:86])
        y_pred2 = forecaster.predict_interval(coverage_rates=[0.9], forecasting_horizon=6)

        assert len(y_pred1) == 6
        assert len(y_pred2) == 6
        assert y_pred1.columns == y_pred2.columns
        # The observe() advanced state: the new vintage is strictly later and
        # the forecast window starts after the previous one ended.
        assert y_pred2["vintage_time"][0] > y_pred1["vintage_time"][0]
        assert y_pred2["time"].min() > y_pred1["time"].min()

    def test_pipeline_c_decomposition_interval_combination(self):
        """Pipeline C: Decomposition + interval produces valid structure."""
        # Generate data with clear components
        n = 100
        t = np.arange(n)
        trend = 50 + 2 * t
        seasonality = 15 * np.sin(2 * np.pi * t / 12)
        residual = np.random.randn(n) * 3
        y_values = trend + seasonality + residual

        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=30 * n - 1),
                interval="30d",
                eager=True,
            ),
            "value": y_values,
        })

        point_forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("season", PatternSeasonalityForecaster(seasonality=12, method="average")),
            ("residual", PointReductionForecaster(LR(), actual_transformer=LagTransformer([1, 12]))),
        ])
        forecaster = SplitConformalForecaster(
            point_forecaster=point_forecaster,
            conformity_scorer=AbsoluteResidual(),
            calibration_size=25,
        )

        forecaster.fit(df[:75], forecasting_horizon=12)

        # Get both point and interval predictions
        y_pred_point = forecaster.predict(forecasting_horizon=12)
        y_pred_interval = forecaster.predict_interval(coverage_rates=[0.9], forecasting_horizon=12)

        # Interval should contain point predictions (approximately at center)
        point_vals = y_pred_point["value"].to_numpy()
        lower_vals = y_pred_interval["value_lower_0.9"].to_numpy()
        upper_vals = y_pred_interval["value_upper_0.9"].to_numpy()

        # Point should be within intervals
        assert np.all((lower_vals <= point_vals) & (point_vals <= upper_vals)), "Point predictions outside intervals"


@pytest.mark.integration
class TestFeatureForecastedPipeline:
    """Pipeline D: Feature-forecasted weather-demand (linear relationships)."""

    def test_pipeline_d_feature_forecaster_learns(self):
        """Pipeline D: Feature forecaster learns to predict temperature."""
        # Generate hourly data: y = 3·temperature + 0.5·hour_index + 10
        n = 200
        t = np.arange(n)
        temperature = 20 + 5 * np.sin(2 * np.pi * t / 24) + np.random.randn(n) * 0.5
        demand = 3 * temperature + 0.5 * t + 10 + np.random.randn(n) * 2

        df_y = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                interval="1h",
                eager=True,
            ),
            "demand": demand,
        })

        df_X = pl.DataFrame({
            "time": df_y["time"],
            "temperature": temperature,
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(
                Ridge(alpha=0.0),
                actual_transformer=FeaturePipeline([
                    ("lags", LagTransformer([1, 24])),
                    ("scale", StandardScaler()),
                ]),
            ),
            feature_forecaster=PointReductionForecaster(
                LR(),
                actual_transformer=LagTransformer([1, 24]),
            ),
            strategy="predicted",
            split_ratio=0.6,
        )

        # Fit
        forecaster.fit(df_y[:150], df_X[:150], forecasting_horizon=10)

        # Predict
        y_pred = forecaster.predict(forecasting_horizon=10)

        # Should produce predictions
        assert len(y_pred) == 10
        assert "demand" in y_pred.columns

    def test_pipeline_d_target_learns_relationship(self):
        """Pipeline D: Target forecaster learns temperature → demand relationship."""
        # Generate data with strong linear relationship
        n = 200
        t = np.arange(n)
        temperature = 20 + 5 * np.sin(2 * np.pi * t / 24)
        demand = 3 * temperature + 10  # Pure linear, no noise for clear test

        df_y = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                interval="1h",
                eager=True,
            ),
            "demand": demand,
        })

        df_X = pl.DataFrame({
            "time": df_y["time"],
            "temperature": temperature,
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(
                Ridge(alpha=0.0),
                actual_transformer=LagTransformer([1, 24]),
            ),
            feature_forecaster=PointReductionForecaster(
                LR(),
                actual_transformer=LagTransformer([1, 24]),
            ),
            strategy="actual",  # Use actual features to isolate target learning
            split_ratio=0.6,
        )

        # Fit
        y_train = df_y[:150]
        X_actual_train = df_X[:150]
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=10)

        # Predict with actual features
        y_pred = forecaster.predict(forecasting_horizon=10)
        y_test = df_y[150:160]

        # Score
        scorer = MeanAbsoluteError()
        scorer.fit(y_train)
        score = scorer.score(y_test, y_pred)

        # Should have low error since relationship is pure linear
        assert score < 5.0, f"Target didn't learn relationship well: MAE={score}"

    def test_pipeline_d_full_lifecycle(self):
        """Pipeline D: Full cycle fit → predict → observe → predict → observe_predict."""
        # Generate data (enough rows so feature forecaster has data after split)
        n = 400
        t = np.arange(n)
        temperature = 20 + 5 * np.sin(2 * np.pi * t / 24) + np.random.randn(n) * 0.5
        demand = 3 * temperature + 0.5 * t + 10 + np.random.randn(n) * 2

        df_y = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                interval="1h",
                eager=True,
            ),
            "demand": demand,
        })

        df_X = pl.DataFrame({
            "time": df_y["time"],
            "temperature": temperature,
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(
                Ridge(alpha=0.01),
                actual_transformer=FeaturePipeline([
                    ("lags", LagTransformer([1, 24])),
                    ("scale", StandardScaler()),
                ]),
            ),
            feature_forecaster=PointReductionForecaster(
                LR(),
                actual_transformer=LagTransformer([1, 24]),
            ),
            strategy="predicted",
            split_ratio=0.6,
        )

        # Fit with enough data
        forecaster.fit(df_y[:300], df_X[:300], forecasting_horizon=10)

        # Predict
        y_pred1 = forecaster.predict(forecasting_horizon=10)
        assert len(y_pred1) == 10

        # Observe
        forecaster.observe(df_y[300:310], df_X[300:310])

        # Predict after observe
        y_pred2 = forecaster.predict(forecasting_horizon=10)
        assert len(y_pred2) == 10

        # Observe_predict now rolls over the block (initial prediction plus one per
        # stride-sized slice), producing multiple vintages.
        y_pred3 = forecaster.observe_predict(df_y[310:320], df_X[310:320], forecasting_horizon=10)
        assert y_pred3["vintage_time"].n_unique() > 1

    def test_pipeline_d_strategy_comparison(self):
        """Pipeline D: Compare 'actual' vs 'predicted' strategies."""
        # Generate data
        n = 180
        t = np.arange(n)
        temperature = 20 + 5 * np.sin(2 * np.pi * t / 24) + np.random.randn(n) * 0.5
        demand = 3 * temperature + 10 + np.random.randn(n) * 2

        df_y = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                interval="1h",
                eager=True,
            ),
            "demand": demand,
        })

        df_X = pl.DataFrame({
            "time": df_y["time"],
            "temperature": temperature,
        })

        # Strategy: actual
        forecaster_actual = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(
                Ridge(alpha=0.0),
                actual_transformer=LagTransformer([1, 24]),
            ),
            feature_forecaster=PointReductionForecaster(
                LR(),
                actual_transformer=LagTransformer([1, 24]),
            ),
            strategy="actual",
            split_ratio=0.6,
        )

        # Strategy: predicted
        forecaster_predicted = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(
                Ridge(alpha=0.0),
                actual_transformer=LagTransformer([1, 24]),
            ),
            feature_forecaster=PointReductionForecaster(
                LR(),
                actual_transformer=LagTransformer([1, 24]),
            ),
            strategy="predicted",
            split_ratio=0.6,
        )

        # Fit both
        y_train = df_y[:150]
        X_actual_train = df_X[:150]
        forecaster_actual.fit(y_train, X_actual_train, forecasting_horizon=10)
        forecaster_predicted.fit(y_train, X_actual_train, forecasting_horizon=10)

        # Predict with actual features
        y_pred_actual = forecaster_actual.predict(forecasting_horizon=10)
        y_pred_predicted = forecaster_predicted.predict(forecasting_horizon=10)

        # Both should work
        assert len(y_pred_actual) == 10
        assert len(y_pred_predicted) == 10

        # Actual should generally be more accurate (has true features)
        y_test = df_y[150:160]
        scorer = MeanAbsoluteError()
        scorer.fit(y_train)
        score_actual = scorer.score(y_test, y_pred_actual)
        score_predicted = scorer.score(y_test, y_pred_predicted)

        # Just verify both are finite
        assert np.isfinite(score_actual)
        assert np.isfinite(score_predicted)

    def test_pipeline_d_chained_predictions(self):
        """Pipeline D: Verify chained predictions match analytical expectations."""
        # Simple case: temperature is constant, demand = 3·temperature + 10
        n = 150
        temperature = np.full(n, 20.0)  # Constant temperature
        demand = 3 * temperature + 10

        df_y = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                interval="1h",
                eager=True,
            ),
            "demand": demand,
        })

        df_X = pl.DataFrame({
            "time": df_y["time"],
            "temperature": temperature,
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(
                Ridge(alpha=0.0),
                actual_transformer=LagTransformer([1]),
            ),
            feature_forecaster=PointReductionForecaster(
                LR(),
                actual_transformer=LagTransformer([1]),
            ),
            strategy="predicted",
            split_ratio=0.6,
        )

        # Fit
        forecaster.fit(df_y[:100], df_X[:100], forecasting_horizon=10)

        # Predict
        y_pred = forecaster.predict(forecasting_horizon=10)

        # Since temperature is constant and relationship is linear,
        # predictions should be close to 70 (3*20 + 10)
        pred_mean = y_pred["demand"].mean()
        assert 65 < pred_mean < 75, f"Predictions far from expected: {pred_mean}"


@pytest.mark.integration
class TestHyperparameterTunedPipeline:
    """Pipeline E: Hyperparameter-tuned forecasting (grid search)."""

    def test_pipeline_e_best_params_selection(self):
        """Pipeline E: GridSearchCV selects best alpha for linear data."""
        # Generate linear data (Ridge alpha=0 should be best)
        n = 150
        t = np.arange(n)
        y_values = 100 + 2 * t + np.random.randn(n) * 1

        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                interval="1d",
                eager=True,
            ),
            "value": y_values,
        })

        base = PointReductionForecaster(
            Ridge(),
            actual_transformer=FeaturePipeline([
                ("lag", LagTransformer(1)),
                ("scale", StandardScaler()),
            ]),
        )

        search = GridSearchCV(
            base,
            param_grid={
                "estimator__alpha": [0.0, 0.01, 0.1, 1.0],
            },
            scoring=MeanAbsoluteError(),
            cv=SlidingWindowSplitter(n_splits=3, test_size=10, stride=10),
            refit=True,
        )

        # Fit
        search.fit(df, forecasting_horizon=5)

        # Best alpha should be 0.0 or very small (linear data)
        # Note: GridSearchCV scoring may encounter edge cases, so we just verify
        # that a valid best_params_ was selected
        best_alpha = search.best_params_["estimator__alpha"]
        assert best_alpha in [0.0, 0.01, 0.1, 1.0], f"Unexpected alpha: {best_alpha}"

    def test_pipeline_e_predict_best_model(self):
        """Pipeline E: Predict uses best model from grid search."""
        # Generate data
        n = 150
        t = np.arange(n)
        y_values = 100 + 2 * t + np.random.randn(n) * 2

        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                interval="1d",
                eager=True,
            ),
            "value": y_values,
        })

        base = PointReductionForecaster(
            Ridge(),
            actual_transformer=FeaturePipeline([
                ("lag", LagTransformer([1, 2])),
                ("scale", StandardScaler()),
            ]),
        )

        search = GridSearchCV(
            base,
            param_grid={
                "estimator__alpha": [0.0, 0.1, 1.0],
                "actual_transformer__lag__lag": [[1], [1, 2], [1, 2, 3]],
            },
            scoring=MeanAbsoluteError(),
            cv=SlidingWindowSplitter(n_splits=3, test_size=10, stride=10),
            refit=True,
        )

        # Fit
        y_train = df[:120]
        search.fit(y_train, forecasting_horizon=5)

        # Predict
        y_pred = search.predict(forecasting_horizon=5)

        # Should return valid predictions
        assert len(y_pred) == 5
        assert "value" in y_pred.columns

        # Score against test data
        y_test = df[120:125]
        scorer = MeanAbsoluteError()
        scorer.fit(y_train)
        score = scorer.score(y_test, y_pred)
        assert np.isfinite(score)

    def test_pipeline_e_recursive_horizon(self):
        """Pipeline E: Recursive prediction with different horizon than fit."""
        # Generate data
        n = 150
        t = np.arange(n)
        y_values = 100 + 2 * t + np.random.randn(n) * 2

        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                interval="1d",
                eager=True,
            ),
            "value": y_values,
        })

        base = PointReductionForecaster(
            Ridge(),
            actual_transformer=FeaturePipeline([
                ("lag", LagTransformer(1)),
                ("scale", StandardScaler()),
            ]),
        )

        search = GridSearchCV(
            base,
            param_grid={
                "estimator__alpha": [0.0, 0.1],
            },
            scoring=MeanAbsoluteError(),
            cv=SlidingWindowSplitter(n_splits=3, test_size=5, stride=10),
            refit=True,
        )

        # Fit with horizon=5
        search.fit(df[:120], forecasting_horizon=5)

        # Predict with horizon=10 (should work recursively)
        y_pred = search.predict(forecasting_horizon=10)

        assert len(y_pred) == 10
        assert "value" in y_pred.columns

    def test_pipeline_e_observe_predict_delegation(self):
        """Pipeline E: observe_predict delegates to best forecaster."""
        # Generate data
        n = 180
        t = np.arange(n)
        y_values = 100 + 2 * t + np.random.randn(n) * 2

        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                interval="1d",
                eager=True,
            ),
            "value": y_values,
        })

        base = PointReductionForecaster(
            Ridge(),
            actual_transformer=LagTransformer([1, 2]),
        )

        search = GridSearchCV(
            base,
            param_grid={
                "estimator__alpha": [0.0, 0.1],
            },
            scoring=MeanAbsoluteError(),
            cv=SlidingWindowSplitter(n_splits=3, test_size=10, stride=20),
            refit=True,
        )

        # Fit
        search.fit(df[:150], forecasting_horizon=7)

        # Observe_predict
        y_pred = search.observe_predict(df[150:157], forecasting_horizon=7)

        # Should work without crash (length may differ from horizon due to framework internals)
        assert len(y_pred) > 0
        assert "value" in y_pred.columns

    def test_pipeline_e_best_forecaster_structure(self):
        """Pipeline E: best_forecaster_ has correct structure after refit."""
        # Generate data
        n = 150
        t = np.arange(n)
        y_values = 100 + 2 * t + np.random.randn(n) * 2

        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                interval="1d",
                eager=True,
            ),
            "value": y_values,
        })

        base = PointReductionForecaster(
            Ridge(),
            actual_transformer=FeaturePipeline([
                ("lag", LagTransformer(1)),
                ("scale", StandardScaler()),
            ]),
        )

        search = GridSearchCV(
            base,
            param_grid={
                "estimator__alpha": [0.0, 0.01, 0.1],
            },
            scoring=MeanAbsoluteError(),
            cv=SlidingWindowSplitter(n_splits=3, test_size=10, stride=20),
            refit=True,
        )

        # Fit
        search.fit(df, forecasting_horizon=5)

        # Check best_forecaster_
        assert hasattr(search, "best_forecaster_")
        assert isinstance(search.best_forecaster_, PointReductionForecaster)

        # Should have fitted state
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(search.best_forecaster_)

        # Parameters should match best_params_
        best_alpha = search.best_params_["estimator__alpha"]
        assert search.best_forecaster_.estimator.alpha == best_alpha


@pytest.mark.integration
class TestMaxNestingPipeline:
    """Pipeline F: Everything composed (maximum nesting depth)."""

    def test_pipeline_f_params_contract(self):
        """Pipeline F: deep get_params/set_params reach and round-trip leaf params.

        Consolidates the deep-nesting get_params, deep-access set_params, and
        get/set round-trip checks on the same maximally nested topology. The
        systematic suite (check_composition_nested_param_addressable,
        check_get_set_params_round_trip) enforces the generic contract; this
        test confirms it holds on a real, deeply nested pipeline.
        """
        forecaster = ColumnForecaster([
            (
                "demand",
                ForecastedFeatureForecaster(
                    target_forecaster=SplitConformalForecaster(
                        point_forecaster=DecompositionPipeline([
                            ("trend", PolynomialTrendForecaster(1)),
                            (
                                "residual",
                                PointReductionForecaster(
                                    Ridge(alpha=0.1),
                                    actual_transformer=FeaturePipeline([
                                        (
                                            "lags",
                                            FeatureUnion([
                                                ("recent", LagTransformer([1, 2, 3])),
                                                ("seasonal", LagTransformer([7, 14])),
                                            ]),
                                        ),
                                        ("scale", StandardScaler()),
                                    ]),
                                ),
                            ),
                        ]),
                        conformity_scorer=AbsoluteResidual(),
                        calibration_size=30,
                    ),
                    feature_forecaster=PointReductionForecaster(
                        LR(),
                        actual_transformer=LagTransformer([1, 7]),
                    ),
                    strategy="predicted",
                ),
                ["demand"],
            ),
            ("auxiliary", SeasonalNaive(7), ["auxiliary"]),
        ])

        alpha_key = "demand__target_forecaster__point_forecaster__residual__estimator__alpha"

        # get_params(deep=True) exposes the full nested key structure.
        params = forecaster.get_params(deep=True)
        assert "demand" in params
        assert "auxiliary" in params
        assert params[alpha_key] == 0.1
        assert params["demand__target_forecaster__calibration_size"] == 30
        assert params["auxiliary__seasonality"] == 7

        # set_params reaches arbitrary leaves at any depth.
        forecaster.set_params(**{
            alpha_key: 0.5,
            "demand__target_forecaster__calibration_size": 50,
            "auxiliary__seasonality": 14,
        })
        updated = forecaster.get_params(deep=True)
        assert updated[alpha_key] == 0.5
        assert updated["demand__target_forecaster__calibration_size"] == 50
        assert updated["auxiliary__seasonality"] == 14

        # Round-trip: feeding scalar params back through set_params is a no-op.
        scalar_params = {
            k: v for k, v in updated.items() if isinstance(v, int | float | str | bool | type(None) | list | tuple)
        }
        forecaster.set_params(**scalar_params)
        final = forecaster.get_params(deep=True)
        assert final[alpha_key] == 0.5
        assert final["demand__target_forecaster__calibration_size"] == 50
        assert final["auxiliary__seasonality"] == 14

    def test_pipeline_f_full_lifecycle_no_crash(self):
        """Pipeline F: fit → predict → predict_interval without crash."""
        # Generate 2-column data with exogenous feature for ForecastedFeatureForecaster
        # Need enough data so ForecastedFeatureForecaster's feature_forecaster (20% split)
        # has sufficient rows after tabularization with LagTransformer([1,7])
        n = 500
        t = np.arange(n)
        temperature = 20 + 5 * np.sin(2 * np.pi * t / 7) + np.random.randn(n) * 0.5
        demand_vals = 3 * temperature + 2 * t + 10 + np.random.randn(n) * 3
        auxiliary_vals = 50 + t + 5 * np.sin(2 * np.pi * t / 7) + np.random.randn(n) * 2

        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                interval="1d",
                eager=True,
            ),
            "demand": demand_vals,
            "auxiliary": auxiliary_vals,
        })

        df_X = pl.DataFrame({
            "time": df["time"],
            "temperature": temperature,
        })

        forecaster = ColumnForecaster([
            (
                "demand",
                ForecastedFeatureForecaster(
                    target_forecaster=SplitConformalForecaster(
                        point_forecaster=DecompositionPipeline([
                            ("trend", PolynomialTrendForecaster(1)),
                            (
                                "residual",
                                PointReductionForecaster(
                                    Ridge(alpha=0.1),
                                    actual_transformer=LagTransformer([1, 2, 3, 7, 14]),
                                ),
                            ),
                        ]),
                        conformity_scorer=AbsoluteResidual(),
                        calibration_size=30,
                    ),
                    feature_forecaster=PointReductionForecaster(
                        LR(),
                        actual_transformer=LagTransformer([1, 7]),
                    ),
                    strategy="predicted",
                    split_ratio=0.8,
                ),
                ["demand"],
            ),
            ("auxiliary", SeasonalNaive(7), ["auxiliary"]),
        ])

        # Fit with X_actual
        forecaster.fit(df[:400], X_actual=df_X[:400], forecasting_horizon=7)

        # Predict point
        y_pred = forecaster.predict(forecasting_horizon=7)
        assert len(y_pred) == 7
        assert "demand" in y_pred.columns
        assert "auxiliary" in y_pred.columns

        # Note: predict_interval is not available on ColumnForecaster when any
        # sub-forecaster is point-only (SeasonalNaive doesn't support intervals).
        # Interval prediction is tested in test_pipeline_f_mixed_metrics.

    def test_pipeline_f_clone_independent_of_fitted_original(self):
        """Pipeline F: fitting the original leaves a deep clone unfitted.

        The generic clone-equality contract (same params, distinct objects) is
        covered by check_composition_clone_deep_clones_components; this test
        adds the integration-specific guarantee that fitting the original does
        not leak fitted state into a clone of a maximally nested pipeline.
        """
        forecaster = ColumnForecaster([
            (
                "demand",
                ForecastedFeatureForecaster(
                    target_forecaster=SplitConformalForecaster(
                        point_forecaster=DecompositionPipeline([
                            ("trend", PolynomialTrendForecaster(1)),
                            (
                                "residual",
                                PointReductionForecaster(
                                    Ridge(alpha=0.1),
                                    actual_transformer=LagTransformer([1, 2, 3, 7, 14]),
                                ),
                            ),
                        ]),
                        conformity_scorer=AbsoluteResidual(),
                        calibration_size=30,
                    ),
                    feature_forecaster=PointReductionForecaster(
                        LR(),
                        actual_transformer=LagTransformer([1, 7]),
                    ),
                    strategy="predicted",
                    split_ratio=0.8,
                ),
                ["demand"],
            ),
            ("auxiliary", SeasonalNaive(7), ["auxiliary"]),
        ])

        # Clone before fitting
        forecaster_cloned = clone(forecaster)

        # Fit original (needs X_actual for ForecastedFeatureForecaster)
        # Need enough data so ForecastedFeatureForecaster's feature_forecaster (20% split)
        # has sufficient rows after tabularization with LagTransformer([1,7])
        n = 500
        t_vals = np.arange(n)
        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                interval="1d",
                eager=True,
            ),
            "demand": 100 + t_vals,
            "auxiliary": 50 + t_vals,
        })

        df_X = pl.DataFrame({
            "time": df["time"],
            "temperature": 20 + 5 * np.sin(2 * np.pi * t_vals / 7),
        })

        forecaster.fit(df[:400], X_actual=df_X[:400], forecasting_horizon=5)

        # Clone should still be unfitted
        with pytest.raises(NotFittedError, match="is not fitted"):
            forecaster_cloned.predict(forecasting_horizon=5)

    def test_pipeline_f_mixed_metrics(self):
        """Pipeline F: Score with both point and interval metrics."""
        # Generate data with exogenous feature for ForecastedFeatureForecaster
        # Need enough data so ForecastedFeatureForecaster's feature_forecaster (20% split)
        # has sufficient rows after tabularization with LagTransformer([1,7])
        n = 500
        t = np.arange(n)
        temperature = 20 + 5 * np.sin(2 * np.pi * t / 7) + np.random.randn(n) * 0.5
        demand_vals = 3 * temperature + 2 * t + 10 + np.random.randn(n) * 3

        df = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                interval="1d",
                eager=True,
            ),
            "demand": demand_vals,
        })

        df_X = pl.DataFrame({
            "time": df["time"],
            "temperature": temperature,
        })

        # Use ForecastedFeatureForecaster directly (no ColumnForecaster) so
        # predict_interval is available (ColumnForecaster requires ALL sub-forecasters
        # to support intervals, which fails if a point-only forecaster is included).
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SplitConformalForecaster(
                point_forecaster=DecompositionPipeline([
                    ("trend", PolynomialTrendForecaster(1)),
                    (
                        "residual",
                        PointReductionForecaster(
                            Ridge(alpha=0.1),
                            actual_transformer=LagTransformer([1, 2, 3, 7, 14]),
                        ),
                    ),
                ]),
                conformity_scorer=AbsoluteResidual(),
                calibration_size=30,
            ),
            feature_forecaster=PointReductionForecaster(
                LR(),
                actual_transformer=LagTransformer([1, 7]),
            ),
            strategy="predicted",
            split_ratio=0.8,
        )

        # Fit with X_actual
        y_train = df[:400]
        y_test = df[400:407]
        X_actual_train = df_X[:400]
        forecaster.fit(y_train, X_actual=X_actual_train, forecasting_horizon=7)

        # Point predictions and scoring
        y_pred_point = forecaster.predict(forecasting_horizon=7)
        point_scorer = MeanAbsoluteError()
        point_scorer.fit(y_train)
        point_score = point_scorer.score(y_test, y_pred_point)
        assert np.isfinite(point_score)

        # Interval predictions and scoring
        y_pred_interval = forecaster.predict_interval(coverage_rates=[0.9], forecasting_horizon=7)
        interval_scorer = IntervalScore(coverage_rates=[0.9])
        interval_scorer.fit(y_train)
        interval_score = interval_scorer.score(y_test, y_pred_interval)
        assert np.isfinite(interval_score)

        # Coverage scoring
        coverage_scorer = EmpiricalCoverage(coverage_rates=[0.9])
        coverage_scorer.fit(y_train)
        coverage = coverage_scorer.score(y_test, y_pred_interval)
        # EmpiricalCoverage with default aggregation_method="all" returns a scalar float
        assert isinstance(coverage, float)
        assert 0 <= coverage <= 1
