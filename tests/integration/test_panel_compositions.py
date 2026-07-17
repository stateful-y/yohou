"""Stress tests for panel data through composition forecasters.

This module tests that all composition types correctly handle panel data
with analytical verification of per-group predictions.
"""

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from yohou.compose import (
    ColumnForecaster,
    ColumnTransformer,
    DecompositionPipeline,
    FeaturePipeline,
    FeatureUnion,
    ForecastedFeatureForecaster,
    LocalPanelForecaster,
)
from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import GridSearchCV, SlidingWindowSplitter
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.preprocessing import (
    LagTransformer,
    MinMaxScaler,
    RollingStatisticsTransformer,
    StandardScaler,
)
from yohou.stationarity import PolynomialTrendForecaster
from yohou.utils.panel import get_group_df, inspect_panel


@pytest.fixture
def linear_panel_5groups():
    """Generate 5-group panel where y_{g,t} = m_g * t + b_g.

    Groups:
    - group_0: slope=1, intercept=10
    - group_1: slope=2, intercept=20
    - group_2: slope=3, intercept=30
    - group_3: slope=4, intercept=40
    - group_4: slope=5, intercept=50
    """
    n = 100
    time_index = pl.datetime_range(
        start=pl.datetime(2020, 1, 1),
        end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
        interval="1d",
        eager=True,
    )

    data = {"time": time_index}
    for group_idx in range(5):
        slope = group_idx + 1
        intercept = (group_idx + 1) * 10
        values = slope * np.arange(n) + intercept
        data[f"g{group_idx}__value"] = values

    return pl.DataFrame(data)


@pytest.fixture
def ar1_panel_3groups():
    """Generate 3-group AR(1) panel where y_{g,t} = φ_g * y_{g,t-1} + ε.

    Groups:
    - group_0: φ=0.3
    - group_1: φ=0.7
    - group_2: φ=0.9
    """
    n = 100
    time_index = pl.datetime_range(
        start=pl.datetime(2020, 1, 1),
        end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
        interval="1d",
        eager=True,
    )

    data = {"time": time_index}
    phis = [0.3, 0.7, 0.9]
    np.random.seed(42)

    for group_idx, phi in enumerate(phis):
        # Generate AR(1) process
        y = np.zeros(n)
        y[0] = 10.0  # Initial value
        noise = np.random.normal(0, 0.5, n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + noise[t]
        data[f"g{group_idx}__value"] = y

    return pl.DataFrame(data)


@pytest.fixture
def linear_panel_10groups():
    """Generate 10-group panel for integration testing scalability."""
    n = 100
    time_index = pl.datetime_range(
        start=pl.datetime(2020, 1, 1),
        end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
        interval="1d",
        eager=True,
    )

    data = {"time": time_index}
    for group_idx in range(10):
        slope = (group_idx + 1) * 0.5
        intercept = (group_idx + 1) * 5
        values = slope * np.arange(n) + intercept
        data[f"g{group_idx}__value"] = values

    return pl.DataFrame(data)


class TestPointReductionPanel:
    """Tests for PointReductionForecaster fit/predict/observe/rewind on panel data."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "actual_transformer",
        [
            None,
            LagTransformer(lag=[1]),
            FeaturePipeline([("lag", LagTransformer(lag=[1])), ("roll", RollingStatisticsTransformer(window_size=3))]),
        ],
        ids=["no_features", "lag_only", "lag_pipeline"],
    )
    def test_point_reduction_panel_fit_predict(self, linear_panel_5groups, actual_transformer):
        """Test PointReductionForecaster fit/predict on panel data with analytical verification."""
        y = linear_panel_5groups
        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            actual_transformer=actual_transformer,
        )

        # Split data
        y_train = y[:80]
        forecasting_horizon = 5

        # Fit and predict
        forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)
        y_pred = forecaster.predict(forecasting_horizon=forecasting_horizon)

        # Verify the mandatory prediction time-column contract on the panel path.
        assert "vintage_time" in y_pred.columns
        assert "time" in y_pred.columns

        # Verify panel structure
        global_cols, panel_groups = inspect_panel(y_pred)
        assert len(panel_groups) == 5

        # Verify each group's predictions analytically
        # For linear trend y = m*t + b, predictions should continue the trend
        for group_idx in range(5):
            group_name = f"g{group_idx}"
            slope = group_idx + 1
            intercept = (group_idx + 1) * 10

            # Extract predicted values for this group
            y_pred_group = get_group_df(y_pred, group_name, schema={"value": pl.Float64})

            # Analytical forecast: continue linear trend from t=80 to t=84
            expected_values = np.array([slope * t + intercept for t in range(80, 85)])

            # Allow tolerance due to reduction approximation
            np.testing.assert_allclose(
                y_pred_group["value"].to_numpy(),
                expected_values,
                rtol=0.2,  # Reduction on panel data may pool groups
                atol=15.0,
            )

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "actual_transformer",
        [
            None,
            LagTransformer(lag=[1]),
            FeaturePipeline([("lag", LagTransformer(lag=[1])), ("roll", RollingStatisticsTransformer(window_size=3))]),
        ],
        ids=["no_features", "lag_only", "lag_pipeline"],
    )
    def test_point_reduction_panel_observe_all_groups(self, linear_panel_5groups, actual_transformer):
        """Test observe() with all groups updates all group states."""
        y = linear_panel_5groups
        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            actual_transformer=actual_transformer,
        )

        # Fit on first 60 rows
        y_train = y[:60]
        forecaster.fit(y_train, forecasting_horizon=5)

        # Update with rows 60-80
        y_update = y[60:80]
        forecaster.observe(y_update)

        # Predict and verify
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Verify panel structure maintained
        global_cols, panel_groups = inspect_panel(y_pred)
        assert len(panel_groups) == 5

        # Verify predictions start from correct position (t=80)
        for group_idx in range(5):
            group_name = f"g{group_idx}"
            y_pred_group = get_group_df(y_pred, group_name, schema={"value": pl.Float64})

            # Should predict from t=80 onwards
            assert len(y_pred_group) == 5

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "actual_transformer",
        [
            None,
            LagTransformer(lag=[1]),
            FeaturePipeline([("lag", LagTransformer(lag=[1])), ("roll", RollingStatisticsTransformer(window_size=3))]),
        ],
        ids=["no_features", "lag_only", "lag_pipeline"],
    )
    def test_point_reduction_panel_observe_subset_groups(self, linear_panel_5groups, actual_transformer):
        """Test observe() with groups filters to specific groups."""
        y = linear_panel_5groups
        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            actual_transformer=actual_transformer,
        )

        # Fit on first 60 rows
        y_train = y[:60]
        forecaster.fit(y_train, forecasting_horizon=5)

        # Store initial prediction state for all groups
        y_pred_before = forecaster.predict(forecasting_horizon=5)

        # Update only groups 1 and 2
        y_update = y[60:80]
        forecaster.observe(y_update, groups=["g1", "g2"])

        # Predict only the updated groups (partial update replaces internal state
        # for those groups only, making predict-all unavailable)
        y_pred_after = forecaster.predict(forecasting_horizon=5, groups=["g1", "g2"])

        # Groups 1 and 2 should have different predictions (updated state)
        for group_idx in [1, 2]:
            group_name = f"g{group_idx}"
            pred_before = get_group_df(y_pred_before, group_name, schema={"value": pl.Float64})
            pred_after = get_group_df(y_pred_after, group_name, schema={"value": pl.Float64})

            # These should differ (hard to predict exact values, but they should be different)
            assert not np.allclose(
                pred_before["value"].to_numpy(),
                pred_after["value"].to_numpy(),
                atol=0.1,
            )

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "actual_transformer",
        [
            None,
            LagTransformer(lag=[1]),
            FeaturePipeline([("lag", LagTransformer(lag=[1])), ("roll", RollingStatisticsTransformer(window_size=3))]),
        ],
        ids=["no_features", "lag_only", "lag_pipeline"],
    )
    def test_point_reduction_panel_rewind_subset_groups(self, linear_panel_5groups, actual_transformer):
        """Test rewind() with groups filters to specific groups."""
        y = linear_panel_5groups
        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            actual_transformer=actual_transformer,
        )

        # Fit on first 80 rows
        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=5)

        # Reset only group_1
        y_reset = y[60:80]  # Last 20 rows

        forecaster.rewind(y_reset, groups=["g1"])

        # Predict only the reset group (partial reset replaces internal state
        # for that group only, making predict-all unavailable)
        y_pred = forecaster.predict(forecasting_horizon=5, groups=["g1"])

        # Verify prediction for the reset group
        assert "g1__value" in y_pred.columns
        assert len(y_pred) == 5


class TestColumnForecasterPanel:
    """Tests for ColumnForecaster with different forecasters and remainders on panel data."""

    @pytest.mark.integration
    def test_column_forecaster_panel_separate_forecasters(self):
        """Test ColumnForecaster with different forecasters per panel column."""
        n = 100
        time_index = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        # Create panel with two target columns (col_a, col_b), each with 2 groups
        data = {
            "time": time_index,
            "cola__g1": np.arange(n) * 1.0 + 10,
            "cola__g2": np.arange(n) * 2.0 + 20,
            "colb__g1": np.arange(n) * 0.5 + 5,
            "colb__g2": np.arange(n) * 1.5 + 15,
        }
        y = pl.DataFrame(data)

        # Assign different forecasters to col_a and col_b
        forecaster = ColumnForecaster(
            forecasters=[
                ("col_a_fc", PointReductionForecaster(estimator=LinearRegression()), ["cola__g1", "cola__g2"]),
                ("col_b_fc", SeasonalNaive(seasonality=7), ["colb__g1", "colb__g2"]),
            ],
        )

        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Verify panel structure preserved
        global_cols, panel_groups = inspect_panel(y_pred)
        assert "cola" in panel_groups
        assert "colb" in panel_groups
        assert len(panel_groups["cola"]) == 2
        assert len(panel_groups["colb"]) == 2

    @pytest.mark.integration
    def test_column_forecaster_panel_with_remainder_passthrough(self):
        """Test ColumnForecaster with remainder='passthrough' on panel data."""
        n = 100
        time_index = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        data = {
            "time": time_index,
            "target__group_1": np.arange(n) * 1.0 + 10,
            "target__group_2": np.arange(n) * 2.0 + 20,
            "other__group_1": np.arange(n) * 0.5,
            "other__group_2": np.arange(n) * 1.5,
        }
        y = pl.DataFrame(data)

        # Only forecast 'target', remainder uses a forecaster
        forecaster = ColumnForecaster(
            forecasters=[
                (
                    "target_fc",
                    PointReductionForecaster(estimator=LinearRegression()),
                    ["target__group_1", "target__group_2"],
                ),
            ],
            remainder=SeasonalNaive(seasonality=7),
        )

        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Verify 'target' is forecasted, 'other' is passed through
        global_cols, panel_groups = inspect_panel(y_pred)
        assert "target" in panel_groups
        assert "other" in panel_groups

    @pytest.mark.integration
    def test_column_forecaster_panel_each_column_panel(self):
        """Test ColumnForecaster where each column is a full panel."""
        n = 100
        time_index = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        # Three panel columns, each with 3 groups
        data = {"time": time_index}
        for col in ["sales", "price", "volume"]:
            for group_idx in range(3):
                slope = (ord(col[0]) - ord("a") + 1) * (group_idx + 1) * 0.1
                intercept = (ord(col[0]) - ord("a") + 1) * (group_idx + 1)
                data[f"{col}__group_{group_idx}"] = slope * np.arange(n) + intercept

        y = pl.DataFrame(data)

        forecaster = ColumnForecaster(
            forecasters=[
                (
                    "sales_fc",
                    PointReductionForecaster(estimator=LinearRegression()),
                    [f"sales__group_{g}" for g in range(3)],
                ),
                ("price_fc", PointReductionForecaster(estimator=Ridge()), [f"price__group_{g}" for g in range(3)]),
                ("volume_fc", SeasonalNaive(seasonality=7), [f"volume__group_{g}" for g in range(3)]),
            ],
        )

        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Verify all columns and groups present
        global_cols, panel_groups = inspect_panel(y_pred)
        assert len(panel_groups) == 3
        for col in ["sales", "price", "volume"]:
            assert col in panel_groups
            assert len(panel_groups[col]) == 3

    @pytest.mark.integration
    def test_column_forecaster_panel_observe_and_predict(self):
        """Test ColumnForecaster observe/predict with panel data."""
        n = 100
        time_index = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        data = {
            "time": time_index,
            "cola__g1": np.arange(n) * 1.0 + 10,
            "cola__g2": np.arange(n) * 2.0 + 20,
        }
        y = pl.DataFrame(data)

        forecaster = ColumnForecaster(
            forecasters=[
                ("col_a_fc", PointReductionForecaster(estimator=LinearRegression()), ["cola__g1", "cola__g2"]),
            ],
        )

        y_train = y[:60]
        forecaster.fit(y_train, forecasting_horizon=5)

        # Update with new data
        y_update = y[60:80]
        forecaster.observe(y_update)

        y_pred = forecaster.predict(forecasting_horizon=5)

        # Verify structure
        global_cols, panel_groups = inspect_panel(y_pred)
        assert "cola" in panel_groups
        assert len(panel_groups["cola"]) == 2


class TestDecompositionPipelinePanel:
    """Tests for DecompositionPipeline trend/residual/seasonality handling on panel data."""

    @pytest.mark.integration
    def test_decomposition_pipeline_panel_per_group_trends(self, linear_panel_5groups):
        """Test a bare (un-wrapped) DecompositionPipeline on panel data.

        This is the cross-group code path: a single DecompositionPipeline fits
        its trend across all groups at once (distinct from
        test_decomposition_pipeline_panel_residuals_near_zero, which wraps the
        pipeline in LocalPanelForecaster for independent per-group fitting). The
        per-group slope still emerges, so each group's forecast must increase by
        its own slope step.
        """
        y = linear_panel_5groups
        y_train = y[:80]

        forecaster = DecompositionPipeline(
            forecasters=[
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("residual", SeasonalNaive(seasonality=7)),
            ],
        )

        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Verify panel structure
        global_cols, panel_groups = inspect_panel(y_pred)
        assert len(panel_groups) == 5

        # Each group continues its own positive linear trend: the step between
        # consecutive forecasts must equal that group's slope (group_idx + 1).
        for group_idx in range(5):
            group_name = f"g{group_idx}"
            y_pred_group = get_group_df(y_pred, group_name, schema={"value": pl.Float64})
            assert len(y_pred_group) == 5
            diffs = np.diff(y_pred_group["value"].to_numpy())
            np.testing.assert_allclose(diffs, float(group_idx + 1), atol=0.5)

    @pytest.mark.integration
    def test_decomposition_pipeline_panel_residuals_near_zero(self, linear_panel_5groups):
        """Test that residuals are near-zero after trend removal per group.

        Uses LocalPanelForecaster to fit independent DecompositionPipelines
        per group, so each group's trend captures its own signal.
        """
        y = linear_panel_5groups
        y_train = y[:80]

        # Wrap DecompositionPipeline with LocalPanelForecaster for per-group fitting
        forecaster = LocalPanelForecaster(
            DecompositionPipeline(
                forecasters=[
                    ("trend", PolynomialTrendForecaster(degree=1)),
                    ("residual", SeasonalNaive(seasonality=7)),
                ],
            ),
        )
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # For purely linear data, trend should capture everything
        # so predictions should be very close to analytical values
        for group_idx in range(5):
            group_name = f"g{group_idx}"
            slope = group_idx + 1
            intercept = (group_idx + 1) * 10
            y_pred_group = get_group_df(y_pred, group_name, schema={"value": pl.Float64})
            expected_values = np.array([slope * t + intercept for t in range(80, 85)])

            # Allow tolerance for SeasonalNaive offset (up to one slope step)
            np.testing.assert_allclose(
                y_pred_group["value"].to_numpy(),
                expected_values,
                atol=float(slope + 1),
            )

    @pytest.mark.integration
    def test_decomposition_pipeline_panel_multi_step(self):
        """Test DecompositionPipeline with trend + residual on panel data."""
        n = 100
        time_index = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        # Create simple linear trends per group
        data = {"time": time_index}
        for group_idx in range(3):
            slope = (group_idx + 1) * 2.0
            intercept = (group_idx + 1) * 10.0
            data[f"g{group_idx}__value"] = slope * np.arange(n) + intercept

        y = pl.DataFrame(data)
        y_train = y[:80]

        # Wrap DecompositionPipeline with LocalPanelForecaster for per-group fitting
        forecaster = LocalPanelForecaster(
            DecompositionPipeline(
                forecasters=[
                    ("trend", PolynomialTrendForecaster(degree=1)),
                    ("residual", SeasonalNaive(seasonality=7)),
                ],
            ),
        )

        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Verify structure
        global_cols, panel_groups = inspect_panel(y_pred)
        assert len(panel_groups) == 3

        # Predictions should approximately match analytical trends (per-group fitting)
        for group_idx in range(3):
            group_name = f"g{group_idx}"
            y_pred_group = get_group_df(y_pred, group_name, schema={"value": pl.Float64})

            slope = (group_idx + 1) * 2.0
            intercept = (group_idx + 1) * 10.0
            expected = np.array([slope * t + intercept for t in range(80, 85)])

            # Per-group trend captures each group's slope; tolerance for SeasonalNaive offset
            np.testing.assert_allclose(
                y_pred_group["value"].to_numpy(),
                expected,
                rtol=1e-2,
                atol=float(slope + 1),
            )

    @pytest.mark.integration
    def test_decomposition_pipeline_panel_with_seasonality(self):
        """Test DecompositionPipeline with trend + seasonality on panel."""
        n = 100
        time_index = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        # Add trend + seasonality per group
        data = {"time": time_index}
        for group_idx in range(2):
            trend = (group_idx + 1) * np.arange(n)
            seasonality = 5 * np.sin(2 * np.pi * np.arange(n) / 7)  # Weekly pattern
            data[f"g{group_idx}__value"] = trend + seasonality

        y = pl.DataFrame(data)
        y_train = y[:80]

        forecaster = DecompositionPipeline(
            forecasters=[
                ("trend", PolynomialTrendForecaster(degree=1)),
                ("residual", SeasonalNaive(seasonality=7)),
            ],
        )

        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Verify panel structure maintained
        global_cols, panel_groups = inspect_panel(y_pred)
        assert len(panel_groups) == 2


class TestForecastedFeatureForecasterPanel:
    """Tests for ForecastedFeatureForecaster strategies and observe on panel data."""

    @pytest.mark.integration
    def test_forecasted_feature_forecaster_panel_matching_structure(self):
        """Test ForecastedFeatureForecaster with matching panel structure for y and X_actual."""
        n = 200
        time_index = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        # Target: linear trends per group
        y_data = {"time": time_index}
        for group_idx in range(3):
            y_data[f"g{group_idx}__target"] = (group_idx + 1) * np.arange(n) + 10

        # Feature: also linear per group (could be correlated)
        X_data = {"time": time_index}
        for group_idx in range(3):
            X_data[f"g{group_idx}__feature"] = (group_idx + 1) * 0.5 * np.arange(n) + 5

        y = pl.DataFrame(y_data)
        X_actual = pl.DataFrame(X_data)

        y_train = y[:160]
        X_actual_train = X_actual[:160]

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(
                estimator=LinearRegression(),
            ),
            feature_forecaster=PointReductionForecaster(
                estimator=LinearRegression(),
            ),
            strategy="predicted",
            split_ratio=0.8,
        )

        forecaster.fit(y_train, X_actual_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Verify panel structure maintained
        global_cols, panel_groups = inspect_panel(y_pred)
        assert len(panel_groups) == 3

    @pytest.mark.integration
    def test_forecasted_feature_forecaster_panel_actual_strategy(self):
        """Test ForecastedFeatureForecaster with strategy='actual' on panel."""
        n = 100
        time_index = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        y_data = {"time": time_index}
        X_data = {"time": time_index}

        for group_idx in range(2):
            y_data[f"g{group_idx}__target"] = (group_idx + 1) * np.arange(n)
            X_data[f"g{group_idx}__feature"] = (group_idx + 1) * 0.5 * np.arange(n)

        y = pl.DataFrame(y_data)
        X_actual = pl.DataFrame(X_data)

        y_train = y[:80]
        X_actual_train = X_actual[:80]

        # strategy="actual" reuses the rewound actual feature buffer at predict
        # time and does not consume a separate future-feature argument, so no
        # X_future is passed to predict() below.
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(
                estimator=LinearRegression(),
            ),
            feature_forecaster=PointReductionForecaster(
                estimator=LinearRegression(),
            ),
            strategy="actual",
        )

        forecaster.fit(y_train, X_actual_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Verify panel structure
        global_cols, panel_groups = inspect_panel(y_pred)
        assert len(panel_groups) == 2

    @pytest.mark.integration
    def test_forecasted_feature_forecaster_panel_observe(self):
        """Test ForecastedFeatureForecaster observe() with panel data."""
        n = 200
        time_index = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        y_data = {"time": time_index}
        X_data = {"time": time_index}

        for group_idx in range(2):
            y_data[f"g{group_idx}__target"] = np.arange(n) * (group_idx + 1)
            X_data[f"g{group_idx}__feature"] = np.arange(n) * (group_idx + 1) * 0.5

        y = pl.DataFrame(y_data)
        X_actual = pl.DataFrame(X_data)

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(
                estimator=LinearRegression(),
            ),
            feature_forecaster=PointReductionForecaster(
                estimator=LinearRegression(),
            ),
            strategy="predicted",
            split_ratio=0.8,
        )

        y_train = y[:120]
        X_actual_train = X_actual[:120]
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=5)

        # Update
        y_update = y[120:160]
        X_actual_update = X_actual[120:160]
        forecaster.observe(y_update, X_actual_update)

        y_pred = forecaster.predict(forecasting_horizon=5)

        # Verify structure
        global_cols, panel_groups = inspect_panel(y_pred)
        assert len(panel_groups) == 2


class TestColumnTransformerPanel:
    """Tests for ColumnTransformer scaling and integration with forecasters on panel data."""

    @pytest.mark.integration
    def test_column_transformer_panel_different_scalers_per_group(self):
        """Test ColumnTransformer with different scalers for different group columns."""
        n = 100
        time_index = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        # Create panel features
        X_data = {
            "time": time_index,
            "feat1__group_1": np.arange(n) * 1.0,
            "feat1__group_2": np.arange(n) * 2.0,
            "feat2__group_1": np.arange(n) * 0.5,
            "feat2__group_2": np.arange(n) * 1.5,
        }
        X_actual = pl.DataFrame(X_data)

        # Apply different scaling to feat1 vs feat2
        transformer = ColumnTransformer(
            transformers=[
                ("scaler1", StandardScaler(), ["feat1__group_1", "feat1__group_2"]),
                ("scaler2", MinMaxScaler(), ["feat2__group_1", "feat2__group_2"]),
            ],
            verbose_feature_names_out=False,
        )

        X_actual_train = X_actual[:80]
        X_transformed = transformer.fit_transform(X_actual_train)

        # Verify panel structure maintained
        assert "feat1__group_1" in X_transformed.columns
        assert "feat2__group_1" in X_transformed.columns

        # Verify StandardScaler applied to feat1 (mean~0, std~1)
        feat1_g1 = X_transformed["feat1__group_1"].to_numpy()
        np.testing.assert_allclose(np.mean(feat1_g1), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.std(feat1_g1, ddof=0), 1.0, atol=1e-10)

        # Verify MinMaxScaler applied to feat2 (range [0, 1])
        feat2_g1 = X_transformed["feat2__group_1"].to_numpy()
        np.testing.assert_allclose(np.min(feat2_g1), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.max(feat2_g1), 1.0, atol=1e-10)

    @pytest.mark.integration
    def test_column_transformer_panel_independent_scaling(self):
        """Test that each panel group is scaled independently."""
        n = 100
        time_index = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        # Groups with different scales
        X_data = {
            "time": time_index,
            "feat__group_1": np.arange(n) * 1.0,  # Range [0, 99]
            "feat__group_2": np.arange(n) * 10.0,  # Range [0, 990]
        }
        X_actual = pl.DataFrame(X_data)

        transformer = StandardScaler()
        X_actual_train = X_actual[:80]
        X_transformed = transformer.fit_transform(X_actual_train)

        # Each group should be independently standardized
        for col in ["feat__group_1", "feat__group_2"]:
            values = X_transformed[col].to_numpy()
            np.testing.assert_allclose(np.mean(values), 0.0, atol=1e-10)
            np.testing.assert_allclose(np.std(values, ddof=0), 1.0, atol=1e-10)

    @pytest.mark.integration
    def test_column_transformer_panel_with_forecaster(self):
        """Test ColumnTransformer as actual_transformer in forecaster on panel."""
        n = 100
        time_index = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        y_data = {"time": time_index}
        X_data = {"time": time_index}

        for group_idx in range(2):
            y_data[f"g{group_idx}__target"] = np.arange(n) * (group_idx + 1)
            X_data[f"g{group_idx}__feature"] = np.arange(n) * (group_idx + 1) * 100

        y = pl.DataFrame(y_data)
        X_actual = pl.DataFrame(X_data)

        # Use ColumnTransformer to scale features before forecasting
        actual_transformer = ColumnTransformer(
            transformers=[
                ("scaler", StandardScaler(), ["feature"]),
            ],
            verbose_feature_names_out=False,
        )

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            actual_transformer=actual_transformer,
        )

        y_train = y[:80]
        X_actual_train = X_actual[:80]
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)

        # Verify panel structure maintained
        global_cols, panel_groups = inspect_panel(y_pred)
        assert len(panel_groups) == 2


class TestGridSearchPanel:
    """Tests for GridSearchCV parameter search and CV splitting on panel data."""

    @pytest.mark.integration
    def test_gridsearch_panel_parameter_search(self, linear_panel_5groups):
        """Test GridSearchCV searches over parameters with panel data."""
        y = linear_panel_5groups
        y_train = y[:80]

        forecaster = PointReductionForecaster(
            estimator=Ridge(),
        )

        param_grid = {
            "estimator__alpha": [0.1, 1.0, 10.0],
        }

        cv = SlidingWindowSplitter(
            train_size=40,
            test_size=5,
            stride=10,
        )

        search = GridSearchCV(
            forecaster,
            param_grid,
            cv=cv,
            scoring=MeanAbsoluteError(),
        )

        search.fit(y_train, forecasting_horizon=5)

        # Verify best forecaster found
        assert hasattr(search, "best_forecaster_")
        assert hasattr(search, "best_params_")

        # Predict via the search wrapper preserves all five panel groups.
        global_cols, panel_groups = inspect_panel(search.predict(forecasting_horizon=5))
        assert len(panel_groups) == 5

        # The unwrapped best_forecaster_ predicts the same panel structure.
        _, best_panel_groups = inspect_panel(search.best_forecaster_.predict(forecasting_horizon=5))
        assert len(best_panel_groups) == 5

    @pytest.mark.integration
    def test_gridsearch_panel_cv_splits_handle_panel(self, linear_panel_5groups):
        """Test that every CV split retains all panel groups."""
        y = linear_panel_5groups
        y_train = y[:80]

        cv = SlidingWindowSplitter(
            train_size=40,
            test_size=5,
            stride=10,
        )

        # Directly verify every train/test slice keeps all five panel groups,
        # rather than relying on "successful fit" to imply it.
        _, full_groups = inspect_panel(y_train)
        full_group_set = set(full_groups.keys())
        assert len(full_group_set) == 5

        splits = list(cv.split(y_train))
        assert len(splits) > 0
        for train_idx, test_idx in splits:
            _, train_groups = inspect_panel(y_train[train_idx.tolist()])
            _, test_groups = inspect_panel(y_train[test_idx.tolist()])
            assert set(train_groups.keys()) == full_group_set
            assert set(test_groups.keys()) == full_group_set

        # And confirm the search itself produces a finite score per param combo.
        param_grid = {"estimator__alpha": [0.1, 1.0]}
        search = GridSearchCV(
            PointReductionForecaster(estimator=Ridge()),
            param_grid,
            cv=cv,
            scoring=MeanAbsoluteError(),
        )
        search.fit(y_train, forecasting_horizon=5)
        assert len(search.cv_results_["params"]) == 2
        assert np.all(np.isfinite(search.cv_results_["mean_test_score"]))

    @pytest.mark.integration
    def test_gridsearch_panel_with_interval_forecaster(self):
        """Test GridSearchCV with interval forecaster on panel data."""
        # Import interval forecaster
        from yohou.interval import SplitConformalForecaster

        # Create simple panel
        n = 100
        time_index = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        y_data = {"time": time_index}
        for group_idx in range(2):
            y_data[f"g{group_idx}__value"] = np.arange(n) * (group_idx + 1) + np.random.normal(0, 1, n)

        y = pl.DataFrame(y_data)
        y_train = y[:80]

        forecaster = SplitConformalForecaster(
            point_forecaster=PointReductionForecaster(
                estimator=Ridge(),
            ),
            calibration_size=20,
        )

        param_grid = {
            "point_forecaster__estimator__alpha": [0.1, 1.0],
        }

        cv = SlidingWindowSplitter(
            train_size=40,
            test_size=5,
            stride=10,
        )

        from yohou.metrics import EmpiricalCoverage

        search = GridSearchCV(
            forecaster,
            param_grid,
            cv=cv,
            scoring=EmpiricalCoverage(),
        )

        search.fit(y_train, forecasting_horizon=5)

        # Predict intervals
        y_pred_interval = search.predict_interval(forecasting_horizon=5, coverage_rates=[0.9])

        # Verify panel structure
        global_cols, panel_groups = inspect_panel(y_pred_interval)
        assert len(panel_groups) == 2


class TestTenGroupPanelScalability:
    """Tests for panel data scalability with 10-group panels and subset operations."""

    @pytest.mark.integration
    def test_ten_group_panel_reduction_forecaster(self, linear_panel_10groups):
        """Test PointReductionForecaster scales to 10-group panel."""
        y = linear_panel_10groups
        y_train = y[:80]

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            actual_transformer=LagTransformer(lag=[1, 2, 3]),
        )

        import time

        start = time.time()
        forecaster.fit(y_train, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 10 seconds even on slow machines)
        assert elapsed < 10.0

        # Verify structure
        global_cols, panel_groups = inspect_panel(y_pred)
        assert len(panel_groups) == 10

    @pytest.mark.integration
    def test_ten_group_panel_observe_subset_at_scale(self, linear_panel_10groups):
        """Test observe with groups filtering at 10-group scale."""
        y = linear_panel_10groups
        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
        )

        y_train = y[:60]
        forecaster.fit(y_train, forecasting_horizon=5)

        # Update only groups 0, 3, 7
        y_update = y[60:80]
        subset_groups = ["g0", "g3", "g7"]
        forecaster.observe(y_update, groups=subset_groups)

        # Predict only the updated groups (partial update replaces internal state
        # for those groups only, making predict-all unavailable)
        y_pred = forecaster.predict(forecasting_horizon=5, groups=subset_groups)

        # Verify structure maintained for requested groups
        global_cols, panel_groups = inspect_panel(y_pred)
        assert len(panel_groups) == 3

    @pytest.mark.integration
    def test_ten_group_panel_predict_subset(self, linear_panel_10groups):
        """Test prediction with groups filtering at scale."""
        y = linear_panel_10groups
        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
        )

        y_train = y[:80]
        forecaster.fit(y_train, forecasting_horizon=5)

        # Predict only subset of groups
        subset_groups = ["g2", "g5", "g8"]
        y_pred_subset = forecaster.predict(
            forecasting_horizon=5,
            groups=subset_groups,
        )

        # Verify only requested groups present
        global_cols, panel_groups = inspect_panel(y_pred_subset)
        assert len(panel_groups) == 3
        assert "g2__value" in y_pred_subset.columns
        assert "g5__value" in y_pred_subset.columns
        assert "g8__value" in y_pred_subset.columns
        assert "g0__value" not in y_pred_subset.columns


class TestPanelGroupPreservation:
    """Tests for verbose feature name output and panel group structure preservation."""

    @pytest.mark.integration
    def test_feature_union_panel_verbose_preserves_groups(self):
        """Test FeatureUnion with verbose_feature_names_out=True preserves panel groups."""
        n = 100
        time_index = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        X_data = {
            "time": time_index,
            "group_1__feat": np.arange(n) * 1.0,
            "group_2__feat": np.arange(n) * 2.0,
        }
        X_actual = pl.DataFrame(X_data)

        transformer = FeatureUnion(
            transformer_list=[
                ("scaler", StandardScaler()),
            ],
            verbose_feature_names_out=True,
        )

        X_actual_train = X_actual[:80]
        X_transformed = transformer.fit_transform(X_actual_train)

        # Panel groups must be preserved
        _, panel_groups = inspect_panel(X_transformed)
        assert set(panel_groups.keys()) == {"group_1", "group_2"}, f"Panel groups not preserved: {panel_groups.keys()}"

        # Column names must use _ separator, not __
        for col in X_transformed.columns:
            if col == "time":
                continue
            parts = col.split("__")
            assert len(parts) <= 2, f"Column '{col}' has multiple __ separators after FeatureUnion prefixing"

    @pytest.mark.integration
    def test_column_transformer_panel_verbose_preserves_groups(self):
        """Test ColumnTransformer with verbose_feature_names_out=True preserves panel groups."""
        n = 100
        time_index = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        X_data = {
            "time": time_index,
            "group_1__feat1": np.arange(n) * 1.0,
            "group_2__feat1": np.arange(n) * 2.0,
            "group_1__feat2": np.arange(n) * 0.5,
            "group_2__feat2": np.arange(n) * 1.5,
        }
        X_actual = pl.DataFrame(X_data)

        transformer = ColumnTransformer(
            transformers=[
                ("scaler1", StandardScaler(), ["group_1__feat1", "group_2__feat1"]),
                ("scaler2", MinMaxScaler(), ["group_1__feat2", "group_2__feat2"]),
            ],
            verbose_feature_names_out=True,
        )

        X_actual_train = X_actual[:80]
        X_transformed = transformer.fit_transform(X_actual_train)

        # Panel groups must be preserved
        _, panel_groups = inspect_panel(X_transformed)
        assert set(panel_groups.keys()) == {"group_1", "group_2"}, f"Panel groups not preserved: {panel_groups.keys()}"

        # Column names must use _ separator, not __
        for col in X_transformed.columns:
            if col == "time":
                continue
            parts = col.split("__")
            assert len(parts) <= 2, f"Column '{col}' has multiple __ separators after ColumnTransformer prefixing"

    @pytest.mark.integration
    def test_stationarity_transformer_panel_preserves_groups(self):
        """Test stationarity transformers preserve panel groups on standalone panel data."""
        from yohou.stationarity import ASinhTransformer, LogTransformer, SeasonalDifferencing

        n = 100
        time_index = pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )

        X_data = {
            "time": time_index,
            "store_1__sales": np.abs(np.arange(n) * 1.0) + 1.0,
            "store_2__sales": np.abs(np.arange(n) * 2.0) + 1.0,
        }
        X_actual = pl.DataFrame(X_data)

        for transformer_cls, kwargs in [
            (LogTransformer, {"offset": 0.0}),
            (ASinhTransformer, {}),
            (SeasonalDifferencing, {"seasonality": 7}),
        ]:
            transformer = transformer_cls(**kwargs)
            X_transformed = transformer.fit_transform(X_actual)

            _, panel_groups = inspect_panel(X_transformed)
            assert set(panel_groups.keys()) == {"store_1", "store_2"}, (
                f"{transformer_cls.__name__} changed panel groups: "
                f"expected {{'store_1', 'store_2'}}, got {set(panel_groups.keys())}. "
                f"Output columns: {X_transformed.columns}"
            )
