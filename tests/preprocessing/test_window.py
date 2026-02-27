"""Tests for window-based transformers.

Tests LagTransformer using the check generator pattern for systematic validation.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.base import clone

from conftest import run_checks
from yohou.preprocessing.window import (
    LagTransformer,
    RollingStatisticsTransformer,
    SlidingWindowFunctionTransformer,
)
from yohou.testing import _yield_yohou_transformer_checks


class TestLagTransformerSystematic:
    """Systematic checks for LagTransformer."""

    @pytest.mark.parametrize(
        "lag",
        [[1], [1, 2], [1, 2, 3, 5]],
        ids=["lag_1", "lag_1_2", "lag_1_2_3_5"],
    )
    def test_systematic_checks(self, lag, time_series_train_test_factory):
        """Run all checks for LagTransformer with different lag configurations."""
        transformer = LagTransformer(lag=lag)
        expected_failures = []  # Empty since invertible=False prevents inverse checks

        min_horizon = max(lag) + 10
        X_train, X_test = time_series_train_test_factory(train_length=min_horizon + 50, test_length=min_horizon + 20)

        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )


class TestLagTransformer:
    """Functional tests for LagTransformer."""

    def test_feature_names(self, time_series_factory):
        """Test LagTransformer generates correct feature names."""
        X = time_series_factory(length=50, n_components=2)
        transformer = LagTransformer(lag=[1, 2])
        transformer.fit(X)

        transformer.transform(X)
        feature_names = transformer.get_feature_names_out()

        # Should have lag_1 and lag_2 for each input feature
        expected_n_features = 2 * 2  # 2 features * 2 lags
        assert len(feature_names) == expected_n_features, (
            f"Expected {expected_n_features} features, got {len(feature_names)}"
        )

        # Check feature naming pattern
        assert all("lag" in name.lower() for name in feature_names), "All features should contain 'lag' in name"

    def test_observation_horizon(self, time_series_factory):
        """Test observation_horizon equals max(lag) + 1."""
        X = time_series_factory(length=50)

        for lag in [[1], [1, 2], [1, 2, 3], [1, 5, 10]]:
            transformer = LagTransformer(lag=lag)
            transformer.fit(X)

            # LagTransformer uses max(lag) + 1 as observation_horizon
            expected_horizon = max(lag)
            assert transformer.observation_horizon == expected_horizon, (
                f"For lag={lag}, expected horizon={expected_horizon}, got {transformer.observation_horizon}"
            )

    def test_output_length(self, time_series_factory):
        """Test output length drops max(lag) rows."""
        X = time_series_factory(length=50)
        transformer = LagTransformer(lag=[1, 2, 3])
        transformer.fit(X)

        X_trans = transformer.transform(X)

        # Output drops max(lag) rows (not observation_horizon)
        expected_length = len(X) - max([1, 2, 3])
        assert len(X_trans) == expected_length, f"Expected output length {expected_length}, got {len(X_trans)}"

    def test_single_lag(self, time_series_factory):
        """Test LagTransformer with single lag value."""
        X = time_series_factory(length=50, n_components=1)
        transformer = LagTransformer(lag=[1])
        transformer.fit(X)

        X_trans = transformer.transform(X)

        # Should have time + lagged features
        assert "time" in X_trans.columns
        assert len([col for col in X_trans.columns if col != "time"]) == 1

    def test_with_panel_data(self, panel_time_series_factory):
        """Test LagTransformer handles panel data."""
        X_panel = panel_time_series_factory(length=50, n_series=3, n_global=2)
        transformer = LagTransformer(lag=[1, 2])

        transformer.fit(X_panel)
        X_trans = transformer.transform(X_panel)

        # Basic validation
        assert "time" in X_trans.columns
        # Output drops max(lag) rows
        expected_length = len(X_panel) - max([1, 2])
        assert len(X_trans) == expected_length, f"Expected {expected_length} rows, got {len(X_trans)}"


@pytest.fixture
def window_data_factory():
    """Fixture factory providing time series data for window tests."""

    def _create(length: int = 50, seed: int = 42) -> pl.DataFrame:
        """Create time series data for testing.

        Parameters
        ----------
        length : int
            Number of samples.
        seed : int
            Random seed.

        Returns
        -------
        pl.DataFrame
            DataFrame with time column and numeric columns.

        """
        np.random.seed(seed)
        time = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(length)]
        return pl.DataFrame({
            "time": time,
            "value": np.cumsum(np.random.randn(length)).tolist(),
            "other": np.cumsum(np.random.randn(length)).tolist(),
        })

    return _create


class TestSlidingWindowFunctionTransformerBasic:
    """Basic functionality tests for SlidingWindowFunctionTransformer."""

    def test_basic_rolling_mean(self, window_data_factory):
        """Test basic rolling mean computation."""
        X = window_data_factory(length=20)

        def rolling_mean(window):
            return window.select(pl.all().exclude("time").mean()).to_numpy().flatten()

        transformer = SlidingWindowFunctionTransformer(func=rolling_mean, window_size=3)
        transformer.fit(X)

        X_t = transformer.transform(X)

        # Output should have (length - window_size + 1) rows
        assert len(X_t) == 20 - 3 + 1  # 18 rows

        # Time should be preserved
        assert "time" in X_t.columns

        # Value column should be present
        assert "value" in X_t.columns

    def test_window_size_1(self, window_data_factory):
        """Test window size 1 (identity)."""
        X = window_data_factory(length=20)

        def identity(window):
            return window.select(pl.all().exclude("time")).to_numpy().flatten()

        transformer = SlidingWindowFunctionTransformer(func=identity, window_size=1)
        transformer.fit(X)

        X_t = transformer.transform(X)

        # Output length should equal input
        assert len(X_t) == len(X)

    def test_observation_horizon(self, window_data_factory):
        """Test observation_horizon is set correctly."""
        X = window_data_factory()

        def rolling_mean(window):
            return window.select(pl.all().exclude("time").mean()).to_numpy().flatten()

        transformer = SlidingWindowFunctionTransformer(func=rolling_mean, window_size=5)
        transformer.fit(X)

        # observation_horizon should be window_size - 1
        assert transformer.observation_horizon == 4

    def test_kw_args(self, window_data_factory):
        """Test kw_args passed to function."""
        X = window_data_factory(length=20)

        def power_mean(window, power=1):
            # Calculate mean then apply power
            data = window.select(pl.all().exclude("time")).to_numpy()
            mean_val = np.mean(data, axis=0)
            return mean_val**power

        transformer = SlidingWindowFunctionTransformer(func=power_mean, window_size=3, kw_args={"power": 2})
        transformer.fit(X)

        X_t = transformer.transform(X)

        assert len(X_t) == 18

    def test_stateful_tag(self):
        """Test that transformer is marked as stateful."""

        def identity(w):
            return w.select(pl.all().exclude("time")).to_numpy()[0]

        transformer = SlidingWindowFunctionTransformer(func=identity, window_size=3)
        tags = transformer.__sklearn_tags__()

        assert tags.transformer_tags.stateful is True

    def test_clone(self):
        """Test cloning preserves parameters."""

        def identity(w):
            return w.select(pl.all().exclude("time")).to_numpy()[0]

        transformer = SlidingWindowFunctionTransformer(func=identity, window_size=5)
        cloned = clone(transformer)

        assert cloned.func is transformer.func
        assert cloned.window_size == transformer.window_size


def _identity_func(window):
    """Identity function for testing - returns last row."""
    return window.select(pl.all().exclude("time")).to_numpy()[-1]


def _rolling_mean_func(window):
    """Rolling mean function for testing."""
    return window.select(pl.all().exclude("time").mean()).to_numpy().flatten()


class TestSlidingWindowFunctionTransformerSystematic:
    """Systematic checks for SlidingWindowFunctionTransformer."""

    @pytest.mark.parametrize(
        "transformer,expected_failures",
        [
            # SlidingWindowFunctionTransformer doesn't have inverse_transform
            (
                SlidingWindowFunctionTransformer(func=_identity_func, window_size=3),
                ["check_inverse_transform_identity"],
            ),
            (
                SlidingWindowFunctionTransformer(func=_rolling_mean_func, window_size=5),
                ["check_inverse_transform_identity"],
            ),
        ],
        ids=["identity_3", "rolling_mean_5"],
    )
    def test_systematic_checks(
        self,
        transformer,
        expected_failures,
        time_series_train_test_factory,
    ):
        """Run all applicable checks for SlidingWindowFunctionTransformer."""
        X_train, X_test = time_series_train_test_factory(
            train_length=60,
            test_length=30,
        )

        # Fit transformer
        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )


class TestRollingStatisticsTransformerSystematic:
    """Systematic checks for RollingStatisticsTransformer."""

    @pytest.mark.parametrize(
        "transformer,expected_failures",
        [
            # Rolling statistics transformers don't have inverse_transform
            (RollingStatisticsTransformer(window_size=3, statistics="mean"), ["check_inverse_transform_identity"]),
            (
                RollingStatisticsTransformer(window_size=5, statistics=["mean", "std"]),
                ["check_inverse_transform_identity"],
            ),
        ],
        ids=["rolling_mean_3", "rolling_multi_5"],
    )
    def test_systematic_checks(
        self,
        transformer,
        expected_failures,
        time_series_train_test_factory,
    ):
        """Run all applicable checks for RollingStatisticsTransformer."""
        X_train, X_test = time_series_train_test_factory(
            train_length=60,
            test_length=30,
        )

        # Fit transformer
        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )


class TestRollingStatisticsTransformerBasic:
    """Basic functionality tests for RollingStatisticsTransformer."""

    def test_single_statistic_mean(self, window_data_factory):
        """Test single rolling mean statistic."""
        X = window_data_factory(length=20)
        transformer = RollingStatisticsTransformer(window_size=3, statistics="mean")
        transformer.fit(X)

        X_t = transformer.transform(X)

        # Output columns should be renamed
        assert "value_mean" in X_t.columns
        assert "other_mean" in X_t.columns
        assert "time" in X_t.columns

        # Original columns should not be present
        assert "value" not in X_t.columns
        assert "other" not in X_t.columns

        # Warmup rows are dropped (observation_horizon = window_size - 1)
        assert len(X_t) == len(X) - transformer.observation_horizon

    def test_multiple_statistics(self, window_data_factory):
        """Test multiple rolling statistics."""
        X = window_data_factory(length=20)
        transformer = RollingStatisticsTransformer(window_size=3, statistics=["mean", "std", "min", "max"])
        transformer.fit(X)

        X_t = transformer.transform(X)

        # Should have all statistics for each column
        for col in ["value", "other"]:
            for stat in ["mean", "std", "min", "max"]:
                assert f"{col}_{stat}" in X_t.columns

        # Total: time + 2 columns * 4 stats = 9 columns
        assert len(X_t.columns) == 9

    def test_all_statistics(self, window_data_factory):
        """Test all supported statistics."""
        X = window_data_factory(length=50)
        transformer = RollingStatisticsTransformer(
            window_size=3,
            statistics=["mean", "std", "min", "max", "median", "sum", "var", "q25", "q75"],
        )
        transformer.fit(X)

        X_t = transformer.transform(X)

        for stat in ["mean", "std", "min", "max", "median", "sum", "var", "q25", "q75"]:
            assert f"value_{stat}" in X_t.columns


class TestRollingStatisticsTransformerValues:
    """Test actual computed values of RollingStatisticsTransformer."""

    def test_rolling_mean_values(self):
        """Test rolling mean computes correct values."""
        time = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(10)]
        X = pl.DataFrame({
            "time": time,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        })

        transformer = RollingStatisticsTransformer(window_size=3, statistics="mean")
        transformer.fit(X)

        X_t = transformer.transform(X)

        # Warmup rows are dropped; first value is the first valid rolling mean
        # mean of [1, 2, 3] = 2.0
        assert X_t["value_mean"][0] == 2.0

        # Second value should be mean of [2, 3, 4] = 3.0
        assert X_t["value_mean"][1] == 3.0

    def test_rolling_sum_values(self):
        """Test rolling sum computes correct values."""
        time = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(10)]
        X = pl.DataFrame({
            "time": time,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        })

        transformer = RollingStatisticsTransformer(window_size=3, statistics="sum")
        transformer.fit(X)

        X_t = transformer.transform(X)

        # First value should be sum of [1, 2, 3] = 6.0 (warmup rows dropped)
        assert X_t["value_sum"][0] == 6.0

        # Second value should be sum of [2, 3, 4] = 9.0
        assert X_t["value_sum"][1] == 9.0


class TestRollingStatisticsTransformerParams:
    """Test parameter handling for RollingStatisticsTransformer."""

    def test_min_samples_equals_window_size(self, window_data_factory):
        """Test min_samples defaults to window_size."""
        X = window_data_factory(length=20)
        transformer = RollingStatisticsTransformer(window_size=5, statistics="mean")
        transformer.fit(X)

        X_t = transformer.transform(X)

        # Warmup rows are dropped; no nulls remain, first value is valid
        assert X_t["value_mean"][0] is not None
        assert len(X_t) == len(X) - transformer.observation_horizon

    def test_invalid_statistic(self, window_data_factory):
        """Test invalid statistic raises ValueError."""
        X = window_data_factory()
        transformer = RollingStatisticsTransformer(window_size=3, statistics="invalid")

        with pytest.raises(ValueError, match="Invalid statistics"):
            transformer.fit(X)

    def test_observation_horizon(self, window_data_factory):
        """Test observation_horizon is set correctly."""
        X = window_data_factory()
        transformer = RollingStatisticsTransformer(window_size=5, statistics="mean")
        transformer.fit(X)

        assert transformer.observation_horizon == 4  # window_size - 1


class TestRollingStatisticsTransformerFeatureNames:
    """Test feature name handling for RollingStatisticsTransformer."""

    def test_feature_names_single_stat(self, window_data_factory):
        """Test feature names with single statistic."""
        X = window_data_factory()
        transformer = RollingStatisticsTransformer(window_size=3, statistics="mean")
        transformer.fit(X)

        names = transformer.get_feature_names_out()

        assert list(names) == ["value_mean", "other_mean"]

    def test_feature_names_multiple_stats(self, window_data_factory):
        """Test feature names with multiple statistics."""
        X = window_data_factory()
        transformer = RollingStatisticsTransformer(window_size=3, statistics=["mean", "std"])
        transformer.fit(X)

        names = transformer.get_feature_names_out()

        assert list(names) == ["value_mean", "value_std", "other_mean", "other_std"]


class TestRollingStatisticsTransformerSklearn:
    """Test sklearn compatibility for RollingStatisticsTransformer."""

    def test_clone(self):
        """Test cloning preserves parameters."""
        transformer = RollingStatisticsTransformer(window_size=5, statistics=["mean", "std"])
        cloned = clone(transformer)

        assert cloned.window_size == transformer.window_size
        assert cloned.statistics == transformer.statistics


class TestWindowTransformersIntegration:
    """Integration tests for window transformers."""

    def test_rolling_mean_consistency(self, window_data_factory):
        """Test RollingStatisticsTransformer produces same results as manual computation."""
        X = window_data_factory(length=20)

        # Using RollingStatisticsTransformer
        transformer = RollingStatisticsTransformer(window_size=3, statistics="mean")
        transformer.fit(X)
        X_t = transformer.transform(X)

        # Manual computation (drop warmup rows to match transformer output)
        manual = X.select([
            pl.col("time"),
            pl.col("value").rolling_mean(3).alias("value_mean"),
            pl.col("other").rolling_mean(3).alias("other_mean"),
        ])[transformer.observation_horizon :]

        assert_frame_equal(X_t, manual)

    def test_sliding_vs_rolling_mean(self, window_data_factory):
        """Compare SlidingWindowFunctionTransformer and RollingStatisticsTransformer for mean."""
        X = window_data_factory(length=20)

        # Using SlidingWindowFunctionTransformer
        def rolling_mean(window):
            return window.select(pl.all().exclude("time").mean()).to_numpy().flatten()

        swft = SlidingWindowFunctionTransformer(func=rolling_mean, window_size=3)
        swft.fit(X)
        X_swft = swft.transform(X)

        # Using RollingStatisticsTransformer
        rst = RollingStatisticsTransformer(window_size=3, statistics="mean")
        rst.fit(X)
        X_rst = rst.transform(X)

        # Both transformers now drop warmup rows, so lengths should match
        np.testing.assert_allclose(
            X_swft["value"].to_numpy(),
            X_rst["value_mean"].to_numpy(),
            rtol=1e-10,
        )
