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

from .conftest import hourly_frame


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


class TestRollingStatisticsTransformerSeasonalSystematic:
    """Systematic checks for RollingStatisticsTransformer with a seasonal window."""

    @pytest.mark.parametrize(
        ("window_size", "seasonality", "statistics"),
        [
            (3, 4, "mean"),
            (3, 24, ["mean", "std"]),
            (7, 4, ["min", "max", "median", "q25"]),
            (7, 24, ["sum", "var", "q75"]),
        ],
        ids=["w3_s4", "w3_s24", "w7_s4", "w7_s24"],
    )
    def test_systematic_checks(self, window_size, seasonality, statistics, time_series_train_test_factory):
        """Run all applicable checks, including batch invariance and observe/rewind."""
        horizon = (window_size - 1) * seasonality
        # Several checks fit on the first half of X_train, so it needs twice the horizon.
        X_train, X_test = time_series_train_test_factory(train_length=2 * horizon + 60, test_length=horizon + 30)
        transformer = RollingStatisticsTransformer(
            window_size=window_size, statistics=statistics, seasonality=seasonality
        )
        transformer.fit(X_train)

        run_checks(transformer, _yield_yohou_transformer_checks(transformer, X_train, None, X_test))


class TestRollingStatisticsTransformerSeasonal:
    """Seasonal window behaviour of RollingStatisticsTransformer."""

    def test_default_unchanged(self):
        """seasonality=1 matches a plain polars rolling window, names and row count included."""
        X = hourly_frame(100)
        X_t = RollingStatisticsTransformer(window_size=24, statistics=["mean", "std"]).fit(X).transform(X)

        expected = X.select(
            "time",
            pl.col("price").rolling_mean(24).alias("price_mean"),
            pl.col("price").rolling_std(24).alias("price_std"),
        )[23:]
        assert_frame_equal(X_t, expected)

    def test_seasonal_window_values(self):
        """The value at t is the statistic over x[t], x[t-24], x[t-48]."""
        X = hourly_frame(200)
        transformer = RollingStatisticsTransformer(window_size=3, seasonality=24, statistics="mean").fit(X)
        X_t = transformer.transform(X)

        assert transformer.observation_horizon == 48
        assert X_t.height == 200 - 48
        assert X_t["time"][0] == X["time"][48]
        price = X["price"].to_numpy()
        expected = [np.mean([price[t], price[t - 24], price[t - 48]]) for t in range(48, 200)]
        np.testing.assert_allclose(X_t["price_s24_mean"].to_numpy(), expected, rtol=1e-12)

    def test_with_panel_data(self, panel_time_series_factory):
        """Group prefixes survive the seasonal rename and each panel column rolls on its own."""
        X_panel = panel_time_series_factory(length=60, n_series=1, n_groups=2)
        transformer = RollingStatisticsTransformer(window_size=3, seasonality=4, statistics="mean")
        X_t = transformer.fit(X_panel).transform(X_panel)

        assert transformer.observation_horizon == 8
        assert set(X_t.columns) == {"time", "group0__series_0_s4_mean", "group1__series_0_s4_mean"}
        assert X_t.height == X_panel.height - 8
        values = X_panel["group1__series_0"].to_numpy()
        expected = [np.mean([values[t], values[t - 4], values[t - 8]]) for t in range(8, X_panel.height)]
        np.testing.assert_allclose(X_t["group1__series_0_s4_mean"].to_numpy(), expected, rtol=1e-12)

    def test_seasonality_zero_rejected(self):
        """seasonality must be a positive integer."""
        X = hourly_frame(50)
        with pytest.raises(ValueError, match="seasonality"):
            RollingStatisticsTransformer(seasonality=0).fit(X)

    def test_window_size_not_in_names(self):
        """Tuning the window size leaves the output names unchanged."""
        X = hourly_frame(500)
        names_7 = RollingStatisticsTransformer(window_size=7, seasonality=24).fit(X).get_feature_names_out()
        names_14 = RollingStatisticsTransformer(window_size=14, seasonality=24).fit(X).get_feature_names_out()
        assert names_7 == names_14 == ["price_s24_mean"]

    def test_seasonal_and_consecutive_names_distinct(self):
        """A union of a consecutive and a seasonal window on one column keeps both columns."""
        from yohou.compose import FeatureUnion

        X = hourly_frame(300)
        union = FeatureUnion(
            [
                ("consecutive", RollingStatisticsTransformer(window_size=24)),
                ("seasonal", RollingStatisticsTransformer(window_size=7, seasonality=24)),
            ],
            verbose_feature_names_out=False,
        )
        X_t = union.fit(X).transform(X)
        assert {"price_mean", "price_s24_mean"} <= set(X_t.columns)

    def test_lag_composition_reproduces_seasonal_lag_mean(self):
        """Lag then seasonal rolling mean equals mean(x[t-24], x[t-48], x[t-72])."""
        from yohou.compose import FeaturePipeline

        X = hourly_frame(300)
        pipeline = FeaturePipeline([
            ("lag", LagTransformer(lag=24)),
            ("mean", RollingStatisticsTransformer(window_size=3, seasonality=24)),
        ])
        X_t = pipeline.fit(X).transform(X)
        price = X["price"].to_numpy()
        offset = X.height - X_t.height
        expected = [np.mean([price[t - 24], price[t - 48], price[t - 72]]) for t in range(offset, X.height)]
        (column,) = [c for c in X_t.columns if c != "time"]
        np.testing.assert_allclose(X_t[column].to_numpy(), expected, rtol=1e-12)
        assert offset == 72


class TestRollingStatisticsTransformerWindowMinimum:
    """Sample statistics need two values, so a single-value window is rejected."""

    @pytest.mark.parametrize("statistics", ["std", "var", ["mean", "var"]])
    def test_single_value_sample_statistic_rejected(self, statistics):
        """std and var over a window of one value raise at fit."""
        transformer = RollingStatisticsTransformer(window_size=1, statistics=statistics)
        with pytest.raises(ValueError, match="set window_size >= 2"):
            transformer.fit(hourly_frame(200))

    @pytest.mark.parametrize("seasonality", [1, 24])
    @pytest.mark.parametrize("statistic", ["mean", "min", "max", "median", "sum", "q25", "q75"])
    def test_single_value_window_accepted_for_other_statistics(self, statistic, seasonality):
        """Every statistic defined over one value still runs, with no nulls."""
        X_t = RollingStatisticsTransformer(window_size=1, statistics=statistic, seasonality=seasonality).fit_transform(
            hourly_frame(200)
        )
        assert all(X_t[c].null_count() == 0 for c in X_t.columns if c != "time")

    @pytest.mark.parametrize("seasonality", [1, 24])
    def test_two_value_window_has_no_nulls(self, seasonality):
        """The smallest accepted window gives std and var a value in every row."""
        X_t = RollingStatisticsTransformer(
            window_size=2, statistics=["std", "var"], seasonality=seasonality
        ).fit_transform(hourly_frame(200))
        assert all(X_t[c].null_count() == 0 for c in X_t.columns if c != "time")

    def test_failed_refit_keeps_statistics(self):
        """A rejected refit leaves the previously fitted statistics in place."""
        transformer = RollingStatisticsTransformer(window_size=3, statistics="mean").fit(hourly_frame(200))
        transformer.set_params(window_size=1, statistics=["std"])
        with pytest.raises(ValueError, match="set window_size >= 2"):
            transformer.fit(hourly_frame(200))
        assert transformer.statistics_ == ["mean"]


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


class TestExponentialMovingAverageSystematic:
    """Systematic checks for ExponentialMovingAverage.

    ``test_common.py`` skips EMA because its ``__init__`` requires ``alpha``
    (no default), so the generic suite never exercises it. Wire it here.
    """

    @pytest.mark.parametrize(
        "alpha,adjust",
        [(0.3, True), (0.5, False)],
        ids=["alpha_0_3_adjust", "alpha_0_5_no_adjust"],
    )
    def test_systematic_checks(self, alpha, adjust, time_series_train_test_factory):
        """Run all applicable checks for ExponentialMovingAverage."""
        from yohou.preprocessing.window import ExponentialMovingAverage

        # EMA is stateless (observation_horizon=0) and not invertible, so no
        # expected failures are needed.
        transformer = ExponentialMovingAverage(alpha=alpha, adjust=adjust)
        X_train, X_test = time_series_train_test_factory(train_length=60, test_length=30)

        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(),
        )


class TestExponentialMovingAverage:
    """Tests for ExponentialMovingAverage transformer."""

    def test_fit_returns_self(self, window_data_factory):
        """Fit returns the transformer instance."""
        from yohou.preprocessing.window import ExponentialMovingAverage

        X = window_data_factory(length=20)
        ema = ExponentialMovingAverage(alpha=0.3)
        result = ema.fit(X)
        assert result is ema

    def test_transform_produces_ewma_columns(self, window_data_factory):
        """Transform output has _ewma suffix on feature columns."""
        from yohou.preprocessing.window import ExponentialMovingAverage

        X = window_data_factory(length=20)
        ema = ExponentialMovingAverage(alpha=0.3)
        ema.fit(X)
        X_t = ema.transform(X)
        assert "time" in X_t.columns
        non_time = [c for c in X_t.columns if c != "time"]
        assert all(c.endswith("_ewma") for c in non_time)

    def test_transform_preserves_row_count(self, window_data_factory):
        """Transform output has same number of rows as input."""
        from yohou.preprocessing.window import ExponentialMovingAverage

        X = window_data_factory(length=30)
        ema = ExponentialMovingAverage(alpha=0.5)
        ema.fit(X)
        X_t = ema.transform(X)
        assert len(X_t) == len(X)

    def test_get_feature_names_out(self, window_data_factory):
        """Feature names output have _ewma suffix."""
        from yohou.preprocessing.window import ExponentialMovingAverage

        X = window_data_factory(length=20)
        ema = ExponentialMovingAverage(alpha=0.3)
        ema.fit(X)
        names = ema.get_feature_names_out()
        assert all(n.endswith("_ewma") for n in names)

    def test_alpha_affects_smoothing(self, window_data_factory):
        """Higher alpha gives less smoothing (closer to raw values)."""
        from yohou.preprocessing.window import ExponentialMovingAverage

        X = window_data_factory(length=30)

        ema_low = ExponentialMovingAverage(alpha=0.1)
        ema_low.fit(X)
        X_low = ema_low.transform(X)

        ema_high = ExponentialMovingAverage(alpha=0.9)
        ema_high.fit(X)
        X_high = ema_high.transform(X)

        ewma_col_low = [c for c in X_low.columns if c.endswith("_ewma")][0]
        ewma_col_high = [c for c in X_high.columns if c.endswith("_ewma")][0]

        raw_col = [c for c in X.columns if c != "time"][0]
        raw_vals = X[raw_col].to_numpy()

        low_vals = X_low[ewma_col_low].to_numpy()
        high_vals = X_high[ewma_col_high].to_numpy()

        low_diff = np.abs(raw_vals - low_vals).mean()
        high_diff = np.abs(raw_vals - high_vals).mean()
        assert low_diff > high_diff

    def test_adjust_parameter(self, window_data_factory):
        """adjust=False uses recursive EWM instead of bias-corrected."""
        from yohou.preprocessing.window import ExponentialMovingAverage

        X = window_data_factory(length=20)
        ema = ExponentialMovingAverage(alpha=0.3, adjust=False)
        ema.fit(X)
        X_t = ema.transform(X)
        assert len(X_t) == len(X)

    def test_multiple_columns(self, time_series_factory):
        """Transform handles multiple numeric columns."""
        from yohou.preprocessing.window import ExponentialMovingAverage

        X = time_series_factory(length=20, n_components=3)
        ema = ExponentialMovingAverage(alpha=0.3)
        ema.fit(X)
        X_t = ema.transform(X)
        ewma_cols = [c for c in X_t.columns if c.endswith("_ewma")]
        assert len(ewma_cols) == 3


def test_mean_lag_transformer_removed():
    """MeanLagTransformer is gone; its seasonal lag average is a LagTransformer + seasonal rolling mean."""
    with pytest.raises(ImportError):
        from yohou.preprocessing import MeanLagTransformer  # noqa: F401
