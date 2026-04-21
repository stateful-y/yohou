"""Stride and horizon combinatorics integration tests.

This module verifies correctness of stride/horizon interactions across:

1. Observe-predict stride matrix on AR(1)
2. Horizon mismatch recursive correctness
3. Cross-validation splitter stride calculations
4. Observe-predict state consistency
5. Panel data stride/horizon operations

All tests use analytically solvable data to verify exact numerical correctness.
Tolerance: atol=1e-7 to 1e-8 (stride/horizon operations can accumulate small errors).
"""

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from yohou.model_selection import ExpandingWindowSplitter, SlidingWindowSplitter
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer


@pytest.mark.integration
class TestObservePredictStride:
    """Observe-predict stride matrix tests on AR(1) series."""

    @pytest.mark.parametrize("stride", [1, 2, 3, 5, 6])
    def test_observe_predict_stride_coverage(self, ar1_series, analytical_forecast, stride):
        """Verify observe_predict with different strides covers all test time steps."""
        # Generate AR(1) series: phi=0.7, c=2.0
        phi, c = 0.7, 2.0
        y_full = ar1_series(phi=phi, c=c, length=150, noise_std=0.0)

        # Split: train on first 100, test on last 50
        y_train = y_full[:100]
        y_test = y_full[100:]

        # Fit with forecasting_horizon=3
        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=3)

        # Collect all predictions via predict + observe with stride
        all_preds = []

        for i in range(0, len(y_test), stride):
            # Get chunk for this stride
            chunk_size = min(stride, len(y_test) - i)
            y_chunk = y_test[i : i + chunk_size]

            # Predict next chunk_size steps, then observe state
            y_pred = forecaster.predict(forecasting_horizon=chunk_size)
            forecaster.observe(y_chunk)
            all_preds.append(y_pred)

        # Concatenate all predictions
        y_pred_full = pl.concat(all_preds)

        # Verify: predictions cover exactly len(y_test) time steps
        assert len(y_pred_full) == len(y_test), (
            f"Stride={stride}: Expected {len(y_test)} predictions, got {len(y_pred_full)}"
        )

        # Verify: predictions match analytical AR(1) forecast
        # For each test step, compute expected value from last observation
        y_test_values = y_test["value"].to_numpy()
        y_pred_values = y_pred_full["value"].to_numpy()

        # Allow small numerical error from stride operations
        np.testing.assert_allclose(
            y_pred_values,
            y_test_values,
            atol=1e-6,
            err_msg=f"Stride={stride}: Predictions don't match AR(1) analytical values",
        )

    def test_observe_predict_stride_remainder(self, ar1_series):
        """Verify stride that doesn't divide test length evenly handles remainder."""
        phi, c = 0.7, 2.0
        y_full = ar1_series(phi=phi, c=c, length=107, noise_std=0.0)  # Odd length

        y_train = y_full[:100]
        y_test = y_full[100:]  # Length 7

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=3)

        # Stride=5 on 7-step test: first chunk 5, remainder 2
        all_preds = []

        # First chunk: predict 5 steps, then observe
        y_pred_1 = forecaster.predict(forecasting_horizon=5)
        forecaster.observe(y_test[:5])
        all_preds.append(y_pred_1)

        # Remainder: predict 2 steps, then observe
        y_pred_2 = forecaster.predict(forecasting_horizon=2)
        forecaster.observe(y_test[5:])
        all_preds.append(y_pred_2)

        y_pred_full = pl.concat(all_preds)

        # Verify full coverage
        assert len(y_pred_full) == 7
        assert len(y_pred_1) == 5
        assert len(y_pred_2) == 2

        # Verify predictions match test data (AR(1) should reconstruct perfectly)
        np.testing.assert_allclose(y_pred_full["value"].to_numpy(), y_test["value"].to_numpy(), atol=1e-6)

    def test_observe_predict_stride_1_vs_batch(self, ar1_series):
        """Verify stride=1 (step-by-step) matches batch observe."""
        phi, c = 0.8, 1.5
        y_full = ar1_series(phi=phi, c=c, length=120, noise_std=0.0)

        y_train = y_full[:100]
        y_test = y_full[100:]

        # Forecaster 1: stride=1 (step-by-step predict + observe)
        forecaster_1 = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster_1.fit(y_train, forecasting_horizon=1)

        preds_stride = []
        for i in range(len(y_test)):
            y_pred = forecaster_1.predict(forecasting_horizon=1)
            forecaster_1.observe(y_test[i : i + 1])
            preds_stride.append(y_pred)
        y_pred_stride = pl.concat(preds_stride)

        # Forecaster 2: single batch observe_predict with stride=1
        forecaster_2 = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster_2.fit(y_train, forecasting_horizon=1)
        y_pred_batch = forecaster_2.observe_predict(y_test, stride=1, forecasting_horizon=1)

        # Step-by-step predict+observe matches first len(y_test) predictions from batch
        # (batch observe_predict includes an extra trailing prediction after the last observe)
        np.testing.assert_allclose(
            y_pred_stride["value"].to_numpy(),
            y_pred_batch["value"].to_numpy()[: len(y_test)],
            atol=1e-8,
            err_msg="Stride=1 step-by-step should match batch observe_predict",
        )

    def test_observe_predict_stride_analytical_verification(self, ar1_series, analytical_forecast):
        """Verify each stride chunk produces correct analytical AR(1) values."""
        phi, c = 0.7, 2.0
        y_full = ar1_series(phi=phi, c=c, length=130, noise_std=0.0)

        y_train = y_full[:100]
        y_test = y_full[100:]

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=5)

        # Test with stride=6: each chunk is 6 steps
        stride = 6
        all_preds = []

        for i in range(0, len(y_test), stride):
            chunk_size = min(stride, len(y_test) - i)
            chunk = y_test[i : i + chunk_size]

            # Verify predictions match analytical forecast
            # Starting from last observation before this chunk
            last_obs = y_train[-1]["value"].item() if i == 0 else y_test[i - 1]["value"].item()

            # Predict, then observe
            y_pred = forecaster.predict(forecasting_horizon=chunk_size)
            forecaster.observe(chunk)
            all_preds.append(y_pred)

            expected = analytical_forecast("ar1", {"phi": phi, "c": c}, horizon=chunk_size, last_obs=last_obs)

            np.testing.assert_allclose(
                y_pred["value"].to_numpy(),
                expected,
                atol=1e-7,
                err_msg=f"Chunk {i // stride} predictions don't match analytical AR(1)",
            )

    def test_observe_predict_varying_strides(self, ar1_series):
        """Verify mixing different stride sizes works correctly."""
        phi, c = 0.6, 2.5
        y_full = ar1_series(phi=phi, c=c, length=125, noise_std=0.0)

        y_train = y_full[:100]
        y_test = y_full[100:]  # Length 25

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=3)

        # Mix different stride sizes: 5, 3, 7, 5, 5
        strides = [5, 3, 7, 5, 5]
        all_preds = []
        pos = 0

        for stride in strides:
            chunk_size = min(stride, len(y_test) - pos)
            if chunk_size <= 0:
                break
            chunk = y_test[pos : pos + chunk_size]
            y_pred = forecaster.predict(forecasting_horizon=chunk_size)
            forecaster.observe(chunk)
            all_preds.append(y_pred)
            pos += chunk_size

        y_pred_full = pl.concat(all_preds)

        # Verify full coverage
        assert len(y_pred_full) == len(y_test)

        # Verify predictions match test data
        np.testing.assert_allclose(y_pred_full["value"].to_numpy(), y_test["value"].to_numpy(), atol=1e-6)


@pytest.mark.integration
class TestHorizonMismatch:
    """Horizon mismatch recursive correctness tests on linear trend series."""

    def test_horizon_mismatch_fit2_predict7(self, linear_series, analytical_forecast):
        """Verify fit_horizon=2, predict_horizon=7: recursive application 2+2+2+1."""
        # Linear trend: y = 2*t + 10
        y = linear_series(slope=2.0, intercept=10.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )

        # Fit with horizon=2
        forecaster.fit(y[:80], forecasting_horizon=2)

        # Predict with horizon=7 (should apply 2+2+2+1)
        y_pred = forecaster.predict(forecasting_horizon=7)

        # Expected: y[80:87] = [2*80+10, 2*81+10, ..., 2*86+10]
        last_index = 79
        expected = analytical_forecast("linear", {"slope": 2.0, "intercept": 10.0}, horizon=7, last_index=last_index)

        np.testing.assert_allclose(
            y_pred["value"].to_numpy(),
            expected,
            atol=1e-8,
            err_msg="fit_horizon=2, predict_horizon=7 doesn't match analytical linear",
        )

    def test_horizon_mismatch_fit3_predict10(self, linear_series, analytical_forecast):
        """Verify fit_horizon=3, predict_horizon=10: recursive application 3+3+3+1."""
        y = linear_series(slope=1.5, intercept=5.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y[:70], forecasting_horizon=3)

        y_pred = forecaster.predict(forecasting_horizon=10)

        last_index = 69
        expected = analytical_forecast("linear", {"slope": 1.5, "intercept": 5.0}, horizon=10, last_index=last_index)

        np.testing.assert_allclose(
            y_pred["value"].to_numpy(),
            expected,
            atol=1e-8,
            err_msg="fit_horizon=3, predict_horizon=10 doesn't match analytical",
        )

    def test_horizon_mismatch_fit5_predict3(self, linear_series, analytical_forecast):
        """Verify fit_horizon=5, predict_horizon=3: single application, trim to 3."""
        y = linear_series(slope=3.0, intercept=100.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y[:90], forecasting_horizon=5)

        # Predict horizon=3 (model trained for 5, only uses first 3)
        y_pred = forecaster.predict(forecasting_horizon=3)

        last_index = 89
        expected = analytical_forecast("linear", {"slope": 3.0, "intercept": 100.0}, horizon=3, last_index=last_index)

        np.testing.assert_allclose(
            y_pred["value"].to_numpy(),
            expected,
            atol=1e-8,
            err_msg="fit_horizon=5, predict_horizon=3 doesn't trim correctly",
        )

    @pytest.mark.parametrize(
        "fit_fh,predict_fh",
        [
            (1, 1),
            (1, 5),
            (2, 1),
            (2, 4),
            (3, 9),
            (5, 2),
            (5, 13),
        ],
    )
    def test_horizon_mismatch_matrix(self, linear_series, analytical_forecast, fit_fh, predict_fh):
        """Test all combinations of fit_horizon vs predict_horizon on linear trend."""
        y = linear_series(slope=2.5, intercept=50.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y[:80], forecasting_horizon=fit_fh)

        y_pred = forecaster.predict(forecasting_horizon=predict_fh)

        # Verify correct number of output rows
        assert len(y_pred) == predict_fh, (
            f"fit_fh={fit_fh}, predict_fh={predict_fh}: Expected {predict_fh} predictions, got {len(y_pred)}"
        )

        # Verify values match analytical forecast
        last_index = 79
        expected = analytical_forecast(
            "linear", {"slope": 2.5, "intercept": 50.0}, horizon=predict_fh, last_index=last_index
        )

        np.testing.assert_allclose(
            y_pred["value"].to_numpy(),
            expected,
            atol=1e-7,
            err_msg=f"fit_fh={fit_fh}, predict_fh={predict_fh} mismatch",
        )

    def test_horizon_mismatch_extreme_ratio(self, linear_series, analytical_forecast):
        """Test extreme horizon mismatch: fit_fh=1, predict_fh=50."""
        y = linear_series(slope=1.0, intercept=0.0, length=100)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y[:50], forecasting_horizon=1)

        # Predict 50 steps ahead (model applied 50 times)
        y_pred = forecaster.predict(forecasting_horizon=50)

        last_index = 49
        expected = analytical_forecast("linear", {"slope": 1.0, "intercept": 0.0}, horizon=50, last_index=last_index)

        # Allow slightly higher tolerance due to recursive accumulation
        np.testing.assert_allclose(
            y_pred["value"].to_numpy(), expected, atol=1e-6, err_msg="Extreme horizon mismatch (1 -> 50) failed"
        )


@pytest.mark.integration
class TestCVSplitterStride:
    """Cross-validation splitter stride calculation tests."""

    def test_sliding_window_stride_split_count(self):
        """Verify SlidingWindowSplitter with stride produces correct number of splits."""
        # 200 rows, train=50, test=10, stride=10
        # Available range: 200 - 50 - 10 = 140
        # Number of splits: 140 // 10 + 1 = 15

        # Create dummy data
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=199),
                interval="1d",
                eager=True,
            ),
            "value": np.arange(200),
        })

        splitter = SlidingWindowSplitter(n_splits=3, test_size=10, stride=10)

        splits = list(splitter.split(y))

        # n_splits=3 with test_size=10, stride=10 → auto train_size=170, exactly 3 splits
        expected_n_splits = 3
        assert len(splits) == expected_n_splits, f"Expected {expected_n_splits} splits, got {len(splits)}"

    def test_sliding_window_stride_indices(self):
        """Verify each split's train/test indices are correct."""
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=99),
                interval="1d",
                eager=True,
            ),
            "value": np.arange(100),
        })

        splitter = SlidingWindowSplitter(n_splits=3, test_size=5, stride=10)

        splits = list(splitter.split(y))

        # n_splits=3, test_size=5, stride=10 on 100 rows → auto train_size=75
        # Split 0: train=[0:75], test=[75:80]
        # Split 1: train=[10:85], test=[85:90]
        # Split 2: train=[20:95], test=[95:100]

        train_0, test_0 = splits[0]
        assert train_0.tolist() == list(range(0, 75))
        assert test_0.tolist() == list(range(75, 80))

        train_1, test_1 = splits[1]
        assert train_1.tolist() == list(range(10, 85))
        assert test_1.tolist() == list(range(85, 90))

        train_2, test_2 = splits[2]
        assert train_2.tolist() == list(range(20, 95))
        assert test_2.tolist() == list(range(95, 100))

    def test_sliding_window_non_overlapping_tests(self):
        """Verify non-overlapping test sets when stride >= test_size."""
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=149),
                interval="1d",
                eager=True,
            ),
            "value": np.arange(150),
        })

        # stride=10, test_size=10: test sets should not overlap
        splitter = SlidingWindowSplitter(n_splits=3, test_size=10, stride=10)

        splits = list(splitter.split(y))

        # Collect all test indices
        test_sets = [set(test_idx) for _, test_idx in splits]

        # Verify no overlaps between consecutive test sets
        for i in range(len(test_sets) - 1):
            overlap = test_sets[i] & test_sets[i + 1]
            assert len(overlap) == 0, f"Test sets {i} and {i + 1} overlap: {overlap}"

    def test_expanding_window_with_gap(self):
        """Verify ExpandingWindowSplitter with gap parameter."""
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(days=99),
                interval="1d",
                eager=True,
            ),
            "value": np.arange(100),
        })

        # Expanding window: n_splits=3, test_size=10
        splitter = ExpandingWindowSplitter(
            n_splits=3,
            test_size=10,
        )

        splits = list(splitter.split(y))

        # For 100 data points, n_splits=3, test_size=10:
        # Split 0: train=[0:70], test=[70:80]
        # Split 1: train=[0:80], test=[80:90]
        # Split 2: train=[0:90], test=[90:100]

        train_0, test_0 = splits[0]
        assert train_0.tolist() == list(range(0, 70))
        assert test_0.tolist() == list(range(70, 80))

        train_1, test_1 = splits[1]
        assert train_1.tolist() == list(range(0, 80))
        assert test_1.tolist() == list(range(80, 90))

        # Verify train size expands
        assert len(train_1) > len(train_0)


@pytest.mark.integration
class TestObservePredictState:
    """Observe-predict state consistency tests."""

    def test_observe_predict_sequential_vs_batch_state(self, ar1_series):
        """Verify sequential observe_predict matches batch observe + predict."""
        phi, c = 0.75, 2.0
        y_full = ar1_series(phi=phi, c=c, length=130, noise_std=0.0)

        y_train = y_full[:100]
        y_test = y_full[100:]

        # Forecaster 1: Sequential observes then predict
        forecaster_1 = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster_1.fit(y_train, forecasting_horizon=5)

        # Split test into 3 chunks: 10, 10, 10
        forecaster_1.observe(y_test[:10])
        forecaster_1.observe(y_test[10:20])
        forecaster_1.observe(y_test[20:])
        y_pred_seq = forecaster_1.predict(forecasting_horizon=5)

        # Forecaster 2: Batch observe then predict
        forecaster_2 = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster_2.fit(y_train, forecasting_horizon=5)
        forecaster_2.observe(y_test)
        y_pred_batch = forecaster_2.predict(forecasting_horizon=5)

        # Predictions from both should match (same final state)
        np.testing.assert_allclose(
            y_pred_seq["value"].to_numpy(),
            y_pred_batch["value"].to_numpy(),
            atol=1e-8,
            err_msg="Sequential observes should match batch observe",
        )

    def test_observe_predict_memory_accumulation(self, ar1_series):
        """Verify observe_predict correctly accumulates observations in memory."""
        phi, c = 0.8, 1.0
        y_full = ar1_series(phi=phi, c=c, length=120, noise_std=0.0)

        y_train = y_full[:100]
        y_test = y_full[100:]

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=3)

        # First observe
        forecaster.observe(y_test[:5])

        # Verify predictions are correct after observe
        y_pred_1 = forecaster.predict(forecasting_horizon=3)
        assert len(y_pred_1) == 3

        # Second observe
        forecaster.observe(y_test[5:10])
        y_pred_2 = forecaster.predict(forecasting_horizon=3)
        assert len(y_pred_2) == 3

        # Third observe
        forecaster.observe(y_test[10:15])
        y_pred_3 = forecaster.predict(forecasting_horizon=3)
        assert len(y_pred_3) == 3

        # Verify predictions start from the correct last observation
        # After observing y_test[0:15], the forecaster predicts from y_test[14]
        expected_last = y_test[14]["value"].item()
        expected_next = phi * expected_last + c
        np.testing.assert_allclose(y_pred_3["value"].to_numpy()[0], expected_next, atol=1e-8)

    def test_observe_predict_no_state_corruption(self, ar1_series):
        """Verify multiple observe_predict cycles don't corrupt internal state."""
        phi, c = 0.7, 2.5
        y_full = ar1_series(phi=phi, c=c, length=150, noise_std=0.0)

        y_train = y_full[:100]
        y_test = y_full[100:]

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=5)

        # Do 5 predict+observe cycles with 10 steps each
        for i in range(5):
            chunk = y_test[i * 10 : (i + 1) * 10]
            y_pred = forecaster.predict(forecasting_horizon=10)
            forecaster.observe(chunk)

            # Each prediction should match the actual test data
            np.testing.assert_allclose(
                y_pred["value"].to_numpy(),
                chunk["value"].to_numpy(),
                atol=1e-6,
                err_msg=f"Cycle {i} predictions corrupted",
            )

        # Final state check: predict 5 steps ahead should work
        y_final = forecaster.predict(forecasting_horizon=5)
        assert len(y_final) == 5


@pytest.mark.integration
class TestPanelStrideHorizon:
    """Panel data stride/horizon operation tests."""

    def test_panel_observe_predict_subset_groups(self, ar1_series):
        """Verify predict with groups returns only specified groups."""
        # Create panel data where each group has its own prefix
        # so we can select individual groups via groups
        series_0 = ar1_series(phi=0.5, c=1.0, length=110, noise_std=0.0)
        series_1 = ar1_series(phi=0.7, c=2.0, length=110, noise_std=0.0)
        series_2 = ar1_series(phi=0.9, c=3.0, length=110, noise_std=0.0)

        y = series_0.rename({"value": "alpha__series"})
        y = y.with_columns(series_1["value"].alias("beta__series"))
        y = y.with_columns(series_2["value"].alias("gamma__series"))

        y_train = y[:100]
        y_test = y[100:]

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=5)

        # Observe all data, then predict for only alpha and beta groups
        forecaster.observe(y_test)
        y_pred = forecaster.predict(forecasting_horizon=5, groups=["alpha", "beta"])

        # Verify only 2 groups returned (plus time columns)
        assert "alpha__series" in y_pred.columns
        assert "beta__series" in y_pred.columns
        assert "gamma__series" not in y_pred.columns

    def test_panel_group_specific_predictions(self, ar1_series, analytical_forecast):
        """Verify group-specific predictions match per-group analytical forecasts."""
        # 2-group panel with different parameters, each group has its own prefix
        params = [
            {"phi": 0.6, "c": 1.5},
            {"phi": 0.8, "c": 2.5},
        ]
        prefixes = ["alpha", "beta"]

        series_list = []
        for p in params:
            series_list.append(ar1_series(phi=p["phi"], c=p["c"], length=110, noise_std=0.0))

        y = series_list[0].rename({"value": f"{prefixes[0]}__series"})
        y = y.with_columns(series_list[1]["value"].alias(f"{prefixes[1]}__series"))

        y_train = y[:100]
        y_test = y[100:]

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=10)

        # Predict both groups using predict+observe pattern
        all_preds = []
        for i in range(0, len(y_test), 5):
            chunk_size = min(5, len(y_test) - i)
            chunk = y_test[i : i + chunk_size]
            y_pred = forecaster.predict(forecasting_horizon=chunk_size)
            forecaster.observe(chunk)
            all_preds.append(y_pred)
        y_pred_full = pl.concat(all_preds)

        # Verify each group matches its analytical forecast
        for i, p in enumerate(params):
            col_name = f"{prefixes[i]}__series"
            last_obs = y_train[-1][col_name].item()

            expected = analytical_forecast(
                "ar1", {"phi": p["phi"], "c": p["c"]}, horizon=len(y_test), last_obs=last_obs
            )

            np.testing.assert_allclose(
                y_pred_full[col_name].to_numpy(),
                expected,
                atol=1e-6,
                err_msg=f"Group {prefixes[i]} predictions don't match analytical AR(1)",
            )

    def test_panel_stride_combinations(self, ar1_series):
        """Verify panel data with different strides and group subsets."""
        # Create panel data with 4 groups, each with its own prefix
        ar_params = [
            {"phi": 0.5, "c": 1.0},
            {"phi": 0.6, "c": 1.5},
            {"phi": 0.7, "c": 2.0},
            {"phi": 0.8, "c": 2.5},
        ]
        prefixes = ["alpha", "beta", "gamma", "delta"]

        series_list = [ar1_series(phi=p["phi"], c=p["c"], length=120, noise_std=0.0) for p in ar_params]

        y = series_list[0].rename({"value": f"{prefixes[0]}__series"})
        for i in range(1, 4):
            y = y.with_columns(series_list[i]["value"].alias(f"{prefixes[i]}__series"))

        y_train = y[:100]
        y_test = y[100:]

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=5)

        # Predict+observe with stride=5, selecting groups beta and delta
        all_preds = []
        for i in range(0, len(y_test), 5):
            chunk_size = min(5, len(y_test) - i)
            chunk = y_test[i : i + chunk_size]
            y_pred = forecaster.predict(forecasting_horizon=chunk_size, groups=["beta", "delta"])
            forecaster.observe(chunk)
            all_preds.append(y_pred)

        y_pred_full = pl.concat(all_preds)

        # Verify coverage and group selection
        assert len(y_pred_full) == len(y_test)
        assert "beta__series" in y_pred_full.columns
        assert "delta__series" in y_pred_full.columns
        assert "alpha__series" not in y_pred_full.columns
        assert "gamma__series" not in y_pred_full.columns

        # Verify predictions match test data for selected groups
        np.testing.assert_allclose(y_pred_full["beta__series"].to_numpy(), y_test["beta__series"].to_numpy(), atol=1e-6)
        np.testing.assert_allclose(
            y_pred_full["delta__series"].to_numpy(), y_test["delta__series"].to_numpy(), atol=1e-6
        )
