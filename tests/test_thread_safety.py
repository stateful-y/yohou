"""Thread safety tests for yohou estimators.

Verifies that concurrent observe/predict calls on *independent* instances
do not interfere with each other via shared global state (e.g. sklearn
metadata routing configuration).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import polars as pl


def _make_y(length: int = 80, seed: int = 42) -> pl.DataFrame:
    """Create a target time series."""
    rng = np.random.default_rng(seed)
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )
    return pl.DataFrame({"time": time, "y_0": rng.standard_normal(length).tolist()})


class TestThreadSafetyFitPredict:
    """Concurrent fit/predict on independent forecaster instances."""

    def test_concurrent_fit_predict_naive(self):
        """Multiple threads fitting/predicting NaiveForecasters independently."""
        from yohou.point import SeasonalNaive

        results: dict[int, pl.DataFrame] = {}
        errors: list[Exception] = []

        def _worker(thread_id: int) -> None:
            try:
                y = _make_y(80, seed=thread_id)
                f = SeasonalNaive(seasonality=1)
                f.fit(y, forecasting_horizon=5)
                pred = f.predict()
                results[thread_id] = pred
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 8
        # Each result should have the correct shape
        for pred in results.values():
            assert len(pred) == 5
            assert "time" in pred.columns

    def test_concurrent_fit_predict_reduction(self):
        """Multiple threads fitting/predicting ReductionForecasters independently."""
        from sklearn.linear_model import Ridge

        from yohou.point import PointReductionForecaster

        results: dict[int, pl.DataFrame] = {}
        errors: list[Exception] = []

        def _worker(thread_id: int) -> None:
            try:
                y = _make_y(80, seed=thread_id)
                f = PointReductionForecaster(estimator=Ridge())
                f.fit(y, forecasting_horizon=5)
                pred = f.predict()
                results[thread_id] = pred
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_worker, i) for i in range(8)]
            for fut in as_completed(futures):
                fut.result()  # Re-raises exceptions

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 8


class TestThreadSafetyObservePredict:
    """Concurrent observe_predict calls on independent instances."""

    def test_concurrent_observe_predict(self):
        """Threaded observe_predict does not corrupt state across instances."""
        from yohou.point import SeasonalNaive

        y_train = _make_y(80, seed=0)
        results: dict[int, pl.DataFrame] = {}
        errors: list[Exception] = []

        def _worker(thread_id: int) -> None:
            try:
                f = SeasonalNaive(seasonality=1)
                f.fit(y_train, forecasting_horizon=3)
                # Each thread observes different new data
                y_new = _make_y(5, seed=100 + thread_id)
                pred = f.observe_predict(y_new)
                results[thread_id] = pred
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 6
        for pred in results.values():
            assert len(pred) > 0
            assert "time" in pred.columns
            assert "vintage_time" in pred.columns


class TestThreadSafetyTransformers:
    """Concurrent transform calls on independent transformer instances."""

    def test_concurrent_fit_transform_scaler(self):
        """Multiple threads fitting StandardScaler independently."""
        from yohou.preprocessing import StandardScaler

        results: dict[int, pl.DataFrame] = {}
        errors: list[Exception] = []

        def _worker(thread_id: int) -> None:
            try:
                rng = np.random.default_rng(thread_id)
                time = pl.datetime_range(
                    start=datetime(2021, 1, 1),
                    end=datetime(2021, 1, 1) + timedelta(seconds=49),
                    interval="1s",
                    eager=True,
                )
                df = pl.DataFrame({"time": time, "val": rng.standard_normal(50).tolist()})
                scaler = StandardScaler()
                scaler.fit(df)
                out = scaler.transform(df)
                results[thread_id] = out
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_worker, i) for i in range(8)]
            for fut in as_completed(futures):
                fut.result()

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 8
        for out in results.values():
            assert "time" in out.columns
            assert len(out) == 50


class TestThreadSafetyMetadataRouting:
    """Verify sklearn metadata routing config is not corrupted by threads."""

    def test_metadata_routing_stable_in_main_thread(self):
        """Metadata routing stays enabled in main thread after concurrent work."""
        from sklearn import get_config

        from yohou.point import SeasonalNaive

        # Verify it's enabled before
        assert get_config()["enable_metadata_routing"] is True

        errors: list[Exception] = []

        def _worker(thread_id: int) -> None:
            try:
                # Note: sklearn get_config() is thread-local, so child threads
                # may NOT inherit the metadata routing flag. We just verify
                # that fit/predict still works regardless.
                y = _make_y(50, seed=thread_id)
                f = SeasonalNaive(seasonality=1)
                f.fit(y, forecasting_horizon=3)
                f.predict()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"

        # Main thread config must remain unaffected
        assert get_config()["enable_metadata_routing"] is True


class TestThreadSafetyScorers:
    """Concurrent scoring on independent instances."""

    def test_concurrent_scoring(self):
        """Multiple threads computing scores simultaneously."""
        from yohou.metrics import MeanAbsoluteError

        results: dict[int, float] = {}
        errors: list[Exception] = []

        def _worker(thread_id: int) -> None:
            try:
                rng = np.random.default_rng(thread_id)
                time = pl.datetime_range(
                    start=datetime(2021, 1, 1),
                    end=datetime(2021, 1, 1) + timedelta(seconds=29),
                    interval="1s",
                    eager=True,
                )
                y_true = pl.DataFrame({"time": time, "val": rng.standard_normal(30).tolist()})
                y_pred = pl.DataFrame({"time": time, "val": rng.standard_normal(30).tolist()})

                scorer = MeanAbsoluteError()
                scorer.fit(y_true)
                score = scorer.score(y_true, y_pred)
                results[thread_id] = score
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_worker, i) for i in range(8)]
            for fut in as_completed(futures):
                fut.result()

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 8
        for score in results.values():
            assert score >= 0.0
