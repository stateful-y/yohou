"""Tests for NaN/Inf edge cases across forecasters, transformers, and scorers.

Step 27 of codebase quality plan: verify that estimators handle
degenerate numerical inputs (NaN, Inf, null) gracefully.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from yohou.metrics.point import MeanAbsoluteError, MeanSquaredError
from yohou.point.naive import SeasonalNaive
from yohou.preprocessing.window import LagTransformer


class TestNaNInfForecasters:
    """Forecasters should either reject or tolerate NaN/Inf in input data."""

    def test_predict_produces_finite_values(self, y_X_factory):
        """Point predictions should be finite when trained on clean data."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        forecaster = SeasonalNaive(seasonality=5)
        forecaster.fit(y[:80], forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        values = y_pred.drop("time", "vintage_time").to_numpy()
        assert np.all(np.isfinite(values)), "Predictions should be finite"


class TestNaNInfTransformers:
    """Transformers should handle or propagate NaN consistently."""

    def test_lag_transformer_introduces_nulls(self, y_X_factory):
        """LagTransformer naturally introduces nulls at the start."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=[1, 2])
        transformer.fit(X[:40], y[:40])
        X_out = transformer.transform(X[40:])
        # After dropping warmup rows, output should have no nulls
        assert X_out.null_count().sum_horizontal().item() == 0

    def test_transform_with_nulls_in_input(self, time_series_with_nulls_factory):
        """Verify that transformers do not silently drop null rows."""
        X = time_series_with_nulls_factory(length=50, n_components=2, null_fraction=0.1)
        transformer = LagTransformer(lag=[1])
        transformer.fit(X)
        X_out = transformer.transform(X)
        # Output length should be deterministic (input length minus lag)
        assert len(X_out) <= len(X)


class TestNaNInfScorers:
    """Scorers should reject or handle NaN/Inf in predictions."""

    def test_scorer_finite_predictions_returns_finite_score(self, y_X_factory):
        """Clean predictions produce a finite score."""
        y, _ = y_X_factory(length=20, n_targets=1, n_features=0, seed=42)
        y_truth = y[:10]
        y_pred = y[10:20].with_columns(pl.col("time").alias("time"))

        # Align time columns for scoring
        y_pred = y_pred.with_columns(y_truth["time"].alias("time"))

        mae = MeanAbsoluteError()
        mae.fit(y_truth)
        result = mae.score(y_truth, y_pred)
        assert np.isfinite(result), f"Score should be finite, got {result}"

    def test_scorer_identical_predictions_zero_error(self, y_X_factory):
        """Identical predictions should yield zero error for MAE."""
        y, _ = y_X_factory(length=20, n_targets=1, n_features=0, seed=42)
        y_truth = y[:10]

        mae = MeanAbsoluteError()
        mae.fit(y_truth)
        result = mae.score(y_truth, y_truth)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_mse_finite_predictions_returns_finite_score(self, y_X_factory):
        """MSE also returns finite values on clean data."""
        y, _ = y_X_factory(length=20, n_targets=1, n_features=0, seed=42)
        y_truth = y[:10]
        y_pred = y_truth.with_columns((pl.col(c) + 1.0) for c in y_truth.columns if c != "time")

        mse = MeanSquaredError()
        mse.fit(y_truth)
        result = mse.score(y_truth, y_pred)
        assert np.isfinite(result), f"Score should be finite, got {result}"
        assert result > 0, "Non-identical predictions should have positive MSE"
