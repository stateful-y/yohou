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

    def test_transform_propagates_nulls(self):
        """LagTransformer propagates input nulls into the corresponding lag column.

        A null at input row ``i`` of column ``a`` must surface (shifted by the
        lag and the dropped warmup row) in ``a_lag_1``, while a null-free column
        must stay null-free. The output row count equals
        ``len(X) - observation_horizon`` (the warmup drop), never a silent
        row-dropping of the null observations.
        """
        from datetime import datetime, timedelta

        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(seconds=9),
            interval="1s",
            eager=True,
        )
        # Two nulls in column "a", none in column "b".
        X = pl.DataFrame({
            "time": time,
            "a": [0.0, 1.0, None, 3.0, 4.0, None, 6.0, 7.0, 8.0, 9.0],
            "b": [10.0] * 10,
        })
        transformer = LagTransformer(lag=[1])
        transformer.fit(X)
        X_out = transformer.transform(X)

        # Warmup drop only: no silent dropping of null rows.
        assert len(X_out) == len(X) - transformer.observation_horizon
        # Both input nulls propagate to the lagged column.
        assert X_out["a_lag_1"].null_count() == 2
        # A null-free input column stays null-free.
        assert X_out["b_lag_1"].null_count() == 0


class TestNaNInfScorers:
    """Scorers should reject or handle NaN/Inf in predictions."""

    def test_scorer_identical_predictions_zero_error(self, y_X_factory):
        """Identical predictions should yield zero error for MAE."""
        y, _ = y_X_factory(length=20, n_targets=1, n_features=0, seed=42)
        y_truth = y[:10]

        mae = MeanAbsoluteError()
        mae.fit(y_truth)
        result = mae.score(y_truth, y_truth)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_mse_nonidentical_predictions_positive(self, y_X_factory):
        """Non-identical predictions yield a strictly positive MSE.

        Finiteness on clean data is already covered for every scorer by
        ``check_scorer_multi_vintage`` in the systematic sweep; the strictly
        positive value for a known nonzero error is the behaviour exercised here.
        """
        y, _ = y_X_factory(length=20, n_targets=1, n_features=0, seed=42)
        y_truth = y[:10]
        y_pred = y_truth.with_columns((pl.col(c) + 1.0) for c in y_truth.columns if c != "time")

        mse = MeanSquaredError()
        mse.fit(y_truth)
        result = mse.score(y_truth, y_pred)
        assert result > 0, "Non-identical predictions should have positive MSE"
