"""Tests for ExponentialTrendForecaster."""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone

from yohou.decomposition import ExponentialTrendForecaster

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from estimator_checks import _yield_yohou_forecaster_checks


@pytest.mark.parametrize(
    "forecaster,tags,expected_failures",
    [
        (
            ExponentialTrendForecaster(),
            {"forecaster_type": "point", "uses_reduction": False},
            [],
        ),
    ],
)
def test_exponential_trend_checks(forecaster, tags, expected_failures, y_X_factory):
    """Run systematic checks on ExponentialTrendForecaster."""
    # Generate data with exponential trend (positive values)
    y, X = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

    # Make values positive and add exponential trend
    y = y.with_columns(
        [
            (pl.col(col).abs() + 10 * pl.Series([np.exp(0.01 * i) for i in range(100)])).alias(col)
            for col in y.columns
            if col != "time"
        ]
    )

    y_train, y_test = y[:80], y[80:]
    X_train, X_test = (X[:80], X[80:]) if X is not None else (None, None)

    # Fit forecaster
    forecaster_fitted = clone(forecaster)
    forecaster_fitted.fit(y_train, X_train, forecasting_horizon=3)

    # Run all generated checks
    expected_failures_set = set(expected_failures)
    for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(
        forecaster_fitted,
        y_train,
        X_train,
        y_test,
        X_test,
        tags=tags,
    ):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(forecaster_fitted, **check_kwargs)


def test_exponential_analytical():
    """Test exponential trend on known exponential process."""
    # Create perfect exponential: y = 10 * exp(0.05*t)
    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=49), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": [10 * np.exp(0.05 * i) for i in range(50)]})

    # Fit on first 40
    forecaster = ExponentialTrendForecaster(alpha=0.0, l1_ratio=0.0)
    forecaster.fit(y[:40], forecasting_horizon=1)

    # Predict 10 steps
    y_pred = forecaster.predict(forecasting_horizon=10)

    # Check predictions are close to expected
    expected_values = [10 * np.exp(0.05 * i) for i in range(40, 50)]

    # Exponential fitting may have small numerical errors
    pred_values = y_pred["value"].to_numpy().flatten()
    assert np.allclose(
        pred_values, expected_values, rtol=0.01
    ), "Exponential trend should match exponential process closely"


def test_exponential_positive_values_required():
    """Test that exponential trend raises error for non-positive values."""
    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=49), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": [-1 * i for i in range(50)]})  # Negative values

    forecaster = ExponentialTrendForecaster()

    with pytest.raises(ValueError, match="positive values"):
        forecaster.fit(y, forecasting_horizon=1)


def test_exponential_zero_values_error():
    """Test that exponential trend raises error for zero values."""
    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=49), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": [0] * 50})  # Zero values

    forecaster = ExponentialTrendForecaster()

    with pytest.raises(ValueError, match="positive values"):
        forecaster.fit(y, forecasting_horizon=1)


def test_exponential_mixed_signs_error():
    """Test that exponential trend raises error for mixed positive/negative."""
    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=49), interval="1d", eager=True
    )
    # Some positive, some negative
    y = pl.DataFrame({"time": time, "value": [10 if i % 2 == 0 else -5 for i in range(50)]})

    forecaster = ExponentialTrendForecaster()

    with pytest.raises(ValueError, match="positive values"):
        forecaster.fit(y, forecasting_horizon=1)


def test_exponential_different_horizons():
    """Test that different forecasting horizons work correctly."""
    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=49), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": [10 * np.exp(0.05 * i) for i in range(50)]})

    forecaster = ExponentialTrendForecaster()
    forecaster.fit(y[:40], forecasting_horizon=5)

    # Predict different horizon than fit
    y_pred_10 = forecaster.predict(forecasting_horizon=10)
    assert len(y_pred_10) == 10

    y_pred_3 = forecaster.predict(forecasting_horizon=3)
    assert len(y_pred_3) == 3


def test_exponential_trend_panel_data(panel_time_series_factory):
    """Test ExponentialTrendForecaster with panel data."""
    y_panel = panel_time_series_factory(length=100, n_series=3, seed=42)

    # Make all values positive and add exponential trends
    # We need to update each column directly
    for i in range(3):
        col_name = f"panel__series_{i}"
        y_panel = y_panel.with_columns(
            (pl.col(col_name).abs() + 10 * pl.Series([np.exp(0.02 * (i + 1) * j) for j in range(100)])).alias(col_name)
        )

    forecaster = ExponentialTrendForecaster()
    forecaster.fit(y_panel[:80], forecasting_horizon=5)

    # Predict all groups
    y_pred = forecaster.predict(forecasting_horizon=5)

    # Should have predictions for all panel series (flat columns)
    assert "panel__series_0" in y_pred.columns
    assert "panel__series_1" in y_pred.columns
    assert "panel__series_2" in y_pred.columns
    assert len(y_pred) == 5


def test_exponential_update_predict():
    """Test update_predict method."""
    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=49), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": [10 * np.exp(0.05 * i) for i in range(50)]})

    forecaster = ExponentialTrendForecaster()
    fit_forecasting_horizon = 5
    forecaster.fit(y[:30], forecasting_horizon=fit_forecasting_horizon)

    # Update with new data and predict
    n_new = 10
    predict_forecasting_horizon = 5
    y_new = y[30:30 + n_new]
    y_pred = forecaster.update_predict(y_new, forecasting_horizon=predict_forecasting_horizon)

    assert len(y_pred) == predict_forecasting_horizon * (1 + n_new // fit_forecasting_horizon)
    assert "value" in y_pred.columns
