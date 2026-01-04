"""Tests for PolynomialTrendForecaster."""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone

from yohou.decomposition import PolynomialTrendForecaster

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from estimator_checks import _yield_yohou_forecaster_checks


@pytest.mark.parametrize(
    "forecaster,tags,expected_failures",
    [
        (
            PolynomialTrendForecaster(degree=1),
            {"forecaster_type": "point", "uses_reduction": False},
            [],
        ),
        (
            PolynomialTrendForecaster(degree=2),
            {"forecaster_type": "point", "uses_reduction": False},
            [],
        ),
        (
            PolynomialTrendForecaster(degree=3),
            {"forecaster_type": "point", "uses_reduction": False},
            [],
        ),
    ],
)
def test_polynomial_trend_checks(forecaster, tags, expected_failures, y_X_factory):
    """Run systematic checks on PolynomialTrendForecaster."""
    # Generate data with trend
    y, X = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

    # Add linear trend to data
    y = y.with_columns(
        [(pl.col(col) + pl.Series(range(len(y)))).alias(col) for col in y.columns if col != "time"]
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


def test_polynomial_linear_analytical():
    """Test linear trend forecaster on known linear process."""
    # Create perfect linear trend: y = 2*t + 5
    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=49), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": [2 * i + 5 for i in range(50)]})

    # Fit on first 40, predict next 10
    forecaster = PolynomialTrendForecaster(degree=1, alpha=0.0, l1_ratio=0.0)
    forecaster.fit(y[:40], forecasting_horizon=1)

    # Predict 10 steps
    y_pred = forecaster.predict(forecasting_horizon=10)

    # Check predictions match exact linear continuation
    expected_values = [2 * i + 5 for i in range(40, 50)]

    # Allow small numerical error (floating point precision)
    pred_values = y_pred["value"].to_numpy().flatten()
    assert np.allclose(
        pred_values, expected_values, atol=1e-10
    ), "Linear trend predictions should match exact linear process"


def test_polynomial_quadratic_analytical():
    """Test polynomial trend on known quadratic process."""
    # Create perfect quadratic: y = 0.5*t^2 + 2*t + 1
    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=49), interval="1d", eager=True
    )
    y = pl.DataFrame(
        {"time": time, "value": [0.5 * i**2 + 2 * i + 1 for i in range(50)]}
    )

    # Fit polynomial degree 2
    forecaster = PolynomialTrendForecaster(degree=2, alpha=0.0, l1_ratio=0.0)
    forecaster.fit(y[:40], forecasting_horizon=1)

    # Predict 10 steps
    y_pred = forecaster.predict(forecasting_horizon=10)

    # Check predictions
    expected_values = [0.5 * i**2 + 2 * i + 1 for i in range(40, 50)]

    # Polynomial fitting may have small numerical errors
    pred_values = y_pred["value"].to_numpy().flatten()
    assert np.allclose(
        pred_values, expected_values, atol=1e-8
    ), "Polynomial trend should match quadratic process closely"


def test_polynomial_different_horizons():
    """Test that different forecasting horizons work correctly."""
    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=49), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": [2 * i + 5 for i in range(50)]})

    forecaster = PolynomialTrendForecaster(degree=1)
    forecaster.fit(y[:40], forecasting_horizon=5)

    # Predict different horizon than fit
    y_pred_10 = forecaster.predict(forecasting_horizon=10)
    assert len(y_pred_10) == 10

    y_pred_3 = forecaster.predict(forecasting_horizon=3)
    assert len(y_pred_3) == 3


def test_polynomial_panel_data(panel_time_series_factory):
    """Test PolynomialTrendForecaster with panel data."""
    y_panel = panel_time_series_factory(length=100, n_series=3, seed=42)

    # Add linear trends with different slopes per series
    for i in range(3):
        col_name = f"panel__series_{i}"
        y_panel = y_panel.with_columns(
            (pl.col(col_name) + (i + 1) * pl.Series(range(100))).alias(col_name)
        )

    forecaster = PolynomialTrendForecaster(degree=1)
    forecaster.fit(y_panel[:80], forecasting_horizon=5)

    # Predict all groups
    y_pred = forecaster.predict(forecasting_horizon=5)

    # Should have predictions for all series
    assert "panel__series_0" in y_pred.columns
    assert "panel__series_1" in y_pred.columns
    assert "panel__series_2" in y_pred.columns
    assert len(y_pred) == 5


def test_polynomial_update_predict():
    """Test update_predict method."""
    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=49), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": [2 * i + 5 for i in range(50)]})

    forecaster = PolynomialTrendForecaster(degree=1)
    fit_forecasting_horizon = 5
    forecaster.fit(y[:30], forecasting_horizon=fit_forecasting_horizon)

    # Update with new data and predict
    n_new = 10
    predict_forecasting_horizon = 5
    y_new = y[30:30 + n_new]
    y_pred = forecaster.update_predict(y_new, forecasting_horizon=predict_forecasting_horizon)

    assert len(y_pred) == predict_forecasting_horizon * (1 + n_new // fit_forecasting_horizon)
    assert "value" in y_pred.columns
