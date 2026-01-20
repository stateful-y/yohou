"""Tests for PolynomialTrendForecaster."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline

from yohou.decomposition import PolynomialTrendForecaster

# Add parent directory to path for imports
from yohou.testing import _yield_yohou_forecaster_checks


@pytest.mark.parametrize(
    "forecaster,expected_failures",
    [
        (
            PolynomialTrendForecaster(degree=1),
            [],
        ),
        (
            PolynomialTrendForecaster(degree=2),
            [],
        ),
        (
            PolynomialTrendForecaster(degree=3),
            [],
        ),
    ],
)
def test_polynomial_trend_checks(forecaster, expected_failures, y_X_factory):
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
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=49),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": [2 * i + 5 for i in range(50)]})

    # Fit on first 40, predict next 10
    forecaster = PolynomialTrendForecaster(degree=1, estimator=ElasticNet(alpha=0.0, l1_ratio=0.0))
    forecaster.fit(y[:40], forecasting_horizon=1)

    # Predict 10 steps
    y_pred = forecaster.predict(forecasting_horizon=10)

    # Check predictions match exact linear continuation
    expected_values = [2 * i + 5 for i in range(40, 50)]

    # Allow small numerical error (floating point precision)
    pred_values = y_pred["value"].to_numpy().flatten()
    assert np.allclose(pred_values, expected_values, atol=1e-10), (
        "Linear trend predictions should match exact linear process"
    )


def test_polynomial_quadratic_analytical():
    """Test polynomial trend on known quadratic process."""
    # Create perfect quadratic: y = 0.5*t^2 + 2*t + 1
    from datetime import timedelta

    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=49),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": [0.5 * i**2 + 2 * i + 1 for i in range(50)]})

    # Fit polynomial degree 2
    forecaster = PolynomialTrendForecaster(degree=2, estimator=ElasticNet(alpha=0.0, l1_ratio=0.0))
    forecaster.fit(y[:40], forecasting_horizon=1)

    # Predict 10 steps
    y_pred = forecaster.predict(forecasting_horizon=10)

    # Check predictions
    expected_values = [0.5 * i**2 + 2 * i + 1 for i in range(40, 50)]

    # Polynomial fitting may have small numerical errors
    pred_values = y_pred["value"].to_numpy().flatten()
    assert np.allclose(pred_values, expected_values, atol=1e-1), (
        "Polynomial trend should match quadratic process closely"
    )


def test_polynomial_different_horizons():
    """Test that different forecasting horizons work correctly."""
    from datetime import timedelta

    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=49),
        interval="1d",
        eager=True,
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
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=49),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": [2 * i + 5 for i in range(50)]})

    forecaster = PolynomialTrendForecaster(degree=1)
    fit_forecasting_horizon = 5
    forecaster.fit(y[:30], forecasting_horizon=fit_forecasting_horizon)

    # Update with new data and predict
    n_new = 10
    predict_forecasting_horizon = 5
    y_new = y[30 : 30 + n_new]
    y_pred = forecaster.update_predict(y_new, forecasting_horizon=predict_forecasting_horizon)

    assert len(y_pred) == predict_forecasting_horizon * (1 + n_new // fit_forecasting_horizon)
    assert "value" in y_pred.columns


@pytest.mark.parametrize("model_panel", [False, True])
def test_polynomial_model_panel_behaviors(panel_time_series_factory, model_panel):
    """Test PolynomialTrendForecaster with both pooled and per-group strategies."""
    y_panel = panel_time_series_factory(length=100, n_series=3, seed=42)

    # Add distinct linear trends with different slopes per series
    for i in range(3):
        col_name = f"panel__series_{i}"
        # Series 0: slope=1, Series 1: slope=3, Series 2: slope=5
        slope = 1 + i * 2
        y_panel = y_panel.with_columns(
            (pl.col(col_name) + slope * pl.Series(range(100))).alias(col_name)
        )

    # Fit forecaster with specified model_panel strategy
    forecaster = PolynomialTrendForecaster(degree=1, model_panel=model_panel)
    forecaster.fit(y_panel[:80], forecasting_horizon=5)

    # Check estimator_ type based on model_panel setting
    if model_panel:
        # Per-group strategy: estimator_ should be dict of estimators (one per panel group)
        assert isinstance(forecaster.estimator_, dict), (
            "model_panel=True should store dict of estimators"
        )
        assert len(forecaster.estimator_) == 1, "Should have 1 estimator for the 'panel' group"
        assert "panel" in forecaster.estimator_, "Should have estimator for 'panel' group"
        # Each estimator in dict should be a Pipeline
        for group_estimator in forecaster.estimator_.values():
            assert isinstance(group_estimator, Pipeline), (
                "Each group estimator should be a Pipeline"
            )
    else:
        # Pooled strategy: estimator_ should be single Pipeline object
        assert isinstance(forecaster.estimator_, Pipeline), (
            "model_panel=False should store single Pipeline"
        )
        assert not isinstance(forecaster.estimator_, dict), "Pooled should not be a dict"

    # Predict all groups
    # TODO: Fix decomposition forecasters to handle dict _y_observed in predict()
    # Currently fails with: AttributeError: 'dict' object has no attribute 'columns'
    # y_pred = forecaster.predict(forecasting_horizon=5)

    # # Basic structure checks
    # assert len(y_pred) == 5
    # assert "panel__series_0" in y_pred.columns
    # assert "panel__series_1" in y_pred.columns
    # assert "panel__series_2" in y_pred.columns


def test_polynomial_model_panel_prediction_differences(panel_time_series_factory):
    """Test that pooled vs per-group strategies produce different predictions on heterogeneous data."""
    y_panel = panel_time_series_factory(length=100, n_series=1, n_groups=3, seed=42)

    # Add VERY distinct linear trends (opposite slopes)
    for i in range(3):
        col_name = f"group{i}__series_0"
        # Series 0: 1.0, Series 1: 3.0, Series 2: 9.0
        # Avg = 4.33. No match.
        slopes = [1.0, 3.0, 9.0]
        slope = slopes[i]
        y_panel = y_panel.with_columns(
            (pl.col(col_name) + slope * pl.Series(range(100))).alias(col_name)
        )

    # Fit both strategies
    forecaster_pooled = PolynomialTrendForecaster(degree=1, model_panel=False)
    forecaster_per_group = PolynomialTrendForecaster(degree=1, model_panel=True)

    forecaster_pooled.fit(y_panel[:80], forecasting_horizon=10)
    forecaster_per_group.fit(y_panel[:80], forecasting_horizon=10)

    # Generate predictions
    y_pred_pooled = forecaster_pooled.predict(forecasting_horizon=10)
    y_pred_per_group = forecaster_per_group.predict(forecasting_horizon=10)

    # Predictions should differ significantly due to heterogeneous trends
    for col in ["group0__series_0", "group1__series_0", "group2__series_0"]:
        pooled_vals = y_pred_pooled[col].to_numpy()
        per_group_vals = y_pred_per_group[col].to_numpy()

        # Calculate absolute difference
        abs_diff = np.abs(pooled_vals - per_group_vals)
        mean_abs_diff = np.mean(abs_diff)

        # With strongly divergent trends, predictions should differ substantially
        # Pooled model averages across all series, per-group captures individual trends
        assert mean_abs_diff > 1.0, (
            f"Predictions for {col} should differ between strategies (got {mean_abs_diff:.4f})"
        )
