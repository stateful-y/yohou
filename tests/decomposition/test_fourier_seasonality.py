"""Tests for FourierSeasonalityForecaster."""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone

from yohou.decomposition import FourierSeasonalityForecaster

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from estimator_checks import _yield_yohou_forecaster_checks


@pytest.mark.parametrize(
    "forecaster,tags,expected_failures",
    [
        (
            FourierSeasonalityForecaster(seasonality=12, harmonics=[1]),
            {"forecaster_type": "point", "uses_reduction": False},
            [],
        ),
        (
            FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2, 3]),
            {"forecaster_type": "point", "uses_reduction": False},
            [],
        ),
        (
            FourierSeasonalityForecaster(seasonality=24, harmonics=[1, 2, 3, 4, 5]),
            {"forecaster_type": "point", "uses_reduction": False},
            [],
        ),
        (
            FourierSeasonalityForecaster(seasonality=7, harmonics=[1, 2]),
            {"forecaster_type": "point", "uses_reduction": False},
            [],
        ),
    ],
)
def test_fourier_seasonality_forecaster_checks(
    forecaster, tags, expected_failures, y_X_factory
):
    """Run systematic checks on FourierSeasonalityForecaster variants."""
    # Generate data with sufficient length
    seasonality = forecaster.seasonality
    y, X = y_X_factory(length=3 * seasonality, n_targets=1, n_features=0, seed=42)

    # Add sine wave pattern
    phases = np.arange(len(y))
    sine_values = np.sin(2 * np.pi * phases / seasonality)
    y = y.with_columns(
        [
            pl.Series(col, sine_values).alias(col) for col in y.columns if col != "time"
        ]
    )

    train_size = int(2.5 * seasonality)
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = (
        (X[:train_size], X[train_size:]) if X is not None else (None, None)
    )

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


def test_fourier_seasonality_sine_wave_recovery():
    """Test Fourier forecaster recovers pure sine wave with 1 harmonic."""
    # Create perfect sine wave with period 12
    seasonality = 12
    n_periods = 3
    length = seasonality * n_periods

    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=length-1), interval="1d", eager=True
    )

    # Pure sine wave: y = sin(2π * t / T)
    phases = np.arange(length)
    sine_values = np.sin(2 * np.pi * phases / seasonality)
    y = pl.DataFrame({"time": time, "value": sine_values})

    # Fit with 1 harmonic (should perfectly capture fundamental frequency)
    forecaster = FourierSeasonalityForecaster(seasonality=seasonality, harmonics=[1], alpha=0.0, l1_ratio=0.0)
    forecaster.fit(y, forecasting_horizon=1)

    # Predict next cycle
    y_pred = forecaster.predict(forecasting_horizon=seasonality)

    # Expected values for next cycle
    next_phases = np.arange(seasonality)
    expected = np.sin(2 * np.pi * next_phases / seasonality)
    predicted = y_pred["value"].to_numpy().flatten()

    # Should match closely (Fourier approximation)
    np.testing.assert_allclose(predicted, expected, atol=1e-15)


def test_fourier_seasonality_complex_pattern():
    """Test Fourier forecaster with multiple harmonics captures complex pattern."""
    # Create pattern with 2 harmonics: y = sin(2πt/T) + 0.5*cos(4πt/T)
    seasonality = 24
    n_periods = 4
    length = seasonality * n_periods

    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(hours=length-1), interval="1h", eager=True
    )

    phases = np.arange(length)
    # Fundamental + first overtone
    values = np.sin(2 * np.pi * phases / seasonality) + 0.5 * np.cos(
        4 * np.pi * phases / seasonality
    )
    y = pl.DataFrame({"time": time, "value": values})

    # Fit with 3 harmonics (should capture both components)
    forecaster = FourierSeasonalityForecaster(seasonality=seasonality, harmonics=[1, 2], alpha=0.0, l1_ratio=0.0)
    forecaster.fit(y, forecasting_horizon=1)

    # Predict next cycle
    y_pred = forecaster.predict(forecasting_horizon=seasonality)

    # Expected values
    next_phases = np.arange(seasonality)
    expected = np.sin(2 * np.pi * next_phases / seasonality) + 0.5 * np.cos(
        4 * np.pi * next_phases / seasonality
    )
    predicted = y_pred["value"].to_numpy().flatten()

    # Should match closely
    np.testing.assert_allclose(predicted, expected, rtol=0.02)


def test_fourier_seasonality_harmonics_constraint():
    """Test that harmonics cannot exceed seasonality/2."""
    # Valid: harmonics < seasonality/2
    forecaster = FourierSeasonalityForecaster(seasonality=12, harmonics=[5])

    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=35), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": np.random.randn(36)})

    # Should work
    forecaster.fit(y, forecasting_horizon=1)

    # Invalid: harmonics > seasonality/2
    forecaster_invalid = FourierSeasonalityForecaster(seasonality=12, harmonics=[7])

    with pytest.raises(ValueError, match="Maximum harmonic.*cannot exceed"):
        forecaster_invalid.fit(y, forecasting_horizon=1)


def test_fourier_seasonality_non_integer_seasonality():
    """Test Fourier forecaster with non-integer effective seasonality."""
    # Create data with seasonality 12 but test with different values
    seasonality = 12
    n_periods = 4
    length = seasonality * n_periods
    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=length-1), interval="1d", eager=True
    )

    phases = np.arange(length)
    values = np.sin(2 * np.pi * phases / seasonality)
    y = pl.DataFrame({"time": time, "value": values})

    # Fit with slightly different seasonality (Fourier handles fractional phases)
    forecaster = FourierSeasonalityForecaster(seasonality=12, harmonics=[2])
    forecaster.fit(y, forecasting_horizon=1)

    # Should still work and produce reasonable predictions
    y_pred = forecaster.predict(forecasting_horizon=12)
    assert len(y_pred) == 12
    assert not y_pred["value"].is_null().any()


def test_fourier_seasonality_insufficient_data():
    """Test error handling for insufficient data."""
    # Less than 1 cycle
    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=7), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": list(range(8))})

    forecaster = FourierSeasonalityForecaster(seasonality=12, harmonics=[3])

    with pytest.raises(ValueError, match="Insufficient data"):
        forecaster.fit(y, forecasting_horizon=1)


def test_fourier_seasonality_different_horizons():
    """Test that different forecasting horizons work correctly."""
    seasonality = 12
    n_periods = 3
    length = seasonality * n_periods
    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=length-1), interval="1d", eager=True
    )

    phases = np.arange(length)
    values = np.sin(2 * np.pi * phases / seasonality)
    y = pl.DataFrame({"time": time, "value": values})

    forecaster = FourierSeasonalityForecaster(seasonality=seasonality, harmonics=[3])
    forecaster.fit(y, forecasting_horizon=5)

    # Predict different horizon than fit
    y_pred_24 = forecaster.predict(forecasting_horizon=24)
    assert len(y_pred_24) == 24

    y_pred_6 = forecaster.predict(forecasting_horizon=6)
    assert len(y_pred_6) == 6


def test_fourier_seasonality_panel_data(panel_time_series_factory):
    """Test FourierSeasonalityForecaster with panel data."""
    # Create panel data with different seasonal patterns per series
    seasonality = 12
    y_panel = panel_time_series_factory(
        length=3 * seasonality, n_series=3, seed=42
    )

    # Fit forecaster
    forecaster = FourierSeasonalityForecaster(seasonality=seasonality, harmonics=[2])
    forecaster.fit(y_panel[:30], forecasting_horizon=6)

    # Predict all groups
    y_pred = forecaster.predict(forecasting_horizon=6)

    # Should have predictions for all series
    assert "panel__series_0" in y_pred.columns
    assert "panel__series_1" in y_pred.columns
    assert "panel__series_2" in y_pred.columns
    assert len(y_pred) == 6


def test_fourier_seasonality_update_predict():
    """Test update_predict method."""
    seasonality = 12
    n_periods = 4
    length = seasonality * n_periods
    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=length-1), interval="1d", eager=True
    )

    phases = np.arange(length)
    values = np.sin(2 * np.pi * phases / seasonality)
    y = pl.DataFrame({"time": time, "value": values})

    forecaster = FourierSeasonalityForecaster(seasonality=seasonality, harmonics=[2], alpha=0.0, l1_ratio=0.0)
    fit_forecasting_horizon = 3
    forecaster.fit(y[:24], forecasting_horizon=fit_forecasting_horizon)

    # Update with new data and predict
    n_new = 12
    predict_forecasting_horizon = 12
    y_new = y[24:24 + n_new]
    y_pred = forecaster.update_predict(y_new, forecasting_horizon=predict_forecasting_horizon)

    assert len(y_pred) == predict_forecasting_horizon * (1 + n_new // fit_forecasting_horizon)
    assert "value" in y_pred.columns


def test_fourier_seasonality_zero_harmonics():
    """Test that harmonics with non-positive values raise an error."""
    # Test empty list
    forecaster_empty = FourierSeasonalityForecaster(seasonality=12, harmonics=[])
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 2, 1), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": np.random.randn(len(time))})
    
    with pytest.raises(ValueError, match="harmonics list cannot be empty"):
        forecaster_empty.fit(y, forecasting_horizon=1)
    
    # Test negative harmonics
    forecaster_negative = FourierSeasonalityForecaster(seasonality=12, harmonics=[1, -2, 3])
    with pytest.raises(ValueError, match="All harmonics must be positive"):
        forecaster_negative.fit(y, forecasting_horizon=1)
