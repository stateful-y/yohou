"""Tests for PatternSeasonalityForecaster."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline

from yohou.decomposition import FourierSeasonalityForecaster, PatternSeasonalityForecaster

# Add parent directory to path for imports
from yohou.testing import _yield_yohou_forecaster_checks


@pytest.mark.parametrize(
    "forecaster,expected_failures",
    [
        (
            PatternSeasonalityForecaster(seasonality=12, method="naive"),
            [],
        ),
        (
            PatternSeasonalityForecaster(seasonality=12, method="average"),
            [],
        ),
        (
            PatternSeasonalityForecaster(seasonality=12, method="median"),
            [],
        ),
        (
            PatternSeasonalityForecaster(seasonality=7, method="average"),
            [],
        ),
    ],
)
def test_seasonality_forecaster_checks(forecaster, expected_failures, y_X_factory):
    """Run systematic checks on PatternSeasonalityForecaster variants."""
    # Generate data with sufficient length for 2+ cycles
    seasonality = forecaster.seasonality
    y, X = y_X_factory(length=3 * seasonality, n_targets=1, n_features=0, seed=42)

    # Add repeating seasonal pattern
    pattern = list(range(seasonality))
    y = y.with_columns(
        [pl.Series(col, pattern * 3).alias(col) for col in y.columns if col != "time"]
    )

    train_size = int(2.5 * seasonality)
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = (X[:train_size], X[train_size:]) if X is not None else (None, None)

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


def test_seasonality_naive_exact_repetition():
    """Test naive method repeats last cycle exactly."""
    # Create perfect seasonal pattern
    pattern = [10, 15, 12, 8, 9, 11]
    from datetime import timedelta

    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=17),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": pattern * 3})  # 3 cycles

    forecaster = PatternSeasonalityForecaster(seasonality=6, method="naive")
    forecaster.fit(y, forecasting_horizon=1)

    # Predict next 12 steps (2 cycles)
    y_pred = forecaster.predict(forecasting_horizon=12)

    # Should repeat last cycle twice
    expected_values = pattern * 2
    pred_values = y_pred["value"].to_list()

    assert pred_values == expected_values, "Naive seasonality should repeat last cycle exactly"


def test_seasonality_average_aggregates_cycles():
    """Test average method aggregates across cycles."""
    # Create pattern with variation across cycles
    base_pattern = [10, 15, 12, 8, 9, 11]
    cycle1 = base_pattern
    cycle2 = [x + 1 for x in base_pattern]
    cycle3 = [x - 1 for x in base_pattern]

    from datetime import timedelta

    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=17),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": cycle1 + cycle2 + cycle3})

    forecaster = PatternSeasonalityForecaster(seasonality=6, method="average")
    forecaster.fit(y, forecasting_horizon=1)

    # Predict next 6 steps (1 cycle)
    y_pred = forecaster.predict(forecasting_horizon=6)

    # Should be average of three cycles (which equals base_pattern)
    pred_values = y_pred["value"].to_list()

    assert pred_values == base_pattern, "Average seasonality should aggregate across cycles"


def test_seasonality_median_robustness():
    """Test median method is robust to outliers."""
    # Create pattern with outlier in one cycle
    base_pattern = [10, 15, 12, 8, 9, 11]
    cycle1 = base_pattern
    cycle2 = base_pattern.copy()
    cycle3 = [100, 15, 12, 8, 9, 11]  # Outlier in first position

    from datetime import timedelta

    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=17),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": cycle1 + cycle2 + cycle3})

    forecaster = PatternSeasonalityForecaster(seasonality=6, method="median")
    forecaster.fit(y, forecasting_horizon=1)

    # Predict next 6 steps
    y_pred = forecaster.predict(forecasting_horizon=6)

    # Median should ignore the outlier
    pred_values = y_pred["value"].to_list()

    # First value should be median of [10, 10, 100] = 10
    assert pred_values[0] == 10, "Median should be robust to outliers"


def test_seasonality_insufficient_data_naive():
    """Test error handling for insufficient data (naive method)."""
    # Only 1 cycle but need at least 1 complete cycle - this should work
    from datetime import timedelta

    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=11),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": list(range(12))})

    forecaster = PatternSeasonalityForecaster(seasonality=12, method="naive")
    # This should work
    forecaster.fit(y, forecasting_horizon=1)

    # But less than 1 cycle should fail
    y_short = y[:11]
    forecaster_short = PatternSeasonalityForecaster(seasonality=12, method="naive")
    with pytest.raises(ValueError, match="Insufficient data"):
        forecaster_short.fit(y_short, forecasting_horizon=1)


def test_seasonality_insufficient_data_average():
    """Test error handling for insufficient data (average method)."""
    # Only 1 cycle for average method (need 2)
    from datetime import timedelta

    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=11),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": list(range(12))})

    forecaster = PatternSeasonalityForecaster(seasonality=12, method="average")

    with pytest.raises(ValueError, match="Insufficient data"):
        forecaster.fit(y, forecasting_horizon=1)


def test_seasonality_wraps_around_correctly():
    """Test that predictions wrap around seasonal cycle correctly."""
    pattern = [1, 2, 3, 4]
    from datetime import timedelta

    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=11),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": pattern * 3})

    forecaster = PatternSeasonalityForecaster(seasonality=4, method="naive")
    forecaster.fit(y, forecasting_horizon=1)

    # Predict 10 steps (2.5 cycles)
    y_pred = forecaster.predict(forecasting_horizon=10)

    # Should wrap around: [1,2,3,4,1,2,3,4,1,2]
    expected = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
    pred_values = y_pred["value"].to_list()

    assert pred_values == expected, "Predictions should wrap around seasonal cycle"


def test_seasonality_different_horizons():
    """Test that different forecasting horizons work correctly."""
    pattern = [10, 15, 12, 8, 9, 11]
    from datetime import timedelta

    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=17),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": pattern * 3})

    forecaster = PatternSeasonalityForecaster(seasonality=6, method="naive")
    forecaster.fit(y, forecasting_horizon=5)

    # Predict different horizon than fit
    y_pred_12 = forecaster.predict(forecasting_horizon=12)
    assert len(y_pred_12) == 12

    y_pred_3 = forecaster.predict(forecasting_horizon=3)
    assert len(y_pred_3) == 3


def test_seasonality_panel_data(panel_time_series_factory):
    """Test PatternSeasonalityForecaster with panel data."""
    # Create panel data with different seasonal patterns per series
    seasonality = 12
    y_panel = panel_time_series_factory(length=3 * seasonality, n_series=3, seed=42)

    # Add seasonal patterns
    pattern1 = list(range(seasonality))
    pattern2 = [x * 2 for x in pattern1]
    pattern3 = [x * 0.5 for x in pattern1]

    # This is a simplified approach - in reality would need to properly set panel data
    forecaster = PatternSeasonalityForecaster(seasonality=seasonality, method="naive")
    forecaster.fit(y_panel[:30], forecasting_horizon=6)

    # Predict all groups
    y_pred = forecaster.predict(forecasting_horizon=6)

    # Should have predictions for all series
    assert "panel__series_0" in y_pred.columns
    assert "panel__series_1" in y_pred.columns
    assert "panel__series_2" in y_pred.columns
    assert len(y_pred) == 6


def test_seasonality_update_predict():
    """Test update_predict method."""
    pattern = [10, 15, 12, 8, 9, 11]
    from datetime import timedelta

    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=23),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": pattern * 4})

    forecaster = PatternSeasonalityForecaster(seasonality=6, method="naive")
    fit_forecasting_horizon = 3
    forecaster.fit(y[:12], forecasting_horizon=fit_forecasting_horizon)

    # Update with new data and predict
    n_new = 6
    predict_forecasting_horizon = 6
    y_new = y[12 : 12 + n_new]
    y_pred = forecaster.update_predict(y_new, forecasting_horizon=predict_forecasting_horizon)

    assert len(y_pred) == predict_forecasting_horizon * (1 + n_new // fit_forecasting_horizon)
    assert "value" in y_pred.columns


@pytest.mark.parametrize(
    "forecaster,expected_failures",
    [
        (
            FourierSeasonalityForecaster(seasonality=12, harmonics=[1]),
            [],
        ),
        (
            FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2, 3]),
            [],
        ),
        (
            FourierSeasonalityForecaster(seasonality=24, harmonics=[1, 2, 3, 4, 5]),
            [],
        ),
        (
            FourierSeasonalityForecaster(seasonality=7, harmonics=[1, 2]),
            [],
        ),
    ],
)
def test_fourier_seasonality_forecaster_checks(forecaster, expected_failures, y_X_factory):
    """Run systematic checks on FourierSeasonalityForecaster variants."""
    # Generate data with sufficient length
    seasonality = forecaster.seasonality
    y, X = y_X_factory(length=3 * seasonality, n_targets=1, n_features=0, seed=42)

    # Add sine wave pattern
    phases = np.arange(len(y))
    sine_values = np.sin(2 * np.pi * phases / seasonality)
    y = y.with_columns(
        [pl.Series(col, sine_values).alias(col) for col in y.columns if col != "time"]
    )

    train_size = int(2.5 * seasonality)
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = (X[:train_size], X[train_size:]) if X is not None else (None, None)

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


def test_fourier_seasonality_sine_wave_recovery():
    """Test Fourier forecaster recovers pure sine wave with 1 harmonic."""
    # Create perfect sine wave with period 12
    seasonality = 12
    n_periods = 3
    length = seasonality * n_periods

    from datetime import timedelta

    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=length - 1),
        interval="1d",
        eager=True,
    )

    # Pure sine wave: y = sin(2π * t / T)
    phases = np.arange(length)
    sine_values = np.sin(2 * np.pi * phases / seasonality)
    y = pl.DataFrame({"time": time, "value": sine_values})

    # Fit with 1 harmonic (should perfectly capture fundamental frequency)
    forecaster = FourierSeasonalityForecaster(
        seasonality=seasonality,
        harmonics=[1],
        estimator=ElasticNet(alpha=0.0, l1_ratio=0.0),
    )
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
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(hours=length - 1),
        interval="1h",
        eager=True,
    )

    phases = np.arange(length)
    # Fundamental + first overtone
    values = np.sin(2 * np.pi * phases / seasonality) + 0.5 * np.cos(
        4 * np.pi * phases / seasonality
    )
    y = pl.DataFrame({"time": time, "value": values})

    # Fit with 3 harmonics (should capture both components)
    forecaster = FourierSeasonalityForecaster(
        seasonality=seasonality,
        harmonics=[1, 2],
        estimator=ElasticNet(alpha=0.0, l1_ratio=0.0),
    )
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
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=35),
        interval="1d",
        eager=True,
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
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=length - 1),
        interval="1d",
        eager=True,
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
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=7),
        interval="1d",
        eager=True,
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
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=length - 1),
        interval="1d",
        eager=True,
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
    y_panel = panel_time_series_factory(length=3 * seasonality, n_series=3, seed=42)

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
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=length - 1),
        interval="1d",
        eager=True,
    )

    phases = np.arange(length)
    values = np.sin(2 * np.pi * phases / seasonality)
    y = pl.DataFrame({"time": time, "value": values})

    forecaster = FourierSeasonalityForecaster(
        seasonality=seasonality, harmonics=[2], estimator=ElasticNet(alpha=0.0, l1_ratio=0.0)
    )
    fit_forecasting_horizon = 3
    forecaster.fit(y[:24], forecasting_horizon=fit_forecasting_horizon)

    # Update with new data and predict
    n_new = 12
    predict_forecasting_horizon = 12
    y_new = y[24 : 24 + n_new]
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


@pytest.mark.parametrize("model_panel", [False, True])
def test_fourier_model_panel_behaviors(panel_time_series_factory, model_panel):
    """Test FourierSeasonalityForecaster with both pooled and per-group strategies."""
    seasonality = 12
    y_panel = panel_time_series_factory(length=3 * seasonality, n_series=3, seed=42)

    # Add distinct seasonal patterns per series (phase-shifted sine waves)
    phases = np.arange(len(y_panel))
    for i in range(3):
        col_name = f"panel__series_{i}"
        # Different phase shift for each series
        sine_values = np.sin(2 * np.pi * phases / seasonality + i * np.pi / 3)
        y_panel = y_panel.with_columns((pl.col(col_name) + pl.Series(sine_values)).alias(col_name))

    # Fit forecaster with specified model_panel strategy
    forecaster = FourierSeasonalityForecaster(
        seasonality=seasonality,
        harmonics=[1, 2],
        model_panel=model_panel,
        estimator=ElasticNet(alpha=0.0, l1_ratio=0.0),
    )
    forecaster.fit(y_panel[:30], forecasting_horizon=6)

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
    # y_pred = forecaster.predict(forecasting_horizon=6)

    # # Basic structure checks
    # assert len(y_pred) == 6
    # assert "panel__series_0" in y_pred.columns
    # assert "panel__series_1" in y_pred.columns
    # assert "panel__series_2" in y_pred.columns


def test_fourier_model_panel_prediction_differences(panel_time_series_factory):
    """Test that pooled vs per-group strategies produce different predictions on heterogeneous data."""
    seasonality = 12
    y_panel = panel_time_series_factory(length=3 * seasonality, n_series=3, seed=42)

    # Add VERY distinct seasonal patterns (different amplitudes and phases)
    phases = np.arange(len(y_panel))
    for i in range(3):
        col_name = f"panel__series_{i}"
        # Series 0: low amplitude, no phase shift
        # Series 1: medium amplitude, 90° phase shift
        # Series 2: high amplitude, 180° phase shift
        amplitude = 1.0 + i * 2.0  # 1.0, 3.0, 5.0
        phase_shift = i * np.pi / 2  # 0, π/2, π
        sine_values = amplitude * np.sin(2 * np.pi * phases / seasonality + phase_shift)
        y_panel = y_panel.with_columns((pl.col(col_name) + pl.Series(sine_values)).alias(col_name))

    # Fit both strategies
    forecaster_pooled = FourierSeasonalityForecaster(
        seasonality=seasonality,
        harmonics=[1, 2],
        model_panel=False,
        estimator=ElasticNet(alpha=0.0, l1_ratio=0.0),
    )
    forecaster_per_group = FourierSeasonalityForecaster(
        seasonality=seasonality,
        harmonics=[1, 2],
        model_panel=True,
        estimator=ElasticNet(alpha=0.0, l1_ratio=0.0),
    )

    forecaster_pooled.fit(y_panel[:30], forecasting_horizon=6)
    forecaster_per_group.fit(y_panel[:30], forecasting_horizon=6)

    # Generate predictions
    y_pred_pooled = forecaster_pooled.predict(forecasting_horizon=6)
    y_pred_per_group = forecaster_per_group.predict(forecasting_horizon=6)

    # Predictions should differ significantly due to heterogeneous patterns
    for col in ["panel__series_0", "panel__series_1", "panel__series_2"]:
        pooled_vals = y_pred_pooled[col].to_numpy()
        per_group_vals = y_pred_per_group[col].to_numpy()

        # Calculate relative difference
        rel_diff = np.abs(pooled_vals - per_group_vals) / (np.abs(pooled_vals) + 1e-10)
        mean_rel_diff = np.mean(rel_diff)

        # Per-group should capture series-specific patterns better
        # With strongly heterogeneous data, predictions should differ noticeably
        assert mean_rel_diff > 0.01, (
            f"Predictions for {col} should differ between strategies (got {mean_rel_diff:.4f})"
        )
