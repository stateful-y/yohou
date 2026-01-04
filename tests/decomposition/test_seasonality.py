"""Tests for SeasonalityForecaster."""

import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from sklearn.base import clone

from yohou.decomposition import SeasonalityForecaster

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from estimator_checks import _yield_yohou_forecaster_checks


@pytest.mark.parametrize(
    "forecaster,tags,expected_failures",
    [
        (
            SeasonalityForecaster(seasonality=12, method="naive"),
            {"forecaster_type": "point", "uses_reduction": False},
            [],
        ),
        (
            SeasonalityForecaster(seasonality=12, method="average"),
            {"forecaster_type": "point", "uses_reduction": False},
            [],
        ),
        (
            SeasonalityForecaster(seasonality=12, method="median"),
            {"forecaster_type": "point", "uses_reduction": False},
            [],
        ),
        (
            SeasonalityForecaster(seasonality=7, method="average"),
            {"forecaster_type": "point", "uses_reduction": False},
            [],
        ),
    ],
)
def test_seasonality_forecaster_checks(forecaster, tags, expected_failures, y_X_factory):
    """Run systematic checks on SeasonalityForecaster variants."""
    # Generate data with sufficient length for 2+ cycles
    seasonality = forecaster.seasonality
    y, X = y_X_factory(length=3 * seasonality, n_targets=1, n_features=0, seed=42)

    # Add repeating seasonal pattern
    pattern = list(range(seasonality))
    y = y.with_columns(
        [
            pl.Series(col, pattern * 3).alias(col)
            for col in y.columns
            if col != "time"
        ]
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
        tags=tags,
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
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=17), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": pattern * 3})  # 3 cycles

    forecaster = SeasonalityForecaster(seasonality=6, method="naive")
    forecaster.fit(y, forecasting_horizon=1)

    # Predict next 12 steps (2 cycles)
    y_pred = forecaster.predict(forecasting_horizon=12)

    # Should repeat last cycle twice
    expected_values = pattern * 2
    pred_values = y_pred["value"].to_list()

    assert (
        pred_values == expected_values
    ), "Naive seasonality should repeat last cycle exactly"


def test_seasonality_average_aggregates_cycles():
    """Test average method aggregates across cycles."""
    # Create pattern with variation across cycles
    base_pattern = [10, 15, 12, 8, 9, 11]
    cycle1 = base_pattern
    cycle2 = [x + 1 for x in base_pattern]
    cycle3 = [x - 1 for x in base_pattern]

    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=17), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": cycle1 + cycle2 + cycle3})

    forecaster = SeasonalityForecaster(seasonality=6, method="average")
    forecaster.fit(y, forecasting_horizon=1)

    # Predict next 6 steps (1 cycle)
    y_pred = forecaster.predict(forecasting_horizon=6)

    # Should be average of three cycles (which equals base_pattern)
    pred_values = y_pred["value"].to_list()

    assert (
        pred_values == base_pattern
    ), "Average seasonality should aggregate across cycles"


def test_seasonality_median_robustness():
    """Test median method is robust to outliers."""
    # Create pattern with outlier in one cycle
    base_pattern = [10, 15, 12, 8, 9, 11]
    cycle1 = base_pattern
    cycle2 = base_pattern.copy()
    cycle3 = [100, 15, 12, 8, 9, 11]  # Outlier in first position

    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=17), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": cycle1 + cycle2 + cycle3})

    forecaster = SeasonalityForecaster(seasonality=6, method="median")
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
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=11), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": list(range(12))})

    forecaster = SeasonalityForecaster(seasonality=12, method="naive")
    # This should work
    forecaster.fit(y, forecasting_horizon=1)

    # But less than 1 cycle should fail
    y_short = y[:11]
    forecaster_short = SeasonalityForecaster(seasonality=12, method="naive")
    with pytest.raises(ValueError, match="Insufficient data"):
        forecaster_short.fit(y_short, forecasting_horizon=1)


def test_seasonality_insufficient_data_average():
    """Test error handling for insufficient data (average method)."""
    # Only 1 cycle for average method (need 2)
    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=11), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": list(range(12))})

    forecaster = SeasonalityForecaster(seasonality=12, method="average")

    with pytest.raises(ValueError, match="Insufficient data"):
        forecaster.fit(y, forecasting_horizon=1)


def test_seasonality_wraps_around_correctly():
    """Test that predictions wrap around seasonal cycle correctly."""
    pattern = [1, 2, 3, 4]
    from datetime import timedelta
    time = pl.datetime_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=11), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": pattern * 3})

    forecaster = SeasonalityForecaster(seasonality=4, method="naive")
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
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=17), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": pattern * 3})

    forecaster = SeasonalityForecaster(seasonality=6, method="naive")
    forecaster.fit(y, forecasting_horizon=5)

    # Predict different horizon than fit
    y_pred_12 = forecaster.predict(forecasting_horizon=12)
    assert len(y_pred_12) == 12

    y_pred_3 = forecaster.predict(forecasting_horizon=3)
    assert len(y_pred_3) == 3


def test_seasonality_panel_data(panel_time_series_factory):
    """Test SeasonalityForecaster with panel data."""
    # Create panel data with different seasonal patterns per series
    seasonality = 12
    y_panel = panel_time_series_factory(
        length=3 * seasonality, n_series=3, seed=42
    )

    # Add seasonal patterns
    pattern1 = list(range(seasonality))
    pattern2 = [x * 2 for x in pattern1]
    pattern3 = [x * 0.5 for x in pattern1]

    # This is a simplified approach - in reality would need to properly set panel data
    forecaster = SeasonalityForecaster(seasonality=seasonality, method="naive")
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
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 1) + timedelta(days=23), interval="1d", eager=True
    )
    y = pl.DataFrame({"time": time, "value": pattern * 4})

    forecaster = SeasonalityForecaster(seasonality=6, method="naive")
    fit_forecasting_horizon = 3
    forecaster.fit(y[:12], forecasting_horizon=fit_forecasting_horizon)

    # Update with new data and predict
    n_new = 6
    predict_forecasting_horizon = 6
    y_new = y[12:12 + n_new]
    y_pred = forecaster.update_predict(y_new, forecasting_horizon=predict_forecasting_horizon)

    assert len(y_pred) == predict_forecasting_horizon * (1 + n_new // fit_forecasting_horizon)
    assert "value" in y_pred.columns
