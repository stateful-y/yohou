"""Tests for validate_data functionality."""

from datetime import datetime

import polars as pl
import pytest

from yohou.point_forecaster import SeasonalNaive


@pytest.fixture
def sample_time_series():
    """Create a sample time series for testing."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 20),
        interval="1d",
        eager=True,
    )
    return pl.DataFrame({"time": time, "value": range(20)})


@pytest.fixture
def sample_forecaster_data():
    """Create sample forecaster data (target + features)."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 20),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "target": range(20)})
    X = pl.DataFrame({"time": time, "feature": range(20, 40)})
    return y, X


def test_validate_data_fit_context(sample_forecaster_data):
    """Test validate_data in fit context (reset=True)."""
    y, X = sample_forecaster_data
    forecaster = SeasonalNaive(seasonality=2)

    # Fit should call validate_data with reset=True
    forecaster.fit(y, X, forecasting_horizon=2)

    # Check that fitted attributes were set
    assert hasattr(forecaster, "interval_")
    assert hasattr(forecaster, "n_features_in_")
    assert hasattr(forecaster, "local_y_schema_")
    # SeasonalNaive uses input_features="y_t|X", so features include y_t + X
    assert forecaster.n_features_in_ == 2


def test_validate_data_update_context(sample_forecaster_data):
    """Test validate_data in update context (reset=False)."""
    y, X = sample_forecaster_data
    forecaster = SeasonalNaive(seasonality=2)
    forecaster.fit(y[:15], X[:15], forecasting_horizon=2)

    # Update should call validate_data with reset=False (schema check only, no interval validation yet)
    forecaster.update(y[15:], X[15:])

    # Should not change n_features_in_
    # SeasonalNaive uses input_features="y_t|X", so features include y_t + X
    assert forecaster.n_features_in_ == 2


def test_validate_data_schema_consistency(sample_forecaster_data):
    """Test that schema consistency is enforced."""
    y, X = sample_forecaster_data
    forecaster = SeasonalNaive(seasonality=2)
    forecaster.fit(y[:15], forecasting_horizon=2)

    # Wrong column name should fail
    y_wrong = y[15:].with_columns(pl.col("target").alias("wrong_col")).drop("target")

    with pytest.raises(Exception):  # ColumnNotFoundError
        forecaster.update(y_wrong)


def test_validate_data_interval_consistency_fit(sample_forecaster_data):
    """Test that interval consistency is checked during fit."""
    y, X = sample_forecaster_data

    # Create data with inconsistent intervals
    time_wrong = [
        datetime(2020, 1, 1),
        datetime(2020, 1, 2),
        datetime(2020, 1, 4),  # Skip day 3 - inconsistent interval
        datetime(2020, 1, 5),
    ]
    y_wrong_interval = pl.DataFrame({"time": time_wrong, "target": [10, 11, 12, 13]})

    forecaster = SeasonalNaive(seasonality=2)
    with pytest.raises(ValueError, match="Time series has inconsistent intervals"):
        forecaster.fit(y_wrong_interval, forecasting_horizon=2)


def test_validate_data_panel_fit():
    """Test validate_data with panel data in fit context."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 10),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame(
        {
            "time": time,
            "sales__store_1": range(10),
            "sales__store_2": range(10, 20),
        }
    )

    forecaster = SeasonalNaive(seasonality=2)
    forecaster.fit(y, forecasting_horizon=2)

    # Should detect panel structure
    assert forecaster.panel_group_names_ == ["sales"]
    assert "store_1" in forecaster.local_y_schema_
    assert "store_2" in forecaster.local_y_schema_


def test_validate_data_panel_update():
    """Test validate_data with panel data in update context."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 15),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame(
        {
            "time": time,
            "sales__store_1": range(15),
            "sales__store_2": range(15, 30),
        }
    )

    forecaster = SeasonalNaive(seasonality=2)
    forecaster.fit(y[:10], forecasting_horizon=2)

    # Update with panel data
    forecaster.update(y[10:])

    # Should maintain panel structure
    assert forecaster.panel_group_names_ == ["sales"]


def test_validate_data_with_none_X():
    """Test validate_data works when X is None."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 10),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "target": range(10)})

    forecaster = SeasonalNaive(seasonality=2)
    forecaster.fit(y, X=None, forecasting_horizon=2)

    # Update with X=None should work
    forecaster.update(y[-2:], X=None)

    assert forecaster._y_observed.shape[0] == 2


def test_validate_data_preserves_column_order():
    """Test that validate_data ensures time column is handled correctly."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 10),
        interval="1d",
        eager=True,
    )
    # Create with columns in expected order
    y = pl.DataFrame(
        {
            "time": time,
            "target": range(10),
        }
    )

    forecaster = SeasonalNaive(seasonality=2)
    forecaster.fit(y, forecasting_horizon=2)

    # Verify that forecaster stores observations correctly
    assert "time" in forecaster._y_observed.columns
    assert "target" in forecaster._y_observed.columns
    assert forecaster._y_observed.shape[0] == 2  # observation_horizon = seasonality = 2
