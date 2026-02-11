"""Tests for validate_data functionality."""

from datetime import datetime

import polars as pl
import pytest

from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import ExpandingWindowSplitter
from yohou.point_forecaster import SeasonalNaive
from yohou.preprocessing import LagTransformer


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
    y = pl.DataFrame({
        "time": time,
        "sales__store_1": range(10),
        "sales__store_2": range(10, 20),
    })

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
    y = pl.DataFrame({
        "time": time,
        "sales__store_1": range(15),
        "sales__store_2": range(15, 30),
    })

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
    y = pl.DataFrame({
        "time": time,
        "target": range(10),
    })

    forecaster = SeasonalNaive(seasonality=2)
    forecaster.fit(y, forecasting_horizon=2)

    # Verify that forecaster stores observations correctly
    assert "time" in forecaster._y_observed.columns
    assert "target" in forecaster._y_observed.columns
    assert forecaster._y_observed.shape[0] == 2  # observation_horizon = seasonality = 2


# ==================== Transformer Tests ====================


@pytest.fixture
def sample_transformer_data():
    """Create sample transformer data."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 20),
        interval="1d",
        eager=True,
    )
    X = pl.DataFrame({"time": time, "feature1": range(20), "feature2": range(20, 40)})
    return X


def test_validate_transformer_fit_context(sample_transformer_data):
    """Test validate_transformer_data in fit context (reset=True)."""
    X = sample_transformer_data
    transformer = LagTransformer(lag=[1, 2])

    # Fit should call validate_transformer_data with reset=True
    transformer.fit(X)

    # Check that fitted attributes were set
    assert hasattr(transformer, "interval_")
    assert hasattr(transformer, "n_features_in_")
    assert transformer.n_features_in_ == 2
    assert transformer.observation_horizon == 2


def test_validate_transformer_transform_context(sample_transformer_data):
    """Test validate_transformer_data in transform context (reset=False)."""
    X = sample_transformer_data
    transformer = LagTransformer(lag=[1, 2])
    transformer.fit(X[:15])

    # Transform should call validate_transformer_data with reset=False
    X_transformed = transformer.transform(X[15:])

    # Should return DataFrame with lag features
    assert isinstance(X_transformed, pl.DataFrame)
    assert "time" in X_transformed.columns
    assert "feature1_lag_1" in X_transformed.columns
    assert "feature2_lag_2" in X_transformed.columns


def test_validate_transformer_schema_consistency(sample_transformer_data):
    """Test that transformer schema consistency is enforced."""
    X = sample_transformer_data
    transformer = LagTransformer(lag=[1, 2])
    transformer.fit(X[:15])

    # Wrong column name should fail
    X_wrong = X[15:].with_columns(pl.col("feature1").alias("wrong_col")).drop("feature1")

    with pytest.raises(Exception):  # ColumnNotFoundError or schema mismatch
        transformer.transform(X_wrong)


def test_validate_transformer_interval_consistency_fit(sample_transformer_data):
    """Test that transformer checks interval consistency during fit."""
    # Create data with inconsistent intervals
    time_wrong = [
        datetime(2020, 1, 1),
        datetime(2020, 1, 2),
        datetime(2020, 1, 4),  # Skip day 3 - inconsistent interval
        datetime(2020, 1, 5),
    ]
    X_wrong_interval = pl.DataFrame({"time": time_wrong, "feature1": [10, 11, 12, 13], "feature2": [20, 21, 22, 23]})

    transformer = LagTransformer(lag=1)
    with pytest.raises(ValueError, match="Time series has inconsistent intervals"):
        transformer.fit(X_wrong_interval)


def test_validate_transformer_panel_fit():
    """Test validate_transformer_data with panel data in fit context."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 10),
        interval="1d",
        eager=True,
    )
    X = pl.DataFrame({
        "time": time,
        "temp__sensor_1": range(10),
        "temp__sensor_2": range(10, 20),
    })

    transformer = LagTransformer(lag=1)
    transformer.fit(X)

    # Should detect panel structure (check internal state)
    assert hasattr(transformer, "interval_")


def test_validate_transformer_update_extends_memory(sample_transformer_data):
    """Test that transformer update extends observation memory."""
    X = sample_transformer_data
    transformer = LagTransformer(lag=[1, 2])
    transformer.fit(X[:15])

    initial_memory_size = transformer._X_observed.shape[0]

    # Update with new data
    transformer.update(X[15:17])

    # Memory should still be bounded by observation_horizon
    assert transformer._X_observed.shape[0] == transformer.observation_horizon


def test_validate_transformer_reset_memory(sample_transformer_data):
    """Test that transformer reset maintains observation horizon."""
    X = sample_transformer_data
    transformer = LagTransformer(lag=[1, 2])
    transformer.fit(X[:10])

    # Reset with new data
    transformer.reset(X[10:15])

    # Memory should be reset to observation_horizon rows
    assert transformer._X_observed.shape[0] == transformer.observation_horizon


# ==================== Scorer Tests ====================


@pytest.fixture
def sample_scorer_data():
    """Create sample scorer data (truth and predictions)."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 10),
        interval="1d",
        eager=True,
    )
    y_true = pl.DataFrame({"time": time, "target": range(10)})
    y_pred = pl.DataFrame({
        "observed_time": time[:10],
        "time": time[:10],
        "target": range(10),  # Use int to match y_true type
    })
    return y_true, y_pred


def test_validate_scorer_basic(sample_scorer_data):
    """Test validate_scorer_data basic functionality."""
    y_true, y_pred = sample_scorer_data
    scorer = MeanAbsoluteError()

    # Score should call validate_scorer_data
    scorer.fit(y_true)
    score = scorer.score(y_true, y_pred)

    # Should return a scalar or DataFrame depending on aggregation
    assert isinstance(score, (float, int, pl.DataFrame))


def test_validate_scorer_missing_column(sample_scorer_data):
    """Test that scorer validation checks column presence."""
    y_true, y_pred = sample_scorer_data
    scorer = MeanAbsoluteError()

    # Remove target column from predictions
    y_pred_wrong = y_pred.drop("target")

    with pytest.raises(ValueError, match="missing in `y_pred`"):
        scorer.fit(y_true)
        scorer.score(y_true, y_pred_wrong)


def test_validate_scorer_type_mismatch(sample_scorer_data):
    """Test that scorer validation checks column types."""
    y_true, y_pred = sample_scorer_data
    scorer = MeanAbsoluteError()

    # Change type of target column
    y_pred_wrong = y_pred.with_columns(pl.col("target").cast(pl.Utf8))

    with pytest.raises(ValueError, match="type mismatch"):
        scorer.fit(y_true)
        scorer.score(y_true, y_pred_wrong)


def test_validate_scorer_panel_consistency():
    """Test that scorer validation checks panel structure consistency."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 10),
        interval="1d",
        eager=True,
    )
    y_true = pl.DataFrame({
        "time": time,
        "sales__store_1": range(10),
        "sales__store_2": range(10, 20),
    })
    # Predictions missing one store
    y_pred = pl.DataFrame({
        "observed_time": time,
        "time": time,
        "sales__store_1": range(10),
        # Missing sales__store_2
    })

    scorer = MeanAbsoluteError()
    with pytest.raises(ValueError, match="missing in `y_pred`"):
        scorer.fit(y_true)
        scorer.score(y_true, y_pred)


def test_validate_scorer_none_inputs():
    """Test that scorer validation rejects None inputs."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 10),
        interval="1d",
        eager=True,
    )
    y_valid = pl.DataFrame({"time": time, "target": range(10)})
    scorer = MeanAbsoluteError()

    with pytest.raises(ValueError, match="Cannot be None"):
        scorer.fit(None)
        scorer.score(None, y_valid)

    with pytest.raises(ValueError, match="cannot be None"):
        scorer.fit(y_valid)
        scorer.score(y_valid, None)


# ==================== Splitter Tests ====================


@pytest.fixture
def sample_splitter_data():
    """Create sample splitter data."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 30),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "target": range(30)})
    X = pl.DataFrame({"time": time, "feature": range(30, 60)})
    return y, X


def test_validate_splitter_fit_context(sample_splitter_data):
    """Test validate_splitter_data in fit context."""
    y, X = sample_splitter_data
    splitter = ExpandingWindowSplitter(n_splits=3, test_size=3)

    # Split should call validate_splitter_data
    splits = list(splitter.split(y, X))

    # Should return valid train/test splits
    assert len(splits) > 0
    for train_idx, test_idx in splits:
        assert len(train_idx) >= 10
        assert len(test_idx) == 3


def test_validate_splitter_interval_consistency_fit(sample_splitter_data):
    """Test that splitter validates interval consistency."""
    # Create data with inconsistent intervals (need at least 3 rows)
    time_wrong = [
        datetime(2020, 1, 1),
        datetime(2020, 1, 2),
        datetime(2020, 1, 4),  # Skip day 3
        datetime(2020, 1, 5),
        datetime(2020, 1, 6),
    ]
    y_wrong = pl.DataFrame({"time": time_wrong, "target": [1, 2, 3, 4, 5]})

    splitter = ExpandingWindowSplitter(n_splits=2, test_size=1)

    with pytest.raises(ValueError, match="inconsistent intervals"):
        list(splitter.split(y_wrong))


def test_validate_splitter_panel_structure():
    """Test that splitter handles panel data structure."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 20),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({
        "time": time,
        "sales__store_1": range(20),
        "sales__store_2": range(20, 40),
    })

    splitter = ExpandingWindowSplitter(n_splits=2, test_size=3)

    # Should detect panel structure during split
    splits = list(splitter.split(y))
    assert len(splits) > 0

    # Splitter should handle panel data (validated internally)
    assert hasattr(splitter, "interval_")


def test_validate_splitter_panel_mismatch():
    """Test that splitter validates panel consistency between y and X."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 20),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({
        "time": time,
        "sales__store_1": range(20),
        "sales__store_2": range(20, 40),
    })
    X = pl.DataFrame({
        "time": time,
        "temp__sensor_1": range(40, 60),  # Different panel group
    })

    splitter = ExpandingWindowSplitter(n_splits=2, test_size=3)

    with pytest.raises(ValueError, match="Panel groups mismatch"):
        list(splitter.split(y, X))


def test_validate_splitter_y_none():
    """Test that splitter validation rejects None y."""
    splitter = ExpandingWindowSplitter(n_splits=2, test_size=3)

    with pytest.raises((ValueError, AttributeError), match="(cannot be None|'NoneType' object has no attribute)"):
        list(splitter.split(None))


def test_validate_splitter_sets_interval(sample_splitter_data):
    """Test that splitter validation sets interval attribute."""
    y, X = sample_splitter_data
    splitter = ExpandingWindowSplitter(n_splits=3, test_size=5)

    # Before split, interval_ should not exist
    assert not hasattr(splitter, "interval_")

    # After split, interval_ should be set
    list(splitter.split(y, X))
    assert hasattr(splitter, "interval_")
    assert splitter.interval_ is not None


def test_validate_splitter_panel_internal_consistency():
    """Test that splitter validates internal panel structure consistency."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 20),
        interval="1d",
        eager=True,
    )
    # Create panel data with mismatched local columns
    y = pl.DataFrame({
        "time": time,
        "sales__store_1": range(20),
        "sales__store_2": range(20, 40),
        "revenue__store_1": range(40, 60),
        # Missing revenue__store_2 - internal inconsistency
    })

    splitter = ExpandingWindowSplitter(n_splits=2, test_size=3)

    with pytest.raises(ValueError, match="Panel structure mismatch"):
        list(splitter.split(y))
