"""Tests for stationarization transformers.

Tests SeasonalDifferencing and SeasonalLogDifferencing transformers
using both the check generator pattern and transformer-specific tests.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.base import clone

from yohou.preprocessing.stationarization import (
    SeasonalDifferencing,
    SeasonalLogDifferencing,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from estimator_checks import _yield_yohou_transformer_checks

length = 52

X = pl.DataFrame(
    {
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 0, 0, length - 1),
            interval="1s",
            eager=True,
        ),
        "a": np.random.rand(length),
        "b": np.random.rand(length),
    }
)


@pytest.mark.parametrize(
    "transformer",
    [
        SeasonalDifferencing(1),
        SeasonalDifferencing(5),
        SeasonalLogDifferencing(1, 2),
        SeasonalLogDifferencing(3, 1),
    ],
)
def test_identity(transformer):
    X_t = transformer.fit_transform(X)
    observation_horizon = transformer.observation_horizon

    X_it = transformer.inverse_transform(X_t=X_t, X_p=X[:observation_horizon])

    assert_frame_equal(X_it, X[observation_horizon:])


# ============================================================================
# COMPREHENSIVE CHECK GENERATOR TESTS
# ============================================================================


@pytest.mark.parametrize(
    "transformer,tags,expected_failures",
    [
        (
            SeasonalDifferencing(seasonality=1),
            {"invertible": True, "stateful": True},
            [],
        ),
        (
            SeasonalLogDifferencing(seasonality=1),
            {"invertible": True, "requires_positive_X": True, "stateful": True},
            [],
        ),
    ],
    ids=["SeasonalDifferencing", "SeasonalLogDifferencing"],
)
def test_stationarization_transformer_checks(
    transformer,
    tags,
    expected_failures,
    time_series_factory,
):
    """Run all applicable checks for stationarization transformers."""
    # Generate test data using fixture factory
    min_horizon = 10
    X_train = time_series_factory(length=min_horizon + 50, seed=42)
    X_test = time_series_factory(length=min_horizon + 20, seed=123)

    # Make data positive for log transforms
    if "requires_positive_X" in tags and tags["requires_positive_X"]:
        X_train = X_train.select([pl.col("time"), (pl.all().exclude("time") + 100.0)])
        X_test = X_test.select([pl.col("time"), (pl.all().exclude("time") + 100.0)])

    # Fit transformer
    transformer_fitted = clone(transformer)
    transformer_fitted.fit(X_train)

    # Run all checks from generator
    expected_failures_set = set(expected_failures)

    for check_name, check_func, check_kwargs in _yield_yohou_transformer_checks(
        transformer_fitted, X_train, X_test, tags=tags
    ):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(transformer_fitted, **check_kwargs)


def test_seasonal_differencing_inverse(time_series_factory):
    """Test SeasonalDifferencing specific round-trip behavior."""
    X = time_series_factory(length=50)
    transformer = SeasonalDifferencing(seasonality=1)
    transformer.fit(X)

    X_trans = transformer.transform(X)
    # inverse_transform requires X_p (past observations)
    horizon = transformer.observation_horizon
    n_dropped = len(X) - len(X_trans)
    X_p = X[n_dropped - horizon : n_dropped]
    X_inv = transformer.inverse_transform(X_trans, X_p)

    # Should recover original shape (for the transformed portion)
    X_expected = X.tail(len(X_trans))
    assert X_expected.shape == X_inv.shape
    # Should have same columns
    assert set(X_expected.columns) == set(X_inv.columns)


def test_seasonal_log_differencing_requires_positive(time_series_factory):
    """Test SeasonalLogDifferencing with non-positive data doesn't crash."""
    X = time_series_factory(length=50)
    # Create data with negative values
    X_negative = X.select([pl.col("time"), (pl.all().exclude("time") - 50.0)])

    transformer = SeasonalLogDifferencing(seasonality=1)

    # Should fit without error
    transformer.fit(X_negative)
    # Transform may produce -inf or NaN but shouldn't crash
    X_trans = transformer.transform(X_negative)

    # Basic validation - should return a dataframe with time column
    assert "time" in X_trans.columns
    assert len(X_trans) > 0


def test_seasonal_differencing_seasonality_2(time_series_factory):
    """Test SeasonalDifferencing with seasonality=2."""
    X = time_series_factory(length=50)
    transformer = SeasonalDifferencing(seasonality=2)
    transformer.fit(X)

    X_trans = transformer.transform(X)

    # Should have time column
    assert "time" in X_trans.columns
    # Output length should be input length - seasonality
    assert len(X_trans) == len(X) - 2


def test_seasonal_differencing_observation_horizon(time_series_factory):
    """Test observation_horizon matches seasonality."""
    X = time_series_factory(length=50)

    for seasonality in [1, 2, 3, 7]:
        transformer = SeasonalDifferencing(seasonality=seasonality)
        transformer.fit(X)

        assert transformer.observation_horizon == seasonality, (
            f"Expected observation_horizon={seasonality}, got {transformer.observation_horizon}"
        )
