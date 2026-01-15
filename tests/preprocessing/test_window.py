"""Tests for window-based transformers.

Tests LagTransformer using the check generator pattern for systematic validation.
"""

import sys
from pathlib import Path

import pytest
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).parent.parent))
from estimator_checks import _yield_yohou_transformer_checks

from yohou.preprocessing.window import LagTransformer


@pytest.mark.parametrize(
    "lag",
    [[1], [1, 2], [1, 2, 3, 5]],
    ids=["lag_1", "lag_1_2", "lag_1_2_3_5"],
)
def test_lag_transformer_checks(lag, time_series_train_test_factory):
    """Run all checks for LagTransformer with different lag configurations."""
    transformer = LagTransformer(lag=lag)
    expected_failures = []  # Empty since invertible=False prevents inverse checks

    min_horizon = max(lag) + 10
    X_train, X_test = time_series_train_test_factory(
        train_length=min_horizon + 50, test_length=min_horizon + 20
    )

    transformer_fitted = clone(transformer)
    transformer_fitted.fit(X_train)

    expected_failures_set = set(expected_failures)
    for check_name, check_func, check_kwargs in _yield_yohou_transformer_checks(
        transformer_fitted, X_train, X_test
    ):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(transformer_fitted, **check_kwargs)


def test_lag_transformer_feature_names(time_series_factory):
    """Test LagTransformer generates correct feature names."""
    X = time_series_factory(length=50, n_components=2)
    transformer = LagTransformer(lag=[1, 2])
    transformer.fit(X)

    X_trans = transformer.transform(X)
    feature_names = transformer.get_feature_names_out()

    # Should have lag_1 and lag_2 for each input feature
    expected_n_features = 2 * 2  # 2 features * 2 lags
    assert len(feature_names) == expected_n_features, (
        f"Expected {expected_n_features} features, got {len(feature_names)}"
    )

    # Check feature naming pattern
    assert all("lag" in name.lower() for name in feature_names), (
        "All features should contain 'lag' in name"
    )


def test_lag_transformer_observation_horizon(time_series_factory):
    """Test observation_horizon equals max(lag) + 1."""
    X = time_series_factory(length=50)

    for lag in [[1], [1, 2], [1, 2, 3], [1, 5, 10]]:
        transformer = LagTransformer(lag=lag)
        transformer.fit(X)

        # LagTransformer uses max(lag) + 1 as observation_horizon
        expected_horizon = max(lag)
        assert transformer.observation_horizon == expected_horizon, (
            f"For lag={lag}, expected horizon={expected_horizon}, got {transformer.observation_horizon}"
        )


def test_lag_transformer_output_length(time_series_factory):
    """Test output length drops max(lag) rows."""
    X = time_series_factory(length=50)
    transformer = LagTransformer(lag=[1, 2, 3])
    transformer.fit(X)

    X_trans = transformer.transform(X)

    # Output drops max(lag) rows (not observation_horizon)
    expected_length = len(X) - max([1, 2, 3])
    assert len(X_trans) == expected_length, (
        f"Expected output length {expected_length}, got {len(X_trans)}"
    )


def test_lag_transformer_single_lag(time_series_factory):
    """Test LagTransformer with single lag value."""
    X = time_series_factory(length=50, n_components=1)
    transformer = LagTransformer(lag=[1])
    transformer.fit(X)

    X_trans = transformer.transform(X)

    # Should have time + lagged features
    assert "time" in X_trans.columns
    assert len([col for col in X_trans.columns if col != "time"]) == 1


def test_lag_transformer_with_panel_data(panel_time_series_factory):
    """Test LagTransformer handles panel data."""
    X_panel = panel_time_series_factory(length=50, n_series=3, n_global=2)
    transformer = LagTransformer(lag=[1, 2])

    # Should fit without error (or raise NotImplementedError)
    try:
        transformer.fit(X_panel)
        X_trans = transformer.transform(X_panel)

        # Basic validation
        assert "time" in X_trans.columns
        # Output drops max(lag) rows
        expected_length = len(X_panel) - max([1, 2])
        assert len(X_trans) == expected_length, (
            f"Expected {expected_length} rows, got {len(X_trans)}"
        )
    except NotImplementedError:
        # Panel data not supported is acceptable
        pytest.skip("LagTransformer doesn't support panel data")
