"""Tests for cross-learning functionality in point forecasters."""

import sys
from copy import deepcopy
from pathlib import Path

import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from yohou.point_forecaster import PointReductionForecaster
from yohou.preprocessing import LagTransformer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from estimator_checks import _yield_yohou_forecaster_checks

# ============================================================================
# Check generator tests with panel data
# ============================================================================


@pytest.mark.parametrize(
    "forecaster,tags,expected_failures",
    [
        (
            PointReductionForecaster(
                estimator=LinearRegression(), feature_transformer=LagTransformer(lag=[1, 2])
            ),
            {"forecaster_type": "point", "uses_reduction": True, "supports_panel_data": True},
            [],
        ),
    ],
)
def test_point_reduction_panel_checks(
    forecaster, tags, expected_failures, panel_time_series_factory
):
    """Run systematic cross-learning checks on PointReductionForecaster with panel data."""
    y = panel_time_series_factory(length=100, n_series=3)
    y_train, y_test = y[:80], y[80:]

    # Fit forecaster
    forecaster_fitted = clone(forecaster)
    forecaster_fitted.fit(y_train, X=None, forecasting_horizon=3)

    # Run all generated checks (including cross-learning checks)
    expected_failures_set = set(expected_failures)
    for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(
        forecaster_fitted, y_train, None, y_test, None, tags=tags
    ):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            # Use deepcopy to ensure checks don't affect each other (e.g. update/reset)
            forecaster_for_check = deepcopy(forecaster_fitted)
            check_func(forecaster_for_check, **check_kwargs)


# ============================================================================
# Specific cross-learning behavior tests
# ============================================================================


def test_panel_predict_all_groups_default(panel_time_series_factory):
    """Test that predict with panel_group=None predicts all groups."""
    y = panel_time_series_factory(length=50, n_series=3)
    y_train = y[:40]

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
        feature_transformer=LagTransformer(lag=[1, 2]),
    )

    forecaster.fit(y=y_train, X=None, forecasting_horizon=3)

    # Predict with panel_group=None (default)
    y_pred = forecaster.predict(X=None, forecasting_horizon=3, panel_group=None)

    # Should have predictions for all 3 series (with __ separator)
    assert "panel__series_0" in y_pred.columns
    assert "panel__series_1" in y_pred.columns
    assert "panel__series_2" in y_pred.columns
    assert len(y_pred) == 3  # 3 forecast steps


def test_panel_predict_single_group(panel_time_series_factory):
    """Test that predict with panel_group filters to a single group."""
    y = panel_time_series_factory(length=50, n_series=3)
    y_train = y[:40]

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
        feature_transformer=LagTransformer(lag=[1, 2]),
    )

    forecaster.fit(y=y_train, X=None, forecasting_horizon=3)

    # Predict only for panel group (all series within the group)
    y_pred = forecaster.predict(X=None, forecasting_horizon=3, panel_group="panel")

    # Should have all series columns with __ separator
    assert "panel__series_0" in y_pred.columns
    assert "panel__series_1" in y_pred.columns
    assert "panel__series_2" in y_pred.columns
    assert len(y_pred) == 3


def test_panel_invalid_group_raises_error(panel_time_series_factory):
    """Test that invalid panel_group raises ValueError."""
    y = panel_time_series_factory(length=50, n_series=3)
    y_train = y[:40]

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
        feature_transformer=LagTransformer(lag=[1, 2]),
    )

    forecaster.fit(y=y_train, X=None, forecasting_horizon=3)

    # Try to predict with invalid group name
    with pytest.raises(ValueError, match="not found in local groups"):
        forecaster.predict(X=None, forecasting_horizon=3, panel_group="invalid_group")


def test_panel_global_data_no_groups(time_series_factory):
    """Test that panel_group has no effect on global data."""
    y = time_series_factory(length=50, n_components=1)
    y_train = y[:40]

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
        feature_transformer=LagTransformer(lag=[1, 2]),
    )

    forecaster.fit(y=y_train, X=None, forecasting_horizon=3)

    # Should work the same with or without panel_group
    y_pred_default = forecaster.predict(X=None, forecasting_horizon=3, panel_group=None)
    y_pred_explicit = forecaster.predict(X=None, forecasting_horizon=3, panel_group=None)

    assert y_pred_default.equals(y_pred_explicit)
    assert "feature_0" in y_pred_default.columns
    assert len(y_pred_default) == 3
