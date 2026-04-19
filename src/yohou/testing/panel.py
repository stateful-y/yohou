"""Check functions for panel data support in forecasters.

This module provides validation functions for cross-learning and panel data
handling in forecasters.
"""

import polars as pl

from yohou.utils import inspect_panel

__all__ = ["check_panel_data", "check_panel_invalid_group_raises", "check_panel_single_group"]


def _call_predict(forecaster, X, forecasting_horizon, panel_group=None, groups=None):
    """Call appropriate predict method based on forecaster type."""
    # Interval forecasters use predict_interval, point forecasters use predict
    if hasattr(forecaster, "predict"):
        if panel_group is not None:
            return forecaster.predict(X=X, forecasting_horizon=forecasting_horizon, panel_group=panel_group)
        elif groups is not None:
            return forecaster.predict(X=X, forecasting_horizon=forecasting_horizon, groups=groups)
        else:
            return forecaster.predict(X=X, forecasting_horizon=forecasting_horizon)
    # Interval forecaster
    elif panel_group is not None:
        return forecaster.predict_interval(X=X, forecasting_horizon=forecasting_horizon, panel_group=panel_group)
    elif groups is not None:
        return forecaster.predict_interval(X=X, forecasting_horizon=forecasting_horizon, groups=groups)
    else:
        return forecaster.predict_interval(X=X, forecasting_horizon=forecasting_horizon)


def _column_present(field: str, columns: list[str]) -> bool:
    """Check if field is present in columns (exact match or prefix for interval bounds)."""
    # Exact match (point forecasters)
    if field in columns:
        return True
    # Interval forecasters: check for field_lower_rate or field_upper_rate patterns
    return any(col.startswith(f"{field}_lower_") or col.startswith(f"{field}_upper_") for col in columns)


def check_panel_data(forecaster, y_panel: pl.DataFrame, X_panel: pl.DataFrame | None = None) -> None:
    """Check cross-learning with panel data predicts all groups by default.

    Validates that when panel_group=None (default), predictions are
    generated for all groups in the panel data columns.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster with panel data
    y_panel : pl.DataFrame
        Panel data with panel columns for testing
    X_panel : pl.DataFrame, optional
        Panel features

    Raises
    ------
    AssertionError
        If default prediction doesn't include all groups

    """
    # Predict with default (panel_group=None)
    y_pred = _call_predict(forecaster, X=X_panel, forecasting_horizon=3, panel_group=None)

    # Check that all local groups from training data are in predictions
    _, y_panel_groups = inspect_panel(y_panel)

    if len(y_panel_groups) > 0:
        # Should have predictions for all group columns (with __ separator)
        for _group_prefix, expected_fields in y_panel_groups.items():
            for field in expected_fields:
                assert _column_present(field, y_pred.columns), (
                    f"Column '{field}' (or interval bounds) missing from predictions. "
                    f"panel_group=None should predict all groups. Got columns: {y_pred.columns}"
                )


def check_panel_single_group(forecaster, y_panel: pl.DataFrame, X_panel: pl.DataFrame | None = None) -> None:
    """Check cross-learning filters to specified panel group.

    Validates that when panel_group is specified, predictions are
    generated only for that panel group (all columns with that prefix).

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster with panel data
    y_panel : pl.DataFrame
        Panel data with panel columns for testing
    X_panel : pl.DataFrame, optional
        Panel features

    Raises
    ------
    AssertionError
        If filtered prediction doesn't match specified group

    """
    _, y_panel_groups = inspect_panel(y_panel)

    if len(y_panel_groups) > 0:
        # Get first group prefix
        first_group = list(y_panel_groups.keys())[0]

        # Predict with specific group
        y_pred = _call_predict(forecaster, X=X_panel, forecasting_horizon=3, panel_group=first_group)

        # Should have columns from the specified group (flat columns with __ separator)
        group_cols = y_panel_groups[first_group]
        assert len(group_cols) > 0, f"Group '{first_group}' should have columns"
        for col in group_cols:
            assert _column_present(col, y_pred.columns), (
                f"Column '{col}' (or interval bounds) from group '{first_group}' "
                f"should be in predictions. Got: {y_pred.columns}"
            )


def check_panel_invalid_group_raises(forecaster, y_panel: pl.DataFrame, X_panel: pl.DataFrame | None = None) -> None:
    """Check that invalid panel_group raises ValueError.

    Validates error handling when panel_group specifies a panel group
    that doesn't exist in the training data.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster with panel data
    y_panel : pl.DataFrame
        Panel data with panel columns for testing
    X_panel : pl.DataFrame, optional
        Panel features

    Raises
    ------
    AssertionError
        If ValueError is not raised for invalid group

    """
    _, y_panel_groups = inspect_panel(y_panel)

    if len(y_panel_groups) > 0:
        # Try to predict with invalid group name
        try:
            _call_predict(forecaster, X=X_panel, forecasting_horizon=3, groups=["invalid_group"])
            raise AssertionError("predict() should raise ValueError for invalid panel_group, but didn't")
        except ValueError as e:
            # Expected - check error message mentions the invalid group
            assert "invalid_group" in str(e) or "not found" in str(e).lower(), (
                f"ValueError message should mention invalid group, got: {e}"
            )
