"""Check functions for panel data support in forecasters.

This module provides validation functions for cross-learning and panel data
handling in forecasters.
"""

try:
    import polars as pl
except ImportError as e:
    raise ImportError(
        "polars.testing is required for yohou.testing module. "
        "Install with: uv sync --group tests"
    ) from e


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
    from yohou.utils.panel import inspect_locality

    # Predict with default (panel_group=None)
    y_pred = forecaster.predict(X=X_panel, forecasting_horizon=3, panel_group=None)

    # Check that all local groups from training data are in predictions
    _, y_panel_groups = inspect_locality(y_panel)

    if len(y_panel_groups) > 0:
        # Should have predictions for all group columns (with __ separator)
        for group_prefix, expected_fields in y_panel_groups.items():
            for field in expected_fields:
                assert field in y_pred.columns, (
                    f"Column '{field}' missing from predictions. "
                    f"panel_group=None should predict all groups."
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
    from yohou.utils.panel import inspect_locality

    _, y_panel_groups = inspect_locality(y_panel)

    if len(y_panel_groups) > 0:
        # Get first group prefix
        first_group = list(y_panel_groups.keys())[0]

        # Predict with specific group
        y_pred = forecaster.predict(X=X_panel, forecasting_horizon=3, panel_group=first_group)

        # Should have columns from the specified group (flat columns with __ separator)
        group_cols = y_panel_groups[first_group]
        assert len(group_cols) > 0, f"Group '{first_group}' should have columns"
        for col in group_cols:
            assert col in y_pred.columns, (
                f"Column '{col}' from group '{first_group}' should be in predictions"
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
    from yohou.utils.panel import inspect_locality

    _, y_panel_groups = inspect_locality(y_panel)

    if len(y_panel_groups) > 0:
        # Try to predict with invalid group name
        try:
            forecaster.predict(
                X=X_panel, forecasting_horizon=3, panel_group_names=["invalid_group"]
            )
            raise AssertionError(
                "predict() should raise ValueError for invalid panel_group, but didn't"
            )
        except ValueError as e:
            # Expected - check error message mentions the invalid group
            assert "invalid_group" in str(e) or "not found" in str(e).lower(), (
                f"ValueError message should mention invalid group, got: {e}"
            )
