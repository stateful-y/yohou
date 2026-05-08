"""Check functions for yohou interval forecasters.

This module provides validation functions specific to interval forecasters
(BaseIntervalForecaster implementations).
"""

import polars as pl
from sklearn.base import clone

from yohou.utils import inspect_panel

__all__ = [
    "check_coverage_rates_parameter",
    "check_coverage_rates_validation",
    "check_interval_bounds",
    "check_interval_prediction_columns",
    "check_interval_prediction_types",
]


def check_interval_prediction_columns(
    forecaster, y_test: pl.DataFrame, X_actual_test: pl.DataFrame | None = None
) -> None:
    """Check interval predictions have {col}_lower_{rate} and {col}_upper_{rate} format.

    Parameters
    ----------
    forecaster : BaseIntervalForecaster
        Fitted interval forecaster instance
    y_test : pl.DataFrame
        Test target data
    X_actual_test : pl.DataFrame, optional
        Test features

    Raises
    ------
    AssertionError
        If interval column naming is incorrect

    """
    forecasting_horizon = min(3, len(y_test))

    # Call predict_interval for interval forecasters
    y_pred = forecaster.predict_interval(forecasting_horizon=forecasting_horizon)

    # Get coverage rates - use fit_coverage_rates_ (set during fit)
    coverage_rates = forecaster.fit_coverage_rates_

    # Check if we have panel data (columns with __ separator)
    _, y_panel_groups = inspect_panel(y_test)

    if len(y_panel_groups) > 0:
        # For panel data, interval columns use __ separator
        # e.g., "stores__store_0_lower_0.1"
        for group_prefix in y_panel_groups:
            # Get fields from the original training data (full column names)
            expected_fields = y_panel_groups[group_prefix]

            for rate in coverage_rates:
                for field in expected_fields:
                    lower_col = f"{field}_lower_{rate}"
                    upper_col = f"{field}_upper_{rate}"

                    assert lower_col in y_pred.columns, f"Missing lower bound column: {lower_col}"
                    assert upper_col in y_pred.columns, f"Missing upper bound column: {upper_col}"
    else:
        # For global data, check individual column pattern: {col}_lower_{rate}
        target_cols = list(forecaster.local_y_schema_.keys())

        # Check each coverage rate has lower and upper bounds for each target
        for rate in coverage_rates:
            for col in target_cols:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"

                assert lower_col in y_pred.columns, f"Missing lower bound column: {lower_col}"
                assert upper_col in y_pred.columns, f"Missing upper bound column: {upper_col}"


def check_interval_bounds(forecaster, y_test: pl.DataFrame, X_actual_test: pl.DataFrame | None = None) -> None:
    """Check upper >= lower for all coverage rates and time steps.

    Parameters
    ----------
    forecaster : BaseIntervalForecaster
        Fitted interval forecaster instance
    y_test : pl.DataFrame
        Test target data
    X_actual_test : pl.DataFrame, optional
        Test features

    Raises
    ------
    AssertionError
        If upper bounds are less than lower bounds

    """
    forecasting_horizon = min(3, len(y_test))
    y_pred = forecaster.predict_interval(forecasting_horizon=forecasting_horizon)

    coverage_rates = forecaster.fit_coverage_rates_

    # Check if we have panel data (columns with __ separator)
    _, y_panel_groups = inspect_panel(y_test)

    if len(y_panel_groups) > 0:
        # For panel data, interval columns use __ separator
        for group_prefix in y_panel_groups:
            # Get fields from the original training data (full column names)
            expected_fields = y_panel_groups[group_prefix]

            for rate in coverage_rates:
                for field in expected_fields:
                    lower_col = f"{field}_lower_{rate}"
                    upper_col = f"{field}_upper_{rate}"

                    lower_vals = y_pred[lower_col].to_numpy()
                    upper_vals = y_pred[upper_col].to_numpy()

                    violations = lower_vals > upper_vals
                    if violations.any():
                        raise AssertionError(
                            f"Found {violations.sum()} violations where lower > upper for {field} at coverage {rate}"
                        )
    else:
        # For global data, check individual columns
        target_cols = list(forecaster.local_y_schema_.keys())

        for rate in coverage_rates:
            for col in target_cols:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"

                lower_vals = y_pred[lower_col].to_numpy()
                upper_vals = y_pred[upper_col].to_numpy()

                violations = lower_vals > upper_vals
                if violations.any():
                    raise AssertionError(
                        f"Found {violations.sum()} violations where lower > upper for {col} at coverage {rate}"
                    )


def check_interval_prediction_types(forecaster) -> None:
    """Check interval forecaster has 'interval' in forecaster_type tag.

    Parameters
    ----------
    forecaster : BaseIntervalForecaster
        Interval forecaster instance

    Raises
    ------
    AssertionError
        If forecaster_type doesn't indicate interval support

    """
    tags = forecaster.__sklearn_tags__()
    forecaster_type = tags.forecaster_tags.forecaster_type if tags.forecaster_tags else None

    assert forecaster_type is not None and "interval" in forecaster_type, (
        f"Interval forecaster should have 'interval' in forecaster_type tag, got {forecaster_type}"
    )


def check_coverage_rates_parameter(forecaster) -> None:
    """Check coverage_rates is list of floats in [0, 1].

    Parameters
    ----------
    forecaster : BaseIntervalForecaster
        Interval forecaster instance

    Raises
    ------
    AssertionError
        If coverage_rates is invalid

    """
    coverage_rates = forecaster.fit_coverage_rates_

    assert isinstance(coverage_rates, list), f"coverage_rates should be list, got {type(coverage_rates)}"

    assert len(coverage_rates) > 0, "coverage_rates should not be empty"

    for rate in coverage_rates:
        assert isinstance(rate, int | float), f"Each coverage rate should be numeric, got {type(rate)} for {rate}"
        assert 0 <= rate <= 1, f"Coverage rates should be in [0, 1], got {rate}"


def check_coverage_rates_validation(
    forecaster,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> None:
    """Check invalid coverage_rates raise ValueError during fit and predict.

    Parameters
    ----------
    forecaster : BaseIntervalForecaster
        Unfitted interval forecaster instance
    y : pl.DataFrame
        Training target data
    X_actual : pl.DataFrame, optional
        Training features

    Raises
    ------
    AssertionError
        If invalid coverage_rates don't raise ValueError

    """
    # Test rate = 1.5 (above 1 - invalid)
    forecaster_clone = clone(forecaster)
    try:
        forecaster_clone.fit(
            y, X_actual, forecasting_horizon=3, coverage_rates=[1.5], X_future=X_future, X_forecast=X_forecast
        )
        raise AssertionError(f"{forecaster_clone.__class__.__name__} should raise ValueError for coverage_rates=[1.5]")
    except ValueError as e:
        assert "coverage" in str(e).lower() or "1" in str(e).lower(), (
            f"ValueError should mention coverage_rates, got: {e}"
        )

    # Test negative rate (invalid)
    forecaster_clone = clone(forecaster)
    try:
        forecaster_clone.fit(
            y, X_actual, forecasting_horizon=3, coverage_rates=[-0.5], X_future=X_future, X_forecast=X_forecast
        )
        raise AssertionError(f"{forecaster_clone.__class__.__name__} should raise ValueError for coverage_rates=[-0.5]")
    except ValueError as e:
        assert "coverage" in str(e).lower() or "negative" in str(e).lower(), (
            f"ValueError should mention coverage_rates, got: {e}"
        )

    # Test predict_interval also validates
    forecaster_clone = clone(forecaster)
    forecaster_clone.fit(
        y, X_actual, forecasting_horizon=3, coverage_rates=[0.95], X_future=X_future, X_forecast=X_forecast
    )

    try:
        forecaster_clone.predict_interval(forecasting_horizon=3, coverage_rates=[1.5])
        raise AssertionError(
            f"{forecaster_clone.__class__.__name__}.predict_interval() should raise ValueError for coverage_rates=[1.5]"
        )
    except ValueError as e:
        assert "coverage" in str(e).lower(), f"ValueError should mention coverage_rates, got: {e}"
