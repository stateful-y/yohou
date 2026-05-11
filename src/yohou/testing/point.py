"""Check functions for yohou point forecasters.

This module provides validation functions specific to point forecasters
(BasePointForecaster implementations).
"""

__all__ = ["check_point_prediction_structure", "check_point_prediction_types"]

try:
    import polars as pl
except ImportError as e:
    raise ImportError("polars.testing is required for yohou.testing module. Install with: uv sync --group tests") from e


def check_point_prediction_structure(
    forecaster, y_test: pl.DataFrame, X_actual_test: pl.DataFrame | None = None
) -> None:
    """Check point predictions have correct column structure.

    Parameters
    ----------
    forecaster : BasePointForecaster
        Fitted point forecaster instance
    y_test : pl.DataFrame
        Test target data
    X_actual_test : pl.DataFrame, optional
        Test features

    Raises
    ------
    AssertionError
        If prediction structure is incorrect

    """
    forecasting_horizon = min(3, len(y_test))
    y_pred = forecaster.predict(forecasting_horizon=forecasting_horizon)

    # Should have vintage_time, time, and target columns
    assert "vintage_time" in y_pred.columns, "Point predictions must have 'vintage_time'"
    assert "time" in y_pred.columns, "Point predictions must have 'time'"

    # Should NOT have interval columns
    interval_cols = [col for col in y_pred.columns if "_lower_" in col or "_upper_" in col]
    assert len(interval_cols) == 0, f"Point predictions should not have interval columns, found: {interval_cols}"

    # Should have target columns
    target_cols = [col for col in y_pred.columns if col not in ["vintage_time", "time"]]
    assert len(target_cols) > 0, "Point predictions must have at least one target column"


def check_point_prediction_types(forecaster) -> None:
    """Check point forecaster has 'point' in forecaster_type tag.

    Parameters
    ----------
    forecaster : BasePointForecaster
        Point forecaster instance

    Raises
    ------
    AssertionError
        If forecaster_type is not 'point'

    """
    tags = forecaster.__sklearn_tags__()
    forecaster_type = tags.forecaster_tags.forecaster_type if tags.forecaster_tags else None

    assert forecaster_type is not None and "point" in forecaster_type, (
        f"Point forecaster should have 'point' in forecaster_type tag, got {forecaster_type}"
    )
