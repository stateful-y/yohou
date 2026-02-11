"""Reusable assertion helpers for yohou tests."""

import polars as pl
import polars.selectors as cs

from yohou.utils.validation import check_interval_consistency


def assert_forecaster_output_valid(
    y_pred: pl.DataFrame,
    expected_length: int,
    expected_columns: list[str] | None = None,
    check_time_column: bool = True,
) -> None:
    """Assert that forecaster output has correct structure.

    Parameters
    ----------
    y_pred : pl.DataFrame
        Forecaster prediction output
    expected_length : int
        Expected number of rows
    expected_columns : list[str] | None
        Expected column names (excluding time columns)
    check_time_column : bool, default=True
        Whether to verify time column presence

    """
    assert isinstance(y_pred, pl.DataFrame)
    assert len(y_pred) == expected_length

    if check_time_column:
        assert "time" in y_pred.columns
        assert "observed_time" in y_pred.columns

    if expected_columns is not None:
        actual_cols = y_pred.select(~cs.by_name("time", "observed_time")).columns
        assert set(actual_cols) == set(expected_columns)


def assert_continuous_time(
    df: pl.DataFrame,
    expected_interval: str | None = None,
) -> None:
    """Assert that time series has continuous time intervals.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with "time" column
    expected_interval : str | None
        Expected interval as string (e.g., "1d", "1h", "1mo")

    """
    if len(df) < 2:
        return  # Single row, can't check continuity

    actual_interval = check_interval_consistency(df)

    if expected_interval is not None:
        assert actual_interval == expected_interval, f"Expected interval {expected_interval}, got {actual_interval}"


def assert_transformer_output_valid(
    X_transformed: pl.DataFrame,
    X_original: pl.DataFrame,
    expected_length: int | None = None,
    check_time_preserved: bool = True,
) -> None:
    """Assert that transformer output has correct structure.

    Parameters
    ----------
    X_transformed : pl.DataFrame
        Transformer output
    X_original : pl.DataFrame
        Original input to transformer
    expected_length : int | None
        Expected number of rows (None to skip check)
    check_time_preserved : bool, default=True
        Whether to verify time column is preserved

    """
    assert isinstance(X_transformed, pl.DataFrame)

    if expected_length is not None:
        assert len(X_transformed) == expected_length

    if check_time_preserved:
        assert "time" in X_transformed.columns
        # Time values should match original (possibly subset)
        original_times = set(X_original["time"])
        transformed_times = set(X_transformed["time"])
        assert transformed_times.issubset(original_times)
