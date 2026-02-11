"""Unit tests for plotting utilities."""

import polars as pl
import pytest

from yohou.plotting.prep import (
    get_numeric_columns,
    resolve_columns,
    validate_dataframe,
    validate_panel_group,
)


def test_validate_dataframe_success():
    """Test DataFrame validation with valid input."""
    df = pl.DataFrame({"time": [1, 2, 3], "y": [10, 20, 30]})
    validate_dataframe(df)  # Should not raise


def test_validate_dataframe_not_polars():
    """Test DataFrame validation rejects non-Polars DataFrames."""
    with pytest.raises(TypeError, match="Expected pl.DataFrame"):
        validate_dataframe({"time": [1, 2, 3]})


def test_validate_dataframe_empty():
    """Test DataFrame validation rejects empty DataFrames."""
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_dataframe(pl.DataFrame())


def test_validate_dataframe_no_time():
    """Test DataFrame validation requires 'time' column."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
    with pytest.raises(ValueError, match="must have a 'time' column"):
        validate_dataframe(df)


def test_get_numeric_columns():
    """Test numeric column extraction."""
    df = pl.DataFrame({"time": [1, 2, 3], "y": [10.0, 20.0, 30.0], "category": ["A", "B", "C"]})
    result = get_numeric_columns(df)
    assert "y" in result
    assert "time" in result
    assert "category" not in result


def test_get_numeric_columns_with_exclude():
    """Test numeric column extraction with exclusions."""
    df = pl.DataFrame({"time": [1, 2, 3], "y": [10.0, 20.0, 30.0], "z": [5.0, 15.0, 25.0]})
    result = get_numeric_columns(df, exclude=["time"])
    assert result == ["y", "z"]


def test_resolve_columns_none():
    """Test column resolution when None (all numeric)."""
    df = pl.DataFrame({"time": [1, 2], "y": [10, 20], "z": [5, 15], "cat": ["A", "B"]})
    result = resolve_columns(df, columns=None, exclude=["time"])
    assert set(result) == {"y", "z"}


def test_resolve_columns_string():
    """Test column resolution with string input."""
    df = pl.DataFrame({"time": [1, 2], "y": [10, 20], "z": [5, 15]})
    result = resolve_columns(df, columns="y")
    assert result == ["y"]


def test_resolve_columns_list():
    """Test column resolution with list input."""
    df = pl.DataFrame({"time": [1, 2], "y": [10, 20], "z": [5, 15]})
    result = resolve_columns(df, columns=["y", "z"])
    assert result == ["y", "z"]


def test_resolve_columns_missing():
    """Test column resolution rejects missing columns."""
    df = pl.DataFrame({"time": [1, 2], "y": [10, 20]})
    with pytest.raises(ValueError, match="Columns not found"):
        resolve_columns(df, columns=["y", "missing"])


def test_validate_panel_group_success():
    """Test panel group validation with valid column."""
    df = pl.DataFrame({"time": [1, 2], "unique_id": ["A", "B"], "y": [10, 20]})
    validate_panel_group(df, "unique_id")  # Should not raise


def test_validate_panel_group_missing():
    """Test panel group validation rejects missing column."""
    df = pl.DataFrame({"time": [1, 2], "y": [10, 20]})
    with pytest.raises(ValueError, match="not found in DataFrame"):
        validate_panel_group(df, "unique_id")
