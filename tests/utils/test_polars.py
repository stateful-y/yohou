"""Tests for polars utility functions."""

from datetime import datetime

import polars as pl
import pytest

from yohou.utils.polars import cast


def test_cast_float_to_int32_with_rounding():
    """Test casting float to Int32 with proper rounding."""
    df = pl.DataFrame({"a": [1.2, 2.5, 3.7, 4.9]})
    schema = {"a": pl.Int32}
    result = cast(df, schema)

    assert result.schema["a"] == pl.Int32
    assert result["a"].to_list() == [1, 2, 4, 5]  # Rounds: 1.2→1, 2.5→2, 3.7→4, 4.9→5


def test_cast_float_to_int8():
    """Test casting float to Int8 (small integer type)."""
    df = pl.DataFrame({"value": [10.3, 20.7, 30.4]})
    schema = {"value": pl.Int8}
    result = cast(df, schema)

    assert result.schema["value"] == pl.Int8
    assert result["value"].to_list() == [10, 21, 30]


def test_cast_float_to_int16():
    """Test casting float to Int16."""
    df = pl.DataFrame({"value": [100.5, 200.9, 300.1]})
    schema = {"value": pl.Int16}
    result = cast(df, schema)

    assert result.schema["value"] == pl.Int16
    assert result["value"].to_list() == [100, 201, 300]


def test_cast_float_to_int64():
    """Test casting float to Int64."""
    df = pl.DataFrame({"value": [1000.5, 2000.9, 3000.1]})
    schema = {"value": pl.Int64}
    result = cast(df, schema)

    assert result.schema["value"] == pl.Int64
    assert result["value"].to_list() == [1000, 2001, 3000]


def test_cast_float64_to_float32():
    """Test casting Float64 to Float32 (no rounding)."""
    df = pl.DataFrame({"b": [1.123456789, 2.987654321]})
    schema = {"b": pl.Float32}
    result = cast(df, schema)

    assert result.schema["b"] == pl.Float32
    # Float32 has less precision, but values should be close
    assert all(abs(result["b"][i] - df["b"][i]) < 0.001 for i in range(len(df)))


def test_cast_multiple_columns():
    """Test casting multiple columns with different target types."""
    df = pl.DataFrame({
        "sales": [10.7, 20.3, 30.9],
        "revenue": [100.5, 200.5, 300.5],
        "name": ["A", "B", "C"],
    })
    schema = {"sales": pl.Int32, "revenue": pl.Float32}
    result = cast(df, schema)

    assert result.schema["sales"] == pl.Int32
    assert result.schema["revenue"] == pl.Float32
    assert result.schema["name"] == pl.String  # Unchanged
    assert result["sales"].to_list() == [11, 20, 31]
    assert result["name"].to_list() == ["A", "B", "C"]


def test_cast_preserves_columns_not_in_schema():
    """Test that columns not in schema are preserved unchanged."""
    df = pl.DataFrame({
        "a": [1.0, 2.0],
        "b": [10, 20],
        "c": ["x", "y"],
        "d": [True, False],
    })
    schema = {"a": pl.Int32}
    result = cast(df, schema)

    assert result.schema["a"] == pl.Int32
    assert result.schema["b"] == pl.Int64  # Unchanged
    assert result.schema["c"] == pl.String  # Unchanged
    assert result.schema["d"] == pl.Boolean  # Unchanged
    assert result["a"].to_list() == [1, 2]
    assert result["b"].to_list() == [10, 20]


def test_cast_ignores_missing_columns():
    """Test that schema columns not in df are silently ignored."""
    df = pl.DataFrame({"a": [1.0, 2.0]})
    schema = {"a": pl.Int32, "b": pl.Float32, "c": pl.String}
    result = cast(df, schema)

    assert result.schema == {"a": pl.Int32}
    assert result["a"].to_list() == [1, 2]


def test_cast_empty_schema():
    """Test casting with empty schema returns unchanged df."""
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [10, 20]})
    schema = {}
    result = cast(df, schema)

    assert result.schema == df.schema
    assert result.equals(df)


def test_cast_with_datetime_column():
    """Test casting preserves datetime columns not in schema."""
    df = pl.DataFrame({
        "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        "value": [10.5, 20.7],
    })
    schema = {"value": pl.Int32}
    result = cast(df, schema)

    assert result.schema["time"] == pl.Datetime
    assert result.schema["value"] == pl.Int32
    assert result["value"].to_list() == [10, 21]


def test_cast_negative_values():
    """Test rounding behavior with negative values."""
    df = pl.DataFrame({"a": [-1.2, -2.5, -3.7, -4.9]})
    schema = {"a": pl.Int32}
    result = cast(df, schema)

    assert result.schema["a"] == pl.Int32
    # Polars round() uses "half away from zero": -2.5→-2 (not -3)
    assert result["a"].to_list() == [-1, -2, -4, -5]


def test_cast_with_nulls():
    """Test casting preserves null values."""
    df = pl.DataFrame({"a": [1.5, None, 3.7, None]})
    schema = {"a": pl.Int32}
    result = cast(df, schema)

    assert result.schema["a"] == pl.Int32
    assert result["a"].to_list() == [2, None, 4, None]


def test_cast_int_to_int():
    """Test casting between integer types (with rounding, though no-op)."""
    df = pl.DataFrame({"a": pl.Series([10, 20, 30], dtype=pl.Int64)})
    schema = {"a": pl.Int32}
    result = cast(df, schema)

    assert result.schema["a"] == pl.Int32
    assert result["a"].to_list() == [10, 20, 30]


def test_cast_string_to_int_fails():
    """Test that invalid casts raise appropriate errors."""
    df = pl.DataFrame({"a": ["1", "2", "3"]})
    schema = {"a": pl.Int32}

    # This should fail because we can't cast string to int directly
    with pytest.raises(Exception):  # polars raises ComputeError or similar
        cast(df, schema)


def test_cast_preserves_column_order():
    """Test that column order is preserved."""
    df = pl.DataFrame({
        "z": [1.0, 2.0],
        "y": [10.0, 20.0],
        "x": [100.0, 200.0],
    })
    schema = {"x": pl.Int32, "y": pl.Int32, "z": pl.Int32}
    result = cast(df, schema)

    assert list(result.columns) == ["z", "y", "x"]
    assert result["z"].to_list() == [1, 2]
    assert result["y"].to_list() == [10, 20]
    assert result["x"].to_list() == [100, 200]


def test_cast_use_case_forecaster_dtype_preservation():
    """Test realistic use case: preserving dtypes after sklearn prediction."""
    # Simulate sklearn returning Float64, need to cast back to original Int32
    df = pl.DataFrame({
        "sales": [45.0, 46.0, 47.0],
        "revenue": [472.5, 483.0, 493.5],
    })
    schema = {"sales": pl.Int32, "revenue": pl.Float32}
    result = cast(df, schema)

    assert result.schema["sales"] == pl.Int32
    assert result.schema["revenue"] == pl.Float32
    assert result["sales"].to_list() == [45, 46, 47]


def test_cast_with_time_columns():
    """Test casting works alongside time columns (common in forecasting)."""
    df = pl.DataFrame({
        "time": [datetime(2020, 1, i) for i in range(1, 4)],
        "observed_time": [datetime(2020, 1, 1)] * 3,
        "value": [10.5, 20.7, 30.9],
    })
    schema = {"value": pl.Int16}
    result = cast(df, schema)

    assert result.schema["time"] == pl.Datetime
    assert result.schema["observed_time"] == pl.Datetime
    assert result.schema["value"] == pl.Int16
    assert result["value"].to_list() == [10, 21, 31]


def test_cast_rounding_behavior_edge_cases():
    """Test edge cases in rounding behavior."""
    df = pl.DataFrame({"a": [0.5, 1.5, 2.5, 3.5, 4.5]})
    schema = {"a": pl.Int32}
    result = cast(df, schema)

    assert result.schema["a"] == pl.Int32
    # Polars uses "round half to even" (banker's rounding):
    # Rounds to nearest even number when exactly halfway between two integers
    # 0.5→0 (even), 1.5→2 (even), 2.5→2 (even), 3.5→4 (even), 4.5→4 (even)
    assert result["a"].to_list() == [0, 2, 2, 4, 4]
