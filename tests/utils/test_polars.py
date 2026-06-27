"""Tests for polars utility functions."""

from datetime import datetime

import polars as pl
import pytest
from polars.exceptions import ComputeError, InvalidOperationError, SchemaError

from yohou.utils.polars import cast, get_categorical_columns, get_numeric_columns, is_categorical_dtype


class TestCast:
    """Tests for the cast utility function."""

    def test_cast_float_to_int32_with_rounding(self):
        """Test casting float to Int32 with proper rounding."""
        df = pl.DataFrame({"a": [1.2, 2.5, 3.7, 4.9]})
        schema = {"a": pl.Int32}
        result = cast(df, schema)

        assert result.schema["a"] == pl.Int32
        assert result["a"].to_list() == [1, 2, 4, 5]  # Rounds: 1.2→1, 2.5→2, 3.7→4, 4.9→5

    @pytest.mark.parametrize(
        ("target_dtype", "values", "expected"),
        [
            (pl.Int8, [10.3, 20.7, 30.4], [10, 21, 30]),
            (pl.Int16, [100.5, 200.9, 300.1], [100, 201, 300]),
            (pl.Int64, [1000.5, 2000.9, 3000.1], [1000, 2001, 3000]),
        ],
    )
    def test_cast_float_to_integer_widths(self, target_dtype, values, expected):
        """Test casting float to each integer width applies the same rounding."""
        df = pl.DataFrame({"value": values})
        result = cast(df, {"value": target_dtype})

        assert result.schema["value"] == target_dtype
        assert result["value"].to_list() == expected

    def test_cast_float64_to_float32(self):
        """Test casting Float64 to Float32 (no rounding)."""
        df = pl.DataFrame({"b": [1.123456789, 2.987654321]})
        schema = {"b": pl.Float32}
        result = cast(df, schema)

        assert result.schema["b"] == pl.Float32
        # Float32 has less precision, but values should be close
        assert all(abs(result["b"][i] - df["b"][i]) < 0.001 for i in range(len(df)))

    def test_cast_multiple_columns(self):
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

    def test_cast_preserves_columns_not_in_schema(self):
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

    def test_cast_ignores_missing_columns(self):
        """Test that schema columns not in df are silently ignored."""
        df = pl.DataFrame({"a": [1.0, 2.0]})
        schema = {"a": pl.Int32, "b": pl.Float32, "c": pl.String}
        result = cast(df, schema)

        assert result.schema == {"a": pl.Int32}
        assert result["a"].to_list() == [1, 2]

    def test_cast_empty_schema(self):
        """Test casting with empty schema returns unchanged df."""
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [10, 20]})
        schema = {}
        result = cast(df, schema)

        assert result.schema == df.schema
        assert result.equals(df)

    def test_cast_with_datetime_column(self):
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

    def test_cast_negative_values(self):
        """Test rounding behavior with negative values."""
        df = pl.DataFrame({"a": [-1.2, -2.5, -3.7, -4.9]})
        schema = {"a": pl.Int32}
        result = cast(df, schema)

        assert result.schema["a"] == pl.Int32
        # Polars uses banker's rounding (round-half-to-even): -2.5→-2 (even), not -3.
        assert result["a"].to_list() == [-1, -2, -4, -5]

    def test_cast_with_nulls(self):
        """Test casting preserves null values."""
        df = pl.DataFrame({"a": [1.5, None, 3.7, None]})
        schema = {"a": pl.Int32}
        result = cast(df, schema)

        assert result.schema["a"] == pl.Int32
        assert result["a"].to_list() == [2, None, 4, None]

    def test_cast_int_to_int(self):
        """Test casting between integer types (with rounding, though no-op)."""
        df = pl.DataFrame({"a": pl.Series([10, 20, 30], dtype=pl.Int64)})
        schema = {"a": pl.Int32}
        result = cast(df, schema)

        assert result.schema["a"] == pl.Int32
        assert result["a"].to_list() == [10, 20, 30]

    def test_cast_string_to_int_fails(self):
        """Test that invalid casts raise appropriate errors."""
        df = pl.DataFrame({"a": ["1", "2", "3"]})
        schema = {"a": pl.Int32}

        # This should fail because we can't cast string to int directly
        # InvalidOperationError was added in polars 1.0.0 for operations on incompatible types
        with pytest.raises((ComputeError, SchemaError, TypeError, InvalidOperationError)):
            cast(df, schema)

    def test_cast_preserves_column_order(self):
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

    def test_cast_use_case_forecaster_dtype_preservation(self):
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

    def test_cast_with_time_columns(self):
        """Test casting works alongside time columns (common in forecasting)."""
        df = pl.DataFrame({
            "time": [datetime(2020, 1, i) for i in range(1, 4)],
            "vintage_time": [datetime(2020, 1, 1)] * 3,
            "value": [10.5, 20.7, 30.9],
        })
        schema = {"value": pl.Int16}
        result = cast(df, schema)

        assert result.schema["time"] == pl.Datetime
        assert result.schema["vintage_time"] == pl.Datetime
        assert result.schema["value"] == pl.Int16
        assert result["value"].to_list() == [10, 21, 31]

    def test_cast_rounding_behavior_edge_cases(self):
        """Test edge cases in rounding behavior."""
        df = pl.DataFrame({"a": [0.5, 1.5, 2.5, 3.5, 4.5]})
        schema = {"a": pl.Int32}
        result = cast(df, schema)

        assert result.schema["a"] == pl.Int32
        # Polars uses "round half to even" (banker's rounding):
        # Rounds to nearest even number when exactly halfway between two integers
        # 0.5→0 (even), 1.5→2 (even), 2.5→2 (even), 3.5→4 (even), 4.5→4 (even)
        assert result["a"].to_list() == [0, 2, 2, 4, 4]

    def test_cast_float_to_string(self):
        """Test casting float column to non-numeric type (String)."""
        df = pl.DataFrame({"a": [1.0, 2.5, 3.0]})
        schema = {"a": pl.String}
        result = cast(df, schema)

        assert result.schema["a"] == pl.String
        assert result["a"].to_list() == ["1.0", "2.5", "3.0"]

    def test_cast_float_to_boolean(self):
        """Test casting float column to Boolean type."""
        df = pl.DataFrame({"a": [1.0, 0.0, 1.0]})
        schema = {"a": pl.Boolean}
        result = cast(df, schema)

        assert result.schema["a"] == pl.Boolean
        assert result["a"].to_list() == [True, False, True]


class TestIsCategoricalDtype:
    """Tests for the is_categorical_dtype utility function."""

    def test_string_is_categorical(self):
        """String dtype is categorical."""
        assert is_categorical_dtype(pl.Series("x", ["a"]).dtype) is True

    def test_categorical_is_categorical(self):
        """Categorical dtype is categorical."""
        assert is_categorical_dtype(pl.Series("x", ["a"]).cast(pl.Categorical).dtype) is True

    def test_enum_is_categorical(self):
        """Enum dtype is categorical."""
        assert is_categorical_dtype(pl.Series("x", ["a"]).cast(pl.Enum(["a", "b"])).dtype) is True

    def test_int64_not_categorical(self):
        """Int64 dtype is not categorical."""
        assert is_categorical_dtype(pl.Series("x", [1]).dtype) is False

    def test_float64_not_categorical(self):
        """Float64 dtype is not categorical."""
        assert is_categorical_dtype(pl.Series("x", [1.0]).dtype) is False

    def test_boolean_not_categorical(self):
        """Boolean dtype is not categorical."""
        assert is_categorical_dtype(pl.Series("x", [True]).dtype) is False

    def test_datetime_not_categorical(self):
        """Datetime dtype is not categorical."""
        from datetime import datetime

        assert is_categorical_dtype(pl.Series("x", [datetime(2020, 1, 1)]).dtype) is False

    def test_series_string_dtype(self):
        """String series dtype is detected as categorical."""
        s = pl.Series("x", ["a", "b", "c"])
        assert is_categorical_dtype(s.dtype) is True

    def test_series_float_dtype(self):
        """Float series dtype is not detected as categorical."""
        s = pl.Series("x", [1.0, 2.0, 3.0])
        assert is_categorical_dtype(s.dtype) is False


class TestGetNumericColumns:
    """Tests for the get_numeric_columns utility function."""

    def test_basic(self):
        """Returns numeric column names."""
        df = pl.DataFrame({"a": [1, 2], "b": [1.0, 2.0], "c": ["x", "y"]})
        assert get_numeric_columns(df) == ["a", "b"]

    def test_exclude(self):
        """Excludes specified columns."""
        df = pl.DataFrame({"time": [1, 2], "y": [10.0, 20.0], "cat": ["A", "B"]})
        assert get_numeric_columns(df, exclude=["time"]) == ["y"]

    def test_no_numeric_columns(self):
        """Returns empty list when no numeric columns exist."""
        df = pl.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        assert get_numeric_columns(df) == []

    def test_all_numeric(self):
        """Returns all columns when all are numeric."""
        df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": pl.Series([5, 6], dtype=pl.UInt32)})
        assert get_numeric_columns(df) == ["a", "b", "c"]

    def test_empty_dataframe(self):
        """Returns empty list for DataFrame with no rows but typed columns."""
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Float64), "b": pl.Series([], dtype=pl.String)})
        assert get_numeric_columns(df) == ["a"]

    def test_exclude_nonexistent_column(self):
        """Excluding a column not in the DataFrame is a no-op."""
        df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        assert get_numeric_columns(df, exclude=["z"]) == ["a", "b"]


class TestGetCategoricalColumns:
    """Tests for the get_categorical_columns utility function."""

    def test_basic(self):
        """Returns string column names."""
        df = pl.DataFrame({"a": [1, 2], "b": [1.0, 2.0], "c": ["x", "y"]})
        assert get_categorical_columns(df) == ["c"]

    def test_exclude(self):
        """Excludes specified columns."""
        df = pl.DataFrame({"cat1": ["a", "b"], "cat2": ["x", "y"], "num": [1, 2]})
        assert get_categorical_columns(df, exclude=["cat1"]) == ["cat2"]

    def test_no_categorical_columns(self):
        """Returns empty list when no categorical columns exist."""
        df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        assert get_categorical_columns(df) == []

    def test_all_categorical(self):
        """Returns all columns when all are categorical."""
        df = pl.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        assert get_categorical_columns(df) == ["a", "b"]

    def test_mixed_dtypes(self):
        """Handles mixed numeric, string, datetime columns."""
        df = pl.DataFrame({
            "num": [1, 2],
            "cat": ["a", "b"],
            "time": pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 2), "1d", eager=True),
            "flag": [True, False],
        })
        assert get_categorical_columns(df) == ["cat"]

    def test_categorical_dtype(self):
        """Detects pl.Categorical dtype columns."""
        df = pl.DataFrame({"a": pl.Series(["x", "y"]).cast(pl.Categorical), "b": [1, 2]})
        assert get_categorical_columns(df) == ["a"]

    def test_enum_dtype(self):
        """Detects pl.Enum dtype columns."""
        df = pl.DataFrame({"a": pl.Series(["x", "y"]).cast(pl.Enum(["x", "y", "z"])), "b": [1, 2]})
        assert get_categorical_columns(df) == ["a"]

    def test_empty_dataframe(self):
        """Returns correctly for DataFrame with no rows."""
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.String), "b": pl.Series([], dtype=pl.Float64)})
        assert get_categorical_columns(df) == ["a"]

    def test_exclude_nonexistent_column(self):
        """Excluding a column not in the DataFrame is a no-op."""
        df = pl.DataFrame({"cat": ["a", "b"], "num": [1, 2]})
        assert get_categorical_columns(df, exclude=["z"]) == ["cat"]
