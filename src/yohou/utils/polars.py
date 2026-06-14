"""Polars utilities."""

import polars as pl

__all__ = ["cast", "get_categorical_columns", "get_numeric_columns", "is_categorical_dtype"]


def cast(
    df: pl.DataFrame,
    schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    """Cast columns according to schema with integer rounding.

    Casts columns to specified dtypes, with special handling for integer types.
    Integer target dtypes trigger rounding before casting to avoid truncation.
    Columns not present in the schema are preserved as-is.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to cast.
    schema : dict of str to pl.DataType
        Mapping from column names to target dtypes. Only columns present
        in this dict will be cast.

    Returns
    -------
    pl.DataFrame
        DataFrame with specified columns cast to target dtypes.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"a": [1.7, 2.3, 3.9], "b": [10.0, 20.0, 30.0], "c": ["x", "y", "z"]})
    >>> schema = {"a": pl.Int32, "b": pl.Float32}
    >>> result = cast(df, schema)
    >>> dict(result.schema)
    {'a': Int32, 'b': Float32, 'c': String}
    >>> result["a"].to_list()
    [2, 2, 4]

    Notes
    -----
    Integer casting behavior:
    - Values are rounded to nearest integer before casting
    - Prevents data loss from truncation (1.9 → 2, not 1)
    - Follows standard statistical rounding rules

    Columns in df but not in schema are left unchanged, allowing
    preservation of extra columns from model predictions.

    See Also
    --------
    - [`get_numeric_columns`][yohou.utils.polars.get_numeric_columns] : List numeric column names from a DataFrame.

    """
    exprs = []

    for col, target_dtype in schema.items():
        if col not in df.columns:
            continue

        if target_dtype.is_integer():
            exprs.append(pl.col(col).round().cast(target_dtype).alias(col))
        else:
            # Floats and non-numeric types are cast directly without rounding.
            exprs.append(pl.col(col).cast(target_dtype).alias(col))

    return df.with_columns(exprs)


def get_numeric_columns(df: pl.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """Get list of numeric column names from a DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    exclude : list[str] | None, default=None
        Column names to exclude from the result.

    Returns
    -------
    list[str]
        List of numeric column names.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.utils.polars import get_numeric_columns
    >>> df = pl.DataFrame({"time": [1, 2, 3], "y": [10.0, 20.0, 30.0], "cat": ["A", "B", "C"]})
    >>> get_numeric_columns(df)
    ['time', 'y']

    >>> get_numeric_columns(df, exclude=["time"])
    ['y']

    See Also
    --------
    - [`cast`][yohou.utils.polars.cast] : Cast DataFrame columns according to a schema.

    """
    exclude = exclude or []
    return [col for col in df.columns if df.schema[col].is_numeric() and col not in exclude]


def is_categorical_dtype(dtype: pl.DataType) -> bool:
    """Check whether a polars dtype represents categorical data.

    Parameters
    ----------
    dtype : pl.DataType
        The polars data type to check.

    Returns
    -------
    bool
        ``True`` if the dtype is ``pl.String``, ``pl.Categorical``, or
        ``pl.Enum``.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.utils.polars import is_categorical_dtype
    >>> is_categorical_dtype(pl.Series("x", ["a"]).dtype)
    True
    >>> is_categorical_dtype(pl.Series("x", [1.0]).dtype)
    False

    """
    return isinstance(dtype, pl.String | pl.Categorical | pl.Enum)


def get_categorical_columns(df: pl.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """Get list of categorical column names from a DataFrame.

    Categorical columns are those with ``pl.String``, ``pl.Categorical``,
    or ``pl.Enum`` dtype.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    exclude : list[str] | None, default=None
        Column names to exclude from the result.

    Returns
    -------
    list[str]
        List of categorical column names.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.utils.polars import get_categorical_columns
    >>> df = pl.DataFrame({"time": [1, 2, 3], "y": [10.0, 20.0, 30.0], "cat": ["A", "B", "C"]})
    >>> get_categorical_columns(df)
    ['cat']

    >>> get_categorical_columns(df, exclude=["cat"])
    []

    See Also
    --------
    - [`get_numeric_columns`][yohou.utils.polars.get_numeric_columns] : List numeric column names from a DataFrame.
    - [`is_categorical_dtype`][yohou.utils.polars.is_categorical_dtype] : Check if a dtype is categorical.

    """
    exclude = exclude or []
    return [col for col in df.columns if is_categorical_dtype(df[col].dtype) and col not in exclude]
