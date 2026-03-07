"""Polars utilities."""

import polars as pl

__all__ = ["cast", "get_numeric_columns"]


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
    `get_numeric_columns` : List numeric column names from a DataFrame.

    """
    exprs = []

    for col, target_dtype in schema.items():
        if col not in df.columns:
            continue

        if target_dtype.is_integer():
            exprs.append(pl.col(col).round().cast(target_dtype).alias(col))
        else:
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
    `cast` : Cast DataFrame columns according to a schema.

    """
    exclude = exclude or []
    numeric_types = [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ]
    return [col for col in df.columns if any(df[col].dtype == dtype for dtype in numeric_types) and col not in exclude]
