"""Polars utilities."""

import polars as pl


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
    >>> df = pl.DataFrame({
    ...     "a": [1.7, 2.3, 3.9],
    ...     "b": [10.0, 20.0, 30.0],
    ...     "c": ["x", "y", "z"]
    ... })
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
