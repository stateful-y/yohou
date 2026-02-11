"""Data preparation and validation utilities for plotting."""

import polars as pl


def validate_dataframe(df: pl.DataFrame) -> None:
    """
    Validate that input is a Polars DataFrame with required structure.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to validate.

    Raises
    ------
    TypeError
        If df is not a Polars DataFrame.
    ValueError
        If DataFrame is empty or missing 'time' column.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting.prep import validate_dataframe
    >>> df = pl.DataFrame({"time": [1, 2, 3], "y": [10, 20, 30]})
    >>> validate_dataframe(df)  # No error raised

    >>> try:
    ...     validate_dataframe(pl.DataFrame())
    ... except ValueError as e:
    ...     print("caught")
    caught
    """
    if not isinstance(df, pl.DataFrame):
        msg = f"Expected pl.DataFrame, got {type(df).__name__}"
        raise TypeError(msg)

    if df.is_empty():
        msg = "DataFrame is empty"
        raise ValueError(msg)

    if "time" not in df.columns:
        msg = "DataFrame must have a 'time' column"
        raise ValueError(msg)


def get_numeric_columns(df: pl.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """
    Get list of numeric column names from DataFrame.

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
    >>> from yohou.plotting.prep import get_numeric_columns
    >>> df = pl.DataFrame({"time": [1, 2, 3], "y": [10.0, 20.0, 30.0], "category": ["A", "B", "C"]})
    >>> get_numeric_columns(df)
    ['time', 'y']

    >>> get_numeric_columns(df, exclude=["time"])
    ['y']
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


def resolve_columns(
    df: pl.DataFrame,
    columns: str | list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    """
    Resolve column selection to list of column names.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    columns : str | list[str] | None, default=None
        Column specification. If None, selects all numeric columns.
        If str, returns single-item list. If list, validates and returns.
    exclude : list[str] | None, default=None
        Column names to exclude when columns=None.

    Returns
    -------
    list[str]
        List of resolved column names.

    Raises
    ------
    ValueError
        If specified columns don't exist in DataFrame.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting.prep import resolve_columns
    >>> df = pl.DataFrame({"time": [1, 2], "y": [10, 20], "z": [5, 15]})

    >>> resolve_columns(df, columns=None, exclude=["time"])
    ['y', 'z']

    >>> resolve_columns(df, columns="y")
    ['y']

    >>> resolve_columns(df, columns=["y", "z"])
    ['y', 'z']
    """
    if columns is None:
        return get_numeric_columns(df, exclude=exclude)

    if isinstance(columns, str):
        columns = [columns]

    # Validate columns exist
    missing = [col for col in columns if col not in df.columns]
    if missing:
        msg = f"Columns not found in DataFrame: {missing}"
        raise ValueError(msg)

    return columns


def validate_panel_group(df: pl.DataFrame, panel_group_name: str) -> None:
    """
    Validate panel group column exists and has appropriate type.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    panel_group_name : str
        Name of the panel grouping column.

    Raises
    ------
    ValueError
        If panel group column doesn't exist.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting.prep import validate_panel_group
    >>> df = pl.DataFrame({"time": [1, 2], "unique_id": ["A", "B"], "y": [10, 20]})
    >>> validate_panel_group(df, "unique_id")  # No error

    >>> try:
    ...     validate_panel_group(df, "missing")
    ... except ValueError as e:
    ...     print("caught")
    caught
    """
    if panel_group_name not in df.columns:
        msg = f"Panel group column '{panel_group_name}' not found in DataFrame"
        raise ValueError(msg)
