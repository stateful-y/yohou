"""Utilities for panel data inspection and filtering."""

import polars as pl


def inspect_locality(df: pl.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    """Inspect DataFrame columns to distinguish global and local (panel) data.

    Global columns apply to all time series (e.g., single univariate series or
    features common across all panels). Local columns are polars Struct columns
    containing different time series for each group (e.g., sales per store).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with potential mix of global and struct columns.
        Must contain a "time" column (which is ignored in the output).

    Returns
    -------
    global_names : list of str
        Names of non-struct columns (excluding "time").

    local_groups : dict of str to list of str
        Mapping from struct column names to their field names.

    Examples
    --------
    >>> import polars as pl
    >>> # Global time series (single series)
    >>> df_global = pl.DataFrame({
    ...     "time": [1, 2, 3],
    ...     "value": [10, 20, 30]
    ... })
    >>> global_names, local_groups = inspect_locality(df_global)
    >>> global_names
    ['value']
    >>> local_groups
    {}

    >>> # Panel data with struct column
    >>> df_panel = pl.DataFrame({
    ...     "time": [1, 2, 3],
    ...     "sales": pl.Series([
    ...         {"store_1": 100, "store_2": 150},
    ...         {"store_1": 110, "store_2": 160},
    ...         {"store_1": 120, "store_2": 170}
    ...     ])
    ... })
    >>> global_names, local_groups = inspect_locality(df_panel)
    >>> global_names
    []
    >>> local_groups
    {'sales': ['store_1', 'store_2']}

    See Also
    --------
    filter_panel_columns : Filter DataFrame to specific struct column for cross-learning
    """
    global_names, local_groups = [], {}
    for col, dtype in df.schema.items():
        if col == "time":
            continue

        if isinstance(dtype, pl.Struct):
            # Cast to Struct to access fields attribute
            struct_dtype = df.schema[col]
            if hasattr(struct_dtype, "fields"):
                local_groups[col] = [field.name for field in struct_dtype.fields]  # type: ignore[attr-defined]
        else:
            global_names.append(col)

    return global_names, local_groups


def filter_panel_columns(
    df: pl.DataFrame,
    cross_learning_group: str,
    local_group_names: list[str] | None,
    include_global: bool = True,
) -> pl.DataFrame:
    """Filter DataFrame to specific struct column for cross-learning.

    For panel data (DataFrames with struct columns representing multiple time series),
    this function filters columns to keep only the "time" column, a specified struct
    column, and optionally global (non-struct) columns.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with potential mix of global and struct columns.
        Must contain a "time" column.

    cross_learning_group : str
        Name of the struct column to keep for cross-learning prediction.

    local_group_names : list of str or None
        Names of all struct columns in the dataset. Used to distinguish
        struct columns from global columns. If None, no filtering is performed.

    include_global : bool, default=True
        Whether to keep global (non-struct) columns in addition to time and
        the specified struct column.
        - True: Keep time + specified struct + all global columns (for X_post/X_ante)
        - False: Keep only time + specified struct (for y target data)

    Returns
    -------
    pl.DataFrame
        Filtered DataFrame containing "time", the specified struct column,
        and optionally global columns.

    Examples
    --------
    >>> import polars as pl
    >>> # Panel data with struct and global columns
    >>> df = pl.DataFrame({
    ...     "time": [1, 2, 3],
    ...     "global_feature": [10.0, 20.0, 30.0],
    ...     "sales": pl.Series([
    ...         {"store_1": 100, "store_2": 150},
    ...         {"store_1": 110, "store_2": 160},
    ...         {"store_1": 120, "store_2": 170}
    ...     ])
    ... })
    >>> # Filter for target (y) - exclude global features
    >>> y_filtered = filter_panel_columns(
    ...     df, "sales", ["sales"], include_global=False
    ... )
    >>> y_filtered.columns
    ['time', 'sales']

    >>> # Filter for features (X) - include global features
    >>> X_filtered = filter_panel_columns(
    ...     df, "sales", ["sales"], include_global=True
    ... )
    >>> X_filtered.columns
    ['time', 'global_feature', 'sales']

    See Also
    --------
    inspect_locality : Inspect DataFrame to identify global and local columns
    """
    # If no local groups, return DataFrame unchanged (no filtering needed)
    if local_group_names is None:
        return df

    if include_global:
        # Keep time + specific struct + all global columns (non-struct columns)
        cols_to_keep = [
            c
            for c in df.columns
            if c == "time" or c == cross_learning_group or c not in local_group_names
        ]
    else:
        # Keep only time + specific struct column
        cols_to_keep = [c for c in df.columns if c == "time" or c == cross_learning_group]

    return df.select(cols_to_keep)
