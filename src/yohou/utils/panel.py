"""Utilities for panel data inspection and filtering."""

import re

import polars as pl


def inspect_locality(df: pl.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    """Inspect DataFrame columns to distinguish global and local (panel) data.

    Global columns apply to all time series (e.g., single univariate series or
    features common across all panels). Local columns use the __ separator to
    indicate panel data groups following the pattern <GROUP>__<SERIES>
    (e.g., sales__store_1, sales__store_2).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with potential mix of global and group columns.
        Must contain a "time" column (which is ignored in the output).

    Returns
    -------
    global_names : list of str
        Names of columns without __ separator (excluding "time").

    local_groups : dict of str to list of str
        Mapping from group prefixes to their full column names.
        Example: {"sales": ["sales__store_1", "sales__store_2"]}

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

    >>> # Panel data with __ separator
    >>> df_panel = pl.DataFrame({
    ...     "time": [1, 2, 3],
    ...     "sales__store_1": [100, 110, 120],
    ...     "sales__store_2": [150, 160, 170]
    ... })
    >>> global_names, local_groups = inspect_locality(df_panel)
    >>> global_names
    []
    >>> local_groups
    {'sales': ['sales__store_1', 'sales__store_2']}

    See Also
    --------
    filter_panel_columns : Filter DataFrame to specific group for cross-learning
    """
    # Pattern to match <GROUP>__<SERIES> format
    group_pattern = re.compile(r'^([^_]+)__(.+)$')
    
    global_names = []
    local_groups: dict[str, list[str]] = {}
    
    for col in df.columns:
        if col == "time":
            continue
        
        match = group_pattern.match(col)
        if match:
            # This is a panel data column
            group_prefix = match.group(1)
            if group_prefix not in local_groups:
                local_groups[group_prefix] = []
            local_groups[group_prefix].append(col)
        else:
            # This is a global column
            global_names.append(col)
    
    return global_names, local_groups


def filter_panel_columns(
    df: pl.DataFrame,
    cross_learning_group: str,
    local_group_names: list[str] | None,
    include_global: bool = True,
) -> pl.DataFrame:
    """Filter DataFrame to specific group prefix for cross-learning.

    For panel data (DataFrames with columns using __ separator for groups),
    this function filters columns to keep only the "time" column, columns
    matching a specified group prefix, and optionally global columns.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with potential mix of global and group columns.
        Must contain a "time" column.

    cross_learning_group : str
        Group prefix to keep for cross-learning prediction (e.g., "sales").
        All columns matching <cross_learning_group>__* will be kept.

    local_group_names : list of str or None
        List of all group prefixes in the dataset. Used to distinguish
        group columns from global columns. If None, no filtering is performed.

    include_global : bool, default=True
        Whether to keep global columns (without __) in addition to time and
        the specified group columns.
        - True: Keep time + specified group + all global columns for X
        - False: Keep only time + specified group (for y target data)

    Returns
    -------
    pl.DataFrame
        Filtered DataFrame containing "time", columns matching the specified
        group prefix, and optionally global columns.

    Examples
    --------
    >>> import polars as pl
    >>> # Panel data with group columns and global column
    >>> df = pl.DataFrame({
    ...     "time": [1, 2, 3],
    ...     "global_feature": [10.0, 20.0, 30.0],
    ...     "sales__store_1": [100, 110, 120],
    ...     "sales__store_2": [150, 160, 170],
    ...     "inventory__store_1": [50, 55, 60],
    ...     "inventory__store_2": [75, 80, 85]
    ... })
    >>> # Filter for target (y) - exclude global features
    >>> y_filtered = filter_panel_columns(
    ...     df, "sales", ["sales", "inventory"], include_global=False
    ... )
    >>> y_filtered.columns
    ['time', 'sales__store_1', 'sales__store_2']

    >>> # Filter for features (X) - include global features
    >>> X_filtered = filter_panel_columns(
    ...     df, "sales", ["sales", "inventory"], include_global=True
    ... )
    >>> set(X_filtered.columns) == {'time', 'global_feature', 'sales__store_1', 'sales__store_2'}
    True

    See Also
    --------
    inspect_locality : Inspect DataFrame to identify global and local columns
    """
    # If no local groups, return DataFrame unchanged (no filtering needed)
    if local_group_names is None:
        return df

    # Determine which columns to keep
    cols_to_keep = ["time"]
    
    for col in df.columns:
        if col == "time":
            continue
        
        # Check if this column belongs to the target group
        if col.startswith(f"{cross_learning_group}__"):
            cols_to_keep.append(col)
        elif include_global:
            # Check if this is a global column (doesn't match any group prefix)
            is_global = True
            for group_prefix in local_group_names:
                if col.startswith(f"{group_prefix}__"):
                    is_global = False
                    break
            if is_global:
                cols_to_keep.append(col)
    
    return df.select(cols_to_keep)
