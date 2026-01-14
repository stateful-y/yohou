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

    panel_groups : dict of str to list of str
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
    >>> global_names, panel_groups = inspect_locality(df_global)
    >>> global_names
    ['value']
    >>> panel_groups
    {}

    >>> # Panel data with __ separator
    >>> df_panel = pl.DataFrame({
    ...     "time": [1, 2, 3],
    ...     "sales__store_1": [100, 110, 120],
    ...     "sales__store_2": [150, 160, 170]
    ... })
    >>> global_names, panel_groups = inspect_locality(df_panel)
    >>> global_names
    []
    >>> panel_groups
    {'sales': ['sales__store_1', 'sales__store_2']}

    See Also
    --------
    select_panel_columns : Filter DataFrame to panel group columns and global columns
    """
    # Pattern to match <GROUP>__<SERIES> format
    group_pattern = re.compile(r"^([^_]+)__(.+)$")

    global_names = []
    panel_groups: dict[str, list[str]] = {}

    for col in df.columns:
        if col == "time":
            continue

        match = group_pattern.match(col)
        if match:
            # This is a panel data column
            group_prefix = match.group(1)
            if group_prefix not in panel_groups:
                panel_groups[group_prefix] = []
            panel_groups[group_prefix].append(col)
        else:
            # This is a global column
            global_names.append(col)

    # Validate that unprefixed panel column names don't conflict with global columns
    if panel_groups and global_names:
        # Extract unprefixed names from all panel columns
        unprefixed_panel_names = set()
        for group_cols in panel_groups.values():
            for col in group_cols:
                # Extract the part after __
                unprefixed_name = col.split("__", 1)[1]
                unprefixed_panel_names.add(unprefixed_name)

        # Check for conflicts with global column names
        conflicts = unprefixed_panel_names.intersection(global_names)
        if conflicts:
            raise ValueError(
                f"Panel column names (after removing group prefix) conflict with global column names: {sorted(conflicts)}. "
                f"Panel columns with __ separator cannot have the same name as global columns. "
                f"For example, if you have 'x__a' and a global column 'a', this creates ambiguity."
            )

    return global_names, panel_groups


def get_group_df(
    df: pl.DataFrame,
    group_name: str,
    schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    """Extract and rename columns for a specific panel group.

    Selects columns matching the group prefix pattern (<group_name>__*),
    renames them to remove the prefix, and returns a DataFrame with "time"
    and the unprefixed columns. Also handles global columns (no prefix) that
    are shared across all groups.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with panel data columns.
        Must contain a "time" column.
    group_name : str
        Group prefix to extract (e.g., "sales", "inventory").
        Columns matching <group_name>__* will be selected.
    schema : dict of str to pl.DataType
        Schema mapping unprefixed column names to their data types.
        Used to determine which columns to extract.
        Can contain both local columns (will have group prefix in df) and
        global columns (no prefix in df).
        Example: {"store_1": pl.Int64, "store_2": pl.Int64, "holiday": pl.Boolean}

    Returns
    -------
    pl.DataFrame
        DataFrame with "time" column and unprefixed group columns.
        Local columns are renamed from <group_name>__<col> to <col>.
        Global columns keep their original names.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "time": [1, 2, 3],
    ...     "sales__store_1": [100, 110, 120],
    ...     "sales__store_2": [150, 160, 170],
    ...     "holiday": [True, False, True],  # Global column
    ...     "inventory__store_1": [50, 55, 60]
    ... })
    >>> # Schema includes both local and global columns
    >>> schema = {"store_1": pl.Int64, "store_2": pl.Int64, "holiday": pl.Boolean}
    >>> df_sales = get_group_df(df, "sales", schema)
    >>> df_sales.columns
    ['time', 'store_1', 'store_2', 'holiday']
    >>> df_sales.shape
    (3, 4)

    See Also
    --------
    inspect_locality : Inspect DataFrame to identify global and local columns
    select_panel_columns : Filter DataFrame to panel group columns and global columns

    Notes
    -----
    This function is used internally by forecasters to extract individual
    panel groups for processing, particularly in the context of the new
    architecture where schemas store unprefixed column names.

    For X (feature) data, the schema typically combines local_X_schema_ and
    global_X_schema_, allowing each group to access both its own features
    and shared global features.
    """
    # Separate local (prefixed) and global (unprefixed) columns
    local_cols = []
    global_cols = []
    rename_map = {}

    for col_name in schema.keys():
        prefixed_col = f"{group_name}__{col_name}"
        if prefixed_col in df.columns:
            # Local column (has group prefix)
            local_cols.append(prefixed_col)
            rename_map[prefixed_col] = col_name
        elif col_name in df.columns:
            # Global column (no prefix)
            global_cols.append(col_name)
        else:
            # Column not found
            raise ValueError(
                f"Column '{col_name}' not found as either '{prefixed_col}' (local) "
                f"or '{col_name}' (global) in DataFrame. "
                f"Available columns: {df.columns}"
            )

    # Select time + local + global columns
    df_group = df.select(["time"] + local_cols + global_cols)

    # Rename only local columns to remove prefix (global columns keep their names)
    if rename_map:
        df_group = df_group.rename(rename_map)

    return df_group


def select_panel_columns(
    df: pl.DataFrame,
    panel_group_names: list[str] | None,
    include_global: bool = True,
) -> pl.DataFrame:
    """Select panel group columns and optionally global columns of a DataFrame.

    For panel data (DataFrames with columns using __ separator for groups),
    this function filters columns to keep only the "time" column, columns
    matching any of the panel group prefixes, and optionally global columns.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with potential mix of global and group columns.
        Must contain a "time" column.

    panel_group_names : list of str or None
        List of all group prefixes in the dataset. All columns matching
        any <group>__* pattern will be kept. If None, no filtering is performed.

    include_global : bool, default=True
        Whether to keep global columns (without __) in addition to time and
        panel group columns.
        - True: Keep time + all panel groups + all global columns for X
        - False: Keep only time + all panel groups (for y target data)

    Returns
    -------
    pl.DataFrame
        Filtered DataFrame containing "time", columns matching any panel
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
    >>> y_filtered = select_panel_columns(
    ...     df, ["sales", "inventory"], include_global=False
    ... )
    >>> set(y_filtered.columns) == {
    ...     'time', 'sales__store_1', 'sales__store_2',
    ...     'inventory__store_1', 'inventory__store_2'
    ... }
    True

    >>> # Filter for features (X) - include global features
    >>> X_filtered = select_panel_columns(
    ...     df, ["sales", "inventory"], include_global=True
    ... )
    >>> set(X_filtered.columns) == {
    ...     'time', 'global_feature', 'sales__store_1', 'sales__store_2',
    ...     'inventory__store_1', 'inventory__store_2'
    ... }
    True

    See Also
    --------
    inspect_locality : Inspect DataFrame to identify global and local columns
    """
    # If no local groups, return DataFrame unchanged (no filtering needed)
    if panel_group_names is None:
        return df

    # Determine which columns to keep
    cols_to_keep = ["time"]

    for col in df.columns:
        if col == "time":
            continue

        # Check if this column belongs to any panel group
        is_panel = False
        for group_prefix in panel_group_names:
            if col.startswith(f"{group_prefix}__"):
                is_panel = True
                break

        if is_panel:
            cols_to_keep.append(col)
        elif include_global:
            # Global column (doesn't match any group prefix)
            cols_to_keep.append(col)

    return df.select(cols_to_keep)


def dict_to_panel(data: dict[str, pl.DataFrame] | pl.DataFrame | None) -> pl.DataFrame | None:
    """Convert a dict of group DataFrames to a single DataFrame with prefixed columns.

    Takes a dictionary mapping group names to DataFrames and combines them into
    a single DataFrame where each group's columns are prefixed with the group name
    using the __ separator pattern (<group_name>__<column>). If the input is already
    a DataFrame, returns it unchanged.

    Parameters
    ----------
    data : dict of str to pl.DataFrame or pl.DataFrame
        Either a dictionary mapping group names to DataFrames, or an already
        combined DataFrame. Each DataFrame in the dict must have a "time" column
        and additional feature columns.

    Returns
    -------
    pl.DataFrame or None
        Combined DataFrame with prefixed columns. The "time" column is shared
        across all groups. Other columns are prefixed as <group_name>__<column>.

    Examples
    --------
    >>> import polars as pl
    >>> # Dictionary of group DataFrames
    >>> data_dict = {
    ...     "sales": pl.DataFrame({
    ...         "time": [1, 2, 3],
    ...         "store_1": [100, 110, 120],
    ...         "store_2": [150, 160, 170]
    ...     }),
    ...     "inventory": pl.DataFrame({
    ...         "time": [1, 2, 3],
    ...         "warehouse_1": [50, 55, 60],
    ...         "warehouse_2": [75, 80, 85]
    ...     })
    ... }
    >>> df_panel = dict_to_panel(data_dict)
    >>> sorted(df_panel.columns)
    ['inventory__warehouse_1', 'inventory__warehouse_2', 'sales__store_1', 'sales__store_2', 'time']

    >>> # Already a DataFrame - returns unchanged
    >>> df_existing = pl.DataFrame({
    ...     "time": [1, 2, 3],
    ...     "sales__store_1": [100, 110, 120]
    ... })
    >>> result = dict_to_panel(df_existing)
    >>> result.equals(df_existing)
    True

    See Also
    --------
    inspect_locality : Inspect DataFrame to identify global and local columns
    get_group_df : Extract a single panel group from a combined DataFrame

    Notes
    -----
    This function is the inverse operation of extracting groups with get_group_df.
    It's commonly used internally by forecasters to convert between the dict
    representation (easier for per-group processing) and the prefixed column
    representation (polars-native format).
    """
    # If already a DataFrame, return as-is
    if data is None or isinstance(data, pl.DataFrame):
        return data

    if all(v is None for v in data.values()):
        return None

    # Convert dict of DataFrames to single DataFrame with prefixed columns
    if not data:
        raise ValueError("Cannot convert empty dict to panel DataFrame")

    # Start with the first group to get the time column
    first_group_name = next(iter(data))
    result = data[first_group_name].select("time")

    # Add each group's columns with prefixes
    for group_name, group_df in data.items():
        # Get all columns except time
        feature_cols = [col for col in group_df.columns if col != "time"]

        # Rename columns with group prefix
        renamed_df = group_df.select(
            ["time"] + [pl.col(col).alias(f"{group_name}__{col}") for col in feature_cols]
        )

        # Join with result (on time)
        result = result.join(renamed_df, on="time", how="inner")

    return result
