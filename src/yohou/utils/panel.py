"""Utilities for panel data inspection and filtering."""

import re
from collections.abc import Callable

import polars as pl
import polars.selectors as cs

__all__ = [
    "dict_to_panel",
    "get_group_df",
    "inspect_panel",
    "panel_aware_prefix",
    "panel_aware_rename",
    "panel_aware_suffix",
    "select_panel_columns",
]


def inspect_panel(df: pl.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    """Inspect DataFrame columns to distinguish global and local (panel) data.

    Global columns apply to all time series (e.g., single univariate series or
    features common across all panels). Local columns use the __ separator to
    indicate panel data groups following the pattern <GROUP>__<SERIES>
    (e.g., store_1__sales, store_2__sales).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with potential mix of global and group columns.
        The "time" column, if present, is excluded from the output.

    Returns
    -------
    global_names : list of str
        Names of columns without __ separator (excluding "time").

    panel_groups : dict of str to list of str
        Mapping from group prefixes to their full column names.
        Example: {"store_1": ["store_1__sales", "store_1__returns"]}

    Examples
    --------
    >>> import polars as pl
    >>> # Global time series (single series)
    >>> df_global = pl.DataFrame({"time": [1, 2, 3], "value": [10, 20, 30]})
    >>> global_names, panel_groups = inspect_panel(df_global)
    >>> global_names
    ['value']
    >>> panel_groups
    {}

    >>> # Panel data with __ separator (<entity>__<variable>)
    >>> df_panel = pl.DataFrame({
    ...     "time": [1, 2, 3],
    ...     "store_1__sales": [100, 110, 120],
    ...     "store_2__sales": [150, 160, 170],
    ... })
    >>> global_names, panel_groups = inspect_panel(df_panel)
    >>> global_names
    []
    >>> panel_groups
    {'store_1': ['store_1__sales'], 'store_2': ['store_2__sales']}

    See Also
    --------
    - [`select_panel_columns`][yohou.utils.panel.select_panel_columns] : Filter DataFrame to panel group columns and global columns
    """
    # Pattern to match <GROUP>__<SERIES> format
    # Non-greedy prefix allows group names with underscores (e.g., new_south_wales__trips)
    group_pattern = re.compile(r"^(.+?)__(.+)$")

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
    key_cols: tuple[str, ...] = ("time",),
) -> pl.DataFrame:
    """Extract and rename columns for a specific panel group.

    Selects columns matching the group prefix pattern (<group_name>__*),
    renames them to remove the prefix, and returns a DataFrame with key
    columns and the unprefixed columns. Also handles global columns (no
    prefix) that are shared across all groups.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with panel data columns.
        Must contain columns listed in ``key_cols``.
    group_name : str
        Group prefix to extract (e.g., "sales", "inventory").
        Columns matching <group_name>__* will be selected.
    schema : dict of str to pl.DataType
        Schema mapping unprefixed column names to their data types.
        Used to determine which columns to extract.
        Can contain both local columns (will have group prefix in df) and
        global columns (no prefix in df).
        Example: {"store_1": pl.Int64, "store_2": pl.Int64, "holiday": pl.Boolean}
    key_cols : tuple of str, default=("time",)
        Index columns to preserve in the output (e.g. ``("time",)`` for
        y/X_actual/X_future, ``("vintage_time", "time")`` for X_forecast).

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
    ...     "inventory__store_1": [50, 55, 60],
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
    - [`inspect_panel`][yohou.utils.panel.inspect_panel] : Inspect DataFrame to identify global and local columns
    - [`select_panel_columns`][yohou.utils.panel.select_panel_columns] : Filter DataFrame to panel group columns and global columns

    Notes
    -----
    This function is used internally by forecasters to extract individual
    panel groups for processing, particularly in the context of the new
    architecture where schemas store unprefixed column names.

    For X (feature) data, the schema typically combines local_X_actual_schema_ and
    shared_X_actual_schema_, allowing each group to access both its own features
    and shared features.
    """
    # Separate local (prefixed) and global (unprefixed) columns
    local_cols = []
    global_cols = []
    rename_map = {}

    for col_name in schema:
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

    # Select key + local + global columns
    df_group = df.select(list(key_cols) + local_cols + global_cols)

    # Rename only local columns to remove prefix (global columns keep their names)
    if rename_map:
        df_group = df_group.rename(rename_map)

    return df_group


def select_panel_columns(
    df: pl.DataFrame,
    groups: list[str] | None,
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

    groups : list of str or None
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
    ...     "inventory__store_2": [75, 80, 85],
    ... })
    >>> # Filter for target (y) - exclude global features
    >>> y_filtered = select_panel_columns(df, ["sales", "inventory"], include_global=False)
    >>> set(y_filtered.columns) == {
    ...     "time",
    ...     "sales__store_1",
    ...     "sales__store_2",
    ...     "inventory__store_1",
    ...     "inventory__store_2",
    ... }
    True

    >>> # Filter for features (X) - include global features
    >>> X_filtered = select_panel_columns(df, ["sales", "inventory"], include_global=True)
    >>> set(X_filtered.columns) == {
    ...     "time",
    ...     "global_feature",
    ...     "sales__store_1",
    ...     "sales__store_2",
    ...     "inventory__store_1",
    ...     "inventory__store_2",
    ... }
    True

    See Also
    --------
    - [`inspect_panel`][yohou.utils.panel.inspect_panel] : Inspect DataFrame to identify global and local columns
    """
    # If no local groups, return DataFrame unchanged (no filtering needed)
    if groups is None:
        return df

    # Index columns ("time" and, when present, "vintage_time") are always kept.
    index_selector = cs.by_name("time", "vintage_time", require_all=False)

    # Columns belonging to a requested panel group start with "<group>__"; the
    # regex stays in the polars engine and preserves original column order.
    panel_regex = "^(?:" + "|".join(re.escape(g) for g in groups) + ")__"
    panel_selector = cs.matches(panel_regex)

    selector = index_selector | panel_selector
    if include_global:
        # Global columns carry no "<group>__" panel prefix at all. Detecting them
        # by the absence of a "__" separator (rather than "does not match a
        # requested group") ensures columns of a *non-requested* group are dropped
        # instead of being mistaken for globals.
        selector = selector | (cs.all() - index_selector - cs.contains("__"))

    return df.select(selector)


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
    ...         "store_2": [150, 160, 170],
    ...     }),
    ...     "inventory": pl.DataFrame({
    ...         "time": [1, 2, 3],
    ...         "warehouse_1": [50, 55, 60],
    ...         "warehouse_2": [75, 80, 85],
    ...     }),
    ... }
    >>> df_panel = dict_to_panel(data_dict)
    >>> sorted(df_panel.columns)
    ['inventory__warehouse_1', 'inventory__warehouse_2', 'sales__store_1', 'sales__store_2', 'time']

    >>> # Already a DataFrame - returns unchanged
    >>> df_existing = pl.DataFrame({"time": [1, 2, 3], "sales__store_1": [100, 110, 120]})
    >>> result = dict_to_panel(df_existing)
    >>> result.equals(df_existing)
    True

    See Also
    --------
    - [`inspect_panel`][yohou.utils.panel.inspect_panel] : Inspect DataFrame to identify global and local columns
    - [`get_group_df`][yohou.utils.panel.get_group_df] : Extract a single panel group from a combined DataFrame

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

    # Reject empty dicts before the all-None check: an empty dict satisfies
    # all(...) vacuously, which would otherwise mask it as an all-None input.
    if not data:
        raise ValueError("Cannot convert empty dict to panel DataFrame")

    if all(v is None for v in data.values()):
        return None

    # Convert dict of DataFrames to single DataFrame with prefixed columns
    # Start with the first group to get the time column
    first_group_name = next(iter(data))
    reference_times = data[first_group_name].get_column("time")

    # All groups must share an identical time axis; an inner join would
    # otherwise silently drop timestamps not present in every group. A set
    # comparison cannot detect duplicate timestamps (e.g. [1, 1, 2] vs
    # [1, 2, 2]), so compare the ordered series directly instead.
    misaligned = [
        name
        for name, df in data.items()
        if name != first_group_name and not df.get_column("time").equals(reference_times)
    ]
    if misaligned:
        raise ValueError(
            f"dict_to_panel requires every group to share an identical 'time' axis, "
            f"but these groups differ from group '{first_group_name}': {misaligned}. "
            f"Align the group time columns (e.g. resample or reindex) before assembling the panel."
        )

    result = data[first_group_name].select("time")

    # Add each group's columns with prefixes
    for group_name, group_df in data.items():
        # Get all columns except time
        feature_cols = [col for col in group_df.columns if col != "time"]

        # Rename columns with group prefix
        renamed_df = group_df.select(["time"] + [pl.col(col).alias(f"{group_name}__{col}") for col in feature_cols])

        # Join with result (on time)
        result = result.join(renamed_df, on="time", how="inner")

    return result


def panel_aware_rename(col: str, fn: Callable[[str], str]) -> str:
    """Apply a rename function to a column name while preserving the panel group prefix.

    For panel data columns following the ``<GROUP>__<SERIES>`` convention,
    the rename function is applied only to the series (member) portion,
    leaving the group prefix intact: ``<GROUP>__fn(<SERIES>)``.

    For global columns (no ``__`` separator), the rename function is applied
    to the entire column name: ``fn(<COLUMN>)``.

    Parameters
    ----------
    col : str
        Column name to rename.
    fn : callable
        Function that takes a string and returns a string.
        Applied to the series part of panel columns or the full name
        for global columns.

    Returns
    -------
    str
        Renamed column name with panel group prefix preserved.

    Examples
    --------
    >>> panel_aware_rename("sales", lambda s: f"log_{s}")
    'log_sales'
    >>> panel_aware_rename("store_1__sales", lambda s: f"log_{s}")
    'store_1__log_sales'
    >>> panel_aware_rename("store_1__sales", lambda s: f"{s}_lag_1")
    'store_1__sales_lag_1'

    See Also
    --------
    - [`panel_aware_prefix`][yohou.utils.panel.panel_aware_prefix] : Convenience wrapper for adding a prefix.
    - [`panel_aware_suffix`][yohou.utils.panel.panel_aware_suffix] : Convenience wrapper for adding a suffix.

    """
    if "__" in col:
        group, member = col.split("__", 1)
        return f"{group}__{fn(member)}"
    return fn(col)


def panel_aware_prefix(col: str, prefix: str) -> str:
    """Add a prefix to a column name while preserving the panel group prefix.

    For panel data columns (``<GROUP>__<SERIES>``), the prefix is inserted
    after the group separator: ``<GROUP>__<PREFIX>_<SERIES>``.

    For global columns, the prefix is prepended: ``<PREFIX>_<COLUMN>``.

    Parameters
    ----------
    col : str
        Column name to prefix.
    prefix : str
        Prefix string to add (joined with ``_``).

    Returns
    -------
    str
        Column name with prefix added in the panel-safe position.

    Examples
    --------
    >>> panel_aware_prefix("sales", "boxcox")
    'boxcox_sales'
    >>> panel_aware_prefix("store_1__sales", "boxcox")
    'store_1__boxcox_sales'

    See Also
    --------
    - [`panel_aware_rename`][yohou.utils.panel.panel_aware_rename] : General-purpose panel-aware rename utility.

    """
    return panel_aware_rename(col, lambda member: f"{prefix}_{member}")


def panel_aware_suffix(col: str, suffix: str) -> str:
    """Add a suffix to a column name while preserving the panel group prefix.

    For panel data columns (``<GROUP>__<SERIES>``), the suffix is appended
    to the series part: ``<GROUP>__<SERIES>_<SUFFIX>``.

    For global columns, the suffix is appended: ``<COLUMN>_<SUFFIX>``.

    Parameters
    ----------
    col : str
        Column name to suffix.
    suffix : str
        Suffix string to add (joined with ``_``).

    Returns
    -------
    str
        Column name with suffix added.

    Examples
    --------
    >>> panel_aware_suffix("sales", "lag_1")
    'sales_lag_1'
    >>> panel_aware_suffix("store_1__sales", "lag_1")
    'store_1__sales_lag_1'

    See Also
    --------
    - [`panel_aware_rename`][yohou.utils.panel.panel_aware_rename] : General-purpose panel-aware rename utility.

    """
    return panel_aware_rename(col, lambda member: f"{member}_{suffix}")
