"""Polars utilities for panel data and struct column manipulation."""

from typing import Literal

import polars as pl
import polars.selectors as cs

from .panel import inspect_locality


def concat_struct(
    items: list[pl.DataFrame], *, how: Literal["vertical", "horizontal", "diagonal"]
) -> pl.DataFrame:
    """Concatenate DataFrames with struct columns preserved.

    This function properly handles concatenation of DataFrames containing struct
    columns representing panel data. Unlike standard polars concat, this maintains
    the struct structure and handles mixed global/local columns correctly.

    Parameters
    ----------
    items : list of pl.DataFrame
        DataFrames to concatenate. Can contain mix of global columns and struct columns.

    how : {"vertical", "horizontal"}
        Concatenation direction:
        - "vertical": Stack rows (like pl.concat with how="vertical")
        - "horizontal": Join columns (like pl.concat with how="horizontal")

    Returns
    -------
    pl.DataFrame
        Concatenated DataFrame with struct columns properly preserved.

    Examples
    --------
    >>> import polars as pl
    >>> # Horizontal concat: adding features
    >>> df1 = pl.DataFrame({
    ...     "time": [1, 2, 3],
    ...     "sales": pl.Series([
    ...         {"store_1": 100, "store_2": 150},
    ...         {"store_1": 110, "store_2": 160},
    ...         {"store_1": 120, "store_2": 170}
    ...     ])
    ... })
    >>> df2 = pl.DataFrame({
    ...     "time": [1, 2, 3],
    ...     "temp": [20.0, 22.0, 21.0]  # Global feature
    ... })
    >>> result = concat_struct([df1, df2], how="horizontal")
    >>> result.columns
    ['time', 'temp', 'sales']

    >>> # Vertical concat: adding time steps
    >>> df_future = pl.DataFrame({
    ...     "time": [4, 5],
    ...     "sales": pl.Series([
    ...         {"store_1": 130, "store_2": 180},
    ...         {"store_1": 140, "store_2": 190}
    ...     ])
    ... })
    >>> result = concat_struct([df1, df_future], how="vertical")
    >>> len(result)
    5

    See Also
    --------
    inspect_locality : Detect global vs local columns
    select_struct : Select specific columns from struct DataFrames

    """
    items_global_names, items_local_groups = [], []
    for df in items:
        global_names, local_groups = inspect_locality(df)

        if "time" in df.columns:
            global_names = ["time"] + global_names

        items_global_names.append(global_names)
        items_local_groups.append(local_groups)

    local_group_names = set([name for local_group in items_local_groups for name in local_group])

    # For horizontal concat, only include time from first item to avoid duplicates
    if how == "horizontal":
        dfs_to_concat = [items[0].select(items_global_names[0])]
        for i, item in enumerate(items[1:], start=1):
            cols = [c for c in items_global_names[i] if c != "time"]
            if cols:
                dfs_to_concat.append(item.select(cols))
        out = pl.concat(dfs_to_concat, how=how)
    else:
        out = pl.concat(
            [item.select(items_global_names[i]) for i, item in enumerate(items)],
            how=how,
        )

    for local_group_name in local_group_names:
        df_group_list = []

        for df in items:
            # Only process DataFrames that actually have this struct column
            if local_group_name not in df.columns:
                continue

            df_group = df[
                [
                    col
                    for col, dtype in df.schema.items()
                    if dtype != pl.Struct or col == local_group_name
                ]
            ].unnest(local_group_name)

            df_group_list.append(df_group)

        df_group = pl.concat(df_group_list, how=how)

        df_group = pl.DataFrame({local_group_name: df_group})

        out = pl.concat([out, df_group], how="horizontal")

    return out


def select_struct(
    df: pl.DataFrame, local_col_names: list[str] | None, select_time: bool = True
) -> pl.DataFrame:
    """Select specific columns from DataFrame with nested struct columns.

    Recursively selects columns from a DataFrame that may contain struct columns.
    Useful for filtering panel data to specific series within struct columns.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with potential struct columns.

    local_col_names : list of str or None
        Names of columns to select from within struct columns. If None, selects all.

    select_time : bool, default=True
        Whether to include the "time" column in the output.

    Returns
    -------
    pl.DataFrame
        DataFrame with selected columns, maintaining struct structure.

    See Also
    --------
    inspect_locality : Detect struct columns in DataFrame
    neg_struct : Negate values in struct columns

    """
    add_time = False
    if select_time and "time" in df.columns:
        add_time = True
        time = df.select(pl.col("time"))
        df = df.select(~cs.by_name("time"))

    out = pl.DataFrame()
    for col, dtype in df.schema.items():
        df_col = df.select(pl.col(col))

        if dtype != pl.Struct and (local_col_names is None or col in local_col_names):
            pass

        elif dtype == pl.Struct:
            df_col = pl.DataFrame(
                {
                    col: select_struct(
                        df_col.unnest(col),
                        local_col_names=local_col_names,
                        select_time=False,
                    )
                }
            )
        else:
            continue

        out = pl.concat([out, df_col], how="horizontal")

    if add_time:
        out = pl.concat([time, out], how="horizontal")

    return out


def neg_struct(
    df: pl.DataFrame, local_col_names: list[str] | None = None, prefix: str = ""
) -> pl.DataFrame:
    """Negate values in DataFrame columns, recursively handling struct columns.

    Applies negation to specified columns, including those nested within struct
    columns. Useful for computing prediction errors or differences in panel data.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with potential struct columns.

    local_col_names : list of str or None, default=None
        Names of columns to negate. If None, negates all non-struct columns.

    prefix : str, default=""
        Prefix to add to negated column names (e.g., "neg_").

    Returns
    -------
    pl.DataFrame
        DataFrame with specified columns negated, maintaining structure.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "time": [1, 2, 3],
    ...     "value": [10, 20, 30]
    ... })
    >>> result = neg_struct(df, prefix="neg_")
    >>> result.select("neg_value")
    shape: (3, 1)
    ┌───────────┐
    │ neg_value │
    │ ---       │
    │ i64       │
    ╞═══════════╡
    │ -10       │
    │ -20       │
    │ -30       │
    └───────────┘

    See Also
    --------
    select_struct : Select columns from struct DataFrames
    inspect_locality : Detect struct columns

    """
    out = pl.DataFrame()

    add_time = False
    if "time" in df.columns:
        add_time = True
        time = df.select(pl.col("time"))
        df = df.select(~cs.by_name("time"))

    for col, dtype in df.schema.items():
        df_col = df.select(pl.col(col))

        if dtype != pl.Struct and (local_col_names is None or col in local_col_names):
            df_col = df_col.select(-pl.col(col).alias(f"{prefix}{col}"))

        elif dtype == pl.Struct:
            df_col = pl.DataFrame(
                {
                    col: neg_struct(
                        df_col.unnest(col), local_col_names=local_col_names, prefix=prefix
                    )
                }
            )

        else:
            # If it's not a Struct and not in local_col_names, keep it as is
            df_col = df_col

        out = pl.concat([out, df_col], how="horizontal")

    if add_time:
        out = pl.concat([time, out], how="horizontal")

    return out
