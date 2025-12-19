import polars as pl
import polars.selectors as cs


def inspect_locality(df: pl.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    global_names, local_groups = [], {}
    for col, dtype in df.schema.items():
        if col == "time":
            continue

        if isinstance(dtype, pl.Struct):
            local_groups[col] = [field.name for field in df.schema[col].fields]
        else:
            global_names.append(col)

    return global_names, local_groups


def concat_struct(items: list[pl.DataFrame], *, how: str) -> pl.DataFrame:
    items_global_names, items_local_groups = [], []
    for df in items:
        global_names, local_groups = inspect_locality(df)

        if "time" in df.columns:
            global_names = ["time"] + global_names

        items_global_names.append(global_names)
        items_local_groups.append(local_groups)

    local_group_names = set([name for local_group in items_local_groups for name in local_group])

    out = pl.concat(
        [item.select(items_global_names[i]) for i, item in enumerate(items)],
        how=how,
    )

    for local_group_name in local_group_names:
        df_group_list = []

        for df in items:
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

        out = pl.concat([out, df_group], how=how)

    return out


def select_struct(
    df: pl.DataFrame, local_col_names: list[str] | None, select_time: bool = True
) -> pl.DataFrame:
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
