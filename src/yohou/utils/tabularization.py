from typing import Sequence

import polars as pl


def tabularize(df_time_series: pl.DataFrame, lags: Sequence[int]) -> pl.DataFrame:
    columns = [col for col in df_time_series.columns if col != "time"]
    df_tabular = (
        df_time_series.with_columns(
            [
                pl.col(col).shift(i).alias(f"{col}_lag_{i}")
                for (col, dtype) in zip(df_time_series.columns, df_time_series.dtypes)
                for i in lags
                if dtype != pl.Datetime
            ]
        )
    ).select(pl.exclude(columns))[max(lags) :]

    return df_tabular
