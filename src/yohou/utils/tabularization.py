"""Tabularization utilities for converting time series tasks to tabular tasks."""

from collections.abc import Sequence

import polars as pl

__all__ = ["tabularize"]


def tabularize(df_time_series: pl.DataFrame, lags: Sequence[int]) -> pl.DataFrame:
    """Convert time series to tabular format using lags.

    Creates a tabular dataset by generating lagged versions of each time series
    column. This is the core operation for reduction-based forecasting, enabling
    use of sklearn estimators for time series prediction.

    Parameters
    ----------
    df_time_series : pl.DataFrame
        Time series DataFrame with columns to be lagged (excluding "time").

    lags : Sequence of int
        Lag values to create. Each value i creates features shifted by i time steps.
        For example, lags=[1, 2, 3] creates lag_1, lag_2, and lag_3 features.

    Returns
    -------
    pl.DataFrame
        Tabularized DataFrame with lagged feature columns. The first max(lags) rows
        are dropped since they would contain null values. Column names follow the
        pattern "{original_column}_lag_{i}".

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> # Original time series with a contract-compliant datetime "time" column
    >>> df = pl.DataFrame({
    ...     "time": pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 5), "1d", eager=True),
    ...     "value": [10, 20, 30, 40, 50],
    ... })
    >>> # Create lag features for lags 1, 2
    >>> df_tabular = tabularize(df, lags=[1, 2])
    >>> df_tabular
    shape: (3, 3)
    ┌─────────────────────┬─────────────┬─────────────┐
    │ time                ┆ value_lag_1 ┆ value_lag_2 │
    │ ---                 ┆ ---         ┆ ---         │
    │ datetime[μs]        ┆ i64         ┆ i64         │
    ╞═════════════════════╪═════════════╪═════════════╡
    │ 2020-01-03 00:00:00 ┆ 20          ┆ 10          │
    │ 2020-01-04 00:00:00 ┆ 30          ┆ 20          │
    │ 2020-01-05 00:00:00 ┆ 40          ┆ 30          │
    └─────────────────────┴─────────────┴─────────────┘

    See Also
    --------
    - [`BaseReductionForecaster`][yohou.base.reduction.BaseReductionForecaster] : Uses tabularize for forecasting
    - [`LagTransformer`][yohou.preprocessing.window.LagTransformer] : Transformer that applies similar lagging logic

    """
    columns = [col for col in df_time_series.columns if col != "time"]
    df_tabular = (
        df_time_series.with_columns([
            pl.col(col).shift(i).alias(f"{col}_lag_{i}")
            for (col, dtype) in zip(df_time_series.columns, df_time_series.dtypes, strict=False)
            for i in lags
            if dtype != pl.Datetime
        ])
    ).select(pl.exclude(columns))[max(lags) :]

    return df_tabular
