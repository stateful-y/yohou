"""Tabularization utilities for converting time series to supervised learning format."""

from typing import Sequence

import polars as pl


def tabularize(df_time_series: pl.DataFrame, lags: Sequence[int]) -> pl.DataFrame:
    """Convert time series to tabular format with lagged features.

    Creates a supervised learning dataset by generating lagged versions of each
    time series column. This is the core operation for reduction-based forecasting,
    enabling use of sklearn regressors for time series prediction.

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
    >>> # Original time series
    >>> df = pl.DataFrame({
    ...     "time": [1, 2, 3, 4, 5],
    ...     "value": [10, 20, 30, 40, 50]
    ... })
    >>> # Create lag features for lags 1, 2
    >>> df_tabular = tabularize(df, lags=[1, 2])
    >>> df_tabular
    shape: (3, 5)
    ┌──────┬────────────┬────────────┬─────────────┬─────────────┐
    │ time ┆ time_lag_1 ┆ time_lag_2 ┆ value_lag_1 ┆ value_lag_2 │
    │ ---  ┆ ---        ┆ ---        ┆ ---         ┆ ---         │
    │ i64  ┆ i64        ┆ i64        ┆ i64         ┆ i64         │
    ╞══════╪════════════╪════════════╪═════════════╪═════════════╡
    │ 3    ┆ 2          ┆ 1          ┆ 20          ┆ 10          │
    │ 4    ┆ 3          ┆ 2          ┆ 30          ┆ 20          │
    │ 5    ┆ 4          ┆ 3          ┆ 40          ┆ 30          │
    └──────┴────────────┴────────────┴─────────────┴─────────────┘

    Notes
    -----
    The transformation is commonly used in reduction forecasting strategies:
    - **Recursive**: Use lag features to predict next step iteratively
    - **Direct**: Use lag features to predict each horizon directly
    - **Multi-output**: Predict all horizons simultaneously from lag features

    The function automatically skips the "time" column and any datetime columns.

    See Also
    --------
    :class:`yohou.base.BaseReductionForecaster` : Uses tabularize for forecasting
    :meth:`yohou.base.BaseReductionForecaster._get_tabularized_dataset` : Wrapper with horizon handling

    """
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
