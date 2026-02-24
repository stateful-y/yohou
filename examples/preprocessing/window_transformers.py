# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Window Transformers: Lags, Rolling Statistics, and EMA.

Demonstrates LagTransformer, RollingStatisticsTransformer, SlidingWindowFunctionTransformer,
and ExponentialMovingAverage.
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Window Transformers

    Window transformers create features from temporal windows of data. They are
    the core building blocks for feature engineering in time series forecasting.

    ## What You'll Learn

    - `LagTransformer`: Create lagged copies of columns
    - `RollingStatisticsTransformer`: Rolling mean, std, min, max, etc.
    - `SlidingWindowFunctionTransformer`: Custom functions over sliding windows
    - `ExponentialMovingAverage`: Exponentially weighted moving average
    - How `observation_horizon` works for stateful transformers
    - Combining transformers with `FeatureUnion`

    ## Prerequisites

    Basic understanding of feature engineering and time series concepts.
    """)
    return

@app.cell(hide_code=True)
def _():
    import polars as pl

    from yohou.compose import FeatureUnion
    from yohou.datasets import fetch_tourism_monthly
    from yohou.plotting import plot_rolling_statistics, plot_time_series
    from yohou.preprocessing import (
        ExponentialMovingAverage,
        LagTransformer,
        RollingStatisticsTransformer,
        SlidingWindowFunctionTransformer,
    )

    return (
        ExponentialMovingAverage,
        FeatureUnion,
        LagTransformer,
        RollingStatisticsTransformer,
        SlidingWindowFunctionTransformer,
        fetch_tourism_monthly,
        pl,
        plot_rolling_statistics,
        plot_time_series,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We load the Monthly Tourism dataset and split it into segments for demonstrating windowed transformations.
    """)
    return

@app.cell
def _(fetch_tourism_monthly):
    y = (
        fetch_tourism_monthly()
        .frame.select("time", "T1__tourists").drop_nulls()
        .rename({"T1__tourists": "passengers"})
    )
    print(f"Shape: {y.shape}")
    y.head()
    return (y,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. LagTransformer

    Creates lagged columns. `lag=3` means the value from 3 steps ago.
    The first `max(lag)` rows become null (the `observation_horizon`).
    """)
    return

@app.cell
def _(LagTransformer, y):
    lag_tf = LagTransformer(lag=[1, 3, 6, 12])
    lag_tf.fit(y)
    y_lagged = lag_tf.transform(y)

    print(f"observation_horizon: {lag_tf.observation_horizon}")
    print(f"Input columns:  {y.columns}")
    print(f"Output columns: {y_lagged.columns}")
    y_lagged.head(15)
    return lag_tf, y_lagged

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. RollingStatisticsTransformer

    Computes rolling statistics over a sliding window. Multiple statistics
    can be computed simultaneously.
    """)
    return

@app.cell
def _(RollingStatisticsTransformer, y):
    roll_tf = RollingStatisticsTransformer(
        window_size=12,
        statistics=["mean", "std", "min", "max"],
    )
    roll_tf.fit(y)
    y_rolled = roll_tf.transform(y)

    print(f"observation_horizon: {roll_tf.observation_horizon}")
    print(f"Output columns: {y_rolled.columns}")
    y_rolled.head(15)
    return roll_tf, y_rolled

@app.cell
def _(plot_rolling_statistics, y):
    plot_rolling_statistics(
        y,
        window_size=12,
        statistics=["mean", "std"],
        title="Rolling Statistics (window=12)",
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. SlidingWindowFunctionTransformer

    Apply any custom function over a sliding window.
    The function receives a polars Series and returns a scalar.
    """)
    return

@app.cell
def _(SlidingWindowFunctionTransformer, y):
    # Custom: range (max - min) over last 6 periods
    # func receives a polars DataFrame window, must return scalar, dict, or ndarray
    range_tf = SlidingWindowFunctionTransformer(
        func=lambda df: {
            col: df[col].max() - df[col].min()
            for col in df.columns
            if col != "time"
        },
        window_size=6,
    )
    range_tf.fit(y)
    y_range = range_tf.transform(y)

    print(f"observation_horizon: {range_tf.observation_horizon}")
    y_range.head(10)
    return range_tf, y_range

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. ExponentialMovingAverage

    EWMA gives more weight to recent observations. Unlike rolling stats,
    it has `observation_horizon=0` (stateless) since it uses all history.
    """)
    return

@app.cell
def _(ExponentialMovingAverage, y):
    ema_tf = ExponentialMovingAverage(alpha=0.3)
    ema_tf.fit(y)
    y_ema = ema_tf.transform(y)

    print(f"observation_horizon: {ema_tf.observation_horizon}")
    y_ema.head()
    return ema_tf, y_ema

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Combining Transformers with FeatureUnion

    Combine multiple window transformers in parallel using `FeatureUnion`.
    All outputs are concatenated horizontally.
    """)
    return

@app.cell
def _(
    ExponentialMovingAverage,
    FeatureUnion,
    LagTransformer,
    RollingStatisticsTransformer,
    y,
):
    union = FeatureUnion(
        transformer_list=[
            ("lags", LagTransformer(lag=[1, 12])),
            ("rolling", RollingStatisticsTransformer(window_size=6, statistics="mean")),
            ("ema", ExponentialMovingAverage(alpha=0.3)),
        ],
    )

    union.fit(y)
    y_combined = union.transform(y)

    print(f"Combined observation_horizon: {union.observation_horizon}")
    print(f"Combined columns: {y_combined.columns}")
    y_combined.head(15)
    return union, y_combined

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - `LagTransformer`: Creates lagged columns; `observation_horizon = max(lag)`
    - `RollingStatisticsTransformer`: Computes rolling stats; multiple statistics at once
    - `SlidingWindowFunctionTransformer`: Any custom function over sliding windows
    - `ExponentialMovingAverage`: EWMA smoothing; `observation_horizon = 0` (stateless)
    - `FeatureUnion` combines transformers in parallel for rich feature engineering
    - Stateful transformers produce nulls for the first `observation_horizon` rows
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Imputation & outliers**: See `data_cleaning.py` for handling nulls and outliers
    - **Resampling**: See `resampling.py` for frequency changes
    - **Using in forecasters**: Pass to `feature_transformer` in `PointReductionForecaster`
    """)
    return

if __name__ == "__main__":
    app.run()
