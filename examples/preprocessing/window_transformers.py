# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Window Transformers",
    "description": "Feature engineering with LagTransformer, RollingStatisticsTransformer, SlidingWindowFunctionTransformer, and ExponentialMovingAverage on time series data.",
}
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

    - [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/): Create lagged copies of columns
    - [`RollingStatisticsTransformer`](/pages/api/generated/yohou.preprocessing.window.RollingStatisticsTransformer/): Rolling mean, std, min, max, etc.
    - [`SlidingWindowFunctionTransformer`](/pages/api/generated/yohou.preprocessing.window.SlidingWindowFunctionTransformer/): Custom functions over sliding windows
    - [`ExponentialMovingAverage`](/pages/api/generated/yohou.preprocessing.window.ExponentialMovingAverage/): Exponentially weighted moving average
    - How `observation_horizon` works for stateful transformers
    - Combining transformers with [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/)

    ## Prerequisites

    Basic understanding of feature engineering and time series concepts.
    """)


@app.cell(hide_code=True)
def _():

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
        plot_rolling_statistics,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We load the monthly tourism series from the Monash repository using
    `fetch_tourism_monthly()`, select a single target column, and rename it
    for readability.
    """)


@app.cell
def _(fetch_tourism_monthly, plot_time_series):
    y = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    plot_time_series(y, title="Monthly Tourism (T1)")
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. LagTransformer

    [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/) creates new columns by shifting the original values
    backward by the specified number of steps. For example, `lag=3` creates
    a column whose value at time *t* equals the original value at *t-3*.
    The first `max(lag)` rows are consumed as the transformer's
    `observation_horizon` because they have no valid lagged value.
    """)


@app.cell
def _(LagTransformer, plot_time_series, y):
    lag_tf = LagTransformer(lag=[1, 3, 6, 12])
    lag_tf.fit(y)
    y_lagged = lag_tf.transform(y)
    plot_time_series(y_lagged, title="LagTransformer (lag=[1, 3, 6, 12])")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. RollingStatisticsTransformer

    [`RollingStatisticsTransformer`](/pages/api/generated/yohou.preprocessing.window.RollingStatisticsTransformer/) computes one or more summary statistics
    over a sliding window of the most recent `window_size` observations.
    Supported statistics include `"mean"`, `"std"`, `"min"`, `"max"`,
    `"median"`, `"sum"`, and `"var"`. Multiple statistics can be computed
    in a single call.
    """)


@app.cell
def _(RollingStatisticsTransformer, plot_time_series, y):
    roll_tf = RollingStatisticsTransformer(
        window_size=12,
        statistics=["mean", "std", "min", "max"],
    )
    roll_tf.fit(y)
    y_rolled = roll_tf.transform(y)
    plot_time_series(y_rolled, title="Rolling Statistics (window=12)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_rolling_statistics`](/pages/api/generated/yohou.plotting.exploration.plot_rolling_statistics/) is a dedicated helper that renders the original
    series together with its rolling mean and standard deviation bands,
    giving a quick visual summary of time-varying location and spread.
    """)


@app.cell
def _(plot_rolling_statistics, y):
    plot_rolling_statistics(
        y,
        window_size=12,
        statistics=["mean", "std"],
        title="Rolling Statistics (window=12)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. SlidingWindowFunctionTransformer

    [`SlidingWindowFunctionTransformer`](/pages/api/generated/yohou.preprocessing.window.SlidingWindowFunctionTransformer/) lets you apply any custom function
    over a sliding window. The function receives a polars DataFrame
    containing `window_size` rows and must return a scalar, a dictionary, or
    an array. Below we compute the range (max minus min) over the last 6
    periods as a volatility proxy.
    """)


@app.cell
def _(SlidingWindowFunctionTransformer, plot_time_series, y):
    # Custom: range (max - min) over last 6 periods
    # func receives a polars DataFrame window, must return scalar, dict, or ndarray
    range_tf = SlidingWindowFunctionTransformer(
        func=lambda df: {col: df[col].max() - df[col].min() for col in df.columns if col != "time"},
        window_size=6,
    )
    range_tf.fit(y)
    y_range = range_tf.transform(y)
    _combined = y.rename({"tourists": "original"}).join(
        y_range.rename({"tourists": "range (max-min)"}),
        on="time",
    )
    plot_time_series(_combined, title="SlidingWindowFunction: Range vs Original")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. ExponentialMovingAverage

    [`ExponentialMovingAverage`](/pages/api/generated/yohou.preprocessing.window.ExponentialMovingAverage/) computes an exponentially weighted moving
    average where recent observations receive more weight than older ones.
    The `alpha` parameter (between 0 and 1) controls the decay: higher
    values respond faster to recent changes. Unlike rolling statistics, EWMA
    has `observation_horizon=0` because it is a purely recursive calculation
    that does not discard initial rows.
    """)


@app.cell
def _(ExponentialMovingAverage, plot_time_series, y):
    ema_tf = ExponentialMovingAverage(alpha=0.3)
    ema_tf.fit(y)
    y_ema = ema_tf.transform(y)
    _combined = y.rename({"tourists": "original"}).join(
        y_ema.rename({"tourists_ewma": "EMA (alpha=0.3)"}),
        on="time",
    )
    plot_time_series(_combined, title="Exponential Moving Average vs Original")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Combining Transformers with FeatureUnion

    [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/) runs multiple transformers in parallel on the same input
    and concatenates their outputs horizontally. The combined
    `observation_horizon` equals the maximum across all transformers, so
    early rows that any single transformer cannot fill are dropped.
    """)


@app.cell
def _(
    ExponentialMovingAverage,
    FeatureUnion,
    LagTransformer,
    RollingStatisticsTransformer,
    plot_time_series,
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
    plot_time_series(y_combined, title="FeatureUnion: All Window Features Combined")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/): Creates lagged columns; `observation_horizon = max(lag)`
    - [`RollingStatisticsTransformer`](/pages/api/generated/yohou.preprocessing.window.RollingStatisticsTransformer/): Computes rolling stats; multiple statistics at once
    - [`SlidingWindowFunctionTransformer`](/pages/api/generated/yohou.preprocessing.window.SlidingWindowFunctionTransformer/): Any custom function over sliding windows
    - [`ExponentialMovingAverage`](/pages/api/generated/yohou.preprocessing.window.ExponentialMovingAverage/): EWMA smoothing; `observation_horizon = 0` (stateless)
    - [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/) combines transformers in parallel for rich feature engineering
    - Stateful transformers produce nulls for the first `observation_horizon` rows
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Imputation & outliers**: See [`data_cleaning.py`](/examples/preprocessing/data_cleaning/) for handling nulls and outliers
    - **Resampling**: See [`resampling.py`](/examples/preprocessing/resampling/) for frequency changes
    - **Using in forecasters**: Pass to `feature_transformer` in [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/)
    """)


if __name__ == "__main__":
    app.run()
