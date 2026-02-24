# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Stationarity Transformers: Differencing, Log, BoxCox, and Returns.

Demonstrates transformations for making time series stationary.
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stationarity Transformers

    Many forecasting methods assume the time series is **stationary** (constant mean
    and variance). Yohou provides invertible transformers to stabilize variance
    and remove trends.

    ## What You'll Learn

    - `LogTransformer` / `BoxCoxTransformer`: Stabilize variance
    - `SeasonalDifferencing`: Remove seasonal patterns
    - `SeasonalLogDifferencing`: Combined log + differencing
    - `SeasonalReturn` / `AbsoluteSeasonalReturn`: Percentage and absolute changes
    - `ASinhTransformer`: Robust symmetric transformation
    - Inverse transforms to recover original scale

    ## Prerequisites

    Understanding of stationarity concepts in time series.
    """)

@app.cell(hide_code=True)
def _():
    import polars as pl

    from yohou.datasets import fetch_tourism_monthly
    from yohou.plotting import plot_time_series
    from yohou.stationarity import (
        AbsoluteSeasonalReturn,
        ASinhTransformer,
        BoxCoxTransformer,
        LogTransformer,
        SeasonalDifferencing,
        SeasonalLogDifferencing,
        SeasonalReturn,
    )

    return (
        ASinhTransformer,
        AbsoluteSeasonalReturn,
        BoxCoxTransformer,
        LogTransformer,
        SeasonalDifferencing,
        SeasonalLogDifferencing,
        SeasonalReturn,
        fetch_tourism_monthly,
        pl,
        plot_time_series,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    Monthly Tourism has seasonality and an upward trend,
    clearly non-stationary.
    """)

@app.cell
def _(fetch_tourism_monthly, plot_time_series):
    y = (
        fetch_tourism_monthly()
        .frame.select("time", "T1__tourists").drop_nulls()
        .rename({"T1__tourists": "tourists"})
    )
    plot_time_series(y, title="Monthly Tourism (Non-Stationary)")
    return (y,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. LogTransformer

    Takes the natural log to stabilize multiplicative variance.
    Stateless and invertible.
    """)

@app.cell
def _(LogTransformer, y):
    log_tf = LogTransformer(offset=0.0)
    log_tf.fit(y)
    y_log = log_tf.transform(y)

    data_col = [c for c in y_log.columns if c != "time"][0]
    print(f"observation_horizon: {log_tf.observation_horizon}")
    print(f"Original range: [{y['tourists'].min()}, {y['tourists'].max()}]")
    print(f"Log range: [{y_log[data_col].min():.2f}, {y_log[data_col].max():.2f}]")

    # Verify inverse transform recovers original (X_p=None for stateless transforms)
    y_recovered = log_tf.inverse_transform(y_log, X_p=None)
    rec_col = [c for c in y_recovered.columns if c != "time"][0]
    log_inv_err = (y["tourists"] - y_recovered[rec_col]).abs().max()
    print(f"Max inverse error: {log_inv_err:.10f}")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. BoxCoxTransformer

    Generalization of LogTransformer. `lmbda=0` is log, `lmbda=1` is identity,
    `lmbda=0.5` is square root.
    """)

@app.cell
def _(BoxCoxTransformer, y):
    for _lmbda in [0.0, 0.25, 0.5, 1.0]:
        _bc = BoxCoxTransformer(lmbda=_lmbda, offset=0.0)
        _bc.fit(y)
        _y_bc = _bc.transform(y)
        _data_col = [c for c in _y_bc.columns if c != "time"][0]
        _rng = _y_bc[_data_col].max() - _y_bc[_data_col].min()
        print(f"lmbda={_lmbda:.2f}  range={_rng:.2f}")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. SeasonalDifferencing

    Removes seasonal patterns by computing `y_t - y_{t-s}`.
    Stateful: `observation_horizon = seasonality`.
    """)

@app.cell
def _(SeasonalDifferencing, y):
    diff_tf = SeasonalDifferencing(seasonality=12)
    diff_tf.fit(y)
    y_diff = diff_tf.transform(y)

    print(f"observation_horizon: {diff_tf.observation_horizon}")
    print(f"Original: {len(y)} rows → Differenced: {len(y_diff)} rows")
    y_diff.head()
    return diff_tf, y_diff

@app.cell
def _(diff_tf, y, y_diff):
    # Inverse transform requires X_p (the observation_horizon prefix from original data)
    y_prefix = y.head(diff_tf.observation_horizon)
    y_inv = diff_tf.inverse_transform(y_diff, X_p=y_prefix)
    _inv_col = [c for c in y_inv.columns if c != "time"][0]
    diff_inv_err = (y["tourists"].tail(len(y_inv)) - y_inv[_inv_col]).abs().max()
    print(f"SeasonalDifferencing inverse error: {diff_inv_err:.10f}")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. SeasonalLogDifferencing

    Combines log transform and seasonal differencing in one step:
    `log(y_t) - log(y_{t-s})`. Particularly effective for multiplicative seasonality.
    """)

@app.cell
def _(SeasonalLogDifferencing, plot_time_series, y):
    sld = SeasonalLogDifferencing(seasonality=12, offset=0.0)
    sld.fit(y)
    y_sld = sld.transform(y)

    print(f"observation_horizon: {sld.observation_horizon}")
    plot_time_series(y_sld, title="Seasonal Log Differenced (should look stationary)")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Seasonal Returns

    - `SeasonalReturn`: Percentage change `(y_t / y_{t-s}) - 1`
    - `AbsoluteSeasonalReturn`: Absolute change `y_t - y_{t-s}`

    Both are invertible and stateful.
    """)

@app.cell
def _(AbsoluteSeasonalReturn, SeasonalReturn, y):
    sr = SeasonalReturn(seasonality=12, offset=0.0)
    sr.fit(y)
    y_sr = sr.transform(y)
    _sr_col = [c for c in y_sr.columns if c != "time"][0]
    print(f"SeasonalReturn (first 5 values): {y_sr[_sr_col].head(5).to_list()}")

    asr = AbsoluteSeasonalReturn(seasonality=12)
    asr.fit(y)
    y_asr = asr.transform(y)
    _asr_col = [c for c in y_asr.columns if c != "time"][0]
    print(f"AbsoluteSeasonalReturn (first 5 values): {y_asr[_asr_col].head(5).to_list()}")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. ASinhTransformer

    Inverse hyperbolic sine: `asinh((x - median) / MAD * scale)`.
    Robust, symmetric, works with zero and negative values.
    """)

@app.cell
def _(ASinhTransformer, y):
    asinh_tf = ASinhTransformer(scale=1.4826)
    asinh_tf.fit(y)
    y_asinh = asinh_tf.transform(y)

    _asinh_col = [c for c in y_asinh.columns if c != "time"][0]
    print(f"observation_horizon: {asinh_tf.observation_horizon}")
    print(f"ASinh range: [{y_asinh[_asinh_col].min():.2f}, {y_asinh[_asinh_col].max():.2f}]")

    # Verify invertibility (X_p=None for stateless)
    y_back = asinh_tf.inverse_transform(y_asinh, X_p=None)
    _back_col = [c for c in y_back.columns if c != "time"][0]
    asinh_inv_err = (y["tourists"] - y_back[_back_col]).abs().max()
    print(f"Inverse error: {asinh_inv_err:.10f}")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Comparison Summary

    We bring together the results from all transforms to compare their effectiveness at inducing stationarity.
    """)

@app.cell
def _(
    ASinhTransformer,
    BoxCoxTransformer,
    LogTransformer,
    SeasonalDifferencing,
    SeasonalLogDifferencing,
    y,
):
    transformers = [
        ("Log", LogTransformer(offset=0.0)),
        ("BoxCox(0.5)", BoxCoxTransformer(lmbda=0.5)),
        ("SeasonalDiff(12)", SeasonalDifferencing(seasonality=12)),
        ("SeasonalLogDiff(12)", SeasonalLogDifferencing(seasonality=12)),
        ("ASinh", ASinhTransformer()),
    ]

    print(f"{'Transformer':<22s}  {'obs_horizon':>11s}  {'invertible':>10s}")
    print("-" * 48)
    for _name, _tf in transformers:
        _tf.fit(y)
        _y_t = _tf.transform(y)
        _has_inv = hasattr(_tf, "inverse_transform")
        print(f"{_name:<22s}  {_tf.observation_horizon:>11d}  {str(_has_inv):>10s}")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Variance stabilization**: `LogTransformer`, `BoxCoxTransformer`, `ASinhTransformer`
    - **Seasonal removal**: `SeasonalDifferencing`, `SeasonalLogDifferencing`
    - **Returns**: `SeasonalReturn` (percentage), `AbsoluteSeasonalReturn` (absolute)
    - All are **invertible**, critical for `target_transformer` in forecasters
    - Stateful transformers have `observation_horizon > 0` (first rows are lost)
    - Combine with `target_transformer` in `PointReductionForecaster` for automatic inverse
    """)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Decomposition**: See `decomposition.py` for trend + seasonality forecasting
    - **Using as target_transformer**: See `point/reduction_forecaster.py`
    - **Preprocessing**: See `preprocessing/` for data cleaning and feature engineering
    """)

if __name__ == "__main__":
    app.run()
