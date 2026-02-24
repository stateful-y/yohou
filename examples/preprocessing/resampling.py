# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///
"""Resampling: Downsampler and Upsampler.

Demonstrates changing time series frequency with aggregation or interpolation.
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
    # Resampling Time Series

    Time series data often needs frequency changes, downsampling (e.g., hourly → daily)
    or upsampling (e.g., monthly → weekly). Yohou provides `Downsampler` and `Upsampler`.

    ## What You'll Learn

    - `Downsampler`: Aggregate to lower frequency (mean, sum, min, max, etc.)
    - `Upsampler`: Interpolate to higher frequency
    - Controlling `closed` and `label` boundaries
    - When to use each transformer

    ## Prerequisites

    None.
    """)

@app.cell(hide_code=True)
def _():
    import polars as pl

    from yohou.datasets import fetch_electricity_demand
    from yohou.plotting import plot_time_series
    from yohou.preprocessing import Downsampler, Upsampler

    return (Downsampler, Upsampler, fetch_electricity_demand, pl, plot_time_series)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load High-Frequency Data

    The Electricity Demand dataset has 30-minute intervals, ideal for demonstrating downsampling.
    """)

@app.cell
def _(fetch_electricity_demand, pl):
    raw = fetch_electricity_demand().frame.select(
        "time", pl.col("vic__demand").alias("demand")
    )
    print(f"Shape: {raw.shape}")
    print(f"Columns: {raw.columns}")
    print(f"Time range: {raw['time'].min()} to {raw['time'].max()}")

    # Use a subset for speed
    y_hf = raw.head(48 * 30)  # ~30 days of 30-min data
    print(f"Subset: {len(y_hf)} observations")
    return raw, y_hf

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Downsampler

    Aggregate from 30-minute to daily frequency. Choose aggregation method:
    `"mean"`, `"sum"`, `"min"`, `"max"`, `"first"`, `"last"`, `"median"`.
    """)

@app.cell
def _(Downsampler, y_hf):
    daily_mean = Downsampler(interval="1d", aggregation="mean")
    daily_mean.fit(y_hf)
    y_daily_mean = daily_mean.transform(y_hf)

    daily_sum = Downsampler(interval="1d", aggregation="sum")
    daily_sum.fit(y_hf)
    y_daily_sum = daily_sum.transform(y_hf)

    print(f"30-min data: {len(y_hf)} rows → daily mean: {len(y_daily_mean)} rows")
    print(f"First day mean demand: {y_daily_mean['demand'][0]:.1f}")
    print(f"First day total demand: {y_daily_sum['demand'][0]:.1f}")
    return daily_mean, daily_sum, y_daily_mean, y_daily_sum

@app.cell
def _(plot_time_series, y_daily_mean):
    plot_time_series(y_daily_mean, title="Daily Mean Electricity Demand")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparing Aggregations

    Let's compare how different aggregation strategies affect the resampled output.
    """)

@app.cell
def _(Downsampler, y_hf):
    for _agg in ["mean", "sum", "min", "max", "median"]:
        _ds = Downsampler(interval="1d", aggregation=_agg)
        _ds.fit(y_hf)
        _result = _ds.transform(y_hf)
        _first_val = _result["demand"][0]
        print(f"aggregation={_agg:>6s}  first day: {_first_val:.1f}  rows: {len(_result)}")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Upsampler

    Interpolate from daily to hourly (or any higher frequency).
    Methods: `"linear"`, `"nearest"`, `"forward"`, `"backward"`.
    """)

@app.cell
def _(Upsampler, y_daily_mean):
    hourly = Upsampler(interval="1h", interpolation="linear")
    hourly.fit(y_daily_mean)
    y_hourly = hourly.transform(y_daily_mean)

    print(f"Daily: {len(y_daily_mean)} rows → hourly: {len(y_hourly)} rows")
    y_hourly.head(30)
    return hourly, y_hourly

@app.cell
def _(Upsampler, y_daily_mean):
    for _method in ["linear", "nearest", "forward", "backward"]:
        _up = Upsampler(interval="1h", interpolation=_method)
        _up.fit(y_daily_mean)
        _result = _up.transform(y_daily_mean)
        # Show first few hourly values to compare methods
        _vals = _result["demand"].head(5).to_list()
        print(f"interpolation={_method:>10s}  first 5h: {[f'{v:.1f}' for v in _vals]}")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Downsampling to Weekly

    We apply the same downsampling approach at a weekly frequency to illustrate flexible interval selection.
    """)

@app.cell
def _(Downsampler, plot_time_series, y_hf):
    weekly = Downsampler(interval="1w", aggregation="mean")
    weekly.fit(y_hf)
    y_weekly = weekly.transform(y_hf)

    print(f"30-min: {len(y_hf)} → weekly: {len(y_weekly)} rows")
    plot_time_series(y_weekly, title="Weekly Mean Electricity Demand")
    return weekly, y_weekly

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - `Downsampler`: Reduces frequency with aggregation (mean, sum, etc.)
    - `Upsampler`: Increases frequency with interpolation (linear, nearest, etc.)
    - Both are stateless transformers (`observation_horizon = 0`)
    - Use downsampling to reduce noise and computation for high-frequency data
    - Use upsampling when models require uniform higher frequency
    - `interval` uses polars duration strings: `"1h"`, `"1d"`, `"1w"`, `"1mo"`
    """)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Window transformers**: See `window_transformers.py` for feature engineering
    - **Data cleaning**: See `data_cleaning.py` for imputation and outliers
    - **In forecasters**: Use resampled data as input to any forecaster
    """)

if __name__ == "__main__":
    app.run()
