# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Resampling",
    "description": "Demonstrate Downsampler and Upsampler for changing time series frequency, including multivariate support, boundary settings, and round-trip information loss.",
}
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
    or upsampling (e.g., monthly → weekly). Yohou provides [`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/) and [`Upsampler`](/pages/api/generated/yohou.preprocessing.resampling.Upsampler/).

    ## What You'll Learn

    - [`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/): Aggregate to lower frequency (mean, sum, min, max, etc.)
    - [`Upsampler`](/pages/api/generated/yohou.preprocessing.resampling.Upsampler/): Interpolate to higher frequency
    - Multivariate DataFrame support
    - Controlling `closed` and `label` boundary settings
    - Round-trip: Downsample then upsample to see information loss

    ## Prerequisites

    None.
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl

    from yohou.datasets import fetch_electricity_demand
    from yohou.plotting import plot_time_series
    from yohou.preprocessing import Downsampler, Upsampler

    return (
        Downsampler,
        Upsampler,
        fetch_electricity_demand,
        pl,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load High-Frequency Data

    The Electricity Demand dataset contains 30-minute interval readings from
    Australian states. We select the Victorian demand column and take a
    30-day subset (about 1 440 observations) for a manageable demonstration.
    """)


@app.cell
def _(fetch_electricity_demand, pl, plot_time_series):
    raw = fetch_electricity_demand().frame.select("time", pl.col("vic__demand").alias("demand"))

    # Use a subset for speed
    y_hf = raw.head(48 * 30)  # ~30 days of 30-min data
    plot_time_series(y_hf, title="Half-Hourly Electricity Demand (30 days)")
    return (y_hf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Downsampler

    [`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/) aggregates a time series to a coarser frequency. You
    specify the target `interval` (e.g. `"1d"` for daily) and the
    `aggregation` method: `"mean"`, `"sum"`, `"min"`, `"max"`, `"first"`,
    `"last"`, or `"median"`. Below we produce daily mean and daily sum
    versions of the half-hourly demand data.
    """)


@app.cell
def _(Downsampler, plot_time_series, y_hf):
    daily_mean = Downsampler(interval="1d", aggregation="mean")
    daily_mean.fit(y_hf)
    y_daily_mean = daily_mean.transform(y_hf)
    plot_time_series(y_daily_mean, title="Hourly -> Daily (mean aggregation)")
    return (y_daily_mean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparing Aggregations

    Different aggregation strategies summarise each interval differently.
    Joining the results into one DataFrame lets us overlay mean, min, max,
    and median on a single [`plot_time_series`](/pages/api/generated/yohou.plotting.exploration.plot_time_series/) chart so the differences are
    immediately visible.
    """)


@app.cell
def _(Downsampler, plot_time_series, y_hf):
    _agg_dfs = {}
    for _agg in ["mean", "min", "max", "median"]:
        _ds = Downsampler(interval="1d", aggregation=_agg)
        _ds.fit(y_hf)
        _agg_dfs[_agg] = _ds.transform(y_hf).rename({"demand": _agg})

    _combined = _agg_dfs["mean"]
    for _agg in ["min", "max", "median"]:
        _combined = _combined.join(_agg_dfs[_agg], on="time")
    plot_time_series(_combined, title="Daily Downsampling: Aggregation Comparison")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Upsampler

    [`Upsampler`](/pages/api/generated/yohou.preprocessing.resampling.Upsampler/) increases the frequency by interpolating between existing
    observations. The `interpolation` parameter controls the method used to
    fill in the new time steps: `"linear"` draws straight lines between
    neighbours, `"nearest"` copies the closest value, and `"forward"` /
    `"backward"` carry the previous or next value.
    """)


@app.cell
def _(Upsampler, plot_time_series, y_daily_mean):
    hourly = Upsampler(interval="1h", interpolation="linear")
    hourly.fit(y_daily_mean)
    y_hourly = hourly.transform(y_daily_mean)
    plot_time_series(y_hourly, title="Daily → Hourly (linear interpolation)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparing Interpolation Methods

    Each interpolation strategy fills in new time steps differently. We
    overlay `linear`, `nearest`, `forward`, and `backward` on one chart
    so you can see how the shape and step patterns compare.
    """)


@app.cell
def _(Upsampler, plot_time_series, y_daily_mean):
    _interp_dfs = {}
    for _method in ["linear", "nearest", "forward", "backward"]:
        _up = Upsampler(interval="1h", interpolation=_method)
        _up.fit(y_daily_mean)
        _interp_dfs[_method] = _up.transform(y_daily_mean).rename({"demand": _method})

    _combined = _interp_dfs["linear"]
    for _method in ["nearest", "forward", "backward"]:
        _combined = _combined.join(_interp_dfs[_method], on="time")
    plot_time_series(_combined, title="Hourly Upsampling: Interpolation Comparison")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Downsampling to Weekly

    [`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/) accepts any polars duration string as the `interval`
    parameter. Here we aggregate the half-hourly data to weekly mean values
    and visualise the result with [`plot_time_series`](/pages/api/generated/yohou.plotting.exploration.plot_time_series/).
    """)


@app.cell
def _(Downsampler, plot_time_series, y_hf):
    weekly = Downsampler(interval="1w", aggregation="mean")
    weekly.fit(y_hf)
    y_weekly = weekly.transform(y_hf)
    plot_time_series(y_weekly, title="Weekly Mean Electricity Demand")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Multivariate Resampling

    [`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/) and [`Upsampler`](/pages/api/generated/yohou.preprocessing.resampling.Upsampler/) handle multivariate DataFrames
    correctly: all non-time columns are processed with the same settings.
    Here we load two demand columns (Victoria and New South Wales) and
    downsample both to daily resolution in a single call.
    """)


@app.cell
def _(Downsampler, fetch_electricity_demand, pl, plot_time_series):
    _elec_mv = (
        fetch_electricity_demand()
        .frame.head(48 * 14)
        .select(
            "time",
            pl.col("vic__demand").alias("demand"),
            pl.col("nsw__demand").alias("nsw_demand"),
        )
    )
    _ds_mv = Downsampler(interval="1d", aggregation="mean")
    _ds_mv.fit(_elec_mv)
    _y_daily_mv = _ds_mv.transform(_elec_mv)
    plot_time_series(_y_daily_mv, title="Daily Mean: Multivariate (VIC + NSW)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Boundary Settings

    The `closed` parameter controls which side of the interval boundary is
    inclusive, and `label` controls which boundary timestamp represents the
    bin. The chart below overlays 6-hour downsampling with `left/left` vs
    `right/right` settings so you can see the timestamp shift.
    """)


@app.cell
def _(Downsampler, plot_time_series, y_hf):
    _ds_left = Downsampler(interval="6h", aggregation="mean", closed="left", label="left")
    _ds_left.fit(y_hf)
    _out_left = _ds_left.transform(y_hf)

    _ds_right = Downsampler(interval="6h", aggregation="mean", closed="right", label="right")
    _ds_right.fit(y_hf)
    _out_right = _ds_right.transform(y_hf)

    _combined = (
        _out_left
        .select("time", "demand")
        .rename({"demand": "left/left"})
        .join(
            _out_right.select("time", "demand").rename({"demand": "right/right"}),
            on="time",
            how="full",
            coalesce=True,
        )
        .sort("time")
    )
    plot_time_series(
        _combined.head(28),
        title="6h Downsampling: Boundary Settings (first 4 days)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Round-Trip: Downsample Then Upsample

    Downsampling discards within-interval variation, so upsampling back to
    the original frequency cannot recover the lost detail. Here we
    downsample to daily mean and then upsample back to half-hourly with
    linear interpolation. Overlaying the original and round-trip versions
    shows the amount of information lost.
    """)


@app.cell
def _(Upsampler, plot_time_series, y_daily_mean, y_hf):
    _us_rt = Upsampler(interval="30m", interpolation="linear")
    _us_rt.fit(y_daily_mean)
    _y_roundtrip = _us_rt.transform(y_daily_mean)
    _n = min(len(y_hf), len(_y_roundtrip))
    _combined = (
        y_hf
        .head(_n)
        .rename({"demand": "original"})
        .join(
            _y_roundtrip.head(_n).rename({"demand": "round-trip"}),
            on="time",
        )
    )
    plot_time_series(
        _combined.head(96),
        title="Round-Trip: Original vs Down-then-Up (first 2 days)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - [`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/): Reduces frequency with aggregation (mean, sum, etc.)
    - [`Upsampler`](/pages/api/generated/yohou.preprocessing.resampling.Upsampler/): Increases frequency with interpolation (linear, nearest, etc.)
    - Both are stateless transformers (`observation_horizon = 0`)
    - Both handle **multivariate** DataFrames: all non-time columns are processed identically
    - **`closed`/`label`** control boundary inclusion and timestamp labelling
    - **Round-trip** (downsample then upsample) always loses within-interval variation
    - `interval` uses polars duration strings: `"1h"`, `"1d"`, `"1w"`, `"1mo"`
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Window transformers**: See [`window_transformers.py`](/examples/preprocessing/window_transformers/) for feature engineering
    - **Data cleaning**: See [`data_cleaning.py`](/examples/preprocessing/data_cleaning/) for imputation and outliers
    - **Signal processing**: See [`signal_processing.py`](/examples/preprocessing/signal_processing/) for filters and differentiators
    - **In forecasters**: Use resampled data as input to any forecaster
    """)


if __name__ == "__main__":
    app.run()
