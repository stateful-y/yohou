# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Advanced Resampling",
    "description": "Advanced Downsampler and Upsampler usage with configurable aggregation (mean/sum/min/max) and interpolation (linear/nearest/forward) strategies.",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Advanced Resampling

    Yohou provides [`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/) and [`Upsampler`](/pages/api/generated/yohou.preprocessing.resampling.Upsampler/) transformers for
    changing the temporal resolution of time series data.

    ## What You'll Learn

    - [`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/): Aggregate to lower frequency (mean, sum, min, max, median)
    - [`Upsampler`](/pages/api/generated/yohou.preprocessing.resampling.Upsampler/): Interpolate to higher frequency (linear, nearest, forward, backward)
    - Boundary / label settings (`closed`, `label`)
    - Combining downsampling and upsampling for round-trip tests
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

    We load a two-week slice of the half-hourly electricity demand dataset.
    Two columns are kept (Victorian and New South Wales demand) to show that
    [`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/) and [`Upsampler`](/pages/api/generated/yohou.preprocessing.resampling.Upsampler/) handle multivariate DataFrames correctly.
    """)


@app.cell
def _(fetch_electricity_demand, pl, plot_time_series):
    elec = fetch_electricity_demand().frame
    # Take a 2-week subset (30-min intervals = 672 rows)
    elec_subset = elec.head(672).select(
        "time",
        pl.col("vic__demand").alias("demand"),
        pl.col("nsw__demand").alias("nsw_demand"),
    )
    plot_time_series(elec_subset, title="Half-Hourly Electricity Demand (2 weeks)")
    return (elec_subset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Downsampler: Mean Aggregation to Hourly

    [`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/) reduces the frequency by applying an aggregation function
    to each interval bin. Setting `interval="1h"` and `aggregation="mean"`
    averages every pair of 30-minute readings into one hourly value. The
    `closed` and `label` parameters control which boundary is inclusive and
    which timestamp represents each bin.
    """)


@app.cell
def _(Downsampler, elec_subset, plot_time_series):
    ds_mean = Downsampler(interval="1h", aggregation="mean", closed="left", label="left")
    ds_mean.fit(elec_subset)
    elec_hourly = ds_mean.transform(elec_subset)
    plot_time_series(elec_hourly.head(168), title="Hourly Mean: First Week")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Different Aggregation Methods

    The chart below overlays five aggregation strategies applied at daily
    resolution on a single plot. Notice how `sum` produces much larger values
    (total demand over the day), while `min` and `max` capture the daily
    extremes.
    """)


@app.cell
def _(Downsampler, elec_subset, plot_time_series):
    _agg_dfs = {}
    for _agg in ["mean", "sum", "min", "max", "median"]:
        _ds = Downsampler(interval="1d", aggregation=_agg)
        _ds.fit(elec_subset)
        _agg_dfs[_agg] = _ds.transform(elec_subset).rename({"demand": _agg, "nsw_demand": f"nsw_{_agg}"})

    _combined = _agg_dfs["mean"]
    for _agg in ["sum", "min", "max", "median"]:
        _combined = _combined.join(_agg_dfs[_agg], on="time")

    # Plot only 'demand' aggregations (exclude nsw for clarity)
    plot_time_series(
        _combined.select("time", "mean", "sum", "min", "max", "median"),
        title="Daily Downsampling: Aggregation Comparison (VIC demand)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Downsampler: Boundary Settings

    `closed` controls which side of the interval boundary is inclusive.
    `label` controls which boundary timestamp represents the bin.
    """)


@app.cell
def _(Downsampler, elec_subset, plot_time_series):
    _ds_left = Downsampler(interval="6h", aggregation="mean", closed="left", label="left")
    _ds_left.fit(elec_subset)
    _out_left = _ds_left.transform(elec_subset)

    _ds_right = Downsampler(interval="6h", aggregation="mean", closed="right", label="right")
    _ds_right.fit(elec_subset)
    _out_right = _ds_right.transform(elec_subset)

    # Overlay left vs right boundary to show timestamp shift
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
    ## 5. Upsampler: Linear Interpolation

    [`Upsampler`](/pages/api/generated/yohou.preprocessing.resampling.Upsampler/) increases the frequency by inserting new timestamps and
    interpolating values between existing observations. Here we first
    downsample to hourly, then upsample back to 30-minute resolution with
    linear interpolation to show the round-trip behaviour.
    """)


@app.cell
def _(Downsampler, Upsampler, elec_subset, plot_time_series):
    # First downsample to hourly, then upsample back to 30-min
    _ds_tmp = Downsampler(interval="1h", aggregation="mean")
    _ds_tmp.fit(elec_subset)
    _hourly = _ds_tmp.transform(elec_subset)

    us_linear = Upsampler(interval="30m", interpolation="linear")
    us_linear.fit(_hourly)
    elec_upsampled = us_linear.transform(_hourly)
    plot_time_series(
        elec_upsampled.select("time", "demand").head(96),
        title="Upsampled (Linear): First 2 Days",
    )
    return (elec_upsampled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Different Interpolation Methods

    The chart below overlays four interpolation strategies when upsampling from
    2-hourly to 30-minute resolution. `linear` produces smooth transitions,
    `nearest` creates step changes at midpoints, while `forward` and
    `backward` hold the previous or next value constant.
    """)


@app.cell
def _(Downsampler, Upsampler, elec_subset, plot_time_series):
    _ds_tmp2 = Downsampler(interval="2h", aggregation="mean")
    _ds_tmp2.fit(elec_subset)
    _coarse = _ds_tmp2.transform(elec_subset)

    _interp_dfs = {}
    for _method in ["linear", "nearest", "forward", "backward"]:
        _us = Upsampler(interval="30m", interpolation=_method)
        _us.fit(_coarse)
        _interp_dfs[_method] = _us.transform(_coarse).select("time", "demand").rename({"demand": _method})

    _combined = _interp_dfs["linear"]
    for _method in ["nearest", "forward", "backward"]:
        _combined = _combined.join(_interp_dfs[_method], on="time")

    plot_time_series(
        _combined.head(48),
        title="Upsampling: Interpolation Comparison (first day)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Round-Trip: Downsample Then Upsample

    Compare original vs down-then-up to see information loss.
    """)


@app.cell
def _(elec_subset, elec_upsampled, plot_time_series):
    # Align lengths for comparison
    _n = min(len(elec_subset), len(elec_upsampled))
    _combined = (
        elec_subset
        .head(_n)
        .select("time", "demand")
        .rename({"demand": "original"})
        .join(
            elec_upsampled.head(_n).select("time", "demand").rename({"demand": "round-trip"}),
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

    - **[`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/)**: Reduces frequency with configurable aggregation (mean, sum, min, max, median)
    - **[`Upsampler`](/pages/api/generated/yohou.preprocessing.resampling.Upsampler/)**: Increases frequency with interpolation (linear, nearest, forward, backward)
    - **`closed`/`label`**: Control boundary inclusion and timestamp labelling
    - **Round-trip**: Downsampling loses information; upsampling cannot fully recover it
    - **Both are stateless** transformers (no observation horizon)

    ## Next Steps

    - **Resampling basics**: See [`examples/preprocessing/resampling.py`](/examples/preprocessing/resampling/)
    - **Signal processing**: See [`examples/preprocessing/signal_processing.py`](/examples/preprocessing/signal_processing/)
    - **Window transformers**: See [`examples/preprocessing/window_transformers.py`](/examples/preprocessing/window_transformers/)
    """)


if __name__ == "__main__":
    app.run()
