"""Advanced Resampling.

Demonstrates Downsampler and Upsampler with different aggregation
strategies, boundary settings, and interpolation methods on
high-frequency data.
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
async def _():
    import sys as _sys

    if "pyodide" in _sys.modules:
        import micropip

        await micropip.install(["plotly", "scikit-learn", "yohou"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Advanced Resampling

    Yohou provides `Downsampler` and `Upsampler` transformers for
    changing the temporal resolution of time series data.

    ## What You'll Learn

    - `Downsampler`: Aggregate to lower frequency (mean, sum, min, max, median)
    - `Upsampler`: Interpolate to higher frequency (linear, nearest, forward, backward)
    - Boundary / label settings (`closed`, `label`)
    - Combining downsampling and upsampling for round-trip tests
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl

    from yohou.datasets import load_vic_electricity
    from yohou.plotting import plot_time_series
    from yohou.preprocessing import Downsampler, Upsampler

    return (
        Downsampler,
        Upsampler,
        load_vic_electricity,
        pl,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load High-Frequency Data
    """)
    return


@app.cell
def _(load_vic_electricity, mo):
    elec = load_vic_electricity()
    # Take a 2-week subset (30-min intervals = 672 rows)
    elec_subset = elec.head(672).select("time", "Demand", "Temperature")
    mo.md(
        f"**Victoria Electricity**: 30-min intervals\n\n"
        f"**Subset**: {len(elec_subset)} rows (2 weeks)\n\n"
        f"**Columns**: {elec_subset.columns}"
    )
    return elec, elec_subset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Downsampler: Mean Aggregation to Hourly
    """)
    return


@app.cell
def _(Downsampler, elec_subset, mo, plot_time_series):
    ds_mean = Downsampler(interval="1h", aggregation="mean", closed="left", label="left")
    ds_mean.fit(elec_subset)
    elec_hourly = ds_mean.transform(elec_subset)

    mo.md(
        f"**30-min → hourly (mean)**: {len(elec_subset)} → {len(elec_hourly)} rows"
    )
    return ds_mean, elec_hourly


@app.cell
def _(elec_hourly, plot_time_series):
    plot_time_series(elec_hourly.head(168), title="Hourly Mean: First Week")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Different Aggregation Methods
    """)
    return


@app.cell
def _(Downsampler, elec_subset, mo, pl):
    _methods = ["mean", "sum", "min", "max", "median"]
    _results = elec_subset.select("time")[:1]  # placeholder

    _rows = []
    for _method in _methods:
        _ds = Downsampler(interval="1d", aggregation=_method)
        _ds.fit(elec_subset)
        _out = _ds.transform(elec_subset)
        _demand_range = _out["Demand"]
        _rows.append({
            "Aggregation": _method,
            "Rows": len(_out),
            "Demand Mean": round(float(_demand_range.mean()), 1),
            "Demand Std": round(float(_demand_range.std()), 1),
        })

    mo.ui.table(pl.DataFrame(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Downsampler: Boundary Settings

    `closed` controls which side of the interval boundary is inclusive.
    `label` controls which boundary timestamp represents the bin.
    """)
    return


@app.cell
def _(Downsampler, elec_subset, mo):
    _ds_left = Downsampler(interval="6h", aggregation="mean", closed="left", label="left")
    _ds_left.fit(elec_subset)
    _out_left = _ds_left.transform(elec_subset)

    _ds_right = Downsampler(interval="6h", aggregation="mean", closed="right", label="right")
    _ds_right.fit(elec_subset)
    _out_right = _ds_right.transform(elec_subset)

    mo.md(
        f"**left/left**: first 3 timestamps: {_out_left['time'].head(3).to_list()}\n\n"
        f"**right/right**: first 3 timestamps: {_out_right['time'].head(3).to_list()}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Upsampler: Linear Interpolation
    """)
    return


@app.cell
def _(Downsampler, Upsampler, elec_subset, mo, plot_time_series):
    # First downsample to hourly, then upsample back to 30-min
    _ds_tmp = Downsampler(interval="1h", aggregation="mean")
    _ds_tmp.fit(elec_subset)
    _hourly = _ds_tmp.transform(elec_subset)

    us_linear = Upsampler(interval="30m", interpolation="linear")
    us_linear.fit(_hourly)
    elec_upsampled = us_linear.transform(_hourly)

    mo.md(
        f"**Hourly → 30-min (linear)**: {len(_hourly)} → {len(elec_upsampled)} rows"
    )
    return elec_upsampled, us_linear


@app.cell
def _(elec_upsampled, plot_time_series):
    plot_time_series(
        elec_upsampled.select("time", "Demand").head(96),
        title="Upsampled (Linear): First 2 Days",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Different Interpolation Methods
    """)
    return


@app.cell
def _(Downsampler, Upsampler, elec_subset, mo, pl):
    _ds_tmp2 = Downsampler(interval="2h", aggregation="mean")
    _ds_tmp2.fit(elec_subset)
    _coarse = _ds_tmp2.transform(elec_subset)

    _methods = ["linear", "nearest", "forward", "backward"]
    _rows = []
    for _method in _methods:
        _us = Upsampler(interval="30m", interpolation=_method)
        _us.fit(_coarse)
        _out = _us.transform(_coarse)
        _demand_std = float(_out["Demand"].std())
        _rows.append({
            "Interpolation": _method,
            "Output Rows": len(_out),
            "Demand Std": round(_demand_std, 2),
        })

    mo.ui.table(pl.DataFrame(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Round-Trip: Downsample Then Upsample

    Compare original vs down-then-up to see information loss.
    """)
    return


@app.cell
def _(elec_subset, elec_upsampled, mo, pl):
    # Align lengths for comparison
    _n = min(len(elec_subset), len(elec_upsampled))
    _orig = elec_subset.head(_n)["Demand"]
    _roundtrip = elec_upsampled.head(_n)["Demand"]
    _mae = ((_orig - _roundtrip).abs()).mean()

    mo.md(
        f"**Round-trip MAE** (30min → 1h mean → 30min linear): {float(_mae):.2f}\n\n"
        "Some information is inevitably lost during downsampling. "
        "Higher aggregation intervals lose more detail."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **`Downsampler`**: Reduces frequency with configurable aggregation (mean, sum, min, max, median)
    - **`Upsampler`**: Increases frequency with interpolation (linear, nearest, forward, backward)
    - **`closed`/`label`**: Control boundary inclusion and timestamp labelling
    - **Round-trip**: Downsampling loses information; upsampling cannot fully recover it
    - **Both are stateless** transformers (no observation horizon)

    ## Next Steps

    - **Resampling basics**: See `examples/preprocessing/resampling.py`
    - **Signal processing**: See `examples/preprocessing/signal_processing.py`
    - **Window transformers**: See `examples/preprocessing/window_transformers.py`
    """)
    return


if __name__ == "__main__":
    app.run()
