"""Signal Processing Transformers.

Demonstrates NumericalFilter, NumericalIntegrator, and NumericalDifferentiator
on high-frequency electricity demand data.
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
    # Signal Processing Transformers

    Yohou provides signal processing transformers for smoothing, trend
    extraction, rate-of-change analysis, and cumulative integration.
    These work best on high-frequency data.

    ## What You'll Learn

    - `NumericalFilter`: Butterworth/Chebyshev/Bessel low/highpass filters
    - `NumericalDifferentiator`: Rate-of-change estimation
    - `NumericalIntegrator`: Cumulative integration
    - Chaining filters for bandpass-like behaviour
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl

    from yohou.datasets import fetch_electricity_demand
    from yohou.plotting import plot_time_series
    from yohou.preprocessing import NumericalDifferentiator, NumericalFilter, NumericalIntegrator

    return (
        NumericalDifferentiator,
        NumericalFilter,
        NumericalIntegrator,
        fetch_electricity_demand,
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
def _(fetch_electricity_demand, mo, pl):
    elec = fetch_electricity_demand().frame
    # Use a manageable subset (first 2 weeks = 672 half-hour periods)
    elec_subset = elec.head(672).select(
        "time", pl.col("vic__demand").alias("demand")
    )
    mo.md(
        f"**Electricity Demand** (30-min intervals)\n\n"
        f"Full dataset: {len(elec)} rows, using first {len(elec_subset)} rows\n\n"
        f"Columns: {elec_subset.columns}"
    )
    return elec, elec_subset


@app.cell
def _(elec_subset, plot_time_series):
    plot_time_series(elec_subset, title="Electricity Demand: Raw (2 Weeks)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Low-Pass Filter (Smoothing)

    A Butterworth low-pass filter removes high-frequency noise,
    revealing the underlying demand trend.
    """)
    return


@app.cell
def _(NumericalFilter, elec_subset, pl, plot_time_series):
    lp_filter = NumericalFilter(
        design="butterworth",
        mode="lowpass",
        order=4,
        cutoff_frequency=0.05,
    )
    lp_filter.fit(elec_subset)
    demand_smooth = lp_filter.transform(elec_subset)

    # Combine for visual comparison
    _combined = elec_subset.join(
        demand_smooth.rename({"demand": "demand_smooth"}),
        on="time",
    )
    plot_time_series(_combined, title="Low-Pass Filter (cutoff=0.05)")
    return demand_smooth, lp_filter


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. High-Pass Filter (Residual Extraction)

    A high-pass filter keeps only the fast-changing component, useful
    for isolating noise or short-term fluctuations.
    """)
    return


@app.cell
def _(NumericalFilter, elec_subset, pl, plot_time_series):
    hp_filter = NumericalFilter(
        design="butterworth",
        mode="highpass",
        order=4,
        cutoff_frequency=0.05,
    )
    hp_filter.fit(elec_subset)
    demand_residual = hp_filter.transform(elec_subset)

    _combined_hp = elec_subset.join(
        demand_residual.rename({"demand": "demand_highpass"}),
        on="time",
    )
    plot_time_series(_combined_hp, title="High-Pass Filter (cutoff=0.05)")
    return demand_residual, hp_filter


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Different Filter Designs

    Compare Butterworth, Chebyshev Type 1, and Bessel filters.
    """)
    return


@app.cell
def _(NumericalFilter, elec_subset, pl, plot_time_series):
    _designs = {
        "butterworth": NumericalFilter(design="butterworth", mode="lowpass", order=4, cutoff_frequency=0.05),
        "chebyshev1": NumericalFilter(
            design="chebyshev1", mode="lowpass", order=4, cutoff_frequency=0.05, passband_ripple=1.0,
        ),
        "bessel": NumericalFilter(design="bessel", mode="lowpass", order=4, cutoff_frequency=0.05),
    }

    _result = elec_subset.select("time")
    for _name, _filt in _designs.items():
        _filt.fit(elec_subset)
        _out = _filt.transform(elec_subset)
        _result = _result.with_columns(_out["demand"].alias(f"demand_{_name}"))

    plot_time_series(
        _result.head(200),
        title="Filter Design Comparison (First 200 Samples)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. NumericalDifferentiator: Rate of Change

    Estimate the derivative (rate of change) of the demand signal.
    """)
    return


@app.cell
def _(NumericalDifferentiator, elec_subset, mo, plot_time_series):
    diff = NumericalDifferentiator(order=1)
    diff.fit(elec_subset)
    demand_rate = diff.transform(elec_subset)

    mo.md(
        f"**Differentiator output**: {demand_rate.columns}\n\n"
        f"**Sampling interval**: {diff.interval_} ({diff.sampling_interval_:.0f} seconds)"
    )
    return demand_rate, diff


@app.cell
def _(demand_rate, plot_time_series):
    plot_time_series(demand_rate.head(200), title="Rate of Change: First 200 Samples")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. NumericalIntegrator: Cumulative Integration

    Integrate the demand signal to get cumulative energy consumption.
    """)
    return


@app.cell
def _(NumericalIntegrator, elec_subset, mo, plot_time_series):
    integ = NumericalIntegrator(method="cumulative_trapezoid")
    integ.fit(elec_subset)
    demand_cumulative = integ.transform(elec_subset)

    mo.md(
        f"**Integrator output columns**: {demand_cumulative.columns}\n\n"
        f"**Sampling interval**: {integ.interval_}"
    )
    return demand_cumulative, integ


@app.cell
def _(demand_cumulative, plot_time_series):
    plot_time_series(demand_cumulative, title="Cumulative Integration: Total Energy")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Chaining: Smooth Then Differentiate

    Apply a low-pass filter first to remove noise, then differentiate
    for a cleaner rate-of-change estimate.
    """)
    return


@app.cell
def _(NumericalDifferentiator, NumericalFilter, elec_subset, pl, plot_time_series):
    _lp = NumericalFilter(design="butterworth", mode="lowpass", order=4, cutoff_frequency=0.05)
    _lp.fit(elec_subset)
    _smooth = _lp.transform(elec_subset)

    _diff2 = NumericalDifferentiator(order=1)
    _diff2.fit(_smooth)
    _rate_smooth = _diff2.transform(_smooth)

    _diff_raw = NumericalDifferentiator(order=1)
    _diff_raw.fit(elec_subset)
    _rate_raw = _diff_raw.transform(elec_subset)

    _compare = _rate_smooth.join(
        _rate_raw.rename({"demand_differentiated": "demand_raw_rate"}),
        on="time",
    ).rename({"demand_differentiated": "demand_smooth_rate"})

    plot_time_series(
        _compare.head(200),
        title="Rate of Change: Raw vs Smoothed",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **`NumericalFilter`**: Butterworth, Chebyshev, Bessel filters in lowpass/highpass/bandpass/bandstop modes
    - **`NumericalDifferentiator`**: Estimates rate of change using `np.gradient`
    - **`NumericalIntegrator`**: Cumulative trapezoidal or Simpson integration
    - **Filter order**: Higher order = sharper cutoff but more phase distortion
    - **Cutoff frequency**: 0-1 as fraction of Nyquist frequency
    - **Chain transformers**: Smooth first, then differentiate for cleaner signals

    ## Next Steps

    - **Resampling**: See `examples/preprocessing/resampling_advanced.py`
    - **Window transformers**: See `examples/preprocessing/window_transformers.py`
    - **Stationarity transforms**: See `examples/stationarity/stationarity_transforms.py`
    """)
    return


if __name__ == "__main__":
    app.run()
