"""Signal Processing -- Spectrum and Phase after Numerical Filtering.

Applies `NumericalFilter` (Butterworth low-pass) to the highly periodic
Victorian electricity Temperature signal and compares the power spectrum
and phase spectrum before and after filtering.

Dataset: vic_electricity (Temperature column -- 30-min periodicity)
Demonstrates: plot_spectrum, plot_phase
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
def _():
    import polars as pl

    from yohou.datasets import load_vic_electricity
    from yohou.plotting import plot_phase, plot_spectrum, plot_time_series
    from yohou.preprocessing import NumericalFilter

    return (
        NumericalFilter,
        load_vic_electricity,
        pl,
        plot_phase,
        plot_spectrum,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Signal Processing

    ## What You'll Learn

    - Applying a `NumericalFilter` (Butterworth low-pass) to a temperature
      signal
    - Comparing raw vs. filtered signals in the time domain
    - Using `plot_spectrum` to see how a low-pass filter removes
      high-frequency noise
    - Using `plot_phase` to examine phase shifts introduced by the filter

    ## The Dataset

    Victorian electricity demand data recorded every **30 minutes** over
    three years. The **Temperature** column contains a strong daily cycle
    (period ≈ 48 samples) plus higher-frequency fluctuations that a
    low-pass filter will suppress.
    """)
    return


@app.cell
def _(load_vic_electricity):
    vic = load_vic_electricity()
    # Take 30 days (1440 half-hours) for readable plots
    vic_short = vic.head(1440)
    return (vic_short,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raw Temperature Signal

    The 30-day window shows a clear daily cycle plus short-term noise.
    """)
    return


@app.cell
def _(plot_time_series, vic_short):
    plot_time_series(
        vic_short,
        columns="Temperature",
        title="Raw Temperature -- 30-day Window",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Applying a Low-Pass Filter

    A **4th-order Butterworth** filter with `cutoff_frequency=0.05`
    (relative to Nyquist) keeps only the slowest oscillations, roughly
    the daily cycle, and removes everything faster.
    """)
    return


@app.cell
def _(NumericalFilter, pl, vic_short):
    lowpass = NumericalFilter(
        design="butterworth",
        mode="lowpass",
        order=1,
        cutoff_frequency=0.05,
    )
    temp_df = vic_short.select("time", "Temperature")
    lowpass.fit(temp_df)
    temp_filtered = lowpass.transform(temp_df)
    # Combine raw and filtered for side-by-side comparison
    combined = temp_df.rename({"Temperature": "Raw"}).join(
        temp_filtered.rename({"Temperature": "Filtered"}),
        on="time",
    )
    return combined, temp_df, temp_filtered


@app.cell
def _(combined, plot_time_series):
    plot_time_series(
        combined,
        columns=["Raw", "Filtered"],
        title="Temperature -- Raw vs. Low-Pass Filtered",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Power Spectrum Comparison

    `plot_spectrum` computes the one-sided power spectral density via FFT.
    The raw signal shows energy at many frequencies; after filtering, only the
    low-frequency band retains power.
    """)
    return


@app.cell
def _(plot_spectrum, temp_df):
    plot_spectrum(
        temp_df,
        columns="Temperature",
        title="Power Spectrum -- Raw Temperature",
    )
    return


@app.cell
def _(plot_spectrum, temp_filtered):
    plot_spectrum(
        temp_filtered,
        columns="Temperature",
        title="Power Spectrum -- After Low-Pass Filter",
    )
    return


@app.cell
def _(plot_spectrum, temp_df):
    plot_spectrum(
        temp_df,
        columns="Temperature",
        log_scale=True,
        title="Power Spectrum -- Raw (Log Scale)",
    )
    return


@app.cell
def _(plot_spectrum, temp_filtered):
    plot_spectrum(
        temp_filtered,
        columns="Temperature",
        log_scale=True,
        title="Power Spectrum -- Filtered (Log Scale)",
    )
    return


@app.cell
def _(plot_spectrum, temp_df):
    plot_spectrum(
        temp_df,
        columns="Temperature",
        title="Power Spectrum -- Raw",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase Spectrum Comparison

    The **phase** angle at each frequency reveals the signal's timing
    structure. `unwrap=True` (default) produces continuous phase;
    `angle_unit="degree"` converts from radians. Notice how the filter
    shifts the phase at higher frequencies that are attenuated.
    """)
    return


@app.cell
def _(plot_phase, temp_df):
    plot_phase(
        temp_df,
        columns="Temperature",
        title="Phase -- Raw Temperature",
    )
    return


@app.cell
def _(plot_phase, temp_filtered):
    plot_phase(
        temp_filtered,
        columns="Temperature",
        title="Phase -- After Low-Pass Filter",
    )
    return


@app.cell
def _(plot_phase, temp_df):
    plot_phase(
        temp_df,
        columns="Temperature",
        angle_unit="degree",
        title="Phase -- Raw (Degrees)",
    )
    return


@app.cell
def _(plot_phase, temp_filtered):
    plot_phase(
        temp_filtered,
        columns="Temperature",
        unwrap=False,
        title="Phase -- Filtered (Wrapped)",
    )
    return


@app.cell
def _(plot_phase, temp_df):
    plot_phase(
        temp_df,
        columns="Temperature",
        angle_unit="degree",
        unwrap=False,
        title="Phase -- Raw (Wrapped Degrees)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Higher-Order and Bandpass Filters

    Changing the filter parameters lets you isolate different frequency
    bands. A **bandpass** filter around the daily cycle (period ≈ 48
    samples → normalised frequency ≈ 0.04) extracts just that
    component.
    """)
    return


@app.cell
def _(NumericalFilter, plot_time_series, temp_df):
    bandpass = NumericalFilter(
        design="butterworth",
        mode="bandpass",
        order=3,
        cutoff_frequency=(0.03, 0.06),
    )
    bandpass.fit(temp_df)
    temp_bp = bandpass.transform(temp_df)
    plot_time_series(
        temp_bp,
        columns="Temperature",
        title="Bandpass Filtered -- Daily Cycle Extracted",
    )
    return (temp_bp,)


@app.cell
def _(plot_spectrum, temp_bp):
    plot_spectrum(
        temp_bp,
        columns="Temperature",
        log_scale=True,
        title="Power Spectrum -- Bandpass (Log Scale)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - `NumericalFilter` applies causal IIR filters (Butterworth, Chebyshev,
      etc.) to polars DataFrames while preserving the `"time"` column
    - `plot_spectrum` shows how filtering reshapes the frequency content;
      `log_scale=True` reveals detail across decades; period is always
      shown in hover tooltips and peak annotations
    - `plot_phase` exposes phase shifts; `unwrap=True` prevents jumps;
      `angle_unit="degree"` aids interpretation
    - Bandpass mode isolates specific periodicities (here the daily cycle)

    ## Next Steps

    - **Similarity heatmaps**: See `examples/plotting/similarity_heatmap.py`
      for distance-based interval forecasting weights
    - **Seasonal diagnostics**: See `examples/plotting/seasonal.py` for
      ACF/PACF and seasonality overlays
    - **Decomposition**: See `examples/plotting/decomposition.py` for
      STL decomposition and calendar heatmaps
    """)
    return


if __name__ == "__main__":
    app.run()
