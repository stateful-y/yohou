# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Signal Processing -- Spectrum and Phase after Numerical Filtering.

Applies `NumericalFilter` (Butterworth low-pass) to the highly periodic
Victorian electricity demand signal and compares the power spectrum
and phase spectrum before and after filtering.

Dataset: electricity_demand (vic__demand column -- 30-min periodicity)
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
def _():
    import polars as pl

    from yohou.datasets import fetch_electricity_demand
    from yohou.plotting import plot_phase, plot_spectrum, plot_time_series
    from yohou.preprocessing import NumericalFilter

    return (
        NumericalFilter,
        fetch_electricity_demand,
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

    - Applying a `NumericalFilter` (Butterworth low-pass) to a demand
      signal
    - Comparing raw vs. filtered signals in the time domain
    - Using `plot_spectrum` to see how a low-pass filter removes
      high-frequency noise
    - Using `plot_phase` to examine phase shifts introduced by the filter

    ## The Dataset

    Australian electricity demand data recorded every **30 minutes** over
    several years. The **vic__demand** column contains a strong daily cycle
    (period ≈ 48 samples) plus higher-frequency fluctuations that a
    low-pass filter will suppress.
    """)
    return

@app.cell
def _(fetch_electricity_demand):
    vic = fetch_electricity_demand().frame
    # Take 30 days (1440 half-hours) for readable plots
    vic_short = vic.head(1440)
    return (vic_short,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raw Demand Signal

    The 30-day window shows a clear daily cycle plus short-term noise.
    """)
    return

@app.cell
def _(plot_time_series, vic_short):
    plot_time_series(
        vic_short,
        columns="vic__demand",
        title="Raw Demand -- 30-day Window",
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
    temp_df = vic_short.select("time", "vic__demand")
    lowpass.fit(temp_df)
    temp_filtered = lowpass.transform(temp_df)
    # Combine raw and filtered for side-by-side comparison
    combined = temp_df.rename({"vic__demand": "Raw"}).join(
        temp_filtered.rename({"vic__demand": "Filtered"}),
        on="time",
    )
    return combined, temp_df, temp_filtered

@app.cell
def _(combined, plot_time_series):
    plot_time_series(
        combined,
        columns=["Raw", "Filtered"],
        title="Demand -- Raw vs. Low-Pass Filtered",
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
        columns="vic__demand",
        title="Power Spectrum -- Raw Demand",
    )
    return

@app.cell
def _(plot_spectrum, temp_filtered):
    plot_spectrum(
        temp_filtered,
        columns="vic__demand",
        title="Power Spectrum -- After Low-Pass Filter",
    )
    return

@app.cell
def _(plot_spectrum, temp_df):
    plot_spectrum(
        temp_df,
        columns="vic__demand",
        log_scale=True,
        title="Power Spectrum -- Raw (Log Scale)",
    )
    return

@app.cell
def _(plot_spectrum, temp_filtered):
    plot_spectrum(
        temp_filtered,
        columns="vic__demand",
        log_scale=True,
        title="Power Spectrum -- Filtered (Log Scale)",
    )
    return

@app.cell
def _(plot_spectrum, temp_df):
    plot_spectrum(
        temp_df,
        columns="vic__demand",
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
        columns="vic__demand",
        title="Phase -- Raw Demand",
    )
    return

@app.cell
def _(plot_phase, temp_filtered):
    plot_phase(
        temp_filtered,
        columns="vic__demand",
        title="Phase -- After Low-Pass Filter",
    )
    return

@app.cell
def _(plot_phase, temp_df):
    plot_phase(
        temp_df,
        columns="vic__demand",
        angle_unit="degree",
        title="Phase -- Raw (Degrees)",
    )
    return

@app.cell
def _(plot_phase, temp_filtered):
    plot_phase(
        temp_filtered,
        columns="vic__demand",
        unwrap=False,
        title="Phase -- Filtered (Wrapped)",
    )
    return

@app.cell
def _(plot_phase, temp_df):
    plot_phase(
        temp_df,
        columns="vic__demand",
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
        columns="vic__demand",
        title="Bandpass Filtered -- Daily Cycle Extracted",
    )
    return (temp_bp,)

@app.cell
def _(plot_spectrum, temp_bp):
    plot_spectrum(
        temp_bp,
        columns="vic__demand",
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
