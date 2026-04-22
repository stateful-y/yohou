# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Signal Processing Plots",
    "description": "Butterworth low-pass filtering with frequency spectrum analysis and phase shift inspection on half-hourly electricity demand data using Plotly.",
    "category": "how-to",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():

    from yohou.datasets import fetch_electricity_demand
    from yohou.plotting import plot_phase, plot_spectrum, plot_time_series
    from yohou.preprocessing import NumericalFilter

    return (
        NumericalFilter,
        fetch_electricity_demand,
        plot_phase,
        plot_spectrum,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Signal Processing

    This notebook shows how to apply Butterworth low-pass filtering with frequency spectrum analysis and phase shift inspection on half-hourly electricity demand data using Plotly.
    """)


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

    [`plot_time_series`](/pages/api/generated/yohou.plotting.exploration.plot_time_series/) renders the raw demand values against the time axis.
    The 30-day window shows a clear daily cycle (period of approximately
    48 half-hour samples) plus short-term noise that the filter will remove.
    """)


@app.cell
def _(plot_time_series, vic_short):
    plot_time_series(
        vic_short,
        columns="vic__demand",
        title="Raw Demand - 30-day Window",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Applying a Low-Pass Filter

    [`NumericalFilter`](/pages/api/generated/yohou.preprocessing.signal.NumericalFilter/) applies causal IIR filters to polars DataFrames.
    Here a **4th-order Butterworth** low-pass filter with `cutoff_frequency=0.05`
    (relative to Nyquist) keeps only the slowest oscillations, roughly
    the daily cycle, and removes everything faster.
    """)


@app.cell
def _(NumericalFilter, vic_short):
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
        title="Demand - Raw vs. Low-Pass Filtered",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Power Spectrum Comparison

    [`plot_spectrum`](/pages/api/generated/yohou.plotting.signal.plot_spectrum/) computes the one-sided power spectral density via FFT.
    The raw signal shows energy at many frequencies; after filtering, only the
    low-frequency band retains power.
    """)


@app.cell
def _(plot_spectrum, temp_df):
    plot_spectrum(
        temp_df,
        columns="vic__demand",
        title="Power Spectrum - Raw Demand",
    )


@app.cell
def _(plot_spectrum, temp_filtered):
    plot_spectrum(
        temp_filtered,
        columns="vic__demand",
        title="Power Spectrum - After Low-Pass Filter",
    )


@app.cell
def _(plot_spectrum, temp_df):
    plot_spectrum(
        temp_df,
        columns="vic__demand",
        log_scale=True,
        title="Power Spectrum - Raw (Log Scale)",
    )


@app.cell
def _(plot_spectrum, temp_filtered):
    plot_spectrum(
        temp_filtered,
        columns="vic__demand",
        log_scale=True,
        title="Power Spectrum - Filtered (Log Scale)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase Plot Comparison

    The **phase** angle at each frequency reveals the signal's timing
    structure. `unwrap=True` (default) produces continuous phase;
    `angle_unit="degree"` converts from radians. Notice how the filter
    shifts the phase at higher frequencies that are attenuated.
    """)


@app.cell
def _(plot_phase, temp_df):
    plot_phase(
        temp_df,
        columns="vic__demand",
        title="Phase - Raw Demand",
    )


@app.cell
def _(plot_phase, temp_filtered):
    plot_phase(
        temp_filtered,
        columns="vic__demand",
        title="Phase - After Low-Pass Filter",
    )


@app.cell
def _(plot_phase, temp_df):
    plot_phase(
        temp_df,
        columns="vic__demand",
        angle_unit="degree",
        title="Phase - Raw (Degrees)",
    )


@app.cell
def _(plot_phase, temp_filtered):
    plot_phase(
        temp_filtered,
        columns="vic__demand",
        unwrap=False,
        title="Phase - Filtered (Wrapped)",
    )


@app.cell
def _(plot_phase, temp_df):
    plot_phase(
        temp_df,
        columns="vic__demand",
        angle_unit="degree",
        unwrap=False,
        title="Phase - Raw (Wrapped Degrees)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Higher-Order and Bandpass Filters

    Changing the filter parameters lets you isolate different frequency
    bands. A **bandpass** filter around the daily cycle (period ≈ 48
    samples → normalised frequency ≈ 0.04) extracts just that
    component.
    """)


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
        title="Bandpass Filtered - Daily Cycle Extracted",
    )
    return (temp_bp,)


@app.cell
def _(plot_spectrum, temp_bp):
    plot_spectrum(
        temp_bp,
        columns="vic__demand",
        log_scale=True,
        title="Power Spectrum - Bandpass (Log Scale)",
    )



if __name__ == "__main__":
    app.run()
