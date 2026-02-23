"""Sunspots - Cyclic Pattern Analysis.

Sunspot dataset demonstrating 11-year solar cycles.

Dataset: Daily sunspot numbers, 1818-2020 (resampled to monthly)
Demonstrates: plot_time_series, plot_rolling_statistics, plot_autocorrelation, plot_spectrum
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import polars as pl

    from yohou.datasets import fetch_sunspot
    from yohou.plotting import (
        plot_autocorrelation,
        plot_spectrum,
        plot_rolling_statistics,
        plot_time_series,
    )

    return (
        fetch_sunspot,
        mo,
        pl,
        plot_autocorrelation,
        plot_spectrum,
        plot_rolling_statistics,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Sunspot Numbers Dataset

    ## What You'll Learn

    This example demonstrates analysis of long-term cyclical patterns using the
    Sunspot Numbers dataset. You'll learn how to:

    - Visualize long historical time series (200+ years)
    - Detect and smooth 11-year solar cycles with rolling statistics
    - Identify periodicity with autocorrelation analysis
    - Find dominant frequencies using spectral analysis

    ## Prerequisites

    None -- this is a standalone dataset exploration.
    """)
    return


@app.cell(hide_code=True)
async def _():
    import sys as _sys

    if "pyodide" in _sys.modules:
        import micropip

        await micropip.install(["plotly", "yohou"])
    return


@app.cell
def _(fetch_sunspot, pl):
    df = (
        fetch_sunspot()
        .frame.group_by_dynamic("time", every="1mo")
        .agg(pl.col("sunspot_number").mean())
    )
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Raw Time Series

    The raw data spans 200+ years of monthly sunspot observations, revealing the
    characteristic ~11-year solar cycle.
    """)
    return


@app.cell
def _(df, plot_time_series):
    plot_time_series(
        df,
        title="Sunspot Numbers (1818-2020)",
        y_label="Sunspot Count",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Rolling Mean (11-Year Cycle)

    An 11-year rolling average smooths out individual cycles and highlights
    longer-term trends in solar activity.
    """)
    return


@app.cell
def _(df, plot_rolling_statistics):
    plot_rolling_statistics(
        df,
        window_size=132,  # 11 years * 12 months
        statistics="mean",
        show_original=True,
        title="11-Year Rolling Mean",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Rolling Median with IQR

    The interquartile range (IQR) envelope shows the variability of sunspot
    activity within each cycle.
    """)
    return


@app.cell
def _(df, plot_rolling_statistics):
    plot_rolling_statistics(
        df,
        window_size=132,
        statistics=["median", "q25", "q75"],
        fill_between=True,
        show_original=False,
        title="11-Year Rolling Median with IQR",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Autocorrelation (Shows 11-Year Cycle)

    Autocorrelation reveals the cyclical structure, with peaks at ~132-month
    intervals confirming the ~11-year solar cycle.
    """)
    return


@app.cell
def _(df, plot_autocorrelation):
    plot_autocorrelation(
        df,
        lags=200,
        title="Autocorrelation Function",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Periodogram (Frequency Domain)

    Spectral analysis identifies the dominant frequency components in the
    sunspot data.
    """)
    return


@app.cell
def _(df, plot_spectrum):
    plot_spectrum(
        df,
        detrend="linear",
        log_scale=True,
        show_peaks=True,
        n_peaks=5,
        title="Periodogram - Dominant Frequencies",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **Solar cycles**: Clear ~11-year periodicity in sunspot activity
    - **Long series**: 200+ years of monthly data reveals secular patterns
    - **Rolling IQR**: `fill_between=True` with quantiles shows uncertainty bands
    - **ACF peaks**: Autocorrelation shows cyclical peaks at ~132-month intervals
    - **Spectral analysis**: Periodogram identifies dominant frequency components

    ## Next Steps

    - For another long series, see `examples/datasets/air_passengers.py`
    - For higher frequency with cycles, see `examples/datasets/vic_electricity.py`
    - For panel data with cycles, see `examples/datasets/australian_tourism.py`
    """)
    return


if __name__ == "__main__":
    app.run()
