# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.19.11"
__gallery__ = {
    "title": "Sunspots",
    "description": "Analyse 200+ years of monthly sunspot numbers with 11-year rolling smoothing, autocorrelation periodicity detection, and spectral frequency analysis.",
    "category": "how-to",
    "section": "data-catalog",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import polars as pl

    from yohou.datasets import fetch_sunspot
    from yohou.plotting import (
        plot_autocorrelation,
        plot_rolling_statistics,
        plot_spectrum,
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

    This notebook shows how to analyse 200+ years of monthly sunspot numbers with 11-year rolling smoothing, autocorrelation periodicity detection, and spectral frequency analysis.

    **Prerequisites:** This is a standalone dataset exploration.
    """)


@app.cell
def _(fetch_sunspot, pl):
    df = fetch_sunspot().frame.group_by_dynamic("time", every="1mo").agg(pl.col("sunspot_number").mean())
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Raw Time Series

    The raw data spans 200+ years of monthly sunspot observations, revealing the
    characteristic ~11-year solar cycle.
    """)


@app.cell
def _(df, plot_time_series):
    plot_time_series(
        df,
        title="Sunspot Numbers (1818-2020)",
        y_label="Sunspot Count",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Rolling Mean (11-Year Cycle)

    An 11-year rolling average smooths out individual cycles and highlights
    longer-term trends in solar activity.
    """)


@app.cell
def _(df, plot_rolling_statistics):
    plot_rolling_statistics(
        df,
        window_size=132,  # 11 years * 12 months
        statistics="mean",
        show_original=True,
        title="11-Year Rolling Mean",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Rolling Median with IQR

    The interquartile range (IQR) envelope shows the variability of sunspot
    activity within each cycle.
    """)


@app.cell
def _(df, plot_rolling_statistics):
    plot_rolling_statistics(
        df,
        window_size=132,
        statistics=["median", "q25", "q75"],
        show_original=False,
        title="11-Year Rolling Median with IQR",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Autocorrelation (Shows 11-Year Cycle)

    Autocorrelation reveals the cyclical structure, with peaks at ~132-month
    intervals confirming the ~11-year solar cycle.
    """)


@app.cell
def _(df, plot_autocorrelation):
    plot_autocorrelation(
        df,
        max_lags=200,
        title="Autocorrelation Function",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Periodogram (Frequency Domain)

    Spectral analysis identifies the dominant frequency components in the
    sunspot data.
    """)


@app.cell
def _(df, plot_spectrum):
    plot_spectrum(
        df,
        log_scale=True,
        show_peaks=True,
        n_peaks=5,
        title="Periodogram - Dominant Frequencies",
    )



if __name__ == "__main__":
    app.run()
