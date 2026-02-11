"""Sunspots - Smoothing and Frequency Analysis.

Time series smoothing and frequency domain analysis using
the classic Wolf sunspot numbers dataset.

Dataset: Monthly sunspot numbers, 1749-1984
Demonstrates: plot_timeseries, plot_exponential_moving_average, plot_rolling_statistics, plot_autocorrelation, plot_periodogram, plot_lag_scatter
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    from yohou.datasets import load_sunspots
    from yohou.plotting import (
        plot_autocorrelation,
        plot_correlation_diagnostics,
        plot_exponential_moving_average,
        plot_lag_scatter,
        plot_partial_autocorrelation,
        plot_periodogram,
        plot_rolling_statistics,
        plot_timeseries,
    )
    return (
        load_sunspots,
        pl,
        plot_autocorrelation,
        plot_correlation_diagnostics,
        plot_exponential_moving_average,
        plot_lag_scatter,
        plot_partial_autocorrelation,
        plot_periodogram,
        plot_rolling_statistics,
        plot_timeseries,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Sunspot Numbers Dataset

        Monthly mean total sunspot numbers from 1749 to 1984. This classic dataset
        exhibits cyclic patterns with ~11-year solar cycles, making it perfect for
        demonstrating smoothing techniques and frequency domain analysis.
        """
    )
    return


@app.cell
def _(load_sunspots):
    # Load dataset
    df = load_sunspots()
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 1. Raw Time Series")
    return


@app.cell
def _(df, plot_timeseries):
    # Original series
    fig1 = plot_timeseries(
        df,
        title="Sunspot Numbers (1749-1984)",
        x_label="Time",
        y_label="Sunspot Count",
    )
    fig1
    return (fig1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 2. Exponential Moving Average Smoothing")
    return


@app.cell
def _(df, plot_exponential_moving_average):
    # EWM with different spans
    fig2 = plot_exponential_moving_average(
        df,
        span=12,
        show_original=True,
        title="12-Month Exponential Moving Average",
    )
    fig2
    return (fig2,)


@app.cell
def _(df, plot_exponential_moving_average):
    # EWM only (no original)
    fig3 = plot_exponential_moving_average(
        df,
        span=24,
        show_original=False,
        smooth_width=3.0,
        title="24-Month Exponential Moving Average (Smooth Only)",
    )
    fig3
    return (fig3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 3. Rolling Statistics Comparison")
    return


@app.cell
def _(df, plot_rolling_statistics):
    # Mean + std bands
    fig4 = plot_rolling_statistics(
        df,
        window_size=132,  # 11 years
        statistics=["mean", "std"],
        fill_between=True,
        title="11-Year Rolling Mean ± 1 Std Dev",
    )
    fig4
    return (fig4,)


@app.cell
def _(df, plot_rolling_statistics):
    # Median with quartiles
    fig5 = plot_rolling_statistics(
        df,
        window_size=132,
        statistics=["median", "q25", "q75"],
        fill_between=True,
        show_original=False,
        title="11-Year Rolling Median with IQR",
    )
    fig5
    return (fig5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 4. Autocorrelation Analysis")
    return


@app.cell
def _(df, plot_autocorrelation):
    # ACF - should show ~11-year cycle
    fig6 = plot_autocorrelation(
        df,
        lags=200,  # ~16 years
        title="Autocorrelation Function (16+ years)",
    )
    fig6
    return (fig6,)


@app.cell
def _(df, plot_partial_autocorrelation):
    # PACF
    fig7 = plot_partial_autocorrelation(
        df,
        lags=100,
        title="Partial Autocorrelation Function",
    )
    fig7
    return (fig7,)


@app.cell
def _(df, plot_correlation_diagnostics):
    # Combined diagnostics
    fig8 = plot_correlation_diagnostics(
        df,
        lags=150,
        title="Correlation Diagnostics",
    )
    fig8
    return (fig8,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 5. Frequency Domain Analysis")
    return


@app.cell
def _(df, plot_periodogram):
    # Periodogram - should show peak at ~11-year period
    fig9 = plot_periodogram(
        df,
        detrend="linear",
        log_scale=False,
        show_peaks=True,
        n_peaks=5,
        title="Periodogram - Dominant Frequencies",
    )
    fig9
    return (fig9,)


@app.cell
def _(df, plot_periodogram):
    # Log scale version
    fig10 = plot_periodogram(
        df,
        detrend="linear",
        log_scale=True,
        title="Periodogram (Log Scale)",
    )
    fig10
    return (fig10,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 6. Lag Dependency Structure")
    return


@app.cell
def _(df, plot_lag_scatter):
    # Lag scatter - visualize cyclic dependency
    fig11 = plot_lag_scatter(
        df,
        lags=[1, 12, 66, 132],  # 1 month, 1 year, 5.5 years, 11 years
        show_diagonal=True,
        show_regression=False,
        title="Lag Scatter Plots (1mo, 1yr, 5.5yr, 11yr)",
    )
    fig11
    return (fig11,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Summary

        The sunspot numbers dataset clearly demonstrates:

        - **Cyclic behavior**: ~11-year solar cycles visible in ACF and periodogram
        - **Smoothing**: EWM and rolling statistics effectively reveal underlying patterns
        - **Frequency analysis**: Periodogram identifies dominant cycle periods
        - **Lag structure**: Scatter plots show strong autocorrelation at cycle multiples

        This dataset is ideal for teaching cyclical time series analysis and
        demonstrating the power of frequency domain methods.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
