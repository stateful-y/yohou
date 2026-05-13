# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.19.11"
__gallery__ = {
    "title": "Hospital Multivariate Analytics",
    "description": "Multivariate analytics on hospital patient data with scatter matrices, correlation heatmaps, STL decomposition, and cross-correlation diagnostics.",
    "category": "how-to",
    "section": "data-catalog",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import polars as pl

    from yohou.datasets import fetch_hospital
    from yohou.plotting import (
        plot_autocorrelation,
        plot_correlation_heatmap,
        plot_cross_correlation,
        plot_decomposition,
        plot_lag_scatter,
        plot_rolling_statistics,
        plot_scatter_matrix,
        plot_time_series,
    )

    return (
        fetch_hospital,
        mo,
        pl,
        plot_autocorrelation,
        plot_decomposition,
        plot_correlation_heatmap,
        plot_cross_correlation,
        plot_lag_scatter,
        plot_rolling_statistics,
        plot_scatter_matrix,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hospital: Multivariate Descriptive Analytics

    The Hospital dataset contains 767 monthly patient count series related
    to medical products (2000-2006). This notebook picks four series and
    examines the **relationships between them** using multivariate
    diagnostic plots.

    This notebook shows how to perform multivariate analytics on hospital patient data with scatter matrices, correlation heatmaps, STL decomposition, and cross-correlation diagnostics.

    **Prerequisites:** See `examples/datasets/hospital.py` for basic panel exploration
    (single-series seasonality and cross-correlation).
    """)


@app.cell
def _(fetch_hospital, pl, plot_time_series):
    _all = fetch_hospital().frame
    # Pick 4 series and rename to readable names
    hospital = _all.select(
        "time",
        pl.col("T1__patients").alias("patients_A"),
        pl.col("T2__patients").alias("patients_B"),
        pl.col("T3__patients").alias("patients_C"),
        pl.col("T4__patients").alias("patients_D"),
    ).drop_nulls()

    plot_time_series(hospital, title="Hospital Patient Counts (4 Series)")
    return (hospital,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Scatter Matrix

    The scatter matrix shows pairwise scatter plots (lower triangle),
    KDE density on the diagonal, and Pearson correlation coefficients
    in the upper triangle. This is a single-glance multivariate overview.
    """)


@app.cell
def _(hospital, plot_scatter_matrix):
    plot_scatter_matrix(
        hospital,
        title="Hospital - Pairwise Scatter Matrix",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Correlation Heatmap

    A focused view of the pairwise Pearson correlation coefficients.
    Strongly correlated series may share common medical drivers.
    """)


@app.cell
def _(hospital, plot_correlation_heatmap):
    plot_correlation_heatmap(
        hospital,
        title="Hospital - Correlation Heatmap",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. STL Decomposition

    Seasonal-Trend decomposition using Loess (STL) separates each series
    into trend, seasonal, and residual components. Monthly data has a
    natural period of 12.
    """)


@app.cell
def _(hospital, plot_decomposition):
    plot_decomposition(
        hospital,
        ["observed", "trend", "seasonal", "residual", "seasonal_adjusted"],
        columns="patients_A",
        method="stl",
        period=12,
        title="Patients A (T1) - STL Decomposition",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Cross-Correlation Between Series

    Cross-correlation between patients A and B quantifies lead/lag
    relationships. A peak at lag 0 suggests synchronous co-movement;
    a peak at positive lag means B leads A.
    """)


@app.cell
def _(hospital, plot_cross_correlation):
    plot_cross_correlation(
        hospital,
        columns=["patients_A", "patients_B"],
        max_lags=24,
        title="Patients A vs B - Cross-Correlation",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Cross-Correlation: A vs C

    Comparing the cross-correlation pattern with a different pair
    reveals whether different product categories have different lag
    structures.
    """)


@app.cell
def _(hospital, plot_cross_correlation):
    plot_cross_correlation(
        hospital,
        columns=["patients_A", "patients_C"],
        max_lags=24,
        title="Patients A vs C - Cross-Correlation",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Autocorrelation

    ACF for a single series reveals the internal temporal dependence
    structure. Strong spikes at lag 12 confirm annual seasonality.
    """)


@app.cell
def _(hospital, plot_autocorrelation):
    plot_autocorrelation(
        hospital,
        columns="patients_A",
        max_lags=36,
        title="Patients A - Autocorrelation",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Lag Scatter

    Scatter y(t) vs y(t-lag) shows the strength of the temporal
    relationship at different horizons. Lags 1 and 12 are compared.
    """)


@app.cell
def _(hospital, plot_lag_scatter):
    plot_lag_scatter(
        hospital,
        columns="patients_A",
        lags=[1, 12],
        show_regression=True,
        title="Patients A - Lag Scatter (1 and 12 months)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Rolling Statistics

    A 12-month rolling mean and standard deviation track the trend
    and volatility over time.
    """)


@app.cell
def _(hospital, plot_rolling_statistics):
    plot_rolling_statistics(
        hospital,
        columns="patients_A",
        window_size=12,
        statistics=["mean", "std"],
        title="Patients A - Rolling Mean and Std (12-month window)",
    )



if __name__ == "__main__":
    app.run()
