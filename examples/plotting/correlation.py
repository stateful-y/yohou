# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "plotly",
#     "yohou",
# ]
# ///
"""Correlation and Scatter Diagnostics.

Demonstrates correlation heatmaps, scatter matrices, cross-correlation, and
lag scatter plots with varied parameter combinations.

Datasets: electricity_demand, hospital
Demonstrates: plot_correlation_heatmap, plot_scatter_matrix,
    plot_cross_correlation, plot_lag_scatter
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

    from yohou.datasets import fetch_electricity_demand, fetch_hospital
    from yohou.plotting import (
        plot_correlation_heatmap,
        plot_cross_correlation,
        plot_lag_scatter,
        plot_scatter_matrix,
    )

    return (
        fetch_electricity_demand,
        fetch_hospital,
        pl,
        plot_correlation_heatmap,
        plot_cross_correlation,
        plot_lag_scatter,
        plot_scatter_matrix,
    )
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Correlation and Scatter Diagnostics

    ## What You'll Learn

    - How to compute and visualize pairwise correlations with `plot_correlation_heatmap`
    - Building scatter matrices with seasonal coloring using `plot_scatter_matrix`
    - Measuring lead-lag relationships with `plot_cross_correlation`
    - Inspecting serial dependence with `plot_lag_scatter` at single or multiple lags

    ## Prerequisites

    Basic understanding of correlation and scatter plots. Familiarity with multivariate time series concepts.
    """)
@app.cell
def _(fetch_electricity_demand, fetch_hospital, pl):
    _elec = fetch_electricity_demand().frame
    # Downsample to daily, rename to plain columns for multivariate analysis
    vic_daily = _elec.group_by_dynamic("time", every="1d").agg(
        pl.col("vic__demand").mean().alias("Victoria"),
        pl.col("nsw__demand").mean().alias("NSW"),
        pl.col("sa__demand").mean().alias("SA"),
    )
    _hosp = fetch_hospital().frame
    # Select 3 series and rename for multivariate analysis
    hospital = _hosp.select(
        "time",
        pl.col("T1__patients").alias("patients_1"),
        pl.col("T2__patients").alias("patients_2"),
        pl.col("T3__patients").alias("patients_3"),
    )
    return hospital, vic_daily
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Correlation Heatmap

    `plot_correlation_heatmap` computes pairwise Pearson correlations and renders
    them as a heatmap. Vary the **colorscale**, toggle **show_values**, and restrict
    to a subset of **columns**.
    """)
@app.cell
def _(plot_correlation_heatmap, vic_daily):
    plot_correlation_heatmap(
        vic_daily,
        title="Pearson Correlation -- Default (RdBu_r, Values Shown)",
    )
@app.cell
def _(plot_correlation_heatmap, vic_daily):
    plot_correlation_heatmap(
        vic_daily,
        colorscale="Viridis",
        show_values=False,
        title="Pearson Correlation -- Viridis, No Annotations",
    )
@app.cell
def _(plot_correlation_heatmap, vic_daily):
    plot_correlation_heatmap(
        vic_daily,
        columns=["Victoria", "NSW"],
        title="Pearson Correlation -- Column Subset",
    )
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Scatter Matrix

    `plot_scatter_matrix` creates an N x N grid of pairwise scatter plots.
    The **diagonal** can be `"kde"`, `"histogram"`, or `"none"`. The **seasonality**
    parameter colors points by season, and **show_correlation** overlays Pearson r.
    """)
@app.cell
def _(plot_scatter_matrix, vic_daily):
    plot_scatter_matrix(
        vic_daily,
        title="Scatter Matrix -- Default (KDE Diagonal)",
    )
@app.cell
def _(plot_scatter_matrix, vic_daily):
    plot_scatter_matrix(
        vic_daily,
        diagonal="histogram",
        title="Scatter Matrix -- Histogram Diagonal",
    )
@app.cell
def _(plot_scatter_matrix, vic_daily):
    plot_scatter_matrix(
        vic_daily,
        diagonal="none",
        show_correlation=False,
        title="Scatter Matrix -- Scatter Only, No Statistics",
    )
@app.cell
def _(plot_scatter_matrix, vic_daily):
    plot_scatter_matrix(
        vic_daily,
        columns=["Victoria", "NSW"],
        seasonality="month",
        title="Scatter Matrix -- Month-Colored Points",
    )
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Cross-Correlation

    `plot_cross_correlation` computes the cross-correlation function (CCF) between
    two columns at integer lags. Vary **lags**, **alpha** (significance level),
    and reverse the column pair to inspect asymmetry.
    """)
@app.cell
def _(plot_cross_correlation, vic_daily):
    plot_cross_correlation(
        vic_daily,
        columns=["Victoria", "NSW"],
        lags=40,
        title="CCF: Victoria vs NSW (40 Lags)",
    )
@app.cell
def _(plot_cross_correlation, vic_daily):
    plot_cross_correlation(
        vic_daily,
        columns=["Victoria", "NSW"],
        lags=20,
        alpha=0.01,
        title="CCF: Victoria vs NSW (20 Lags, alpha=0.01)",
    )
@app.cell
def _(plot_cross_correlation, vic_daily):
    plot_cross_correlation(
        vic_daily,
        columns=["NSW", "Victoria"],
        lags=40,
        title="CCF: NSW vs Victoria (Reversed Direction)",
    )
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Lag Scatter

    `plot_lag_scatter` plots $y(t)$ vs $y(t - k)$ for one or more **lags**.
    Passing a list of lags produces a subplot grid. The **seasonality** parameter
    colors points by season, **show_regression** adds a trend line, and
    **show_diagonal** draws the $y = x$ identity.
    """)
@app.cell
def _(hospital, plot_lag_scatter):
    plot_lag_scatter(
        hospital,
        columns="patients_1",
        lags=1,
        title="Lag 1 Scatter -- Hospital patients_1",
    )
@app.cell
def _(hospital, plot_lag_scatter):
    plot_lag_scatter(
        hospital,
        columns="patients_1",
        lags=[1, 6, 12, 24],
        title="Multi-Lag Grid -- patients_1 at Lags 1, 6, 12, 24",
    )
@app.cell
def _(hospital, plot_lag_scatter):
    plot_lag_scatter(
        hospital,
        columns="patients_1",
        lags=[1, 12],
        seasonality="month",
        show_regression=True,
        title="Lag Scatter -- Month-Colored with Regression Line",
    )
@app.cell
def _(hospital, plot_lag_scatter):
    plot_lag_scatter(
        hospital,
        columns="patients_1",
        lags=[1, 7],
        show_diagonal=False,
        marker_opacity=0.3,
        title="Lag Scatter -- No Diagonal, Transparent Markers",
    )
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Correlation diagnostics** provide a quick overview of linear relationships; use `colorscale` and `show_values` to customise the heatmap
    - **Scatter matrices** reveal pairwise nonlinear patterns; `seasonality` coloring exposes seasonal clustering in the scatter space
    - **Cross-correlation** measures lead-lag effects between two series; reversing the pair reveals asymmetric relationships
    - **Lag scatter** at multiple lags shows how auto-dependence decays; `show_regression` and `seasonality` add interpretive layers

    ## Next Steps

    - **Seasonal diagnostics**: See `examples/plotting/seasonal.py` for seasonality overlays, ACF/PACF, and frequency-domain analysis
    - **STL decomposition**: See `examples/plotting/decomposition.py` for decomposition and calendar heatmaps
    - **Exploration**: See `examples/plotting/exploration.py` for rolling statistics and missing data audits
    """)
if __name__ == "__main__":
    app.run()
