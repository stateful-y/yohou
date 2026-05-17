# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "How to Visualize Correlations",
    "description": "Pairwise correlation heatmaps, scatter matrices, cross-correlation at multiple lags, and lag scatter plots for multivariate time series diagnostics.",
    "category": "tutorial",
    "section": "visualization",
    "api_references": ["plot_correlation_heatmap", "plot_cross_correlation", "plot_lag_scatter", "plot_scatter_matrix"],
}
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

    This notebook shows how to create pairwise correlation heatmaps, scatter matrices, cross-correlation at multiple lags, and lag scatter plots for multivariate time series diagnostics.

    **Prerequisites:** Basic understanding of correlation and scatter plots. Familiarity with multivariate time series concepts.
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load and Prepare Data

    We load two datasets using [`fetch_electricity_demand`](/pages/api/generated/yohou.datasets._fetchers.fetch_electricity_demand/) and [`fetch_hospital`](/pages/api/generated/yohou.datasets._fetchers.fetch_hospital/).
    The electricity demand data is downsampled from half-hourly to daily means for
    three Australian states (Victoria, NSW, South Australia). The hospital data
    provides three monthly patient count series.
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

    [`plot_correlation_heatmap`](/pages/api/generated/yohou.plotting.diagnostics.plot_correlation_heatmap/) computes pairwise Pearson correlations and renders
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

    [`plot_scatter_matrix`](/pages/api/generated/yohou.plotting.diagnostics.plot_scatter_matrix/) creates an N x N grid of pairwise scatter plots.
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

    [`plot_cross_correlation`](/pages/api/generated/yohou.plotting.diagnostics.plot_cross_correlation/) computes the cross-correlation function (CCF) between
    two columns at integer lags. Vary **lags**, **alpha** (significance level),
    and reverse the column pair to inspect asymmetry.
    """)


@app.cell
def _(plot_cross_correlation, vic_daily):
    plot_cross_correlation(
        vic_daily,
        columns=["Victoria", "NSW"],
        max_lags=40,
        title="CCF: Victoria vs NSW (40 Lags)",
    )


@app.cell
def _(plot_cross_correlation, vic_daily):
    plot_cross_correlation(
        vic_daily,
        columns=["Victoria", "NSW"],
        max_lags=20,
        confidence_level=0.99,
        title="CCF: Victoria vs NSW (20 Lags, 99% CI)",
    )


@app.cell
def _(plot_cross_correlation, vic_daily):
    plot_cross_correlation(
        vic_daily,
        columns=["NSW", "Victoria"],
        max_lags=40,
        title="CCF: NSW vs Victoria (Reversed Direction)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Lag Scatter

    [`plot_lag_scatter`](/pages/api/generated/yohou.plotting.diagnostics.plot_lag_scatter/) plots $y(t)$ vs $y(t - k)$ for one or more **lags**.
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
    mo.md(
        r"""
        ## Next Steps

        - [Visualize Forecasts](/pages/how-to/visualize-forecasts/) for the full guide
        - [Exploratory Visualization Tutorial](/pages/tutorials/exploratory-visualization/) for related techniques
        """
    )
    return

if __name__ == "__main__":
    app.run()
