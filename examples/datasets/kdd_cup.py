# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "KDD Cup Air Quality",
    "description": "Explore the KDD Cup 2018 multivariate panel dataset with hourly air quality readings from 59 Beijing and London monitoring stations.",
    "category": "how-to",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from yohou.datasets import fetch_kdd_cup
    from yohou.plotting import (
        plot_boxplot,
        plot_correlation_heatmap,
        plot_time_series,
    )
    from yohou.utils.panel import inspect_panel

    return (
        fetch_kdd_cup,
        inspect_panel,
        mo,
        plot_boxplot,
        plot_correlation_heatmap,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # KDD Cup 2018 Air Quality

    This notebook shows how to explore the KDD Cup 2018 multivariate panel dataset with hourly air quality readings from 59 Beijing and London monitoring stations.

    **Prerequisites:** None. This is a standalone dataset exploration.
    """)


@app.cell
def _(fetch_kdd_cup, plot_time_series):
    _bunch = fetch_kdd_cup(n_groups=2)
    df = _bunch.frame.head(24 * 7 * 52)
    # Show all stations and pollutants
    plot_time_series(df, title="KDD Cup Air Quality")
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Inspect Panel Structure

    `fetch_kdd_cup(n_groups=5)` loads 5 complete Beijing stations, each
    with 6 air quality measurements (PM2.5, PM10, NO2, CO, O3, SO2).
    The `station__measurement` naming convention makes each station a
    panel group with 6 members.
    """)


@app.cell
def _(df, inspect_panel, mo):
    global_cols, panel_groups = inspect_panel(df)
    mo.md(f"""
    **Global columns**: {global_cols}

    **Panel groups** ({len(panel_groups)} stations):

    {chr(10).join(f"- **{k}**: {v}" for k, v in panel_groups.items())}
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Multi-Pollutant View

    Unlike univariate panel datasets (one value per group), each station
    here has **multiple measurements**. This makes it a true multivariate
    panel, ideal for testing `panel_strategy="multivariate"` where
    cross-variable interactions matter.
    """)


@app.cell
def _(df, plot_time_series):
    plot_time_series(
        df,
        groups=["beijing_aotizhongxin_aq"],
        title="Beijing Aotizhongxin - All 6 Pollutants",
        y_label="Concentration",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Cross-Station Comparison

    Comparing the same pollutant across stations reveals spatial patterns
    and station-level heterogeneity.
    """)


@app.cell
def _(df, plot_time_series):
    plot_time_series(
        df,
        groups=[],
        columns=["pm2.5"],
        title="PM2.5 Across All Stations",
        y_label="PM2.5 (µg/m³)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Pollutant Distributions

    Boxplots show how pollutant distributions vary across stations and
    measurements.
    """)


@app.cell
def _(df, plot_boxplot):
    plot_boxplot(
        df,
        title="Pollutant Distributions Across Stations",
        y_label="Concentration",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Within-Station Correlations

    Air pollutants within a station are often correlated (e.g. PM2.5 and
    PM10 share common emission sources). The correlation heatmap reveals
    these relationships.
    """)


@app.cell
def _(df, plot_correlation_heatmap):
    _station_cols = [c for c in df.columns if c.startswith("beijing_aotizhongxin_aq__")]
    plot_correlation_heatmap(
        df,
        columns=_station_cols,
        title="Aotizhongxin Station - Pollutant Correlations",
    )



if __name__ == "__main__":
    app.run()
