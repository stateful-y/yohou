# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.19.11"
__gallery__ = {
    "title": "Hospital",
    "description": "Explore the Hospital panel dataset (767 series) with multi-series visualisation, cross-correlation analysis across lags, and monthly seasonality.",
    "category": "how-to",
    "section": "data-catalog",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from yohou.datasets import fetch_hospital
    from yohou.plotting import (
        plot_cross_correlation,
        plot_seasonality,
        plot_time_series,
    )

    return (
        fetch_hospital,
        mo,
        plot_cross_correlation,
        plot_seasonality,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Hospital Dataset

    This notebook shows how to explore the Hospital panel dataset (767 series) with multi-series visualisation, cross-correlation analysis across lags, and monthly seasonality.

    **Prerequisites:** None. this is a standalone dataset exploration.
    """)


@app.cell
def _(fetch_hospital):
    _all = fetch_hospital().frame
    # Select first 6 series for a manageable exploration
    _cols = ["time"] + [c for c in _all.columns if c != "time"][:6]
    df = _all.select(_cols)
    df.head(20)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Multiple Series Visualization

    Plotting several patient count series together shows how different
    medical product categories co-move over time.
    """)


@app.cell
def _(df, plot_time_series):
    _cols = [c for c in df.columns if c.endswith("__patients")][:4]

    plot_time_series(
        df,
        columns=_cols,
        title="Hospital Patient Counts - First 4 Series",
        y_label="Patients",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Cross-Correlation Between Series

    Cross-correlation between T1 and T2 quantifies the strength and
    timing of the relationship between two patient count series.
    """)


@app.cell
def _(df, plot_cross_correlation):
    plot_cross_correlation(
        df,
        columns=["T1__patients", "T2__patients"],
        max_lags=24,
        title="T1 vs T2 Patient Counts Cross-Correlation",
    )
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Cross-Correlation: T1 vs T3

    Comparing cross-correlation with a different series reveals whether
    different categories have different lag structures.
    """)


@app.cell
def _(df, plot_cross_correlation):
    plot_cross_correlation(
        df,
        columns=["T1__patients", "T3__patients"],
        max_lags=24,
        title="T1 vs T3 Patient Counts Cross-Correlation",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Monthly Seasonality

    Aggregating by month reveals seasonal healthcare demand patterns
    driven by illness seasonality and administrative cycles.
    """)


@app.cell
def _(df, plot_seasonality):
    plot_seasonality(
        df,
        columns="T1__patients",
        seasonality="month",
        title="T1 - Average Patient Count by Month",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Quarterly Patterns

    Quarterly aggregation shows broader seasonal trends in patient counts.
    """)


@app.cell
def _(df, plot_seasonality):
    plot_seasonality(
        df,
        columns="T1__patients",
        seasonality="quarter",
        title="T1 - Average Patient Count by Quarter",
    )



if __name__ == "__main__":
    app.run()
