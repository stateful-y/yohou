# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "plotly",
#     "yohou",
# ]
# ///
"""Hospital - Monthly Patient Counts Panel Analysis.

Monthly patient counts for medical products in Yohou panel format.

Dataset: 767 monthly patient count series (exploring first 6)
Demonstrates: plot_time_series, plot_cross_correlation, plot_seasonality
"""

import marimo

__generated_with = "0.19.11"
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

    ## What You'll Learn

    This example demonstrates monthly panel time series analysis with the
    Hospital dataset (767 series of patient counts). You'll learn how to:

    - Visualize multiple patient count series simultaneously
    - Analyze cross-correlation between different series
    - Identify monthly and quarterly seasonality patterns in healthcare data
    - Understand relationships across panel groups

    ## Prerequisites

    None. this is a standalone dataset exploration.
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

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **Panel format**: 767 monthly patient count series using `Tn__patients` convention
    - **Cross-series correlation**: Different medical product series show varying correlation strength
    - **Monthly seasonality**: Healthcare demand follows seasonal illness patterns
    - **Monthly resolution**: Captures medium-term trends in patient counts
    - **Large panel**: 767 series available; select a subset for tractable exploration

    ## Next Steps

    - For high-frequency panel data, see `examples/datasets/vic_electricity.py`
    - For weekly panel, see `examples/datasets/store_sales.py`
    - For univariate cyclic patterns, see `examples/datasets/sunspots.py`
    """)

if __name__ == "__main__":
    app.run()
