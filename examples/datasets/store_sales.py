"""Dominick - Weekly Retail Panel Analysis.

Weekly retail profit for SKUs from Dominick's Finer Foods in Yohou panel format.

Dataset: 115704 weekly profit series (exploring first 9)
Demonstrates: inspect_locality, plot_time_series, plot_boxplot, plot_seasonality
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import polars as pl

    from yohou.datasets import fetch_dominick
    from yohou.plotting import (
        plot_boxplot,
        plot_seasonality,
        plot_time_series,
    )
    from yohou.utils.panel import inspect_locality

    return (
        fetch_dominick,
        inspect_locality,
        mo,
        pl,
        plot_boxplot,
        plot_seasonality,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Dominick Dataset

    ## What You'll Learn

    This example demonstrates panel data analysis with the Dominick dataset
    (Dominick's Finer Foods), pre-formatted in Yohou's native `__` panel
    convention. You'll learn how to:

    - Inspect panel structure with `inspect_locality`
    - Visualize panel columns directly without manual pivoting
    - Compare profit across SKUs
    - Analyze distributions across panel groups with boxplots

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
def _(fetch_dominick):
    _all = fetch_dominick().frame
    # Select first 9 series for a manageable panel
    _cols = ["time"] + [c for c in _all.columns if c != "time"][:9]
    df = _all.select(_cols)
    df.head(10)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Inspect Panel Structure

    The full dataset has 115704 series using the `Tn__profit` convention.
    Here we work with the first 9 series.
    """)
    return


@app.cell
def _(df, inspect_locality, mo):
    global_cols, panel_groups = inspect_locality(df)
    mo.md(f"""
    **Global columns**: {global_cols}

    **Panel groups** ({len(panel_groups)} groups):

    {chr(10).join(f'- **{k}**: {v}' for k, v in panel_groups.items())}
    """)
    return (panel_groups,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. First Three Series

    Visualizing the first three SKU profit series shows how different products
    behave over time.
    """)
    return


@app.cell
def _(df, plot_time_series):
    first_3_cols = [c for c in df.columns if c.endswith("__profit")][:3]

    plot_time_series(
        df,
        columns=first_3_cols,
        title="Dominick - First 3 SKUs",
        y_label="Profit",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. All Selected Series

    Comparing all 9 selected series reveals scale differences and shared
    patterns across SKUs.
    """)
    return


@app.cell
def _(df, plot_time_series):
    all_profit_cols = [c for c in df.columns if c.endswith("__profit")]

    plot_time_series(
        df,
        columns=all_profit_cols,
        title="Dominick - 9 SKUs",
        y_label="Profit",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Profit Distribution by Panel Group

    Boxplots reveal the distribution of weekly profit for each SKU,
    making it easy to compare variability and outliers.
    """)
    return


@app.cell
def _(df, plot_boxplot):
    plot_boxplot(
        df,
        title="Profit Distribution by SKU",
        y_label="Profit",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Monthly Patterns

    Aggregating by month shows seasonal profit cycles across the year.
    """)
    return


@app.cell
def _(df, plot_seasonality):
    _first_col = [c for c in df.columns if c.endswith("__profit")][0]
    plot_seasonality(
        df,
        columns=_first_col,
        feature="month",
        aggregation="mean",
        title="T1 - Average Profit by Month",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **Native panel format**: Columns use `Tn__profit` convention: no pivoting needed
    - **`inspect_locality`**: Automatically discovers panel groups from column names
    - **Direct plotting**: Panel columns can be plotted directly with `plot_time_series`
    - **Boxplots**: Compare distributions across panel groups to spot outliers and variability
    - **Large dataset**: 115704 series available; select a subset for exploration

    ## Next Steps

    - **Hourly panel**: See `examples/datasets/walmart_sales.py`
    - **Quarterly panel data**: See `examples/datasets/australian_tourism.py`
    - **Panel forecasting**: See `examples/point/panel_forecasting.py` for global vs local models
    """)
    return


if __name__ == "__main__":
    app.run()
