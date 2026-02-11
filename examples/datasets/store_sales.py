"""Store Sales - Panel Retail Analysis.

Daily retail sales for 3 stores × 3 items in Yohou panel format.

Dataset: 9 panel groups (store_X_item_Y__sales), 1826 daily observations
Demonstrates: inspect_locality, plot_time_series, plot_boxplot, plot_seasonality
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import polars as pl

    from yohou.datasets import load_store_sales
    from yohou.plotting import (
        plot_boxplot,
        plot_seasonality,
        plot_time_series,
    )
    from yohou.utils.panel import inspect_locality

    return (
        inspect_locality,
        load_store_sales,
        mo,
        pl,
        plot_boxplot,
        plot_seasonality,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Store Sales Dataset

    ## What You'll Learn

    This example demonstrates panel data analysis with the Store Sales dataset,
    which is pre-formatted in Yohou's native `__` panel convention. You'll learn
    how to:

    - Inspect panel structure with `inspect_locality`
    - Visualize panel columns directly without manual pivoting
    - Compare sales across stores and items
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
def _(load_store_sales):
    df = load_store_sales()
    df.head(10)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Inspect Panel Structure

    The dataset uses Yohou's `__` separator convention: `store_X_item_Y__sales`.
    `inspect_locality` identifies global columns and panel groups automatically.
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
    ## 2. Store Comparison: All Items for Store 1

    Visualizing all items for a single store shows how different product categories
    contribute to total sales.
    """)
    return


@app.cell
def _(df, plot_time_series):
    store_1_cols = [c for c in df.columns if c.startswith("store_1")]

    plot_time_series(
        df,
        columns=store_1_cols,
        title="Store 1 - All Items",
        y_label="Sales",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Item Comparison Across Stores

    Comparing the same item across stores reveals location-specific demand patterns.
    """)
    return


@app.cell
def _(df, plot_time_series):
    item_1_cols = [c for c in df.columns if c.endswith("item_1__sales")]

    plot_time_series(
        df,
        columns=item_1_cols,
        title="Item 1 - All Stores",
        y_label="Sales",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Sales Distribution by Panel Group

    Boxplots reveal the distribution of daily sales for each store-item
    combination, making it easy to compare variability and outliers.
    """)
    return


@app.cell
def _(df, plot_boxplot):
    plot_boxplot(
        df,
        title="Sales Distribution by Store-Item",
        y_label="Sales",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Day-of-Week Patterns

    Aggregating by day of week shows the weekly sales cycle and identifies peak
    shopping days.
    """)
    return


@app.cell
def _(df, plot_seasonality):
    plot_seasonality(
        df,
        columns="store_1_item_1__sales",
        feature="dayofweek",
        aggregation="mean",
        title="Store 1 Item 1 - Average Sales by Day of Week",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **Native panel format**: Columns use `store_X_item_Y__sales` convention: no pivoting needed
    - **`inspect_locality`**: Automatically discovers panel groups from column names
    - **Direct plotting**: Panel columns can be plotted directly with `plot_time_series`
    - **Boxplots**: Compare distributions across panel groups to spot outliers and variability
    - **Weekday patterns**: Clear day-of-week effects in retail sales

    ## Next Steps

    - **Branch-level panel**: See `examples/datasets/walmart_sales.py`
    - **Quarterly panel data**: See `examples/datasets/australian_tourism.py`
    - **Panel forecasting**: See `examples/point/panel_forecasting.py` for global vs local models
    """)
    return


if __name__ == "__main__":
    app.run()
