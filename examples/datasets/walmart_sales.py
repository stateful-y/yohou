"""Walmart Sales - Branch-Level Panel Analysis.

Daily sales and ratings for 3 branches in Yohou panel format.

Dataset: 3 branches (A, B, C) with total sales and ratings, 89 daily observations
Demonstrates: inspect_locality, plot_time_series, plot_boxplot, plot_seasonality
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from yohou.datasets import load_walmart_sales
    from yohou.plotting import (
        plot_boxplot,
        plot_seasonality,
        plot_time_series,
    )
    from yohou.utils.panel import inspect_locality

    return (
        inspect_locality,
        load_walmart_sales,
        mo,
        plot_boxplot,
        plot_seasonality,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Walmart Sales Dataset

    ## What You'll Learn

    This example demonstrates branch-level panel analysis with the Walmart Sales
    dataset, pre-formatted in Yohou's native `__` panel convention. You'll learn
    how to:

    - Inspect panel structure with `inspect_locality`
    - Compare daily sales and ratings across branches
    - Visualize distributions across branches with boxplots
    - Analyze day-of-week effects

    ## Prerequisites

    None. this is a standalone dataset exploration.
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
def _(load_walmart_sales):
    df = load_walmart_sales()
    df.head(10)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Inspect Panel Structure

    The dataset has 3 branches (A, B, C), each with `total` and `rating` variables.
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Branch Sales Comparison

    Comparing total sales across branches A, B, and C shows relative revenue
    performance and temporal alignment.
    """)
    return


@app.cell
def _(df, plot_time_series):
    total_cols = [c for c in df.columns if c.endswith("__total")]

    plot_time_series(
        df,
        columns=total_cols,
        title="Daily Sales by Branch",
        y_label="Sales ($)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Branch Ratings Comparison

    Customer ratings across branches reveal service quality differences and
    potential seasonal patterns.
    """)
    return


@app.cell
def _(df, plot_time_series):
    rating_cols = [c for c in df.columns if c.endswith("__rating")]

    plot_time_series(
        df,
        columns=rating_cols,
        title="Average Daily Rating by Branch",
        y_label="Rating",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Sales Distribution by Branch

    Boxplots compare total sales distributions across branches, revealing
    relative performance and variability in this short 89-day window.
    """)
    return


@app.cell
def _(df, plot_boxplot):
    _total_cols = [c for c in df.columns if c.endswith("__total")]

    plot_boxplot(
        df,
        columns=_total_cols,
        title="Total Sales Distribution by Branch",
        y_label="Sales ($)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Day-of-Week Patterns

    Weekly aggregation reveals which days drive the most sales activity across
    all branches.
    """)
    return


@app.cell
def _(df, plot_seasonality):
    plot_seasonality(
        df,
        columns="branch_a__total",
        feature="dayofweek",
        aggregation="mean",
        title="Branch A - Average Sales by Day of Week",
        y_label="Average Sales ($)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **Native panel format**: Columns use `branch_X__total` and `branch_X__rating` convention
    - **Multi-variable panels**: Each branch has both `total` and `rating` variables
    - **Short horizon**: 89 days of data (Jan–Mar 2019): suitable for short-term analysis
    - **Branch comparison**: Direct multi-series plotting without manual pivoting
    - **`inspect_locality`**: Automatically discovers panel groups from column names

    ## Next Steps

    - **Larger retail panel**: See `examples/datasets/store_sales.py`
    - **Quarterly panel data**: See `examples/datasets/australian_tourism.py`
    - **Short-horizon forecasting**: See `examples/datasets/walmart_forecasting.py` for panel forecasting on this dataset
    """)
    return


if __name__ == "__main__":
    app.run()
