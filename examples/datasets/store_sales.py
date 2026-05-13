# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.19.11"
__gallery__ = {
    "title": "Dominick Store Sales",
    "description": "Explore the Dominick's Finer Foods panel dataset with per-SKU profit visualisation, cross-group distribution boxplots, and weekly seasonality.",
    "category": "how-to",
    "section": "data-catalog",
}
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
    from yohou.utils.panel import inspect_panel

    return (
        fetch_dominick,
        inspect_panel,
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

    This notebook shows how to explore the Dominick's Finer Foods panel dataset with per-SKU profit visualisation, cross-group distribution boxplots, and weekly seasonality.

    **Prerequisites:** This is a standalone dataset exploration.
    """)


@app.cell
def _(fetch_dominick, plot_time_series):
    _all = fetch_dominick().frame
    # Select first 9 series for a manageable panel
    _cols = ["time"] + [c for c in _all.columns if c != "time"][:9]
    df = _all.select(_cols)
    plot_time_series(df, title="Dominick Store Sales (9 Series)")
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Inspect Panel Structure

    By default, `fetch_dominick()` returns 50 series using the `Tn__profit`
    convention (pass `n_series=None` for all 115704). Here we work with the
    first 9 series.
    """)


@app.cell
def _(df, inspect_panel, mo):
    global_cols, panel_groups = inspect_panel(df)
    mo.md(f"""
    **Global columns**: {global_cols}

    **Panel groups** ({len(panel_groups)} groups):

    {chr(10).join(f"- **{k}**: {v}" for k, v in panel_groups.items())}
    """)
    return (panel_groups,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. First Three Series

    Visualizing the first three SKU profit series shows how different products
    behave over time.
    """)


@app.cell
def _(df, plot_time_series):
    first_3_cols = [c for c in df.columns if c.endswith("__profit")][:3]

    plot_time_series(
        df,
        columns=first_3_cols,
        title="Dominick - First 3 SKUs",
        y_label="Profit",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. All Selected Series

    Comparing all 9 selected series reveals scale differences and shared
    patterns across SKUs.
    """)


@app.cell
def _(df, plot_time_series):
    all_profit_cols = [c for c in df.columns if c.endswith("__profit")]

    plot_time_series(
        df,
        columns=all_profit_cols,
        title="Dominick - 9 SKUs",
        y_label="Profit",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Profit Distribution by Panel Group

    Boxplots reveal the distribution of weekly profit for each SKU,
    making it easy to compare variability and outliers.
    """)


@app.cell
def _(df, plot_boxplot):
    plot_boxplot(
        df,
        title="Profit Distribution by SKU",
        y_label="Profit",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Monthly Patterns

    Aggregating by month shows seasonal profit cycles across the year.
    """)


@app.cell
def _(df, plot_seasonality):
    _first_col = [c for c in df.columns if c.endswith("__profit")][0]
    plot_seasonality(
        df,
        columns=_first_col,
        seasonality="month",
        title="T1 - Average Profit by Month",
    )



if __name__ == "__main__":
    app.run()
