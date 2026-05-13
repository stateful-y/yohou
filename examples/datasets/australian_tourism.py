# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.19.11"
__gallery__ = {
    "title": "Tourism Quarterly",
    "description": "Explore the Tourism Quarterly panel dataset with inspect_panel structure analysis, cross-group demand comparison, and seasonal boxplots across 8 series.",
    "category": "how-to",
    "section": "data-catalog",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from yohou.datasets import fetch_tourism_quarterly
    from yohou.plotting import (
        plot_boxplot,
        plot_seasonality,
        plot_time_series,
    )
    from yohou.utils.panel import inspect_panel

    return (
        fetch_tourism_quarterly,
        inspect_panel,
        mo,
        plot_boxplot,
        plot_seasonality,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Tourism Quarterly Dataset

    This notebook shows how to explore the Tourism Quarterly panel dataset with inspect_panel structure analysis, cross-group demand comparison, and seasonal boxplots across 8 series.

    **Prerequisites:** None. this is a standalone dataset exploration.
    """)


@app.cell
def _(fetch_tourism_quarterly, plot_time_series):
    _all = fetch_tourism_quarterly().frame
    # Select first 8 series for a manageable panel
    _cols = ["time"] + [c for c in _all.columns if c != "time"][:8]
    df = _all.select(_cols)
    plot_time_series(df, title="Australian Tourism (8 Series)")
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Inspect Panel Structure

    The full dataset has 427 panel groups using the `Tn__tourists`
    convention. Here we work with the first 8 series.
    """)


@app.cell
def _(df, inspect_panel, mo):
    global_cols, panel_groups = inspect_panel(df)
    mo.md(f"""
    **Global columns**: {global_cols}

    **Panel groups** ({len(panel_groups)} groups):

    {chr(10).join(f"- **{k}**: {v}" for k, v in panel_groups.items())}
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. First Three Series Comparison

    Comparing the first three tourism series shows demand patterns and
    seasonal differences across panel groups.
    """)


@app.cell
def _(df, plot_time_series):
    major = [c for c in df.columns if c.endswith("__tourists")][:3]

    plot_time_series(
        df,
        columns=major,
        title="Tourism Quarterly - First 3 Series",
        y_label="Tourists",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. All Selected Series Overview

    Plotting all 8 series together reveals scale differences and shared
    seasonal patterns.
    """)


@app.cell
def _(df, plot_time_series):
    all_trip_cols = [c for c in df.columns if c.endswith("__tourists")]

    plot_time_series(
        df,
        columns=all_trip_cols,
        title="Tourism Quarterly - 8 Series",
        y_label="Tourists",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Quarterly Seasonality

    Quarterly aggregation highlights peak tourism seasons across the year.
    """)


@app.cell
def _(df, plot_seasonality):
    _first_col = [c for c in df.columns if c.endswith("__tourists")][0]
    plot_seasonality(
        df,
        columns=_first_col,
        seasonality="quarter",
        title="T1 - Average Tourism by Quarter",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Annual Distribution

    Box plots show the year-to-year variability in tourism demand for each series.
    """)


@app.cell
def _(df, plot_boxplot):
    plot_boxplot(
        df,
        period="1y",
        title="Annual Tourism Distribution",
    )



if __name__ == "__main__":
    app.run()
