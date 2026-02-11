"""Australian Tourism - State-Level Panel Data.

Quarterly tourism trips aggregated by Australian state.

Dataset: 8 states, 80 quarterly observations (1998-2017)
Demonstrates: inspect_locality, plot_time_series, plot_seasonality, plot_boxplot
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from yohou.datasets import load_australian_tourism
    from yohou.plotting import (
        plot_boxplot,
        plot_seasonality,
        plot_time_series,
    )
    from yohou.utils.panel import inspect_locality

    return (
        inspect_locality,
        load_australian_tourism,
        mo,
        plot_boxplot,
        plot_seasonality,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Australian Tourism Dataset

    ## What You'll Learn

    This example demonstrates quarterly panel data analysis with the Australian
    Tourism dataset, pre-formatted in Yohou's native `__` panel convention. You'll
    learn how to:

    - Inspect panel structure with `inspect_locality`
    - Compare tourism demand across Australian states
    - Analyze quarterly seasonal patterns
    - Use box plots for distribution analysis

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
def _(load_australian_tourism):
    df = load_australian_tourism()
    df.head(10)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Inspect Panel Structure

    The dataset has 8 panel groups, one per Australian state, using
    the `state_name__trips` convention.
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
    ## 2. Major States Comparison

    Comparing NSW and Victoria, the two largest tourism markets, shows
    dominant demand patterns and seasonal differences.
    """)
    return


@app.cell
def _(df, plot_time_series):
    major = [
        "new_south_wales__trips",
        "victoria__trips",
        "queensland__trips",
    ]

    plot_time_series(
        df,
        columns=major,
        title="Tourism by Major States",
        y_label="Trips",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. All States Overview

    Plotting all 8 states together reveals the scale differences and shared
    seasonal patterns across Australia.
    """)
    return


@app.cell
def _(df, plot_time_series):
    all_trip_cols = [c for c in df.columns if c.endswith("__trips")]

    plot_time_series(
        df,
        columns=all_trip_cols,
        title="Tourism - All States",
        y_label="Trips",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Quarterly Seasonality

    Quarterly aggregation highlights peak tourism seasons, particularly Q1 and
    Q4 (summer/holiday periods in Australia).
    """)
    return


@app.cell
def _(df, plot_seasonality):
    plot_seasonality(
        df,
        columns="new_south_wales__trips",
        feature="quarter",
        aggregation="mean",
        title="NSW - Average Tourism by Quarter",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Annual Distribution

    Box plots show the year-to-year variability in tourism demand for each state.
    """)
    return


@app.cell
def _(df, plot_boxplot):
    plot_boxplot(
        df,
        period="1y",
        title="Annual Tourism Distribution",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **Native panel format**: Columns use `state_name__trips` convention: no pivoting needed
    - **Quarterly frequency**: 80 observations spanning 20 years (1998–2017)
    - **State-level aggregation**: Pre-aggregated from regional data for clean panel analysis
    - **Seasonal patterns**: Q1 and Q4 are typically peak tourism quarters
    - **Scale differences**: NSW and Victoria dominate, but all states show consistent patterns

    ## Next Steps

    - For daily panel data, see `examples/datasets/store_sales.py`
    - For branch-level panel, see `examples/datasets/walmart_sales.py`
    - **Panel forecasting**: See `examples/datasets/australian_tourism_forecasting.py` for end-to-end panel forecasting
    """)
    return


if __name__ == "__main__":
    app.run()
