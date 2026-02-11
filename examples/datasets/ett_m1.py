"""ETTm1 - Electricity Transformer Multivariate Analysis.

Multivariate analysis of transformer temperature and load features.

Dataset: 7 temperature features at 15-minute intervals
Demonstrates: plot_time_series, plot_cross_correlation, plot_seasonality
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from yohou.datasets import load_ett_m1
    from yohou.plotting import (
        plot_cross_correlation,
        plot_seasonality,
        plot_time_series,
    )

    return (
        load_ett_m1,
        mo,
        plot_cross_correlation,
        plot_seasonality,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # ETTm1 Dataset

    ## What You'll Learn

    This example demonstrates multivariate time series analysis with the ETTm1
    (Electricity Transformer Temperature) dataset. You'll learn how to:

    - Visualize multiple correlated features simultaneously
    - Analyze cross-correlation between temperature and load features
    - Identify hourly and weekly seasonality patterns in industrial IoT data
    - Understand lead-lag relationships in multivariate systems

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
def _(load_ett_m1):
    df = load_ett_m1()
    df.head(20)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Multiple Features Visualization

    Plotting all features together shows how oil temperature (OT) and load
    variables co-move over time.
    """)
    return


@app.cell
def _(df, plot_time_series):
    df_week = df.head(96 * 7)

    plot_time_series(
        df_week,
        columns=["OT", "HUFL", "MUFL", "LUFL"],
        title="Oil Temperature and Load Features (7 Days)",
        y_label="Temperature",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Cross-Correlation with High Load

    Cross-correlation with HUFL (High UseFul Load) quantifies the strength and
    timing of the temperature-load relationship.
    """)
    return


@app.cell
def _(df, plot_cross_correlation):
    df_month = df.head(96 * 30)

    plot_cross_correlation(
        df_month,
        x_column="OT",
        y_column="HUFL",
        lags=96,
        title="Oil Temperature vs High Load Cross-Correlation",
    )
    return (df_month,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Cross-Correlation with Medium Load

    Comparing cross-correlation with MUFL reveals whether different load
    categories have different lag structures.
    """)
    return


@app.cell
def _(df_month, plot_cross_correlation):
    plot_cross_correlation(
        df_month,
        x_column="OT",
        y_column="MUFL",
        lags=96,
        title="Oil Temperature vs Medium Load Cross-Correlation",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Hourly Seasonality

    Aggregating by hour reveals the 24-hour demand cycle driven by business
    hours and daily routines.
    """)
    return


@app.cell
def _(df, plot_seasonality):
    plot_seasonality(
        df,
        columns="OT",
        feature="hour",
        aggregation="mean",
        title="Average Oil Temperature by Hour",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Day-of-Week Patterns

    Weekly patterns show slightly higher temperatures on weekdays compared to
    weekends.
    """)
    return


@app.cell
def _(df, plot_seasonality):
    plot_seasonality(
        df,
        columns="OT",
        feature="dayofweek",
        aggregation="mean",
        title="Average Oil Temperature by Day of Week",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **Strong feature correlation**: Oil temperature (OT) closely tracks load features
    - **High load dominates**: HUFL (High UseFul Load) shows strongest correlation with target
    - **Near-instantaneous response**: Cross-correlation peaks near zero lag
    - **24-hour cycles**: Clear intraday patterns driven by business hours
    - **Weekday effects**: Temperature slightly higher on weekdays vs weekends
    - **15-minute resolution**: Captures rapid temperature fluctuations

    ## Next Steps

    - For simpler multivariate data, see `examples/datasets/vic_electricity.py`
    - For high-frequency panel data, see `examples/datasets/vic_electricity.py`
    - For univariate cyclic patterns, see `examples/datasets/sunspots.py`
    """)
    return


if __name__ == "__main__":
    app.run()
