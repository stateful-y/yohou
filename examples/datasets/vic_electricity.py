"""Victoria Electricity - Multivariate Analysis.

Electricity demand with temperature and cross-correlation analysis.

Dataset: 30-min electricity demand and temperature
Demonstrates: plot_time_series, plot_cross_correlation, plot_seasonality, plot_rolling_statistics
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from yohou.datasets import load_vic_electricity
    from yohou.plotting import (
        plot_cross_correlation,
        plot_rolling_statistics,
        plot_seasonality,
        plot_time_series,
    )

    return (
        load_vic_electricity,
        mo,
        plot_cross_correlation,
        plot_rolling_statistics,
        plot_seasonality,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Victoria Electricity Dataset

    ## What You'll Learn

    This example demonstrates high-frequency time series analysis with the
    Victoria Electricity dataset. You'll learn how to:

    - Visualize 30-minute electricity demand patterns
    - Analyze demand with temperature as an external covariate
    - Use cross-correlation to find lagged relationships
    - Apply intraday seasonality analysis (hour of day)

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
def _(load_vic_electricity):
    df = load_vic_electricity()
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Demand and Temperature Together

    Plotting demand and temperature on the same chart reveals their co-movement:
    demand rises with extreme temperatures.
    """)
    return


@app.cell
def _(df, plot_time_series):
    plot_time_series(
        df,
        columns=["Demand", "Temperature"],
        title="Electricity Demand and Temperature",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Demand Rolling Average

    A rolling average of demand highlights the underlying daily and weekly patterns
    beneath the noisy 30-minute readings.
    """)
    return


@app.cell
def _(df, plot_rolling_statistics):
    plot_rolling_statistics(
        df,
        columns="Demand",
        window_size=48,  # 24 hours
        statistics="mean",
        show_original=True,
        title="24-Hour Rolling Average",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Temperature Range

    The min/max envelope of temperature shows the daily temperature swing and how
    it varies across seasons.
    """)
    return


@app.cell
def _(df, plot_rolling_statistics):
    plot_rolling_statistics(
        df,
        columns="Temperature",
        window_size=48,
        statistics=["mean", "min", "max"],
        fill_between=True,
        show_original=False,
        title="24-Hour Temperature Min/Mean/Max",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Cross-Correlation Analysis

    Cross-correlation quantifies the lagged relationship between temperature and
    electricity demand.
    """)
    return


@app.cell
def _(df, plot_cross_correlation):
    plot_cross_correlation(
        df,
        columns=["Temperature", "Demand"],
        max_lags=100,
        title="Temperature vs Demand Cross-Correlation",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Hourly Patterns

    Aggregating by hour of day reveals the demand peak times and the intraday
    consumption pattern.
    """)
    return


@app.cell
def _(df, plot_seasonality):
    plot_seasonality(
        df,
        columns="Demand",
        seasonality="hour",
        title="Average Demand by Hour",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **High frequency**: 30-minute data captures intraday patterns
    - **External regressors**: Temperature as a covariate explains demand variation
    - **Cross-correlation**: Identifies lagged relationship between temperature and demand
    - **Multiple statistics**: `["mean", "min", "max"]` with `fill_between=True` shows range
    - **Intraday seasonality**: Clear hour-of-day effects (peak demand times)

    ## Next Steps

    - For hourly data, see `examples/datasets/ett_m1.py`
    - For daily data with external features, see `examples/datasets/walmart_sales.py`
    - For panel data, see `examples/datasets/store_sales.py`
    """)
    return


if __name__ == "__main__":
    app.run()
