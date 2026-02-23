"""Tourism Monthly - Trend and Seasonality Analysis.

Monthly tourism time series from the Monash forecasting competition.

Dataset: 366 monthly tourism series (exploring first series T1)
Demonstrates: plot_time_series, plot_rolling_statistics, plot_seasonality
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from yohou.datasets import fetch_tourism_monthly
    from yohou.plotting import (
        plot_rolling_statistics,
        plot_seasonality,
        plot_time_series,
    )

    return (
        fetch_tourism_monthly,
        mo,
        plot_rolling_statistics,
        plot_seasonality,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Tourism Monthly Dataset

    ## What You'll Learn

    - Visualize raw monthly tourism time series with trend
    - Apply rolling statistics to smooth and highlight patterns
    - Analyze seasonal patterns across different time periods

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
def _(fetch_tourism_monthly):
    bunch = fetch_tourism_monthly()
    df = bunch.frame.select("time", "T1__tourists").rename({"T1__tourists": "Passengers"})
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Raw Time Series Visualization

    We explore the first series (T1) from the Tourism Monthly collection, renamed to "Passengers" for readability. The data reveals trend and seasonal patterns in monthly tourism counts.
    """)
    return


@app.cell
def _(df, plot_time_series):
    plot_time_series(
        df,
        title="Tourism Monthly - Series T1",
        x_label="Year",
        y_label="Passengers",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Rolling Statistics - Mean with Original

    A 12-month rolling average smooths out seasonal noise and highlights the underlying trend.
    """)
    return


@app.cell
def _(df, plot_rolling_statistics):
    plot_rolling_statistics(
        df,
        window_size=12,
        statistics="mean",
        show_original=True,
        title="12-Month Rolling Average",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Rolling Statistics - Min/Max Envelope

    The min/max envelope shows how the range of tourism counts varies over time, highlighting seasonality amplitude changes.
    """)
    return


@app.cell
def _(df, plot_rolling_statistics):
    plot_rolling_statistics(
        df,
        window_size=12,
        statistics=["min", "max"],
        fill_between=True,
        show_original=False,
        title="12-Month Range Envelope",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Seasonal Pattern Analysis

    Aggregating by month reveals the seasonal shape of tourism demand across the year.
    """)
    return


@app.cell
def _(df, plot_seasonality):
    plot_seasonality(
        df,
        feature="month",
        aggregation="mean",
        title="Average Tourism by Month",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **Tourism Monthly**: 366 monthly tourism series from Monash competition, explored here via series T1
    - **Seasonal patterns**: Monthly aggregation reveals peak tourism periods
    - **Rolling statistics reveal trend**: 12-month moving average smooths seasonal noise
    - **Parameter variability**: `show_original`, `fill_between`, and multiple statistics demonstrate function flexibility

    ## Next Steps

    - For cyclic patterns without trend, see `examples/datasets/sunspots.py`
    - For panel data with multiple series, see `examples/datasets/store_sales.py`
    - For high-frequency data, see `examples/datasets/vic_electricity.py`
    """)
    return


if __name__ == "__main__":
    app.run()
