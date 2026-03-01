# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.19.11"
__gallery__ = {
    "title": "Tourism Monthly",
    "description": "Explore a single monthly tourism series with raw trend visualisation, 12-month rolling mean smoothing, and min/max envelope seasonality analysis.",
}
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

    None - this is a standalone dataset exploration.
    """)


@app.cell
def _(fetch_tourism_monthly):
    bunch = fetch_tourism_monthly()
    df = bunch.frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Raw Time Series Visualization

    We explore the first series (T1) from the Tourism Monthly collection, renamed to "tourists" for readability. The data reveals trend and seasonal patterns in monthly tourism counts.
    """)


@app.cell
def _(df, plot_time_series):
    plot_time_series(
        df,
        title="Tourism Monthly - Series T1",
        x_label="Year",
        y_label="tourists",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Rolling Statistics - Mean with Original

    A 12-month rolling average smooths out seasonal noise and highlights the underlying trend.
    """)


@app.cell
def _(df, plot_rolling_statistics):
    plot_rolling_statistics(
        df,
        window_size=12,
        statistics="mean",
        show_original=True,
        title="12-Month Rolling Average",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Rolling Statistics - Min/Max Envelope

    The min/max envelope shows how the range of tourism counts varies over time, highlighting seasonality amplitude changes.
    """)


@app.cell
def _(df, plot_rolling_statistics):
    plot_rolling_statistics(
        df,
        window_size=12,
        statistics=["min", "max"],
        show_original=False,
        title="12-Month Range Envelope",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Seasonal Pattern Analysis

    Aggregating by month reveals the seasonal shape of tourism demand across the year.
    """)


@app.cell
def _(df, plot_seasonality):
    plot_seasonality(
        df,
        seasonality="month",
        title="Average Tourism by Month",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **Tourism Monthly**: 366 monthly tourism series from Monash competition, explored here via series T1
    - **Seasonal patterns**: Monthly aggregation reveals peak tourism periods
    - **Rolling statistics reveal trend**: 12-month moving average smooths seasonal noise
    - **Parameter variability**: `show_original` and multiple statistics demonstrate function flexibility

    ## Next Steps

    - For cyclic patterns without trend, see [`examples/datasets/sunspots.py`](/examples/datasets/sunspots/)
    - For panel data with multiple series, see [`examples/datasets/store_sales.py`](/examples/datasets/store_sales/)
    - For high-frequency data, see [`examples/datasets/vic_electricity.py`](/examples/datasets/vic_electricity/)
    """)


if __name__ == "__main__":
    app.run()
