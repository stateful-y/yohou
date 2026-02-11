"""Air Passengers - Trend and Seasonality Analysis.

Classic time series with strong trend and monthly seasonality.

Dataset: Monthly airline passengers, 1949-1960
Demonstrates: plot_time_series, plot_rolling_statistics, plot_seasonality
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from yohou.datasets import load_air_passengers
    from yohou.plotting import (
        plot_rolling_statistics,
        plot_seasonality,
        plot_time_series,
    )

    return (
        load_air_passengers,
        mo,
        plot_rolling_statistics,
        plot_seasonality,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Air Passengers Dataset

    ## What You'll Learn

    - Visualize raw time series data with clear trends
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
def _(load_air_passengers):
    df = load_air_passengers()
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Raw Time Series Visualization

    The raw data shows monthly airline passenger totals from 1949 to 1960, revealing a strong upward trend with seasonal fluctuations.
    """)
    return


@app.cell
def _(df, plot_time_series):
    plot_time_series(
        df,
        title="Air Passengers (1949-1960)",
        x_label="Year",
        y_label="Passengers (thousands)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Rolling Statistics - Mean with Original

    A 12-month rolling average smooths out seasonal noise and highlights the underlying growth trend.
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

    The min/max envelope shows how the range of passenger counts widens over time -- a hallmark of multiplicative seasonality.
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

    Aggregating by month reveals the seasonal shape: summer months (July/August) consistently have the highest passenger numbers.
    """)
    return


@app.cell
def _(df, plot_seasonality):
    plot_seasonality(
        df,
        feature="month",
        aggregation="mean",
        title="Average Passengers by Month",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **Strong exponential trend**: Passenger numbers grow dramatically from 1949 to 1960
    - **Multiplicative seasonality**: Seasonal variations increase proportionally with the trend
    - **Peak summer months**: July and August consistently show highest passenger numbers
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
