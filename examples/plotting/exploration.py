# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "yohou",
# ]
# ///
"""Exploratory Time Series Visualization.

Demonstrates four exploration plotting functions with varied parameter
combinations across univariate, multivariate, and panel datasets.

Datasets: tourism_monthly, electricity_demand, tourism_quarterly, dominick
Demonstrates: plot_time_series, plot_rolling_statistics,
    plot_boxplot, plot_missing_data
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)

@app.cell(hide_code=True)
def _():
    import polars as pl

    from yohou.datasets import (
        fetch_dominick,
        fetch_electricity_demand,
        fetch_tourism_monthly,
        fetch_tourism_quarterly,
    )
    from yohou.plotting import (
        plot_boxplot,
        plot_missing_data,
        plot_rolling_statistics,
        plot_time_series,
    )

    return (
        fetch_dominick,
        fetch_electricity_demand,
        fetch_tourism_monthly,
        fetch_tourism_quarterly,
        pl,
        plot_boxplot,
        plot_missing_data,
        plot_rolling_statistics,
        plot_time_series,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploratory Time Series Visualization

    ## What You'll Learn

    - How to visualize raw time series across univariate, multivariate, and panel datasets
    - Using rolling statistics with different window sizes, statistics, and envelope bands
    - Generating boxplot distributions grouped by different periods
    - Auditing missing data with heatmap, bar, and matrix visualizations

    ## Prerequisites

    None -- this is a standalone exploration tutorial.
    """)

@app.cell
def _(
    fetch_tourism_monthly,
    fetch_tourism_quarterly,
    fetch_dominick,
    fetch_electricity_demand,
):
    tourism_monthly = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    vic = fetch_electricity_demand().frame
    tourism_quarterly = fetch_tourism_quarterly().frame
    _dom_full = fetch_dominick().frame
    _profit_cols = [c for c in _dom_full.columns if c.endswith("__profit")][:6]
    store = _dom_full.select("time", *_profit_cols)
    return tourism_monthly, tourism_quarterly, store, vic

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Raw Time Series

    `plot_time_series` renders one or more numeric columns against the `"time"` axis.
    Varying **columns**, **panel_group_names**, and styling kwargs shows how the same
    function adapts to different data shapes.
    """)

@app.cell
def _(tourism_monthly, plot_time_series):
    plot_time_series(tourism_monthly, title="Monthly Tourism -- Single Column")

@app.cell
def _(plot_time_series, vic):
    plot_time_series(
        vic,
        columns=["vic__demand", "nsw__demand"],
        title="Electricity Demand -- Multi-Column Overlay",
    )

@app.cell
def _(plot_time_series, tourism_quarterly):
    plot_time_series(
        tourism_quarterly,
        panel_group_names=["T1", "T2"],
        title="Tourism Quarterly -- Panel Faceting (T1 & T2)",
    )

@app.cell
def _(tourism_monthly, plot_time_series):
    plot_time_series(
        tourism_monthly,
        line_dash="dash",
        color_palette=["#DC2626"],
        title="Monthly Tourism -- Dashed Red Line",
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Rolling Statistics

    `plot_rolling_statistics` computes windowed statistics over a time series.
    Key parameters include **window_size**, **statistics** (single string or list),
    **show_original**, and the **fill_between** kwarg for envelope bands.
    """)

@app.cell
def _(tourism_monthly, plot_rolling_statistics):
    plot_rolling_statistics(
        tourism_monthly,
        window_size=12,
        statistics="mean",
        title="12-Month Rolling Mean",
    )

@app.cell
def _(tourism_monthly, plot_rolling_statistics):
    plot_rolling_statistics(
        tourism_monthly,
        window_size=12,
        statistics=["mean", "std"],
        title="12-Month Rolling Mean and Standard Deviation",
    )

@app.cell
def _(tourism_monthly, plot_rolling_statistics):
    plot_rolling_statistics(
        tourism_monthly,
        window_size=12,
        statistics=["min", "max", "mean"],
        fill_between=True,
        title="12-Month Min/Max Envelope with Mean",
    )

@app.cell
def _(plot_rolling_statistics, vic):
    plot_rolling_statistics(
        vic,
        columns="vic__demand",
        window_size=48,
        statistics="mean",
        show_original=False,
        title="48-Step Rolling Mean on Vic Demand (Smoothed Only)",
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Distribution by Period

    `plot_boxplot` groups values by a time **period** and shows the distribution
    within each group. Vary the period, toggle **show_points**, or pass
    **panel_group_names** for panel data.
    """)

@app.cell
def _(tourism_monthly, plot_boxplot):
    plot_boxplot(tourism_monthly, period="1mo", title="Monthly Boxplot (Monthly Tourism)")

@app.cell
def _(tourism_monthly, plot_boxplot):
    plot_boxplot(tourism_monthly, period="1q", title="Quarterly Boxplot (Monthly Tourism)")

@app.cell
def _(tourism_monthly, plot_boxplot):
    plot_boxplot(
        tourism_monthly,
        period="1mo",
        show_points="all",
        title="Monthly Boxplot -- All Points Shown",
    )

@app.cell
def _(plot_boxplot, tourism_quarterly):
    plot_boxplot(
        tourism_quarterly,
        period="1y",
        panel_group_names=["T1", "T2"],
        title="Yearly Boxplot -- Tourism Quarterly Panel (T1 & T2)",
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Missing Data Audit

    `plot_missing_data` visualises null patterns using three **kind** options:
    `"heatmap"`, `"bars"`, and `"matrix"`. Additional kwargs like
    **time_aggregation** and custom colors provide further control.
    """)

@app.cell(hide_code=True)
def _(pl, store):
    import numpy as np

    _rng = np.random.default_rng(42)
    _cols = [c for c in store.columns if c != "time"]
    _mask = {
        c: [
            None if _rng.random() < 0.05 else v
            for v in store[c].to_list()
        ]
        for c in _cols
    }
    store_nan = store.select("time").with_columns(
        [pl.Series(c, vals) for c, vals in _mask.items()]
    )
    return (store_nan,)

@app.cell
def _(plot_missing_data, store_nan):
    plot_missing_data(store_nan, kind="heatmap", title="Missing Data -- Heatmap")

@app.cell
def _(plot_missing_data, store_nan):
    plot_missing_data(store_nan, kind="bars", title="Missing Data -- Bar Chart")

@app.cell
def _(plot_missing_data, store_nan):
    plot_missing_data(store_nan, kind="matrix", title="Missing Data -- Matrix")

@app.cell
def _(plot_missing_data, store_nan):
    plot_missing_data(
        store_nan,
        kind="heatmap",
        time_aggregation="1mo",
        color_missing="#000000",
        color_present="#22c55e",
        title="Missing Data -- Monthly Aggregation, Custom Colors",
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **plot_time_series** adapts to single-column, multi-column, and panel data via `columns` and `panel_group_names`
    - **Rolling statistics** reveal trends (`"mean"`), variability (`"std"`), and range (`"min"/"max"` with `fill_between`)
    - **Boxplots** aggregate at different time granularities (`period`); `show_points="all"` shows every observation
    - **Missing data** visualizations come in three forms: `"heatmap"` for patterns over time, `"bars"` for column-level counts, `"matrix"` for a compact binary view

    ## Next Steps

    - **Correlation diagnostics**: See `examples/plotting/correlation.py` for scatter matrices and cross-correlation
    - **Seasonal analysis**: See `examples/plotting/seasonal.py` for seasonality overlays and frequency-domain plots
    - **Forecast visualization**: See `examples/plotting/forecasting_visualization.py` for model comparison and prediction intervals
    """)

if __name__ == "__main__":
    app.run()
