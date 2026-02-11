"""Store Sales - Calendar Heatmaps and Retail Patterns.

Retail sales visualization using store and item-level data.

Dataset: Daily sales for 10 stores × 50 items
Demonstrates: plot_calendar_heatmap, plot_timeseries, plot_seasonality, plot_boxplot
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    from yohou.datasets import load_store_sales
    from yohou.plotting import (
        plot_boxplot,
        plot_calendar_heatmap,
        plot_seasonality,
        plot_timeseries,
    )

    return (
        load_store_sales,
        pl,
        plot_boxplot,
        plot_calendar_heatmap,
        plot_seasonality,
        plot_timeseries,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Store Sales Dataset

    Daily sales data for 10 stores and 50 items (500 time series).
    This panel dataset is ideal for demonstrating:
    - Calendar heatmaps showing day-of-week patterns
    - Store-level aggregations
    - Item performance comparison
    - Weekly and monthly seasonality
    """)
    return


@app.cell
def _(load_store_sales):
    # Load panel dataset
    df = load_store_sales()
    df.head(10)
    return (df,)


@app.cell
def _(df):
    # Dataset structure
    n_stores = df["store"].n_unique()
    n_items = df["item"].n_unique()
    date_range = (df["time"].min(), df["time"].max())

    print(f"Stores: {n_stores}")
    print(f"Items: {n_items}")
    print(f"Date range: {date_range[0]} to {date_range[1]}")
    print(f"Total series: {n_stores * n_items}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Calendar Heatmap - Total Sales

    Visualize daily sales patterns across the entire year for all stores combined.
    Calendar heatmaps show weekday vs weekend effects and identify specific
    high/low sales days.
    """)
    return


@app.cell
def _(df, pl, plot_calendar_heatmap):
    # Aggregate all stores and items for calendar view
    df_total = df.group_by("time").agg(pl.col("sales").sum())

    # Get the first complete year
    first_year = df_total["time"].dt.year().min()

    fig1 = plot_calendar_heatmap(
        df_total,
        column="sales",
        aggregation="sum",
        year=first_year,
        colorscale="Viridis",
        title=f"Total Daily Sales - {first_year} (All Stores)",
    )
    fig1
    return (first_year,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Store-Level Calendar Pattern
    """)
    return


@app.cell
def _(df, first_year, pl, plot_calendar_heatmap):
    # Calendar for a single store (store ID is integer)
    df_store_1 = df.filter(pl.col("store") == 1).group_by("time").agg(pl.col("sales").sum())

    fig2 = plot_calendar_heatmap(
        df_store_1,
        column="sales",
        aggregation="sum",
        year=first_year,
        colorscale="Blues",
        title=f"Store 1 Daily Sales - {first_year}",
    )
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Time Series by Store
    """)
    return


@app.cell
def _(df, pl, plot_timeseries):
    # Aggregate by store and time
    df_by_store = df.group_by(["store", "time"]).agg(pl.col("sales").sum()).sort("time")

    # Select first 3 stores (IDs are integers) and pivot to wide format
    stores_subset = [1, 2, 3]
    df_subset = df_by_store.filter(pl.col("store").is_in(stores_subset))

    # Pivot to wide format for multivariate plot
    df_wide = df_subset.pivot(
        on="store",
        index="time",
        values="sales",
    ).sort("time")

    # Rename columns to strings for plotting
    df_wide = df_wide.rename({str(i): f"Store {i}" for i in stores_subset})

    fig3 = plot_timeseries(
        df_wide,
        columns=[f"Store {i}" for i in stores_subset],
        title="Daily Sales by Store (3 Stores)",
        x_label="Time",
        y_label="Sales",
    )
    fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Day-of-Week Patterns
    """)
    return


@app.cell
def _(df, pl, plot_seasonality):
    # Aggregate to daily totals first to avoid plotting 500 series
    df_daily_total = df.group_by("time").agg(pl.col("sales").sum())

    fig4 = plot_seasonality(
        df_daily_total,
        columns="sales",
        feature="dayofweek",
        aggregation="mean",
        title="Average Sales by Day of Week (All Stores)",
    )
    fig4
    return (df_daily_total,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Weekly Distribution
    """)
    return


@app.cell
def _(df_daily_total, plot_boxplot):
    # Weekly boxplot showing variability (using aggregated daily totals)
    fig5 = plot_boxplot(
        df_daily_total,
        columns="sales",
        period="1w",
        title="Weekly Sales Distribution (All Stores)",
    )
    fig5
    return (fig5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Top-Performing Items
    """)
    return


@app.cell
def _(df, pl):
    # Find top 5 items by total sales
    top_items = (
        df.group_by("item")
        .agg(pl.col("sales").sum().alias("total_sales"))
        .sort("total_sales", descending=True)
        .head(5)
    )
    top_items
    return (top_items,)


@app.cell
def _(df, pl, plot_timeseries, top_items):
    # Plot top items over time (aggregated across stores)
    top_item_ids = top_items["item"].to_list()
    df_top_items = (
        df.filter(pl.col("item").is_in(top_item_ids))
        .group_by(["item", "time"])
        .agg(pl.col("sales").sum())
        .sort("time")
    )

    # Pivot to wide format (item IDs are integers)
    df_items_wide = df_top_items.pivot(
        on="item",
        index="time",
        values="sales",
    ).sort("time")

    # Rename columns to strings
    df_items_wide = df_items_wide.rename({str(i): f"Item {i}" for i in top_item_ids})

    fig6 = plot_timeseries(
        df_items_wide,
        columns=[f"Item {i}" for i in top_item_ids],
        title="Top 5 Items - Daily Sales (All Stores)",
        x_label="Time",
        y_label="Sales",
    )
    fig6
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Monthly Seasonality
    """)
    return


@app.cell
def _(df_daily_total, plot_seasonality):
    # Month-level patterns (using aggregated daily totals)
    fig7 = plot_seasonality(
        df_daily_total,
        columns="sales",
        feature="month",
        aggregation="sum",
        title="Total Sales by Month (All Stores)",
    )
    fig7
    return (fig7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    The Store Sales dataset demonstrates:

    - **Calendar heatmaps**: Excellent for identifying weekly patterns and specific high-sales days
    - **Panel data pivoting**: Convert long format to wide for multivariate visualization
    - **Weekday effects**: Clear difference between weekday and weekend sales
    - **Seasonal patterns**: Monthly and weekly trends visible across the dataset
    - **Item rankings**: Identifying top performers for inventory management

    Key insights for retail analytics:
    - Calendar heatmaps reveal day-of-week effects (weekends differ from weekdays)
    - Store-level aggregation enables location-specific analysis
    - Item-level tracking enables SKU optimization
    - Weekly aggregation smooths daily noise while preserving trends

    **Note**: Full panel support with `panel_group_name` and interactive dropdowns
    is planned for future releases. Current workflows use filtering and pivoting.

    This workflow applies to any retail, e-commerce, or transaction-based time series.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
