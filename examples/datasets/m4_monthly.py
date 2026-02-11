"""M4 Monthly - Panel Data with Facets and Dropdowns.

Panel data visualization using M4 Monthly competition data.

Dataset: 50 monthly time series from M4 competition
Demonstrates: plot_timeseries, plot_boxplot, plot_seasonality
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    from yohou.datasets import load_m4_monthly
    from yohou.plotting import (
        plot_boxplot,
        plot_seasonality,
        plot_timeseries,
    )
    return (
        load_m4_monthly,
        pl,
        plot_boxplot,
        plot_seasonality,
        plot_timeseries,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # M4 Monthly Dataset (Panel Data)

        Panel of 50 monthly time series from the M4 Forecasting Competition.
        Each series represents a different forecasting task with unique characteristics.

        This example demonstrates how to visualize multiple time series simultaneously
        using facets and interactive dropdowns.
        """
    )
    return


@app.cell
def _(load_m4_monthly):
    # Load panel dataset
    df = load_m4_monthly()
    df.head(15)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 1. Single Series Visualization

        With panel data, we first filter to a single series for visualization.
        Future versions will support interactive dropdowns via `panel_group_name`.
        """
    )
    return


@app.cell
def _(df, plot_timeseries):
    # Select first series for visualization
    df_single = df.filter(pl.col("unique_id") == "M1")

    fig1 = plot_timeseries(
        df_single,
        title="M4 Series M1 - Monthly Data",
        x_label="Time",
        y_label="Value",
    )
    fig1
    return df_single, fig1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 2. Multiple Series Comparison

        Compare a few selected series on the same plot.
        Future versions will support facets for automatic small multiples.
        """
    )
    return


@app.cell
def _(df, plot_timeseries):
    # Compare 3 series with different patterns
    series_subset = ["M1", "M10", "M20"]
    df_comparison = df.filter(pl.col("unique_id").is_in(series_subset))

    # Since panel_group_name not implemented, we plot each series as a column
    # Reshape: pivot to wide format
    df_wide = df_comparison.pivot(
        on="unique_id",
        index="time",
        values="y",
    ).sort("time")

    fig2 = plot_timeseries(
        df_wide,
        columns=series_subset,
        title="M4 Monthly - Comparing 3 Series",
        x_label="Time",
        y_label="Value",
    )
    fig2
    return df_comparison, df_wide, fig2, series_subset


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 3. Panel Data Statistics

        Analyze distributional properties across all series.
        """
    )
    return


@app.cell
def _(df):
    # Summary statistics by series
    summary = (
        df.group_by("unique_id")
        .agg([
            pl.col("y").count().alias("n_obs"),
            pl.col("y").mean().alias("mean"),
            pl.col("y").std().alias("std"),
            pl.col("y").min().alias("min"),
            pl.col("y").max().alias("max"),
        ])
        .sort("unique_id")
    )
    summary.head(10)
    return (summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 4. Boxplot for Panel Data

        Visualize value distributions aggregated across all series.
        This shows the overall pattern in the M4 monthly data.
        """
    )
    return


@app.cell
def _(df, pl, plot_boxplot):
    # Aggregate all series to show overall monthly patterns
    # Take first 12 months from each series for comparison
    df_first_year = (
        df.sort(["unique_id", "time"])
        .group_by("unique_id")
        .head(12)
        .with_row_index()
        .with_columns(
            (pl.col("index") % 12 + 1).alias("month_of_year")
        )
        .with_columns(
            pl.datetime(2000, pl.col("month_of_year"), 1).alias("aligned_time")
        )
        .drop(["index", "month_of_year", "unique_id", "time"])
        .rename({"aligned_time": "time"})
    )

    fig3 = plot_boxplot(
        df_first_year,
        columns="y",
        period="1mo",
        title="M4 Monthly Distribution (First Year Across All Series)",
    )
    fig3
    return df_first_year, fig3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 5. Seasonal Patterns Across Panel

        Examine if there are common seasonal patterns across all series.
        """
    )
    return


@app.cell
def _(df, pl, plot_seasonality):
    # Pre-aggregate across all 50 series to avoid plotting too much data
    df_monthly_agg = (
        df.with_columns(pl.col("time").dt.month().alias("month"))
        .group_by("month")
        .agg(pl.col("y").median().alias("y"))
        .sort("month")
        .with_columns(
            pl.datetime(2000, pl.col("month"), 1).alias("time")
        )
    )

    fig4 = plot_seasonality(
        df_monthly_agg,
        columns="y",
        feature="month",
        aggregation="mean",  # Just mean since already aggregated
        title="Median Seasonal Pattern by Month (All Series)",
    )
    fig4
    return (df_monthly_agg, fig4)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 6. Individual Series Analysis

        Analyze specific series in detail.
        """
    )
    return


@app.cell
def _(df, plot_timeseries):
    # Deep dive into one series
    df_m15 = df.filter(pl.col("unique_id") == "M15")

    fig5 = plot_timeseries(
        df_m15,
        columns="y",
        title="M4 Series M15 - Detailed View",
        x_label="Time",
        y_label="Value",
    )
    fig5
    return df_m15, fig5


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Summary

        The M4 Monthly dataset demonstrates panel data workflows:

        - **Series filtering**: Select specific series from the panel for analysis
        - **Wide format pivoting**: Convert long panel data to wide format for multivariate plots
        - **Aggregated statistics**: Combine information across all panel members
        - **Seasonal patterns**: Identify common trends across heterogeneous series

        Key insights:
        - The M4 dataset contains 50 series with varying scales and patterns
        - Most series show some seasonal structure
        - Pivoting enables direct comparison of multiple series
        - Aggregation reveals patterns common across the panel

        **Note**: Full panel support with `panel_group_name`, `facet_by`, and interactive dropdowns
        is planned for future releases. Current workflows use filtering and pivoting.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
