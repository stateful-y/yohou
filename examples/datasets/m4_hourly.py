"""M4 Hourly - High-Frequency Intraday Patterns.

High-frequency time series analysis using M4 Hourly data.

Dataset: 50 hourly time series from M4 competition
Demonstrates: plot_timeseries, plot_seasonality, plot_boxplot
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from yohou.datasets import load_m4_hourly
    from yohou.plotting import (
        plot_boxplot,
        plot_seasonality,
        plot_timeseries,
    )
    return (
        load_m4_hourly,
        mo,
        pl,
        plot_boxplot,
        plot_seasonality,
        plot_timeseries,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # M4 Hourly Dataset (High-Frequency Panel Data)

        Panel of 50 hourly time series from the M4 Forecasting Competition.
        High-frequency data exhibits:
        - Intraday patterns (hourly cycles)
        - Day-of-week effects
        - Peak/off-peak periods

        This example demonstrates how to analyze hourly patterns and
        identify recurring cycles at multiple timescales.
        """
    )
    return


@app.cell
def _(load_m4_hourly):
    # Load panel dataset
    df = load_m4_hourly()
    df.head(24)  # First 24 hours
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 1. Single Series - 7 Days")
    return


@app.cell
def _(df, plot_timeseries):
    # Visualize one week of hourly data for H1
    df_h1 = df.filter(pl.col("unique_id") == "H1")

    # First 7 days (168 hours)
    df_h1_week = df_h1.head(24 * 7)

    fig1 = plot_timeseries(
        df_h1_week,
        title="M4 Hourly - Series H1 (First 7 Days)",
        x_label="Time",
        y_label="Value",
    )
    fig1
    return df_h1, df_h1_week, fig1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 2. Intraday Patterns - Hour-of-Day Analysis

        Aggregate by hour of day to identify intraday cycles.
        """
    )
    return


@app.cell
def _(df_h1, plot_seasonality):
    # Hour-of-day patterns
    fig2 = plot_seasonality(
        df_h1,
        feature="hour",
        aggregation="mean",
        title="H1 - Average Value by Hour of Day",
        x_label="Hour (0-23)",
        y_label="Average Value",
    )
    fig2
    return (fig2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 3. Day-of-Week Patterns

        Analyze how patterns differ across weekdays.
        """
    )
    return


@app.cell
def _(df_h1, plot_seasonality):
    # Day of week patterns
    fig3 = plot_seasonality(
        df_h1,
        feature="dayofweek",
        aggregation="mean",
        title="H1 - Average Value by Day of Week",
        x_label="Day of Week (0=Mon, 6=Sun)",
        y_label="Average Value",
    )
    fig3
    return (fig3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 4. Multiple Series Comparison

        Compare 3 different hourly series (first 7 days each).
        """
    )
    return


@app.cell
def _(df, plot_timeseries):
    # Compare H1, H5, H10 for first week
    series_subset = ["H1", "H5", "H10"]
    df_comparison = (
        df
        .filter(pl.col("unique_id").is_in(series_subset))
        .with_columns(
            pl.col("time").rank("dense").over("unique_id").alias("hour_rank")
        )
        .filter(pl.col("hour_rank") <= 168)  # First 7 days
        .drop("hour_rank")
    )

    # Pivot to wide format
    df_wide = df_comparison.pivot(
        on="unique_id",
        index="time",
        values="y",
    ).sort("time")

    fig4 = plot_timeseries(
        df_wide,
        columns=series_subset,
        title="M4 Hourly - Comparing 3 Series (First 7 Days)",
        x_label="Time",
        y_label="Value",
    )
    fig4
    return df_comparison, df_wide, fig4, series_subset


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 5. Hourly Distribution Analysis

        Box plots showing how values distribute across hours.
        """
    )
    return


@app.cell
def _(df_h1, plot_boxplot):
    # Box plots by 6-hour periods
    fig5 = plot_boxplot(
        df_h1,
        period="6h",
        title="H1 - Distribution by 6-Hour Periods",
        x_label="6-Hour Period",
        y_label="Value",
    )
    fig5
    return (fig5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 6. Aggregate Intraday Pattern

        Average intraday pattern across all 50 series.
        """
    )
    return


@app.cell
def _(df, pl):
    # Aggregate across all series by hour-of-day
    df_agg_hour = (
        df
        .with_columns(pl.col("time").dt.hour().alias("hour"))
        .group_by("hour")
        .agg(pl.col("y").mean().alias("y"))
        .sort("hour")
        .with_columns(
            pl.datetime(2000, 1, 1).alias("base_date")
        )
        .with_columns(
            (pl.col("base_date") + pl.duration(hours=pl.col("hour"))).alias("time")
        )
        .drop("base_date")
        .select(["time", "y"])
    )

    df_agg_hour
    return (df_agg_hour,)


@app.cell
def _(df_agg_hour, plot_seasonality):
    # Plot aggregate hourly pattern
    fig6 = plot_seasonality(
        df_agg_hour,
        feature="hour",
        aggregation="mean",
        title="M4 Hourly - Average Intraday Pattern (All 50 Series)",
        x_label="Hour of Day",
        y_label="Average Value",
    )
    fig6
    return (fig6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 7. Daily Box Plots

        View how entire days distribute across the dataset.
        """
    )
    return


@app.cell
def _(df_h1, plot_boxplot):
    # Box plots by day
    fig7 = plot_boxplot(
        df_h1,
        period="1d",
        title="H1 - Daily Distributions",
        x_label="Day",
        y_label="Value",
    )
    fig7
    return (fig7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Summary

        The M4 Hourly dataset demonstrates:

        **Key Insights:**
        - High-frequency data reveals strong intraday patterns
        - Hour-of-day analysis identifies peak and off-peak periods
        - Day-of-week effects show weekday vs weekend differences
        - 6-hour aggregations capture major shifts in behavior
        - Aggregating across all series reveals common patterns

        **Functions Used:**
        - `plot_timeseries` - High-frequency line plots
        - `plot_seasonality` - Hour-of-day and day-of-week analysis
        - `plot_boxplot` - Distribution analysis at multiple timescales

        **Next Steps:**
        - For monthly data, see `examples/m4_monthly.py`
        - For quarterly patterns, see `examples/m4_quarterly.py`
        - For multivariate high-frequency data, see `examples/vic_electricity.py`
        """
    )
    return


if __name__ == "__main__":
    app.run()
