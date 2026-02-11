"""M4 Quarterly - Quarterly Seasonality Patterns.

Seasonal subseries analysis using M4 Quarterly data.

Dataset: 50 quarterly time series from M4 competition
Demonstrates: plot_timeseries, plot_seasonality, plot_boxplot
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from yohou.datasets import load_m4_quarterly
    from yohou.plotting import (
        plot_boxplot,
        plot_seasonality,
        plot_timeseries,
    )
    return (
        load_m4_quarterly,
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
        # M4 Quarterly Dataset (Panel Data)

        Panel of 50 quarterly time series from the M4 Forecasting Competition.
        This dataset is ideal for demonstrating quarterly seasonal patterns
        and subseries analysis.

        Quarterly data exhibits:
        - Clear 4-quarter cycles (annual patterns)
        - Business cycle effects
        - Seasonal subseries comparisons
        """
    )
    return


@app.cell
def _(load_m4_quarterly):
    # Load panel dataset
    df = load_m4_quarterly()
    df.head(15)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 1. Individual Series Visualization")
    return


@app.cell
def _(df, plot_timeseries):
    # Visualize a single quarterly series
    df_single = df.filter(pl.col("unique_id") == "Q1")

    fig1 = plot_timeseries(
        df_single,
        title="M4 Quarterly - Series Q1",
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

        Compare several quarterly series using multivariate plot via pivoting.
        """
    )
    return


@app.cell
def _(df, plot_timeseries):
    # Compare 4 series with different scales
    series_subset = ["Q1", "Q5", "Q10", "Q20"]
    df_comparison = df.filter(pl.col("unique_id").is_in(series_subset))

    # Pivot to wide format for multivariate plotting
    df_wide = df_comparison.pivot(
        on="unique_id",
        index="time",
        values="y",
    ).sort("time")

    fig2 = plot_timeseries(
        df_wide,
        columns=series_subset,
        title="M4 Quarterly - Comparing 4 Series",
        x_label="Time",
        y_label="Value",
    )
    fig2
    return df_comparison, df_wide, fig2, series_subset


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 3. Quarterly Seasonality Analysis

        Examine seasonal patterns by quarter (Q1, Q2, Q3, Q4).
        """
    )
    return


@app.cell
def _(df_single, plot_seasonality):
    # Quarterly subseries plot
    fig3 = plot_seasonality(
        df_single,
        feature="quarter",
        aggregation="mean",
        title="Q1 - Average Value by Quarter",
        x_label="Quarter",
        y_label="Average Value",
    )
    fig3
    return (fig3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 4. Aggregate Quarterly Patterns

        Aggregate across all 50 series to find common quarterly patterns.
        """
    )
    return


@app.cell
def _(df, plot_seasonality):
    # Add quarter feature
    df_with_quarter = df.with_columns(
        pl.col("time").dt.quarter().alias("quarter")
    )

    # Aggregate across all series
    df_agg = (
        df_with_quarter
        .group_by("quarter")
        .agg(pl.col("y").mean().alias("y"))
        .sort("quarter")
        .with_columns(
            pl.datetime(2000, pl.col("quarter") * 3, 1).alias("time")
        )
    )

    fig4 = plot_seasonality(
        df_agg,
        feature="quarter",
        aggregation="mean",
        title="M4 Quarterly - Average Pattern Across All 50 Series",
        x_label="Quarter",
        y_label="Average Value",
    )
    fig4
    return df_agg, df_with_quarter, fig4


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 5. Quarterly Distribution Analysis

        Box plots showing value distributions by quarter.
        """
    )
    return


@app.cell
def _(df_single, plot_boxplot):
    # Quarterly boxplots for single series
    fig5 = plot_boxplot(
        df_single,
        period="1q",
        title="Q1 - Quarterly Distributions",
        x_label="Quarter",
        y_label="Value",
    )
    fig5
    return (fig5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 6. Year-by-Year Comparison

        Visualize how patterns evolve across years.
        """
    )
    return


@app.cell
def _(df_single, pl):
    # Add year and quarter features
    df_yearly = df_single.with_columns([
        pl.col("time").dt.year().alias("year"),
        pl.col("time").dt.quarter().alias("quarter"),
    ])

    # Pivot to compare years (aggregate if duplicates exist)
    df_yearly_pivot = (
        df_yearly
        .group_by(["quarter", "year"])
        .agg(pl.col("y").mean())
        .pivot(
            on="year",
            index="quarter",
            values="y",
        )
        .sort("quarter")
    )

    df_yearly_pivot.head(10)
    return df_yearly, df_yearly_pivot


@app.cell
def _(df_yearly_pivot, plot_timeseries):
    # Plot multiple years as separate series
    years = [col for col in df_yearly_pivot.columns if col != "quarter"][:5]  # First 5 years

    # Create time axis aligned by quarter
    df_yearly_time = df_yearly_pivot.with_columns(
        pl.datetime(2000, pl.col("quarter") * 3, 1).alias("time")
    )

    fig6 = plot_timeseries(
        df_yearly_time,
        columns=years,
        title="Q1 - Year-over-Year Comparison",
        x_label="Quarter",
        y_label="Value",
    )
    fig6
    return df_yearly_time, fig6, years


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Summary

        The M4 Quarterly dataset demonstrates:

        **Key Insights:**
        - Quarterly data shows clear 4-period seasonal cycles
        - Aggregation across series reveals common business cycle patterns
        - Box plots help identify outliers and distributional changes
        - Year-over-year comparison shows trend evolution

        **Functions Used:**
        - `plot_timeseries` - Single series and multivariate comparisons
        - `plot_seasonality` - Quarterly subseries analysis
        - `plot_boxplot` - Quarterly distribution visualization

        **Next Steps:**
        - For higher-frequency data, see `examples/m4_hourly.py`
        - For monthly patterns, see `examples/m4_monthly.py`
        - For hierarchical data, see `examples/australian_tourism.py`
        """
    )
    return


if __name__ == "__main__":
    app.run()
