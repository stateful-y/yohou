"""Walmart Sales - Retail Patterns with Covariates.

Retail sales analysis using the Walmart/Supermarket Sales
dataset with covariates.

Dataset: 3 branches with product lines and segments
Demonstrates: plot_timeseries, plot_calendar_heatmap, plot_seasonality, plot_boxplot
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from yohou.datasets import load_walmart_sales
    from yohou.plotting import (
        plot_boxplot,
        plot_calendar_heatmap,
        plot_seasonality,
        plot_timeseries,
    )
    return (
        load_walmart_sales,
        mo,
        pl,
        plot_boxplot,
        plot_calendar_heatmap,
        plot_seasonality,
        plot_timeseries,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Walmart Sales Dataset

        Transaction-level supermarket sales data across 3 branches (A, B, C) with:
        - Product line categories
        - Customer ratings
        - City information
        - Transaction timestamps

        This example demonstrates how to analyze retail patterns with exogenous variables.
        """
    )
    return


@app.cell
def _(load_walmart_sales):
    # Load dataset
    df = load_walmart_sales()
    df.head(10)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 1. Data Aggregation

        Aggregate transaction-level data to daily totals for time series analysis.
        """
    )
    return


@app.cell
def _(df, pl):
    # Aggregate to daily branch totals
    df_daily = (
        df
        .with_columns(pl.col("time").dt.date().alias("date"))
        .group_by(["date", "Branch"])
        .agg(pl.col("Total").sum().alias("Total"))
        .with_columns(pl.col("date").cast(pl.Datetime).alias("time"))
        .drop("date")
        .sort("time", "Branch")
    )

    df_daily.head(15)
    return (df_daily,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 2. Branch Comparison")
    return


@app.cell
def _(df_daily, plot_timeseries):
    # Compare all 3 branches
    df_branch_wide = df_daily.pivot(
        on="Branch",
        index="time",
        values="Total",
    ).sort("time")

    # Rename columns for clarity
    df_branch_wide = df_branch_wide.rename({
        "A": "Branch A",
        "B": "Branch B",
        "C": "Branch C",
    })

    fig1 = plot_timeseries(
        df_branch_wide,
        columns=["Branch A", "Branch B", "Branch C"],
        title="Walmart Sales - Daily Totals by Branch",
        x_label="Date",
        y_label="Sales ($)",
    )
    fig1
    return df_branch_wide, fig1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 3. Calendar Heatmap

        Visualize daily sales patterns using calendar heatmap.
        """
    )
    return


@app.cell
def _(df_daily, plot_calendar_heatmap):
    # Branch A calendar heatmap for 2019
    df_branch_a = df_daily.filter(pl.col("Branch") == "A")

    fig2 = plot_calendar_heatmap(
        df_branch_a,
        column="Total",
        aggregation="sum",
        year=2019,
        title="Branch A - Daily Sales Calendar (2019)",
    )
    fig2
    return df_branch_a, fig2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 4. Day-of-Week Seasonality

        Analyze weekly patterns across all branches.
        """
    )
    return


@app.cell
def _(df_daily, pl):
    # Aggregate all branches
    df_total = (
        df_daily
        .group_by("time")
        .agg(pl.col("Total").sum())
        .sort("time")
    )
    df_total.head()
    return (df_total,)


@app.cell
def _(df_total, plot_seasonality):
    # Day of week patterns
    fig3 = plot_seasonality(
        df_total,
        feature="dayofweek",
        aggregation="mean",
        title="Walmart Sales - Average by Day of Week",
        x_label="Day of Week (0=Mon, 6=Sun)",
        y_label="Average Sales ($)",
    )
    fig3
    return (fig3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 5. Product Line Analysis

        Compare sales distributions across different product categories.
        """
    )
    return


@app.cell
def _(df, pl):
    # Aggregate by product line and date
    df_product = (
        df
        .with_columns(pl.col("time").dt.date().alias("date"))
        .group_by(["date", "Product line"])
        .agg(pl.col("Total").sum().alias("Total"))
        .with_columns(pl.col("date").cast(pl.Datetime).alias("time"))
        .drop("date")
        .sort("time", "Product line")
    )

    df_product.head(15)
    return (df_product,)


@app.cell
def _(df_product, plot_boxplot):
    # Box plots by product line (using weekly aggregation)
    fig4 = plot_boxplot(
        df_product,
        period="1w",
        title="Sales Distribution by Product Line (Weekly)",
        x_label="Week",
        y_label="Sales ($)",
    )
    fig4
    return (fig4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 6. Branch Performance by Product Line

        Pivot view showing which branches excel in which product lines.
        """
    )
    return


@app.cell
def _(df, pl):
    # Branch × Product line aggregation
    df_branch_product = (
        df
        .group_by(["Branch", "Product line"])
        .agg(pl.col("Total").mean().alias("avg_sales"))
        .sort("Branch", "Product line")
    )

    df_branch_product
    return (df_branch_product,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 7. Rating Analysis

        Examine relationship between customer ratings and sales.
        """
    )
    return


@app.cell
def _(df, pl):
    # Aggregate by rating bins
    df_rating = (
        df
        .with_columns(
            (pl.col("Rating") // 1).alias("rating_bin")
        )
        .group_by("rating_bin")
        .agg([
            pl.col("Total").mean().alias("avg_sales"),
            pl.col("Total").count().alias("count"),
        ])
        .sort("rating_bin")
    )

    df_rating
    return (df_rating,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Summary

        The Walmart Sales dataset demonstrates:

        **Key Insights:**
        - Transaction-level data must be aggregated for time series analysis
        - Calendar heatmaps reveal day-of-week patterns
        - Different branches show varying performance levels
        - Product lines have distinct sales distributions
        - Exogenous variables (ratings, product categories) provide rich context

        **Functions Used:**
        - `plot_timeseries` - Branch comparison over time
        - `plot_calendar_heatmap` - Daily pattern visualization
        - `plot_seasonality` - Day-of-week analysis
        - `plot_boxplot` - Product line distributions

        **Next Steps:**
        - For simpler retail data, see `examples/store_sales.py`
        - For multivariate analysis, see `examples/vic_electricity.py`
        - For panel data, see `examples/m4_monthly.py`
        """
    )
    return


if __name__ == "__main__":
    app.run()
