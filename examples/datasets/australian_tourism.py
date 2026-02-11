"""Australian Tourism - Hierarchical Panel Data.

Hierarchical time series visualization using the Australian
Tourism dataset with regional groupings.

Dataset: Quarterly tourism demand by Region, State, Purpose
Demonstrates: plot_timeseries, plot_seasonality, plot_boxplot
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from yohou.datasets import load_australian_tourism
    from yohou.plotting import (
        plot_boxplot,
        plot_seasonality,
        plot_timeseries,
    )

    return (
        load_australian_tourism,
        mo,
        pl,
        plot_boxplot,
        plot_seasonality,
        plot_timeseries,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Australian Tourism Dataset

    Hierarchical quarterly tourism demand data with:
    - **Regions**: 76 tourism regions across Australia
    - **States**: 8 Australian states/territories
    - **Purposes**: 4 travel purposes (Business, Holiday, Visiting, Other)
    - **Frequency**: Quarterly observations

    This example demonstrates hierarchical data analysis and regional comparisons.
    """)
    return


@app.cell
def _(load_australian_tourism):
    # Load dataset
    df = load_australian_tourism()
    df.head(12)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. State-Level Aggregation
    """)
    return


@app.cell
def _(df, pl):
    # Aggregate to state level
    df_state = (
        df
        .group_by(["State", "time"])
        .agg(pl.col("Trips").sum())
        .sort("State", "time")
    )

    df_state.head(15)
    return (df_state,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Major State Comparison
    """)
    return


@app.cell
def _(df_state, pl, plot_timeseries):
    # Compare major states
    major_states = ["New South Wales", "Victoria", "Queensland"]
    df_major = df_state.filter(pl.col("State").is_in(major_states))

    # Pivot to wide format
    df_major_wide = df_major.pivot(
        on="State",
        index="time",
        values="Trips",
    ).sort("time")

    fig1 = plot_timeseries(
        df_major_wide,
        columns=major_states,
        title="Australian Tourism - Major States (Quarterly)",
        x_label="Time",
        y_label="Tourism Demand",
    )
    fig1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Regional Focus - Sydney vs Melbourne

    Compare two major tourism regions.
    """)
    return


@app.cell
def _(df, pl, plot_timeseries):
    # Focus on Sydney and Melbourne
    df_cities = df.filter(
        pl.col("Region").is_in(["Sydney", "Melbourne"])
    ).group_by(["Region", "time"]).agg(pl.col("Trips").sum()).sort("time")

    # Pivot for comparison
    df_cities_wide = df_cities.pivot(
        on="Region",
        index="time",
        values="Trips",
    ).sort("time")

    fig2 = plot_timeseries(
        df_cities_wide,
        columns=["Sydney", "Melbourne"],
        title="Tourism Demand - Sydney vs Melbourne",
        x_label="Time",
        y_label="Tourism Demand",
    )
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Purpose Analysis - Holiday vs Business Travel

    Examine different travel purposes.
    """)
    return


@app.cell
def _(df, pl):
    # Aggregate by purpose
    df_purpose = (
        df
        .group_by(["Purpose", "time"])
        .agg(pl.col("Trips").sum())
        .sort("Purpose", "time")
    )

    df_purpose.head(20)
    return (df_purpose,)


@app.cell
def _(df_purpose, plot_timeseries):
    # Pivot purposes for comparison
    df_purpose_wide = df_purpose.pivot(
        on="Purpose",
        index="time",
        values="Trips",
    ).sort("time")

    fig3 = plot_timeseries(
        df_purpose_wide,
        columns=["Holiday", "Business", "Visiting", "Other"],
        title="Australian Tourism by Purpose",
        x_label="Time",
        y_label="Tourism Demand",
    )
    fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Quarterly Seasonality Patterns

    Analyze seasonal effects across all regions.
    """)
    return


@app.cell
def _(df, pl):
    # Total tourism demand
    df_total = (
        df
        .group_by("time")
        .agg(pl.col("Trips").sum())
        .sort("time")
    )
    df_total.head()
    return (df_total,)


@app.cell
def _(df_total, plot_seasonality):
    # Quarterly patterns
    fig4 = plot_seasonality(
        df_total,
        feature="quarter",
        aggregation="mean",
        title="Australian Tourism - Average by Quarter",
        x_label="Quarter",
        y_label="Average Tourism Demand",
    )
    fig4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Holiday Travel Seasonality

    Focus on holiday-specific patterns.
    """)
    return


@app.cell
def _(df, pl):
    # Holiday travel only
    df_holiday = (
        df
        .filter(pl.col("Purpose") == "Holiday")
        .group_by("time")
        .agg(pl.col("Trips").sum())
        .sort("time")
    )
    df_holiday.head()
    return (df_holiday,)


@app.cell
def _(df_holiday, plot_seasonality):
    # Holiday quarterly patterns
    fig5 = plot_seasonality(
        df_holiday,
        feature="quarter",
        aggregation="mean",
        title="Holiday Travel - Average by Quarter",
        x_label="Quarter (1=Jan-Mar, 4=Oct-Dec)",
        y_label="Average Demand",
    )
    fig5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Regional Distribution Analysis

    Box plots showing how regions compare.
    """)
    return


@app.cell
def _(df_state, plot_boxplot):
    # Box plots by state (annual aggregation)
    fig6 = plot_boxplot(
        df_state,
        period="1y",
        title="State-Level Tourism Demand Distribution (Annual)",
        x_label="Year",
        y_label="Tourism Demand",
    )
    fig6
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. Sydney Region by Purpose

    Drill down into a single region across purposes.
    """)
    return


@app.cell
def _(df, pl, plot_timeseries):
    # Sydney by purpose
    df_sydney_purpose = (
        df
        .filter(pl.col("Region") == "Sydney")
        .group_by(["Purpose", "time"])
        .agg(pl.col("Trips").sum())
        .sort("time")
    )

    # Pivot
    df_sydney_wide = df_sydney_purpose.pivot(
        on="Purpose",
        index="time",
        values="Trips",
    ).sort("time")

    fig7 = plot_timeseries(
        df_sydney_wide,
        columns=["Holiday", "Business", "Visiting", "Other"],
        title="Sydney Tourism by Purpose",
        x_label="Time",
        y_label="Tourism Demand",
    )
    fig7
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    The Australian Tourism dataset demonstrates:

    **Key Insights:**
    - Hierarchical structure enables multi-level aggregation
    - State-level patterns show clear regional differences
    - Holiday travel dominates overall demand with strong seasonality
    - Q1 (Jan-Mar) and Q4 (Oct-Dec) are peak tourism quarters in Australia
    - Major cities (Sydney, Melbourne) drive significant share of demand
    - Different purposes exhibit distinct seasonal patterns

    **Functions Used:**
    - `plot_timeseries` - Multi-level comparisons (state, region, purpose)
    - `plot_seasonality` - Quarterly pattern analysis
    - `plot_boxplot` - Distributional comparisons across regions

    **Hierarchical Analysis:**
    - **Top level**: Total national tourism
    - **State level**: 8 states/territories
    - **Region level**: 76 tourism regions
    - **Purpose level**: 4 travel purposes

    **Next Steps:**
    - For simple panel data, see `examples/m4_monthly.py`
    - For retail hierarchies, see `examples/store_sales.py`
    - For quarterly seasonality, see `examples/m4_quarterly.py`
    """)
    return


if __name__ == "__main__":
    app.run()
