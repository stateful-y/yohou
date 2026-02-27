# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Electricity Demand Dataset",
    "description": "High-frequency panel analysis of half-hourly Australian electricity demand across five states with cross-correlation diagnostics and rolling statistics.",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from yohou.datasets import fetch_electricity_demand
    from yohou.plotting import (
        plot_cross_correlation,
        plot_rolling_statistics,
        plot_seasonality,
        plot_time_series,
    )
    from yohou.utils.panel import inspect_panel

    return (
        fetch_electricity_demand,
        inspect_panel,
        mo,
        plot_cross_correlation,
        plot_rolling_statistics,
        plot_seasonality,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Electricity Demand Dataset

    ## What You'll Learn

    This example demonstrates high-frequency panel time series analysis with the
    Australian Electricity Demand dataset. You'll learn how to:

    - Inspect the panel structure of 5 Australian state demands
    - Visualize and compare half-hourly demand across states
    - Use cross-correlation to find relationships between states
    - Apply intraday seasonality analysis (hour of day)

    ## Prerequisites

    None; this is a standalone dataset exploration.
    """)


@app.cell
def _(fetch_electricity_demand, plot_time_series):
    df = fetch_electricity_demand().frame.head(10000)
    plot_time_series(df, title="Victorian Electricity Demand")
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Inspect Panel Structure

    The dataset has 5 panel groups, one per Australian state, using the
    `state__demand` convention.
    """)


@app.cell
def _(df, inspect_panel, mo):
    global_cols, panel_groups = inspect_panel(df)
    mo.md(f"""
    **Global columns**: {global_cols}

    **Panel groups** ({len(panel_groups)} groups):

    {chr(10).join(f"- **{k}**: {v}" for k, v in panel_groups.items())}
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. All States Demand Comparison

    Plotting demand from all 5 states reveals scale differences: NSW and
    Victoria have the highest demand, while Tasmania is the smallest.
    """)


@app.cell
def _(df, plot_time_series):
    _demand_cols = [c for c in df.columns if c.endswith("__demand")]
    # Show first 48*7 rows (one week of 30-min data) for readability
    plot_time_series(
        df.head(48 * 7),
        columns=_demand_cols,
        title="Electricity Demand - All States (1 Week)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Rolling Average - NSW Demand

    A 24-hour (48-step) rolling average of NSW demand highlights daily and
    weekly patterns beneath the noisy half-hourly readings.
    """)


@app.cell
def _(df, plot_rolling_statistics):
    plot_rolling_statistics(
        df,
        columns="nsw__demand",
        window_size=48,  # 24 hours at 30-min intervals
        statistics="mean",
        show_original=True,
        title="NSW - 24-Hour Rolling Average",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Cross-Correlation Between States

    Cross-correlation quantifies how demand in one state relates to demand
    in another, revealing synchronised patterns.
    """)


@app.cell
def _(df, plot_cross_correlation):
    plot_cross_correlation(
        df,
        columns=["nsw__demand", "vic__demand"],
        max_lags=100,
        title="NSW vs Victoria Demand Cross-Correlation",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Hourly Patterns

    Aggregating by hour of day reveals intraday demand peaks and the
    consumption pattern across all states.
    """)


@app.cell
def _(df, plot_seasonality):
    plot_seasonality(
        df,
        columns="nsw__demand",
        seasonality="hour",
        title="NSW - Average Demand by Hour",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **Panel format**: 5 state-level demand series using `state__demand` convention
    - **High frequency**: Half-hourly data captures intraday patterns
    - **Cross-state correlation**: States show strongly correlated demand patterns
    - **Intraday seasonality**: Clear hour-of-day effects (peak demand times)
    - **Scale differences**: NSW and Victoria dominate total demand

    ## Next Steps

    - For monthly panel data, see [`examples/datasets/hospital.py`](/examples/datasets/hospital/)
    - For weekly panel data, see [`examples/datasets/store_sales.py`](/examples/datasets/store_sales/)
    - For advanced panel analytics, see [`examples/datasets/australian_tourism_forecasting.py`](/examples/datasets/australian_tourism_forecasting/)
    """)


if __name__ == "__main__":
    app.run()
