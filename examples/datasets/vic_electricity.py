"""Victoria Electricity - Multivariate Time Series.

Multivariate time series analysis using electricity demand
data from Victoria, Australia with temperature.

Dataset: 30-min electricity demand, temperature, holidays
Demonstrates: plot_timeseries, plot_rolling_statistics, plot_boxplot, plot_seasonality, plot_cross_correlation
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    from yohou.datasets import load_vic_electricity
    from yohou.plotting import (
        plot_boxplot,
        plot_cross_correlation,
        plot_rolling_statistics,
        plot_seasonality,
        plot_timeseries,
    )
    return (
        load_vic_electricity,
        pl,
        plot_boxplot,
        plot_cross_correlation,
        plot_rolling_statistics,
        plot_seasonality,
        plot_timeseries,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Victoria Electricity Demand Dataset

        30-minute electricity demand for Victoria, Australia, with temperature and
        holiday indicators. This multivariate dataset is ideal for demonstrating
        cross-correlation analysis and understanding relationships between demand
        and external factors.
        """
    )
    return


@app.cell
def _(load_vic_electricity):
    # Load dataset
    df = load_vic_electricity()
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 1. Multivariate Time Series Visualization")
    return


@app.cell
def _(df, plot_timeseries):
    # Plot both demand and temperature
    fig1 = plot_timeseries(
        df,
        columns=["Demand", "Temperature"],
        title="Electricity Demand and Temperature",
        x_label="Time",
        y_label="Value",
    )
    fig1
    return (fig1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 2. Rolling Statistics for Each Series")
    return


@app.cell
def _(df, plot_rolling_statistics):
    # 48-period (24 hours) rolling mean for demand
    fig2 = plot_rolling_statistics(
        df,
        columns="Demand",
        window_size=48,
        statistics="mean",
        show_original=True,
        title="Electricity Demand - 24-Hour Rolling Average",
    )
    fig2
    return (fig2,)


@app.cell
def _(df, plot_rolling_statistics):
    # Temperature rolling statistics with bands
    fig3 = plot_rolling_statistics(
        df,
        columns="Temperature",
        window_size=48,
        statistics=["mean", "min", "max"],
        fill_between=True,
        show_original=False,
        title="Temperature - 24-Hour Min/Mean/Max",
    )
    fig3
    return (fig3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 3. Time-of-Day Patterns")
    return


@app.cell
def _(df, plot_boxplot):
    # Daily patterns - aggregate to hourly
    fig4 = plot_boxplot(
        df,
        columns="Demand",
        period="1d",
        title="Daily Electricity Demand Distribution",
    )
    fig4
    return (fig4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 4. Seasonal Patterns (Hour of Day)")
    return


@app.cell
def _(df, plot_seasonality):
    # Average demand by hour of day
    fig5 = plot_seasonality(
        df,
        columns="Demand",
        feature="hour",
        aggregation="mean",
        title="Average Electricity Demand by Hour",
    )
    fig5
    return (fig5,)


@app.cell
def _(df, plot_seasonality):
    # Temperature patterns by hour
    fig6 = plot_seasonality(
        df,
        columns="Temperature",
        feature="hour",
        aggregation="mean",
        title="Average Temperature by Hour",
    )
    fig6
    return (fig6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 5. Cross-Correlation Analysis

        Analyze the relationship between electricity demand and temperature.
        Negative lags indicate temperature *leads* demand (temperature changes,
        then demand responds). Positive lags indicate demand *leads* temperature
        (less meaningful in this context).
        """
    )
    return


@app.cell
def _(df, plot_cross_correlation):
    # CCF between demand and temperature
    fig7 = plot_cross_correlation(
        df,
        x_column="Temperature",
        y_column="Demand",
        lags=100,  # ~50 hours
        title="Cross-Correlation: Temperature vs Electricity Demand",
    )
    fig7
    return (fig7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 6. Day of Week Patterns

        Explore weekly patterns in both demand and temperature.
        """
    )
    return


@app.cell
def _(df, plot_seasonality):
    # Demand by day of week
    fig8 = plot_seasonality(
        df,
        columns=["Demand", "Temperature"],
        feature="dayofweek",
        aggregation="mean",
        title="Weekly Patterns: Demand and Temperature",
    )
    fig8
    return (fig8,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Summary

        The Victoria Electricity dataset demonstrates:

        - **Multivariate visualization**: Plotting demand and temperature together
        - **Rolling statistics**: 24-hour windows reveal daily patterns
        - **Time-of-day effects**: Clear diurnal patterns in both series
        - **Cross-correlation**: Temperature leads demand by several hours (cooling/heating response)
        - **Weekly patterns**: Weekday vs weekend differences

        This type of analysis is crucial for:
        - Energy demand forecasting
        - Understanding driver-response relationships
        - Identifying lagged effects in multivariate systems
        - Planning grid operations based on weather
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
