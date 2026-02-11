"""ETTm1 - Electricity Transformer Multivariate Analysis.

Multivariate time series analysis using the ETTm1 benchmark.

Dataset: 7 temperature features at 15-minute intervals
Demonstrates: plot_timeseries, plot_cross_correlation, plot_seasonality, plot_boxplot
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from yohou.datasets import load_ett_m1
    from yohou.plotting import (
        plot_boxplot,
        plot_cross_correlation,
        plot_seasonality,
        plot_timeseries,
    )
    return (
        load_ett_m1,
        mo,
        pl,
        plot_boxplot,
        plot_cross_correlation,
        plot_seasonality,
        plot_timeseries,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # ETTm1 Dataset (Electricity Transformer Temperature)

        High-frequency multivariate time series with:
        - **Target**: OT (Oil Temperature)
        - **Features**: 6 temperature measurements
          - HUFL, HULL: High UseFul Load (temperature)
          - MUFL, MULL: Medium UseFul Load
          - LUFL, LULL: Low UseFul Load
        - **Frequency**: 15-minute intervals
        - **Period**: 2 years (July 2016 - June 2018)

        This dataset is a benchmark for transformer forecasting and
        demonstrates multivariate dependencies in industrial IoT data.
        """
    )
    return


@app.cell
def _(load_ett_m1):
    # Load dataset
    df = load_ett_m1()
    df.head(20)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 1. Overview - First 7 Days

        Visualize target and all features for one week.
        """
    )
    return


@app.cell
def _(df, plot_timeseries):
    # First 7 days (15-min intervals: 96 per day × 7 = 672)
    df_week = df.head(96 * 7)

    # Plot target + key features
    fig1 = plot_timeseries(
        df_week,
        columns=["OT", "HUFL", "MUFL", "LUFL"],
        title="ETTm1 - Oil Temperature and Load Features (First 7 Days)",
        x_label="Time",
        y_label="Temperature",
    )
    fig1
    return df_week, fig1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 2. Target vs High Load Feature

        Compare oil temperature (OT) with high load feature (HUFL).
        """
    )
    return


@app.cell
def _(df, plot_timeseries):
    # First 30 days for clearer patterns
    df_month = df.head(96 * 30)

    fig2 = plot_timeseries(
        df_month,
        columns=["OT", "HUFL"],
        title="ETTm1 - Oil Temperature vs High Load (First 30 Days)",
        x_label="Time",
        y_label="Temperature",
    )
    fig2
    return df_month, fig2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 3. Cross-Correlation Analysis

        Examine lead-lag relationships between features and target.
        """
    )
    return


@app.cell
def _(df_month, plot_cross_correlation):
    # Cross-correlation between target (y) and HUFL
    fig3 = plot_cross_correlation(
        df_month,
        x_column="OT",
        y_column="HUFL",
        lags=96,  # 1 day of 15-min intervals
        title="ETTm1 - Cross-Correlation: Oil Temp (OT) vs HUFL",
    )
    fig3
    return (fig3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 4. Another Cross-Correlation - MUFL

        Check medium load feature correlation.
        """
    )
    return


@app.cell
def _(df_month, plot_cross_correlation):
    # Cross-correlation with MUFL
    fig4 = plot_cross_correlation(
        df_month,
        x_column="OT",
        y_column="MUFL",
        lags=96,
        title="ETTm1 - Cross-Correlation: Oil Temp (OT) vs MUFL",
    )
    fig4
    return (fig4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 5. Hourly Seasonality Patterns

        Analyze intraday patterns in oil temperature.
        """
    )
    return


@app.cell
def _(df, plot_seasonality):
    # Hour-of-day patterns
    fig5 = plot_seasonality(
        df,
        columns="OT",
        feature="hour",
        aggregation="mean",
        title="ETTm1 - Average Oil Temperature by Hour of Day",
        x_label="Hour (0-23)",
        y_label="Average Temperature",
    )
    fig5
    return (fig5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 6. Feature Seasonality - HUFL

        Compare hourly patterns across features.
        """
    )
    return


@app.cell
def _(df, plot_seasonality):
    # HUFL hourly patterns
    fig6 = plot_seasonality(
        df,
        columns="HUFL",
        feature="hour",
        aggregation="mean",
        title="ETTm1 - Average HUFL by Hour of Day",
        x_label="Hour (0-23)",
        y_label="Average Temperature",
    )
    fig6
    return (fig6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 7. Daily Box Plots

        Examine daily distributions of oil temperature.
        """
    )
    return


@app.cell
def _(df, plot_boxplot):
    # Daily aggregation (first 60 days)
    df_60d = df.head(96 * 60)

    fig7 = plot_boxplot(
        df_60d,
        columns="OT",
        period="1d",
        title="ETTm1 - Daily Oil Temperature Distribution (First 60 Days)",
        x_label="Day",
        y_label="Temperature",
    )
    fig7
    return df_60d, fig7


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 8. Multi-Feature Comparison - All Loads

        Compare all 6 load features simultaneously.
        """
    )
    return


@app.cell
def _(df_week, plot_timeseries):
    # All 6 features for first week
    features = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]

    fig8 = plot_timeseries(
        df_week,
        columns=features,
        title="ETTm1 - All Load Features (First 7 Days)",
        x_label="Time",
        y_label="Temperature",
    )
    fig8
    return features, fig8


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 9. Day-of-Week Patterns

        Check for weekly cycles in transformer behavior.
        """
    )
    return


@app.cell
def _(df, plot_seasonality):
    # Day of week analysis
    fig9 = plot_seasonality(
        df,
        columns="OT",
        feature="dayofweek",
        aggregation="mean",
        title="ETTm1 - Average Oil Temperature by Day of Week",
        x_label="Day of Week (0=Mon, 6=Sun)",
        y_label="Average Temperature",
    )
    fig9
    return (fig9,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Summary

        The ETTm1 dataset demonstrates:

        **Key Insights:**
        - Oil temperature (OT) shows strong correlation with all load features
        - HUFL (High UseFul Load) has the strongest correlation with target
        - Clear intraday patterns with 24-hour cycles
        - Weekday vs weekend effects present but subtle
        - 15-minute resolution captures rapid temperature changes
        - Cross-correlation reveals near-instantaneous dependencies

        **Functions Used:**
        - `plot_timeseries` - Multivariate visualization
        - `plot_cross_correlation` - Feature dependency analysis
        - `plot_seasonality` - Hourly and weekly pattern detection
        - `plot_boxplot` - Daily distribution analysis

        **Industrial IoT Insights:**
        - Transformer temperature is highly dependent on load
        - High load features lead temperature changes
        - Time-of-day effects dominate (business hours vs off-hours)
        - Useful for predictive maintenance and anomaly detection

        **Next Steps:**
        - For simpler multivariate data, see `examples/vic_electricity.py`
        - For high-frequency panel data, see `examples/m4_hourly.py`
        - For seasonal decomposition, see `examples/air_passengers.py`
        """
    )
    return


if __name__ == "__main__":
    app.run()
