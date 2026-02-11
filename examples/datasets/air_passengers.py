"""Air Passengers - Comprehensive Plotting Examples.

Yohou's time series plotting with the classic Box & Jenkins
airline passengers dataset.

Dataset: Monthly airline passengers, 1949-1960
Demonstrates: plot_timeseries, plot_rolling_statistics, plot_boxplot, plot_seasonality, plot_autocorrelation, plot_prediction_interval, plot_periodogram, plot_lag_scatter, plot_residuals, plot_forecast, plot_comparison
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import polars as pl
    from yohou.datasets import load_air_passengers
    from yohou.plotting import (
        plot_autocorrelation,
        plot_boxplot,
        plot_comparison,
        plot_correlation_diagnostics,
        plot_forecast,
        plot_lag_scatter,
        plot_partial_autocorrelation,
        plot_periodogram,
        plot_prediction_interval,
        plot_residuals,
        plot_rolling_statistics,
        plot_seasonality,
        plot_timeseries,
    )

    return (
        load_air_passengers,
        pl,
        plot_autocorrelation,
        plot_boxplot,
        plot_comparison,
        plot_correlation_diagnostics,
        plot_forecast,
        plot_lag_scatter,
        plot_partial_autocorrelation,
        plot_periodogram,
        plot_prediction_interval,
        plot_residuals,
        plot_rolling_statistics,
        plot_seasonality,
        plot_timeseries,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Air Passengers Dataset

    Monthly international airline passengers (1949-1960). This classic dataset
    exhibits strong trend and seasonal patterns, making it ideal for demonstrating
    time series visualization techniques.
    """)
    return


@app.cell
def _(load_air_passengers):
    # Load dataset
    df = load_air_passengers()
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Basic Time Series Plot
    """)
    return


@app.cell
def _(df, plot_timeseries):
    # Simple line plot
    fig1 = plot_timeseries(
        df,
        title="Air Passengers (1949-1960)",
        x_label="Time",
        y_label="Passengers (thousands)",
    )
    fig1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Rolling Statistics
    """)
    return


@app.cell
def _(df, plot_rolling_statistics):
    # 12-month rolling mean with original data
    fig2 = plot_rolling_statistics(
        df,
        window_size=12,
        statistics="mean",
        show_original=True,
        title="12-Month Rolling Average",
    )
    fig2
    return


@app.cell
def _(df, plot_rolling_statistics):
    # Min/max envelope
    fig3 = plot_rolling_statistics(
        df,
        window_size=12,
        statistics=["min", "max"],
        fill_between=True,
        show_original=False,
        title="12-Month Min/Max Envelope",
    )
    fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Seasonal Patterns
    """)
    return


@app.cell
def _(df, plot_boxplot):
    # Monthly boxplots
    fig4 = plot_boxplot(
        df,
        period="1mo",
        title="Monthly Distribution Patterns",
    )
    fig4
    return


@app.cell
def _(df, plot_seasonality):
    # Average by month
    fig5 = plot_seasonality(
        df,
        feature="month",
        aggregation="mean",
        title="Average Passengers by Month",
    )
    fig5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Autocorrelation Analysis
    """)
    return


@app.cell
def _(df, plot_autocorrelation):
    # ACF plot
    fig6 = plot_autocorrelation(
        df,
        lags=40,
        title="Autocorrelation Function",
    )
    fig6
    return


@app.cell
def _(df, plot_partial_autocorrelation):
    # PACF plot
    fig7 = plot_partial_autocorrelation(
        df,
        lags=40,
        title="Partial Autocorrelation Function",
    )
    fig7
    return


@app.cell
def _(df, plot_correlation_diagnostics):
    # Combined ACF + PACF
    fig8 = plot_correlation_diagnostics(
        df,
        lags=36,
        title="Correlation Diagnostics",
    )
    fig8
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Frequency Domain Analysis
    """)
    return


@app.cell
def _(df, plot_periodogram):
    # Periodogram
    fig9 = plot_periodogram(
        df,
        detrend="linear",
        log_scale=True,
        show_peaks=True,
        n_peaks=5,
        title="Periodogram (Log Scale)",
    )
    fig9
    return


@app.cell
def _(df, plot_lag_scatter):
    # Lag scatter plots
    fig10 = plot_lag_scatter(
        df,
        lags=[1, 6, 12, 24],
        show_diagonal=True,
        title="Lag Scatter Plots",
    )
    fig10
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Forecasting Demonstration

    For demonstration purposes, we'll create simple forecasts using rolling statistics.
    """)
    return


@app.cell
def _(df, pl):
    # Create simple forecasts (last 12 months as "forecast")
    # Get the last 12 rows (1 year of monthly data)
    n_forecast = 12

    df_with_forecast = df.with_columns(
        [
            # Mark last 12 rows as forecast
            pl.lit(False).alias("is_forecast"),
            # Compute fitted values (12-month rolling mean)
            pl.col("Passengers").rolling_mean(12, center=False).alias("fitted"),
        ]
    ).with_row_index("idx").with_columns(
        [
            # Mark last n_forecast rows as forecast
            (pl.col("idx") >= pl.col("idx").max() - n_forecast + 1).alias("is_forecast"),
        ]
    ).drop("idx").with_columns(
        [
            # Residuals
            (pl.col("Passengers") - pl.col("fitted")).alias("residuals"),
            # Prediction intervals (±20%)
            (pl.col("fitted") * 0.8).alias("lower_bound"),
            (pl.col("fitted") * 1.2).alias("upper_bound"),
        ]
    )

    df_with_forecast.tail(15)
    return (df_with_forecast,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Prediction Intervals
    """)
    return


@app.cell
def _(df_with_forecast, pl, plot_prediction_interval):
    # Plot with prediction intervals
    fig11 = plot_prediction_interval(
        df_with_forecast.filter(pl.col("fitted").is_not_null()),
        columns="fitted",
        lower_bound_column="lower_bound",
        upper_bound_column="upper_bound",
        title="Fitted Values with 80% Prediction Interval",
    )
    fig11
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Residual Diagnostics
    """)
    return


@app.cell
def _(df_with_forecast, pl, plot_residuals):
    # Residual diagnostics (4-panel)
    fig12 = plot_residuals(
        df_with_forecast.filter(pl.col("fitted").is_not_null()),
        residuals_column="residuals",
        fitted_column="fitted",
        title="Residual Diagnostics",
    )
    fig12
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Forecast Visualization
    """)
    return


@app.cell
def _(df_with_forecast, plot_forecast):
    # Historical + forecast plot
    fig13 = plot_forecast(
        df_with_forecast,
        columns="Passengers",
        is_forecast_column="is_forecast",
        lower_bound_column="lower_bound",
        upper_bound_column="upper_bound",
        n_history=36,
        title="Historical Data + Forecast (Last 12 Months)",
    )
    fig13
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Comparison of Smoothing Methods
    """)
    return


@app.cell
def _(df, pl):
    # Add multiple smoothing methods
    df_smooth = df.with_columns(
        [
            pl.col("Passengers").rolling_mean(6).alias("MA_6"),
            pl.col("Passengers").rolling_mean(12).alias("MA_12"),
            pl.col("Passengers").ewm_mean(span=6).alias("EWM_6"),
            pl.col("Passengers").ewm_mean(span=12).alias("EWM_12"),
        ]
    )

    df_smooth.tail()
    return (df_smooth,)


@app.cell
def _(df_smooth, plot_comparison):
    # Overlay comparison
    fig14 = plot_comparison(
        df_smooth,
        columns=["Passengers", "MA_6", "MA_12", "EWM_6", "EWM_12"],
        comparison_mode="overlay",
        title="Comparison: Original vs Smoothed Series",
    )
    fig14
    return


@app.cell
def _(df_smooth, pl, plot_comparison):
    # Difference from original
    fig15 = plot_comparison(
        df_smooth.filter(pl.col("MA_12").is_not_null()),
        columns=["MA_6", "MA_12", "EWM_6", "EWM_12"],
        comparison_mode="difference",
        reference_column="Passengers",
        title="Smoothing Methods - Difference from Original",
    )
    fig15
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    This notebook demonstrated yohou's comprehensive time series plotting
    capabilities using the Air Passengers dataset:

    - **Basic plots**: Line plots, rolling statistics
    - **Seasonal analysis**: Boxplots, seasonal aggregation
    - **Diagnostics**: ACF, PACF, residual analysis
    - **Frequency analysis**: Periodogram, lag scatter
    - **Forecasting**: Prediction intervals, forecast visualization
    - **Comparison**: Multiple series overlay and difference plots

    All plots are interactive (Plotly) and follow consistent styling.
    """)
    return


if __name__ == "__main__":
    app.run()
