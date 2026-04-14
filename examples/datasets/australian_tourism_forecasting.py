# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "statsmodels",
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Tourism Quarterly Analytics",
    "description": "Diagnostic analytics on quarterly tourism panel data with autocorrelation, correlation heatmaps, lag scatter, and seasonal subseries plots.",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from yohou.datasets import fetch_tourism_quarterly
    from yohou.plotting import (
        plot_autocorrelation,
        plot_correlation_heatmap,
        plot_lag_scatter,
        plot_partial_autocorrelation,
        plot_rolling_statistics,
        plot_subseasonality,
        plot_time_series,
    )

    return (
        fetch_tourism_quarterly,
        mo,
        plot_autocorrelation,
        plot_correlation_heatmap,
        plot_lag_scatter,
        plot_partial_autocorrelation,
        plot_rolling_statistics,
        plot_subseasonality,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tourism Quarterly - Advanced Panel Analytics

    This notebook builds on the basic Tourism Quarterly exploration with
    deeper diagnostic plots. We use 8 quarterly tourism series (T3-T10)
    from the Monash forecasting competition.

    ## What You'll Learn

    - Autocorrelation and partial autocorrelation for quarterly lags
    - Inter-series correlation heatmap across panel groups
    - Lag scatter plots to visualise temporal dependencies
    - Seasonal subseries for quarter-by-quarter patterns
    - Rolling statistics for trend tracking

    ## Prerequisites

    See `examples/datasets/australian_tourism.py` for basic panel
    exploration of this dataset.
    """)


@app.cell
def _(fetch_tourism_quarterly, plot_time_series):
    _all = fetch_tourism_quarterly().frame
    _cols = ["time"] + [c for c in _all.columns if c != "time"][2:10]
    tourism = _all.select(_cols).drop_nulls()

    plot_time_series(tourism, title="Tourism Quarterly - 8 Series (T3-T10)")
    return (tourism,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Autocorrelation

    The ACF reveals the temporal dependence structure. With quarterly data
    we expect significant spikes at lags 4, 8, 12, etc. reflecting the
    annual seasonal cycle.
    """)


@app.cell
def _(plot_autocorrelation, tourism):
    _col = [c for c in tourism.columns if c.endswith("__tourists")][0]
    plot_autocorrelation(
        tourism,
        columns=_col,
        max_lags=20,
        title="T3 - Autocorrelation (Quarterly Lags)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Partial Autocorrelation

    PACF isolates the direct effect of each lag after removing the
    influence of intermediate lags. A significant spike at lag 4 confirms
    the direct annual seasonal link.
    """)


@app.cell
def _(plot_partial_autocorrelation, tourism):
    _col = [c for c in tourism.columns if c.endswith("__tourists")][0]
    plot_partial_autocorrelation(
        tourism,
        columns=_col,
        max_lags=16,
        title="T3 - Partial Autocorrelation",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Correlation Heatmap

    Pairwise Pearson correlations across all 8 panel series. High
    correlations indicate series that share common drivers (e.g.
    economic conditions, holiday timing).
    """)


@app.cell
def _(plot_correlation_heatmap, tourism):
    plot_correlation_heatmap(
        tourism,
        title="Inter-Series Correlation Heatmap",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Lag Scatter Plots

    Scatter y(t) vs y(t-lag) for lags 1 and 4. A tight diagonal cluster
    at lag 4 confirms the annual seasonal structure.
    """)


@app.cell
def _(plot_lag_scatter, tourism):
    _col = [c for c in tourism.columns if c.endswith("__tourists")][0]
    plot_lag_scatter(
        tourism,
        columns=_col,
        lags=[1, 4],
        show_regression=True,
        title="T3 - Lag Scatter (1-quarter and 1-year)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Seasonal Subseries

    Each subplot shows observations from one quarter across all years,
    with the mean line highlighting whether particular quarters are
    consistently higher or lower than others.
    """)


@app.cell
def _(plot_subseasonality, tourism):
    _col = [c for c in tourism.columns if c.endswith("__tourists")][0]
    plot_subseasonality(
        tourism,
        columns=_col,
        seasonality="quarter",
        title="T3 - Seasonal Subseries (by Quarter)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Rolling Statistics

    A 4-quarter (1-year) rolling mean smooths out seasonal noise and
    reveals the underlying trend in tourism demand.
    """)


@app.cell
def _(plot_rolling_statistics, tourism):
    _col = [c for c in tourism.columns if c.endswith("__tourists")][0]
    plot_rolling_statistics(
        tourism,
        columns=_col,
        window_size=12,
        statistics=["mean", "std"],
        title="T3 - Rolling Mean and Std (4-quarter window)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **ACF/PACF**: Quarterly data shows strong seasonal spikes at lag 4 (annual cycle)
    - **Correlation heatmap**: Reveals which panel series share common patterns
    - **Lag scatter**: Tight diagonal clusters at lag 4 confirm annual seasonality
    - **Subseries**: Quarter-by-quarter view isolates within-year variation
    - **Rolling statistics**: 4-quarter window reveals long-term trends

    ## Next Steps

    - **Basic panel exploration**: See [`examples/datasets/australian_tourism.py`](/examples/datasets/australian_tourism/)
    - **Weekly panel data**: See [`examples/datasets/store_sales.py`](/examples/datasets/store_sales/)
    - **Hourly panel data**: See [`examples/datasets/pedestrian_counts.py`](/examples/datasets/pedestrian_counts/)
    """)


if __name__ == "__main__":
    app.run()
