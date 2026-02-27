# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Pedestrian Counts Analytics",
    "description": "Advanced sensor analytics on Melbourne pedestrian data with autocorrelation, partial autocorrelation, rolling statistics, and day-of-week seasonality.",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from yohou.datasets import fetch_pedestrian_counts
    from yohou.plotting import (
        plot_autocorrelation,
        plot_partial_autocorrelation,
        plot_rolling_statistics,
        plot_seasonality,
        plot_time_series,
    )
    from yohou.utils.panel import inspect_panel

    return (
        fetch_pedestrian_counts,
        inspect_panel,
        mo,
        plot_autocorrelation,
        plot_partial_autocorrelation,
        plot_rolling_statistics,
        plot_seasonality,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pedestrian Counts - Advanced Sensor Analytics

    This notebook builds on the basic Pedestrian Counts exploration with
    deeper diagnostic plots. We use hourly data from Melbourne sensors to
    examine temporal dependencies, spectral structure, and seasonal
    patterns at multiple scales.

    ## What You'll Learn

    - Autocorrelation and partial autocorrelation for hourly/daily lags
    - Day-of-week seasonal patterns
    - Rolling statistics to track trend and volatility

    ## Prerequisites

    See `examples/datasets/pedestrian_counts.py` for basic panel
    exploration (boxplots, hour-of-day patterns).
    """)


@app.cell
def _(fetch_pedestrian_counts, inspect_panel, plot_time_series):
    _all = fetch_pedestrian_counts().frame
    _cols = ["time"] + [c for c in _all.columns if c != "time"][:4]
    pedestrian = _all.select(_cols).head(24 * 7 * 52)
    _globals, _groups = inspect_panel(pedestrian)

    # Show one week for a clear view
    plot_time_series(
        pedestrian.head(24 * 7),
        title="Pedestrian Counts - 4 Sensors (1 Week)",
    )
    return (pedestrian,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Autocorrelation

    With hourly data, strong spikes at lag 24 (daily) and lag 168 (weekly)
    reveal the natural periodic structure of pedestrian traffic.
    """)


@app.cell
def _(pedestrian, plot_autocorrelation):
    _col = [c for c in pedestrian.columns if c.endswith("__count")][0]
    plot_autocorrelation(
        pedestrian,
        columns=_col,
        max_lags=200,
        title="T1 - Autocorrelation (Hourly Lags)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Partial Autocorrelation

    PACF identifies which lags have *direct* predictive value. Typical
    spikes at lags 1, 24, and 168 suggest that a model needs short-term,
    daily, and weekly lag features.
    """)


@app.cell
def _(pedestrian, plot_partial_autocorrelation):
    _col = [c for c in pedestrian.columns if c.endswith("__count")][0]
    plot_partial_autocorrelation(
        pedestrian,
        columns=_col,
        max_lags=50,
        title="T1 - Partial Autocorrelation",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Day-of-Week Seasonality

    Overlaying each day of the week reveals how pedestrian patterns differ
    between weekdays and weekends (e.g. commuter peaks vs. leisure traffic).
    """)


@app.cell
def _(pedestrian, plot_seasonality):
    _col = [c for c in pedestrian.columns if c.endswith("__count")][0]
    plot_seasonality(
        pedestrian,
        columns=_col,
        seasonality="weekday",
        title="T1 - Day-of-Week Seasonality",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Rolling Statistics

    A 168-hour (1-week) rolling mean and standard deviation track how
    average pedestrian traffic and its variability evolve over time.
    """)


@app.cell
def _(pedestrian, plot_rolling_statistics):
    _col = [c for c in pedestrian.columns if c.endswith("__count")][0]
    plot_rolling_statistics(
        pedestrian,
        columns=_col,
        window_size=168,
        statistics=["mean", "std"],
        title="T1 - Rolling Mean and Std (1-week window)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **ACF/PACF**: Hourly data shows strong daily (lag 24) and weekly (lag 168) cycles
    - **Day-of-week seasonality**: Weekday vs weekend patterns are visually distinct
    - **Rolling statistics**: 1-week window tracks evolving traffic levels

    ## Next Steps

    - **Basic sensor exploration**: See [`examples/datasets/pedestrian_counts.py`](/examples/datasets/pedestrian_counts/)
    - **Panel data analysis**: See [`examples/datasets/australian_tourism.py`](/examples/datasets/australian_tourism/)
    - **Signal diagnostics**: See [`examples/datasets/sunspots.py`](/examples/datasets/sunspots/) for spectral analysis
    """)


if __name__ == "__main__":
    app.run()
