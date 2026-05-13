# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Pedestrian Counts Dataset",
    "description": "Sensor-level panel exploration of Melbourne pedestrian counts with per-sensor boxplots, hourly seasonal patterns, and panel structure inspection.",
    "category": "how-to",
    "section": "data-catalog",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from yohou.datasets import fetch_pedestrian_counts
    from yohou.plotting import (
        plot_boxplot,
        plot_seasonality,
        plot_time_series,
    )
    from yohou.utils.panel import inspect_panel

    return (
        fetch_pedestrian_counts,
        inspect_panel,
        mo,
        plot_boxplot,
        plot_seasonality,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Pedestrian Counts Dataset

    This notebook shows how to perform sensor-level panel exploration of Melbourne pedestrian counts with per-sensor boxplots, hourly seasonal patterns, and panel structure inspection.

    **Prerequisites:** None. this is a standalone dataset exploration.
    """)


@app.cell
def _(fetch_pedestrian_counts, plot_time_series):
    _all = fetch_pedestrian_counts().frame
    # Select first 6 sensors for a manageable panel
    _cols = ["time"] + [c for c in _all.columns if c != "time"][:6]
    df = _all.select(_cols).head(24 * 7 * 52)
    plot_time_series(df, title="Pedestrian Counts (6 Sensors)")
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Inspect Panel Structure

    By default, `fetch_pedestrian_counts()` returns 20 sensors using the
    `Tn__count` convention (pass `n_series=None` for all 66). Here we work
    with the first 6 sensors.
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
    ## 2. Sensor Comparison

    Comparing counts across sensors shows how pedestrian traffic varies
    by location. Some sensors are in high-traffic areas while others
    capture quieter streets.
    """)


@app.cell
def _(df, plot_time_series):
    _count_cols = [c for c in df.columns if c.endswith("__count")][:3]

    plot_time_series(
        df.head(24 * 7),  # One week of hourly data
        columns=_count_cols,
        title="Pedestrian Counts - First 3 Sensors (1 Week)",
        y_label="Count",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. All Selected Sensors

    Plotting all 6 selected sensors reveals scale differences and shared
    temporal patterns across locations.
    """)


@app.cell
def _(df, plot_time_series):
    _count_cols = [c for c in df.columns if c.endswith("__count")]

    plot_time_series(
        df.head(24 * 14),  # Two weeks of hourly data
        columns=_count_cols,
        title="Pedestrian Counts - 6 Sensors (2 Weeks)",
        y_label="Count",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Count Distribution by Sensor

    Boxplots reveal the distribution of hourly counts for each sensor,
    making it easy to compare traffic levels and variability.
    """)


@app.cell
def _(df, plot_boxplot):
    plot_boxplot(
        df,
        title="Pedestrian Count Distribution by Sensor",
        y_label="Count",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Hour-of-Day Patterns

    Aggregating by hour reveals the daily pedestrian traffic cycle, with
    peaks during commute hours and lunch time.
    """)


@app.cell
def _(df, plot_seasonality):
    _first_col = [c for c in df.columns if c.endswith("__count")][0]
    plot_seasonality(
        df,
        columns=_first_col,
        seasonality="hour",
        title="T1 - Average Pedestrian Count by Hour",
        y_label="Average Count",
    )



if __name__ == "__main__":
    app.run()
