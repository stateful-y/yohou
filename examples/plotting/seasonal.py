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
    "title": "Seasonal Analysis",
    "description": "Seasonal overlays, subseasonal structure, ACF/PACF correlation patterns, and STL decomposition for monthly, quarterly, and long-cycle datasets.",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    from yohou.datasets import (
        fetch_electricity_demand,
        fetch_sunspot,
        fetch_tourism_monthly,
        fetch_tourism_quarterly,
    )
    from yohou.plotting import (
        plot_autocorrelation,
        plot_decomposition,
        plot_partial_autocorrelation,
        plot_seasonal_heatmap,
        plot_seasonality,
        plot_subseasonality,
    )

    return (
        fetch_electricity_demand,
        fetch_sunspot,
        fetch_tourism_monthly,
        fetch_tourism_quarterly,
        plot_autocorrelation,
        plot_decomposition,
        plot_partial_autocorrelation,
        plot_seasonal_heatmap,
        plot_seasonality,
        plot_subseasonality,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Seasonal and Frequency Domain Analysis

    ## What You'll Learn

    - Overlaying seasonal cycles with [`plot_seasonality`](/pages/api/generated/yohou.plotting.diagnostics.plot_seasonality/) and highlighting specific years
    - Inspecting subseasonal structure with [`plot_subseasonality`](/pages/api/generated/yohou.plotting.diagnostics.plot_subseasonality/)
    - Identifying autocorrelation and partial autocorrelation patterns for AR/MA order selection
    - Applying these diagnostics to monthly, quarterly, and long-cycle datasets
    - Visualising seasonal patterns with 2-D heatmaps via [`plot_seasonal_heatmap`](/pages/api/generated/yohou.plotting.diagnostics.plot_seasonal_heatmap/)
    - STL decomposition with [`plot_decomposition`](/pages/api/generated/yohou.plotting.forecasting.plot_decomposition/) and flat decomposition parameters

    ## Prerequisites

    Basic understanding of seasonality, autocorrelation, and the frequency domain. Familiarity with time series terminology.
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Datasets

    We load three datasets using [`fetch_tourism_monthly`](/pages/api/generated/yohou.datasets._fetchers.fetch_tourism_monthly/), [`fetch_tourism_quarterly`](/pages/api/generated/yohou.datasets._fetchers.fetch_tourism_quarterly/),
    and [`fetch_sunspot`](/pages/api/generated/yohou.datasets._fetchers.fetch_sunspot/). The monthly tourism data provides a strong yearly cycle,
    the quarterly data demonstrates coarser granularity, and the sunspot data
    offers a long-cycle (approximately 11-year) periodic signal.
    """)


@app.cell
def _(fetch_sunspot, fetch_tourism_monthly, fetch_tourism_quarterly):
    tourism_monthly = (
        fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    )
    sunspots = fetch_sunspot().frame
    tourism_quarterly = fetch_tourism_quarterly().frame
    return tourism_monthly, tourism_quarterly


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Seasonal Overlay

    [`plot_seasonality`](/pages/api/generated/yohou.plotting.diagnostics.plot_seasonality/) overlays one line per cycle on the same seasonal axis (FPP3
    gg_season style). Vary the **seasonality**, use **highlight** to emphasise
    specific years, and pass **groups** for panel data.
    """)


@app.cell
def _(plot_seasonality, tourism_monthly):
    plot_seasonality(
        tourism_monthly,
        seasonality="month",
        title="Seasonal Overlay - Monthly",
    )


@app.cell
def _(plot_seasonality, tourism_monthly):
    plot_seasonality(
        tourism_monthly,
        seasonality="quarter",
        title="Seasonal Overlay - Quarterly",
    )


@app.cell
def _(plot_seasonality, tourism_monthly):
    plot_seasonality(
        tourism_monthly,
        seasonality="month",
        highlight=[2000, 2005],
        title="Seasonal Overlay - Highlighting 2000 and 2005",
    )


@app.cell
def _(plot_seasonality, tourism_quarterly):
    plot_seasonality(
        tourism_quarterly,
        seasonality="quarter",
        groups=["T1", "T2"],
        title="Seasonal Overlay - Tourism Quarterly Panel (T1 & T2)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Subseries Analysis

    [`plot_subseasonality`](/pages/api/generated/yohou.plotting.diagnostics.plot_subseasonality/) creates one mini-plot per season category (e.g. January
    values across all years). Use **kind** to control the visualization style
    and toggle **show_mean** for a horizontal reference.
    """)


@app.cell
def _(plot_subseasonality, tourism_monthly):
    plot_subseasonality(
        tourism_monthly,
        seasonality="month",
        show_mean=True,
        title="Subseries - Mean (default)",
    )


@app.cell
def _(plot_subseasonality, tourism_monthly):
    plot_subseasonality(
        tourism_monthly,
        seasonality="month",
        kind="lines",
        title="Subseries - Per-cycle lines overlay",
    )


@app.cell
def _(plot_subseasonality, tourism_monthly):
    plot_subseasonality(
        tourism_monthly,
        seasonality="quarter",
        kind="box",
        facet_n_cols=2,
        title="Subseries - Box (quarterly, 2-column grid)",
    )


@app.cell
def _(plot_subseasonality, tourism_monthly):
    plot_subseasonality(
        tourism_monthly,
        seasonality="month",
        kind="violin",
        title="Subseries - Violin",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Autocorrelation

    [`plot_autocorrelation`](/pages/api/generated/yohou.plotting.diagnostics.plot_autocorrelation/) plots the ACF up to **max_lags**. Significant seasonal
    peaks appear at multiples of the seasonal period (lag 12, 24, 36 for monthly
    data). Vary **confidence_level** and toggle **show_confidence**.
    """)


@app.cell
def _(plot_autocorrelation, tourism_monthly):
    plot_autocorrelation(
        tourism_monthly,
        max_lags=36,
        title="ACF - 36 Lags (Default 95% CI)",
    )


@app.cell
def _(plot_autocorrelation, tourism_monthly):
    plot_autocorrelation(
        tourism_monthly,
        max_lags=48,
        confidence_level=0.99,
        title="ACF - 48 Lags, 99% CI",
    )


@app.cell
def _(plot_autocorrelation, tourism_monthly):
    plot_autocorrelation(
        tourism_monthly,
        max_lags=36,
        show_confidence=False,
        title="ACF - Bars Only, No Confidence Bands",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Partial Autocorrelation

    [`plot_partial_autocorrelation`](/pages/api/generated/yohou.plotting.diagnostics.plot_partial_autocorrelation/) estimates the PACF, removing the effect of
    intermediate lags. Compare **method** options (`"yw"` vs `"ols"`) and vary
    **confidence_level** to assess significance.
    """)


@app.cell
def _(plot_partial_autocorrelation, tourism_monthly):
    plot_partial_autocorrelation(
        tourism_monthly,
        max_lags=36,
        title="PACF - Yule-Walker (Default)",
    )


@app.cell
def _(plot_partial_autocorrelation, tourism_monthly):
    plot_partial_autocorrelation(
        tourism_monthly,
        max_lags=36,
        method="ols",
        title="PACF - OLS Method",
    )


@app.cell
def _(plot_partial_autocorrelation, tourism_monthly):
    plot_partial_autocorrelation(
        tourism_monthly,
        max_lags=24,
        confidence_level=0.70,
        title="PACF - 24 Lags, 90% CI",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. STL Decomposition

    [`plot_decomposition`](/pages/api/generated/yohou.plotting.forecasting.plot_decomposition/) applies STL (Seasonal and Trend decomposition using Loess)
    and displays selected **components**. By default all five panels are shown:
    observed, trend, seasonal, residual, and seasonal-adjusted.
    Pass STL tuning parameters directly (e.g. `period`, `seasonal_window`, `robust`).
    """)


@app.cell
def _(plot_decomposition, tourism_monthly):
    plot_decomposition(
        tourism_monthly,
        ["observed", "trend", "seasonal", "residual", "seasonal_adjusted"],
        method="stl",
        title="STL - All Components (Default)",
    )


@app.cell
def _(plot_decomposition, tourism_monthly):
    plot_decomposition(
        tourism_monthly,
        ["trend", "seasonal"],
        method="stl",
        title="STL - Trend and Seasonal Only",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. STL Parameter Tuning

    Control the decomposition by setting **robust** (outlier resistance),
    explicit **period**, and **seasonal_window** size as flat parameters.
    Comparing robust vs non-robust helps gauge the impact of outliers on
    the trend estimate.
    """)


@app.cell
def _(plot_decomposition, tourism_monthly):
    plot_decomposition(
        tourism_monthly,
        ["observed", "trend", "seasonal", "residual", "seasonal_adjusted"],
        method="stl",
        robust=False,
        title="STL - Non-Robust Estimation",
    )


@app.cell
def _(plot_decomposition, tourism_monthly):
    plot_decomposition(
        tourism_monthly,
        ["observed", "trend", "seasonal", "residual", "seasonal_adjusted"],
        method="stl",
        period=12,
        seasonal_window=15,
        title="STL - Explicit Period=12, Seasonal Window=15",
    )


@app.cell
def _(plot_decomposition, tourism_monthly):
    plot_decomposition(
        tourism_monthly,
        ["residual"],
        method="stl",
        robust=True,
        title="STL - Residual Only (Robust)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Seasonal Heatmap

    [`plot_seasonal_heatmap`](/pages/api/generated/yohou.plotting.diagnostics.plot_seasonal_heatmap/) aggregates values onto a 2-D grid
    of two temporal periods. Vary **x_period**, **y_period**, and **agg** to
    reveal different seasonal structures.
    """)


@app.cell
def _(fetch_electricity_demand):
    _elec = fetch_electricity_demand().frame.head(8760)
    elec_heatmap = _elec.select("time", _elec.columns[1])
    return (elec_heatmap,)


@app.cell
def _(elec_heatmap, plot_seasonal_heatmap):
    plot_seasonal_heatmap(
        elec_heatmap,
        x_period="hour",
        y_period="month",
        title="Seasonal Heatmap - Hour x Month (Mean)",
    )


@app.cell
def _(elec_heatmap, plot_seasonal_heatmap):
    plot_seasonal_heatmap(
        elec_heatmap,
        x_period="hour",
        y_period="day_of_week",
        title="Seasonal Heatmap - Hour x Day of Week",
    )


@app.cell
def _(elec_heatmap, plot_seasonal_heatmap):
    plot_seasonal_heatmap(
        elec_heatmap,
        x_period="hour",
        y_period="month",
        agg="max",
        colorscale="Hot",
        reverse_y=True,
        title="Seasonal Heatmap - Max, Hot Colorscale, Reversed Y",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Seasonal overlays** ([`plot_seasonality`](/pages/api/generated/yohou.plotting.diagnostics.plot_seasonality/)) make year-over-year comparisons immediate; `highlight` draws attention to specific cycles
    - **Subseries plots** ([`plot_subseasonality`](/pages/api/generated/yohou.plotting.diagnostics.plot_subseasonality/)) reveal within-season trends over time; the mean line shows each season's average level
    - **ACF** identifies repeating correlation patterns at seasonal multiples; **PACF** isolates direct lag effects for AR order selection
    - **Seasonal heatmaps** ([`plot_seasonal_heatmap`](/pages/api/generated/yohou.plotting.diagnostics.plot_seasonal_heatmap/)) aggregate values across two time dimensions; the `x_period`/`y_period` pair reveals different seasonal structures
    - **STL decomposition** separates trend, seasonal, and residual components; `robust=True` reduces outlier influence
    - **Seasonal window** tuning controls seasonal component flexibility; larger values produce a more stable pattern

    ## Next Steps

    - **Correlation diagnostics**: See [`correlation.py`](/examples/plotting/correlation/) for scatter matrices and cross-correlation
    - **Forecast visualization**: See [`forecasting_visualization.py`](/examples/plotting/forecasting_visualization/) for model comparison
    - **Exploration**: See [`exploration.py`](/examples/plotting/exploration/) for rolling statistics and missing data audits
    """)


if __name__ == "__main__":
    app.run()
