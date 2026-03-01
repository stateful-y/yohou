# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.19.11"
__gallery__ = {
    "title": "Decomposition Plots",
    "description": "Visualise STL decomposition of monthly tourism data into trend, seasonal, residual, and seasonal-adjusted components using plot_components in STL mode.",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    from yohou.datasets import (
        fetch_tourism_monthly,
    )
    from yohou.plotting import plot_components

    return (
        fetch_tourism_monthly,
        plot_components,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # STL Decomposition

    ## What You'll Learn

    - Decomposing a time series into trend, seasonal, and residual components with [`plot_components`](/pages/api/generated/yohou.plotting.forecasting.plot_components/) in STL mode
    - Selecting specific components and tuning STL parameters (window sizes, robustness) via `stl_kwargs`

    ## Prerequisites

    Understanding of additive decomposition concepts (trend, seasonality, residuals).
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Data

    We load the Monthly Tourism dataset via [`fetch_tourism_monthly`](/pages/api/generated/yohou.datasets._fetchers.fetch_tourism_monthly/) and extract a
    single univariate series (tourists from region T1) for decomposition.
    """)


@app.cell
def _(fetch_tourism_monthly):
    tourism = (
        fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    )
    return (tourism,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Full STL Decomposition

    [`plot_components`](/pages/api/generated/yohou.plotting.forecasting.plot_components/) applies STL (Seasonal and Trend decomposition using Loess)
    and displays selected **components**. By default all five panels are shown:
    observed, trend, seasonal, residual, and seasonal-adjusted.
    Pass STL tuning parameters via the `stl_kwargs` dict.
    """)


@app.cell
def _(tourism, plot_components):
    plot_components(
        tourism,
        ["observed", "trend", "seasonal", "residual", "seasonal_adjusted"],
        title="STL - All Components (Default)",
    )


@app.cell
def _(tourism, plot_components):
    plot_components(
        tourism,
        ["trend", "seasonal"],
        title="STL - Trend and Seasonal Only",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. STL Parameter Tuning

    Control the decomposition by setting **robust** (outlier resistance),
    explicit **period**, and **seasonal_window** size via `stl_kwargs`.
    Comparing robust vs non-robust helps gauge the impact of outliers on
    the trend estimate.
    """)


@app.cell
def _(tourism, plot_components):
    plot_components(
        tourism,
        ["observed", "trend", "seasonal", "residual", "seasonal_adjusted"],
        stl_kwargs={"robust": False},
        title="STL - Non-Robust Estimation",
    )


@app.cell
def _(tourism, plot_components):
    plot_components(
        tourism,
        ["observed", "trend", "seasonal", "residual", "seasonal_adjusted"],
        stl_kwargs={"period": 12, "seasonal_window": 15},
        title="STL - Explicit Period=12, Seasonal Window=15",
    )


@app.cell
def _(tourism, plot_components):
    plot_components(
        tourism,
        ["residual"],
        stl_kwargs={"robust": True},
        title="STL - Residual Only (Robust)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **STL decomposition** separates trend, seasonal, and residual components; use the `components` list to focus on specific parts
    - **Robust STL** (`stl_kwargs={"robust": True}`) reduces the influence of outliers on the trend and seasonal estimates
    - **Seasonal window** tuning controls how flexible the seasonal component is, larger values produce a more stable pattern

    ## Next Steps

    - **Seasonal diagnostics**: See [`examples/plotting/seasonal.py`](/examples/plotting/seasonal/) for overlays, ACF/PACF, and frequency domain analysis
    - **Forecast visualization**: See [`examples/plotting/forecasting_visualization.py`](/examples/plotting/forecasting_visualization/) for model comparison and prediction intervals
    - **Exploration**: See [`examples/plotting/exploration.py`](/examples/plotting/exploration/) for rolling statistics and missing data audits
    """)


if __name__ == "__main__":
    app.run()
