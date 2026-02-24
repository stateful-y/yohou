# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "plotly",
#     "yohou",
# ]
# ///
"""STL Decomposition.

Demonstrates STL decomposition with different component selections, robustness
settings, and window parameters.

Datasets: tourism_monthly
Demonstrates: plot_stl_components
"""

import marimo

__generated_with = "0.19.11"
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
    from yohou.plotting import plot_stl_components

    return (
        fetch_tourism_monthly,
        plot_stl_components,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # STL Decomposition

    ## What You'll Learn

    - Decomposing a time series into trend, seasonal, and residual components with `plot_stl_components`
    - Selecting specific components and tuning STL parameters (window sizes, robustness)

    ## Prerequisites

    Understanding of additive decomposition concepts (trend, seasonality, residuals).
    """)

@app.cell
def _(fetch_tourism_monthly):
    tourism = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    return (tourism,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Full STL Decomposition

    `plot_stl_components` applies STL (Seasonal and Trend decomposition using Loess)
    and displays selected **components**. By default all five panels are shown:
    observed, trend, seasonal, residual, and seasonal-adjusted.
    """)

@app.cell
def _(tourism, plot_stl_components):
    plot_stl_components(
        tourism,
        title="STL -- All Components (Default)",
    )

@app.cell
def _(tourism, plot_stl_components):
    plot_stl_components(
        tourism,
        components=["trend", "seasonal"],
        title="STL -- Trend and Seasonal Only",
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. STL Parameter Tuning

    Control the decomposition by setting **robust** (outlier resistance),
    explicit **period**, and **seasonal_window** size. Comparing robust vs
    non-robust helps gauge the impact of outliers on the trend estimate.
    """)

@app.cell
def _(tourism, plot_stl_components):
    plot_stl_components(
        tourism,
        robust=False,
        title="STL -- Non-Robust Estimation",
    )

@app.cell
def _(tourism, plot_stl_components):
    plot_stl_components(
        tourism,
        period=12,
        seasonal_window=15,
        title="STL -- Explicit Period=12, Seasonal Window=15",
    )

@app.cell
def _(tourism, plot_stl_components):
    plot_stl_components(
        tourism,
        components=["residual"],
        robust=True,
        title="STL -- Residual Only (Robust)",
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **STL decomposition** separates trend, seasonal, and residual components; use `components` to focus on specific parts
    - **Robust STL** (`robust=True`) reduces the influence of outliers on the trend and seasonal estimates
    - **Seasonal window** tuning controls how flexible the seasonal component is, larger values produce a more stable pattern

    ## Next Steps

    - **Seasonal diagnostics**: See `examples/plotting/seasonal.py` for overlays, ACF/PACF, and frequency domain analysis
    - **Forecast visualization**: See `examples/plotting/forecasting_visualization.py` for model comparison and prediction intervals
    - **Exploration**: See `examples/plotting/exploration.py` for rolling statistics and missing data audits
    """)

if __name__ == "__main__":
    app.run()
