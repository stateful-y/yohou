"""Seasonal and Frequency Domain Analysis.

Demonstrates seasonality overlays, subseries plots, ACF/PACF, power spectrum, and
phase spectrum with varied parameter combinations.

Datasets: tourism_monthly, sunspot, tourism_quarterly
Demonstrates: plot_seasonality, plot_subseasonality, plot_autocorrelation,
    plot_partial_autocorrelation
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
async def _():
    import sys as _sys

    if "pyodide" in _sys.modules:
        import micropip

        await micropip.install(["plotly", "yohou"])
    return


@app.cell(hide_code=True)
def _():
    from yohou.datasets import (
        fetch_sunspot,
        fetch_tourism_monthly,
        fetch_tourism_quarterly,
    )
    from yohou.plotting import (
        plot_autocorrelation,
        plot_partial_autocorrelation,
        plot_seasonality,
        plot_subseasonality,
    )

    return (
        fetch_sunspot,
        fetch_tourism_monthly,
        fetch_tourism_quarterly,
        plot_autocorrelation,
        plot_partial_autocorrelation,
        plot_seasonality,
        plot_subseasonality,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Seasonal and Frequency Domain Analysis

    ## What You'll Learn

    - Overlaying seasonal cycles with `plot_seasonality` and highlighting specific years
    - Inspecting subseasonal structure with `plot_subseasonality`
    - Identifying autocorrelation and partial autocorrelation patterns for AR/MA order selection
    - Applying these diagnostics to monthly, quarterly, and long-cycle datasets

    ## Prerequisites

    Basic understanding of seasonality, autocorrelation, and the frequency domain. Familiarity with time series terminology.
    """)
    return


@app.cell
def _(fetch_tourism_monthly, fetch_tourism_quarterly, fetch_sunspot):
    air = fetch_tourism_monthly().frame.select("time", "T1__tourists").rename({"T1__tourists": "passengers"})
    sunspots = fetch_sunspot().frame
    tourism = fetch_tourism_quarterly().frame
    return air, sunspots, tourism


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Seasonal Overlay

    `plot_seasonality` overlays one line per cycle on the same seasonal axis (FPP3
    gg_season style). Vary the **seasonality**, use **highlight** to emphasise
    specific years, and pass **panel_group_names** for panel data.
    """)
    return


@app.cell
def _(air, plot_seasonality):
    plot_seasonality(
        air,
        seasonality="month",
        title="Seasonal Overlay -- Monthly",
    )
    return


@app.cell
def _(air, plot_seasonality):
    plot_seasonality(
        air,
        seasonality="quarter",
        title="Seasonal Overlay -- Quarterly",
    )
    return


@app.cell
def _(air, plot_seasonality):
    plot_seasonality(
        air,
        seasonality="month",
        highlight=[1958, 1960],
        title="Seasonal Overlay -- Highlighting 1958 and 1960",
    )
    return


@app.cell
def _(plot_seasonality, tourism):
    plot_seasonality(
        tourism,
        seasonality="quarter",
        panel_group_names=["T1", "T2"],
        title="Seasonal Overlay -- Tourism Quarterly Panel (T1 & T2)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Subseries Analysis

    `plot_subseasonality` creates one mini-plot per season category (e.g. January
    values across all years). Toggle **show_mean** for a horizontal reference, and
    adjust **n_cols** to change the grid layout.
    """)
    return


@app.cell
def _(air, plot_subseasonality):
    plot_subseasonality(
        air,
        seasonality="month",
        show_mean=True,
        title="Subseries -- Monthly with Mean Line",
    )
    return


@app.cell
def _(air, plot_subseasonality):
    plot_subseasonality(
        air,
        seasonality="quarter",
        facet_n_cols=2,
        title="Subseries -- Quarterly (2-Column Grid)",
    )
    return


@app.cell
def _(air, plot_subseasonality):
    plot_subseasonality(
        air,
        seasonality="month",
        show_mean=False,
        title="Subseries -- Monthly, No Mean Line",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Autocorrelation

    `plot_autocorrelation` plots the ACF up to **max_lags**. Significant seasonal
    peaks appear at multiples of the seasonal period (lag 12, 24, 36 for monthly
    data). Vary **confidence_level** and toggle **show_confidence**.
    """)
    return


@app.cell
def _(air, plot_autocorrelation):
    plot_autocorrelation(
        air,
        max_lags=36,
        title="ACF -- 36 Lags (Default 95% CI)",
    )
    return


@app.cell
def _(air, plot_autocorrelation):
    plot_autocorrelation(
        air,
        max_lags=48,
        confidence_level=0.99,
        title="ACF -- 48 Lags, 99% CI",
    )
    return


@app.cell
def _(air, plot_autocorrelation):
    plot_autocorrelation(
        air,
        max_lags=36,
        show_confidence=False,
        title="ACF -- Bars Only, No Confidence Bands",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Partial Autocorrelation

    `plot_partial_autocorrelation` estimates the PACF, removing the effect of
    intermediate lags. Compare **method** options (`"yw"` vs `"ols"`) and vary
    **confidence_level** to assess significance.
    """)
    return


@app.cell
def _(air, plot_partial_autocorrelation):
    plot_partial_autocorrelation(
        air,
        max_lags=36,
        title="PACF -- Yule-Walker (Default)",
    )
    return


@app.cell
def _(air, plot_partial_autocorrelation):
    plot_partial_autocorrelation(
        air,
        max_lags=36,
        method="ols",
        title="PACF -- OLS Method",
    )
    return


@app.cell
def _(air, plot_partial_autocorrelation):
    plot_partial_autocorrelation(
        air,
        max_lags=24,
        confidence_level=0.70,
        title="PACF -- 24 Lags, 90% CI",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Seasonal overlays** (`plot_seasonality`) make year-over-year comparisons immediate; `highlight` draws attention to specific cycles
    - **Subseries plots** (`plot_subseasonality`) reveal within-season trends over time; the mean line shows each season's average level
    - **ACF** identifies repeating correlation patterns at seasonal multiples; **PACF** isolates direct lag effects for AR order selection

    ## Next Steps

    - **STL decomposition**: See `examples/plotting/decomposition.py` for trend/seasonal/residual decomposition
    - **Correlation diagnostics**: See `examples/plotting/correlation.py` for scatter matrices and cross-correlation
    - **Forecast visualization**: See `examples/plotting/forecasting_visualization.py` for model comparison
    """)
    return


if __name__ == "__main__":
    app.run()
