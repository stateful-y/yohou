"""Trend and Seasonality Forecasters with DecompositionPipeline.

Demonstrates PolynomialTrendForecaster, PatternSeasonalityForecaster,
FourierSeasonalityForecaster, and DecompositionPipeline.
"""

import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
async def _():
    import sys

    if "pyodide" in sys.modules:
        import micropip

        await micropip.install(["plotly", "scikit-learn", "yohou"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Trend and Seasonality Decomposition

    Classical decomposition splits a time series into **trend**, **seasonality**,
    and **residual** components. Yohou provides specialized forecasters for each
    component and `DecompositionPipeline` to chain them.

    ## What You'll Learn

    - `PolynomialTrendForecaster`: Fit polynomial trend
    - `PatternSeasonalityForecaster`: Repeat seasonal patterns (naive, average, median)
    - `FourierSeasonalityForecaster`: Fourier-based seasonal modeling
    - `DecompositionPipeline`: Chain trend + seasonality + residual forecasters
    - Visualizing decomposition with `plot_components`

    ## Prerequisites

    Understanding of trend and seasonality concepts.
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import DecompositionPipeline
    from yohou.datasets import fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_components, plot_forecast
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer
    from yohou.stationarity import (
        FourierSeasonalityForecaster,
        LogTransformer,
        PatternSeasonalityForecaster,
        PolynomialTrendForecaster,
    )

    return (
        DecompositionPipeline,
        FourierSeasonalityForecaster,
        LagTransformer,
        LogTransformer,
        MeanAbsoluteError,
        PatternSeasonalityForecaster,
        PointReductionForecaster,
        PolynomialTrendForecaster,
        Ridge,
        fetch_tourism_monthly,
        plot_components,
        plot_forecast,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We load a time series dataset suitable for demonstrating trend and seasonality decomposition.
    """)
    return


@app.cell
def _(fetch_tourism_monthly):
    y = (
        fetch_tourism_monthly()
        .frame.select("time", "T1__tourists")
        .rename({"T1__tourists": "passengers"})
    )
    y_train = y.head(280)
    y_test = y.tail(65)
    fh = len(y_test)

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    return fh, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. PolynomialTrendForecaster

    Fits a polynomial to the time index. `degree=1` is linear, `degree=2` is quadratic.
    """)
    return


@app.cell
def _(PolynomialTrendForecaster, fh, plot_forecast, y_test, y_train):
    trend_fc = PolynomialTrendForecaster(degree=1)
    trend_fc.fit(y_train, forecasting_horizon=fh)
    y_pred_trend = trend_fc.predict(forecasting_horizon=fh)

    print(f"Trend prediction (first 5): {y_pred_trend['passengers'].head(5).to_list()}")
    plot_forecast(y_test, y_pred_trend, y_train=y_train, title="Linear Trend Forecast")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. PatternSeasonalityForecaster

    Captures recurring patterns by averaging (or median/naive) over past seasons.

    - `method="naive"`: Repeat last season's exact values
    - `method="average"`: Average across all observed seasons
    - `method="median"`: Median across seasons
    """)
    return


@app.cell
def _(PatternSeasonalityForecaster, fh, y_train):
    for method in ["naive", "average", "median"]:
        season_fc = PatternSeasonalityForecaster(seasonality=12, method=method)
        season_fc.fit(y_train, forecasting_horizon=fh)
        pred = season_fc.predict(forecasting_horizon=fh)
        print(f"method={method:>7s}  first pred: {pred['passengers'][0]:.1f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. FourierSeasonalityForecaster

    Models seasonality using Fourier basis functions. More harmonics
    capture more complex seasonal shapes.
    """)
    return


@app.cell
def _(FourierSeasonalityForecaster, fh, plot_forecast, y_test, y_train):
    fourier_fc = FourierSeasonalityForecaster(
        seasonality=12.0,
        harmonics=[1, 2, 3],
    )
    fourier_fc.fit(y_train, forecasting_horizon=fh)
    y_pred_fourier = fourier_fc.predict(forecasting_horizon=fh)

    plot_forecast(
        y_test, y_pred_fourier, y_train=y_train, title="Fourier Seasonality (3 harmonics)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. DecompositionPipeline

    Chain multiple component forecasters sequentially. Each forecaster in the pipeline
    fits on the **residuals** from the previous one.

    `trend → seasonality → residual`
    """)
    return


@app.cell
def _(
    DecompositionPipeline,
    LagTransformer,
    LogTransformer,
    PatternSeasonalityForecaster,
    PointReductionForecaster,
    PolynomialTrendForecaster,
    Ridge,
    fh,
    y_train,
):
    decomp = DecompositionPipeline(
        forecasters=[
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", PatternSeasonalityForecaster(seasonality=12, method="average")),
            (
                "residual",
                PointReductionForecaster(
                    estimator=Ridge(),
                    feature_transformer=LagTransformer(lag=list(range(1, 13))),
                ),
            ),
        ],
        target_transformer=LogTransformer(),
        store_residuals=True,
    )

    decomp.fit(y_train, forecasting_horizon=fh)
    y_pred_decomp = decomp.predict(forecasting_horizon=fh)
    print(f"DecompositionPipeline prediction: {len(y_pred_decomp)} steps")
    return decomp, y_pred_decomp


@app.cell
def _(MeanAbsoluteError, y_pred_decomp, y_test, y_train):
    mae = MeanAbsoluteError()
    mae.fit(y_train)
    score = mae.score(y_test, y_pred_decomp)
    print(f"Decomposition MAE: {score:.2f}")
    return


@app.cell
def _(plot_forecast, y_pred_decomp, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_decomp,
        y_train=y_train,
        title="DecompositionPipeline: Trend + Season + Residual",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Visualizing Components

    With `store_residuals=True`, access each component's contribution via
    `named_forecasters_`.
    """)
    return


@app.cell
def _(decomp, fh, plot_components, y_test):
    components = {}
    for name, fc, *_ in decomp.forecasters_:
        comp_pred = fc.predict(forecasting_horizon=fh)
        components[name] = comp_pred

    plot_components(
        y_test,
        components,
        title="Decomposition Components",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - `PolynomialTrendForecaster`: Polynomial trend modeling (linear, quadratic, etc.)
    - `PatternSeasonalityForecaster`: Pattern-based seasonality (naive, average, median)
    - `FourierSeasonalityForecaster`: Smooth seasonal curves via harmonics
    - `DecompositionPipeline`: Chain forecasters; each fits on previous residuals
    - `store_residuals=True` enables component inspection
    - The pipeline prediction is the **sum** of all component predictions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Stationarity transforms**: See `stationarity_transforms.py` for differencing
    - **Reduction forecasting**: See `point/reduction_forecaster.py`
    - **Interval forecasting**: See `interval/` for prediction intervals
    """)
    return


if __name__ == "__main__":
    app.run()
