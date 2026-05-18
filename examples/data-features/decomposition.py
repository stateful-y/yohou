# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Decomposition",
    "description": "Chain PolynomialTrendForecaster, PatternSeasonalityForecaster, and FourierSeasonalityForecaster inside DecompositionPipeline with component visualisation.",
    "category": "tutorial",
    "section": "data-features",
    "companion": "/pages/tutorials/decomposition/",
    "api_references": ["DecompositionPipeline", "FourierSeasonalityForecaster", "LogTransformer", "PatternSeasonalityForecaster", "PointReductionForecaster", "PolynomialTrendForecaster", "plot_decomposition", "plot_score_per_step"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Trend and Seasonality Decomposition

    Classical decomposition splits a time series into **trend**, **seasonality**,
    and **residual** components. Yohou provides specialized forecasters for each
    component and [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) to chain them.

    ## Prerequisites

    Understanding of trend and seasonality concepts.
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    from sklearn.linear_model import LinearRegression, Ridge

    from yohou.compose import DecompositionPipeline
    from yohou.datasets import fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import train_test_split
    from yohou.plotting import (
        plot_decomposition,
        plot_forecast,
        plot_score_per_step,
        plot_time_series,
    )
    from yohou.point import PointReductionForecaster
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
        LinearRegression,
        LogTransformer,
        MeanAbsoluteError,
        PatternSeasonalityForecaster,
        PointReductionForecaster,
        PolynomialTrendForecaster,
        Ridge,
        deepcopy,
        fetch_tourism_monthly,
        plot_decomposition,
        plot_forecast,
        plot_score_per_step,
        plot_time_series,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We load a time series dataset suitable for demonstrating trend and seasonality decomposition.
    """)


@app.cell
def _(fetch_tourism_monthly, plot_time_series, train_test_split):
    y = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    y_train, y_test = train_test_split(y, test_size=65)
    fh = len(y_test)

    plot_time_series(y, title="Monthly Tourism (T1)")
    return fh, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. PolynomialTrendForecaster

    Fits a polynomial to the time index. `degree=1` is linear, `degree=2` is quadratic.
    """)


@app.cell
def _(PolynomialTrendForecaster, fh, plot_forecast, y_test, y_train):
    trend_fc = PolynomialTrendForecaster(degree=1)
    trend_fc.fit(y_train, forecasting_horizon=fh)
    y_pred_trend = trend_fc.predict(forecasting_horizon=fh)

    print(f"Trend prediction (first 5): {y_pred_trend['tourists'].head(5).to_list()}")
    plot_forecast(y_test, y_pred_trend, y_train=y_train, title="Linear Trend Forecast")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. PatternSeasonalityForecaster

    Captures recurring patterns by averaging (or median/naive) over past seasons.

    - `method="naive"`: Repeat last season's exact values
    - `method="average"`: Average across all observed seasons
    - `method="median"`: Median across seasons
    """)


@app.cell
def _(PatternSeasonalityForecaster, fh, y_train):
    for method in ["naive", "average", "median"]:
        season_fc = PatternSeasonalityForecaster(seasonality=12, method=method)
        season_fc.fit(y_train, forecasting_horizon=fh)
        pred = season_fc.predict(forecasting_horizon=fh)
        print(f"method={method:>7s}  first pred: {pred['tourists'][0]:.1f}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. FourierSeasonalityForecaster

    Models seasonality using Fourier basis functions. More harmonics
    capture more complex seasonal shapes.
    """)


@app.cell
def _(
    FourierSeasonalityForecaster,
    LinearRegression,
    fh,
    plot_forecast,
    y_test,
    y_train,
):
    fourier_fc = FourierSeasonalityForecaster(
        estimator=LinearRegression(),
        seasonality=12.0,
        harmonics=[1, 2, 3],
    )
    fourier_fc.fit(y_train, forecasting_horizon=fh)
    y_pred_fourier = fourier_fc.predict(forecasting_horizon=fh)

    plot_forecast(y_test, y_pred_fourier, y_train=y_train, title="Fourier Seasonality (3 harmonics)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. DecompositionPipeline

    Chain multiple component forecasters sequentially. Each forecaster in the pipeline
    fits on the **residuals** from the previous one.

    `trend → seasonality → residual`
    """)


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
                    estimator=Ridge(alpha=100),
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


@app.cell
def _(plot_forecast, y_pred_decomp, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_decomp,
        y_train=y_train,
        title="DecompositionPipeline: Trend + Season + Residual",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Visualizing Components

    With `store_residuals=True`, access each component's contribution via
    `named_forecasters_`.
    """)


@app.cell
def _(decomp, fh, plot_decomposition, y_test):
    components = {}
    for name, fc, *_ in decomp.forecasters_:
        comp_pred = fc.predict(forecasting_horizon=fh)
        components[name] = comp_pred

    plot_decomposition(
        y_test,
        components,
        title="Decomposition Components",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-vintage Scoring

    The `observe_predict` method with `stride=1` produces one forecast per
    observation point, creating multiple *vintages*. Each vintage represents
    a different forecast origin, so you can analyse how accuracy evolves as
    the model absorbs more data.
    """)


@app.cell
def _(deepcopy, fh, trend_fc, y_test):
    _vintage_model = deepcopy(trend_fc)
    y_pred_vintages = _vintage_model.observe_predict(
        y=y_test,
        stride=1,
        forecasting_horizon=fh,
    )
    print(f"Vintages: {y_pred_vintages['vintage_time'].n_unique()}")
    y_pred_vintages.head(10)
    return (y_pred_vintages,)


@app.cell
def _(MeanAbsoluteError, y_train):
    vintage_scorer = MeanAbsoluteError()
    vintage_scorer.fit(y_train)
    return (vintage_scorer,)


@app.cell
def _(vintage_scorer, plot_score_per_step, y_pred_vintages, y_test):
    plot_score_per_step(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="MAE per Forecast Step",
        y_label="MAE",
        height=380,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Stationarity transforms**: See [`stationarity_transforms.py`](/examples/data-features/stationarity_transforms/) for differencing
    - **Reduction forecasting**: See [`point/reduction_forecaster.py`](/examples/getting-started/reduction_forecaster/)
    - **Interval forecasting**: See [Interval](/examples/#interval-forecasting) for prediction intervals
    """)


if __name__ == "__main__":
    app.run()
