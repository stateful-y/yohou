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
    "title": "Observe-Predict Workflow",
    "description": "Walk through a test set in batches, updating forecasts as new data arrives with observe_predict.",
    "category": "tutorial",
    "section": "getting-started",
    "companion": "/pages/tutorials/observe-predict/",
    "api_references": ["MeanAbsoluteScaledError", "PointReductionForecaster", "plot_forecast"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Observe-Predict Workflow

    In production forecasting you rarely retrain from scratch each time new
    data arrives. Instead you **observe** new data (update memory/state) and
    then **predict** using the already-fitted model. This notebook walks
    through a two-year test set in six-month batches, comparing a
    single-shot prediction against a full walk-forward loop.

    ## Prerequisites

    Familiarity with [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) and `fit/predict`
    (see [Getting Started](/pages/tutorials/getting-started/)).
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import FeaturePipeline
    from yohou.datasets import fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteScaledError
    from yohou.model_selection import train_test_split
    from yohou.plotting import plot_forecast
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou.stationarity import SeasonalDifferencing

    return (
        FeaturePipeline,
        LagTransformer,
        MeanAbsoluteScaledError,
        PointReductionForecaster,
        Ridge,
        SeasonalDifferencing,
        fetch_tourism_monthly,
        pl,
        plot_forecast,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load and Prepare Data

    We use a single tourism series (T1) and hold out the final 24 months
    as the test set with a 6-month forecasting horizon.
    """)


@app.cell
def _(fetch_tourism_monthly, train_test_split):
    bunch = fetch_tourism_monthly()
    y = (
        bunch.frame
        .select("time", "T1__tourists")
        .rename({"T1__tourists": "tourists"})
        .drop_nulls()
    )

    forecasting_horizon = 6
    y_train, y_test = train_test_split(y, test_size=24)

    print(f"Series length: {len(y)} months")
    print(f"Train: {len(y_train)} months, Test: {len(y_test)} months")
    return forecasting_horizon, y, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Fit the Forecaster

    Build a Ridge forecaster with seasonal differencing and lag features.
    """)


@app.cell
def _(
    FeaturePipeline,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SeasonalDifferencing,
    forecasting_horizon,
    y_train,
):
    forecaster = PointReductionForecaster(
        estimator=Ridge(),
        target_transformer=SeasonalDifferencing(seasonality=12),
        actual_transformer=FeaturePipeline([
            ("lags", LagTransformer(lag=[1, 2, 3, 12])),
        ]),
    )
    forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)
    return (forecaster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Single-Shot Predict

    A single-shot forecast produces one batch of predictions from the end
    of the training data, covering the first six months of the test period.
    All rows share the same `vintage_time`.
    """)


@app.cell
def _(forecaster, forecasting_horizon):
    y_pred_single = forecaster.predict(forecasting_horizon=forecasting_horizon)
    print(y_pred_single)
    return (y_pred_single,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Walk-Forward with observe_predict

    `observe_predict` steps through the test set in batches of size `stride`,
    observing actual values and issuing a fresh forecast after each batch.
    Setting `stride=forecasting_horizon` tiles the test set with no gaps
    or overlaps.
    """)


@app.cell
def _(forecaster, forecasting_horizon, y_test):
    y_pred_loop = forecaster.observe_predict(y=y_test, stride=forecasting_horizon)
    print(f"Total predictions: {len(y_pred_loop)}")
    print(f"Distinct vintage_times: {y_pred_loop['vintage_time'].n_unique()}")
    print(y_pred_loop.head(12))
    return (y_pred_loop,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Score and Compare

    Filter predictions to the test period and compare single-shot and
    walk-forward MASE.
    """)


@app.cell
def _(MeanAbsoluteScaledError, forecasting_horizon, pl, y_pred_loop, y_pred_single, y_test, y_train):
    y_pred_in_test = y_pred_loop.filter(pl.col("time") <= y_test["time"][-1])

    scorer = MeanAbsoluteScaledError(seasonality=12)
    scorer.fit(y_train)
    mase_loop = scorer.score(y_test, y_pred_in_test)
    mase_single = scorer.score(y_test[:forecasting_horizon], y_pred_single)

    print(f"Single-shot MASE (months 1 to 6):  {mase_single:.3f}")
    print(f"Walk-forward MASE (all 24 months): {mase_loop:.3f}")
    return (y_pred_in_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Visualize the Predictions

    Plot all walk-forward prediction vintages against the actual test values.
    Each vintage appears as a separate colored segment overlaid on the
    actual test series.
    """)


@app.cell
def _(plot_forecast, y_pred_in_test, y_test, y_train):
    plot_forecast(y_test, y_pred_in_test, y_train=y_train[-24:])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Exogenous features**: See the [Exogenous Features](/pages/tutorials/exogenous-features/) tutorial for passing external regressors through the observe/predict loop
    - **Interval forecasting**: See the [Interval Forecasting](/pages/tutorials/interval-forecasting/) tutorial for prediction intervals with conformal methods
    """)


if __name__ == "__main__":
    app.run()
