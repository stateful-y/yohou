# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "catboost",
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Point Forecasting with CatBoost.

Demonstrates PointReductionForecaster with CatBoostRegressor as the wrapped
sklearn-compatible estimator.
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point Forecasting with CatBoost

    [CatBoost](https://catboost.ai/) is a gradient-boosting library that works
    seamlessly as a drop-in sklearn estimator inside `PointReductionForecaster`.

    ## What You'll Learn

    - Wrapping `CatBoostRegressor` in `PointReductionForecaster`
    - Silencing CatBoost training output with `verbose=0`
    - Comparing CatBoost with a linear baseline
    """)
    return

@app.cell(hide_code=True)
def _():
    import polars as pl
    from catboost import CatBoostRegressor
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor

    from yohou.datasets import fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        CatBoostRegressor,
        LagTransformer,
        MeanAbsoluteError,
        MultiOutputRegressor,
        PointReductionForecaster,
        Ridge,
        fetch_tourism_monthly,
        pl,
        plot_forecast,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data
    """)
    return

@app.cell
def _(fetch_tourism_monthly):
    y = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "passengers"})

    split_idx = int(len(y) * 0.8)
    y_train = y.head(split_idx)
    y_test = y.tail(len(y) - split_idx)
    forecasting_horizon = len(y_test)

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    return forecasting_horizon, y_test, y_train

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. CatBoost Forecaster

    `CatBoostRegressor` implements the sklearn `fit`/`predict` API, so it
    plugs directly into `PointReductionForecaster`.  Set `verbose=0` to
    suppress per-iteration training logs.
    """)
    return

@app.cell
def _(
    CatBoostRegressor,
    LagTransformer,
    MultiOutputRegressor,
    PointReductionForecaster,
    forecasting_horizon,
    y_train,
):
    catboost_fc = PointReductionForecaster(
        estimator=MultiOutputRegressor(
            CatBoostRegressor(
                iterations=200,
                depth=4,
                learning_rate=0.1,
                verbose=0,
            )
        ),
        feature_transformer=LagTransformer(lag=list(range(1, 13))),
    )

    catboost_fc.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_cb = catboost_fc.predict(forecasting_horizon=forecasting_horizon)

    y_pred_cb.head()
    return catboost_fc, y_pred_cb

@app.cell
def _(plot_forecast, y_pred_cb, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_cb,
        y_train=y_train,
        title="CatBoost Point Forecast",
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Compare with Linear Baseline
    """)
    return

@app.cell
def _(
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    forecasting_horizon,
    y_pred_cb,
    y_test,
    y_train,
):
    ridge_fc = PointReductionForecaster(
        estimator=Ridge(),
        feature_transformer=LagTransformer(lag=list(range(1, 13))),
    )
    ridge_fc.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_ridge = ridge_fc.predict(forecasting_horizon=forecasting_horizon)

    scorer = MeanAbsoluteError()
    scorer.fit(y_train)

    mae_cb = scorer.score(y_test, y_pred_cb)
    mae_ridge = scorer.score(y_test, y_pred_ridge)
    print(f"CatBoost MAE: {mae_cb:.2f}")
    print(f"Ridge    MAE: {mae_ridge:.2f}")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - Any sklearn-compatible regressor works with `PointReductionForecaster`
    - CatBoost often captures nonlinear patterns better than linear models
    - Use `verbose=0` to keep notebook output clean
    - Consider `GridSearchCV` for tuning `iterations`, `depth`, `learning_rate`
    """)
    return

if __name__ == "__main__":
    app.run()
