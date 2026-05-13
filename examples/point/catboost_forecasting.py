# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "catboost",
#     "scikit-learn",
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "CatBoost Forecasting",
    "description": "Plug CatBoostRegressor into PointReductionForecaster as a drop-in sklearn estimator, compare gradient-boosted versus Ridge linear baseline, and demonstrate the direct reduction strategy with tree-based models.",
    "category": "how-to",
    "section": "forecasting-models",
}

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
    seamlessly as a drop-in sklearn estimator inside [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/).

    ## What You'll Learn

    - Wrapping `CatBoostRegressor` in [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/)
    - Silencing CatBoost training output with `verbose=0`
    - Comparing CatBoost with a linear baseline
    - Using `reduction_strategy="direct"` with tree-based estimators
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    import polars as pl
    from catboost import CatBoostRegressor
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor

    from yohou.datasets import fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import (
        plot_forecast,
        plot_score_per_step,
        plot_score_time_series,
        plot_time_series,
    )
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        CatBoostRegressor,
        LagTransformer,
        MeanAbsoluteError,
        MultiOutputRegressor,
        PointReductionForecaster,
        Ridge,
        deepcopy,
        fetch_tourism_monthly,
        pl,
        plot_forecast,
        plot_score_per_step,
        plot_score_time_series,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We load the Monthly Tourism dataset via [`fetch_tourism_monthly`](/pages/api/generated/yohou.datasets._fetchers.fetch_tourism_monthly/), extract a
    single univariate series, and split it 80/20 into training and test sets
    while preserving temporal order.
    """)


@app.cell
def _(fetch_tourism_monthly, plot_time_series):
    y = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})

    split_idx = int(len(y) * 0.8)
    y_train = y.head(split_idx)
    y_test = y.tail(len(y) - split_idx)
    forecasting_horizon = len(y_test)

    plot_time_series(y, title="Monthly Tourism (T1)")
    return forecasting_horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. CatBoost Forecaster

    `CatBoostRegressor` implements the sklearn `fit`/`predict` API, so it
    plugs directly into [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/).  Set `verbose=0` to
    suppress per-iteration training logs.
    """)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) overlays the predicted values against the test actuals,
    optionally showing the training history for context.
    """)


@app.cell
def _(plot_forecast, y_pred_cb, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_cb,
        y_train=y_train,
        title="CatBoost Point Forecast",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Compare with Linear Baseline

    We fit a [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) backed by `Ridge` regression using the
    same lag features and compare its MAE against CatBoost to quantify the
    benefit of gradient boosting on this dataset.
    """)


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
    return (y_pred_ridge,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. CatBoost with Direct Strategy

    Gradient-boosted trees optimise a scalar loss per boosting round, so
    they are inherently single-output learners. The default `"multi-output"`
    strategy wraps CatBoost in `MultiOutputRegressor`, which clones one
    model per horizon step anyway.

    For a **univariate** target, `reduction_strategy="direct"` removes
    this redundancy: yohou trains **H independent CatBoost models**
    directly, each predicting a single scalar. Every model's splits are
    optimised exclusively for its own horizon step, giving it full
    capacity to capture step-specific patterns.

    With a **multivariate** target (multiple columns in `y`), each
    per-step model still faces a multi-output problem, so a
    `MultiOutputRegressor` wrapper remains necessary.
    """)


@app.cell
def _(
    CatBoostRegressor,
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    forecasting_horizon,
    plot_forecast,
    y_pred_cb,
    y_pred_ridge,
    y_test,
    y_train,
):
    catboost_direct_fc = PointReductionForecaster(
        estimator=CatBoostRegressor(
            iterations=200,
            depth=4,
            learning_rate=0.1,
            verbose=0,
        ),
        reduction_strategy="direct",
        feature_transformer=LagTransformer(lag=list(range(1, 13))),
    )
    catboost_direct_fc.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_cb_direct = catboost_direct_fc.predict(forecasting_horizon=forecasting_horizon)

    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    print(f"CatBoost Direct MAE: {_scorer.score(y_test, y_pred_cb_direct):.2f}")

    plot_forecast(
        y_test,
        {
            "CatBoost (multi-output)": y_pred_cb,
            "CatBoost (direct)": y_pred_cb_direct,
            "Ridge": y_pred_ridge,
        },
        y_train=y_train,
        title="CatBoost Multi-Output vs Direct vs Ridge",
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
def _(catboost_fc, deepcopy, forecasting_horizon, y_test):
    _vintage_model = deepcopy(catboost_fc)
    y_pred_vintages = _vintage_model.observe_predict(
        y=y_test,
        stride=1,
        forecasting_horizon=forecasting_horizon,
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


@app.cell
def _(vintage_scorer, plot_score_time_series, y_pred_vintages, y_test):
    plot_score_time_series(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="MAE over Time",
        y_label="MAE",
        height=380,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - Any sklearn-compatible regressor works with [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/)
    - CatBoost often captures nonlinear patterns better than linear models
    - Tree-based models pair well with `reduction_strategy="direct"` (no `MultiOutputRegressor` wrapper needed)
    - Use `verbose=0` to keep notebook output clean
    - Consider [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) for tuning `iterations`, `depth`, `learning_rate`
    - See [`reduction_strategies.py`](/examples/point/reduction_strategies/) for a full strategy comparison
    """)


if __name__ == "__main__":
    app.run()
