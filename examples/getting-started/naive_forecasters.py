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
    "title": "Naive Forecasters",
    "description": "Baseline forecasting (the first portion of the First Forecast tutorial) with SeasonalNaive using different seasonality periods, the observe/predict streaming workflow, and rolling evaluation patterns.",
    "category": "tutorial",
    "section": "getting-started",
    "companion": "/pages/tutorials/getting-started/",
    "api_references": ["MeanSeasonalNaive", "SeasonalNaive", "plot_score_heatmap", "plot_score_per_step", "plot_score_time_series"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Naive Forecasters as Baselines

    Every forecasting project needs a **baseline** to judge model quality.
    Naive methods repeat past patterns with no learning, yet they are surprisingly
    hard to beat on many datasets.

    ## Prerequisites

    Basic understanding of seasonality in time series.
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    from yohou.datasets import fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import train_test_split
    from yohou.plotting import (
        plot_forecast,
        plot_score_heatmap,
        plot_score_per_step,
        plot_score_time_series,
        plot_time_series,
    )
    from yohou.point import SeasonalNaive

    return (
        MeanAbsoluteError,
        SeasonalNaive,
        deepcopy,
        fetch_tourism_monthly,
        plot_forecast,
        plot_score_heatmap,
        plot_score_per_step,
        plot_score_time_series,
        plot_time_series,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load and Split Data

    We begin by loading the Monthly Tourism dataset and splitting it into training and test sets for evaluating the forecasters.
    """)


@app.cell
def _(fetch_tourism_monthly, train_test_split):
    y = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    y_train, y_test = train_test_split(y, test_size=36)
    forecasting_horizon = 12

    print(f"Train: {len(y_train)} obs, Test: {len(y_test)} obs")
    return forecasting_horizon, y, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_time_series`](/pages/api/generated/yohou.plotting.exploration.plot_time_series/) renders the full dataset to reveal the yearly seasonal
    pattern and upward trend that the naive forecaster will attempt to capture.
    """)


@app.cell
def _(plot_time_series, y):
    plot_time_series(y, title="Monthly Tourism")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. SeasonalNaive Forecaster

    [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/) repeats the last `seasonality` observations cyclically.
    With `seasonality=12` on monthly data, it predicts each month
    using the same month from the previous year.
    """)


@app.cell
def _(SeasonalNaive, forecasting_horizon, y_train):
    naive = SeasonalNaive(seasonality=12)
    naive.fit(y_train, forecasting_horizon=forecasting_horizon)

    y_pred_naive = naive.predict(forecasting_horizon=forecasting_horizon)
    print(f"Predicted {len(y_pred_naive)} steps ahead")
    y_pred_naive.head()
    return naive, y_pred_naive


@app.cell
def _(MeanAbsoluteError, plot_forecast, y_pred_naive, y_test, y_train):
    mae = MeanAbsoluteError()
    y_test_h = y_test.head(len(y_pred_naive))
    mae.fit(y_train)
    score = mae.score(y_test_h, y_pred_naive)
    print(f"Seasonal Naive MAE: {score:.2f}")

    plot_forecast(
        y_test_h,
        y_pred_naive,
        y_train=y_train,
        title="Seasonal Naive Forecast (seasonality=12)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Comparing Seasonality Periods

    The `seasonality` parameter controls how far back the forecaster looks.
    Let's compare `seasonality=1` (repeat last value), `6`, and `12`.
    """)


@app.cell
def _(MeanAbsoluteError, SeasonalNaive, forecasting_horizon, y_test, y_train):
    results = {}
    for period in [1, 6, 12]:
        model = SeasonalNaive(seasonality=period)
        model.fit(y_train, forecasting_horizon=forecasting_horizon)
        _pred = model.predict(forecasting_horizon=forecasting_horizon)

        scorer = MeanAbsoluteError()
        y_t = y_test.head(len(_pred))
        scorer.fit(y_train)
        results[f"seasonality={period}"] = {
            "mae": scorer.score(y_t, _pred),
            "pred": _pred,
        }
        print(f"seasonality={period:>2d}  MAE: {results[f'seasonality={period}']['mae']:.2f}")
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) overlays predictions from all three seasonality settings
    against the test actuals, making it easy to see which period best captures
    the seasonal pattern.
    """)


@app.cell
def _(plot_forecast, results, y_test, y_train):
    preds_dict = {name: r["pred"] for name, r in results.items()}
    plot_forecast(
        y_test.head(12),
        preds_dict,
        y_train=y_train,
        title="Comparing Seasonality Periods",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. predict vs observe_predict

    `predict` generates forecasts from the last fitted state.
    `observe_predict` first ingests new observations, then predicts,
    giving the model fresh data to work with. Comparing both on the
    same test set reveals the benefit of incremental observation.
    """)


@app.cell
def _(
    MeanAbsoluteError,
    SeasonalNaive,
    plot_score_time_series,
    y_test,
    y_train,
):
    fc = SeasonalNaive(seasonality=12)
    fc.fit(y_train, forecasting_horizon=12)

    # predict: forecast from the last fitted state
    _y_pred = fc.predict(forecasting_horizon=len(y_test))

    # observe_predict: observe test data first, then predict
    _y_obs_pred = fc.observe_predict(y_test, forecasting_horizon=12)

    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    plot_score_time_series(
        _scorer,
        y_test,
        {"predict": _y_pred, "observe_predict": _y_obs_pred},
        title="MAE Over Time: predict vs observe_predict",
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
def _(deepcopy, forecasting_horizon, naive, y_test):
    _vintage_model = deepcopy(naive)
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
def _(vintage_scorer, plot_score_heatmap, y_pred_vintages, y_test):
    plot_score_heatmap(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="Score Heatmap (Step x Vintage)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Reduction forecasting**: See [`reduction_forecaster.py`](/examples/getting-started/reduction_forecaster/) for ML-based approach
    - **Scoring**: See [Metrics](/examples/#metrics) for comprehensive evaluation metrics
    - **Cross-validation**: See [Model Selection](/examples/#model-selection) for temporal CV strategies
    """)


if __name__ == "__main__":
    app.run()
