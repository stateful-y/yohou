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
    "title": "How to Score Multi-Vintage Forecasts",
    "description": "Generate multi-vintage predictions with observe_predict, score per step and per vintage, and visualize with heatmap, per-step, and per-vintage plots.",
    "category": "how-to",
    "companion": "/pages/how-to/multi-vintage-scoring/",
    "section": "evaluation-search",
    "api_references": ["PointReductionForecaster", "SeasonalDifferencing", "SeasonalNaive", "plot_score_heatmap", "plot_score_per_step", "plot_score_per_vintage"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multi-vintage Scoring

    Score forecasts across multiple observation origins (vintages) to assess
    how accuracy varies over time and across forecast horizon steps.
    Use [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.point.MeanAbsoluteError/)
    and [`RootMeanSquaredError`](/pages/api/generated/yohou.metrics.point.RootMeanSquaredError/)
    with the `"vintagewise"` and `"stepwise"` aggregation modes.
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    from yohou.datasets import fetch_sunspot
    from yohou.metrics import MeanAbsoluteError, RootMeanSquaredError
    from yohou.plotting import (
        plot_score_heatmap,
        plot_score_per_step,
        plot_score_per_vintage,
    )
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer
    from yohou.stationarity import SeasonalDifferencing

    return (
        HistGradientBoostingRegressor,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        RootMeanSquaredError,
        SeasonalDifferencing,
        SeasonalNaive,
        deepcopy,
        fetch_sunspot,
        plot_score_heatmap,
        plot_score_per_step,
        plot_score_per_vintage,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data
    """)


@app.cell
def _(fetch_sunspot, train_test_split):
    y = fetch_sunspot().frame
    forecasting_horizon = 12
    y_train, y_test = train_test_split(y, test_size=60, shuffle=False)

    print(f"Train: {len(y_train)} rows, Test: {len(y_test)} rows")
    return forecasting_horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Fit Two Models
    """)


@app.cell
def _(
    HistGradientBoostingRegressor,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SeasonalDifferencing,
    SeasonalNaive,
    forecasting_horizon,
    y_train,
):
    naive = SeasonalNaive(seasonality=132)
    naive.fit(y_train, forecasting_horizon=forecasting_horizon)

    ridge = PointReductionForecaster(
        estimator=Ridge(),
        target_transformer=SeasonalDifferencing(seasonality=132),
        feature_transformer=LagTransformer(lag=[1, 2, 3, 6, 12]),
    )
    ridge.fit(y_train, forecasting_horizon=forecasting_horizon)

    print("Both models fitted.")
    return naive, ridge


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Generate Multi-vintage Predictions

    Use `observe_predict` with `stride=1` to produce one vintage per test row.
    Always `deepcopy` the forecaster before calling `observe_predict` because
    the method mutates internal state.
    """)


@app.cell
def _(deepcopy, forecasting_horizon, naive, ridge, y_test):
    y_pred_naive = deepcopy(naive).observe_predict(
        y_test, forecasting_horizon=forecasting_horizon, stride=1
    )
    y_pred_ridge = deepcopy(ridge).observe_predict(
        y_test, forecasting_horizon=forecasting_horizon, stride=1
    )

    print(f"Naive vintages: {y_pred_naive['vintage_time'].n_unique()}")
    print(f"Ridge vintages: {y_pred_ridge['vintage_time'].n_unique()}")
    return y_pred_naive, y_pred_ridge


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Score Per Horizon Step

    [`plot_score_per_step`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_step/) shows how accuracy degrades across the forecast window.
    """)


@app.cell
def _(
    MeanAbsoluteError,
    plot_score_per_step,
    y_pred_naive,
    y_pred_ridge,
    y_test,
    y_train,
):
    mae = MeanAbsoluteError()
    mae.fit(y_train)

    plot_score_per_step(
        mae,
        y_test,
        {"SeasonalNaive": y_pred_naive, "Ridge": y_pred_ridge},
        title="MAE by Forecast Step",
    )
    return (mae,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Score Per Vintage

    [`plot_score_per_vintage`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_vintage/) tracks accuracy over successive forecast origins.
    An upward trend signals model degradation.
    """)


@app.cell
def _(mae, plot_score_per_vintage, y_pred_ridge, y_test):
    plot_score_per_vintage(
        mae,
        y_test,
        y_pred_ridge,
        show_trend=True,
        title="Ridge MAE by Vintage",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Step x Vintage Heatmap

    [`plot_score_heatmap`](/pages/api/generated/yohou.plotting.evaluation.plot_score_heatmap/) combines both dimensions into a single view.
    Each cell is the error for a specific step at a specific vintage.
    """)


@app.cell
def _(mae, plot_score_heatmap, y_pred_ridge, y_test):
    plot_score_heatmap(
        mae,
        y_test,
        y_pred_ridge,
        title="Ridge MAE: Step x Vintage",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Compare Multiple Metrics
    """)


@app.cell
def _(
    MeanAbsoluteError,
    RootMeanSquaredError,
    plot_score_per_step,
    y_pred_naive,
    y_pred_ridge,
    y_test,
    y_train,
):
    scorers = {
        "MAE": MeanAbsoluteError(),
        "RMSE": RootMeanSquaredError(),
    }
    for _s in scorers.values():
        _s.fit(y_train)

    plot_score_per_step(
        scorers,
        y_test,
        {"SeasonalNaive": y_pred_naive, "Ridge": y_pred_ridge},
        title="MAE & RMSE by Forecast Step",
    )



@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Score Multi-Vintage Forecasts](/pages/how-to/multi-vintage-scoring/) for the full guide
        - [Evaluate Forecast Accuracy](/pages/how-to/evaluate-forecast-accuracy/) for related techniques
        """
    )
    return

if __name__ == "__main__":
    app.run()
