# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Multi-Metric Search",
    "description": "Evaluate hyperparameter configurations against multiple metrics simultaneously with dict-of-scorers, refit strategies, and Pareto-optimal selection.",
    "category": "how-to",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multi-Metric Hyperparameter Search

    Evaluate model configurations against multiple metrics simultaneously.

    ## What You'll Learn

    - Passing a **dict of scorers** to [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/)
    - Understanding `cv_results_` with multiple metrics
    - Refit strategies: select best by MAE, RMSE, or MAPE
    - [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/) with distributions for efficient search
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from scipy.stats import loguniform
    from sklearn.linear_model import ElasticNet, Ridge

    from copy import deepcopy

    from yohou.datasets import fetch_tourism_monthly
    from yohou.metrics import (
        MeanAbsoluteError,
        MeanAbsolutePercentageError,
        RootMeanSquaredError,
    )
    from yohou.model_selection import GridSearchCV, RandomizedSearchCV
    from yohou.model_selection.split import ExpandingWindowSplitter
    from yohou.plotting import (
        plot_cv_results_scatter,
        plot_forecast,
        plot_score_summary,
        plot_time_series,
    )
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        ElasticNet,
        ExpandingWindowSplitter,
        GridSearchCV,
        LagTransformer,
        MeanAbsoluteError,
        MeanAbsolutePercentageError,
        PointReductionForecaster,
        RandomizedSearchCV,
        Ridge,
        RootMeanSquaredError,
        deepcopy,
        fetch_tourism_monthly,
        loguniform,
        pl,
        plot_cv_results_scatter,
        plot_forecast,
        plot_score_summary,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We load the Tourism Monthly dataset and split it into training and test
    partitions. The test length defines the forecasting horizon for all
    search experiments below.
    """)


@app.cell
def _(fetch_tourism_monthly, mo):
    tourism = (
        fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    )
    _split = int(len(tourism) * 0.85)
    y_train = tourism.head(_split)
    y_test = tourism.tail(len(tourism) - _split)
    horizon = len(y_test)

    mo.md(f"**Train**: {len(y_train)} months, **Test**: {len(y_test)} months")
    return horizon, tourism, y_test, y_train


@app.cell
def _(plot_time_series, tourism):
    plot_time_series(tourism, title="Tourism Monthly")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Multi-Metric GridSearchCV

    Pass a dict of `BaseScorer` instances. When using a dict, `refit`
    must be a key name (string) or `False`, using `refit=True` raises
    an error.

    Scores for "lower is better" metrics are **negated** internally
    (sklearn convention: higher = better).
    """)


@app.cell
def _(
    ExpandingWindowSplitter,
    GridSearchCV,
    LagTransformer,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    PointReductionForecaster,
    Ridge,
    RootMeanSquaredError,
    horizon,
    y_train,
):
    _base = PointReductionForecaster(
        estimator=Ridge(),
        feature_transformer=LagTransformer(lag=[1, 12]),
    )
    multi_scoring = {
        "mae": MeanAbsoluteError(),
        "rmse": RootMeanSquaredError(),
        "mape": MeanAbsolutePercentageError(),
    }
    multi_gs = GridSearchCV(
        forecaster=_base,
        param_grid={
            "estimator__alpha": [0.01, 0.1, 1.0, 10.0],
        },
        scoring=multi_scoring,
        refit="mae",  # Best model selected by MAE
        cv=ExpandingWindowSplitter(n_splits=2),
    )
    multi_gs.fit(y_train, forecasting_horizon=horizon)
    return (multi_gs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Inspect cv_results_

    With multiple scorers, each metric gets its own `mean_test_{name}`,
    `std_test_{name}`, and `rank_test_{name}` columns.
    """)


@app.cell
def _(mo, multi_gs, pl):
    _results = pl.DataFrame(multi_gs.cv_results_)
    _cols = [c for c in _results.columns if "mean_test" in c or "rank_test" in c or "param_" in c]
    mo.ui.table(_results.select(_cols))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Refit Strategies

    Compare which alpha each metric selects as "best".
    """)


@app.cell
def _(
    ExpandingWindowSplitter,
    GridSearchCV,
    LagTransformer,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    PointReductionForecaster,
    Ridge,
    RootMeanSquaredError,
    horizon,
    mo,
    pl,
    y_train,
):
    _rows = []
    for _metric_name in ["mae", "rmse", "mape"]:
        _gs = GridSearchCV(
            forecaster=PointReductionForecaster(
                estimator=Ridge(),
                feature_transformer=LagTransformer(lag=[1, 12]),
            ),
            param_grid={"estimator__alpha": [0.01, 0.1, 1.0, 10.0]},
            scoring={
                "mae": MeanAbsoluteError(),
                "rmse": RootMeanSquaredError(),
                "mape": MeanAbsolutePercentageError(),
            },
            refit=_metric_name,
            cv=ExpandingWindowSplitter(n_splits=2),
        )
        _gs.fit(y_train, forecasting_horizon=horizon)
        _rows.append({
            "Refit Metric": _metric_name.upper(),
            "Best Alpha": _gs.best_params_["estimator__alpha"],
            "Best Score": -round(float(_gs.best_score_), 4),
        })
    mo.ui.table(pl.DataFrame(_rows))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. RandomizedSearchCV with Distributions

    For larger parameter spaces, random sampling is more efficient.
    """)


@app.cell
def _(
    ElasticNet,
    ExpandingWindowSplitter,
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    RandomizedSearchCV,
    RootMeanSquaredError,
    horizon,
    loguniform,
    mo,
    y_train,
):
    rand_search = RandomizedSearchCV(
        forecaster=PointReductionForecaster(
            estimator=ElasticNet(),
            feature_transformer=LagTransformer(lag=[1, 12]),
        ),
        param_distributions={
            "estimator__alpha": loguniform(0.001, 10.0),
            "estimator__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        scoring={
            "mae": MeanAbsoluteError(),
            "rmse": RootMeanSquaredError(),
        },
        refit="mae",
        n_iter=10,
        cv=ExpandingWindowSplitter(n_splits=2),
    )
    rand_search.fit(y_train, forecasting_horizon=horizon)

    mo.md(f"**Best params**: {rand_search.best_params_}\n\n**Best MAE (negated)**: {rand_search.best_score_:.4f}")
    return (rand_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_cv_results_scatter`](/pages/api/generated/yohou.plotting.model_selection.plot_cv_results_scatter/) shows how each randomly sampled alpha value
    performed according to the refit metric. This helps identify whether
    the search has explored enough of the parameter space.
    """)


@app.cell
def _(plot_cv_results_scatter, rand_search):
    plot_cv_results_scatter(rand_search.cv_results_, "estimator__alpha")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) displays the best model's predictions against the actual
    test values, along with a trailing window of training history for context.
    """)


@app.cell
def _(horizon, plot_forecast, rand_search, y_test, y_train):
    y_pred_best = rand_search.predict(forecasting_horizon=horizon)
    plot_forecast(
        y_test,
        y_pred_best,
        y_train=y_train,
        n_history=36,
        title="Best Forecaster (RandomizedSearchCV, multi-metric)",
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
    return


@app.cell
def _(deepcopy, horizon, multi_gs, y_test):
    _vintage_model = deepcopy(multi_gs.best_forecaster_)
    y_pred_vintages = _vintage_model.observe_predict(
        y=y_test,
        stride=1,
        forecasting_horizon=horizon,
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
def _(vintage_scorer, plot_score_summary, y_pred_vintages, y_test):
    plot_score_summary(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="Model Score Summary",
    )
    return
