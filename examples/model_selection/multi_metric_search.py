# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Multi-Metric Hyperparameter Search.

Demonstrates multi-metric scoring and refit strategies with GridSearchCV
and RandomizedSearchCV.
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
    # Multi-Metric Hyperparameter Search

    Evaluate model configurations against multiple metrics simultaneously.

    ## What You'll Learn

    - Passing a **dict of scorers** to `GridSearchCV`
    - Understanding `cv_results_` with multiple metrics
    - Refit strategies: select best by MAE, RMSE, or MAPE
    - `RandomizedSearchCV` with distributions for efficient search
    """)
    return

@app.cell(hide_code=True)
def _():
    import polars as pl
    from scipy.stats import loguniform, randint
    from sklearn.linear_model import ElasticNet, Ridge

    from yohou.datasets import fetch_tourism_monthly
    from yohou.metrics import (
        MeanAbsoluteError,
        MeanAbsolutePercentageError,
        RootMeanSquaredError,
    )
    from yohou.model_selection import GridSearchCV, RandomizedSearchCV
    from yohou.model_selection.split import ExpandingWindowSplitter
    from yohou.plotting import plot_cv_results_scatter, plot_forecast
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
        fetch_tourism_monthly,
        loguniform,
        pl,
        plot_cv_results_scatter,
        plot_forecast,
        randint,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data
    """)
    return

@app.cell
def _(fetch_tourism_monthly, mo):
    ap = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "passengers"})
    _split = int(len(ap) * 0.85)
    y_train = ap.head(_split)
    y_test = ap.tail(len(ap) - _split)
    horizon = len(y_test)

    mo.md(f"**Train**: {len(y_train)} months, **Test**: {len(y_test)} months")
    return ap, horizon, y_test, y_train

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
    return

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
        cv=ExpandingWindowSplitter(n_splits=3),
    )
    multi_gs.fit(y_train, forecasting_horizon=horizon)
    return multi_gs, multi_scoring

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Inspect cv_results_

    With multiple scorers, each metric gets its own `mean_test_{name}`,
    `std_test_{name}`, and `rank_test_{name}` columns.
    """)
    return

@app.cell
def _(mo, multi_gs, pl):
    _results = pl.DataFrame(multi_gs.cv_results_)
    _cols = [c for c in _results.columns if "mean_test" in c or "rank_test" in c or "param_" in c]
    mo.ui.table(_results.select(_cols))
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Refit Strategies

    Compare which alpha each metric selects as "best".
    """)
    return

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
            cv=ExpandingWindowSplitter(n_splits=3),
        )
        _gs.fit(y_train, forecasting_horizon=horizon)
        _rows.append({
            "Refit Metric": _metric_name.upper(),
            "Best Alpha": _gs.best_params_["estimator__alpha"],
            "Best Score (negated)": round(float(_gs.best_score_), 4),
        })
    mo.ui.table(pl.DataFrame(_rows))
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. RandomizedSearchCV with Distributions

    For larger parameter spaces, random sampling is more efficient.
    """)
    return

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
        cv=ExpandingWindowSplitter(n_splits=3),
    )
    rand_search.fit(y_train, forecasting_horizon=horizon)

    mo.md(
        f"**Best params**: {rand_search.best_params_}\n\n"
        f"**Best MAE (negated)**: {rand_search.best_score_:.4f}"
    )
    return (rand_search,)

@app.cell
def _(plot_cv_results_scatter, rand_search):
    plot_cv_results_scatter(rand_search.cv_results_, "estimator__alpha")
    return

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
    return (y_pred_best,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - Pass `scoring={"name": BaseScorer()}` for multi-metric evaluation
    - `refit` must be a scorer key when using dict scoring (not `True`)
    - Scores are **negated** for "lower is better" metrics
    - `cv_results_` contains per-metric mean/std/rank columns
    - Different metrics may select different best parameters
    - `RandomizedSearchCV` efficiently samples continuous distributions

    ## Next Steps

    - **Interval search**: See `examples/model_selection/interval_search.py`
    - **Optuna search**: See `examples/model_selection/optuna_search.py`
    - **Panel CV**: See `examples/model_selection/panel_cross_validation.py`
    """)
    return

if __name__ == "__main__":
    app.run()
