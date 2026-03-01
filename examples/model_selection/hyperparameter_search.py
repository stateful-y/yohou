# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.19.11"
__gallery__ = {
    "title": "Hyperparameter Search",
    "description": "Tune forecaster hyperparameters with GridSearchCV and RandomizedSearchCV using temporal cross-validation splitters and result scatter visualisation.",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hyperparameter Search for Time Series

    Yohou's [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) and [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/) combine sklearn's hyperparameter
    search with time-respecting cross-validation. They work with any forecaster,
    scorer, and splitter.

    ## What You'll Learn

    - [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) with exhaustive grid search
    - [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/) for larger search spaces
    - Using time series splitters with search
    - Inspecting `cv_results_` and `best_params_`
    - Visualizing results with [`plot_cv_results_scatter`](/pages/api/generated/yohou.plotting.model_selection.plot_cv_results_scatter/)

    ## Prerequisites

    Familiarity with splitters (see `cv_splitters.py`) and scorers (see `metrics/`).
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from scipy.stats import uniform
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import (
        ExpandingWindowSplitter,
        GridSearchCV,
        RandomizedSearchCV,
    )
    from yohou.plotting import plot_cv_results_scatter, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        ExpandingWindowSplitter,
        GridSearchCV,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        RandomizedSearchCV,
        Ridge,
        fetch_tourism_monthly,
        pl,
        plot_cv_results_scatter,
        plot_time_series,
        uniform,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Setup

    We load the data and define the forecaster, parameter grid, and cross-validation splitter used throughout the search examples.
    """)


@app.cell
def _(fetch_tourism_monthly):
    y = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})

    y_train = y.head(120)
    y_test = y.tail(24)
    fh = 12

    print(f"Train: {len(y_train)}, Test: {len(y_test)}, Horizon: {fh}")
    return fh, y, y_test, y_train


@app.cell
def _(plot_time_series, y):
    plot_time_series(y, title="Tourism Monthly")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. GridSearchCV

    Exhaustive grid search over parameter combinations.
    Uses [`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.ExpandingWindowSplitter/) for temporal CV and [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.point.MeanAbsoluteError/) for scoring.
    """)


@app.cell
def _(
    ExpandingWindowSplitter,
    GridSearchCV,
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    fh,
    y_train,
):
    base_fc = PointReductionForecaster(
        estimator=Ridge(),
        feature_transformer=LagTransformer(lag=list(range(1, 7))),
    )

    grid_search = GridSearchCV(
        forecaster=base_fc,
        param_grid={
            "estimator__alpha": [0.01, 0.1, 1.0, 10.0],
            "feature_transformer__lag": [
                list(range(1, 7)),
                list(range(1, 13)),
            ],
        },
        scoring=MeanAbsoluteError(),
        cv=ExpandingWindowSplitter(n_splits=2, test_size=fh),
    )

    grid_search.fit(y_train, forecasting_horizon=fh)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best score:  {grid_search.best_score_:.2f}")
    return (grid_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Inspecting Results

    `cv_results_` is a dict with per-parameter-combination scores, similar to sklearn.
    """)


@app.cell
def _(grid_search, pl):
    cv_results = grid_search.cv_results_
    results_df = pl.DataFrame({
        "params": [str(p) for p in cv_results["params"]],
        "mean_test_score": cv_results["mean_test_score"],
        "rank_test_score": cv_results["rank_test_score"],
    }).sort("rank_test_score")

    results_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_cv_results_scatter`](/pages/api/generated/yohou.plotting.model_selection.plot_cv_results_scatter/) plots each parameter combination against its
    mean test score, showing how performance varies across the grid.
    """)


@app.cell
def _(grid_search, plot_cv_results_scatter):
    plot_cv_results_scatter(
        grid_search.cv_results_,
        param_name="estimator__alpha",
        title="Grid Search: Alpha vs Score",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Predict with Best Model

    After fitting, [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) acts as a forecaster using the best-found parameters.
    """)


@app.cell
def _(MeanAbsoluteError, grid_search, y_test, y_train):
    y_pred = grid_search.predict(forecasting_horizon=len(y_test))

    mae = MeanAbsoluteError()
    mae.fit(y_train)
    score = mae.score(y_test, y_pred)
    print(f"Best model MAE on test: {score:.2f}")
    y_pred.head()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. RandomizedSearchCV

    For larger search spaces, [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/) samples parameter combinations
    randomly. Use `n_iter` to control how many to try.
    """)


@app.cell
def _(
    ExpandingWindowSplitter,
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    RandomizedSearchCV,
    Ridge,
    fh,
    uniform,
    y_train,
):
    rand_search = RandomizedSearchCV(
        forecaster=PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(lag=list(range(1, 7))),
        ),
        param_distributions={
            "estimator__alpha": uniform(loc=0.01, scale=20),
        },
        n_iter=8,
        scoring=MeanAbsoluteError(),
        cv=ExpandingWindowSplitter(n_splits=2, test_size=fh),
        random_state=42,
    )

    rand_search.fit(y_train, forecasting_horizon=fh)
    print(f"Best params: {rand_search.best_params_}")
    print(f"Best score:  {rand_search.best_score_:.2f}")
    return (rand_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_cv_results_scatter`](/pages/api/generated/yohou.plotting.model_selection.plot_cv_results_scatter/) visualises the randomly sampled alpha values
    against their mean test scores. The more dispersed distribution compared
    to the grid search illustrates how randomised search explores the space.
    """)


@app.cell
def _(plot_cv_results_scatter, rand_search):
    plot_cv_results_scatter(
        rand_search.cv_results_,
        param_name="estimator__alpha",
        title="Randomized Search: Alpha vs Score",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/): Exhaustive search: best for small parameter grids
    - [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/): Random sampling: better for large/continuous spaces
    - Both use time-respecting splitters (never leak future data)
    - `scoring` takes scorer **instances** (e.g., `MeanAbsoluteError()`)
    - Access results via `best_params_`, `best_score_`, `cv_results_`
    - Use [`plot_cv_results_scatter`](/pages/api/generated/yohou.plotting.model_selection.plot_cv_results_scatter/) to visualize parameter-score relationships
    - After fitting, use the search object directly as a forecaster
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Splitters**: See [`cv_splitters.py`](/examples/model_selection/cv_splitters/) for splitter details
    - **Scoring**: See [Metrics](/examples/#metrics) for all available scorers
    - **Interval search**: Use [`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/) with search for interval tuning
    """)


if __name__ == "__main__":
    app.run()
