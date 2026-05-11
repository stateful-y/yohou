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
    "category": "how-to",
    "companion": "/pages/explanation/model-selection/",
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

    Yohou's [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) and
    [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/)
    combine sklearn's hyperparameter search with time-respecting cross-validation.
    They work with any forecaster, scorer, and splitter.

    ## Prerequisites

    Familiarity with splitters (see `cv_splitters.py`) and scorers (see `metrics/`).
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    import polars as pl
    from scipy.stats import uniform
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    from yohou.class_proba import ClassProbaReductionForecaster
    from yohou.datasets import fetch_air_quality_classification, fetch_tourism_monthly
    from yohou.metrics import LogLoss, MeanAbsoluteError
    from yohou.model_selection import (
        ExpandingWindowSplitter,
        GridSearchCV,
        RandomizedSearchCV,
    )
    from yohou.plotting import (
        plot_cv_results_scatter,
        plot_forecast,
        plot_score_per_step,
        plot_time_series,
    )
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        ClassProbaReductionForecaster,
        DecisionTreeClassifier,
        ExpandingWindowSplitter,
        GridSearchCV,
        LagTransformer,
        LogLoss,
        MeanAbsoluteError,
        PointReductionForecaster,
        RandomizedSearchCV,
        Ridge,
        deepcopy,
        fetch_air_quality_classification,
        fetch_tourism_monthly,
        pl,
        plot_cv_results_scatter,
        plot_forecast,
        plot_score_per_step,
        plot_time_series,
        train_test_split,
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
    ## 6. GridSearchCV for Classification

    [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) works identically with class-probability forecasters.
    Pass a classification scorer such as [`LogLoss`](/pages/api/generated/yohou.metrics.class_proba.LogLoss/) and search over the
    classifier and feature transformer hyperparameters. We use the
    [`fetch_air_quality_classification`](/pages/api/generated/yohou.datasets._fetchers.fetch_air_quality_classification/)
    dataset (4-class air quality target, 5 pollutant features).
    """)


@app.cell
def _(fetch_air_quality_classification, train_test_split):
    cls_data = fetch_air_quality_classification()
    cls_y, cls_X = cls_data.y, cls_data.X_actual
    cls_y_train, cls_y_test, cls_X_train, cls_X_test = train_test_split(
        cls_y,
        cls_X,
        test_size=200,
        shuffle=False,
    )
    cls_fh = 24

    print(f"Classes: {cls_data.classes}")
    print(f"Train: {len(cls_y_train)}, Test: {len(cls_y_test)}, Horizon: {cls_fh}")
    return cls_X_test, cls_X_train, cls_fh, cls_y_test, cls_y_train


@app.cell
def _(
    ClassProbaReductionForecaster,
    DecisionTreeClassifier,
    ExpandingWindowSplitter,
    GridSearchCV,
    LagTransformer,
    LogLoss,
    cls_X_train,
    cls_fh,
    cls_y_train,
):
    cls_base = ClassProbaReductionForecaster(
        estimator=DecisionTreeClassifier(random_state=42),
        feature_transformer=LagTransformer(lag=[1, 2, 3]),
    )

    cls_grid_search = GridSearchCV(
        forecaster=cls_base,
        param_grid={
            "estimator__max_depth": [3, 5, 10, None],
            "feature_transformer__lag": [
                [1, 2, 3],
                [1, 2, 3, 6, 12, 24],
            ],
        },
        scoring=LogLoss(),
        cv=ExpandingWindowSplitter(n_splits=2, test_size=cls_fh),
    )

    cls_grid_search.fit(cls_y_train, X_actual=cls_X_train, forecasting_horizon=cls_fh)

    print(f"Best params: {cls_grid_search.best_params_}")
    print(f"Best LogLoss: {cls_grid_search.best_score_:.4f}")
    return (cls_grid_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Inspecting Classification Search Results

    The `cv_results_` dict contains per-combination scores just like the
    point forecaster search. [`plot_cv_results_scatter`](/pages/api/generated/yohou.plotting.model_selection.plot_cv_results_scatter/) works with any
    parameter name from the grid.
    """)


@app.cell
def _(cls_grid_search, pl):
    cls_cv_results = cls_grid_search.cv_results_
    cls_results_df = pl.DataFrame({
        "params": [str(p) for p in cls_cv_results["params"]],
        "mean_test_score": cls_cv_results["mean_test_score"],
        "rank_test_score": cls_cv_results["rank_test_score"],
    }).sort("rank_test_score")

    cls_results_df


@app.cell
def _(cls_grid_search, plot_cv_results_scatter):
    plot_cv_results_scatter(
        cls_grid_search.cv_results_,
        param_name="estimator__max_depth",
        title="Classification Grid Search: max_depth vs LogLoss",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Predict with Best Classification Model

    After fitting, the search object exposes both `predict()` (hard labels)
    and `predict_class_proba()` (soft probabilities). [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/)
    auto-detects the prediction type and renders step charts for categorical
    data and stacked area charts for class probabilities.
    """)


@app.cell
def _(cls_X_test, cls_fh, cls_grid_search, cls_y_test, cls_y_train, plot_forecast):
    cls_y_pred_labels = cls_grid_search.predict(
        X_future=cls_X_test[:cls_fh],
        forecasting_horizon=cls_fh,
    )
    plot_forecast(
        cls_y_test,
        cls_y_pred_labels,
        y_train=cls_y_train,
        n_history=50,
        title="Best Model - Hard-Label Forecast",
    )


@app.cell
def _(cls_X_test, cls_fh, cls_grid_search, cls_y_test, plot_forecast):
    cls_y_proba = cls_grid_search.predict_class_proba(
        X_future=cls_X_test[:cls_fh],
        forecasting_horizon=cls_fh,
    )
    plot_forecast(
        cls_y_test,
        cls_y_proba,
        title="Best Model - Class-Probability Forecast",
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
def _(deepcopy, fh, grid_search, y, y_train):
    _vintage_model = deepcopy(grid_search.best_forecaster_)
    y_after_train = y.slice(len(y_train))
    y_pred_vintages = _vintage_model.observe_predict(
        y=y_after_train,
        forecasting_horizon=fh,
    )
    print(f"Vintages: {y_pred_vintages['vintage_time'].n_unique()}")
    y_pred_vintages.head(10)
    return (y_after_train, y_pred_vintages)


@app.cell
def _(MeanAbsoluteError, y_train):
    vintage_scorer = MeanAbsoluteError()
    vintage_scorer.fit(y_train)
    return (vintage_scorer,)


@app.cell
def _(vintage_scorer, plot_score_per_step, y_after_train, y_pred_vintages):
    plot_score_per_step(
        vintage_scorer,
        y_after_train,
        y_pred_vintages,
        title="MAE per Forecast Step",
        y_label="MAE",
        height=380,
    )


if __name__ == "__main__":
    app.run()
