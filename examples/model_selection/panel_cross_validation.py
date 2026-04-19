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
    "title": "Panel Cross-Validation",
    "description": "Time series cross-validation on panel data with GridSearchCV, selective group observation, rewind operations, and groupwise performance comparison.",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Panel Cross-Validation

    Time series CV on panel data splits by time while preserving all
    groups. Each split trains and evaluates on all groups simultaneously.

    ## What You'll Learn

    - Running [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) on panel data to find optimal hyperparameters
    - Using `observe` and `predict` with `groups` on the best
      model returned by the search
    - Using `rewind` to reset the forecast origin for specific groups
    - Visualising how observation shifts the forecast origin per group
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_tourism_quarterly
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        ExpandingWindowSplitter,
        GridSearchCV,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        deepcopy,
        fetch_tourism_quarterly,
        pl,
        plot_forecast,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Panel Data

    We load the Tourism Quarterly dataset and select eight panel groups.
    Each column follows the `GROUP__MEMBER` naming convention, which yohou
    uses to identify panel structure automatically.
    """)


@app.cell
def _(fetch_tourism_quarterly, mo):
    tourism = fetch_tourism_quarterly().frame.select("time", *[f"T{i}__tourists" for i in range(3, 11)]).drop_nulls()
    fh = 8
    y_train = tourism.head(len(tourism) - fh)
    y_test = tourism.tail(fh)
    mo.md(
        f"**Shape**: {tourism.shape}\n\n"
        f"**Columns**: {tourism.columns}\n\n"
        f"**Quarterly**: {len(tourism)} observations, last {fh} held out for testing"
    )
    return fh, tourism, y_test, y_train


@app.cell
def _(plot_time_series, tourism):
    plot_time_series(tourism, title="Australian Tourism Quarterly (Panel)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. GridSearchCV on Panel Data

    Search over forecaster hyperparameters using time series CV. The
    scorer evaluates all panel groups together.
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
    _forecaster = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 4]),
    )

    _param_grid = {
        "estimator__alpha": [0.1, 1.0, 10.0],
    }

    _cv = ExpandingWindowSplitter(n_splits=3, test_size=8)

    search = GridSearchCV(
        forecaster=_forecaster,
        param_grid=_param_grid,
        scoring=MeanAbsoluteError(),
        cv=_cv,
        refit=True,
    )
    search.fit(y_train, forecasting_horizon=fh)
    return (search,)


@app.cell
def _(mo, pl, search):
    _results = search.cv_results_
    _table = pl.DataFrame({
        "alpha": [p["estimator__alpha"] for p in _results["params"]],
        "mean_score": _results["mean_test_score"],
        "std_score": _results["std_test_score"],
        "rank": _results["rank_test_score"],
    }).sort("rank")

    mo.vstack([
        mo.md(f"**Best parameters**: `{search.best_params_}`\n\n**Best score**: {search.best_score_:.4f}"),
        mo.ui.table(_table),
    ])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Observe, Predict, and Rewind the Best Model

    After `refit=True`, the search object delegates `predict`, `observe`,
    and `rewind` to its `best_forecaster_`. All three accept
    `groups` for selective group operations.

    We demonstrate three steps on groups **T5** and **T8**:

    1. **Predict from training window**: baseline forecast
    2. **Observe** the first half of test data for those groups, then
       predict again: the forecast origin moves forward
    3. **Rewind** those groups back to the training window and predict
       once more: the origin returns to its original position
    """)


@app.cell
def _(deepcopy, fh, mo, plot_forecast, search, y_test, y_train):
    _groups = ["T5", "T8"]
    _search = deepcopy(search)

    # 1) Predict from training window
    _y_pred_baseline = _search.predict(forecasting_horizon=fh, groups=_groups)
    _fig_baseline = plot_forecast(
        y_test,
        _y_pred_baseline,
        y_train=y_train,
        groups=_groups,
        n_history=16,
        title="Step 1 - Predict from training window",
    )

    # 2) Observe first half of test data for those groups, then predict
    _y_obs = y_test.head(fh // 2)
    _search.observe(_y_obs, groups=_groups)
    _y_pred_observed = _search.predict(forecasting_horizon=fh, groups=_groups)
    _fig_observed = plot_forecast(
        y_test,
        _y_pred_observed,
        y_train=y_train,
        groups=_groups,
        n_history=16,
        title="Step 2 - After observing first half of test (origin moves forward)",
    )

    # 3) Rewind back and predict again
    _search.rewind(y_train, groups=_groups)
    _y_pred_rewound = _search.predict(forecasting_horizon=fh, groups=_groups)
    _fig_rewound = plot_forecast(
        y_test,
        _y_pred_rewound,
        y_train=y_train,
        groups=_groups,
        n_history=16,
        title="Step 3 - After rewind (origin returns)",
    )

    mo.vstack([
        _fig_baseline,
        mo.md("**After `observe`**: the forecast origin shifts forward by 4 quarters for T5 and T8:"),
        _fig_observed,
        mo.md("**After `rewind`**: the forecast origin resets to the end of the training window:"),
        _fig_rewound,
    ])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Panel CV splits by time**: All groups share the same train/test time
      boundaries; [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) evaluates them together
    - **`refit=True`**: The best model is fitted on all training data after
      search, ready for `predict` / `observe` / `rewind`
    - **`observe` with `groups`**: Feed new data to specific
      groups, shifting their forecast origin forward
    - **`rewind` with `groups`**: Reset specific groups back to
      a previous observation window without refitting
    - **`predict` with `groups`**: Generate forecasts for a
      subset of groups only

    ## Next Steps

    - **Multi-metric search**: See [`examples/model_selection/multi_metric_search.py`](/examples/model_selection/multi_metric_search/)
    - **Interval search**: See [`examples/model_selection/interval_search.py`](/examples/model_selection/interval_search/)
    - **CV splitters**: See [`examples/model_selection/cv_splitters.py`](/examples/model_selection/cv_splitters/)
    - **Panel forecasting**: See [`examples/point/panel_forecasting.py`](/examples/point/panel_forecasting/)
    """)


if __name__ == "__main__":
    app.run()
