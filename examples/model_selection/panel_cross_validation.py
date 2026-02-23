"""Panel Cross-Validation.

Demonstrates time series cross-validation on panel data with
ExpandingWindowSplitter, SlidingWindowSplitter, and GridSearchCV.
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
async def _():
    import sys as _sys

    if "pyodide" in _sys.modules:
        import micropip

        await micropip.install(["plotly", "scikit-learn", "yohou"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Panel Cross-Validation

    Time series CV on panel data splits by time while preserving all
    groups. Each split trains and evaluates on all groups simultaneously.

    ## What You'll Learn

    - `ExpandingWindowSplitter` and `SlidingWindowSplitter` on panel data
    - Visualising panel CV splits with `plot_splits`
    - `GridSearchCV` with panel data: hyperparameter search across groups
    - Per-group results analysis from grid search
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_tourism_quarterly
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV, SlidingWindowSplitter
    from yohou.plotting import plot_splits
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        ExpandingWindowSplitter,
        GridSearchCV,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        SlidingWindowSplitter,
        fetch_tourism_quarterly,
        pl,
        plot_splits,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Panel Data
    """)
    return


@app.cell
def _(fetch_tourism_quarterly, mo):
    tourism = fetch_tourism_quarterly().frame.select("time", *[f"T{i}__tourists" for i in range(1, 9)])
    mo.md(
        f"**Shape**: {tourism.shape}\n\n"
        f"**Columns**: {tourism.columns}\n\n"
        f"**Quarterly**: {len(tourism)} observations"
    )
    return (tourism,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Expanding Window Splits

    Each fold trains on an expanding window and tests on a fixed-size
    window. All panel groups share the same time splits.
    """)
    return


@app.cell
def _(ExpandingWindowSplitter, tourism):
    ew_splitter = ExpandingWindowSplitter(n_splits=3, test_size=8, gap=0)
    ew_splitter
    return (ew_splitter,)


@app.cell
def _(ew_splitter, plot_splits, tourism):
    plot_splits(tourism, ew_splitter, title="Expanding Window: 3 Splits")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Sliding Window Splits

    Fixed-size train and test windows that slide forward.
    """)
    return


@app.cell
def _(SlidingWindowSplitter, plot_splits, tourism):
    sw_splitter = SlidingWindowSplitter(n_splits=3, test_size=8, stride=8, gap=0)
    plot_splits(tourism, sw_splitter, title="Sliding Window: Fixed Size")
    return (sw_splitter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. GridSearchCV on Panel Data

    Search over forecaster hyperparameters using time series CV. The
    scorer evaluates all panel groups together.
    """)
    return


@app.cell
def _(
    ExpandingWindowSplitter,
    GridSearchCV,
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    tourism,
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
    search.fit(tourism, forecasting_horizon=8)
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

    mo.md(
        f"**Best parameters**: {search.best_params_}\n\n"
        f"**Best score**: {search.best_score_:.4f}"
    )
    return


@app.cell
def _(mo, pl, search):
    _results = search.cv_results_
    _table = pl.DataFrame({
        "alpha": [p["estimator__alpha"] for p in _results["params"]],
        "mean_score": _results["mean_test_score"],
        "std_score": _results["std_test_score"],
        "rank": _results["rank_test_score"],
    }).sort("rank")
    mo.ui.table(_table)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Predict with Best Model

    When `refit=True`, the search object can predict directly using the
    best estimator fitted on the full training data.
    """)
    return


@app.cell
def _(mo, search, tourism):
    _y_pred_best = search.predict(forecasting_horizon=8)
    mo.md(
        f"**Prediction columns**: {_y_pred_best.columns}\n\n"
        f"**Prediction shape**: {_y_pred_best.shape}\n\n"
        "All panel groups are predicted using the best hyperparameters."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Panel CV splits by time**: All groups share the same train/test time boundaries
    - **`ExpandingWindowSplitter`**: Growing train window, fixed test window
    - **`SlidingWindowSplitter`**: Fixed train + test windows sliding forward
    - **`GridSearchCV`**: Evaluates all groups together using the scorer's aggregation mode
    - **`refit=True`**: Best model is fitted on full data after search, ready for prediction
    - **Scoring**: Pass `BaseScorer` instances (not strings) to `scoring=`

    ## Next Steps

    - **Multi-metric search**: See `examples/model_selection/multi_metric_search.py`
    - **Interval search**: See `examples/model_selection/interval_search.py`
    - **CV splitters**: See `examples/model_selection/cv_splitters.py`
    """)
    return


if __name__ == "__main__":
    app.run()
