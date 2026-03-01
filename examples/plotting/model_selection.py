# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Model Selection Plots",
    "description": "Visualise CV fold geometry with expanding and sliding window splitters and hyperparameter search results with plot_splits and plot_cv_results_scatter.",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():

    from yohou.datasets import fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import (
        ExpandingWindowSplitter,
        GridSearchCV,
        SlidingWindowSplitter,
    )
    from yohou.plotting import plot_cv_results_scatter, plot_splits
    from yohou.point import SeasonalNaive

    return (
        ExpandingWindowSplitter,
        GridSearchCV,
        MeanAbsoluteError,
        SeasonalNaive,
        SlidingWindowSplitter,
        fetch_tourism_monthly,
        plot_cv_results_scatter,
        plot_splits,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Selection Visualization

    ## What You'll Learn

    - Visualizing train/test CV splits with [`plot_splits`](/pages/api/generated/yohou.plotting.model_selection.plot_splits/)
    - Comparing expanding vs sliding window strategies
    - Using gap periods to prevent data leakage
    - Plotting hyperparameter search results with [`plot_cv_results_scatter`](/pages/api/generated/yohou.plotting.model_selection.plot_cv_results_scatter/)

    ## Prerequisites

    Familiarity with cross-validation concepts. See `examples/cross_validation.py`
    for a detailed introduction.
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Data

    We load the Monthly Tourism dataset via [`fetch_tourism_monthly`](/pages/api/generated/yohou.datasets._fetchers.fetch_tourism_monthly/) and extract
    a single univariate series for the cross-validation and search demonstrations.
    """)


@app.cell
def _(fetch_tourism_monthly):
    tourism = (
        fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    )
    return (tourism,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Cross-Validation Splits

    [`plot_splits`](/pages/api/generated/yohou.plotting.model_selection.plot_splits/) renders each fold as a horizontal bar, with colour-coded
    train, test, and optional gap segments. Vary the splitter type, number
    of splits, and custom colours.
    """)


@app.cell
def _(ExpandingWindowSplitter, plot_splits, tourism):
    plot_splits(
        tourism,
        ExpandingWindowSplitter(n_splits=3, test_size=12),
        title="Expanding Window - 3 Folds, 12-Step Test",
    )


@app.cell
def _(ExpandingWindowSplitter, plot_splits, tourism):
    plot_splits(
        tourism,
        ExpandingWindowSplitter(n_splits=5, test_size=12),
        title="Expanding Window - 5 Folds",
    )


@app.cell
def _(SlidingWindowSplitter, plot_splits, tourism):
    plot_splits(
        tourism,
        SlidingWindowSplitter(n_splits=5, test_size=12),
        title="Sliding Window - 5 Folds / 12 Test",
    )


@app.cell
def _(SlidingWindowSplitter, plot_splits, tourism):
    plot_splits(
        tourism,
        SlidingWindowSplitter(n_splits=5, test_size=12, stride=6),
        title="Sliding Window - Stride 6 (Overlapping)",
    )


@app.cell
def _(ExpandingWindowSplitter, plot_splits, tourism):
    plot_splits(
        tourism,
        ExpandingWindowSplitter(n_splits=3, test_size=12, gap=6),
        train_color="#059669",
        test_color="#dc2626",
        gap_color="#fbbf24",
        title="Expanding Window - Gap=6, Custom Colours",
    )


@app.cell
def _(SlidingWindowSplitter, plot_splits, tourism):
    plot_splits(
        tourism,
        SlidingWindowSplitter(n_splits=3, test_size=12, gap=3),
        title="Sliding Window - Gap=3",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Hyperparameter Search Results

    [`plot_cv_results_scatter`](/pages/api/generated/yohou.plotting.model_selection.plot_cv_results_scatter/) visualises the relationship between a
    hyperparameter and the cross-validation score. We first run a
    [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) over [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/) with varying `seasonality` values,
    using [`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.ExpandingWindowSplitter/) for temporal CV and [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.point.MeanAbsoluteError/)
    as the scorer. The resulting `cv_results_` dictionary is then passed
    to [`plot_cv_results_scatter`](/pages/api/generated/yohou.plotting.model_selection.plot_cv_results_scatter/) to inspect score sensitivity.
    """)


@app.cell
def _(
    ExpandingWindowSplitter,
    GridSearchCV,
    MeanAbsoluteError,
    SeasonalNaive,
    tourism,
):
    fh = 12
    y_train = tourism.head(len(tourism) - fh)

    search = GridSearchCV(
        forecaster=SeasonalNaive(),
        cv=ExpandingWindowSplitter(n_splits=3, test_size=fh),
        param_grid={"seasonality": [3, 6, 12, 24]},
        scoring=MeanAbsoluteError(),
    )
    search.fit(y_train, forecasting_horizon=fh)
    cv_results = search.cv_results_
    return (cv_results,)


@app.cell
def _(cv_results, plot_cv_results_scatter):
    plot_cv_results_scatter(
        cv_results,
        param_name="seasonality",
        title="CV Results - Default",
    )


@app.cell
def _(cv_results, plot_cv_results_scatter):
    plot_cv_results_scatter(
        cv_results,
        param_name="seasonality",
        highlight_best=False,
        title="CV Results - No Best Highlight",
    )


@app.cell
def _(cv_results, plot_cv_results_scatter):
    plot_cv_results_scatter(
        cv_results,
        param_name="seasonality",
        show_std=False,
        title="CV Results - No Error Bars",
    )


@app.cell
def _(cv_results, plot_cv_results_scatter):
    plot_cv_results_scatter(
        cv_results,
        param_name="seasonality",
        best_marker_color="#059669",
        marker_size=14.0,
        title="CV Results - Custom Marker Style",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **plot_splits** colour-codes train / test / gap segments per fold; custom colours via `train_color`, `test_color`, `gap_color`
    - **ExpandingWindowSplitter** grows the training window; **SlidingWindowSplitter** slides a fixed window with optional `stride`
    - Setting `gap > 0` inserts a buffer between train and test to prevent data leakage
    - **plot_cv_results_scatter** plots hyperparameter vs score; `highlight_best` marks the optimal value; `show_std` toggles error bars

    ## Next Steps

    - **Evaluation**: See [`examples/plotting/evaluation.py`](/examples/plotting/evaluation/) for residual and score distribution plots
    - **Forecasting**: See [`examples/plotting/forecasting_visualization.py`](/examples/plotting/forecasting_visualization/) for forecast and comparison plots
    - **Similarity**: See `examples/plotting/similarity_heatmap.py` for distance-based interval weights
    - **Signal processing**: See [`examples/plotting/signal_processing.py`](/examples/plotting/signal_processing/) for spectrum and phase analysis
    """)


if __name__ == "__main__":
    app.run()
