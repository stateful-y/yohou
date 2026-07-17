# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "How to Visualize Model Selection Results",
    "description": "Visualise CV fold geometry with expanding and sliding window splitters and hyperparameter search results with plot_splits and plot_cv_results_scatter.",
    "category": "how-to",
    "section": "visualization",
    "companion": "/pages/how-to/visualize-scores/",
    "api_references": ["GridSearchCV", "SeasonalNaive", "plot_cv_results_scatter", "plot_splits"],
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

    **Prerequisites:** Familiarity with cross-validation concepts. See `examples/cross_validation.py`
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
        ExpandingWindowSplitter(n_splits=3, test_size=12),
        train_color="#059669",
        test_color="#dc2626",
        title="Expanding Window - Custom Colours",
    )


@app.cell
def _(SlidingWindowSplitter, plot_splits, tourism):
    plot_splits(
        tourism,
        SlidingWindowSplitter(n_splits=3, test_size=12),
        title="Sliding Window - 3 Folds",
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

    from yohou.model_selection import train_test_split as _tts

    y_train, _ = _tts(tourism, test_size=fh)

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
    ## Next Steps

    - **Evaluation**: See [`examples/plotting/evaluation.py`](/examples/visualization/evaluation/) for residual and score distribution plots
    - **Forecasting**: See [`examples/plotting/forecasting_visualization.py`](/examples/visualization/forecasting_visualization/) for forecast and comparison plots
    - **Similarity**: See `examples/plotting/similarity_heatmap.py` for distance-based interval weights
    - **Signal processing**: See [`examples/visualization/signal_processing_visualization.py`](/examples/signal_processing_visualization/) for spectrum and phase analysis
    """)


if __name__ == "__main__":
    app.run()
