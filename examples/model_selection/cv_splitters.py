# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.19.11"
__gallery__ = {
    "title": "CV Splitters",
    "description": "Demonstrate ExpandingWindowSplitter and SlidingWindowSplitter for temporal cross-validation with configurable gap, test_size, and fold visualisation.",
    "category": "tutorial",
    "section": "getting-started",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cross-Validation Splitters for Time Series

    Standard k-fold CV violates temporal order. Yohou provides time-respecting
    splitters that always train on past data and test on future data.

    ## What You'll Learn

    - [`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.ExpandingWindowSplitter/): Growing training window
    - [`SlidingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.SlidingWindowSplitter/): Fixed-size sliding window
    - Visualizing splits with [`plot_splits`](/pages/api/generated/yohou.plotting.model_selection.plot_splits/)

    ## Prerequisites

    Basic understanding of cross-validation concepts.
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl

    from yohou.datasets import fetch_tourism_monthly
    from yohou.model_selection import ExpandingWindowSplitter, SlidingWindowSplitter
    from yohou.plotting import plot_splits, plot_time_series

    return (
        ExpandingWindowSplitter,
        SlidingWindowSplitter,
        fetch_tourism_monthly,
        pl,
        plot_splits,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We create a time series dataset to demonstrate how each cross-validation splitter partitions the data into training and test folds.
    """)


@app.cell
def _(fetch_tourism_monthly):
    y = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    print(f"Total observations: {len(y)}")
    return (y,)


@app.cell
def _(plot_time_series, y):
    plot_time_series(y, title="Tourism Monthly")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. ExpandingWindowSplitter

    The training set **grows** with each split. The earliest data is always included.
    `n_splits` controls how many train/test pairs are generated.
    """)


@app.cell
def _(ExpandingWindowSplitter, y):
    expanding = ExpandingWindowSplitter(n_splits=4, test_size=12)

    print(f"Number of splits: {expanding.get_n_splits(y)}")
    for _i, (_train_idx, _test_idx) in enumerate(expanding.split(y)):
        print(f"  Split {_i}: train={len(_train_idx)}, test={len(_test_idx)}")
    return (expanding,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_splits`](/pages/api/generated/yohou.plotting.model_selection.plot_splits/) renders each CV fold as a horizontal bar, with training
    periods in one colour and test periods in another. This makes it easy
    to verify that the splitter respects temporal order.
    """)


@app.cell
def _(expanding, plot_splits, y):
    plot_splits(y, expanding, title="Expanding Window Splitter (4 splits, test=12)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. SlidingWindowSplitter

    The training window has a **fixed size** that slides forward.
    Older data is dropped, keeping model training focused on recent patterns.
    """)


@app.cell
def _(SlidingWindowSplitter, y):
    sliding = SlidingWindowSplitter(n_splits=5, test_size=12)

    print(f"Number of splits: {sliding.get_n_splits()}")
    for _i, (_train_idx, _test_idx) in enumerate(sliding.split(y)):
        print(f"  Split {_i}: train={len(_train_idx)}, test={len(_test_idx)}")
    return (sliding,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_splits`](/pages/api/generated/yohou.plotting.model_selection.plot_splits/) now shows the sliding window structure, where each fold
    has a fixed-size training window that moves forward through time.
    """)


@app.cell
def _(plot_splits, sliding, y):
    plot_splits(y, sliding, title="Sliding Window Splitter (5 folds, test=12)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Controlling Window Size

    `max_train_size` on [`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.ExpandingWindowSplitter/) limits growth, effectively
    becoming a sliding window that fills up from the expanding start.
    """)


@app.cell
def _(ExpandingWindowSplitter, plot_splits, y):
    capped = ExpandingWindowSplitter(n_splits=4, test_size=12, max_train_size=80)

    for _i, (_train_idx, _test_idx) in enumerate(capped.split(y)):
        print(f"  Split {_i}: train={len(_train_idx)} (capped at 80), test={len(_test_idx)}")

    plot_splits(y, capped, title="Expanding Window capped at 80")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. SlidingWindowSplitter with Stride

    `stride` controls how many observations the window advances between splits.
    Default is `test_size` (non-overlapping test sets).
    """)


@app.cell
def _(SlidingWindowSplitter, plot_splits, y):
    strided = SlidingWindowSplitter(n_splits=5, test_size=12, stride=6)

    print(f"Splits with stride=6: {strided.get_n_splits()}")
    plot_splits(y, strided, title="Sliding Window with stride=6 (overlapping tests)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - [`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.ExpandingWindowSplitter/): Training window grows: uses all historical data
    - [`SlidingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.SlidingWindowSplitter/): Fixed training window: focuses on recent data
    - `max_train_size`: Caps expanding window growth
    - `stride`: Controls step size between splits (for overlapping test sets)
    - [`plot_splits`](/pages/api/generated/yohou.plotting.model_selection.plot_splits/) visualizes the temporal structure of CV splits
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Hyperparameter search**: See [`hyperparameter_search.py`](/examples/model_selection/hyperparameter_search/) for GridSearchCV
    - **Scoring**: See [Metrics](/examples/#metrics) for evaluation metrics used in CV
    """)


if __name__ == "__main__":
    app.run()
