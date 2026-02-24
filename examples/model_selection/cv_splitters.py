# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Cross-Validation Splitters for Time Series.

Demonstrates ExpandingWindowSplitter and SlidingWindowSplitter with visualization.
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
    # Cross-Validation Splitters for Time Series

    Standard k-fold CV violates temporal order. Yohou provides time-respecting
    splitters that always train on past data and test on future data.

    ## What You'll Learn

    - `ExpandingWindowSplitter`: Growing training window
    - `SlidingWindowSplitter`: Fixed-size sliding window
    - Controlling `gap` between train and test
    - Visualizing splits with `plot_splits`

    ## Prerequisites

    Basic understanding of cross-validation concepts.
    """)
    return

@app.cell(hide_code=True)
def _():
    import polars as pl

    from yohou.datasets import fetch_tourism_monthly
    from yohou.model_selection import ExpandingWindowSplitter, SlidingWindowSplitter
    from yohou.plotting import plot_splits

    return (
        ExpandingWindowSplitter,
        SlidingWindowSplitter,
        fetch_tourism_monthly,
        pl,
        plot_splits,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We create a time series dataset to demonstrate how each cross-validation splitter partitions the data into training and test folds.
    """)
    return

@app.cell
def _(fetch_tourism_monthly):
    y = (
        fetch_tourism_monthly()
        .frame.select("time", "T1__tourists").drop_nulls()
        .rename({"T1__tourists": "passengers"})
    )
    print(f"Total observations: {len(y)}")
    return (y,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. ExpandingWindowSplitter

    The training set **grows** with each split. The earliest data is always included.
    `n_splits` controls how many train/test pairs are generated.
    """)
    return

@app.cell
def _(ExpandingWindowSplitter, y):
    expanding = ExpandingWindowSplitter(n_splits=4, test_size=12)

    print(f"Number of splits: {expanding.get_n_splits(y)}")
    for _i, (_train_idx, _test_idx) in enumerate(expanding.split(y)):
        print(f"  Split {_i}: train={len(_train_idx)}, test={len(_test_idx)}")
    return (expanding,)

@app.cell
def _(expanding, plot_splits, y):
    plot_splits(y, expanding, title="Expanding Window Splitter (4 splits, test=12)")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. SlidingWindowSplitter

    The training window has a **fixed size** that slides forward.
    Older data is dropped, keeping model training focused on recent patterns.
    """)
    return

@app.cell
def _(SlidingWindowSplitter, y):
    sliding = SlidingWindowSplitter(n_splits=5, test_size=12)

    print(f"Number of splits: {sliding.get_n_splits()}")
    for _i, (_train_idx, _test_idx) in enumerate(sliding.split(y)):
        print(f"  Split {_i}: train={len(_train_idx)}, test={len(_test_idx)}")
    return (sliding,)

@app.cell
def _(plot_splits, sliding, y):
    plot_splits(y, sliding, title="Sliding Window Splitter (5 folds, test=12)")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Using `gap` to Simulate Forecast Delay

    In practice, there's often a gap between the last known observation and when
    forecasts are needed. The `gap` parameter simulates this.
    """)
    return

@app.cell
def _(ExpandingWindowSplitter, plot_splits, y):
    expanding_gap = ExpandingWindowSplitter(n_splits=3, test_size=12, gap=6)

    for _i, (_train_idx, _test_idx) in enumerate(expanding_gap.split(y)):
        print(f"  Split {_i}: train ends at {_train_idx[-1]}, test starts at {_test_idx[0]} (gap=6)")

    plot_splits(y, expanding_gap, title="Expanding Window with gap=6")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Controlling Window Size

    `max_train_size` on `ExpandingWindowSplitter` limits growth, effectively
    becoming a sliding window that fills up from the expanding start.
    """)
    return

@app.cell
def _(ExpandingWindowSplitter, plot_splits, y):
    capped = ExpandingWindowSplitter(n_splits=4, test_size=12, max_train_size=80)

    for _i, (_train_idx, _test_idx) in enumerate(capped.split(y)):
        print(f"  Split {_i}: train={len(_train_idx)} (capped at 80), test={len(_test_idx)}")

    plot_splits(y, capped, title="Expanding Window capped at 80")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. SlidingWindowSplitter with Stride

    `stride` controls how many observations the window advances between splits.
    Default is `test_size` (non-overlapping test sets).
    """)
    return

@app.cell
def _(SlidingWindowSplitter, plot_splits, y):
    strided = SlidingWindowSplitter(n_splits=5, test_size=12, stride=6)

    print(f"Splits with stride=6: {strided.get_n_splits()}")
    plot_splits(y, strided, title="Sliding Window with stride=6 (overlapping tests)")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - `ExpandingWindowSplitter`: Training window grows: uses all historical data
    - `SlidingWindowSplitter`: Fixed training window: focuses on recent data
    - `gap`: Simulates forecast delay between train and test periods
    - `max_train_size`: Caps expanding window growth
    - `stride`: Controls step size between splits (for overlapping test sets)
    - `plot_splits` visualizes the temporal structure of CV splits
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Hyperparameter search**: See `hyperparameter_search.py` for GridSearchCV
    - **Scoring**: See `metrics/` for evaluation metrics used in CV
    """)
    return

if __name__ == "__main__":
    app.run()
