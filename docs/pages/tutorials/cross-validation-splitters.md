# Cross-Validation Splitters

In this tutorial, we will create temporal cross-validation folds that respect time ordering, visualize them with [`plot_splits`](/pages/api/generated/yohou.plotting.plot_splits/), and compare how expanding and sliding window strategies affect fold geometry. Along the way, we will control training set growth with `max_train_size` and adjust fold overlap with `stride`.

<!-- COMPANION_NOTEBOOKS -->

## Prerequisites

- Completed [Getting Started](getting-started.md)

## 1. Prepare Data

We use a single series loaded with [`fetch_tourism_monthly`](/pages/api/generated/yohou.datasets.fetch_tourism_monthly/) from the Tourism Monthly dataset: 187 months of visitor arrivals from January 1979 to July 1994.

```python
from yohou.datasets import fetch_tourism_monthly

bunch = fetch_tourism_monthly(n_series=1)
y = bunch.frame
print(f"Total rows: {len(y)}")
print(y.head(3))
```

```text
Total rows: 187
shape: (3, 2)
┌─────────────────────┬──────────┐
│ time                ┆ tourists │
│ ---                 ┆ ---      │
│ datetime[μs]        ┆ f64      │
╞═════════════════════╪══════════╡
│ 1979-01-01 00:00:00 ┆ 1149.87  │
│ 1979-02-01 00:00:00 ┆ 1053.8   │
│ 1979-03-01 00:00:00 ┆ 1388.88  │
└─────────────────────┴──────────┘
```

## 2. Expanding Window

[`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.ExpandingWindowSplitter/) grows the training set with each fold while keeping the test window fixed. Each successive training set is a superset of the previous one:

```python
from yohou.model_selection import ExpandingWindowSplitter
from yohou.plotting import plot_splits

expanding = ExpandingWindowSplitter(n_splits=4, test_size=12)
print(f"Number of folds: {expanding.get_n_splits()}")
```

```text
Number of folds: 4
```

Visualize the folds with [`plot_splits`](/pages/api/generated/yohou.plotting.plot_splits/):

```python
fig = plot_splits(y, expanding)
fig.show()
```

Notice that each fold's training region (left bar) grows longer while the test region (right bar) stays the same width. Fold 1 uses the least training data; fold 4 uses the most. The test windows do not overlap, so each fold evaluates a different time period.

## 3. Sliding Window

[`SlidingWindowSplitter`](/pages/api/generated/yohou.model_selection.SlidingWindowSplitter/) uses a fixed-size training window that slides forward. This is useful when older data may no longer be representative (concept drift):

!!! note "Concept drift"
    Many real-world time series change their statistical properties over time.
    Consumer behavior shifts, markets evolve, and sensor characteristics degrade.
    When the data-generating process changes, older observations can mislead
    the model. A sliding window limits training to recent data, allowing the
    model to adapt to the current regime rather than averaging across regimes
    that may no longer be relevant.

```python
from yohou.model_selection import SlidingWindowSplitter

sliding = SlidingWindowSplitter(n_splits=4, train_size=60, test_size=12)

fig = plot_splits(y, sliding)
fig.show()
```

Notice that every fold's training region is the same width (60 rows). Both the training and test windows move forward together. The oldest data drops off as newer data enters, keeping the model focused on recent patterns.

## 4. Control the Training Size

[`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.ExpandingWindowSplitter/) can cap the maximum training size with `max_train_size`. This combines the benefits of using recent data with some expansion:

```python
expanding_capped = ExpandingWindowSplitter(
    n_splits=4, test_size=12, max_train_size=80
)

fig = plot_splits(y, expanding_capped)
fig.show()
```

Notice that the early folds grow as before, but once the training region reaches 80 rows it stops expanding. This gives you some of the recency benefit of a sliding window while still using more data than a fixed window in the early folds.

## 5. Sliding Window with Stride

By default, [`SlidingWindowSplitter`](/pages/api/generated/yohou.model_selection.SlidingWindowSplitter/) moves forward by `test_size` rows between folds. The `stride` parameter lets you control the step size independently:

```python
sliding_stride = SlidingWindowSplitter(
    n_splits=4, train_size=60, test_size=12, stride=6
)

fig = plot_splits(y, sliding_stride)
fig.show()
```

Notice that consecutive test windows now overlap: each fold advances by only 6 rows instead of 12. Overlapping folds produce more evaluation points from the same data, but the fold scores will be correlated because they share test observations.

## What You Built

In this tutorial, we:

- Created temporal cross-validation folds with [`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.ExpandingWindowSplitter/) and [`SlidingWindowSplitter`](/pages/api/generated/yohou.model_selection.SlidingWindowSplitter/)
- Visualized splits with [`plot_splits`](/pages/api/generated/yohou.plotting.plot_splits/) to verify temporal ordering and fold geometry
- Capped training growth with `max_train_size` to balance data volume against recency
- Controlled fold overlap with `stride` to trade more evaluation points for correlated scores

## Next Steps

- [Forecasting Workflow](forecasting-workflow.md) to use splitters inside [`GridSearchCV`](/pages/api/generated/yohou.model_selection.GridSearchCV/)
- [Model Selection](../explanation/model-selection.md) for the conceptual background on temporal CV
- [How to Tune Hyperparameters](../how-to/tune-hyperparameters.md) for using splitters inside grid search and randomized search
