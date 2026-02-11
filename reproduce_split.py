import numpy as np
import polars as pl

from yohou.model_selection import ExpandingWindowSplitter


def test_splitter(n_samples, n_splits, test_size=None, gap=0):
    print(f"--- n_samples={n_samples}, n_splits={n_splits}, test_size={test_size}, gap={gap} ---")
    y = pl.DataFrame({"time": np.arange(n_samples), "y": np.arange(n_samples)})
    splitter = ExpandingWindowSplitter(n_splits=n_splits, test_size=test_size, gap=gap)

    print(f"get_n_splits: {splitter.get_n_splits(y)}")

    splits = list(splitter.split(y))
    print(f"Actual splits yielded: {len(splits)}")

    for i, (train, test) in enumerate(splits):
        print(f"Split {i}: Train {train}, Test {test}")
        if len(train) > 0 and len(test) > 0:
            print(f"  Train max: {train.max()}, Test min: {test.min()}")
            if train.max() >= test.min():
                print("  OVERLAP DETECTED!")
            else:
                print(f"  Gap observed: {test.min() - train.max() - 1}")
        elif len(train) == 0:
            print("  Empty train set!")


# Case 1: Standard
test_splitter(n_samples=100, n_splits=3, test_size=10, gap=0)

# Case 2: Gap
test_splitter(n_samples=100, n_splits=3, test_size=10, gap=5)

# Case 3: Insufficient samples for requested splits (Should drop splits)
test_splitter(n_samples=10, n_splits=3, test_size=4, gap=0)

# Case 4: Negative starting index but valid due to gap?
# Range starts at negative, but range + gap is non-negative?
# n_samples=10, n_splits=1, test_size=5, gap=3.
# start = 10 - 3 - 5 = 2. Positive.
# Let's try to force start < 0 but start + gap >= 0
# n_samples=5, n_splits=1, test_size=5, gap=3.
# start = 5 - 3 - 5 = -3.
# start + gap = 0.
test_splitter(n_samples=5, n_splits=1, test_size=5, gap=3)

# Case 5: Overlap check
test_splitter(n_samples=10, n_splits=2, test_size=2, gap=0)
