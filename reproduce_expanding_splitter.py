from datetime import datetime, timedelta

import polars as pl

from yohou.model_selection import ExpandingWindowSplitter


def main():
    # Create dummy data
    length = 115
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(length)]

    # yohou requires a 'time' column in polars DataFrame
    y_train = pl.DataFrame({"time": dates, "value": range(length)})

    # Setup splitter as requested
    n_splits = 2
    test_size = 12

    # We use step_size=test_size only if that's the default or desired behavior.
    # Usually ExpandingWindowSplitter might deduce step_size or require it.
    # Looking at standard implementations (like sklearn custom ones), usually step_size defaults to test_size if not specified.
    # I'll stick to the user's specific parameters: n_splits=2, test_size=12.
    splitter = ExpandingWindowSplitter(n_splits=n_splits, test_size=test_size)

    print(f"Data length: {len(y_train)}")
    print(f"Splitter config: n_splits={n_splits}, test_size={test_size}")

    # Iterate and check
    for i, (train_idx, test_idx) in enumerate(splitter.split(y_train)):
        train_data = y_train[train_idx]
        test_data = y_train[test_idx]

        train_times = train_data["time"]
        test_times = test_data["time"]

        # Determine ranges
        if len(train_times) > 0:
            train_start, train_end = train_times.min(), train_times.max()
        else:
            train_start, train_end = "N/A", "N/A"

        if len(test_times) > 0:
            test_start, test_end = test_times.min(), test_times.max()
        else:
            test_start, test_end = "N/A", "N/A"

        print(f"\nSplit {i}:")
        print(f"  Train indices: {len(train_idx)} samples.")
        print(f"  Test indices:  {len(test_idx)} samples.")

        # Print first and last index to see continuity
        if len(train_idx) > 0:
            print(f"  Train index range: {train_idx[0]} -> {train_idx[-1]}")
        if len(test_idx) > 0:
            print(f"  Test index range:  {test_idx[0]} -> {test_idx[-1]}")

        print(f"  Train time range: {train_start} to {train_end}")
        print(f"  Test time range:  {test_start} to {test_end}")

        # Check overlap
        train_set = set(train_times)
        test_set = set(test_times)
        overlap = train_set.intersection(test_set)

        if overlap:
            print(f"  WARNING: Time overlap detected! {len(overlap)} timestamps overlap.")
            print(f"  Example overlap: {list(overlap)[:5]}")
        else:
            print("  No time overlap between Train and Test.")

        # Check index overlap
        idx_overlap = set(train_idx).intersection(set(test_idx))
        if idx_overlap:
            print(f"  WARNING: Index overlap detected! {len(idx_overlap)} indices overlap.")


if __name__ == "__main__":
    main()
