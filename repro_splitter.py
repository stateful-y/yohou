from datetime import datetime

import polars as pl

from yohou.model_selection import ExpandingWindowSplitter


def main():
    # monthly dates from 1949 to 1958 (10 years = 120 months)
    start_date = datetime(1949, 1, 1)
    # create 120 months
    dates = [
        start_date.replace(
            year=start_date.year + (start_date.month + i - 1) // 12, month=(start_date.month + i - 1) % 12 + 1
        )
        for i in range(120)
    ]

    y = pl.DataFrame({"time": dates, "value": range(120)})

    print(f"Data shape: {y.shape}")
    print(f"Start date: {y['time'].min()}")
    print(f"End date: {y['time'].max()}")

    splitter = ExpandingWindowSplitter(n_splits=2, test_size=12)

    print("\nIterating splits:")
    for i, (train_idx, test_idx) in enumerate(splitter.split(y)):
        train_dates = y[train_idx]["time"]
        test_dates = y[test_idx]["time"]

        train_max = train_dates.max()
        test_min = test_dates.min()

        print(f"Split {i}:")
        print(f"  Train indices: {len(train_idx)} samples, max index: {train_idx.max()}")
        print(f"  Test indices: {len(test_idx)} samples, min index: {test_idx.min()}")
        print(f"  Train Max Date: {train_max}")
        print(f"  Test Min Date:  {test_min}")

        if test_min < train_max:
            print("  [FAIL] Test Min < Train Max")
        else:
            print("  [PASS] Test Min >= Train Max")


if __name__ == "__main__":
    main()
