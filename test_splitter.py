from datetime import datetime, timedelta

import polars as pl

from yohou.model_selection import ExpandingWindowSplitter

# Create time series
time = [datetime(1949, 1, 1) + timedelta(days=30 * i) for i in range(144)]  # 12 years
y = pl.DataFrame({"time": time, "value": range(144)})

print(f"Total samples: {len(y)}")
print(f"Date range: {y['time'].min()} to {y['time'].max()}")

# Create splitter matching the tutorial
splitter = ExpandingWindowSplitter(n_splits=2, test_size=12)

for i, (train_idx, test_idx) in enumerate(splitter.split(y)):
    print(f"\nFold {i}:")
    print(f"  Train indices: [{train_idx[0]}...{train_idx[-1]}] (len={len(train_idx)})")
    print(f"  Train dates: {y[int(train_idx[0]), 'time']} to {y[int(train_idx[-1]), 'time']}")
    print(f"  Test indices: [{test_idx[0]}...{test_idx[-1]}] (len={len(test_idx)})")
    print(f"  Test dates: {y[int(test_idx[0]), 'time']} to {y[int(test_idx[-1]), 'time']}")
