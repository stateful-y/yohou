
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.metaestimators import _safe_split
from yohou.datasets import load_air_passengers
from yohou.model_selection import ExpandingWindowSplitter

# Load and split like quickstart
y = load_air_passengers()
y = y.rename({"Passengers": "passengers"}).with_columns(
    pl.col("time").cast(pl.Datetime)
)
y_train, y_test = train_test_split(y, test_size=0.2, shuffle=False)

print(f"y_train len: {len(y_train)}")

# Emulate GridSearchCV Splitter
test_size = 12
splitter = ExpandingWindowSplitter(n_splits=2, test_size=test_size)

print("Splitter created. Iterating splits on y_train...")

for i, (train_idx, test_idx) in enumerate(splitter.split(y_train)):
    print(f"Split {i}")
    print(f"Train idx range: {train_idx.min()} - {train_idx.max()}")
    print(f"Test idx range: {test_idx.min()} - {test_idx.max()}")

    # Emulate _safe_split
    # Note: _safe_split signature is (estimator, X, y, indices, train_indices=None)
    # But usually called as (estimator, X, y, indices)
    # Actually, SKLearn GridSearchCV calls it as:
    # X_train, y_train = _safe_split(estimator, X, y, train_train_idxs)
    # Wait, _safe_split returns (X_subset, y_subset) or just X_subset?
    # Actually yohou calling convention: y_train, X_train = _safe_split(forecaster, y, X, train)

    # We pass None for estimator as it's not used for simple slicing usually
    y_sub_train, _ = _safe_split(None, y_train, None, train_idx)
    y_sub_test, _ = _safe_split(None, y_train, None, test_idx)

    print(f"Train Date Max: {y_sub_train['time'].max()}")
    print(f"Test Date Min: {y_sub_test['time'].min()}")

    if y_sub_test['time'].min() <= y_sub_train['time'].max():
        print("FATAL: Test overlaps Train")
    else:
        print("PASS")
