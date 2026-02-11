
import polars as pl
from sklearn.model_selection import train_test_split
from yohou.datasets import load_air_passengers

y = load_air_passengers()

# Simulate the split used in quickstart.py
y_train, y_test = train_test_split(y, test_size=0.2, shuffle=False)

print(f"Type of y_train: {type(y_train)}")
if isinstance(y_train, pl.DataFrame):
     print(f"Train max: {y_train['time'].max()}")
     print(f"Test min: {y_test['time'].min()}")

if y_train['time'].max() >= y_test['time'].min():
    print("FATAL: Train overlaps Test")
else:
    print("Split looks correct")

# Check if y is sorted
is_sorted = y['time'].is_sorted()
print(f"Original y is sorted: {is_sorted}")
