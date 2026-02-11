# Getting Started

Get up and running with Yohou in minutes.

## Installation

### Using uv (Recommended)

```bash
# Add to project
uv add yohou

# Install for development
git clone https://github.com/yourusername/yohou.git
cd yohou
just install  # Installs dev dependencies + pre-commit hooks
```

### Using pip

```bash
pip install yohou
```

### Verify Installation

```python
import yohou
print(yohou.__version__)
```

## Quick Start

### 1. Load a Dataset

Yohou includes 10 built-in time series datasets:

```python
from yohou.datasets import load_air_passengers

df = load_air_passengers()
print(df.head())
```

Output:
```
shape: (5, 2)
┌────────────┬─────┐
│ time       ┆ y   │
│ ---        ┆ --- │
│ date       ┆ i64 │
╞════════════╪═════╡
│ 1949-01-01 ┆ 112 │
│ 1949-02-01 ┆ 118 │
│ 1949-03-01 ┆ 132 │
│ 1949-04-01 ┆ 129 │
│ 1949-05-01 ┆ 121 │
└────────────┴─────┘
```

### 2. Create Your First Plot

```python
from yohou.plotting import plot_timeseries

fig = plot_timeseries(
    df,
    title="Monthly Airline Passengers",
    x_label="Date",
    y_label="Passengers"
)
fig.show()
```

### 3. Analyze Seasonality

```python
from yohou.plotting import plot_seasonality

fig = plot_seasonality(
    df,
    feature="month",
    aggregation="mean",
    title="Average Passengers by Month"
)
fig.show()
```

### 4. Examine Autocorrelation

```python
from yohou.plotting import plot_correlation_diagnostics

fig = plot_correlation_diagnostics(
    df,
    lags=36,
    title="ACF and PACF Analysis"
)
fig.show()
```

## Working with Your Own Data

### From CSV

```python
import polars as pl

# Load your data
df = pl.read_csv("your_data.csv")

# Ensure you have a 'time' column
df = df.rename({"date": "time"})

# Convert to datetime if needed
df = df.with_columns(
    pl.col("time").str.to_datetime()
)

# Plot
from yohou.plotting import plot_timeseries
plot_timeseries(df)
```

### From Pandas

```python
import pandas as pd
import polars as pl

# Load with pandas
df_pandas = pd.read_csv("your_data.csv")

# Convert to Polars
df = pl.from_pandas(df_pandas)

# Ensure time column
df = df.rename({"date": "time"})
```

### Expected Format

All yohou functions expect:
- A column named `"time"` with datetime values
- One or more numeric columns to plot
- For univariate: columns `["time", "y"]`
- For multivariate: columns `["time", "y", "feature1", "feature2", ...]`
- For panel data: columns `["unique_id", "time", "y"]`

## Common Tasks

### Smoothing Time Series

```python
from yohou.plotting import plot_rolling_statistics

# Rolling mean with ±1 std bands
fig = plot_rolling_statistics(
    df,
    window_size=12,
    statistics=["mean", "std"],
    fill_between=True
)
fig.show()
```

### Comparing Multiple Series

```python
import polars as pl
from yohou.datasets import load_vic_electricity
from yohou.plotting import plot_timeseries

df = load_vic_electricity()

# Plot multiple columns
fig = plot_timeseries(
    df,
    columns=["y", "Temperature"],
    title="Demand vs Temperature"
)
fig.show()
```

### Frequency Analysis

```python
from yohou.plotting import plot_periodogram

fig = plot_periodogram(
    df,
    detrend="linear",
    log_scale=True
)
fig.show()
```

### Panel Data

```python
from yohou.datasets import load_m4_monthly
import polars as pl

df = load_m4_monthly()

# Filter to specific series
df_subset = df.filter(
    pl.col("unique_id").is_in(["M1", "M5", "M10"])
)

# Pivot to wide format
df_wide = df_subset.pivot(
    on="unique_id",
    index="time",
    values="y"
).sort("time")

# Plot
from yohou.plotting import plot_timeseries
plot_timeseries(df_wide, columns=["M1", "M5", "M10"])
```

### Step 3: [Use the main functionality]

```python
# [Add realistic example showing actual usage]
# For example:
# result = instance.process(data)
# output = instance.transform(input_data)

# Example with the provided function
greeting = hello("Python")
print(greeting)
```

## Complete Example

Here's a complete working example:

```python
from yohou.example import hello

# [Replace with realistic multi-step example]
# Step 1: Initialize
names = ["Alice", "Bob", "Charlie"]

# Step 2: Process
greetings = [hello(name) for name in names]

# Step 3: Display results
for greeting in greetings:
    print(greeting)
```

## Try Interactive Examples

For hands-on learning with interactive notebooks, see the [Examples](examples.md) page where you can:

- Run code directly in your browser
- Experiment with different parameters
- See visual outputs in real-time
- Download standalone HTML versions

Or run locally:

=== "just"

    ```bash
    just example
    ```

=== "uv run"

    ```bash
    uv run marimo edit examples/hello.py
    ```

## Next Steps

Now that you have Yohou installed and running:

- **Learn the concepts**: Read the [User Guide](user-guide.md) to understand core concepts and capabilities
- **Explore examples**: Check out the [Examples](examples.md) for real-world use cases
- **Dive into the API**: Browse the [API Reference](api-reference.md) for detailed documentation
- **Get help**: Visit [GitHub Discussions](https://github.com/stateful-y/yohou/discussions) or [open an issue](https://github.com/stateful-y/yohou/issues)
