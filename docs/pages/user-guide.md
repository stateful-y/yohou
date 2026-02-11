# User Guide

Comprehensive guide to using Yohou for time series analysis and visualization.

## Overview

Yohou is a time series forecasting and visualization package built on **Polars** and **Scikit-Learn**, with interactive plotting powered by **Plotly**. It provides a clean, functional API for working with univariate, multivariate, and panel time series data.

### Design Philosophy

- **Polars-first**: All data operations use Polars DataFrames (no pandas dependency)
- **Plotly-only**: Interactive visualizations with consistent styling
- **Functional API**: Separate functions with clear responsibilities
- **Type-safe**: Full type hints and Pydantic configuration models
- **Batteries included**: 10 built-in datasets covering common time series patterns

### When to Use Yohou

Yohou excels at:
- Exploratory time series analysis with rich visualizations
- Quick prototyping of forecasting workflows
- Panel/hierarchical time series visualization
- Educational projects demonstrating time series concepts
- Projects requiring Polars integration

## Prerequisites

### Polars DataFrames

Yohou uses [Polars](https://pola.rs/) for all data operations. Polars is a fast DataFrame library built in Rust with a Python API similar to pandas but with better performance and memory efficiency.

Key concepts:
- **Lazy evaluation**: Use `.lazy()` for query optimization
- **Expression API**: Chainable operations with `pl.col()`
- **Type safety**: Strong type system with clear error messages

Learn more: [Polars Documentation](https://pola-rs.github.io/polars/)

### Plotly Interactive Plots

Visualizations use [Plotly](https://plotly.com/python/) to create interactive figures that can be displayed in notebooks, saved as HTML, or embedded in web applications.

Learn more: [Plotly Python Documentation](https://plotly.com/python/)

## Core Concepts

### Time Column Standard

All yohou datasets and plotting functions expect a column named `"time"` containing datetime values. This convention simplifies the API by eliminating the need for a `date_column` parameter in every function.

```python
import polars as pl
from yohou.datasets import load_air_passengers

df = load_air_passengers()
print(df.columns)  # ['time', 'y']

# If your data uses a different name, rename it:
df = df.rename({"date": "time"})
```

### Univariate vs Multivariate

**Univariate**: Single time series with columns `["time", "y"]`

```python
from yohou.datasets import load_air_passengers
from yohou.plotting import plot_timeseries

df = load_air_passengers()
plot_timeseries(df)  # Plots the 'y' column
```

**Multivariate**: Multiple numeric columns alongside time

```python
from yohou.datasets import load_vic_electricity
from yohou.plotting import plot_timeseries

df = load_vic_electricity()
# Columns: ['time', 'y', 'Temperature', 'Holiday']

# Plot specific columns
plot_timeseries(df, columns=["y", "Temperature"])

# Or plot all numeric columns
plot_timeseries(df)  # Plots y, Temperature, Holiday
```

### Panel Data (Grouped Time Series)

Panel data contains multiple time series identified by grouping columns (e.g., `unique_id`, `store`, `region`).

```python
from yohou.datasets import load_m4_monthly

df = load_m4_monthly()
# Columns: ['unique_id', 'time', 'y']
# 50 different series identified by unique_id

# Filter to specific series
df_single = df.filter(pl.col("unique_id") == "M1")

# Pivot to wide format for multivariate plotting
df_wide = (
    df.filter(pl.col("unique_id").is_in(["M1", "M5", "M10"]))
    .pivot(on="unique_id", index="time", values="y")
    .sort("time")
)
```

**Note**: Full panel support with `panel_group_name`, `facet_by`, and dropdown parameters is planned for future releases. Current examples demonstrate filtering + pivoting workflows.

## Datasets

Yohou includes 10 built-in datasets spanning different time series types:

### Univariate Datasets

- **air_passengers**: Monthly airline passengers (trend + seasonality)
- **sunspots**: Monthly sunspot counts (cyclic patterns)

### Multivariate Datasets

- **vic_electricity**: 30-min electricity demand with temperature
- **ett_m1**: Transformer temperature (7 features, 15-min intervals)

### Panel Datasets

- **m4_monthly**: 50 monthly series
- **m4_quarterly**: 50 quarterly series
- **m4_hourly**: 50 hourly series
- **store_sales**: 10 stores × 50 items, daily sales
- **walmart_sales**: 3 branches with product categories
- **australian_tourism**: 76 regions × 4 purposes, quarterly

```python
from yohou.datasets import (
    load_air_passengers,
    load_sunspots,
    load_vic_electricity,
    load_m4_monthly,
    load_store_sales,
)

df = load_air_passengers()
df.head()
```

## Plotting Functions

All plotting functions follow a consistent signature:

```python
def plot_<type>(
    df: pl.DataFrame,
    *,  # Keyword-only arguments
    columns: str | list[str] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,  # Function-specific styling
) -> go.Figure
```

### Core Time Series Plots

**plot_timeseries()**
```python
from yohou.plotting import plot_timeseries

# Single series
plot_timeseries(df)

# Multiple series
plot_timeseries(df, columns=["y", "Temperature"])

# Styling
plot_timeseries(df, line_width=3.0, line_color="#DC2626")
```

**plot_rolling_statistics()**
```python
from yohou.plotting import plot_rolling_statistics

# Rolling mean
plot_rolling_statistics(df, window_size=12, statistics="mean")

# Mean ± std bands
plot_rolling_statistics(
    df,
    window_size=12,
    statistics=["mean", "std"],
    fill_between=True
)
```

**plot_exponential_moving_average()**
```python
from yohou.plotting import plot_exponential_moving_average

plot_exponential_moving_average(df, span=12, show_original=True)
```

**plot_boxplot()**
```python
from yohou.plotting import plot_boxplot

# Monthly boxplots
plot_boxplot(df, period="1mo")

# Weekly with points
plot_boxplot(df, period="1w", show_points="all")
```

### Diagnostic Plots

**plot_seasonality()**
```python
from yohou.plotting import plot_seasonality

# Monthly patterns
plot_seasonality(df, feature="month", aggregation="mean")

# Hour-of-day patterns
plot_seasonality(df, feature="hour", aggregation="median")
```

**plot_autocorrelation() / plot_partial_autocorrelation()**
```python
from yohou.plotting import (
    plot_autocorrelation,
    plot_partial_autocorrelation,
    plot_correlation_diagnostics,
)

# ACF
plot_autocorrelation(df, lags=40)

# PACF
plot_partial_autocorrelation(df, lags=40)

# Combined ACF + PACF
plot_correlation_diagnostics(df, lags=40)
```

### Frequency Analysis

**plot_periodogram()**
```python
from yohou.plotting import plot_periodogram

plot_periodogram(df, detrend="linear", log_scale=True)
```

**plot_lag_scatter()**
```python
from yohou.plotting import plot_lag_scatter

plot_lag_scatter(df, lags=[1, 6, 12, 24])
```

### Forecasting & Comparison

**plot_forecast()**
```python
from yohou.plotting import plot_forecast

# Requires df with is_forecast column
df_combined = pl.concat([df_historical, df_forecast])
plot_forecast(df_combined, n_history=36)
```

**plot_prediction_interval()**
```python
from yohou.plotting import plot_prediction_interval

# Requires lower_bound and upper_bound columns
df_with_bounds = df.with_columns([
    (pl.col("y") * 0.9).alias("lower_bound"),
    (pl.col("y") * 1.1).alias("upper_bound"),
])
plot_prediction_interval(df_with_bounds)
```

**plot_residuals()**
```python
from yohou.plotting import plot_residuals

# Requires residuals and fitted columns
df_model = df.with_columns([
    (pl.col("y") - pl.col("y").rolling_mean(12)).alias("residuals"),
    pl.col("y").rolling_mean(12).alias("fitted"),
])
plot_residuals(df_model)
```

### Specialized Plots

**plot_cross_correlation()**
```python
from yohou.plotting import plot_cross_correlation

# Requires multivariate data
plot_cross_correlation(
    df,
    x_column="y",
    y_column="Temperature",
    lags=48
)
```

**plot_calendar_heatmap()**
```python
from yohou.plotting import plot_calendar_heatmap

# Daily patterns in calendar view
plot_calendar_heatmap(
    df,
    column="sales",
    aggregation="sum",
    year=2019
)
```

## Common Workflows

### Exploratory Data Analysis

```python
from yohou.datasets import load_air_passengers
from yohou.plotting import (
    plot_timeseries,
    plot_seasonality,
    plot_correlation_diagnostics,
)

df = load_air_passengers()

# 1. Visualize raw data
plot_timeseries(df, title="Air Passengers Time Series")

# 2. Check seasonal patterns
plot_seasonality(df, feature="month", title="Monthly Seasonality")

# 3. Examine autocorrelation
plot_correlation_diagnostics(df, lags=36)
```

### Panel Data Analysis

```python
from yohou.datasets import load_m4_monthly
from yohou.plotting import plot_timeseries, plot_boxplot
import polars as pl

df = load_m4_monthly()

# Filter to specific series
series_ids = ["M1", "M5", "M10"]
df_subset = df.filter(pl.col("unique_id").is_in(series_ids))

# Pivot to wide format
df_wide = df_subset.pivot(
    on="unique_id",
    index="time",
    values="y"
).sort("time")

# Compare series
plot_timeseries(df_wide, columns=series_ids)

# Analyze distributions
plot_boxplot(df_subset, period="1y")
```

### Multivariate Correlation

```python
from yohou.datasets import load_vic_electricity
from yohou.plotting import (
    plot_timeseries,
    plot_cross_correlation,
)

df = load_vic_electricity()

# Visualize both variables
plot_timeseries(df, columns=["y", "Temperature"])

# Check cross-correlation
plot_cross_correlation(
    df,
    x_column="y",
    y_column="Temperature",
    lags=48
)
```

## Next Steps

- Explore the [Examples](examples.md) for complete workflows
- Browse the [API Reference](api-reference.md) for detailed function documentation
- Check the [Getting Started](getting-started.md) guide for installation

[Detailed explanation of this feature]

**Example:**

```python
# Realistic example code
```

### [Feature 3: Feature Name]

[Detailed explanation with focus on practical use]

### [Feature 4: Feature Name]

[Explanation including trade-offs and best practices]

### [Feature 5: Feature Name]

[Explanation of advanced or specialized feature]

### [Feature 6: Feature Name] *(Experimental)*

[Explanation of experimental or future features, noting maturity level and potential changes]

> **Note**: This feature is experimental and may change in future versions.

## Configuration

### Basic Configuration

[Explanation of how to configure the package]

```python
from yohou import [ConfigClass]

config = [ConfigClass](
    option_1="value",  # Description of what this controls
    option_2=True,     # Description of what this controls
)
```

Or using a configuration file:

```yaml
# config.yaml
yohou:
  option_1: value
  option_2: true
```

### Advanced Configuration

[More advanced configuration options and patterns]

## Best Practices

### 1. [Best Practice Category 1]

**Do:**
- [Specific recommendation with brief explanation]
- [Another recommendation]

**Don't:**
- [Anti-pattern to avoid with explanation why]

### 2. [Best Practice Category 2]

[Explanation of recommended patterns and approaches]

### 3. [Best Practice Category 3]

[Additional guidance on effective usage]

## Limitations and Considerations

Understanding the limitations helps you make informed decisions:

1. **[Limitation 1]**: [Explanation of the trade-off or constraint and why it exists]

2. **[Limitation 2]**: [Explanation of what's not supported and potential workarounds]

3. **[Consideration 3]**: [Important factor to consider when using this package]

## Troubleshooting

### [Common Issue 1]

**Problem**: [Description of the problem users encounter]

**Solution**: [Step-by-step solution with code examples]

```python
# Example showing the solution
```

### [Common Issue 2]

**Problem**: [Description]

**Solution**: [Resolution with explanation]

**Solution**: [Resolution with explanation]

## FAQ

### [Question 1]?

[Detailed answer addressing the question comprehensively]

### [Question 2]?

[Answer with examples or links as needed]

### [Question 3]?

[Answer that might reference other sections or external resources]

## Next Steps

Now that you understand the core concepts and features:

- Follow the [Getting Started](getting-started.md) guide to start using Yohou
- Explore the [Examples](examples.md) for real-world use cases
- Check the [API Reference](api-reference.md) for detailed API documentation
- Join the community on [GitHub Discussions](https://github.com/stateful-y/yohou/discussions)
