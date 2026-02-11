# Examples

Explore real-world time series visualization with Yohou through interactive Marimo notebooks.

## Available Examples

All examples are located in the `examples/` directory and can be run as:
- **Interactive notebooks**: `marimo edit examples/<name>.py`
- **Standalone scripts**: `python examples/<name>.py`

### 1. Air Passengers - Comprehensive Demonstration

**Dataset**: Monthly airline passengers (1949-1960)
**Functions**: 13+ plotting functions
**Topics**: Trend analysis, seasonal decomposition, forecasting, residual diagnostics

The most comprehensive example showing:
- Basic time series visualization
- Rolling statistics and smoothing
- Autocorrelation analysis (ACF, PACF)
- Frequency domain analysis (periodogram)
- Forecasting with prediction intervals
- Residual diagnostics

[View: examples/air_passengers.py](../../examples/air_passengers.py)

---

### 2. Sunspots - Smoothing & Frequency Analysis

**Dataset**: Monthly sunspot counts (1749-1983)
**Functions**: 7 plotting functions
**Topics**: Smoothing techniques, cyclic patterns, frequency analysis

Demonstrates:
- Exponential moving averages
- Rolling statistics with bands
- Periodogram for cycle detection
- Lag scatter plots

[View: examples/sunspots.py](../../examples/sunspots.py)

---

### 3. Victoria Electricity - Multivariate Analysis

**Dataset**: 30-minute electricity demand with temperature
**Functions**: 5 plotting functions
**Topics**: Multivariate visualization, cross-correlation, weather effects

Shows how to:
- Plot multiple time series together
- Analyze cross-correlation between variables
- Examine seasonal patterns with covariates

[View: examples/vic_electricity.py](../../examples/vic_electricity.py)

---

### 4. M4 Monthly - Panel Data Workflows

**Dataset**: 50 monthly time series from M4 Competition
**Functions**: 3 plotting functions
**Topics**: Panel data, filtering, pivoting, aggregation

Demonstrates:
- Working with panel/grouped time series
- Filtering and selecting subsets
- Pivoting to wide format for multivariate plots
- Aggregated statistics across series

[View: examples/m4_monthly.py](../../examples/m4_monthly.py)

---

### 5. Store Sales - Calendar Heatmaps

**Dataset**: Daily sales for 10 stores × 50 items
**Functions**: 4 plotting functions
**Topics**: Calendar visualization, retail patterns, panel data

Features:
- Calendar heatmap for daily patterns
- Store-level comparisons
- Seasonal analysis by time features

[View: examples/store_sales.py](../../examples/store_sales.py)

---

### 6. M4 Quarterly - Quarterly Seasonality

**Dataset**: 50 quarterly time series
**Functions**: 3 plotting functions
**Topics**: Quarterly patterns, year-over-year comparison

Covers:
- 4-quarter seasonal cycles
- Year-by-year comparison
- Quarterly aggregation

[View: examples/m4_quarterly.py](../../examples/m4_quarterly.py)

---

### 7. Walmart Sales - Retail with Covariates

**Dataset**: Transaction-level sales across 3 branches
**Functions**: 4 plotting functions
**Topics**: Transaction aggregation, product analysis, covariates

Shows:
- Aggregating transaction data to time series
- Branch performance comparison
- Product line and rating analysis

[View: examples/walmart_sales.py](../../examples/walmart_sales.py)

---

### 8. M4 Hourly - High-Frequency Patterns

**Dataset**: 50 hourly time series
**Functions**: 3 plotting functions
**Topics**: Intraday patterns, hour-of-day effects

Demonstrates:
- High-frequency visualization
- Hour-of-day seasonality
- Multiple timescale aggregations

[View: examples/m4_hourly.py](../../examples/m4_hourly.py)

---

### 9. Australian Tourism - Hierarchical Data

**Dataset**: Quarterly tourism by 76 regions, 8 states, 4 purposes
**Functions**: 3 plotting functions
**Topics**: Hierarchical aggregation, multi-level analysis

Features:
- Multi-level aggregation (region → state → national)
- Purpose-based segmentation
- Regional comparisons

[View: examples/australian_tourism.py](../../examples/australian_tourism.py)

---

### 10. ETTm1 - Multivariate Industrial IoT

**Dataset**: Electricity transformer temperature (7 features, 15-min intervals)
**Functions**: 4 plotting functions
**Topics**: Multivariate dependencies, cross-correlation, IoT

Covers:
- Multivariate feature visualization
- Cross-correlation analysis
- High-frequency industrial data patterns

[View: examples/ett_m1.py](../../examples/ett_m1.py)

---

## Running Examples

### Interactive Editing with Marimo

```bash
# Install marimo if needed
uv pip install marimo

# Open any example in interactive mode
marimo edit examples/air_passengers.py
```

### Run as Python Scripts

```bash
# Execute directly
python examples/air_passengers.py

# Or using uv
uv run python examples/air_passengers.py
```

### Testing Examples

Examples are tested as part of the CI pipeline:

```bash
# Test all examples
pytest tests/test_examples.py

# Test specific example
pytest tests/test_examples.py -k air_passengers
```

## Function Coverage

All 16 implemented plotting functions are demonstrated across the examples:

| Function | Examples |
|----------|----------|
| `plot_timeseries` | All 10 |
| `plot_rolling_statistics` | 3 |
| `plot_exponential_moving_average` | 2 |
| `plot_boxplot` | 7 |
| `plot_seasonality` | 7 |
| `plot_autocorrelation` | 2 |
| `plot_partial_autocorrelation` | 2 |
| `plot_correlation_diagnostics` | 2 |
| `plot_periodogram` | 2 |
| `plot_lag_scatter` | 2 |
| `plot_cross_correlation` | 2 |
| `plot_residuals` | 1 |
| `plot_forecast` | 1 |
| `plot_comparison` | 1 |
| `plot_prediction_interval` | 1 |
| `plot_calendar_heatmap` | 2 |

## Next Steps

- Browse the [API Reference](api-reference.md) for detailed function documentation
- Check the [User Guide](user-guide.md) for core concepts
- See [Getting Started](getting-started.md) for installation instructions
