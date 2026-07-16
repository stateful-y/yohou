# Seasonal Analysis

In this tutorial, we will confirm a seasonal pattern in the tourism dataset, measure its strength, and understand its shape using five diagnostic views: seasonal overlays, subseasonal boxplots, autocorrelation analysis, STL decomposition, and a seasonal heatmap.

<!-- COMPANION_NOTEBOOKS -->

## Prerequisites

- Completed [Exploratory Visualization](exploratory-visualization.md)

## 1. Load the Data

Continue with the same tourism dataset from the previous tutorial:

```python
from yohou.datasets import fetch_tourism_monthly

bunch = fetch_tourism_monthly(n_series=1)
y = bunch.frame
```

## 2. Does the Pattern Repeat Each Year?

[`plot_seasonality`](/pages/api/generated/yohou.plotting.plot_seasonality/) overlays each seasonal cycle on the same axis, with one line per year. This makes repeating shapes immediately visible:

```python
from yohou.plotting import plot_seasonality

fig = plot_seasonality(y, seasonality="month")
fig.show()
```

**What to look for:**

- **Lines that cluster tightly**: the seasonal pattern is strong and stable across years. A fixed seasonal component like [`PatternSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.PatternSeasonalityForecaster/) will capture this well.
- **Lines that diverge over time**: the pattern is evolving. Consider Fourier features or an adaptive seasonal model instead of fixed pattern extraction.
- **No consistent shape**: seasonality is weak or absent. A non-seasonal forecaster may be sufficient.

## 3. How Much Does Each Season Vary?

[`plot_subseasonality`](/pages/api/generated/yohou.plotting.plot_subseasonality/) breaks the series into seasonal positions and shows the distribution at each position:

```python
from yohou.plotting import plot_subseasonality

fig = plot_subseasonality(y, kind="box")
fig.show()
```

**What to look for:**

- **Tall boxes at specific months**: those months have high year-to-year variability. The seasonal component there is less predictable.
- **Boxes at clearly different levels**: confirms a strong seasonal effect. The median line traces the average seasonal shape.
- **Similar box heights and positions**: the seasonal pattern is weak and may not justify a dedicated seasonal component.

## 4. What Does the Correlation Structure Tell You?

The overlays and boxplots confirmed the seasonal pattern visually. Now use the autocorrelation function (ACF) and partial autocorrelation function (PACF) to understand how the seasonal dependence behaves, which informs your deseasonalization strategy.

[`plot_autocorrelation`](/pages/api/generated/yohou.plotting.plot_autocorrelation/) shows how correlated the series is with lagged versions of itself:

```python
from yohou.plotting import plot_autocorrelation

fig = plot_autocorrelation(y, max_lags=36)
fig.show()
```

[`plot_partial_autocorrelation`](/pages/api/generated/yohou.plotting.plot_partial_autocorrelation/) isolates the direct correlation at each lag by removing the influence of intermediate lags:

```python
from yohou.plotting import plot_partial_autocorrelation

fig = plot_partial_autocorrelation(y, max_lags=36, method="yw")  # Yule-Walker estimation
fig.show()
```

**What to look for:**

- **ACF spikes at lag 12 that decay slowly**: the seasonal component is persistent and non-stationary. Apply [`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.SeasonalDifferencing/) or use a [`DecompositionPipeline`](/pages/api/generated/yohou.compose.DecompositionPipeline/) that subtracts the seasonal component.
- **ACF spikes at lag 12 that decay quickly**: the seasonal effect is stationary. A [`PatternSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.PatternSeasonalityForecaster/) can extract it directly.
- **Significant PACF at both lags 1 and 12**: short-term autocorrelation and seasonal dependence coexist. A reduction forecaster with [`LagTransformer`](/pages/api/generated/yohou.preprocessing.LagTransformer/) features at both lags will capture both effects.

## 5. Can You Separate Trend, Season, and Residual?

STL (Seasonal and Trend decomposition using Loess) splits the series into three components. [`plot_decomposition`](/pages/api/generated/yohou.plotting.plot_decomposition/) shows each component in its own subplot:

```python
from yohou.plotting import plot_decomposition

fig = plot_decomposition(y, ["trend", "seasonal", "residual"], method="stl", period=12)
fig.show()
```

**What to look for:**

- **Clean seasonal subplot with a repeating pattern**: the seasonal shape is well-defined. This confirms the overlay findings from step 2.
- **Small, structureless residuals**: the decomposition captured most of the signal. A [`DecompositionPipeline`](/pages/api/generated/yohou.compose.DecompositionPipeline/) that models trend and seasonality separately will work well.
- **Residuals with remaining seasonal patterns**: the decomposition missed part of the signal. The seasonal shape may be changing over time, calling for a more flexible model.

## 6. How Does the Pattern Evolve Year Over Year?

[`plot_seasonal_heatmap`](/pages/api/generated/yohou.plotting.plot_seasonal_heatmap/) arranges values on a two-dimensional calendar grid, with rows for years and columns for months. Color intensity shows the value at each position:

```python
from yohou.plotting import plot_seasonal_heatmap

fig = plot_seasonal_heatmap(y, x_period="month", y_period="year")
fig.show()
```

**What to look for:**

- **Consistent color bands across rows**: the seasonal shape is stable from year to year.
- **Color intensity shifting over time** (e.g., bottom rows darker than top rows): the level is trending, which confirms the need for a trend component alongside the seasonal one.
- **Abrupt color changes in specific years**: possible structural breaks or regime changes. Consider truncating the history or adding a regime indicator.

## What You Built

We confirmed a 12-month seasonal cycle using five complementary diagnostics:

1. **Seasonal overlays**: the overall pattern shape and its stability across years
2. **Subseasonal boxplots**: the distribution and variability at each seasonal position
3. **ACF and PACF**: the correlation structure that informs your deseasonalization strategy
4. **STL decomposition**: a clean separation of trend, season, and residual
5. **Seasonal heatmap**: year-over-year evolution of the pattern

These findings directly inform model selection: strong fixed seasonality points to [`PatternSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.PatternSeasonalityForecaster/), while evolving patterns call for Fourier features or adaptive approaches.

## Next Steps

- [Time Series Patterns](../explanation/time-series-patterns.md) for translating visual observations into modeling decisions
- [Decomposition](decomposition.md) for building forecasters that explicitly model trend and seasonality
- [How to Handle Complex Seasonality](../how-to/handle-complex-seasonality.md) for practical strategies with multiple seasonal periods
