# Exploratory Visualization

In this tutorial, we will explore a tourism dataset through eight diagnostic questions that guide every downstream modeling decision. Along the way, we will check for completeness, trends, variance stability, seasonality, seasonal evolution, outliers, and the right modeling frequency.

## Prerequisites

- Completed [Getting Started](getting-started.md)

<!-- COMPANION_NOTEBOOKS -->

## 1. Get a First Look

Start by plotting the raw series with [`plot_time_series`](/pages/api/generated/yohou.plotting.plot_time_series/). This gives you an initial impression of length, frequency, scale, and any obvious patterns:

```python
from yohou.datasets import fetch_tourism_monthly
from yohou.plotting import plot_time_series

bunch = fetch_tourism_monthly(n_series=1)
y = bunch.frame

fig = plot_time_series(y)
fig.show()
```

Scan the plot and ask yourself: does the level stay roughly constant, or does it drift upward or downward? Do you see repeating peaks and troughs? Are there sudden jumps or gaps? Keep those first impressions in mind as you move through the diagnostics below.

## 2. Are There Gaps in the Data?

Missing values will propagate silently through pipelines if left unchecked, so check for them early. [`plot_missing_data`](/pages/api/generated/yohou.plotting.plot_missing_data/) visualizes gaps as a heatmap, bar chart, or matrix:

```python
from yohou.plotting import plot_missing_data

fig = plot_missing_data(y, kind="heatmap")
fig.show()
```

**What to look for:**

- **Scattered missing values** (isolated white cells): usually safe to interpolate with [`SimpleImputer`](/pages/api/generated/yohou.preprocessing.SimpleImputer/) or [`SeasonalImputer`](/pages/api/generated/yohou.preprocessing.SeasonalImputer/).
- **Long contiguous blocks**: may require truncating the series before the gap, or using a model that handles missing data natively.
- **Entire columns missing**: indicates a data pipeline issue, not a modeling problem.

Switch to `kind="bars"` for a per-column summary or `kind="matrix"` for a row-level view. Once you know where the gaps are, see [Handle Missing Data](../how-to/handle-missing-data.md) for imputation strategies.

## 3. Is There a Trend, and Is the Variance Stable?

[`plot_rolling_statistics`](/pages/api/generated/yohou.plotting.plot_rolling_statistics/) overlays a rolling mean and standard deviation on the raw series. The rolling mean reveals trends; the rolling standard deviation reveals volatility changes:

```python
from yohou.plotting import plot_rolling_statistics

fig = plot_rolling_statistics(y, window_size=12, statistics=["mean", "std"])
fig.show()
```

**What to look for:**

- **Rising or falling rolling mean**: a trend is present. You will likely need a [`DecompositionPipeline`](/pages/api/generated/yohou.compose.DecompositionPipeline/) with a trend forecaster, or differencing via [`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.SeasonalDifferencing/).
- **Stable rolling mean but repeating peaks**: seasonality without trend. A simpler pipeline with just a seasonality component may suffice.
- **Widening rolling standard deviation**: the variance grows with the level (heteroscedasticity). Apply a [`LogTransformer`](/pages/api/generated/yohou.stationarity.LogTransformer/) or [`BoxCoxTransformer`](/pages/api/generated/yohou.stationarity.BoxCoxTransformer/) before modeling to stabilize it.
- **Constant standard deviation**: no variance stabilization needed.

## 4. Is There a Seasonal Pattern?

[`plot_seasonality`](/pages/api/generated/yohou.plotting.plot_seasonality/) overlays one line per cycle (e.g., one line per year) on the same seasonal axis (e.g., months 1 through 12). This makes it easy to see whether the same months are consistently high or low across years:

```python
from yohou.plotting import plot_seasonality

fig = plot_seasonality(y, seasonality="month")
fig.show()
```

**What to look for:**

- **Lines that follow the same shape each year**: a strong, stable seasonal pattern. A [`DecompositionPipeline`](/pages/api/generated/yohou.compose.DecompositionPipeline/) with a seasonal component will capture this well.
- **Lines that diverge over time** (recent years higher or lower than earlier years): seasonality combined with a trend. You will need both a trend and a seasonality component.
- **No consistent shape across years**: seasonality is weak or absent. A non-seasonal forecaster may be sufficient.

Use the `highlight` parameter to draw attention to specific years, for example `highlight=[2018, 2019]` to compare two recent periods side by side. Change `seasonality` to `"quarter"` or `"weekday"` if your data operates at a different frequency.

## 5. Does the Seasonal Shape Change Over Time?

[`plot_subseasonality`](/pages/api/generated/yohou.plotting.plot_subseasonality/) creates one mini subplot per season (e.g., 12 subplots for months). Within each subplot the values for that season across all years are connected chronologically, with an optional horizontal mean line:

```python
from yohou.plotting import plot_subseasonality

fig = plot_subseasonality(y, seasonality="month", kind="mean")
fig.show()
```

**What to look for:**

- **Flat lines within each subplot**: the seasonal level is stable over time. A fixed seasonal component is appropriate.
- **Upward or downward trends within subplots**: the seasonal effect is changing. Consider a model that allows the seasonal component to evolve, or apply [`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.SeasonalDifferencing/).
- **Different slopes across subplots** (e.g., summer months trending up but winter months flat): the seasonal shape itself is shifting. This calls for a more flexible seasonal model.

Switch to `kind="box"` or `kind="violin"` to see the full distribution within each season rather than just the trajectory.

## 6. Does the Distribution Change Over Time?

[`plot_boxplot`](/pages/api/generated/yohou.plotting.plot_boxplot/) groups values by a calendar period to reveal how the distribution shifts:

```python
from yohou.plotting import plot_boxplot

fig = plot_boxplot(y, period="1y")
fig.show()
```

**What to look for:**

- **Boxes shifting upward over time**: confirms the trend you saw in the rolling mean.
- **Growing box sizes**: confirms heteroscedasticity; reinforces the need for variance stabilization.
- **Consistent box positions and sizes**: the series is stationary in level and spread, which is the ideal starting point for most forecasters.

Try changing `period` to `"1q"` to see quarterly distributions, which may reveal finer seasonal structure.

## 7. Are There Outliers or Genuine Business Events?

[`plot_outliers`](/pages/api/generated/yohou.plotting.plot_outliers/) highlights observations that exceed a statistical threshold. Choose from `"zscore"`, `"iqr"`, or `"percentile"` methods:

```python
from yohou.plotting import plot_outliers

fig = plot_outliers(y, method="zscore")
fig.show()
```

**What to look for:**

- **Isolated spikes at random times**: likely data errors. Consider clipping or replacing them with [`OutlierThresholdHandler`](/pages/api/generated/yohou.preprocessing.OutlierThresholdHandler/) before modeling.
- **Outliers clustering at the same calendar period each year** (e.g., December spikes in tourism): these are genuine business events, not errors. Keep them and consider adding holiday features instead of removing the values.
- **A sudden level shift that persists**: this is a structural break, not an outlier. You may need to truncate the history before the break or add a regime indicator.

The z-score method flags observations more than 3 standard deviations from the mean by default (configurable via the `threshold` parameter). See [Handle Outliers](../how-to/handle-outliers.md) for pipeline integration.

## 8. What Frequency Should You Model At?

[`plot_resampling_comparison`](/pages/api/generated/yohou.plotting.plot_resampling_comparison/) shows the same series at different temporal resolutions side by side:

```python
from yohou.plotting import plot_resampling_comparison
from yohou.preprocessing import Downsampler

downsampler = Downsampler(interval="1q")
y_quarterly = downsampler.fit_transform(y)

fig = plot_resampling_comparison(y, y_quarterly)
fig.show()
```

**What to look for:**

- **Too fine a frequency** (e.g., daily when the signal is monthly): the plot becomes noisy and dominated by measurement error. Model at a coarser frequency or use [`Downsampler`](/pages/api/generated/yohou.preprocessing.Downsampler/) to aggregate.
- **Too coarse** (e.g., yearly when seasonal patterns are monthly): important patterns disappear. Keep the finer frequency.
- **The "sweet spot"**: the resolution where patterns are clearly visible but noise is manageable. This is your modeling frequency.

## What You Built

We answered eight diagnostic questions about a raw time series:

1. **First look**: overall length, frequency, scale, and visible patterns
2. **Data completeness**: where gaps exist and how to handle them
3. **Trend and variance stability**: whether detrending or variance stabilization is needed
4. **Seasonal pattern**: whether a repeating seasonal cycle is present
5. **Seasonal evolution**: whether the seasonal shape is stable or changing over time
6. **Distribution evolution**: whether the series is stationary in level and spread
7. **Outliers vs. events**: whether extreme values should be removed or preserved
8. **Modeling frequency**: the right temporal resolution for your forecaster

These answers directly inform which preprocessing transforms to apply and which forecaster architecture to choose in subsequent tutorials.

The companion notebook also covers [`plot_distribution`](/pages/api/generated/yohou.plotting.plot_distribution/), which summarizes the overall value distribution as a histogram with an optional density curve.

## Next Steps

- [Seasonal Analysis](seasonal-analysis.md) for deeper seasonal diagnostics with ACF/PACF and heatmaps
- [Visualization](../explanation/visualization.md) for the conceptual overview of the plotting module
