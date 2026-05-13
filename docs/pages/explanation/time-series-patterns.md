# Time Series Patterns

Before choosing a forecasting method or a stationarity transform, you need to understand what kind of structure lives in the data. Time series patterns fall into a small number of categories, each with distinct implications for how yohou components should be configured. Getting this classification right is the single most important step in the forecasting workflow, because it determines whether you need decomposition, differencing, feature engineering, or simply a seasonal naive baseline.

## Trend

A trend is a long-term increase or decrease in the level of the series. It does not have to be linear: retail sales might grow exponentially, a population might follow logistic saturation, or an economic indicator might alternate between periods of growth and contraction. What matters is that the mean of the series changes slowly over time.

Trend has direct consequences for yohou configuration. A series with a rising trend violates the stationarity assumption that reduction forecasters rely on. You can address this with [`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/) (which removes the trend by subtracting lagged values), or by fitting an explicit trend model inside a [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) using [`PolynomialTrendForecaster`](/pages/api/generated/yohou.stationarity.trend.PolynomialTrendForecaster/).

## Seasonality

Seasonality refers to periodic fluctuations that repeat at a known, fixed frequency: daily patterns in electricity demand, weekly patterns in supermarket sales, annual patterns in tourism. The key distinction from trends is that seasonal patterns have a fixed and known period. The amplitude may be constant (additive seasonality) or proportional to the series level (multiplicative seasonality).

Additive seasonality means the seasonal swings stay roughly the same magnitude regardless of the overall level. If January sales are always about 200 units above the annual average, that is additive. Multiplicative seasonality means the swings scale with the level: if January sales are always about 15% above the annual average, the absolute swing grows as total sales grow.

This distinction matters for transform selection. Additive seasonality is handled directly by [`PatternSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.PatternSeasonalityForecaster/) or [`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/). Multiplicative seasonality requires a log transform first (via [`LogTransformer`](/pages/api/generated/yohou.stationarity.transformers.LogTransformer/) or [`SeasonalLogDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalLogDifferencing/)) to convert the multiplicative relationship into an additive one in log-space.

## Cycles

Cycles are rises and falls that are not of a fixed period. Business cycles, for example, might last anywhere from two to ten years. Unlike seasonality, cycles have variable length and are not tied to a calendar frequency. In practice, cycles are difficult to forecast because their period is unknown, and most time series datasets are too short to contain multiple complete cycles.

Yohou's reduction approach handles cycles implicitly: lagged features capture recent momentum, and the regressor learns associations between recent patterns and future values. There is no dedicated "cycle forecaster" because the variable period makes explicit modeling unreliable for most practical horizons.

## Autocorrelation

Autocorrelation measures how strongly a series correlates with its own lagged values. A series with high autocorrelation at lag 1 means that today's value is closely related to yesterday's. High autocorrelation at lag 12 (for monthly data) suggests an annual seasonal pattern.

The autocorrelation function (ACF) and partial autocorrelation function (PACF) are the primary diagnostic tools for identifying these relationships. In yohou, [`plot_autocorrelation`](/pages/api/generated/yohou.plotting.diagnostics.plot_autocorrelation/) and [`plot_partial_autocorrelation`](/pages/api/generated/yohou.plotting.diagnostics.plot_partial_autocorrelation/) produce these plots:

- **ACF**: Shows total correlation at each lag (including indirect effects through intermediate lags). A slowly decaying ACF suggests a trend; sharp spikes at seasonal lags suggest seasonality.
- **PACF**: Shows the direct correlation at each lag after removing the effect of shorter lags. A significant spike at lag `k` with no significant spikes at shorter lags suggests that a lag-`k` feature would be informative for a reduction forecaster.

These plots guide feature engineering decisions. If the PACF shows significant spikes at lags 1, 7, and 14, those are strong candidates for [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/) or window-based features.

## White Noise

White noise is a series where all values are independently and identically distributed with zero autocorrelation at all lags. It looks like random scatter with no discernible pattern. White noise is the ideal residual after a good model has captured all the structure in the data.

In practice, you check whether a series (or model residuals) resembles white noise by examining the ACF plot: if all autocorrelation values fall within the significance bounds (the dashed lines in [`plot_autocorrelation`](/pages/api/generated/yohou.plotting.diagnostics.plot_autocorrelation/)), the series is consistent with white noise. This is the target state after applying stationarity transforms and fitting a forecaster. If significant autocorrelation remains in the residuals, the model is leaving predictable structure on the table.

## Identifying Patterns Visually

The starting point for any forecasting project is visual inspection. Yohou's plotting
module provides the essential diagnostic toolkit:

- [`plot_time_series`](/pages/api/generated/yohou.plotting.exploration.plot_time_series/): The raw series. Look for trends (upward or downward drift), seasonal patterns (repeating waves), and obvious anomalies (sudden jumps or outliers).
- [`plot_seasonality`](/pages/api/generated/yohou.plotting.diagnostics.plot_seasonality/): Overlays multiple seasonal cycles to reveal the shape and stability of the seasonal pattern.
- [`plot_autocorrelation`](/pages/api/generated/yohou.plotting.diagnostics.plot_autocorrelation/): Identifies the lag structure. Slowly decaying ACF means trend; spikes at seasonal lags mean seasonality.
- [`plot_spectrum`](/pages/api/generated/yohou.plotting.signal.plot_spectrum/): Frequency-domain view that reveals dominant periodicities. Peaks correspond to seasonal frequencies.
- [`plot_lag_scatter`](/pages/api/generated/yohou.plotting.diagnostics.plot_lag_scatter/): Scatter plots of `y(t)` vs `y(t-k)` for various lags. Strong linear patterns indicate useful lags for feature engineering.
- [`plot_boxplot`](/pages/api/generated/yohou.plotting.exploration.plot_boxplot/): Distribution summaries grouped by time period. Useful for spotting seasonal variation and outliers.

The combination of these views gives a fairly complete picture of the data's structure before you write a single line of modeling code.

For guidance on translating these patterns into yohou component choices, see [Choose a Forecasting Method](../how-to/choose-forecasting-method.md).

## References

- Hyndman, R.J. & Athanasopoulos, G. (2021). [Forecasting: principles and practice](https://otexts.com/fpp3/), 3rd edition, OTexts. Chapters 2 and 4.

## Connections

The patterns identified here inform transform selection in
[Stationarity](stationarity.md) and forecaster configuration in
[Forecasting](forecasting.md). For transforming raw features (lags, rolling windows,
scaling), see [Preprocessing](preprocessing.md). For checking whether a model has
captured the patterns, see [Residual Diagnostics](residual-diagnostics.md).

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Categorical Patterns

Categorical time series exhibit patterns distinct from numeric series:

- **Transition frequencies**: How often the series switches between classes. High
  transition rates suggest a volatile process; low rates suggest regime persistence.
- **Class persistence**: The tendency for a class to repeat across consecutive
  timesteps. Many real-world categorical series are "sticky" (weather, equipment
  state, demand level).
- **Seasonal class patterns**: Some classes may be more likely during certain
  seasons (e.g., "High" demand in summer, "Low" in spring).
