# Time Series Patterns

Time series patterns fall into a small number of categories, each with distinct implications for model and transform selection. Understanding this taxonomy is the prerequisite for every step in the forecasting workflow, from deciding whether to difference a series to choosing which feature transformer to apply. Classifying the structure that lives in the data is what determines whether a problem calls for decomposition, differencing, feature engineering, or simply a seasonal naive baseline.

## Trend

A trend is a long-term increase or decrease in the level of the series. It does not have to be linear: retail sales might grow exponentially, a population might follow logistic saturation, or an economic indicator might alternate between periods of growth and contraction. What matters is that the mean of the series changes slowly over time.

A series with a rising or falling trend violates the constant-mean assumption that reduction forecasters rely on. You can address this with [`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/) (which removes the trend by subtracting lagged values), or by fitting an explicit trend model inside a [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) using [`PolynomialTrendForecaster`](/pages/api/generated/yohou.stationarity.trend.PolynomialTrendForecaster/). The decomposition approach is generally preferred when you need the trend component itself (for example, to report the underlying growth rate), while differencing is simpler and often sufficient when you only care about the forecast.

[`plot_rolling_statistics`](/pages/api/generated/yohou.plotting.exploration.plot_rolling_statistics/) makes trend visible by smoothing out short-term fluctuations, and [`plot_decomposition`](/pages/api/generated/yohou.plotting.forecasting.plot_decomposition/) separates it from seasonal and residual components after fitting a `DecompositionPipeline`.

## Seasonality

Seasonality refers to periodic fluctuations that repeat at a known, fixed frequency: daily patterns in electricity demand, weekly patterns in supermarket sales, annual patterns in tourism. The key distinction from trends is that seasonal patterns have a fixed and known period. The amplitude may be constant (additive seasonality) or proportional to the series level (multiplicative seasonality).

Additive seasonality means the seasonal swings stay roughly the same magnitude regardless of the overall level. If January sales are always about 200 units above the annual average, that is additive. Multiplicative seasonality means the swings scale with the level: if January sales are always about 15% above the annual average, the absolute swing grows as total sales grow.

This distinction matters for transform selection. Additive seasonality is handled directly by [`PatternSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.PatternSeasonalityForecaster/) or [`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/). Multiplicative seasonality requires a log transform first (via [`LogTransformer`](/pages/api/generated/yohou.stationarity.transformers.LogTransformer/) or [`SeasonalLogDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalLogDifferencing/)) to convert the multiplicative relationship into an additive one in log-space.

### Multiple Seasonality

Many real-world series exhibit more than one seasonal pattern simultaneously. Electricity demand, for example, has daily cycles (usage peaks in the evening), weekly cycles (weekdays differ from weekends), and annual cycles (summer cooling, winter heating). Retail sales commonly combine weekly and annual patterns.

`PatternSeasonalityForecaster` captures a single seasonal frequency per component, so handling multiple seasons requires either nesting multiple `DecompositionPipeline` stages (removing one pattern at a time) or using [`FourierSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.FourierSeasonalityForecaster/), which represents seasonality as a sum of sine and cosine terms and can capture multiple frequencies in a single component. For reduction forecasters, [`FourierFeatureTransformer`](/pages/api/generated/yohou.preprocessing.time_features.FourierFeatureTransformer/) adds Fourier terms as exogenous features, letting the regressor learn the seasonal shape from data.

[`plot_seasonality`](/pages/api/generated/yohou.plotting.diagnostics.plot_seasonality/) overlays multiple seasonal cycles to reveal the primary seasonal shape, while [`plot_subseasonality`](/pages/api/generated/yohou.plotting.diagnostics.plot_subseasonality/) helps identify finer patterns within a broader seasonal period (e.g., day-of-week effects within a monthly cycle). [`plot_spectrum`](/pages/api/generated/yohou.plotting.signal.plot_spectrum/) provides a frequency-domain view where each dominant periodicity appears as a peak, making it straightforward to count the number of active seasonal frequencies. For practical guidance on configuring these components, see [How to Handle Complex Seasonality](../how-to/handle-complex-seasonality.md).

## Cycles

Cycles are rises and falls that are not of a fixed period. Business cycles, for example, might last anywhere from two to ten years. Unlike seasonality, cycles have variable length and are not tied to a calendar frequency. In practice, cycles are difficult to forecast because their period is unknown, and most time series datasets are too short to contain multiple complete cycles.

Yohou's reduction approach handles cycles implicitly: lagged features capture recent momentum, and the regressor learns associations between recent patterns and future values. There is no dedicated "cycle forecaster" because the variable period makes explicit modeling unreliable for most practical horizons.

## Structural Breaks

A structural break is an abrupt, permanent change in the statistical properties of the series: a sudden jump in level (the series shifts up or down), a change in trend slope, or a shift in seasonal amplitude. Common causes include policy changes, market disruptions, equipment replacements, and pandemics.

Structural breaks differ from trends in that the shift is sudden rather than gradual, and they differ from outliers in that the change persists. A one-time price increase that permanently raises the series level is a structural break; a single-day spike that reverts is an outlier.

Breaks pose a particular challenge for forecasters trained on historical data, because observations before the break may no longer be representative. The most practical approaches are truncating the training data to start after the break (if the break is recent and identifiable) or letting the regressor adapt through recent-lag features that naturally down-weight old patterns. [`plot_time_series`](/pages/api/generated/yohou.plotting.exploration.plot_time_series/) and [`plot_rolling_statistics`](/pages/api/generated/yohou.plotting.exploration.plot_rolling_statistics/) are the primary tools for spotting breaks visually: a sudden jump in the rolling mean or a sharp discontinuity in the raw series both signal a structural break.

## Changing Variance

Many time series exhibit variance that changes over time: the fluctuations grow larger as the level increases, or volatility clusters during certain periods. This pattern, called heteroscedasticity, has two practical consequences. First, reduction forecasters trained on squared-error loss will over-weight high-variance periods, biasing the model. Second, prediction intervals that assume constant variance will be too narrow during volatile periods and too wide during calm ones.

The most common case is variance that scales with the level (e.g., the absolute size of seasonal swings grows as sales increase). This is closely related to multiplicative seasonality, and the same log transform that linearizes multiplicative seasonality also stabilizes the variance. [`LogTransformer`](/pages/api/generated/yohou.stationarity.transformers.LogTransformer/) works for strictly positive series. For series with zeros or near-zero values, [`BoxCoxTransformer`](/pages/api/generated/yohou.stationarity.transformers.BoxCoxTransformer/) offers a more flexible power transform, and [`ASinhTransformer`](/pages/api/generated/yohou.stationarity.transformers.ASinhTransformer/) handles series that include zeros or negative values.

You can check for changing variance by inspecting [`plot_rolling_statistics`](/pages/api/generated/yohou.plotting.exploration.plot_rolling_statistics/), which shows the rolling standard deviation alongside the rolling mean. If the rolling standard deviation tracks the rolling mean, a variance-stabilizing transform is appropriate.

## Autocorrelation

Autocorrelation measures how strongly a series correlates with its own lagged values. A series with high autocorrelation at lag 1 means that today's value is closely related to yesterday's. High autocorrelation at lag 12 (for monthly data) suggests an annual seasonal pattern.

The autocorrelation function (ACF) and partial autocorrelation function (PACF) are the primary diagnostic tools for identifying these relationships. In yohou, [`plot_autocorrelation`](/pages/api/generated/yohou.plotting.diagnostics.plot_autocorrelation/) and [`plot_partial_autocorrelation`](/pages/api/generated/yohou.plotting.diagnostics.plot_partial_autocorrelation/) produce these plots:

- **ACF**: Shows total correlation at each lag (including indirect effects through intermediate lags). A slowly decaying ACF suggests a trend; sharp spikes at seasonal lags suggest seasonality.
- **PACF**: Shows the direct correlation at each lag after removing the effect of shorter lags. A significant spike at lag `k` with no significant spikes at shorter lags suggests that a lag-`k` feature would be informative for a reduction forecaster.

These plots guide feature engineering decisions. If the PACF shows significant spikes at lags 1, 7, and 14, those are strong candidates for [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/) or window-based features like [`RollingStatisticsTransformer`](/pages/api/generated/yohou.preprocessing.window.RollingStatisticsTransformer/).

## White Noise

White noise is a series where all values are independently and identically distributed with zero autocorrelation at all lags. It looks like random scatter with no discernible pattern. White noise represents the forecasting floor: no model can extract predictive signal from a white noise process, and the best possible forecast is simply the mean.

In practice, you check whether a series (or model residuals) resembles white noise by examining the ACF plot: if all autocorrelation values fall within the significance bounds (the dashed lines in [`plot_autocorrelation`](/pages/api/generated/yohou.plotting.diagnostics.plot_autocorrelation/)), the series is consistent with white noise. This is the target state for model residuals after applying stationarity transforms and fitting a forecaster. If significant autocorrelation remains in the residuals, the model is leaving predictable structure on the table. For a detailed treatment of residual evaluation, see [Residual Diagnostics](residual-diagnostics.md).

## Categorical Patterns

When the target variable takes discrete values (e.g., demand level as Low/Medium/High, equipment state as Running/Idle/Fault, weather condition as Clear/Rain/Snow), the relevant patterns differ from those in numeric series.

- **Transition frequencies**: How often the series switches between classes. High transition rates suggest a volatile process; low rates suggest regime persistence. A machine that rarely transitions from Running to Fault has a different forecasting challenge (rare-event detection) than one that switches state hourly.
- **Class persistence**: The tendency for a class to repeat across consecutive timesteps. Many real-world categorical series are "sticky": tomorrow's weather condition is strongly predicted by today's condition, and demand levels persist for days before shifting. High persistence means lag features are highly informative.
- **Seasonal class patterns**: Some classes may be more likely during certain seasons (e.g., "High" demand in summer, "Low" in spring). This is the categorical analogue of numeric seasonality.
- **Class imbalance over time**: If one class dominates the series (e.g., 90% "Normal", 10% "Alert"), the forecaster must be evaluated with proper scoring rules rather than simple accuracy. See [Class-Probability Forecasting](class-probability-forecasting.md) for the evaluation approach.

## Identifying Patterns Visually

The starting point for any forecasting project is visual inspection. Yohou's plotting module provides the essential diagnostic toolkit:

- [`plot_time_series`](/pages/api/generated/yohou.plotting.exploration.plot_time_series/): The raw series. Look for trends (upward or downward drift), seasonal patterns (repeating waves), structural breaks (sudden jumps), and obvious anomalies (spikes or outliers).
- [`plot_rolling_statistics`](/pages/api/generated/yohou.plotting.exploration.plot_rolling_statistics/): Rolling mean and standard deviation over time. A drifting mean reveals trend; a changing standard deviation reveals heteroscedasticity.
- [`plot_seasonality`](/pages/api/generated/yohou.plotting.diagnostics.plot_seasonality/): Overlays multiple seasonal cycles to reveal the shape and stability of the seasonal pattern.
- [`plot_seasonal_heatmap`](/pages/api/generated/yohou.plotting.diagnostics.plot_seasonal_heatmap/): A grid view of values indexed by season and year, making it easy to spot both seasonal shapes and year-over-year changes.
- [`plot_autocorrelation`](/pages/api/generated/yohou.plotting.diagnostics.plot_autocorrelation/): Identifies the lag structure. Slowly decaying ACF means trend; spikes at seasonal lags mean seasonality.
- [`plot_spectrum`](/pages/api/generated/yohou.plotting.signal.plot_spectrum/): Frequency-domain view that reveals dominant periodicities. Peaks correspond to seasonal frequencies; multiple peaks indicate multiple seasonality.
- [`plot_lag_scatter`](/pages/api/generated/yohou.plotting.diagnostics.plot_lag_scatter/): Scatter plots of `y(t)` vs `y(t-k)` for various lags. Strong linear patterns indicate useful lags for feature engineering.
- [`plot_boxplot`](/pages/api/generated/yohou.plotting.exploration.plot_boxplot/): Distribution summaries grouped by time period. Useful for spotting seasonal variation and outliers.
- [`plot_decomposition`](/pages/api/generated/yohou.plotting.forecasting.plot_decomposition/): After fitting a `DecompositionPipeline`, visualizes the extracted trend, seasonal, and residual components side by side.

The combination of these views gives a complete picture of the data's structure before you write a single line of modeling code.

For guidance on translating these patterns into yohou component choices, see [Choose a Forecasting Method](../how-to/choose-forecasting-method.md).

## References

- Hyndman, R.J. & Athanasopoulos, G. (2021). [Forecasting: principles and practice](https://otexts.com/fpp3/), 3rd edition, OTexts. Chapters 2 and 4.

## Connections

The patterns identified here inform transform selection in [Stationarity](stationarity.md) and forecaster configuration in [Reduction Forecasting](reduction-forecasting.md). For transforming raw features (lags, rolling windows, scaling), see [Preprocessing](preprocessing.md). For checking whether a model has captured the patterns, see [Residual Diagnostics](residual-diagnostics.md).

For practical recipes, see [How to Handle Complex Seasonality](../how-to/handle-complex-seasonality.md) and [How to Apply Stationarity Transforms](../how-to/apply-stationarity-transforms.md).
