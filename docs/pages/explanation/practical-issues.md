# Practical Issues

Textbook forecasting assumes clean data, a single seasonal period, and a series long enough to learn from. Production forecasting rarely offers these luxuries. This page covers the messy realities that arise in practice and how yohou's components address them.

## Complex Seasonality

Many real-world series exhibit multiple seasonal patterns operating simultaneously. Daily electricity demand has an intra-day pattern (peak during business hours), a weekly pattern (lower on weekends), and an annual pattern (air conditioning in summer). Sub-daily data can have even more layers: a 5-minute-resolution series might show patterns at the hourly, daily, weekly, and annual scales.

Standard seasonal differencing handles one period at a time. For complex seasonality, there are several approaches in yohou:

- **Fourier terms**: [`FourierSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.FourierSeasonalityForecaster/) can include harmonics at multiple frequencies in a single model. This is the most flexible approach and avoids the data loss from repeated differencing.
- **Nested decomposition**: A [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) with multiple seasonality forecasters, each targeting a different period. The pipeline fits them sequentially on successive residuals.
- **Feature engineering**: Encode each seasonal pattern as exogenous features (day-of-week dummies, month indicators, Fourier pairs) and let the reduction forecaster learn the combined effect.

The right approach depends on the data volume. Fourier terms work well when periods are known and the seasonal shape is smooth. Pattern-based decomposition works better when the seasonal shape is irregular but you have enough complete cycles to estimate it.

## Missing Values

Time series data frequently has gaps: sensor outages, holidays when markets close, or irregular reporting schedules. Missing values break the temporal continuity that forecasters rely on, and different components handle them differently.

Yohou provides several imputation transformers suited to temporal data.
[`SimpleTimeImputer`](/pages/api/generated/yohou.preprocessing.imputation.SimpleTimeImputer/)
fills gaps using strategies like forward-fill (which preserves jump discontinuities)
or linear interpolation (which is better for smooth series).
[`SeasonalImputer`](/pages/api/generated/yohou.preprocessing.imputation.SeasonalImputer/)
accounts for periodic patterns when filling gaps, and
[`TransformedSpaceKNNImputer`](/pages/api/generated/yohou.preprocessing.imputation.TransformedSpaceKNNImputer/)
uses nearest-neighbor imputation in a transformed feature space. Custom imputation
logic can be implemented via
[`FunctionTransformer`](/pages/api/generated/yohou.preprocessing.function.FunctionTransformer/).

The key decision is where imputation happens in the pipeline. Imputing before stationarity transforms ensures the differencing and decomposition algorithms see complete data. Imputing after may introduce artifacts if the imputation method does not account for trends or seasonality.

## Outliers

Outliers in time series can be genuine (a flash crash, a once-in-a-decade storm) or spurious (a data entry error, a sensor malfunction). The distinction matters because genuine outliers carry information and spurious ones add noise.

[`OutlierThresholdHandler`](/pages/api/generated/yohou.preprocessing.outlier.OutlierThresholdHandler/)
and
[`OutlierPercentileHandler`](/pages/api/generated/yohou.preprocessing.outlier.OutlierPercentileHandler/)
detect and handle outliers in the preprocessing stage. Typical strategies include
clipping to a threshold or percentile range, replacing with interpolated values, or
flagging outliers as a binary feature that the regressor can learn from.

For conformal prediction, outliers in the calibration set affect interval width. A single extreme conformity score from an outlier can widen all prediction intervals. Using [`GammaResidual`](/pages/api/generated/yohou.metrics.conformity.GammaResidual/) (which normalizes by prediction magnitude) can mitigate this when outliers correlate with the level of the series.

## Forecast Combinations

Combining forecasts from multiple models often outperforms any single model. This is one of the most reliable findings in forecasting research: even a simple average of two decent models typically beats either model alone, because different models make different errors that partially cancel.

Yohou supports combinations through composition. A
[`ColumnForecaster`](/pages/api/generated/yohou.compose.column_forecaster.ColumnForecaster/)
can assign different forecasters to different target columns. For combining
predictions from multiple forecasters on the same target, fit multiple forecasters
independently, collect their predictions, and average (or apply other combination
logic) using polars operations or a
[`FunctionTransformer`](/pages/api/generated/yohou.preprocessing.function.FunctionTransformer/).

Weighted combinations, where models receive weights proportional to their recent
accuracy, can be implemented using cross-validation scores from
[`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/)
or custom scoring logic.

## Short Series

Short series present a fundamental challenge: there may not be enough data to estimate seasonal patterns, validate via cross-validation, or calibrate conformal intervals. Some guidelines:

- **Too short for seasonal decomposition**: If the series has fewer than 2-3 complete seasonal cycles, pattern-based seasonality estimation is unreliable. Consider using Fourier terms (which need fewer observations to fit) or treat the series as non-seasonal.
- **Too short for cross-validation**: Standard expanding-window CV requires enough data for a meaningful training set, validation set, and multiple splits. With very short series, leave-one-out approaches or simple train/test splits may be all that is feasible.
- **Too short for scaled metrics**: MASE and RMSSE require enough training data to estimate the naive baseline error. With fewer observations than the seasonal period, these metrics cannot be computed.

In many panel data scenarios, individual series may be short even though the aggregate dataset is large. Yohou's panel strategies (global, multivariate) address this by pooling information across groups: the regressor trains on all series simultaneously, learning shared patterns that transfer to series with limited individual history.

## Very Long Series

At the other extreme, very long series can cause problems of a different kind. Patterns from decades ago may be irrelevant to today's dynamics: a model trained on 1990s retail data includes shopping patterns before e-commerce existed. The `observation_horizon` property in yohou determines how much history each forecaster retains in its sliding window (see [Core Concepts](core-concepts.md#observation-horizon)). Configuring transformers with shorter memory windows prevents ancient data from diluting the regressor's focus on current patterns.

Time-weighted evaluation is another approach:
[`exponential_decay_weight`](/pages/api/generated/yohou.utils.weighting.exponential_decay_weight/)
and
[`linear_decay_weight`](/pages/api/generated/yohou.utils.weighting.linear_decay_weight/)
give more importance to recent errors during model selection, naturally favoring models
that track current dynamics.

## Resampling

Data may arrive at a frequency that does not match the forecast requirement. Hourly data when you need daily forecasts, or irregular timestamps that need alignment to a regular grid.
[`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/)
handles frequency reduction with configurable aggregation (mean, sum, first, last).
[`Upsampler`](/pages/api/generated/yohou.preprocessing.resampling.Upsampler/)
handles frequency increases, though upsampling is generally discouraged in forecasting
because it creates artificial data points without genuine information content. If a
higher-frequency forecast is needed, it is usually better to obtain higher-frequency
input data rather than interpolate lower-frequency data.

## References

- Hyndman, R.J. & Athanasopoulos, G. (2021). [Forecasting: principles and practice](https://otexts.com/fpp3/), 3rd edition, OTexts. Chapter 13.
- Bates, J.M. & Granger, C.W.J. (1969). "The combination of forecasts." Operational Research Quarterly, 20(4), 451-468.

## Connections

Missing value and outlier handling are part of the [Preprocessing](preprocessing.md) pipeline. Complex seasonality ties into [Stationarity](stationarity.md) transforms and decomposition. Forecast combinations relate to the composition patterns in [Composition](composition.md). For evaluating whether a model is adequate given these practical challenges, see [Forecast Accuracy](forecast-accuracy.md) and [Residual Diagnostics](residual-diagnostics.md).

Practical examples: [Resampling](/examples/preprocessing/resampling/),
[Advanced Imputation](/examples/preprocessing/advanced_imputation/), and
[Data Cleaning](/examples/preprocessing/data_cleaning/).

## Categorical Data Considerations

Categorical time series introduce additional practical challenges:

- **Class imbalance**: Rare categories may be underrepresented in training data,
  leading to poor predictions for minority classes. Consider stratified evaluation
  using per-class metrics.
- **Label encoding**: [`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.reduction.ClassProbaReductionForecaster/) encodes labels internally.
  Ensure class labels are consistent between training and test data.
- **Temporal class persistence**: Many categorical series exhibit "sticky" behavior
  where the same class persists for long runs (e.g., weather patterns). Models that
  learn this persistence tend to outperform those that treat each timestep
  independently.
