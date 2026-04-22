# Exogenous Features

Most real-world forecasting problems involve more than just the target series. Electricity demand depends on temperature, retail sales respond to promotions, and hospital admissions correlate with flu season indicators. These external predictors, known as exogenous features, can dramatically improve forecast accuracy when used correctly. But they come with a constraint that distinguishes time series forecasting from standard supervised learning: features must be known in advance, at the time the forecast is made.

## Target vs. Exogenous

Yohou follows a clear convention: `y` is the target (what you want to forecast) and `X` is the exogenous feature set (the information available to help). Both are polars DataFrames with a mandatory `"time"` column, but they serve different roles in the forecasting pipeline.

The target variable is what gets tabularized into lagged features by the reduction machinery. It is the series whose future values you are predicting. The exogenous features are additional columns that the regressor can use alongside the lagged target values. They enter the model as-is (after any feature transformers are applied) and enrich the feature space beyond what the target's own history can provide.

## The Ex-Ante Requirement

The single most important constraint on exogenous features is: they must be known at forecast time. This is the ex-ante requirement, and violating it is one of the most common mistakes in time series forecasting.

If you are forecasting sales 7 days ahead, you can use features that are known today (day of week, month, holiday indicator, planned promotions) but not features that depend on future observations (actual temperature next week, competitor pricing next Tuesday). Features that are determined by the calendar (seasonal indicators, Fourier terms, time-of-day) always satisfy this requirement. Features that are measured in real time (temperature, stock prices) only satisfy it if you either use lagged values or forecast the features themselves.

Yohou enforces this through the `X` parameter in `predict()`. Whatever features you pass to `predict()` must cover the forecast horizon. If a feature column is missing or has NaN values in the forecast period, the model cannot use it.

## Useful Predictor Types

Exogenous features fall into a few recurring categories, each with different
relationships to the ex-ante requirement.

### Calendar and seasonal features

Day of week, month of year, hour of day, holiday indicators. These are always known
in advance because they are determined by the calendar, not by observations.
[`CalendarFeatureTransformer`](/pages/api/generated/yohou.preprocessing.calendar.CalendarFeatureTransformer/) and [`HolidayFeatureTransformer`](/pages/api/generated/yohou.preprocessing.calendar.HolidayFeatureTransformer/) derive these directly from the time column, so no external data source is needed.

Fourier terms (sine and cosine pairs at seasonal frequencies) serve a similar purpose
but approximate smooth periodic patterns with fewer features than dummy variables.
[`FourierFeatureTransformer`](/pages/api/generated/yohou.preprocessing.time_features.FourierFeatureTransformer/) constructs these as exogenous columns derived from the time column. Fourier terms can also handle non-integer seasonality (like 365.25 days per year), which calendar dummies cannot. [`FourierSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.FourierSeasonalityForecaster/) uses the same idea internally for decomposition.

### Lagged features

Lagged values of the target or exogenous series capture autoregressive
relationships. For exogenous features, lagged values are inherently ex-ante
valid: last week's temperature is known today. The reduction machinery creates
lagged target features internally during tabularization, but lagged exogenous
features must be constructed explicitly through
[`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/).

### Trend indicators

Numeric time indices, polynomial time features, or piecewise linear trends help
the regressor capture gradual level changes.
[`PolynomialTrendForecaster`](/pages/api/generated/yohou.stationarity.trend.PolynomialTrendForecaster/)
handles this within a decomposition pipeline. For reduction forecasters,
[`TimeIndexTransformer`](/pages/api/generated/yohou.preprocessing.time_features.TimeIndexTransformer/)
converts the time column into numeric indices and optional polynomial terms.

### Rolling statistics

Rolling means, medians, and standard deviations of recent observations capture
local dynamics.
[`RollingStatisticsTransformer`](/pages/api/generated/yohou.preprocessing.window.RollingStatisticsTransformer/)
creates these as feature columns. Because they depend only on past data, they
satisfy the ex-ante requirement automatically.

## Multivariate Target Forecasting

Sometimes `y` itself has multiple columns: you want to forecast sales for three product categories simultaneously. Yohou handles this natively through panel data conventions (columns prefixed with group names separated by `__`) or simply as a multi-column target DataFrame. The reduction forecaster tabularizes each target column and fits the regressor on the combined feature matrix.

For situations where columns influence each other (e.g., forecasting temperature and humidity jointly), the lagged values of all target columns become available as features through the tabularization process, allowing the regressor to learn cross-column relationships.

## Forecasting Unknown Features

A common challenge is when exogenous features are informative but not known in
advance. Temperature improves energy demand forecasts, but you do not know next
week's temperature today.
[`ForecastedFeatureForecaster`](/pages/api/generated/yohou.compose.forecasted_feature_forecaster.ForecastedFeatureForecaster/)
solves this by chaining two models: a feature forecaster that predicts future `X`,
and a target forecaster that uses those predicted features. At prediction time, the
feature forecaster runs first, and its output feeds into the target forecaster.

The uncertainty compounds: errors in feature forecasts propagate to target forecasts.
But the additional information often outweighs the noise, especially when the
exogenous feature has a strong, stable relationship with the target. The
`strategy` parameter controls how the target forecaster is trained to account for
this distribution shift (see [Composition and Pipelines](composition.md) for the
three available strategies).

## References

- Hyndman, R.J. & Athanasopoulos, G. (2021). [Forecasting: principles and practice](https://otexts.com/fpp3/), 3rd edition, OTexts. Chapters 7 and 10.

## Connections

The reduction framework described in [Forecasting](forecasting.md) is where
exogenous features enter the model through tabularization. Feature transformers
for constructing lags, rolling statistics, and other derived features are covered
in [Preprocessing](preprocessing.md). For composing multi-stage forecasters that
chain target and feature predictions, see
[Composition and Pipelines](composition.md).

For constructing calendar, Fourier, and trend features from the time column, see
the [time features section in Preprocessing](preprocessing.md#time-features) and the
[how-to guide](../how-to/time-features.md).
