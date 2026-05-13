# Stationarity

Most regression models (whether linear, tree-based, or neural) assume that the statistical relationship between inputs and outputs stays the same across the training data. When applied to time series, this assumption quietly breaks down. A regressor trained on sales data from January cannot reliably predict July sales if the series has a rising trend or a pronounced summer spike. The model has no mechanism to distinguish "the underlying pattern changed" from "the data is noisy." Stationarity transforms address this gap by removing time-dependent structure before the data reaches the regressor.

A stationary series has constant mean, constant variance, and autocovariance that depends only on lag, not on absolute position in time. Real-world series almost never satisfy these conditions out of the box. Retail sales grow year over year (trend), electricity demand peaks every afternoon (seasonality), and financial volatility clusters in bursts (heteroscedasticity). Stationarity transforms peel away these patterns so the regressor sees a well-behaved residual that looks roughly the same whether you sample from the beginning or the end of the training window.

Yohou provides two complementary approaches to stationarity: decomposition pipelines that model each component with a dedicated forecaster, and standalone transformers that apply invertible mathematical operations to the raw series.

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Decomposition

The classical approach to time series decomposition splits a series into additive components:

```text
y(t) = trend(t) + seasonality(t) + residual(t)
```

[`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) automates this pattern. It accepts a list of `(name, forecaster)` tuples and fits them sequentially: the first forecaster models the full series, the second forecaster models the residuals left after subtracting the first forecaster's in-sample predictions, and so on. At prediction time, the component forecasts are summed to produce the final output.

A typical setup pairs a trend forecaster with a seasonality forecaster and a residual forecaster:

```python
DecompositionPipeline([
    ("trend", PolynomialTrendForecaster(degree=1)),
    ("seasonality", PatternSeasonalityForecaster(seasonality=12)),
    ("residual", PointReductionForecaster(regressor=Ridge())),
])
```

The pipeline handles all the bookkeeping: computing residuals between stages, aligning time indices, and reconstructing the final prediction. Setting `store_residuals=True` exposes the intermediate residuals for inspection, which is useful for diagnosing whether the trend forecaster captured enough of the slow-moving level change before the seasonality forecaster tries to model what remains.

For multiplicative decomposition (where seasonal amplitude grows proportionally with the level), pass `target_transformer=LogTransformer()`. This converts the problem to additive in log-space, since `log(trend * season * residual) = log(trend) + log(season) + log(residual)`.

## Trend Estimation

Trend forecasters in the [`yohou.stationarity`](/pages/api/generated/yohou.stationarity/) module estimate and remove slowly-varying level changes. The base class `_BaseTrendForecaster` provides the shared infrastructure for converting datetime indices to numeric features and fitting a regression model.

[`PolynomialTrendForecaster`](/pages/api/generated/yohou.stationarity.trend.PolynomialTrendForecaster/) fits a polynomial of configurable degree using ElasticNet regularization. With `degree=1` it produces a linear trend; `degree=2` gives a quadratic curve. Higher degrees are technically possible but risk overfitting: a cubic trend that wiggles through the training data will extrapolate wildly. For exponential trends, a more robust strategy is to combine `degree=1` with `target_transformer=LogTransformer()`, which fits a linear model in log-space and produces exponential growth in the original scale.

## Seasonality Estimation

Seasonality forecasters model repeating periodic patterns. The module provides two approaches with different trade-offs.

[`PatternSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.PatternSeasonalityForecaster/) extracts a discrete seasonal profile by averaging (or taking the median of) values at each position within the seasonal cycle. With `method="average"` and `seasonality=12`, it computes the mean January value, the mean February value, and so on, then tiles this fixed pattern into the future. The "median" method is more robust to outliers in individual years. The "naive" method simply repeats the last complete cycle. This approach works well when the seasonal shape is stable and the period aligns exactly with the data frequency.

[`FourierSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.FourierSeasonalityForecaster/) represents seasonality as a sum of sine and cosine waves at specified harmonics, fitted via ElasticNet regression. Fourier representation has two notable advantages: it handles non-integer seasonality (such as 365.25 days per year, accounting for leap years) and it produces smooth, differentiable seasonal curves rather than a piecewise-constant pattern. The `harmonics` parameter controls the complexity: more harmonics capture sharper seasonal features, while fewer harmonics produce a gentler curve.

## Standalone Transforms

Not every situation calls for a full decomposition pipeline. Sometimes a single invertible transform is enough to make the residual well-behaved, especially when combined with a flexible regressor inside a [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) or [`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/). The transforms in [`yohou.stationarity`](/pages/api/generated/yohou.stationarity/) fall into two categories: differencing-based (which remove trend and seasonality) and variance-stabilizing (which address heteroscedasticity).

### Differencing

[`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/) computes `y(t) - y(t - s)` where `s` is the seasonal period. With `seasonality=1` this is ordinary first differencing, which removes a linear trend. With `seasonality=12` on monthly data, it subtracts last January from this January, removing both the annual seasonal pattern and any trend that is roughly constant over one cycle. The first `s` values are consumed as history for the lag, so the output is shorter than the input. The transform is invertible: given the lagged values, the original series can be reconstructed exactly.

[`SeasonalLogDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalLogDifferencing/) applies a log transform before differencing. Mathematically this computes `log(y(t)) - log(y(t-s))`, which equals `log(y(t) / y(t-s))`, the log-ratio between current and lagged values. This is the natural choice for series with multiplicative seasonality, where the amplitude of seasonal swings grows proportionally with the level.

[`SeasonalReturn`](/pages/api/generated/yohou.stationarity.transformers.SeasonalReturn/) and [`AbsoluteSeasonalReturn`](/pages/api/generated/yohou.stationarity.transformers.AbsoluteSeasonalReturn/) provide alternative formulations. `SeasonalReturn` computes `(y(t) / y(t-s)) - 1`, the percentage change relative to the seasonal lag. `AbsoluteSeasonalReturn` computes the raw difference `y(t) - y(t-s)`, which is functionally similar to `SeasonalDifferencing` but offers a consistent API with `SeasonalReturn`, including an `offset` parameter for handling near-zero denominators.

All differencing-based transforms are stateful: they set `observation_horizon` equal to the seasonality parameter, meaning they need `s` prior observations to produce output. This state is managed automatically when used inside yohou pipelines.

### Variance Stabilization

Even after removing trend and seasonality, the residual may have non-constant variance. Financial returns, for instance, are roughly zero-mean but their volatility changes over time. Variance-stabilizing transforms compress the range of the data so that the residual variance is approximately uniform.

[`LogTransformer`](/pages/api/generated/yohou.stationarity.transformers.LogTransformer/) applies `log(y + offset)`. It is the simplest variance stabilizer and works well for strictly positive series where larger values exhibit proportionally larger fluctuations. The `offset` parameter shifts the data to avoid taking the log of zero.

[`BoxCoxTransformer`](/pages/api/generated/yohou.stationarity.transformers.BoxCoxTransformer/) generalizes the log transform with a tunable power parameter `lmbda`. When `lmbda=0` it reduces to the log transform; `lmbda=0.5` gives a square root; `lmbda=1` is the identity (no transformation). The Box-Cox family covers a broad range of variance-stabilizing behaviors, making it a good data-driven choice when the right transform is not obvious in advance. Like the log transform, it requires strictly positive input (after applying the offset).

[`ASinhTransformer`](/pages/api/generated/yohou.stationarity.transformers.ASinhTransformer/) applies the inverse hyperbolic sine after centering by the median and scaling by the Median Absolute Deviation (MAD). Unlike log or Box-Cox, `asinh` is defined for all real numbers: it handles zeros, negatives, and extreme outliers without issue. For large positive values it behaves approximately like `log(2x)`, so it still compresses the upper tail. For values near zero it behaves approximately linearly, avoiding the singularity that plagues log transforms. This makes it a practical default when the data contains zeros or can go negative.

## Choosing a Transform

The right transform depends on the structure of the series.

For **additive seasonality with stable variance**,
[`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/)
is usually sufficient. Set `seasonality` to the period length (7 for weekly, 12 for
monthly, 365 for daily-to-yearly).

For **multiplicative seasonality** (where seasonal amplitude grows with the level),
apply
[`LogTransformer`](/pages/api/generated/yohou.stationarity.transformers.LogTransformer/)
followed by
[`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/),
or use
[`SeasonalLogDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalLogDifferencing/)
which combines both steps. A
[`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/)
with `target_transformer=LogTransformer()` is an alternative when you want explicit
component modeling.

For **series with zeros**,
[`ASinhTransformer`](/pages/api/generated/yohou.stationarity.transformers.ASinhTransformer/)
handles zeros gracefully where log-based transforms would fail or require an
artificial offset.

When the **distribution is unknown**,
[`BoxCoxTransformer`](/pages/api/generated/yohou.stationarity.transformers.BoxCoxTransformer/)
with a tuned `lmbda` value adapts to whatever power transform best stabilizes
variance. Pair it with differencing if the series also has trend or seasonality.

For **complex structure** (trend plus multiple seasonal periods), a
[`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/)
assigns a dedicated forecaster to each component, giving explicit control over how
trend, seasonality, and residual are modeled separately.

In practice, trying two or three options and comparing forecast accuracy using
[cross-validation](model-selection.md) is the most reliable way to choose. The
"right" transform is the one that yields the best out-of-sample scores for your
specific series and forecasting horizon.

## References

- Hyndman, R.J. & Athanasopoulos, G. (2021). [Forecasting: principles and practice](https://otexts.com/fpp3/), 3rd edition, OTexts. Chapters 3 (decomposition), 3.1 (transformations), and 9.1 (differencing).
- Box, G.E.P. & Cox, D.R. (1964). "An analysis of transformations." *Journal of the Royal Statistical Society, Series B*, 26(2), 211-252.
- Johnson, N.L. (1949). "Systems of frequency curves generated by methods of translation." *Biometrika*, 36, 149-176.

## Connections

Stationarity transforms feed into the [Forecasting](forecasting.md) pipeline as
`target_transformer` parameters, and the decomposition approach is a complementary
alternative to standalone transforms. The [Preprocessing](preprocessing.md) page
covers non-stationarity transforms (scaling, windowing, imputation) that operate on
features rather than targets. For how residuals reveal whether a stationarity
transform has done its job, see [Residual Diagnostics](residual-diagnostics.md).

Practical examples: [Decomposition](/examples/stationarity/decomposition/),
[Decomposition Variations](/examples/compose/decomposition_variations/),
[Fourier Tuning](/examples/stationarity/fourier_tuning/), and
[Stationarity Transforms](/examples/stationarity/stationarity_transforms/).
