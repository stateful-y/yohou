# Stationarity

Time series forecasting models require some form of temporal regularity in the
data they learn from. The exact requirement depends on the modeling approach, but
the practical consequence is the same: raw series with trends, seasonal swings,
or changing variance violate these assumptions and produce unreliable forecasts.
Stationarity transforms remove this time-dependent structure so the model sees
well-behaved data.

## IID and Stationarity: Two Related Assumptions

Two distinct assumptions appear repeatedly in forecasting, and understanding how
they differ clarifies when and why transforms are necessary.

**The IID assumption (reduction forecasters).** Scikit-Learn regressors assume
training samples are *independent and identically distributed* (i.i.d.): each
row is drawn from the same distribution, and knowing one row tells you nothing
about another. When a reduction forecaster tabularizes a time series into
feature/target rows, the resulting table should approximate this assumption. If
the original series has a rising trend, the early rows have systematically lower
targets than the later rows, so the rows are not identically distributed. A
seasonal pattern creates the same problem at a different frequency. The
regressor cannot distinguish "the data-generating process changed" from "the
data is noisy," because it treats each row as exchangeable.

**The stationarity assumption (native time series models).** Classical
econometric models (ARIMA, exponential smoothing, and their variants available
through [`yohou-nixtla`](/pages/reference/extensions/)) operate directly on the
ordered sequence of observations rather than on tabularized rows. These models
assume *stationarity*: the joint distribution of any collection of time points
depends only on the relative spacing between them, not on their absolute
position. In practice, weak stationarity suffices: constant mean, constant
variance, and autocovariance that depends only on lag. ARIMA enforces this by
differencing the series until a unit-root test (ADF or KPSS) no longer rejects;
exponential smoothing handles trend and seasonality through explicit state
components.

**Overlap.** Both assumptions break down in the same situations (trends,
seasonality, heteroscedasticity), and many of the same transforms fix both.
Differencing a trended series removes the trend whether the downstream consumer
is a Ridge regressor or an ARIMA model. A log transform stabilizes variance
regardless of the modeling approach.

**Distinction.** IID is stronger than stationarity because it additionally
requires *independence*: no autocorrelation between samples. A stationary AR(1)
process is stationary but not IID, because each value depends on the previous
one. In the reduction setting, the tabularized rows are not truly independent
either (overlapping windows share observations), but the regressor still works
well in practice because the features explicitly encode the temporal dependence
that the IID assumption otherwise forbids. The key concern for reduction
forecasters is identical distribution across rows (no trend or seasonal shift in
the target), not literal independence. For native time series models, mild
violations of stationarity are often tolerable because the model's own structure
(autoregressive terms, seasonal components) captures the temporal dependence
that stationarity permits.

**Practical consequence.** Whether you use a reduction forecaster or a native
time series model, the same stationarity transforms help: they remove
deterministic structure (trend, seasonality, heteroscedasticity) that violates
the model's assumptions, leaving a residual that the model can learn from
effectively.

## Approaches in Yohou

Yohou provides two complementary approaches to stationarity: decomposition
pipelines that model each component with a dedicated forecaster, and standalone
transformers that apply invertible mathematical operations to the raw series.

## Decomposition

The classical approach to time series decomposition splits a series into additive components:

```text
y(t) = trend(t) + seasonality(t) + residual(t)
```

[`DecompositionPipeline`](/pages/api/generated/yohou.compose.DecompositionPipeline/) automates this pattern. It accepts a list of `(name, forecaster)` tuples and fits them sequentially: the first forecaster models the full series, the second forecaster models the residuals left after subtracting the first forecaster's in-sample predictions, and so on. At prediction time, the component forecasts are summed to produce the final output.

A typical setup pairs a trend forecaster with a seasonality forecaster and a residual forecaster:

```python
DecompositionPipeline([
    ("trend", PolynomialTrendForecaster(degree=1)),
    ("seasonality", PatternSeasonalityForecaster(seasonality=12)),
    ("residual", PointReductionForecaster(estimator=Ridge())),
])
```

The pipeline handles all the bookkeeping: computing residuals between stages, aligning time indices, and reconstructing the final prediction. Setting `store_residuals=True` exposes the intermediate residuals for inspection, which is useful for diagnosing whether the trend forecaster captured enough of the slow-moving level change before the seasonality forecaster tries to model what remains.

For multiplicative decomposition (where seasonal amplitude grows proportionally with the level), pass `target_transformer=LogTransformer()`. This converts the problem to additive in log-space, since `log(trend * season * residual) = log(trend) + log(season) + log(residual)`.

## Trend Estimation

Trend forecasters in the [`yohou.stationarity`](/pages/api/stationarity/) module estimate and remove slowly varying level changes. They convert datetime indices to numeric features internally, then fit a regression model to capture the trend shape.

[`PolynomialTrendForecaster`](/pages/api/generated/yohou.stationarity.PolynomialTrendForecaster/) fits a polynomial of configurable degree using ElasticNet regularization. With `degree=1` it produces a linear trend; `degree=2` gives a quadratic curve. Higher degrees are technically possible but risk overfitting: a cubic trend that wiggles through the training data will extrapolate wildly. For exponential trends, a more robust strategy is to combine `degree=1` with `target_transformer=LogTransformer()`, which fits a linear model in log-space and produces exponential growth in the original scale.

## Seasonality Estimation

Seasonality forecasters model repeating periodic patterns. The module provides two approaches with different trade-offs.

[`PatternSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.PatternSeasonalityForecaster/) extracts a discrete seasonal profile by averaging (or taking the median of) values at each position within the seasonal cycle. With `method="average"` and `seasonality=12`, it computes the mean January value, the mean February value, and so on, then tiles this fixed pattern into the future. The "median" method is more robust to outliers in individual years. The "naive" method simply repeats the last observed cycle and requires only one complete seasonal cycle of training data, while "average" and "median" require at least two. This approach works well when the seasonal shape is stable and the period aligns exactly with the data frequency.

[`FourierSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.FourierSeasonalityForecaster/) represents seasonality as a sum of sine and cosine waves at specified harmonics, fitted via a regression model (ElasticNet by default, configurable through the `estimator` parameter). Fourier representation has two notable advantages: it handles non-integer seasonality (such as 365.25 days per year, accounting for leap years) and it produces smooth, differentiable seasonal curves rather than a piecewise-constant pattern. The `harmonics` parameter controls the complexity: more harmonics capture sharper seasonal features, while fewer harmonics produce a gentler curve.

## Standalone Transforms

Not every situation calls for a full decomposition pipeline. Sometimes a single invertible transform is enough to make the residual well-behaved, especially when passed as `target_transformer` to a [`PointReductionForecaster`](/pages/api/generated/yohou.point.PointReductionForecaster/). The transforms in [`yohou.stationarity`](/pages/api/stationarity/) fall into two categories: differencing-based (which remove trend and seasonality) and variance-stabilizing (which address heteroscedasticity).

### Differencing

[`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.SeasonalDifferencing/) computes `y(t) - y(t - s)` where `s` is the seasonal period. With `seasonality=1` this is ordinary first differencing, which removes a linear trend. With `seasonality=12` on monthly data, it subtracts last January from this January, removing both the annual seasonal pattern and any trend that is roughly constant over one cycle. The first `s` values are consumed as history for the lag, so the output is shorter than the input. The transform is invertible: given the lagged values, the original series can be reconstructed exactly.

[`SeasonalLogDifferencing`](/pages/api/generated/yohou.stationarity.SeasonalLogDifferencing/) applies a log transform before differencing. Mathematically this computes `log(y(t)) - log(y(t-s))`, which equals `log(y(t) / y(t-s))`, the log-ratio between current and lagged values. This is the natural choice for series with multiplicative seasonality, where the amplitude of seasonal swings grows proportionally with the level.

[`SeasonalReturn`](/pages/api/generated/yohou.stationarity.SeasonalReturn/) and [`AbsoluteSeasonalReturn`](/pages/api/generated/yohou.stationarity.AbsoluteSeasonalReturn/) provide alternative formulations. [`SeasonalReturn`](/pages/api/generated/yohou.stationarity.SeasonalReturn/) computes `(y(t) / y(t-s)) - 1`, the percentage change relative to the seasonal lag. [`AbsoluteSeasonalReturn`](/pages/api/generated/yohou.stationarity.AbsoluteSeasonalReturn/) computes the raw difference `y(t) - y(t-s)`, which is functionally similar to [`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.SeasonalDifferencing/) but offers a consistent API with `SeasonalReturn`, including an `offset` parameter for handling near-zero denominators.

All differencing-based transforms are stateful: they set `observation_horizon` equal to the seasonality parameter. This means the first `s` rows of the input are consumed as lag context rather than appearing in the output. When used as a `target_transformer` inside a reduction forecaster, the pipeline reserves these lag observations automatically before tabularization, so no manual adjustment is needed.

### Variance Stabilization

Even after removing trend and seasonality, the residual may have non-constant variance. Financial returns, for instance, are roughly zero-mean but their volatility changes over time. Variance-stabilizing transforms compress the range of the data so that the residual variance is approximately uniform.

[`LogTransformer`](/pages/api/generated/yohou.stationarity.LogTransformer/) applies `log(y + offset)`. It is the simplest variance stabilizer and works well for strictly positive series where larger values exhibit proportionally larger fluctuations. The `offset` parameter shifts the data to avoid taking the log of zero.

[`BoxCoxTransformer`](/pages/api/generated/yohou.stationarity.BoxCoxTransformer/) generalizes the log transform with a tunable power parameter `lmbda`. It computes $((y + \text{offset})^{\lambda} - 1) / \lambda$ when $\lambda \neq 0$, and $\log(y + \text{offset})$ when $\lambda = 0$. Setting `lmbda=0.5` gives a square root transform; `lmbda=1` is the identity. The Box-Cox family covers a broad range of variance-stabilizing behaviors, making it a good data-driven choice when the right transform is not obvious in advance. Like the log transform, it requires strictly positive input (after applying the offset).

[`ASinhTransformer`](/pages/api/generated/yohou.stationarity.ASinhTransformer/) centers each column by its median, scales by the Median Absolute Deviation (MAD), then applies the inverse hyperbolic sine: $\operatorname{asinh}((y - \tilde{y}) / \text{MAD})$. During `fit`, it stores the per-column median and MAD so that `inverse_transform` can reverse the operation exactly. Unlike log or Box-Cox, `asinh` is defined for all real numbers: it handles zeros, negatives, and extreme outliers without issue. For large positive values it behaves approximately like $\log(2x)$, compressing the upper tail. For values near zero it behaves approximately linearly, avoiding the singularity that plagues log transforms. This makes it a practical default when the data contains zeros or can go negative.

## Trade-offs Between Transform Approaches

The two broad approaches (standalone invertible transforms and decomposition pipelines)
represent different trade-offs between explicitness and flexibility.

Standalone transforms like
[`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.SeasonalDifferencing/)
and
[`SeasonalLogDifferencing`](/pages/api/generated/yohou.stationarity.SeasonalLogDifferencing/)
are the simplest path to stationarity: a single invertible operation removes a
predictable pattern without introducing any additional model parameters. Their strength
is transparency and inversion. Because the transform is fully reversible with a known
formula, predictions in the differenced space map back to the original scale exactly.
Their limitation is rigidity: differencing assumes the pattern to remove (trend or
seasonal period) is stable and known in advance. A series whose seasonal amplitude
changes over time, or whose seasonal period drifts, will still produce structured
residuals after a fixed differencing step.

Decomposition pipelines via
[`DecompositionPipeline`](/pages/api/generated/yohou.compose.DecompositionPipeline/)
take the opposite stance. Each component (trend, seasonality, residual) gets its own
dedicated forecaster, which can adapt to whatever shape the component takes. A
[`PolynomialTrendForecaster`](/pages/api/generated/yohou.stationarity.PolynomialTrendForecaster/) can model non-linear growth; a
[`FourierSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.FourierSeasonalityForecaster/) can represent a seasonal pattern that changes smoothly
across years. The cost is model complexity: each component forecaster introduces
hyperparameters, increases fitting time, and requires the user to decide how many
components to include. Inspecting intermediate residuals (via `store_residuals=True`)
is essential for verifying that each forecaster removes the pattern it is supposed to
remove without over-fitting.

Variance-stabilizing transforms occupy a different dimension from trend and seasonality
removal. They address heteroscedasticity, not location shifts, and can be composed
freely with either standalone differencing or decomposition pipelines.

The final arbiter is always out-of-sample accuracy. Comparing two or three candidate
transforms using [cross-validation](model-selection.md) reveals which actually produces
a more predictable residual for your specific series and horizon. No heuristic about
additive versus multiplicative seasonality substitutes for empirical evaluation.

## References

- Hyndman, R.J. & Athanasopoulos, G. (2021). [Forecasting: principles and practice](https://otexts.com/fpp3/), 3rd edition, OTexts. Chapters 3 (decomposition), 3.1 (transformations), and 9.1 (differencing).
- Hamilton, J.D. (1994). *Time Series Analysis*. Princeton University Press. Chapter 3 (stationarity and ergodicity).
- Box, G.E.P. & Cox, D.R. (1964). "An analysis of transformations." *Journal of the Royal Statistical Society, Series B*, 26(2), 211-252.
- Johnson, N.L. (1949). "Systems of frequency curves generated by methods of translation." *Biometrika*, 36, 149-176.

## Connections

Stationarity transforms feed into the [Reduction Forecasting](reduction-forecasting.md) pipeline as
`target_transformer` parameters, and the decomposition approach is a complementary
alternative to standalone transforms. Native time series models available through
[`yohou-nixtla`](/pages/reference/extensions/) handle stationarity internally
(ARIMA differences automatically, ETS models trend and seasonality as state
components), so explicit transforms are optional when using those forecasters.
The [Preprocessing](preprocessing.md) page
covers non-stationarity transforms (scaling, windowing, imputation) that operate on
features rather than targets. For how residuals reveal whether a stationarity
transform has done its job, see [Residual Diagnostics](residual-diagnostics.md).
The [Interval Forecasting](interval-forecasting.md) page discusses how traditional
statistical models assume Gaussian residuals (a stronger condition than stationarity)
while conformal prediction requires only exchangeability of calibration errors.

For practical recipes, see [How to Apply Stationarity Transforms](../how-to/apply-stationarity-transforms.md).
For a hands-on exploration of seasonal structure, see the
[Seasonal Analysis Tutorial](../tutorials/seasonal-analysis.md).

Interactive examples: [Decomposition](/examples/decomposition/),
[Decomposition Variations](/examples/decomposition_variations/),
[Fourier Tuning](/examples/fourier_tuning/), and
[Stationarity Transforms](/examples/stationarity_transforms/).
