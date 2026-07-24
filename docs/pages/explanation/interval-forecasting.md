# Interval Forecasting

A point forecast produces a single predicted value ("sales will be 150 units
tomorrow"). An interval forecast produces a range ("sales will be between 120 and
180 units tomorrow, with 90% probability"). The range is more honest: it
acknowledges uncertainty and gives decision-makers the information they need to
plan for risk. How wide should you set inventory buffers? How much capacity should
you reserve? These questions require knowing not just the expected outcome, but how
wrong the forecast might be.

Yohou provides two approaches to interval forecasting: conformal prediction
(distribution-free, wraps any point forecaster) and quantile regression (learns
interval bounds directly). Both produce prediction intervals at user-specified
coverage rates and integrate with yohou's standard `fit`/`predict_interval`/`observe`
lifecycle.

## Conformal Prediction

Traditional statistical methods (ARIMA, exponential smoothing) produce prediction
intervals by assuming the forecast errors follow a known distribution, typically
Gaussian. Under this assumption, interval width is derived analytically from the
estimated error variance. The advantage is simplicity; the risk is that the
assumption is wrong. Heavy-tailed, skewed, or heteroscedastic errors produce
intervals that are too narrow and undercover in practice.

Conformal prediction avoids this problem entirely. It is a distribution-free
framework that constructs prediction intervals from calibration data alone. The
core idea: if you have a collection of past prediction errors, the quantile of
those errors tells you how wide the interval needs to be. If the 90th percentile
of past absolute errors is 15, then adding and subtracting 15 from a point
prediction gives an interval that would have covered 90% of past observations.

The formal guarantee is marginal coverage: over the calibration set, the intervals
contain the true value at least as often as the stated coverage rate. This
guarantee holds regardless of the underlying data distribution, the forecasting
model, or the time series characteristics. The only assumption is exchangeability
of the calibration residuals, roughly meaning that the calibration errors are
representative of future errors.

In practice, this assumption can be violated when the data distribution shifts over
time. Yohou's `observe` mechanism helps here: as new data arrives, the calibration
set is updated incrementally, keeping the conformity scores aligned with recent
behavior.

Yohou focuses on distribution-free approaches because they pair naturally with the
reduction framework: any sklearn regressor can serve as the base model, and the
interval construction does not depend on the regressor's internal assumptions.

## Split Conformal Forecasting

[`SplitConformalForecaster`](/pages/api/generated/yohou.interval.SplitConformalForecaster/)
implements the split conformal approach. It divides the training data into two
portions: a training set and a calibration set. The wrapped point forecaster trains
on the first portion, then generates predictions on the held-out calibration portion.
The differences between those predictions and the actual calibration values become the
conformity scores that determine interval width.

The process works as follows:

1. **Split**: The last `calibration_size` observations form the calibration set;
   the remainder is the training set.
2. **Train**: The point forecaster fits on the training set.
3. **Calibrate**: The fitted forecaster predicts across the calibration set using
   a rolling observe-predict loop (stride of 1), producing one set of conformity
   scores per forecast horizon step.
4. **Predict**: At inference time, the quantile of the calibration scores at the
   desired coverage rate sets the interval bounds around the point prediction.

```python
from yohou.point import SeasonalNaive
from yohou.interval import SplitConformalForecaster

forecaster = SplitConformalForecaster(
    point_forecaster=SeasonalNaive(seasonality=7),
    calibration_size=100,
)
forecaster.fit(y=train, forecasting_horizon=7, coverage_rates=[0.9])
intervals = forecaster.predict_interval(coverage_rates=[0.9])
```

Because the calibration scores are computed per horizon step, step 1 and step 7
predictions can have different interval widths. This reflects the natural behavior
that uncertainty grows with the forecast horizon.

An important nuance: the coverage guarantee is marginal, meaning it holds on average
across the calibration set. It does not guarantee that any specific individual
prediction interval will contain the true value. In regions where the model performs
poorly, the actual coverage can be lower; in regions where the model is accurate, it
can be higher.

### Updating with New Observations

After initial fitting, calling `observe()` with new data updates the conformity
scores *before* updating the underlying point forecaster. This ordering ensures that
the next `predict_interval()` call reflects both the refreshed calibration set and the
updated model. As new observations arrive, the conformity score distribution evolves
to reflect recent forecast accuracy, keeping the intervals well-calibrated even as the
data changes.

The `rewind()` method reverses post-fit observations, removing their conformity scores
from the calibration set. It will not remove data that was part of the original fit.

## Conformity Scorers

The choice of conformity scorer controls how the calibration residuals translate into
interval bounds. Different scorers produce intervals with different geometric
properties.

**Signed residuals**:
[`Residual`](/pages/api/generated/yohou.metrics.Residual/)
computes $s = y - \hat{y}$. Because positive and negative errors are preserved
separately, the lower and upper quantiles can differ. This produces **asymmetric**
intervals where the point prediction is not necessarily at the center. Asymmetric
intervals are appropriate when the error distribution is skewed (for example, when
demand forecasts tend to underpredict more than they overpredict). This is the default
scorer.

**Absolute residuals**:
[`AbsoluteResidual`](/pages/api/generated/yohou.metrics.AbsoluteResidual/)
computes $s = |y - \hat{y}|$. A single quantile is added and subtracted from the
point prediction, producing **symmetric** intervals centered on the forecast. This
works well when errors are roughly symmetric around zero.

**Gamma (relative) residuals**:
[`GammaResidual`](/pages/api/generated/yohou.metrics.GammaResidual/)
computes $s = (y - \hat{y}) / (\hat{y} + \epsilon)$. By normalizing the error by the
prediction magnitude, this scorer produces intervals that scale with the level of the
series. When the target value is large, the interval is wide; when it is small, the
interval is narrow. This is the right choice for data with multiplicative seasonality
or heteroscedastic variance that grows proportionally with the signal.
[`AbsoluteGammaResidual`](/pages/api/generated/yohou.metrics.AbsoluteGammaResidual/)
is the symmetric variant.

Switching scorers requires no changes to the forecaster; pass a different
`conformity_scorer` to the
[`SplitConformalForecaster`](/pages/api/generated/yohou.interval.SplitConformalForecaster/)
constructor.

## Adaptive Intervals

Standard conformal prediction uses the same set of calibration scores for every
prediction, regardless of context. This means the interval width is constant: a
prediction during a holiday peak gets the same interval as a prediction on a quiet
Tuesday. In many applications, the uncertainty genuinely varies across different
conditions.

Similarity-based weighting addresses this. Instead of treating all calibration
residuals equally when computing quantiles, it assigns higher weights to calibration
points that are "similar" to the current prediction context. The weighted quantile
then produces intervals that adapt to local conditions.

### Distance Similarity

[`DistanceSimilarity`](/pages/api/generated/yohou.interval.DistanceSimilarity/)
computes distances between the current prediction context (the feature vector derived
from the point forecaster's observation window at predict time) and each calibration
feature vector stored during `fit()`. It then converts distances to weights using a
numerically stable softmax of negative distances with uniform mass reserved for the
test point:

$$w_{ji} = \frac{\exp(-(d_{ji} - \max_k d_{jk}))}{1 + \sum_k \exp(-(d_{jk} - \max_k d_{jk}))}$$

Calibration points close to the current prediction get exponentially higher weights
than distant ones. The distance metric is configurable: euclidean, cityblock, cosine,
or any metric supported by `scipy.spatial.distance.cdist`.

The leading `1` in the denominator reserves uniform mass for the hypothetical test
point, so each weight row is non-negative and sums to a value strictly below 1. This
follows the non-exchangeable conformal construction of
[Barber et al. (2023)](https://doi.org/10.1214/23-AOS2276): the test
point, whose residual is unknown, is treated as one more calibration candidate that
always holds a baseline share of the mass. That reserved share shrinks as more
calibration points fall close to the prediction context, and grows when none do.

```python
from yohou.interval import SplitConformalForecaster, DistanceSimilarity

forecaster = SplitConformalForecaster(
    similarity=DistanceSimilarity(metric="euclidean"),
    calibration_size=100,
)
```

### Seasonal Similarity

[`SeasonalSimilarity`](/pages/api/generated/yohou.interval.SeasonalSimilarity/)
captures seasonal patterns by extracting Fourier features (sine and cosine components)
from timestamps at specified seasonal periods. Predictions at similar seasonal
positions (for example, all Mondays, or all January observations) receive higher
calibration weights.

```python
from yohou.interval import SplitConformalForecaster, SeasonalSimilarity

forecaster = SplitConformalForecaster(
    similarity=SeasonalSimilarity(seasonalities=[7.0, 365.25]),
    calibration_size=100,
)
```

The `seasonalities` parameter accepts a list of period lengths. A weekly seasonality
of 7.0 groups similar days of the week; an annual seasonality of 365.25 groups similar
times of year. The `harmonics` parameter controls how many sine/cosine pairs are
generated per seasonality, allowing finer or coarser seasonal grouping.

### Composite Similarity

[`CompositeSimilarity`](/pages/api/generated/yohou.interval.CompositeSimilarity/)
combines multiple similarity measures into a single weighting scheme. This is useful
when both feature-space proximity and temporal proximity matter.

```python
from yohou.interval import (
    SplitConformalForecaster,
    CompositeSimilarity,
    DistanceSimilarity,
    SeasonalSimilarity,
)

forecaster = SplitConformalForecaster(
    similarity=CompositeSimilarity(
        similarities=[
            ("dist", DistanceSimilarity(metric="euclidean")),
            ("seasonal", SeasonalSimilarity(seasonalities=[7.0])),
        ],
        combination="multiply",
    ),
    calibration_size=100,
)
```

The `combination` parameter controls how weight matrices are merged: `"multiply"`
takes the element-wise product (both similarities must agree for a calibration point
to receive high weight), while `"mean"` takes the weighted average. An optional
`weights` list assigns relative importance to each sub-similarity.

### Tradeoffs

The tradeoff with all similarity measures is effective sample size. Heavily weighting
a few nearby calibration points reduces variance but can increase bias if the local
neighborhood is too small. Larger calibration sets help by providing more data points
in each local region.

## Adaptive Conformal Inference

Similarity weighting adapts *which* residuals count. Adaptive conformal
inference adapts something orthogonal: the *quantile level* itself. A static
calibration set fixes the miscoverage level once, so if the series drifts, the
realized coverage drifts away from the target and never corrects. The adaptive
conformal inference update (Gibbs and Candes, 2021) closes that loop. After each
observation it compares the target miscoverage to the realized miscoverage and
shifts the effective level accordingly:

$$\alpha_{t+1} = \alpha_t + \gamma\,(\alpha^{*} - \mathrm{err}_t)$$

where $\alpha^{*} = 1 - \text{coverage rate}$ is the target, $\mathrm{err}_t$ is 1
when the truth fell outside the interval, and $\gamma$ (the `step_size`) sets how
fast the level reacts. A run of misses lowers $\alpha_t$ and widens the interval;
a run of covers raises it and narrows the interval.

In Yohou this is an optional
[`AdaptiveConformalInference`](/pages/api/generated/yohou.interval.AdaptiveConformalInference/)
adapter passed to `SplitConformalForecaster`. It lives in the same
`observe` / `predict_interval` / `rewind` lifecycle as the rest of the library,
so a backtest and a production stream restore coverage identically. Because the
two mechanisms are orthogonal, the adapter composes with similarity weighting:
the similarity sets the weights, the adapter sets the level. The two axes are
summarized below.

| Mechanism | What it adapts | Set by |
| --- | --- | --- |
| Similarity weighting | which residuals count | `similarity` |
| Adaptive conformal inference | how far into their tail to reach | `adapter` |

The adapter tracks one level per horizon step (or a single shared level with
`alpha_pooling="shared"`), and one level per tail for asymmetric conformity
scorers so a lopsided error distribution corrects each side separately.

## Quantile Reduction Intervals

[`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.IntervalReductionForecaster/)
takes a fundamentally different approach. Instead of wrapping a point forecaster and
calibrating intervals after the fact, it trains quantile regression models that
directly predict the interval bounds. For a coverage rate $\alpha$, it fits two
models: one for the lower quantile and one for the upper quantile:

$$\hat{y}^{\text{lower}} = f_{(1-\alpha)/2}(\mathbf{x}_t), \quad \hat{y}^{\text{upper}} = f_{(1+\alpha)/2}(\mathbf{x}_t)$$

For a 90% coverage rate, this means one model at the 5th percentile (lower bound) and
one at the 95th percentile (upper bound).

```python
from yohou.interval import IntervalReductionForecaster

forecaster = IntervalReductionForecaster()
forecaster.fit(y=train, forecasting_horizon=7, coverage_rates=[0.9])
intervals = forecaster.predict_interval(coverage_rates=[0.9])
```

### Reduction Strategies

Like
[`PointReductionForecaster`](/pages/api/generated/yohou.point.PointReductionForecaster/),
the interval variant supports three `reduction_strategy` options:

- **Multi-output** (`"multi-output"`, the default): a single model predicts all
  horizon steps simultaneously per quantile.
- **Direct** (`"direct"`): fits independent models per horizon step per quantile.
  With `n_jobs`, these can run in parallel.
- **Dir-rec** (`"dir-rec"`): sequential models where later steps see predictions
  from earlier steps.

The same tradeoffs apply: multi-output is fastest, direct avoids error accumulation,
and dir-rec captures inter-step dependencies. See
[Reduction Strategies](reduction-forecasting.md#reduction-strategies) for the full
treatment of these approaches.

### Multi-Quantile Estimators

By default, `IntervalReductionForecaster` uses `QuantileRegressor` from sklearn and
fits separate models for each quantile (two per coverage rate). Some gradient boosting
libraries support predicting multiple quantiles in a single model, which is
significantly faster.

The forecaster automatically detects multi-quantile capability in two cases:

- **CatBoost**: when `loss_function` starts with `"MultiQuantile"`, a single model is
  trained for all requested quantiles simultaneously.
- **LightGBM**: when `objective="quantile"` with an `alpha` parameter.

When a multi-quantile estimator is detected, all quantiles for all coverage rates are
combined into a single training pass. For example, coverage rates `[0.9, 0.95]`
produce quantiles `[0.025, 0.05, 0.5, 0.95, 0.975]` in one model rather than ten
separate models.

The MultiQuantile path is restricted to `forecasting_horizon=1`. For multi-step
horizons, a standard single-quantile estimator such as `QuantileRegressor` lets
`IntervalReductionForecaster` fit multiple models via the `direct` strategy instead.

### Comparison with Conformal Prediction

Because the quantile models learn the conditional distribution directly from features,
their intervals naturally adapt to heteroscedastic data without needing explicit
similarity weighting. The disadvantage is that quantile regression does not carry the
same finite-sample coverage guarantee as conformal prediction, and its accuracy depends
entirely on how well the model captures the conditional quantiles.

## Panel Data

Both
[`SplitConformalForecaster`](/pages/api/generated/yohou.interval.SplitConformalForecaster/)
and
[`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.IntervalReductionForecaster/)
support panel data through the `panel_strategy` parameter:

- **`"global"`** (default): fits a single shared model across all groups, but
  maintains per-group transformer state and observation buffers. Groups share a
  single calibration set or quantile model. For independent per-group models, use
  [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.LocalPanelForecaster/).
- **`"multivariate"`**: pools data across groups, sharing calibration scores or
  quantile models. This is useful when individual groups have limited data and
  borrowing strength across entities improves interval quality.

The choice mirrors the panel strategies available in point forecasting. See
[Panel Data](panel-data.md) for the full treatment.

## Coverage Rates

Coverage rates are specified as floats in the range (0, 1]. Multiple rates can be
requested in a single call. Higher coverage rates produce wider intervals: a 95%
interval must be wider than a 90% interval to capture more of the distribution.

Coverage rates are first specified at `fit()` time (defaulting to `[0.95]` if
omitted). At `predict_interval()` time, you can request different coverage rates
without re-fitting. For `SplitConformalForecaster`, this works because the conformity
scores are stored and the quantile computation is applied at prediction time; no
re-calibration is needed. For `IntervalReductionForecaster`, new coverage rates
require that the underlying estimator can produce predictions at the corresponding
quantile levels.

The right coverage rate reflects the cost asymmetry of the decision. Safety-critical
applications (capacity planning, risk management) warrant high coverage because the
cost of being caught outside the interval is severe. Low-stakes decisions can tolerate
narrower intervals that are more actionable even if they miss more often.

## References

- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random
  World*. Springer. (foundational conformal prediction framework)
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R.J., & Wasserman, L. (2018).
  Distribution-free predictive inference for regression. *Journal of the American
  Statistical Association*, 113(523), 1094-1111.
  [DOI:10.1080/01621459.2017.1307116](https://doi.org/10.1080/01621459.2017.1307116)
- Barber, R.F., Candes, E.J., Ramdas, A., & Tibshirani, R.J. (2023). Conformal
  prediction beyond exchangeability. *Annals of Statistics*, 51(2), 816-845.
  [DOI:10.1214/23-AOS2276](https://doi.org/10.1214/23-AOS2276)
- Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*,
  3rd edition, [Chapter 5.5](https://otexts.com/fpp3/prediction-intervals.html)
  (prediction intervals).

## Connections

Interval forecasting builds on yohou's
[Reduction Forecasting](reduction-forecasting.md) foundation. Every interval method
either wraps a point forecaster or extends the same reduction machinery. The
`observe`/`predict_interval` lifecycle mirrors the point forecasting API, so
switching between point and interval forecasts requires minimal code changes.

For evaluating interval forecasts, see the interval metrics in
[Forecast Accuracy](forecast-accuracy.md). Coverage rate and interval width metrics
help diagnose whether intervals are well-calibrated: too narrow means the stated
coverage is not achieved, too wide means the intervals are uninformative.

For cross-validation with interval forecasters, the
[Model Selection](model-selection.md) tools work with `predict_interval` in the same
way they work with `predict`.

[`VotingIntervalForecaster`](/pages/api/generated/yohou.ensemble.VotingIntervalForecaster/)
provides an ensemble approach to combining prediction intervals from multiple models.
It supports three aggregation methods: averaging bounds, taking medians, or taking the
envelope (minimum of lower bounds, maximum of upper bounds for the most conservative
intervals). See [Ensemble Forecasting](ensemble-forecasting.md) for details.

For practical recipes, see
[How to Forecast with Prediction Intervals](../how-to/interval-forecasting.md).
For a hands-on introduction, see the [Interval Forecasting Tutorial](../tutorials/interval-forecasting.md).

Interactive examples:
[Conformal Forecasting](/examples/conformal_forecasting/),
[Conformity Scorers](/examples/conformity_scorers/), and
[Distance Similarity](/examples/distance_similarity/).
