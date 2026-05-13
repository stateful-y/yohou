# Interval Forecasting

A point forecast tells you "sales will be 150 units tomorrow." An interval forecast
tells you "sales will be between 120 and 180 units tomorrow, with 90% probability."
The single number is easier to communicate, but the range is more honest: it
acknowledges that the future is uncertain and gives decision-makers the information
they need to plan for risk. How wide should you set inventory buffers? How much
capacity should you reserve? These questions require knowing not just the expected
outcome, but how wrong the forecast might be.

Yohou provides two approaches to interval forecasting: conformal prediction
(distribution-free, wraps any point forecaster) and quantile regression (learns
intervals directly). Both produce prediction intervals at user-specified coverage
rates and integrate with yohou's standard `fit`/`predict_interval`/`observe` lifecycle.

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Conformal Prediction

Traditional statistical methods (ARIMA, exponential smoothing) produce prediction
intervals by assuming the forecast errors follow a known distribution, typically
Gaussian. Under this assumption, interval width is derived analytically from the
estimated error variance. The advantage is simplicity; the risk is that the assumption
is wrong. Heavy-tailed, skewed, or heteroscedastic errors produce intervals that are
too narrow and undercover in practice.

Conformal prediction avoids this problem entirely. It is a distribution-free framework
that constructs prediction intervals from calibration data alone. The core idea is
simple: if you have a collection of past prediction errors, the quantile of those
errors tells you how wide the interval needs to be. If the 90th percentile of past
absolute errors is 15, then adding and subtracting 15 from a point prediction gives
an interval that would have covered 90% of past observations.

The formal guarantee is marginal coverage: over the calibration set, the intervals
contain the true value at least as often as the stated coverage rate. This guarantee
holds regardless of the underlying data distribution, the forecasting model, or the
time series characteristics. The only assumption is exchangeability of the calibration
residuals, roughly meaning that the calibration errors are representative of future errors.

In practice, this assumption can be violated when the data distribution shifts over
time. Yohou's `observe` mechanism helps here: as new data arrives, the calibration
set can be updated incrementally.

Yohou focuses on distribution-free approaches because they pair naturally with the
reduction framework: any sklearn regressor can serve as the base model, and the
interval construction does not depend on the regressor's internal assumptions.


## Split Conformal Forecasting

[`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/)
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
    point_forecaster=SeasonalNaive(seasonal_period=7),
    calibration_size=100,
)
forecaster.fit(y=train, forecasting_horizon=7, coverage_rates=[0.9])
intervals = forecaster.predict_interval(coverage_rates=[0.9])
```

Because the calibration scores are computed per horizon step, step-1 and step-7
predictions can have different interval widths. This reflects the natural behavior
that uncertainty grows with the forecast horizon.

An important nuance: the coverage guarantee is marginal, meaning it holds on average
across the calibration set. It does not guarantee that any specific individual
prediction interval will contain the true value. In regions where the model performs
poorly, the actual coverage can be lower; in regions where the model is accurate, it
can be higher.


## Conformity Scores

The choice of conformity scorer controls how the calibration residuals translate into
interval bounds. Different scorers produce intervals with different properties.

**Signed residuals**:
[`Residual`](/pages/api/generated/yohou.metrics.conformity.Residual/)
computes $s = y - \hat{y}$. Because positive and negative errors are preserved
separately, the lower and upper quantiles can differ. This produces **asymmetric**
intervals where the point prediction is not necessarily at the center. Asymmetric
intervals are useful when the error distribution is skewed. For example, when
demand forecasts tend to underpredict more than they overpredict.

**Absolute residuals**:
[`AbsoluteResidual`](/pages/api/generated/yohou.metrics.conformity.AbsoluteResidual/)
computes $s = |y - \hat{y}|$. A single quantile is added and subtracted from the
point prediction, producing **symmetric** intervals centered on the forecast. This is
the simpler choice and works well when errors are roughly symmetric around zero.

**Gamma (relative) residuals**:
[`GammaResidual`](/pages/api/generated/yohou.metrics.conformity.GammaResidual/)
computes $s = (y - \hat{y}) / (\hat{y} + \epsilon)$. By normalizing the error by the
prediction magnitude, this scorer produces intervals that scale with the level of the
series. When the target value is large, the interval is wide; when it is small, the
interval is narrow. This is the right choice for data with multiplicative seasonality
or heteroscedastic variance that grows proportionally with the signal.
[`AbsoluteGammaResidual`](/pages/api/generated/yohou.metrics.conformity.AbsoluteGammaResidual/)
is the symmetric variant.

The default conformity scorer is `Residual`. Switching scorers requires no changes to
the forecaster itself; pass a different `conformity_scorer` to the
[`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/)
constructor.


## Similarity-Based Adaptive Intervals

Standard conformal prediction uses the same set of calibration scores for every
prediction, regardless of context. This means the interval width is constant: a
prediction during a holiday peak gets the same interval as a prediction on a quiet
Tuesday. In many applications, the uncertainty genuinely varies across different
conditions.

Similarity-based weighting addresses this. Instead of treating all calibration
residuals equally when computing quantiles, it assigns higher weights to calibration
points that are "similar" to the current prediction context. The weighted quantile
then produces intervals that adapt to local conditions.

[`DistanceSimilarity`](/pages/api/generated/yohou.interval.similarity.DistanceSimilarity/)
implements this using scipy distance metrics. It computes distances between the
current prediction context and each calibration point in feature space, then converts
distances to weights using a softmax of negative distances:

$$w_{ji} = \frac{\exp(-d(x_j, x_i))}{\sum_k \exp(-d(x_j, x_k))}$$

Calibration points close to the current prediction get exponentially higher weights
than distant ones. The distance metric is configurable (euclidean, cityblock, cosine,
or any metric supported by `scipy.spatial.distance.cdist`.

To use similarity weighting with
[`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/):

```python
from yohou.interval import SplitConformalForecaster
from yohou.interval.similarity import DistanceSimilarity

forecaster = SplitConformalForecaster(
    similarity=DistanceSimilarity(metric="euclidean"),
    calibration_size=100,
)
```

The tradeoff is effective sample size. Heavily weighting a few nearby calibration
points reduces variance but can increase bias if the local neighborhood is too small.
Larger calibration sets help by providing more data points in each local region.


## Quantile Regression Intervals

[`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/)
takes a fundamentally different approach. Instead of wrapping a point forecaster and
calibrating intervals after the fact, it trains quantile regression models that
directly predict the interval bounds. For a 90% coverage rate, it fits two models:
one for the 5th percentile (lower bound) and one for the 95th percentile (upper
bound).

```python
from yohou.interval import IntervalReductionForecaster

forecaster = IntervalReductionForecaster()
forecaster.fit(y=train, forecasting_horizon=7, coverage_rates=[0.9])
intervals = forecaster.predict_interval(coverage_rates=[0.9])
```

The default estimator is `MultiOutputRegressor(QuantileRegressor())`, but any sklearn
estimator with `fit` and `predict` methods works. Multi-quantile estimators like
CatBoost with `MultiQuantile` loss are detected automatically and trained as a single
model for all quantiles simultaneously, which can be significantly faster.

Because the quantile models learn the conditional distribution directly from features,
their intervals naturally adapt to heteroscedastic data without needing explicit
similarity weighting. The disadvantage is that quantile regression does not carry the
same finite-sample coverage guarantee as conformal prediction, and its accuracy depends
entirely on how well the model captures the conditional quantiles.


## Choosing a Coverage Rate

Coverage rates are specified as floats in the range (0, 1]. Multiple rates can be
requested in a single `predict_interval` call. Higher coverage rates produce wider
intervals: a 95% interval must be wider than a 90% interval to capture more of the
distribution.

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

Interval forecasting builds on yohou's [Point Forecasting](forecasting.md)
foundation. Every interval method either wraps a point forecaster or extends the
same reduction machinery. The `observe`/`predict_interval` lifecycle mirrors the
point forecasting API, so switching between point and interval forecasts requires
minimal code changes.

For evaluating interval forecasts, see the interval metrics in
[Forecast Accuracy](forecast-accuracy.md). Coverage rate and interval width metrics
help diagnose whether intervals are well-calibrated: too narrow means the stated
coverage is not achieved, too wide means the intervals are uninformative.

For cross-validation with interval forecasters, the
[Model Selection](model-selection.md) tools work with `predict_interval` in the same
way they work with `predict`.

[`VotingIntervalForecaster`](/pages/api/generated/yohou.ensemble.voting_interval.VotingIntervalForecaster/)
provides an ensemble approach to combining prediction intervals from multiple models.
See [Ensemble Forecasting](ensemble-forecasting.md) for details.

Practical examples: [Conformal Forecasting](/examples/interval/conformal_forecasting/),
[Conformity Scorers](/examples/metrics/conformity_scorers/), and
[Distance Similarity](/examples/interval/distance_similarity/).
