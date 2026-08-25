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

### Strided Calibration

The stride-1 replay scores every calibration row as an origin. A production
system that forecasts once per day never predicts from most of those origins,
so their scores describe decisions the system does not make. The optional
`calibration_stride` parameter (default `None`, meaning the stride-1 replay)
restricts scoring to origins `calibration_stride` rows apart, tail-anchored on
the last calibration row, while the replay still observes every row so the
wrapped forecaster ends fit observed through the block.

`calibration_size` keeps meaning rows, whatever the stride. With block size $C$
and stride $k$, horizon step $h$ collects $C/k - \lceil h/k \rceil + 1$ scores:
origins near the block end contribute nothing for steps whose targets fall past
it, so the deepest step always has the fewest.

Fit validates that $C$ is a multiple of $k$ and refuses a configuration whose
deepest step would fall below the required score count, reporting the binding
coverage rate and the smallest `calibration_size` that would pass. The
requirement scales with the requested coverage rates: each step must hold at
least $\lceil m/t \rceil$ scores, where $m$ is `MIN_TAIL_SAMPLES` (3) and the
tail mass $t$ is $1 - cr$ for a symmetric conformity scorer (absolute
residuals fold both tails into one quantile) and $(1 - cr)/2$ for an
asymmetric one, so the same coverage rate needs twice the scores under signed
residuals.

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

**Dispersion-normalised residuals**:
[`NormalizedResidual`](/pages/api/generated/yohou.metrics.NormalizedResidual/)
computes $s = (y - \hat{y}) / \sigma_c$, dividing by a scale fitted per value column at
`fit` time. Where gamma residuals remove differences in *magnitude* between columns, this
also removes differences in *volatility*, so two columns of equal size but unequal noise
still produce comparable scores. That is what makes global calibration possible, and it is
also useful on its own: the interval width tracks each column's own dispersion.
[`AbsoluteNormalizedResidual`](/pages/api/generated/yohou.metrics.AbsoluteNormalizedResidual/)
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
feature vector stored during `fit()`. Each feature column is first divided by its
own spread, measured at `fit()` time. Then the distances are divided by a fitted
distance scale (the median pairwise distance among the calibration features) and by
the `bandwidth` parameter. Writing that scaled distance as
$\tilde{d} = d / (\text{bandwidth} \cdot \text{distance scale})$, the weights are a
numerically stable softmax of negative scaled distances with uniform mass reserved
for the test point:

$$w_{ji} = \frac{\exp(-(\tilde{d}_{ji} - \min_k \tilde{d}_{jk}))}{1 + \sum_k \exp(-(\tilde{d}_{jk} - \min_k \tilde{d}_{jk}))}$$

Calibration points close to the current prediction get exponentially higher weights
than distant ones. The distance metric is configurable: euclidean, cityblock, cosine,
or any metric supported by `scipy.spatial.distance.cdist`.

The two scalings each fix a different failure. Dividing by the fitted distance scale
makes the weights invariant to a rescaling of the data: without it, the softmax has no
width of its own, so how concentrated the weights are depends on the units. The same
series measured in thousands rather than units would put nearly all its weight on a
single calibration row, and the resulting interval would be read off one conformity
score. Dividing each column by its own spread stops a single high-magnitude column
from deciding the neighbourhood for every other column, which is what happens on a
panel whose entities differ in size.

`bandwidth` is the knob left over once the scale is handled. Below 1 concentrates
weight on nearer calibration rows, above 1 flattens toward uniform. Reach for it when
the default neighbourhood is wider or narrower than the structure you know is in your
data.

The two similarities default it differently, and the reason is worth knowing.
`DistanceSimilarity` defaults to `1.0`. `SeasonalSimilarity` defaults to `0.5`, because
its harmonic features are bounded on the unit circle: their median pairwise distance is
around 1.5, so dividing by it flattens a spread that was already narrow. At `1.0` a
weekly seasonality keeps roughly 41 of 50 calibration rows as effective sample size,
near enough to uniform that the weighted quantile usually lands on the same order
statistic as the unweighted one, which makes the similarity do nothing visible. At `0.5`
it keeps roughly 25 and the weighting bites.

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

The adapter tracks one level per horizon step and value column (or one level per
value column shared across steps with `alpha_pooling="shared"`), and one level per
tail for asymmetric conformity scorers so a lopsided error distribution corrects
each side separately. `alpha_pooling` pools along the horizon axis only: two
entities' levels are never fused, so a chronically miscovered entity widens its own
intervals and nobody else's.

Which value to pick is a question about what you are willing to assume. The default
resolves each horizon separately, which matches the intuition that a one-step and a
twelve-step forecast drift differently, but running one recursion per horizon over
overlapping data is a pragmatic extension rather than something the original result
covers. `"shared"` collapses them to a single trajectory per entity, which stays
inside the single-sequence setting the theory addresses, at the cost of horizon
resolution.

Under `"shared"` the forecaster allocates one adapter per value column and points
every horizon-step key at that same object, so `adapters_["step_1"][col]` and
`adapters_["step_2"][col]` are the same adapter rather than identical copies of it.

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
  maintains per-group transformer state and observation buffers. For independent
  per-group models, use
  [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.LocalPanelForecaster/).
- **`"multivariate"`**: skips panel detection entirely and treats `__`-prefixed
  columns as ordinary multivariate columns, so one transformer and one model see
  the full wide frame. Use it when cross-group feature interactions matter.

The choice mirrors the panel strategies available in point forecasting. See
[Panel Data](panel-data.md) for the full treatment.

### What `panel_strategy` does not govern

`panel_strategy` decides how the *point model* is fitted and how per-group state
is kept. It does not decide the calibration axis. Conformal calibration is always
per value column, under either strategy: each `{group}__{variable}` column takes
the quantile of its own conformity scores, and the adaptive level, when an
`adapter` is configured, is likewise tracked per column.

This matters when entities differ in magnitude. A store selling 50 units a day
and one selling 5,000 need interval widths two orders of magnitude apart. Sharing
one calibration across them would over-cover the small entity and under-cover the
large one, and the sharing would be invisible: both intervals are well-formed, one
is simply too wide and the other too narrow.

Calibration can optionally be shared across entities: global calibration, enabled with
`calibration_strategy="global"`. It is off by default, and whether it helps depends
entirely on your data. Three things govern that, and they are not equally tractable.

The first is magnitude. Pooling raw residuals from entities of different size is what
produces the failure above, and a scale-free score such as
[`GammaResidual`](/pages/api/generated/yohou.metrics.conformity.GammaResidual/), which
divides by the predicted level, removes it.

The second is volatility, and the level-based score does not remove it. Two entities of
the same size whose errors differ threefold in spread still contribute incomparable
scores. Making them comparable needs a score divided by a per-entity dispersion
estimate, which yohou does not currently provide.

The third is dependence, and no conformity score removes it. Entities observed at the
same timestamp share shocks, so their scores are correlated within a timestamp and much
less so across timestamps. A combined sequence with that block structure is not
exchangeable, which is the assumption the finite-sample guarantee rests on. It also
means global calibration buys far less than the entity count suggests: under same-timestamp
correlation `rho` the effective gain saturates near `1 / rho`, so two hundred entities
that move together are worth about as much as ten.

The first is solved by
[`NormalizedResidual`](/pages/api/generated/yohou.metrics.NormalizedResidual/), which
divides each residual by that column's own dispersion rather than by its predicted
level. Global calibration requires it, or another scorer declaring cross-column comparability, and
`calibration_strategy="global"` raises at fit otherwise: pooling incomparable scores
produces an interval that is wrong rather than merely imprecise.

The second and third are not solved, only bounded, which is why the mode is opt-in.

### Deciding whether to calibrate globally

Do not decide from the numbers above. Measure your own data with
[`diagnose_global_calibration`](/pages/api/generated/yohou.interval.diagnose_global_calibration/), which
reports the cross-sectional correlation of a fitted forecaster's conformity scores and
how comparable those scores are across columns. It reports and does not choose, because
the right answer also depends on which coverage rates you need.

Global calibration earns its place when a coverage rate is out of reach per column. On
`fetch_hospital`, 40 series with 28 calibration scores each at a nominal 99%:

| strategy | realized coverage |
| --- | --- |
| `"local"` | 92.3%, and the forecaster warns the rate is unreachable |
| `"global"` | 99.6% |

It costs something when an entity's errors differ in shape from its neighbours, not just
in scale, since normalization equalizes dispersion and not tail weight. With nine
light-tailed series and one heavy-tailed one at a nominal 90%, the heavy-tailed entity
falls from 88.9% under `"local"` to 85.9% under `"global"`: its interval is drawn largely
from better-behaved neighbours. The light-tailed nine are unaffected.

Note that `panel_strategy` and `calibration_strategy` both accept `"global"` and neither
implies the other. The first shares the point *model* across entities, the second shares
the *calibration*. A shared model with per-entity calibration is the default and a
perfectly sensible pairing.

The adapter has no `entity_pooling` parameter alongside its `alpha_pooling`, and the
adaptive level stays per entity under both calibration strategies.

Note also that entity count never starves calibration here. `calibration_size` slices
rows, and every entity shares the time index, so each entity receives the full
calibration count no matter how many entities there are. What can starve it is a short
history: a coverage rate of `1 - alpha` needs at least `1/alpha - 1` calibration scores
to be expressible at all, meaning 19 for 95% and 99 for 99%.

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

### How many calibration scores a rate needs

A conformal bound is an order statistic of the calibration scores, specifically the
`ceil((n + 1) * q)`-th, where `q` is the tail level. That index exists only while it
stays within `n`, which means a rate needs at least `q / (1 - q)` scores per value
column to be representable at all: 9 for a symmetric 90%, 99 for a symmetric 99%, and
roughly double each for an asymmetric scorer, which splits the miscoverage across two
tails. Beyond that point the correct bound is unbounded, and any finite number the
implementation returns under-covers.

`SplitConformalForecaster` warns when you ask for a rate its calibration set cannot
express, naming the horizon step and the count you would need. Note that the constraint
is per value column but `calibration_size` slices rows, so every entity of a panel gets
the full count no matter how many entities there are. A short history is what runs you
out, not a wide panel.

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
