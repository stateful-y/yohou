# Ensemble Forecasting

A single forecaster captures one view of the data-generating process. Its errors
are systematic: model misspecification, sensitivity to outliers, overfitting to
training patterns. Ensemble methods combine predictions from multiple forecasters
to reduce these errors through diversity.

The core idea is variance reduction through aggregation. If base forecasters make
uncorrelated errors, averaging their predictions cancels out individual mistakes
while preserving the shared signal. Yohou implements this through three voting
forecasters, one for each prediction type: [`VotingPointForecaster`](/pages/api/generated/yohou.ensemble.voting_point.VotingPointForecaster/),
[`VotingIntervalForecaster`](/pages/api/generated/yohou.ensemble.voting_interval.VotingIntervalForecaster/), and [`VotingClassProbaForecaster`](/pages/api/generated/yohou.ensemble.voting_class_proba.VotingClassProbaForecaster/).

## Variance Reduction Through Diversity

Consider $K$ base forecasters producing predictions $\hat{y}_{t,1}, \ldots, \hat{y}_{t,K}$.
If each has expected error variance $\sigma^2$ and pairwise correlation $\rho$, the
variance of their average is:

$$\text{Var}\left(\frac{1}{K}\sum_{k=1}^K \hat{y}_{t,k}\right) = \frac{\sigma^2}{K}(1 + (K-1)\rho)$$

When base models are perfectly correlated ($\rho = 1$), averaging provides no benefit.
When they are uncorrelated ($\rho = 0$), variance shrinks by a factor of $K$. In
practice, forecasters trained on the same data are always somewhat correlated, but
using different model families (e.g., linear + tree-based + naive) or different feature
sets increases diversity.

The bias of the ensemble is the average bias of the base forecasters. Ensembles do
not fix systematic bias; they reduce variance. This is the bias-variance perspective
on why ensembles work.

## Bias-Variance Decomposition

The expected loss of any forecaster can be decomposed into three terms:

$$\mathbb{E}[(\hat{y}_t - y_t)^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

**Bias** measures how far the model's average prediction is from the true value across
repeated training runs. A model that consistently under-predicts demand by 10% has high
bias.

**Variance** measures how much the prediction changes across different training samples.
A deep decision tree memorizes training data and produces wildly different forecasts
when retrained on a slightly different sample.

**Irreducible noise** is the inherent randomness in the data that no model can capture.

Averaging $K$ forecasters leaves bias unchanged (the average of $K$ biased estimates
is still biased by the same amount) but reduces variance. The reduction depends on
how correlated the errors are. This is the fundamental reason ensembles work: they
trade zero increase in bias for potentially large decreases in variance.

In time series forecasting, variance is often the dominant source of error for
flexible models (gradient-boosted trees, neural networks, reduction forecasters with
many lags). Simpler models like [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/) have high bias but low variance.
Ensembling across the complexity spectrum (simple + flexible) can balance both terms.

## Achieving Diversity

Diversity is the critical ingredient. Two forecasters making identical errors provide
no benefit when combined. The goal is forecasters that are individually accurate but
make errors in different places.

### Different Model Families

The most reliable way to achieve diversity is combining structurally different models.
A linear model, a tree-based model, and a seasonal naive forecaster capture different
aspects of the data:

- **Linear models** (Ridge, ElasticNet) capture trends and linear feature relationships
  but miss nonlinear interactions
- **Tree-based models** (LightGBM, XGBoost, RandomForest) capture nonlinear patterns
  and interactions but can overfit to training noise
- **Naive models** (SeasonalNaive) capture strong seasonal patterns with zero variance
  but cannot adapt to trends or exogenous effects

### Different Feature Sets

Even within the same model family, using different feature transformers creates
diversity. One forecaster with lag features and another with Fourier seasonality
terms will capture different signal components:

```python
from sklearn.linear_model import Ridge

from yohou.ensemble import VotingPointForecaster
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer, FourierFeatureTransformer

lag_forecaster = PointReductionForecaster(
    estimator=Ridge(),
    feature_transformer=LagTransformer([1, 2, 3, 7, 14]),
)
fourier_forecaster = PointReductionForecaster(
    estimator=Ridge(),
    feature_transformer=FourierFeatureTransformer(seasonality=7, harmonics=[1, 2, 3]),
)
ensemble = VotingPointForecaster(
    forecasters=[("lag", lag_forecaster), ("fourier", fourier_forecaster)],
)
```

### Different Training Windows

Training on different historical windows introduces diversity through exposure to
different data regimes. A forecaster trained on the last 6 months emphasizes recent
patterns, while one trained on 2 years captures longer cycles. Since Yohou fits
each base forecaster on the same `y` passed to `fit()`, this strategy requires
slicing the training data before constructing the ensemble, giving each base
forecaster a different view of history.

## Aggregation Methods

All voting forecasters accept an optional `weights` parameter (a list of floats,
one per base forecaster). Weights do not need to sum to 1; they are normalized
internally. Weights only apply to mean-based aggregation. Median and envelope
methods ignore them.

### Point Ensembles

[`VotingPointForecaster`](/pages/api/generated/yohou.ensemble.voting_point.VotingPointForecaster/) combines point predictions using two methods controlled by the `method` parameter:

**Mean** (default) computes a weighted average of predictions across base
forecasters. With uniform weights this is optimal under squared-error loss when
base forecasters have Gaussian errors. Setting weights inversely proportional to
validation error gives better-performing models more influence.

**Median** takes the median prediction, ignoring weights. One rogue forecaster
cannot pull the ensemble off course, making median a safer choice when base models
have heavy-tailed error distributions.

### Interval Ensembles

[`VotingIntervalForecaster`](/pages/api/generated/yohou.ensemble.voting_interval.VotingIntervalForecaster/) combines prediction intervals using three methods controlled by the `method` parameter:

**Envelope** (default) takes the minimum of all lower bounds and the maximum of
all upper bounds. This guarantees that the ensemble interval contains every
individual interval, producing wider (more conservative) intervals. Useful when
undercoverage is costly. Weights are ignored.

**Mean** computes a weighted average of the lower and upper bounds separately,
producing intervals closer to the average width. This can undercover if individual
models are already miscalibrated.

**Median** takes the median of each bound, ignoring weights. Offers robustness to
outlier intervals without the conservatism of envelope.

When base forecasters also support point predictions, `VotingIntervalForecaster`
exposes `predict()` alongside `predict_interval()`. Point predictions are
aggregated separately using the `point_method` parameter (`"mean"` or `"median"`),
independent of the interval `method`.

### Class-Probability Ensembles

[`VotingClassProbaForecaster`](/pages/api/generated/yohou.ensemble.voting_class_proba.VotingClassProbaForecaster/) combines probability distributions using two methods controlled by the `method` parameter:

**Soft voting** (default) computes a weighted average of class probabilities across
base forecasters. It preserves calibration better than hard voting because it
operates on the full probability simplex. If one model assigns 80% probability and
another assigns 20% to the same class, their average reflects genuine uncertainty.

**Hard voting** lets each base forecaster vote for its argmax class, and the
majority class wins. Weights are ignored. Ties are broken deterministically by
alphabetical class order. Hard voting discards probability information and is
generally inferior to soft voting, but it allows ensembling forecasters that do not
produce well-calibrated probabilities.

All base forecasters in a class-probability ensemble must discover the same set of
classes during training. If they disagree, fitting raises a `ValueError`.

## Fault Tolerance

If a base forecaster raises an exception during fitting, the ensemble skips it with
a warning rather than failing entirely. The remaining forecasters continue, and
weights are automatically adjusted to account for the survivors. Only if every base
forecaster fails does the ensemble raise a `RuntimeError`. This makes ensembles
more robust in production settings where a single model configuration might fail on
certain data splits without bringing down the entire pipeline.

## When Ensembles Help

Ensembles are most effective when:

- Base models make **different errors** (diversity). Combining five linear models
  trained on the same features with the same regularization provides negligible
  improvement over a single one.
- The prediction task has **moderate noise**. When the signal is very strong, a
  single well-specified model suffices. When noise dominates, even ensembles struggle.
- You can afford the **computational cost**. Ensembles multiply training and
  inference time by the number of base models. Setting `n_jobs` to fit base
  forecasters in parallel helps, but memory usage still scales linearly.

Diminishing returns set in quickly. Going from 1 to 3 models often captures most of
the ensemble benefit. Going from 10 to 20 rarely helps.

## Connections

[`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) can tune ensemble weights or compare an ensemble against its individual members, as described in [Model Selection](model-selection.md). [`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/) can preprocess data before passing to an ensemble, and ensembles can serve as components in larger [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) workflows. All voting forecasters support panel data transparently: each base forecaster receives the full panel, and aggregation happens per group.

[Class-Probability Forecasting](class-probability-forecasting.md) covers categorical ensembles, and [Interval Forecasting](interval-forecasting.md) discusses interval ensemble context. For practical recipes, see [How to Combine Forecasters with Ensembles](../how-to/ensemble-forecasting.md). The full API is documented in the [yohou.ensemble reference](../api/ensemble.md), and interactive examples are available in the [Ensemble Examples](../examples/forecasting-models.md).
