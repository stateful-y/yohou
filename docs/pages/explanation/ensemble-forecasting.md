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

## Aggregation Methods

### Point Ensembles

[`VotingPointForecaster`](/pages/api/generated/yohou.ensemble.voting_point.VotingPointForecaster/) combines point predictions using the **mean** (default) or
**median**.

The mean is optimal under squared-error loss when base forecasters have Gaussian
errors. Weights can be set manually (e.g., inversely proportional to validation
error) or left uniform. The median is robust to outliers: one rogue forecaster
cannot pull the ensemble off course, making it a safer default when base models
have heavy-tailed error distributions.

### Interval Ensembles

[`VotingIntervalForecaster`](/pages/api/generated/yohou.ensemble.voting_interval.VotingIntervalForecaster/) combines prediction intervals using three methods:

**Envelope** (default) takes the minimum of all lower bounds and the maximum of
all upper bounds. This guarantees that the ensemble interval contains every
individual interval, producing wider (more conservative) intervals. Useful when
undercoverage is costly.

**Mean** averages the lower and upper bounds separately, producing intervals
closer to the average width. This can undercover if individual models are already
miscalibrated.

**Median** takes the median of each bound, offering robustness to outlier
intervals.

### Class-Probability Ensembles

[`VotingClassProbaForecaster`](/pages/api/generated/yohou.ensemble.voting_class_proba.VotingClassProbaForecaster/) combines probability distributions using **soft voting**
(default) or **hard voting**.

Soft voting averages class probabilities across base forecasters. It preserves
calibration better than hard voting because it operates on the full probability
simplex. If one model is 80% confident and another is 20% confident about the
same class, their average reflects genuine uncertainty.

Hard voting lets each base forecaster vote for its argmax class, and the majority
class wins. This discards probability information and is generally inferior to
soft voting, but it allows ensembling forecasters that do not produce
well-calibrated probabilities.

## When Ensembles Help

Ensembles are most effective when:

- Base models make **different errors** (diversity). Combining five linear models
  trained on the same features with the same regularization provides negligible
  improvement over a single one.
- The prediction task has **moderate noise**. When signal-to-noise is very high, a
  single well-specified model suffices. When noise dominates, even ensembles struggle.
- You can afford the **computational cost**. Ensembles multiply training and
  inference time by the number of base models.

Diminishing returns set in quickly. Going from 1 to 3 models often captures most of
the ensemble benefit. Going from 10 to 20 rarely helps.

## Fault Tolerance

Voting forecasters handle base model failures gracefully. If a base forecaster raises
an exception during `fit()`, it is excluded from the ensemble with a warning rather
than crashing the entire pipeline. The ensemble continues with the remaining
forecasters. This behavior is always enabled and cannot be disabled.

This makes ensembles practical in production settings where one model may fail on
unusual data while others succeed. It also means a four-model ensemble can
degrade to three models silently, so monitoring which base models are active
matters.

## Connections

- **Model Selection**: Use [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) to tune ensemble weights or compare an
  ensemble against its individual members
- **Composition**: [`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/) can preprocess data before passing to an
  ensemble, and ensembles can serve as components in larger [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/)
  workflows
- **Panel data**: All voting forecasters support panel data transparently. Each
  base forecaster receives the full panel, and aggregation happens per-group.

### Related pages

- [Class-Probability Forecasting](class-probability-forecasting.md): categorical ensembles
- [Interval Forecasting](interval-forecasting.md): interval ensemble context
- [How to Combine Forecasters with Ensembles](../how-to/ensemble-forecasting.md): step-by-step guide
- [API Reference: yohou.ensemble](../api/ensemble.md)
- [Ensemble Examples](../examples/ensemble.md)
