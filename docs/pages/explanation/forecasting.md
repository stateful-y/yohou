# Forecasting

Yohou's central insight is that time series forecasting can be treated as a supervised
learning problem. Instead of designing specialized forecasting algorithms from scratch,
yohou converts the temporal structure of a time series into rows and columns that any
scikit-learn regressor can learn from. This page explains how that conversion works,
what choices it involves, and where the tradeoffs lie.

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## The Forecasting Workflow

The forecasting lifecycle is cyclical rather than sequential. Problem definition establishes what variable to predict, over what horizon, and which decisions the forecast will drive, shaping every subsequent choice from data granularity to evaluation criteria.

Data preparation addresses the realities of collecting and cleaning historical observations and exogenous predictors. Common scenarios such as missing values, outliers, and frequency mismatches are covered in [Practical Issues](practical-issues.md). Exploration of the cleaned series reveals [temporal patterns](time-series-patterns.md) such as trend, seasonality, cycles, and structural breaks, which guide the choice of transformers and model configurations.

Method selection and evaluation form the core iterative loop. A candidate configuration is fitted and its accuracy measured using temporal cross-validation and appropriate [accuracy metrics](forecast-accuracy.md). Unsatisfactory results send the process back to exploration or data preparation rather than to model tuning alone. [Model Selection](model-selection.md) covers strategies for navigating this cycle efficiently.

Production deployment uses the `observe`/`predict` lifecycle to generate new forecasts as observations arrive. [Residual Diagnostics](residual-diagnostics.md) tracks whether the model continues to perform well over time, and significant degradation initiates a return to the earlier phases.


## Forecasting as Supervised Learning Reduction

A forecaster needs to predict future values given past observations. A regressor needs to
predict target values given feature columns. These two problems have the same shape;
the difference is just how the inputs are arranged.

The reduction approach makes this explicit. Given a time series like
`[10, 20, 30, 40, 50]`, a reduction forecaster slides a window over the data to produce
training samples:

| Past values (features) | Future value (target) |
|---|---|
| 10, 20, 30 | 40 |
| 20, 30, 40 | 50 |

Each row is one training example. The past values become feature columns; the future
values become targets. Once the data is in this form, any sklearn regressor (linear
regression, random forests, gradient boosting) can learn the mapping from past to
future.

[`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/)
implements this idea. It accepts an `estimator` parameter (any sklearn regressor) and
handles the conversion internally. A `LinearRegression` forecaster and a
`GradientBoostingRegressor` forecaster share the same tabularization logic; only the
learning algorithm differs:

```python
from sklearn.ensemble import GradientBoostingRegressor
from yohou.point import PointReductionForecaster

forecaster = PointReductionForecaster(
    estimator=GradientBoostingRegressor(),
)
forecaster.fit(y=train, forecasting_horizon=7)
predictions = forecaster.predict(forecasting_horizon=7)
```

This is the main advantage of the reduction approach: the full ecosystem of sklearn
regressors, including hyperparameter tuning and model selection, becomes available for
forecasting with no additional implementation.


## Tabularization

The conversion from time series to tabular format is handled by
[`tabularize()`](/pages/api/generated/yohou.utils.tabularization.tabularize/). It takes
a DataFrame and a sequence of lag values, then produces a new DataFrame where each
row contains shifted versions of the original series.

For a concrete example, given a series `[10, 20, 30, 40, 50]` and `lags=[1, 2]`,
`tabularize` produces:

| value_lag_1 | value_lag_2 |
|---|---|
| 20 | 10 |
| 30 | 20 |
| 40 | 30 |

Each lag shifts the series by that many steps. `value_lag_1` is the value one step before
the current row; `value_lag_2` is two steps before. The first `max(lags)` rows are dropped
because they would contain nulls.

Inside
[`BaseReductionForecaster`](/pages/api/generated/yohou.base.reduction.BaseReductionForecaster/),
tabularization serves two roles. First, it creates the target matrix: future values at
lags `[1, 2, ..., H]` are renamed from `lag_1` to `step_1`, `lag_2` to `step_2`, and so
on. Second, it participates in building the feature matrix, which includes lagged versions
of both the target and any exogenous variables. The last `H` rows of the feature matrix
are discarded because they do not have corresponding future targets.

The forecasting horizon `H` determines the shape of the target matrix. With `H=3`, each
training sample has three target columns representing one-step-ahead, two-step-ahead, and
three-step-ahead predictions. How those columns are used depends on the reduction
strategy.


## Reduction Strategies

The `reduction_strategy` parameter on
[`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/)
controls how the estimator relates to the multi-step horizon. There are three options.

**Multi-output** (`"multi-output"`, the default) trains a single model that predicts all
`H` horizon steps simultaneously. The target matrix has shape `(n_samples, H)`, and the
regressor learns to produce all steps at once. This is the simplest and fastest strategy.
It works well when the relationship between features and targets is similar across horizon
steps, but it asks one model to handle both near-term and far-term predictions with the
same parameters.

**Direct** (`"direct"`) fits `H` independent models, one per horizon step. Model 1
specializes in one-step-ahead, model 2 in two-step-ahead, and so on. Each model sees the
same features but trains on a different target column. This avoids the constraint of a
single model covering all steps, and it naturally sidesteps error accumulation since
each model predicts directly from the original features rather than from prior
predictions. The cost is computational: fitting `H` models takes roughly `H` times
longer. The `n_jobs` parameter enables parallel fitting to offset this.

**Dir-rec** (`"dir-rec"`) is a direct-recursive hybrid. Like the direct strategy, it fits
`H` models sequentially. But model `h` receives an augmented feature matrix that includes
the in-sample predictions from models `1` through `h-1`. This lets later models
incorporate information about the predicted trajectory so far, combining per-step
specialization with inter-step information flow. The augmentation happens at training time
using in-sample predictions, so each model sees realistic inputs rather than perfect
future values.

The choice depends on the problem. Multi-output is a good default for short horizons and
fast iteration. Direct is worth considering when error accumulation is a concern or when
the relationship between features and target changes substantially across horizon steps.
Dir-rec adds complexity but can improve accuracy on longer horizons where step-to-step
dependencies matter.


## Target and Feature Transformers

Reduction forecasters support two transformer pipelines that serve distinct purposes.

The **target transformer** (`target_transformer`) operates on `y` before tabularization.
Its job is to transform the prediction target into a space where the regressor can learn
more effectively. Common examples include
[`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/)
(which removes seasonal patterns) and
[`LogTransformer`](/pages/api/generated/yohou.stationarity.transformers.LogTransformer/)
(which stabilizes variance in exponentially growing series). After the regressor
produces predictions in the transformed space, the forecaster automatically applies
`inverse_transform` to return predictions to the original scale.

The **feature transformer** (`feature_transformer`) creates additional input features
from the (possibly already transformed) target. Transformers like
[`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/) and
[`RollingStatisticsTransformer`](/pages/api/generated/yohou.preprocessing.window.RollingStatisticsTransformer/)
produce lagged values, moving averages, or other derived signals that the regressor uses
as predictors. These features are never inverted; they flow into the regressor as
inputs, not outputs.

The distinction matters because it determines what the regressor learns. A target
transformer changes the *question* being asked (predict differenced values instead of
raw values). A feature transformer changes the *information* available to answer it
(give the regressor rolling statistics alongside raw lags). In practice, many forecasters
use both: a target transformer for stationarity and a feature transformer for richer
input signals.

The `target_as_feature` parameter controls one more detail: whether the target itself
(raw or transformed) appears among the features. The default (`"transformed"`) includes
the transformed target as a feature column alongside whatever the feature transformer
produces. Setting it to `"raw"` includes the pre-transformation target instead, which
can be useful when the regressor benefits from seeing original values even though the
prediction happens in the transformed space. Setting it to `None` uses only exogenous
features and feature transformer outputs, excluding the target entirely.


## Window Length and Observation Horizon

Every transformer has an `observation_horizon` that declares how many past time
steps it needs (see [Core Concepts](core-concepts.md#observation-horizon) for the
full mechanism). The forecaster computes an effective observation horizon as the
maximum across all attached transformers and uses it to maintain a fixed-size
sliding window of recent data.

The practical question for reduction forecasting is: how much history should the
regressor see? The transformers you choose set a hard minimum. A
[`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/)
with `lags=[1, 7]` needs at least 7 rows. Adding a
[`RollingStatisticsTransformer`](/pages/api/generated/yohou.preprocessing.window.RollingStatisticsTransformer/)
with `window_size=14` pushes the requirement to 14.

Beyond the minimum, adding more context is a tradeoff. Longer lookback windows
give the regressor access to older patterns and longer-range dependencies, which
helps when the series has slow-moving dynamics. But they also introduce older,
potentially irrelevant data that can dilute the signal. In practice, the window
length is determined by the transformer configuration, and you adjust it by
choosing transformers with appropriate lookback requirements.


## Incremental Observation

Once a forecaster is fitted, new data arrives incrementally. The `observe` method
appends fresh rows to the forecaster's internal buffers without refitting, and
`rewind` trims those buffers back to the observation horizon. The composite
`observe_predict` combines both steps: it updates state with newly arrived data,
then generates predictions from the updated observation point.

This pattern is central to rolling evaluation. A cross-validation loop calls
`observe_predict` repeatedly, advancing one stride at a time through the test set.
Each call produces one *vintage*: a set of predictions anchored to a specific
`vintage_time`. The resulting predictions can be scored per vintage (vintagewise)
or per horizon step (stepwise) to diagnose where and when the model struggles.
See [Forecast Accuracy](forecast-accuracy.md#vintage-based-evaluation) for details.


## Naive Baselines

Every forecasting exercise should start with simple benchmarks. Naive methods require
no model fitting yet are surprisingly hard to beat on many series. They serve two
purposes: as sanity checks (any useful model must outperform them) and as components
in combination forecasts and decomposition pipelines.

[`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/) is the most
important baseline in time series forecasting. It predicts by repeating the last
seasonal cycle: with `seasonality=7` on daily data, each forecast day copies the value
from the same weekday in the previous week. When `seasonality=1`, it reduces to a
standard naive (random walk) forecast that repeats the last observed value.

The classical forecasting literature describes four naive methods: random walk, seasonal
naive, mean, and drift. Yohou implements random walk and seasonal naive as special
cases of `SeasonalNaive` (with `seasonality=1` and `seasonality=k` respectively). Mean
and drift forecasters are not implemented as standalone classes because
`PointReductionForecaster` with a simple regressor (intercept-only linear regression
for mean, linear regression on a time index for drift) achieves the same result while
fitting naturally into the reduction framework.

`SeasonalNaive` implements the same API as reduction forecasters (`fit`, `predict`,
`observe`, `observe_predict`) but without tabularization or a regressor. It simply
stores the last `seasonality` observations and cycles through them. This makes it a
drop-in replacement for comparison.

A reduction forecaster that cannot beat `SeasonalNaive` is adding complexity without
value: the regressor is failing to learn anything beyond repeating recent history.
Conversely, the margin by which a forecaster exceeds the naive baseline quantifies
the actual contribution of the features and model. The scaled metrics
[`MeanAbsoluteScaledError`](/pages/api/generated/yohou.metrics.point.MeanAbsoluteScaledError/)
and
[`RootMeanSquaredScaledError`](/pages/api/generated/yohou.metrics.point.RootMeanSquaredScaledError/)
formalize this comparison by expressing forecast error as a ratio to the naive
baseline's error.


## References

- Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*,
  3rd edition. OTexts. [Chapter 1](https://otexts.com/fpp3/intro.html) (forecasting
  workflow), [Chapter 5.2](https://otexts.com/fpp3/simple-methods.html) (naive
  benchmarks), [Chapter 5.1](https://otexts.com/fpp3/forecasting-residuals.html)
  (training-set evaluation).
- Bontempi, G., Ben Taieb, S., & Le Borgne, Y.-A. (2013). Machine learning strategies
  for time series forecasting. *European Business Intelligence Summer School*, 62-77.


## Connections

The reduction approach described here produces **point forecasts**: single-valued
predictions for each future time step. For prediction intervals that quantify
uncertainty, see [Interval Forecasting](interval-forecasting.md), which covers
conformal prediction and similarity-based interval estimation built on top of the same
reduction machinery.

The transformers mentioned above (target transformers for stationarity, feature
transformers for signal enrichment) are discussed in depth in
[Preprocessing](preprocessing.md) and [Stationarity](stationarity.md).

Practical examples: [Reduction Forecaster](/examples/point/reduction_forecaster/) walks
through building a basic reduction forecaster, and
[Reduction Strategies](/examples/point/reduction_strategies/) compares multi-output,
direct, and dir-rec on the same dataset.
[Panel Reduction Forecasting](/examples/point/panel_reduction/) demonstrates panel
strategies (global, multivariate, local) for multi-entity data.

The reduction pattern extends naturally to categorical targets through
[`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.reduction.ClassProbaReductionForecaster/), which wraps sklearn classifiers instead of
regressors. See [Class-Probability Forecasting](class-probability-forecasting.md)
for the full treatment.
