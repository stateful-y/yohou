# Reduction Forecasting

Yohou implements several forecasting approaches that share a common API (`fit`,
`predict`, `observe`, `observe_predict`) but differ in how they generate predictions.
The most flexible is the **reduction approach**, which converts time series into tabular
data so that any Scikit-Learn estimator can power a forecaster. This page focuses on
the reduction approach: how tabularization works, what strategy options are available,
and how target and feature transformers shape the learning problem. For decomposition
and composition, see [Forecaster Composition](forecaster-composition.md).

## The Reduction Approach

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
handles the conversion internally. A [`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) forecaster and a
[`GradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) forecaster share the same tabularization logic; only the
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
[`tabularize`](/pages/api/generated/yohou.utils.tabularization.tabularize/) produces:

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
tabularization builds the target matrix. It tabularizes the (transformed) target `y`
with lags `[0, 1, ..., H]`, then renames the columns so that `step_1` is the
one-step-ahead value, `step_2` is two steps ahead, and so on. The feature matrix is
built separately by the feature transformer pipeline (see
[Target and Feature Transformers](#target-and-feature-transformers) below), and the
last `H` rows are discarded because they have no corresponding future targets.

The forecasting horizon `H` determines the shape of the target matrix. With `H=3`, each
training sample has three target columns representing one-step-ahead, two-step-ahead, and
three-step-ahead predictions. For multivariate targets, each component gets its own set
of step columns, so the target matrix has `H * n_targets` columns. How those columns are
used depends on the reduction strategy.

## Reduction Strategies

The `reduction_strategy` parameter on
[`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/)
controls how the estimator relates to the multi-step horizon. There are three options.
In all cases, $\mathbf{x}_t$ denotes the full feature vector at time $t$. This vector
is assembled from the feature transformer output (lagged values of the target and any
`X_actual` exogenous columns), plus forward step columns from `X_future` and
`X_forecast`. Its exact composition depends on the transformer configuration,
`target_as_feature`, and which exogenous inputs are provided (see
[Target and Feature Transformers](#target-and-feature-transformers) below and
[Exogenous Features](exogenous-features.md) for details).

**Multi-output** (`"multi-output"`, the default) trains a single model that predicts all
$H$ horizon steps simultaneously:

$$[\hat{y}_{t+1},\; \hat{y}_{t+2},\; \ldots,\; \hat{y}_{t+H}] = f(\mathbf{x}_t)$$

The target matrix has shape $(n_\text{samples},\; H \times n_\text{targets})$, and the
regressor learns to produce all steps at once. The estimator must natively support
multi-output (as `LinearRegression`, `RandomForestRegressor`, and
`DecisionTreeRegressor` do). For estimators that only handle single-output targets,
wrap them with
[`MultiOutputRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html)
or use the direct strategy instead. Multi-output is the simplest and fastest strategy.
It works well when the relationship between features and targets is similar across horizon
steps, but it asks one model to handle both near-term and far-term predictions with the
same parameters.

**Direct** (`"direct"`) fits $H$ independent models, one per horizon step:

$$\hat{y}_{t+h} = f_h(\mathbf{x}_t) \quad \text{for } h = 1, 2, \ldots, H$$

Model 1 $f_1$
specializes in one-step-ahead, model 2 $f_2$ in two-step-ahead, and so on. Each model sees the
same features but trains on a different target column. This avoids the constraint of a
single model covering all steps, and it naturally sidesteps error accumulation since
each model predicts directly from the original features rather than from prior
predictions. The cost is computational: fitting $H$ models takes roughly $H$ times
longer. The `n_jobs` parameter enables parallel fitting to offset this.

When the feature vector includes step-indexed exogenous columns (from `X_future` or
`X_forecast`), the `step_feature_alignment` parameter controls which step columns each
model sees. `"all"` (the default) gives every model all step columns. `"matched"` gives
model $f_h$ only the columns for step $h$, so it sees only the exogenous value at its
own prediction time. `"cumulative"` gives model $f_h$ the step columns for steps 1
through $h$, progressively expanding the information available to later models.

**Dir-rec** (`"dir-rec"`) is a direct-recursive hybrid that fits $H$ models sequentially,
where each model receives the original embedding augmented with in-sample predictions
from all previous models:

$$\hat{y}_{t+h} = f_h(\mathbf{x}_t,\; \hat{y}_{t+1},\; \hat{y}_{t+2},\; \ldots,\; \hat{y}_{t+h-1}) \quad \text{for } h = 1, 2, \ldots, H$$

This lets later models
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
from the target and any `X_actual` exogenous columns. What the feature transformer
receives as input depends on `target_as_feature`: when set to `"transformed"` (the
default), it receives the transformed target concatenated with `X_actual`; when set to
`"raw"`, it receives the original untransformed target instead. Transformers like
[`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/) and
[`RollingStatisticsTransformer`](/pages/api/generated/yohou.preprocessing.window.RollingStatisticsTransformer/)
produce lagged values, moving averages, or other derived signals that the regressor uses
as predictors. These features are never inverted; they flow into the regressor as
inputs, not outputs. After the feature transformer runs, any step-indexed columns from
`X_future` and `X_forecast` are joined onto the result, bypassing the transformer
entirely.

The distinction matters because it determines what the regressor learns. A target
transformer changes the *question* being asked (predict differenced values instead of
raw values). A feature transformer changes the *information* available to answer it
(give the regressor rolling statistics alongside raw lags). In practice, many forecasters
use both: a target transformer for stationarity and a feature transformer for richer
input signals.

The `target_as_feature` parameter controls whether the target series appears among
the feature transformer's inputs. The default (`"transformed"`) feeds the transformed
target (after any target transformer) alongside `X_actual` into the feature
transformer. Setting it to `"raw"` feeds the original untransformed target instead,
which can be useful when the regressor benefits from seeing original values even though
the prediction target is in the transformed space. Setting it to `None` excludes the
target entirely and passes only `X_actual` to the feature transformer, which requires
that `X_actual` is provided when a feature transformer is set.

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

## Prediction at Inference Time

At prediction time, the forecaster does not re-tabularize the full training set. During
`fit()` (and each subsequent `observe()` call), the forecaster stores the last row of
the transformed feature matrix as `_X_t_observed`. When `predict()` is called, this
single-row feature vector is passed to the fitted estimator's `predict()`, producing
an output of shape $(1, H \times n_\text{targets})$. The forecaster reshapes this into
one predicted value per step and per target column, then applies
`inverse_transform` if a target transformer was used.

This design means prediction is always $O(1)$ with respect to the training set size.
The observation lifecycle (`observe`, `rewind`) updates `_X_t_observed` so that
subsequent predictions reflect newly arrived data without refitting. See
[Core Concepts](core-concepts.md#observation-horizon) for the full observation mechanism.

## Recursive Prediction

When `predict(forecasting_horizon=P)` is called with $P$ greater than the horizon used
at fit time ($P > H$), the forecaster enters recursive mode. It deep-copies itself to
avoid mutating internal state, then loops in blocks of $H$ steps: predict one block,
feed those predictions back as observations via `observe()`, and predict the next block.
This continues until $P$ steps are covered.

Recursive prediction introduces error accumulation because each block's predictions
(which may be imperfect) become the input features for the next block. It is also
incompatible with `X_forecast`, because forecast step columns are vintage-dependent and
cannot be re-derived across blocks. The forecaster raises a `ValueError` if recursive
prediction is attempted with `X_forecast`.

## Sample Weighting

[`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/)
accepts `time_weight` and `vintage_weight` parameters at `fit()` time, which control
how much influence each training sample has on the learned model.

**`time_weight`** assigns weights based on the sample's position in the series. The
most common use is exponential decay, which emphasizes recent observations over older
ones. Because each tabularized sample spans a window of $H$ future steps, the
`sample_weight_alignment` parameter controls how per-timestep weights are collapsed
to a single per-sample weight: `"first_step"` uses the weight of the first predicted
step, `"mean_step"` averages across all steps in the window, and other options
(`"max_weight_step"`, `"min_weight_step"`, `"weighted_mean_step"`) provide alternative
aggregations.

**`vintage_weight`** assigns weights based on the sample's observation time. This is
useful when certain vintages are more representative or trustworthy than others.

When both are provided, the weights are multiplied element-wise and normalized so their
sum equals the number of samples. The combined weight vector is passed to the
estimator as `sample_weight` during fitting, so the estimator must support that
parameter (most sklearn regressors do).

## NaN Handling

After tabularization, the training matrix may contain NaN values. These can originate
from several sources:

1. **Target lags**: if `y` contains NaN at position $t$, the lag feature
   `value_lag_k` will carry that NaN into every row whose window includes $t$.
2. **X_future step columns**: if `X_future` has a gap at time $t$, the step column
   `feature_step_h` will be NaN for any row whose horizon lands on $t$.
3. **X_forecast step columns**: similarly, missing vintages in `X_forecast` propagate
   as NaN into the pivoted step columns.
4. **Target columns (y_tab)**: if `y` itself has NaN at the positions that become
   the supervised target after tabularization.

The `nan_handling` parameter on
[`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/)
(and all other reduction forecasters) controls what happens next:

| Value | Behavior |
|-------|----------|
| `"pass"` (default) | NaN values are left in place. The estimator receives them as-is. Tree-based models (LightGBM, XGBoost, CatBoost, HistGradientBoosting) handle NaN natively, so this is the zero-effort path for those estimators. |
| `"drop"` | Any tabularized row where X or y contains at least one NaN is removed before fitting. A warning is emitted with the count and percentage of dropped rows. `sample_weight` is filtered in lockstep. |

For the **direct** strategy, NaN filtering happens per step: step 1's estimator may
retain rows that step 3's estimator drops (because step 3's features reference
different future positions). This maximizes the training data available to each
individual model.

For the **multi-output** and **dir-rec** strategies, a single unified mask is applied
across all steps, since all steps share one model (multi-output) or the same initial
feature matrix (dir-rec).

If `nan_handling="drop"` removes all rows, a `ValueError` is raised indicating that
no training samples remain.

## References

- Bontempi, G., Ben Taieb, S., & Le Borgne, Y.-A. (2013). Machine learning strategies
  for time series forecasting. *European Business Intelligence Summer School*, 62-77.
## Connections

The reduction approach described here produces **point forecasts**: single-valued
predictions for each future time step. For prediction intervals that quantify
uncertainty, see [Interval Forecasting](interval-forecasting.md), which covers
conformal prediction and quantile reduction built on top of the same reduction
machinery.

[Forecaster Composition](forecaster-composition.md) covers the remaining approaches:
decomposition pipelines, column forecasters, panel-local forecasters, and
forecasted-feature chains. Ensemble methods that combine multiple forecasters via
voting are also described there.

The transformers mentioned above (target transformers for stationarity, feature
transformers for signal enrichment) are discussed in depth in
[Preprocessing](preprocessing.md) and [Stationarity](stationarity.md).

Practical examples: [Reduction Forecaster](/examples/reduction_forecaster/) walks
through building a basic reduction forecaster, and
[Reduction Strategies](/examples/reduction_strategies/) compares multi-output,
direct, and dir-rec on the same dataset.
[Panel Reduction Forecasting](/examples/panel_reduction/) demonstrates panel
strategies (global, multivariate, local) for multi-entity data.

The reduction pattern extends naturally to categorical targets through
[`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.reduction.ClassProbaReductionForecaster/), which wraps sklearn classifiers instead of
regressors. See [Class-Probability Forecasting](class-probability-forecasting.md)
for the full treatment.

For practical recipes, see [How to Build a Reduction Forecaster](../how-to/build-reduction-forecasters.md) and [How to Choose a Forecasting Method](../how-to/choose-forecasting-method.md).
