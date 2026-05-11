# Model Selection

Selecting the right forecasting model and tuning its hyperparameters requires evaluating candidate configurations on held-out data. In tabular machine learning this is straightforward: shuffle the rows, partition into folds, and measure performance. Time series data, however, carries temporal dependencies that make shuffling invalid. Yohou's model selection module provides splitters and search utilities that respect chronological order while fitting naturally into the sklearn-style `fit` / `predict` workflow.

## Standard Cross-Validation Fails for Time Series

Standard k-fold cross-validation randomly shuffles observations into folds. Each fold trains on a subset and tests on the remainder, with no concern for ordering. When applied to a time series this creates a fundamental problem: the model can train on data that comes *after* some of its test observations. A regression model might learn the Tuesday value while being evaluated on Monday, producing optimistically biased scores that do not reflect real forecasting performance.

The bias is not subtle. Any feature derived from recent history (lagged values, rolling averages, seasonal differences) becomes contaminated because the "recent history" in the training set includes future observations relative to the test set. Leak-free evaluation requires that every training sample precedes every test sample within each fold. This is the principle that yohou's splitters enforce: folds respect the arrow of time, and evaluation mimics the sequential nature of real forecasting.

The standard term for this procedure is **rolling origin evaluation** (also called
**time series cross-validation** or **walk-forward validation**). The "origin" is the
last training observation in each fold; it rolls forward through the series, producing
a sequence of train-test pairs that simulate how the forecaster would be deployed in
practice. Yohou's expanding and sliding window splitters are both variants of rolling
origin evaluation.

## Expanding Window Splitting

[`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.ExpandingWindowSplitter/) implements the most common temporal cross-validation strategy. The first fold trains on the earliest portion of the series and tests on the window that follows. Each subsequent fold keeps all the previous training data and appends the next slice, so the training set grows monotonically:

```text
Fold 1:  [=== train ===][test]..............
Fold 2:  [====== train ======][test]........
Fold 3:  [========= train =========][test]..
```

The expanding approach reflects a natural assumption: more historical data generally helps a model. It also mirrors production deployments where you periodically retrain on all available history before generating the next round of forecasts.

Key parameters control the geometry of the folds. `n_splits` sets the number of folds, `test_size` fixes the length of each test window, and `max_train_size` optionally caps the training set if memory or computation becomes a concern. When `max_train_size` is set, the splitter still marches forward in time but trims the oldest training observations to stay within the limit, creating a hybrid between expanding and sliding behavior.

## Sliding Window Splitting

[`SlidingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.SlidingWindowSplitter/) takes a different stance: the training window has a fixed size and slides forward with each fold. As new data enters the training set, equally old data drops off:

```text
Fold 1:  [=== train ===][test]..............
Fold 2:  ...[=== train ===][test]...........
Fold 3:  ......[=== train ===][test]........
```

This strategy suits series where older observations become less relevant over time, such as situations involving concept drift, regime changes, or evolving consumer behavior. Because each fold trains on the same amount of data, it also keeps computation per fold constant, which can matter for large datasets.

The `stride` parameter controls how far the window advances between folds. By default it equals `test_size`, producing non-overlapping test sets. Setting stride smaller than `test_size` creates overlapping test windows for finer-grained evaluation at the cost of correlated fold scores. When `train_size` is omitted, the splitter computes it automatically from `n_splits` and the data length so that the requested number of folds fits exactly.

## Checking Splitter Alignment

When `test_size` is not an exact multiple of `stride`, some forecast steps may be
evaluated on more vintages than others, producing unbalanced scores.
[`check_cv_alignment`](/pages/api/generated/yohou.model_selection.split.check_cv_alignment/)
inspects this relationship before you run a full search:

```python
from yohou.model_selection import SlidingWindowSplitter, check_cv_alignment

cv = SlidingWindowSplitter(n_splits=3, test_size=10, stride=4)
info = check_cv_alignment(cv, forecasting_horizon=4)
print(info["is_balanced"])  # False
```

The returned dictionary includes the number of vintages per fold, how many vintages
cover each forecast step, and whether the distribution is balanced. Call this early
to avoid surprises in evaluation results.

## Gap-Based Splitting

Both splitters accept a `gap` parameter that inserts a buffer of time steps between the end of the training set and the start of the test set:

```text
[=== train ===]---gap---[test]
```

In many real forecasting scenarios, predictions are needed several steps ahead. A retailer forecasting weekly demand may need the forecast by Wednesday for the following week, so the most recent five days of data are unavailable at prediction time. Setting `gap=5` simulates this lead-time constraint during evaluation.

The gap also guards against a subtler form of leakage. Some transformers (rolling averages, exponential smoothers) blend information across neighboring time steps. Without a gap, the last few training observations may carry information that bleeds into the test period through these smoothed features. A small gap provides an extra margin of safety.

## Time-Weighted Scoring

Not all test errors deserve equal attention. A forecast that performed well last month but poorly six months ago may still be the right choice for production. Yohou scorers accept a `time_weight` parameter through sklearn's metadata routing that assigns different importance to each test-set time step.

The [`exponential_decay_weight`](/pages/api/generated/yohou.utils.weighting.exponential_decay_weight/) utility generates weights that decrease geometrically into the past, giving recent performance the greatest influence on the final score. [`seasonal_emphasis_weight`](/pages/api/generated/yohou.utils.weighting.seasonal_emphasis_weight/) takes a different approach: it upweights time steps that fall on specific seasonal boundaries (year-end, quarter-end, peak season) where accurate forecasts matter most. The [`linear_decay_weight`](/pages/api/generated/yohou.utils.weighting.linear_decay_weight/) function offers a simpler ramp that transitions smoothly from low weight on the oldest test step to full weight on the most recent.

These weighting functions return polars DataFrames with a `"time"` column and a weight column, matching the structure that scorers expect. During cross-validation, the weights are routed to the scorer automatically through metadata routing, requiring no manual plumbing.

## Hyperparameter Search

[`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) and [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/) combine time series splitters with parameter search to find the best forecaster configuration. They follow the same interface as their sklearn counterparts but operate on yohou forecasters and scorers:

```python
from yohou.model_selection import GridSearchCV, ExpandingWindowSplitter
from yohou.metrics import MeanAbsoluteError

search = GridSearchCV(
    forecaster=my_forecaster,
    param_grid={"estimator__alpha": [0.1, 1.0, 10.0]},
    scoring=MeanAbsoluteError(),
    cv=ExpandingWindowSplitter(n_splits=3, test_size=10),
)
search.fit(y, X_actual=X, forecasting_horizon=7)
```

For each candidate parameter combination, the search clones the forecaster, fits it on the training fold, and evaluates predictions on the test fold using the provided scorer. Results accumulate into a `cv_results_` dictionary containing per-fold scores, mean scores, standard deviations, rankings, and timing information.

When `refit=True` (the default), the search refits the best configuration on the entire dataset after evaluation. The resulting `best_forecaster_` supports all standard yohou methods (`predict`, `observe`, `rewind`, `observe_predict`) so the search object can be used directly in place of a bare forecaster.

Multi-metric evaluation is supported by passing a dictionary of scorers. In this case, `refit` must name the scorer to optimize or be set to `False`.

`RandomizedSearchCV` works identically but samples a fixed number of parameter combinations from specified distributions rather than exhaustively evaluating every point on the grid. This is more practical when the parameter space is large or continuous.

Both search classes parallelize fold evaluation via `n_jobs`, and both integrate with sklearn's metadata routing so that `time_weight` and other metadata flow through to scorers without extra configuration.


## Choosing a Forecasting Method

With many possible models, the practical question is where to start. The most
productive approach is incremental: begin with the simplest model and add complexity
only when cross-validation shows it helps.

A [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/) forecaster
is a natural starting point because it establishes a baseline that any useful model
must beat. If a
[`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/)
with a linear regressor and a few lags improves on that baseline, there is learnable
structure beyond seasonal repetition, and further complexity (richer transformers,
nonlinear regressors) is worth exploring.

Hyperparameter search should come after model structure is settled. Tuning
hyperparameters on an underspecified model wastes computation, while a well-structured
model often performs acceptably even with default parameters. Use `GridSearchCV` for
small discrete grids and `RandomizedSearchCV` when the parameter space is large or
continuous.

Finally, a single metric can be misleading. Evaluating candidates on both
scale-dependent (MAE, RMSE) and scaled (MASE) metrics confirms the ranking is robust.
See [Forecast Accuracy](forecast-accuracy.md) for metric selection guidance.


## References

- Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*,
  3rd edition. [Chapter 5.10](https://otexts.com/fpp3/tscv.html) (time series
  cross-validation), [Chapter 8.1](https://otexts.com/fpp3/ses.html) (time weighting
  and exponential smoothing).
- Tashman, L.J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and
  review. *International Journal of Forecasting*, 16(4), 437-450.
  [DOI:10.1016/S0169-2070(00)00065-0](https://doi.org/10.1016/S0169-2070(00)00065-0)


## Connections

The splitters and search utilities tie together several other parts of yohou.
Scorers from [Forecast Accuracy](forecast-accuracy.md) define the objective.
Weighting functions shape how errors are aggregated across time. Forecasters
from the [Point Forecasting](forecasting.md) and
[Interval Forecasting](interval-forecasting.md) modules provide the candidates.

[`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/)
and
[`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/)
work with all forecaster types: point, interval, and class-probability. For
classification forecasters, pass a class-proba scorer such as
[`LogLoss()`](/pages/api/generated/yohou.metrics.class_proba.LogLoss/) as the
`scoring` parameter.

Practical examples: [CV Splitters](/examples/cv_splitters/),
[Hyperparameter Search](/examples/hyperparameter_search/), and
[Time-Weighted Scoring](/examples/time_weighted_scoring/).
