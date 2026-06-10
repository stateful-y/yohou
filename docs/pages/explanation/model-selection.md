# Model Selection

Selecting the right forecasting model and tuning its hyperparameters requires evaluating candidate configurations on held-out data. In tabular machine learning this is straightforward: shuffle the rows, partition into folds, and measure performance. Time series data, however, carries temporal dependencies that make shuffling invalid. Yohou's model selection module provides splitters, search utilities, and convenience functions that respect chronological order while fitting naturally into the sklearn-style `fit` / `predict` workflow.

## Standard Cross-Validation Fails for Time Series

Standard k-fold cross-validation randomly shuffles observations into folds. Each fold trains on a subset and tests on the remainder, with no concern for ordering. When applied to a time series this creates a fundamental problem: the model can train on data that comes *after* some of its test observations. A regression model might learn the Tuesday value while being evaluated on Monday, producing optimistically biased scores that do not reflect real forecasting performance.

The bias is not subtle. Any feature derived from recent history (lagged values, rolling averages, seasonal differences) becomes contaminated because the "recent history" in the training set includes future observations relative to the test set. Leak-free evaluation requires that every training sample precedes every test sample within each fold. This is the principle that yohou's splitters enforce: folds respect the arrow of time, and evaluation mimics the sequential nature of real forecasting.

The standard term for this procedure is **rolling origin evaluation** (also called **time series cross-validation** or **walk-forward validation**). The "origin" is the last training observation in each fold; it rolls forward through the series, producing a sequence of train/test pairs that simulate how the forecaster would be deployed in practice. Yohou's expanding and sliding window splitters are both variants of rolling origin evaluation.

## Expanding Window Splitting

[`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.ExpandingWindowSplitter/) implements the most common temporal cross-validation strategy. The first fold trains on the earliest portion of the series and tests on the window that follows. Each subsequent fold keeps all the previous training data and appends the next slice, so the training set grows monotonically:

```text
Fold 1:  [=== train ===][test]..............
Fold 2:  [====== train ======][test]........
Fold 3:  [========= train =========][test]..
```

The expanding approach reflects a natural assumption: more historical data generally helps a model. It also mirrors production deployments where you periodically retrain on all available history before generating the next round of forecasts.

Three parameters control the geometry of the folds:

- `n_splits` sets the number of folds (minimum 2).
- `test_size` fixes the length of each test window. When omitted, it defaults to `n_samples // (n_splits + 1)`.
- `max_train_size` optionally caps the training set if memory or computation becomes a concern. The splitter still marches forward in time but trims the oldest training observations to stay within the limit, creating a hybrid between expanding and sliding behavior.

## Sliding Window Splitting

[`SlidingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.SlidingWindowSplitter/) takes a different stance: the training window has a fixed size and slides forward with each fold. As new data enters the training set, equally old data drops off:

```text
Fold 1:  [=== train ===][test]..............
Fold 2:  ...[=== train ===][test]...........
Fold 3:  ......[=== train ===][test]........
```

This strategy suits series where older observations become less relevant over time, such as situations involving concept drift, regime changes, or evolving consumer behavior. Because each fold trains on the same amount of data, it also keeps computation per fold constant, which can matter for large datasets.

The `stride` parameter controls how far the window advances between folds. By default it equals `test_size`, producing non-overlapping test sets. Setting stride smaller than `test_size` creates overlapping test windows for finer-grained evaluation at the cost of correlated fold scores. Setting stride larger than `test_size` leaves gaps between test windows, skipping some observations entirely. When `train_size` is omitted, the splitter computes it automatically from `n_splits` and the data length so that the requested number of folds fits exactly.

## Train/Test Split

For a simple one-time partition without cross-validation, [`train_test_split`](/pages/api/generated/yohou.model_selection.split.train_test_split/) splits one or more polars DataFrames chronologically. It accepts either an integer (`test_size=30` for 30 rows) or a float (`test_size=0.2` for 20% of the data) and returns alternating train/test pairs:

```python
from yohou.model_selection import train_test_split

y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.2)
```

This is useful for quick sanity checks and holdout evaluation before committing to a full cross-validation run.

## Panel Data Support

All splitters operate on row indices, so they work identically for univariate, multivariate, and panel datasets. When the target DataFrame contains multiple groups (identified by the `__` column separator convention), the splitter partitions by row position across all groups simultaneously. This ensures that every group shares the same temporal split boundaries, which is essential for panel forecasters that learn across groups.

The search classes inherit panel support from the underlying forecaster. After fitting, `groups_` exposes the panel group names discovered during search.

## Checking Splitter Alignment

When `test_size` is not an exact multiple of `stride`, some forecast steps may be evaluated on more vintages than others, producing unbalanced scores. [`check_cv_alignment`](/pages/api/generated/yohou.model_selection.split.check_cv_alignment/) inspects this relationship before you run a full search:

```python
from yohou.model_selection import SlidingWindowSplitter, check_cv_alignment

cv = SlidingWindowSplitter(n_splits=3, test_size=10, stride=4)
info = check_cv_alignment(cv, forecasting_horizon=4)
print(info["is_balanced"])  # False
```

The returned dictionary includes:

- `n_vintages`: the number of predict calls per fold.
- `steps_per_vintage`: step counts per vintage (all equal to `forecasting_horizon` except possibly the last).
- `step_counts`: maps each forecast step (1-based) to how many vintages include it.
- `is_balanced`: `True` when every step appears in the same number of vintages.

Call this early to avoid surprises in evaluation results.

The helper [`check_cv`](/pages/api/generated/yohou.model_selection.split.check_cv/) normalizes various CV specifications into a splitter instance. Passing `None` produces a default `ExpandingWindowSplitter` with 5 folds; passing an integer produces an expanding window with that many folds. This is the same normalization that the search classes apply internally.

## Time-Weighted Scoring

Not all test errors deserve equal attention. A forecast that performed well last month but poorly six months ago may still be the right choice for production. Yohou scorers carry their weighting as a constructor parameter: pass a weighter to `time_weighter` to assign different importance to each test-set time step.

Three built-in weighters map a key `pl.Series` to a weight `pl.Series`:

- [`ExponentialDecayWeighter`](/pages/api/generated/yohou.utils.weighting.ExponentialDecayWeighter/) generates weights that decrease geometrically into the past, controlled by a `half_life` parameter. Recent performance receives the greatest influence on the final score.
- [`SeasonalEmphasisWeighter`](/pages/api/generated/yohou.utils.weighting.SeasonalEmphasisWeighter/) upweights time steps that fall on specific seasonal boundaries (year-end, quarter-end, peak season) where accurate forecasts matter most.
- [`LinearDecayWeighter`](/pages/api/generated/yohou.utils.weighting.LinearDecayWeighter/) offers a simpler ramp that transitions smoothly from low weight on the oldest test step to full weight on the most recent.

Because the weighting lives on the scorer instance, a weighted scorer is a self-contained cross-validation objective; there is no per-call weight argument to route:

```python
from yohou.metrics import MeanAbsoluteError
from yohou.utils.weighting import ExponentialDecayWeighter

scoring = MeanAbsoluteError(time_weighter=ExponentialDecayWeighter(half_life=90))
```

### Weighters Are Tunable Hyperparameters

Constructor-residence has a second payoff: a weighter's settings become ordinary searchable hyperparameters, addressed with the `__` syntax. You can tune the recency half-life (or the decay basis) alongside the model's own parameters, and the search clones and varies them directly (no metadata routing involved):

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster
from yohou.model_selection import GridSearchCV, ExpandingWindowSplitter
from yohou.utils.weighting import ExponentialDecayWeighter

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    time_weighter=ExponentialDecayWeighter(half_life=365),
)

search = GridSearchCV(
    forecaster,
    param_grid={
        "estimator__alpha": [0.1, 1.0, 10.0],
        "time_weighter__half_life": [90, 180, 365],
        "time_weighter__scale": ["elapsed", "position"],
    },
    cv=ExpandingWindowSplitter(n_splits=5, test_size=14),
)
search.fit(y, forecasting_horizon=14)
```

See [How to Use Time Weighting](../how-to/time-weighting.md) for the full set of weighter classes and tuning recipes.

## Cross-Validation

The simplest way to compute a cross-validated score is [`cross_val_score`](/pages/api/generated/yohou.model_selection.validation.cross_val_score/). It takes a forecaster, target data, a scorer, and an optional splitter, then returns a `pl.DataFrame` with `split` (0-indexed fold identifier) and `score` columns:

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster
from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import cross_val_score, ExpandingWindowSplitter

forecaster = PointReductionForecaster(estimator=Ridge())
scores = cross_val_score(
    forecaster,
    y,
    scoring=MeanAbsoluteError(),
    cv=ExpandingWindowSplitter(n_splits=5, test_size=14),
    forecasting_horizon=14,
)
print(scores)  # DataFrame with split and score columns
print(f"Mean: {scores['score'].mean():.2f} (+/- {scores['score'].std():.2f})")
```

### The `cross_validate` Function

For richer output, [`cross_validate`](/pages/api/generated/yohou.model_selection.validation.cross_validate/) returns a `pl.DataFrame` with a `split` column, timing columns (`fit_time`, `score_time`), and score columns. In single-scorer mode the score column is `test_score`. When `return_train_score=True` is set, a `train_score` column is added. Two additional optional flags change the return type to a dictionary: `return_forecaster` stores the fitted forecaster from each fold, and `return_indices` stores the train/test index arrays. When either flag is set, the result is a dictionary with a `"results"` key containing the DataFrame, plus `"forecaster"` and/or `"indices"` keys.

When a dictionary of scorers is passed, the score columns follow the pattern `test_{name}` and (if requested) `train_{name}` for each scorer name:

```python
from yohou.metrics import MeanAbsoluteError, RootMeanSquaredError
from yohou.model_selection import cross_validate, ExpandingWindowSplitter

results = cross_validate(
    forecaster,
    y,
    scoring={
        "mae": MeanAbsoluteError(),
        "rmse": RootMeanSquaredError(),
    },
    cv=ExpandingWindowSplitter(n_splits=5, test_size=14),
    forecasting_horizon=14,
)
print(results["test_mae"])   # per-fold MAE column
print(results["test_rmse"])  # per-fold RMSE column
print(results["fit_time"])   # time spent fitting each fold
```

`return_train_score` defaults to `False` to save computation, since training scores require an additional scoring pass over the (often much larger) training set.

### Obtaining Predictions by Cross-Validation

[`cross_val_predict`](/pages/api/generated/yohou.model_selection.validation.cross_val_predict/) generates out-of-fold predictions rather than scores. For each fold the forecaster is fitted on the training data and predictions are produced on the test data. The function concatenates all fold predictions into a single `pl.DataFrame` with a `split` column identifying the originating fold.

```python
from yohou.model_selection import cross_val_predict, ExpandingWindowSplitter

predictions = cross_val_predict(
    forecaster,
    y,
    cv=ExpandingWindowSplitter(n_splits=5, test_size=14),
    forecasting_horizon=14,
)
print(predictions.head())  # columns include "time", predictions, and "split"
```

These predictions are useful for visualizing how the forecaster performs across different folds and for model blending (stacking), where out-of-fold predictions serve as features for a second-level model. Note that scoring the concatenated predictions is not equivalent to the per-fold averaged scores from `cross_val_score`, because each prediction comes from a model trained on a different subset of the data.

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

For each candidate parameter combination, the search clones the forecaster, fits it on the training fold, and evaluates predictions on the test fold using the provided scorer. Results accumulate into `cv_results_`, a dictionary of NumPy arrays containing per-fold scores (`split0_test_score`, `split1_test_score`, ...), mean and standard deviation across folds, rankings, parameter values, and timing information. Scores follow a "higher is better" sign convention: metrics where lower values are better (like MAE) are negated in `cv_results_` so that the best candidate always has the highest mean score.

`best_params_` holds the winning parameter combination, `best_score_` the corresponding mean score, and `best_index_` points into the `cv_results_` arrays. Setting `return_train_score=True` adds training scores to the results, which is useful for diagnosing overfitting but requires the forecaster to support `rewind()`.

### Refitting and Using the Best Model

When `refit=True` (the default), the search refits the best configuration on the entire dataset after evaluation. The resulting `best_forecaster_` supports all standard yohou methods (`predict`, `predict_interval`, `predict_class_proba`, `observe`, `rewind`, `observe_predict`, and their interval/class-probability variants) so the search object can be used directly in place of a bare forecaster.

The `refit` parameter also accepts a string (to name the scorer for multi-metric optimization) or a callable that receives `cv_results_` and returns the `best_index_`, enabling custom selection strategies like choosing the simplest model within one standard deviation of the best score.

### Multi-Metric Evaluation

Passing a dictionary of scorers enables simultaneous evaluation on multiple metrics. In this case, `refit` must name the scorer to optimize or be set to `False`:

```python
search = GridSearchCV(
    forecaster=my_forecaster,
    param_grid=param_grid,
    scoring={"mae": MeanAbsoluteError(), "rmse": RootMeanSquaredError()},
    cv=cv,
    refit="mae",
)
```

### Randomized Search

[`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/) samples a fixed number of parameter combinations (`n_iter`, default 10) from specified distributions rather than exhaustively evaluating every point on the grid. This is more practical when the parameter space is large or continuous. A `random_state` parameter ensures reproducibility.

### Parallelization and Error Handling

Both search classes parallelize fold evaluation via `n_jobs` and control dispatch with `pre_dispatch` to limit memory usage. When a candidate fails to fit, `error_score` determines the behavior: set it to `np.nan` (the default) to record the failure and continue, or to `"raise"` to abort immediately. Failed fits produce a `FitFailedWarning` with the traceback.

Both classes integrate with sklearn's metadata routing so that `time_weight` and other metadata flow through to scorers without extra configuration.

## Choosing a Forecasting Method

With many possible models, the practical question is where to start. The incremental approach works well because each step isolates one source of improvement. A naive baseline reveals whether there is learnable structure at all. A linear model on a few lags shows whether regression adds value over repetition. Richer transformers and nonlinear regressors add capacity, but only improve scores when the data has patterns that simpler models cannot capture.

Hyperparameter search is most valuable after model structure is settled. Tuning hyperparameters on an underspecified model wastes computation, while a well-structured model often performs acceptably even with default parameters.

Evaluating candidates on multiple metrics (scale-dependent and scaled) confirms that rankings are robust rather than artifacts of a single summary statistic. See [Forecast Accuracy](forecast-accuracy.md) for metric selection guidance and [Choose a Forecasting Method](../how-to/choose-forecasting-method.md) for practical step-by-step guidance.

## References

- Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*,
  3rd edition. [Chapter 5.10](https://otexts.com/fpp3/tscv.html) (time series
  cross-validation), [Chapter 8.1](https://otexts.com/fpp3/ses.html) (time weighting
  and exponential smoothing).
- Tashman, L.J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and
  review. *International Journal of Forecasting*, 16(4), 437-450.
  [DOI:10.1016/S0169-2070(00)00065-0](https://doi.org/10.1016/S0169-2070(00)00065-0)

## Connections

The splitters and search utilities tie together several other parts of yohou. Scorers from [Forecast Accuracy](forecast-accuracy.md) define the objective. Weighting functions shape how errors are aggregated across time. Forecasters from the [Reduction Forecasting](reduction-forecasting.md) and [Interval Forecasting](interval-forecasting.md) modules provide the candidates.

[`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) and [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/) work with all forecaster types: point, interval, and class-probability. For classification forecasters, pass a class-proba scorer such as [`LogLoss()`](/pages/api/generated/yohou.metrics.class_proba.LogLoss/) as the `scoring` parameter.

For practical recipes, see [How to Tune Hyperparameters](../how-to/tune-hyperparameters.md).

Interactive examples: [CV Splitters](/examples/cv_splitters/), [Cross-Validation](/examples/cross_validation/), [Hyperparameter Search](/examples/hyperparameter_search/), and [Time-Weighted Scoring](/examples/time_weighted_scoring/).
