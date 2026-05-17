# Glossary

Key terms used across Yohou documentation.

## Forecasting

Forecasting horizon { #forecasting-horizon }
:   The number of future timesteps to predict. Specified at `fit()` time and
    optionally overridden at `predict()` time.

Observation horizon { #observation-horizon }
:   The number of recent time steps a stateful component must retain in memory to
    produce output. A [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/)
    with `lags=[1, 7]` has an observation horizon of 7. Forecasters derive theirs
    from the maximum across their transformers. See
    [Core Concepts](core-concepts.md#observation-horizon).

Memory buffer { #memory-buffer }
:   The internal store of recent rows (up to `observation_horizon` in length) that a
    stateful component maintains. Updated by `observe()` and trimmed by `rewind()`.

Observe { #observe }
:   Updating a fitted forecaster's or transformer's internal buffers with newly
    arrived data, without refitting. Called via `observe()`, typically as part of the
    composite `observe_predict()` during rolling evaluation.

Rewind { #rewind }
:   Resetting observation buffers to the last `observation_horizon` rows. Called
    automatically after `observe()` to keep memory bounded.

Composite method { #composite-method }
:   A method that combines two operations in sequence: `observe_predict`,
    `observe_predict_interval`, `observe_predict_class_proba`, `observe_transform`,
    and `rewind_transform`. These are routed together through metadata routing.
    See [Core Concepts](core-concepts.md).

Point forecast { #point-forecast }
:   A single numeric prediction per timestep, produced by `predict()`.

Interval forecast { #interval-forecast }
:   A pair of bounds $[L, U]$ per timestep and coverage rate, produced by
    `predict_interval()`. Designed so that the true value falls within the interval
    at least $\alpha$% of the time.

Class-probability forecast { #class-probability-forecast }
:   A probability distribution over categorical classes per timestep, produced by
    `predict_class_proba()`. Each row sums to 1. See
    [Class-Probability Forecasting](class-probability-forecasting.md).

Coverage rate { #coverage-rate }
:   The target probability $\alpha$ that an interval forecast should contain the true
    value. Common values: 0.90, 0.95.

Calibration { #calibration }
:   The degree to which predicted probabilities or intervals match observed
    frequencies. A well-calibrated 90% interval contains the true value 90% of the
    time.

Recursive prediction { #recursive-prediction }
:   How a forecaster predicts beyond its training horizon by feeding its own
    predictions back as inputs and applying itself iteratively. See
    [Reduction Forecasting](reduction-forecasting.md#reduction-strategies).

Error accumulation { #error-accumulation }
:   The compounding of prediction errors when recursive prediction feeds earlier
    predictions back as inputs. The direct reduction strategy avoids this by fitting
    independent models per step. See [Reduction Forecasting](reduction-forecasting.md).

Vintage { #vintage }
:   A single forecast origin: the point in time at which the forecaster last
    observed data before predicting. Each call to `observe_predict` during rolling
    evaluation produces one vintage. See
    [Core Concepts](core-concepts.md#the-time-column-contract).

`vintage_time` { #vintage-time }
:   The column in a prediction DataFrame that records the last observed timestamp
    for each vintage. Used by scorers to support vintagewise aggregation and by
    plotting functions like [`plot_score_per_vintage`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_vintage/).

Step { #step }
:   A single position within the forecasting horizon (1-based). Step 1 is the
    one-step-ahead prediction; step $H$ is the $H$-step-ahead prediction. Used by
    scorers with `aggregation_method="stepwise"` and by [`plot_score_per_step`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_step/).

Stride { #stride }
:   The number of time steps the cross-validation window advances between folds.
    A stride equal to `test_size` produces non-overlapping test windows; a smaller
    stride produces overlapping windows for finer-grained evaluation. Also used
    in `observe_predict` to control how many time steps to advance between
    successive prediction vintages.

Rolling evaluation { #rolling-evaluation }
:   A procedure for assessing forecaster performance over time. The forecaster is
    trained once, then repeatedly observes new data and predicts the next horizon,
    producing one vintage per iteration. Also called walk-forward validation. See
    [Model Selection](model-selection.md).

## Data

Time column contract { #time-column-contract }
:   Every DataFrame in Yohou must have a `"time"` column containing datetime values.
    This column is preserved through all transformations.
    See [Core Concepts](core-concepts.md#the-time-column-contract).

Polars-native { #polars-native }
:   Yohou uses Polars DataFrames as its internal data representation throughout the
    entire pipeline (fitting, transforming, predicting, scoring). All user-facing
    inputs and outputs are Polars DataFrames.

Panel data { #panel-data }
:   Multiple related time series handled together. Groups are identified by a prefix
    in column names using the `__` separator. See [Panel Data](panel-data.md).

Group prefix (`__`) { #group-prefix }
:   The double-underscore separator between a panel group name and a column name.
    For example, `store_A__sales` identifies column `sales` in group `store_A`.

Panel strategy { #panel-strategy }
:   Controls how a forecaster handles panel data. `"global"` (default) fits one model
    with per-group transformer state. `"multivariate"` skips panel detection and treats
    `__`-prefixed columns as ordinary wide-format columns.

Exogenous features { #exogenous-features }
:   Additional input columns (beyond the target) that may improve forecasts. Passed
    as the `X` parameter in `fit()` and `predict()`. See
    [Exogenous Features](exogenous-features.md).

Known-future features { #known-future-features }
:   Exogenous features whose values are available for the prediction horizon at
    forecast time (e.g., holidays, scheduled events). Passed in the `X` argument of
    `predict()`. See [Exogenous Features](exogenous-features.md).

Step-indexed columns { #step-indexed-columns }
:   Feature columns named with a `_step_h` suffix that carry different values for
    each forecasting step. Used to provide step-specific exogenous information to
    direct or dir-rec reduction strategies. See
    [Exogenous Features](exogenous-features.md).

Univariate { #univariate }
:   A single target column (plus time). The simplest forecasting setting, where
    the model predicts one variable using only its own past values and optionally
    exogenous features.

Multivariate { #multivariate }
:   Multiple target columns forecasted simultaneously. Each column receives its
    own predictions, but all columns share the same time index and can influence
    each other through shared features or model structure.

## Modeling

Tabularization { #tabularization }
:   The process of converting a time series into a flat feature matrix that a standard
    sklearn estimator can consume. Lag features, rolling statistics, and calendar
    attributes are typical columns in the tabularized output.

Reduction strategy { #reduction-strategy }
:   Converting a time series forecasting problem into a tabular supervised learning
    problem. Three strategies control how multi-step horizons are handled:
    **multi-output** fits one model that predicts all steps simultaneously,
    **direct** fits one independent model per step, and **dir-rec** fits models
    sequentially so each step can use predictions from earlier steps. See
    [Reduction Forecasting](reduction-forecasting.md).

Step feature alignment { #step-feature-alignment }
:   Controls which step-indexed columns each direct estimator receives in a direct or
    dir-rec reduction strategy. Options: `"all"` (every step column), `"matched"`
    (only the column matching the current step), and `"cumulative"` (columns for the
    current step and all preceding steps).

Pipeline { #pipeline }
:   A sequence of transformers followed by a forecaster, executed in order. Yohou
    pipelines propagate `observe()` and `rewind()` calls through each step and
    accumulate observation horizons. See [Feature Pipelines](feature-pipelines.md).

Target transformer { #target-transformer }
:   A [`BaseTransformer`](/pages/api/generated/yohou.base.transformer.BaseTransformer/)
    applied to the target series `y` before tabularization. Used for operations like
    differencing or scaling that should be inverted after prediction.

Feature transformer { #feature-transformer }
:   A [`BaseTransformer`](/pages/api/generated/yohou.base.transformer.BaseTransformer/)
    applied to the feature matrix `X` before tabularization. Used for creating
    lag features, rolling statistics, or other derived inputs.

Stateful transformer { #stateful-transformer }
:   A [`BaseTransformer`](/pages/api/generated/yohou.base.transformer.BaseTransformer/)
    that maintains an internal observation window of recent values and updates
    it during `observe()`. Stateful transformers have a non-zero `observation_horizon`
    because they need to remember past values (e.g., lag features, rolling statistics)
    to transform new data. See [Preprocessing](preprocessing.md) for details on
    how observation horizons propagate through pipelines.

Stateless transformer { #stateless-transformer }
:   A [`BaseTransformer`](/pages/api/generated/yohou.base.transformer.BaseTransformer/)
    whose output depends only on its fitted parameters and the current input, with no
    dependence on prior observations. Has an `observation_horizon` of 0. Examples
    include scaling, log transforms, and calendar feature extraction.

Ensemble { #ensemble }
:   Combining predictions from multiple forecasters to reduce variance. Implemented
    via the [`VotingPointForecaster`](/pages/api/generated/yohou.ensemble.voting_point.VotingPointForecaster/), [`VotingIntervalForecaster`](/pages/api/generated/yohou.ensemble.voting_interval.VotingIntervalForecaster/), and
    [`VotingClassProbaForecaster`](/pages/api/generated/yohou.ensemble.voting_class_proba.VotingClassProbaForecaster/) classes.
    See [Ensemble Forecasting](ensemble-forecasting.md).

Forecaster composition { #forecaster-composition }
:   Building complex forecasting workflows by combining simpler components. Includes
    pipelines, column forecasters (different forecasters per column),
    decomposition pipelines (trend, seasonality, residual), and forecasted-feature
    forecasters (two-stage prediction of exogenous inputs). See
    [Forecaster Composition](forecaster-composition.md).

Decomposition { #decomposition }
:   Separating a time series into additive components (trend, seasonality, residual) for
    separate modeling, implemented in [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/).
    See [Stationarity](stationarity.md).

Variance stabilization { #variance-stabilization }
:   Transforming a time series so that its error variance is approximately constant
    over time. Common transforms include logarithms, Box-Cox, and inverse hyperbolic
    sine. Useful when error magnitude scales with the level of the series. See
    [Stationarity](stationarity.md).

Stationarity { #stationarity }
:   A time series is stationary when its statistical properties (mean, variance) do
    not change over time. Many forecasting methods assume or benefit from stationarity,
    and Yohou provides transforms like [`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/) and [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/)
    to achieve it before modeling. See [Stationarity](stationarity.md).

Conformal prediction { #conformal-prediction }
:   A distribution-free method for constructing prediction intervals with finite-sample
    coverage guarantees, implemented in [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/).
    See [Interval Forecasting](interval-forecasting.md).

Conformity score { #conformity-score }
:   The residual measurement computed on a held-out calibration set during conformal
    prediction. The quantile of conformity scores determines the width of prediction
    intervals. Yohou provides signed, absolute, gamma, and quantile variants.

Calibration set { #calibration-set }
:   A portion of the training data held out from model fitting and used exclusively to
    compute conformity scores for interval construction. The quality of conformal
    intervals depends on the calibration set being representative of future data.

Similarity measure { #similarity-measure }
:   A function that weights conformity scores based on the similarity between the
    current prediction context and past calibration contexts. Produces locally adaptive
    intervals that are wider in unfamiliar regimes and narrower in well-represented
    ones. See [Interval Forecasting](interval-forecasting.md).

Metadata routing { #metadata-routing }
:   The mechanism by which sample-level metadata (e.g., `time_weight`, `step_weight`,
    `vintage_weight`) flows from a composite estimator (pipeline, search, ensemble) down
    to its child components. Yohou enables scikit-learn's metadata routing globally and
    registers seven additional routable methods. See
    [Metadata Routing](metadata-routing.md).

## Evaluation

Scorer { #scorer }
:   An object that computes a numeric quality measure from predicted and actual values.
    Yohou provides point scorers (e.g., MAE, RMSE, MASE), interval scorers
    (e.g., `IntervalScore`, `EmpiricalCoverage`), and class-probability scorers
    (e.g., `LogLoss`, `BrierScore`). All scorers share a common `score()` interface.
    See [Forecast Accuracy](forecast-accuracy.md).

Proper scoring rule { #proper-scoring-rule }
:   A metric that is uniquely minimized (or maximized) when the predicted distribution
    matches the true distribution. `LogLoss` and `BrierScore` are proper scoring rules;
    `Accuracy` is not.

Aggregation method { #aggregation-method }
:   Controls which dimensions a scorer collapses when computing its result. Options:
    `"timewise"` (across time), `"componentwise"` (across multivariate columns),
    `"groupwise"` (across panel groups), `"stepwise"` (across forecasting steps),
    `"vintagewise"` (across vintages), `"coveragewise"` (across coverage rates,
    intervals only), and `"all"` (collapse every dimension to a single scalar). See
    [Forecast Accuracy](forecast-accuracy.md).

Forecast error { #forecast-error }
:   The difference between a predicted value and the corresponding actual value,
    computed on out-of-sample data. Distinct from a residual, which is computed
    on training data. See [Forecast Accuracy](forecast-accuracy.md).

Scale-dependent metric { #scale-dependent-metric }
:   A metric whose value depends on the scale of the data (e.g., MAE, RMSE). Cannot
    be compared directly across series with different units or magnitudes.

Scale-independent metric { #scale-independent-metric }
:   A metric that normalizes errors so they can be compared across series of different
    scales (e.g., MAPE, MASE).

Cross-validation { #cross-validation }
:   Evaluating model performance by repeatedly splitting data into training and test
    sets. Time series cross-validation uses temporal splits to prevent data leakage.

Temporal split { #temporal-split }
:   A train/test split that respects time ordering: training data always precedes test
    data chronologically. This prevents future information from leaking into training.

Expanding window splitter { #expanding-window-splitter }
:   A cross-validation splitter where each fold grows the training window while keeping
    the test window a fixed size. The training set includes all data from the beginning
    of the series up to each cutoff point. See [Model Selection](model-selection.md).

Sliding window splitter { #sliding-window-splitter }
:   A cross-validation splitter that maintains a fixed-size training window that slides
    forward in time. Older data is dropped from training as newer data is added,
    making it better suited for data with concept drift or regime changes. See
    [Model Selection](model-selection.md).

Concept drift { #concept-drift }
:   A shift in the statistical properties of the data over time, causing a model
    trained on older data to become less accurate. Sliding window splitters
    mitigate this by limiting training to recent observations.

Leakage { #leakage }
:   Using information from the future (test period) during training, leading to
    artificially optimistic performance estimates. In time series, leakage commonly
    occurs from improper cross-validation splits or applying global normalization
    before splitting. See [Model Selection](model-selection.md) for how Yohou's
    splitters prevent temporal leakage.

Time weighting { #time-weighting }
:   Applying non-uniform weights to observations or errors so that specific time
    periods carry more or less influence. Yohou supports three weight types:
    `time_weight` (per-timestep), `step_weight` (per-forecasting-step), and
    `vintage_weight` (per-forecast-origin). See
    [Weighting](weighting.md) for formats and normalization.

`time_weight` { #time-weight }
:   A per-timestep weight. In training, it controls the emphasis on individual
    observations. In scoring, it controls how much each time point contributes to
    the aggregated metric. Passed as a column or Series alongside the data.

Step weight { #step-weight }
:   A weight applied per forecasting step (1-step-ahead, 2-step-ahead, etc.)
    during scoring. Controls how much each lead time contributes to the
    aggregated score. Only available in scoring, not training.

Vintage weight { #vintage-weight }
:   A weight applied per vintage (forecast origin date). In training, controls
    per-observation emphasis; in scoring, controls per-vintage score
    aggregation. Available in both `fit()` and `score()`.
