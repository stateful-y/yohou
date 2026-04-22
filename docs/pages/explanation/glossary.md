# Glossary

Key terms used across Yohou documentation.

## Forecasting

Forecasting horizon
:   The number of future timesteps to predict. Specified at `fit()` time and
    optionally overridden at `predict()` time.

Observation horizon
:   The number of recent time steps a stateful component must retain in memory to
    produce output. A [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/)
    with `lags=[1, 7]` has an observation horizon of 7. Forecasters derive theirs
    from the maximum across their transformers. See
    [Core Concepts](core-concepts.md#observation-horizon).

Observe
:   Updating a fitted forecaster's or transformer's internal buffers with newly
    arrived data, without refitting. Called via `observe()`, typically as part of the
    composite `observe_predict()` during rolling evaluation.

Rewind
:   Resetting observation buffers to the last `observation_horizon` rows. Called
    automatically after `observe()` to keep memory bounded.

Point forecast
:   A single numeric prediction per timestep, produced by `predict()`.

Interval forecast
:   A pair of bounds $[L, U]$ per timestep and coverage rate, produced by
    `predict_interval()`. Designed so that the true value falls within the interval
    at least $\alpha$% of the time.

Class-probability forecast
:   A probability distribution over categorical classes per timestep, produced by
    `predict_class_proba()`. Each row sums to 1.

Coverage rate
:   The target probability $\alpha$ that an interval forecast should contain the true
    value. Common values: 0.90, 0.95.

Calibration
:   The degree to which predicted probabilities or intervals match observed
    frequencies. A well-calibrated 90% interval contains the true value 90% of the
    time.

Recursive prediction
:   How a forecaster predicts beyond its training horizon by feeding its own
    predictions back as inputs and applying itself iteratively. See
    [Advanced Topics](advanced.md#recursive-prediction).

Vintage
:   A single forecast origin: the point in time at which the forecaster last
    observed data before predicting. Each call to `observe_predict` during rolling
    evaluation produces one vintage. See
    [Core Concepts](core-concepts.md#the-time-column-contract).

`vintage_time`
:   The column in a prediction DataFrame that records the last observed timestamp
    for each vintage. Used by scorers to support vintagewise aggregation and by
    plotting functions like `plot_score_per_vintage`.

Step
:   A single position within the forecasting horizon (1-based). Step 1 is the
    one-step-ahead prediction; step $H$ is the $H$-step-ahead prediction. Used by
    scorers with `aggregation_method="stepwise"` and by `plot_score_per_step`.

Stride
:   The number of time steps the cross-validation window advances between folds.
    A stride equal to `test_size` produces non-overlapping test windows; a smaller
    stride produces overlapping windows for finer-grained evaluation.

## Data

Time column contract
:   Every DataFrame in Yohou must have a `"time"` column containing datetime values.
    This column is preserved through all transformations.

Panel data
:   Multiple related time series handled together. Groups are identified by a prefix
    in column names using the `__` separator.

Group prefix (`__`)
:   The double-underscore separator between a panel group name and a column name.
    For example, `store_A__sales` identifies column `sales` in group `store_A`.

Panel strategy
:   Controls how a forecaster handles panel data. `"global"` (default) fits one model
    with per-group transformer state. `"multivariate"` skips panel detection and treats
    `__`-prefixed columns as ordinary wide-format columns.

Exogenous features
:   Additional input columns (beyond the target) that may improve forecasts. Passed
    as the `X` parameter in `fit()` and `predict()`.

Univariate
:   A single target column (plus time). The simplest forecasting setting.

Multivariate
:   Multiple target columns forecasted simultaneously.

## Modeling

Tabularization
:   The process of converting a time series into a flat feature matrix that a standard
    sklearn estimator can consume. Lag features, rolling statistics, and calendar
    attributes are typical columns in the tabularized output.

Reduction strategy
:   Converting a time series forecasting problem into a tabular supervised learning
    problem. Three strategies control how multi-step horizons are handled:
    **multi-output** fits one model that predicts all steps simultaneously,
    **direct** fits one independent model per step, and **dir-rec** fits models
    sequentially so each step can use predictions from earlier steps.

Target transformer
:   A [`BaseTransformer`](/pages/api/generated/yohou.base.base.BaseTransformer/)
    applied to the target series `y` before tabularization. Used for operations like
    differencing or scaling that should be inverted after prediction.

Feature transformer
:   A [`BaseTransformer`](/pages/api/generated/yohou.base.base.BaseTransformer/)
    applied to the feature matrix `X` before tabularization. Used for creating
    lag features, rolling statistics, or other derived inputs.

Ensemble
:   Combining predictions from multiple forecasters to reduce variance. Implemented
    via the [`VotingPointForecaster`](/pages/api/generated/yohou.ensemble.voting_point.VotingPointForecaster/), [`VotingIntervalForecaster`](/pages/api/generated/yohou.ensemble.voting_interval.VotingIntervalForecaster/), and
    [`VotingClassProbaForecaster`](/pages/api/generated/yohou.ensemble.voting_class_proba.VotingClassProbaForecaster/) classes.

Conformal prediction
:   A distribution-free method for constructing prediction intervals with finite-sample
    coverage guarantees, implemented in [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/).

Decomposition
:   Separating a time series into components (trend, seasonality, residual) for
    separate modeling, implemented in [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/).

Stationarity
:   A time series is stationary when its statistical properties (mean, variance) do
    not change over time. Many forecasting methods assume or benefit from stationarity.

## Evaluation

Proper scoring rule
:   A metric that is uniquely minimized (or maximized) when the predicted distribution
    matches the true distribution. Log loss and Brier score are proper scoring rules;
    accuracy is not.

Cross-validation
:   Evaluating model performance by repeatedly splitting data into training and test
    sets. Time series cross-validation uses temporal splits to prevent data leakage.

Temporal split
:   A train/test split that respects time ordering: training data always precedes test
    data chronologically. This prevents future information from leaking into training.

Leakage
:   Using information from the future (test period) during training, leading to
    artificially optimistic performance estimates.

Time weighting
:   Applying non-uniform weights to observations or errors so that specific time
    periods carry more or less influence. Yohou supports three weight types:
    `time_weight` (per-timestep), `step_weight` (per-forecasting-step), and
    `vintage_weight` (per-forecast-origin). See
    [Weighting](/pages/explanation/weighting/) for formats and normalization.

Step weight
:   A weight applied per forecasting step (1-step-ahead, 2-step-ahead, etc.)
    during scoring. Controls how much each lead time contributes to the
    aggregated score. Only available in scoring, not training.

Vintage weight
:   A weight applied per vintage (forecast origin date). In training, controls
    per-observation emphasis; in scoring, controls per-vintage score
    aggregation. Available in both `fit()` and `score()`.
