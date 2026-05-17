*[forecasting horizon]: The number of future timesteps to predict.
*[observation horizon]: The number of recent time steps a stateful component must retain in memory to produce output.
*[memory buffer]: The internal store of recent rows that a stateful component maintains.
*[composite method]: A method that combines two operations in sequence, such as observe_predict or observe_transform.
*[point forecast]: A single numeric prediction per timestep, produced by predict().
*[interval forecast]: A pair of bounds per timestep and coverage rate, produced by predict_interval().
*[class-probability forecast]: A probability distribution over categorical classes per timestep, produced by predict_class_proba().
*[coverage rate]: The target probability that an interval forecast should contain the true value.
*[recursive prediction]: Predicting beyond the training horizon by feeding predictions back as inputs iteratively.
*[error accumulation]: The compounding of prediction errors when recursive prediction feeds earlier predictions back as inputs.
*[rolling evaluation]: Assessing forecaster performance by repeatedly observing new data and predicting, producing one vintage per iteration.
*[time column contract]: Every DataFrame in Yohou must have a "time" column containing datetime values.
*[panel data]: Multiple related time series handled together, with groups identified by the __ separator in column names.
*[group prefix]: The double-underscore separator between a panel group name and a column name.
*[panel strategy]: Controls how a forecaster handles panel data: "global" fits one model with per-group transformer state.
*[exogenous features]: Additional input columns beyond the target that may improve forecasts.
*[known-future features]: Exogenous features whose values are available for the prediction horizon at forecast time.
*[step-indexed columns]: Feature columns named with a _step_h suffix that carry different values for each forecasting step.
*[reduction strategy]: Converting a time series forecasting problem into a tabular supervised learning problem.
*[step feature alignment]: Controls which step-indexed columns each direct estimator receives in a direct or dir-rec reduction strategy.
*[target transformer]: A transformer applied to the target series before tabularization.
*[feature transformer]: A transformer applied to the feature matrix before tabularization.
*[stateful transformer]: A transformer that maintains an internal observation window and updates it during observe().
*[stateless transformer]: A transformer whose output depends only on its fitted parameters and the current input.
*[forecaster composition]: Building complex forecasting workflows by combining simpler components.
*[variance stabilization]: Transforming a time series so that its error variance is approximately constant over time.
*[conformal prediction]: A distribution-free method for constructing prediction intervals with finite-sample coverage guarantees.
*[conformity score]: The residual measurement computed on a held-out calibration set during conformal prediction.
*[calibration set]: A portion of the training data held out from model fitting, used to compute conformity scores.
*[similarity measure]: A function that weights conformity scores based on the similarity between prediction contexts.
*[metadata routing]: The mechanism by which sample-level metadata flows from a composite estimator down to its child components.
*[aggregation method]: Controls which dimensions a scorer collapses when computing its result.
*[forecast error]: The difference between a predicted value and the corresponding actual value, computed on out-of-sample data.
*[scale-dependent metric]: A metric whose value depends on the scale of the data, such as MAE or RMSE.
*[scale-independent metric]: A metric that normalizes errors so they can be compared across series of different scales.
*[cross-validation]: Evaluating model performance by repeatedly splitting data into training and test sets.
*[temporal split]: A train/test split that respects time ordering, with training data always preceding test data.
*[expanding window splitter]: A cross-validation splitter where each fold grows the training window while keeping the test window fixed.
*[sliding window splitter]: A cross-validation splitter that maintains a fixed-size training window sliding forward in time.
*[concept drift]: A shift in the statistical properties of the data over time, causing older models to become less accurate.
*[time weighting]: Applying non-uniform weights to observations or errors so that specific time periods carry more or less influence.
*[proper scoring rule]: A metric uniquely minimized when the predicted distribution matches the true distribution.
*[ACF]: Autocorrelation function, measuring the linear correlation between a time series and a lagged copy of itself.
*[baseline]: A simple reference model (such as SeasonalNaive) used to judge whether more complex approaches add value.
*[calibration]: The degree to which predicted probabilities or interval coverage rates match observed frequencies.
*[dir-rec strategy]: A hybrid reduction strategy that trains sequential models, each receiving predictions from all previous steps as features.
*[direct strategy]: A reduction strategy that trains one independent model per forecast step in the horizon.
*[ensemble]: A method that combines predictions from multiple independent forecasters to produce a single forecast.
*[feature matrix]: The tabular matrix of input features produced during tabularization, used as input to the estimator.
*[fold]: A single train/test partition produced by a cross-validation splitter.
*[harmonics]: Integer multiples of a base frequency used to capture seasonal patterns in Fourier features.
*[heteroscedasticity]: A condition where the variance of a time series changes over time, often growing with the level.
*[multi-output strategy]: A reduction strategy that trains a single model to predict all forecast steps simultaneously.
*[outlier]: An observation that deviates significantly from the expected pattern of a time series.
*[PACF]: Partial autocorrelation function, measuring correlation after removing intermediate lag effects.
*[prediction intervals]: A pair of lower and upper bounds per timestep that quantifies forecast uncertainty at a given coverage rate.
*[residuals]: The differences between predicted and actual values, used to diagnose model performance.
*[scorer]: An object that computes a metric from truth and prediction DataFrames after fitting on training data.
*[stride]: The number of time steps observed between consecutive predict calls in a rolling evaluation.
*[trend]: A long-term upward or downward movement in the level of a time series.
*[vintage]: A set of predictions issued at the same point in time, identified by the vintage_time column.
*[walk-forward evaluation]: Another name for rolling evaluation: stepping through a test set chronologically, observing actual values before issuing each forecast.
*[seasonality]: A repeating, predictable pattern in a time series that occurs at regular intervals (daily, weekly, yearly).
*[lag]: An observation from a previous time step used as an input feature for prediction.
*[STL]: Seasonal and Trend decomposition using Loess, a method that separates a time series into trend, seasonal, and remainder components.
*[autoregressive features]: Input features derived from past values of the target series itself, such as lags or rolling statistics.
*[tail errors]: Large forecast errors in the extreme quantiles of the error distribution, indicating poor performance on unusual observations.
*[structural break]: An abrupt, permanent change in the statistical properties of a time series caused by an external event.
*[regime change]: A shift in the underlying data-generating process that produces a new statistical pattern.
*[homoscedasticity]: A condition where the variance of a time series remains constant over time.
*[MRO]: Method Resolution Order, the sequence Python follows when searching for a method across a class hierarchy.
*[vintage weight]: A metadata weight applied per vintage in multi-vintage scoring to control the relative importance of each forecast vintage.
*[step weight]: A metadata weight applied per forecast step to control the relative importance of predictions at different horizons.
