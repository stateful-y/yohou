# About Exogenous Features

Forecasting rarely happens in isolation. Electricity prices depend on weather,
retail demand responds to holidays, and industrial output tracks commodity
indices. These external signals are *exogenous features*, and getting them into
a forecasting model correctly is surprisingly subtle. Yohou's three-parameter
API (`X_actual`, `X_future`, `X_forecast`) exists because a single `X`
parameter cannot capture the temporal semantics that matter in production
forecasting.

**API Reference**: [`yohou.base`](../api/base.md) · [`yohou.utils`](../api/utils.md)

## The Three Categories

External data that feeds a forecasting model falls into exactly three categories,
each with distinct temporal properties:

### X_actual: Observation Features

Actual measurements available up to the current observation point. Temperature
readings, sensor data, realized demand, settled prices. These values are
*historical* by definition: you cannot know tomorrow's actual temperature
today.

`X_actual` flows through the `feature_transformer` pipeline. Lag features,
rolling statistics, and other time-dependent transformations apply to it
just as they do to the target variable. At predict time, `X_actual` is not
available (the future hasn't happened yet), so it never appears in the
`predict()` signature.

### X_future: Known-Future Features

Deterministic values available for any date, past or future. Holiday
calendars, day-of-week indicators, scheduled auction prices, planned
maintenance windows. Looking up whether December 25th is a holiday gives the
same answer whether you check in January or November.

`X_future` bypasses the `feature_transformer` entirely. Instead, the framework
windows it forward from each observation point to produce *step-indexed*
columns (`is_holiday_step_1`, `is_holiday_step_2`, ..., `is_holiday_step_H`).
Each step column tells the estimator what the holiday status will be at that
specific forecast horizon.

### X_forecast: External Forecasts

Predictions from external models, each issued at a specific time (the
*vintage*). Weather model output, demand projections, competitor price
forecasts. The 6:00 AM weather forecast and the 9:30 AM forecast for the
same target hour typically differ because the model was updated with newer
data.

`X_forecast` requires a `vintage_time` column that identifies when each
forecast was issued. Like `X_future`, it bypasses `feature_transformer` and
produces step-indexed columns. Unlike `X_future`, different vintages produce
different step values, enabling multi-vintage prediction from a single
observation state.

## Benefits of the Three-Parameter API

Separating exogenous data into three parameters unlocks four capabilities:

**Leakage-free walk-forward evaluation.** The `observe_predict` loop
separates `X_actual` (observation-only, never passed to `predict`) from
`X_future` (predict-safe). This eliminates an entire class of data leakage
where future actual measurements (e.g., tomorrow's temperature) would
otherwise appear in each prediction step.

**Partial features at predict time.** `predict(X_future=holidays)` works
without providing observation features. The API accepts only the data
categories that are relevant at prediction time, so schema validation
passes cleanly.

**Explicit predict-time semantics.** `predict()` does not accept `X_actual`
at all. The estimator uses the stored `_X_t_observed` buffer from fit, and
step columns from `X_future`/`X_forecast` are the only features that can be
overridden at predict time. There is no ambiguity about which features are
used.

**Native support for vintage-indexed data.** `X_forecast` accepts tidy
tables with `vintage_time` columns directly. The framework handles the
pivot from `[vintage_time, time, col]` to step-indexed format internally,
removing manual preprocessing.

## Step-Indexed Columns

Both `X_future` and `X_forecast` become *step-indexed columns* in the
internal feature matrix. This pivoting transforms temporal data into the
tabular format that sklearn estimators expect.

For a forecasting horizon of $H$ and a feature column `temperature`:

$$
\text{temperature\_step\_}h = \text{temperature at } T + h \cdot \Delta t
$$

where $T$ is the observation time and $\Delta t$ is the time series frequency.

The resulting feature matrix has columns
`temperature_step_1, temperature_step_2, ..., temperature_step_H` alongside
the transformer-derived features (`target_lag_1`, `temp_rolling_mean_7`, etc.).

Two public utilities handle this pivoting:

- `pivot_forecasts()` converts tidy `[vintage_time, time, col1, col2]` to
  wide `[time, col1_step_1, col1_step_2, ...]`
- `window_futures()` converts flat `[time, col1, col2]` to wide format by
  windowing forward from each observation time

Both are called internally by `_derive_step_columns()`, but are available as
public utilities for data preparation workflows.

## The Bypass Principle

A key design decision: step-indexed columns bypass `feature_transformer`
entirely. The `feature_transformer` operates on `X_actual` (and optionally
on the target via `target_as_feature`) to produce lags, rolling statistics,
and other observation-derived features. Step columns from `X_future` and
`X_forecast` are already forward-looking by construction: `is_holiday_step_3`
*is* the feature for horizon 3. Applying lag or rolling transformations to
step columns would be meaningless.

This bypass has a practical benefit: at predict time, the framework can swap
step columns without re-running the transformer. Five different weather
forecast vintages produce five different predictions from a single
`predict()` call each, with no deepcopy and no transformer refit.

## Step Feature Alignment

When using the `"direct"` reduction strategy (which fits $H$ independent
estimators, one per forecast horizon), the `step_feature_alignment` parameter
controls which step columns each estimator sees:

| Mode | Estimator $h$ receives | Use case |
|---|---|---|
| `"all"` (default) | All step columns `*_step_1..H` | Maximum information, backward compatible |
| `"matched"` | Only `*_step_h` | Cleanest signal, each estimator sees only its horizon's forecast |
| `"cumulative"` | `*_step_1..h` | All information up to horizon $h$ |

For the electricity pricing use case, `step_feature_alignment="matched"` means
estimator $h$ trains on `(wind_step_h, price_step_h)`: the weather forecast
for time $T+h$ predicting the price at $T+h$. This avoids cross-horizon
information that could confuse simpler estimators.

## Predict-Time Override (Column Swap)

When `predict(X_forecast=...)` is called with new vintage data, the framework
temporarily replaces all step columns in `_X_t_observed` with freshly derived
values. The save-swap-restore flow:

1. Resolve effective raws (provided override or stored `_X_future_raw_`/`_X_forecast_raw_`)
2. Re-derive ALL step columns via `_derive_step_columns()`
3. Save current step columns and raws from `_X_t_observed`
4. Swap raws and step columns into `_X_t_observed`
5. Call the estimator's predict
6. Restore saved raws and step columns (in a `finally` block)

The forecaster's state is unchanged after the call. Five consecutive
`predict()` calls with five different `X_forecast` values return five different
results, all independent.

!!! warning "Thread Safety"
    The column-swap mechanism mutates and restores `_X_t_observed` in place.
    For parallel multi-vintage predictions, `copy.deepcopy(forecaster)` once
    per thread.

## Partial Coverage and Null Handling

Not every `X_forecast` vintage covers the full forecast horizon. If the weather
model issues a 12-step forecast but the model was trained with `H=24`, the
left join produces null step columns for steps 13 through 24. This is by design:
tree-based estimators (XGBoost, LightGBM, HistGradientBoosting) handle null
values natively. Linear models require imputation or complete coverage.

Similarly, if `X_forecast` doesn't cover all training observation times, the
uncovered rows produce null step columns. This is common when forecast archives
start later than the target series.

## Cross-Validation with Exogenous Data

In cross-validation, the three parameters receive different splitting treatment:

- **X_actual** is split by time indices, same as the target `y`
- **X_future** requires no splitting (deterministic data, available for all dates)
- **X_forecast** is filtered by `vintage_time <= T` where $T$ is the fold's training cutoff

The `vintage_time` filter on `X_forecast` prevents future forecast vintages from
leaking into training folds. A forecast issued on Wednesday cannot be used to
train a model whose observation point is Monday.

## Composition Forecasters

All composition forecasters propagate the three parameters:

- **ColumnForecaster**: Routes `X_actual`, `X_future`, `X_forecast` to each
  child forecaster. Children that don't use exogenous features ignore the
  parameters via `ignores_exogenous` tag.

- **DecompositionPipeline**: Passes all three parameters to the residual
  forecaster after trend/seasonality removal.

- **ForecastedFeatureForecaster**: `X_actual` trains the feature forecaster
  (treated as its y) and provides lag features for the target forecaster.
  `X_future` and `X_forecast` pass through to the target forecaster directly.
  At predict time, the target forecaster uses its stored observation window
  for X_actual features; the feature forecaster is not called.

- **VotingForecaster**: All ensemble members receive the same three parameters.

- **SplitConformalForecaster**: Forwards all parameters to the wrapped point
  forecaster.

## Connections

- [Exogenous Features Tutorial](exogenous-tutorial.md): hands-on introduction
  with synthetic data
- [How to Use Exogenous Features](exogenous-howto.md): production workflow
  recipes
- [Forecasting](forecasting.md): general forecasting concepts
- [`pivot_forecasts` API Reference](../api/utils.md): utility for pivoting
  vintage data
- [`window_futures` API Reference](../api/utils.md): utility for windowing
  known-future data
