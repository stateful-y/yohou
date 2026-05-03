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

## Why Three Parameters, Not One

The single-`X` API creates four problems that cannot be fixed without
separating the data categories:

**Data leakage in walk-forward evaluation.** The `observe_predict` loop
passes the full `X` to `predict()` at each step. If `X` contains actual
temperature measurements alongside holiday indicators, future temperature
values leak into each prediction. Separating `X_actual` (observation-only)
from `X_future` (predict-safe) eliminates this class of leakage.

**No partial features at predict time.** With a single `X`, calling
`predict(X=holidays)` fails schema validation because the model was fitted
with temperature columns too. The three-parameter API lets you pass
`predict(X_future=holidays)` without providing observation features.

**Silent ignoring of X in predict.** The current `_X_t_observed` buffer
from fit is what the estimator actually uses for prediction; `X` passed to
`predict()` is silently ignored in non-recursive mode. Users pass future
actuals thinking they improve predictions, but the values are never used.
The three-parameter API makes this explicit: `predict()` does not accept
`X_actual` at all.

**No support for vintage-indexed data.** Weather model output arrives as
tidy tables with issuance timestamps. A single `X` parameter has no place
for `vintage_time`, forcing users to manually pivot and join before calling
`fit()`.

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
- `window_future()` converts flat `[time, col1, col2]` to wide format by
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

1. Compute effective raws (provided override or stored `_X_future_raw_`/`_X_forecast_raw_`)
2. Re-derive ALL step columns via `_derive_step_columns()`
3. Save current step columns from `_X_t_observed`
4. Replace step columns
5. Call the estimator's predict
6. Restore saved step columns

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

- **ForecastedFeatureForecaster**: The feature forecaster receives `X_actual`
  and `X_future`. Its predictions become the main forecaster's `X_forecast`.

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
- [`window_future` API Reference](../api/utils.md): utility for windowing
  known-future data
