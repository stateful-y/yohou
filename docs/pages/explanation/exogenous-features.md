# Exogenous Features

Forecasting rarely happens in isolation. Electricity prices depend on weather,
retail demand responds to holidays, and industrial output tracks commodity
indices. These external signals are *exogenous features*, and getting them into
a forecasting model correctly is surprisingly subtle. A single `X` parameter
cannot capture the temporal semantics that matter in production forecasting,
so Yohou separates exogenous data into three parameters: `X_actual`,
`X_future`, and `X_forecast`. All three appear in `fit()` and `observe()`;
only `X_future` and `X_forecast` appear in `predict()`, because observation
features are not available for future time steps.

## The Three Categories

External data that feeds a forecasting model falls into exactly three
categories, each with distinct temporal availability:

### X_actual: Observation Features

Actual measurements available up to the current observation point. Temperature
readings, sensor data, realized demand, settled prices. These values are
*historical* by definition: you cannot know tomorrow's actual temperature
today.

`X_actual` flows through the `feature_transformer` pipeline. Lag features,
rolling statistics, and other time-dependent transformations apply to it
just as they do to the target variable. Because `X_actual` is unavailable for
future time steps, it does not appear in the `predict()` signature. The
forecaster stores the most recent observation window internally and uses it
automatically at predict time.

### X_future: Known-Future Features

Values you can look up for a future timestamp but cannot derive from the
observation point. Holiday calendars, scheduled auction prices, planned
maintenance windows, announced promotions. Looking up whether December 25th is
a holiday gives the same answer whether you check in January or November, but
no amount of inspecting today's date tells you when Easter falls next year.
That takes a calendar.

`X_future` bypasses the `feature_transformer` entirely. Instead, the framework
windows it forward from each observation point to produce *step-indexed*
columns (`is_holiday_step_1`, `is_holiday_step_2`, ..., `is_holiday_step_H`).
Each step column tells the estimator what the holiday status will be at that
specific forecast horizon.

#### What Belongs Here

`feature_transformer` only ever runs on the observation frame, which stops at
the observation point `T`. It can compute a feature for `T`, for `T-1`, for any
timestamp already observed, and for none beyond. That single fact decides the
channel:

> Does the feature's value at `T` determine its value at `T+h`?

If it does, `feature_transformer` already suffices and `X_future` is the wrong
home. A Fourier pair is the clearest case: `sin(2*pi*(t+h)/S)` is a fixed linear
combination of the sine and cosine at `t`, so once the estimator sees the pair
at `T` it can express the value at every horizon. Windowing it forward across
`H` steps adds columns without adding information, and those columns are exactly
collinear with the pair the model already had.

If it does not, nothing computed at `T` can stand in, and the forward window is
the only way to put the value in front of the estimator. Whether next Tuesday is
a public holiday is not a function of anything measurable today. It is a lookup.

The same test, in the form you can apply while writing the `fit()` call: **can
you compute it from the timestamp alone, or do you need an external table?** A
timestamp alone means a *clock feature*, which belongs in `feature_transformer`
(see [`FourierFeatureTransformer`](/pages/api/generated/yohou.preprocessing.time_features.FourierFeatureTransformer/)
and [`CalendarFeatureTransformer`](/pages/api/generated/yohou.preprocessing.calendar.CalendarFeatureTransformer/)).
An external table means an *event feature*, which belongs in `X_future`.

Note that "deterministic and knowable in advance" does not separate the two. A
day-of-week indicator and a holiday calendar are both perfectly deterministic
and both knowable for any future date, yet they belong in different channels.
Determinism is not the question; derivability from `T` is.

#### Holidays Sit On Both Sides

Holidays deserve a note, because they can legitimately appear on either channel
and the two uses are complementary rather than competing.

[`HolidayFeatureTransformer`](/pages/api/generated/yohou.preprocessing.calendar.HolidayFeatureTransformer/)
on `X_actual` gives *past* holiday effects: whether recently observed timestamps
were holidays, which is what you want when demand rebounds the day after a
closure. It runs on the observation frame, so it cannot say anything about a
holiday next week.

A holiday calendar passed through `X_future` gives *future* holiday effects:
whether each forecast step lands on a holiday, which is what you want when the
holiday itself moves demand.

Reaching for one does not rule out the other. A model that needs both the
closure and the rebound uses both.

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

Bypassing `feature_transformer` is a statement about that slot, not about the
frame. `feature_transformer` operates on single-axis `X_actual` data, and an
`X_forecast` frame carries a second time axis it cannot read. The frame can still
be transformed, just earlier and by a transformer built for its shape: derive the
features first, then pass the result in through `X_forecast`. See
[Transformer Kinds](transformer-kinds.md) for the two transformer kinds and why
the vintage axis needs its own.

## Benefits of the Three-Parameter API

Separating exogenous data into three parameters provides four capabilities:

**Leakage-free walk-forward evaluation.** The `observe`/`predict` loop
separates `X_actual` (observation-only, never passed to `predict`) from
`X_future` and `X_forecast` (both available at predict time). This eliminates
data leakage where future actual measurements (e.g., tomorrow's temperature)
would otherwise appear in each prediction step.

**Partial features at predict time.** `predict(X_future=holidays)` or
`predict(X_forecast=weather)` works without providing observation features.
The API accepts only the data categories relevant at prediction time, so
schema validation passes cleanly. Both overrides are optional and can be
used independently or together.

**Explicit predict-time semantics.** `predict()` accepts `X_future` and
`X_forecast` but not `X_actual`. The estimator uses the stored observation
buffer (`_X_t_observed`) from fit for observation-derived features, and
step columns from `X_future`/`X_forecast` are the only features that can be
overridden at predict time. There is no ambiguity about which features
contribute to a given prediction.

**Native support for vintage-indexed data.** `X_forecast` accepts tidy
tables with a `vintage_time` column directly. The framework handles the
pivot from `[vintage_time, time, col]` to step-indexed format internally,
removing manual preprocessing.

## Step-Indexed Columns

Both `X_future` and `X_forecast` become *step-indexed columns* in the
internal feature matrix. This pivoting transforms temporal data into the
tabular format that sklearn estimators expect.

For a forecasting horizon $H$ and a feature column `temperature`:

$$
\text{temperature\_step\_}h = \text{temperature}(T + h \cdot \Delta t)
$$

where $T$ is the observation time and $\Delta t$ is the time series frequency.

The resulting feature matrix has columns
`temperature_step_1, temperature_step_2, ..., temperature_step_H` alongside
any transformer-derived features (`target_lag_1`, `temp_rolling_mean_7`, etc.).

Two public utilities handle this pivoting:

- [`window_forecasts()`](/pages/api/generated/yohou.utils.pivot.window_forecasts/) selects the latest vintage at or before each
  observation time (as-of matching) and converts tidy
  `[vintage_time, time, col1, col2]` to wide
  `[time, col1_step_1, col1_step_2, ...]`
- [`window_futures()`](/pages/api/generated/yohou.utils.pivot.window_futures/) converts flat `[time, col1, col2]` to wide format by
  looking forward from each observation time

Both are called internally by `_derive_step_columns()` but are available as
public utilities for data preparation workflows.

## The Bypass Principle

Step-indexed columns bypass `feature_transformer` entirely. The
`feature_transformer` operates on `X_actual` (and optionally on the target
via `target_as_feature`) to produce lags, rolling statistics, and other
observation-derived features. Step columns from `X_future` and `X_forecast`
are already forward-looking by construction: `is_holiday_step_3` *is* the
feature for horizon 3. Applying lag or rolling transformations to them would
be meaningless.

This bypass also brings a practical benefit: at predict time, the framework
can swap step columns without re-running the transformer. Five different
weather forecast vintages produce five different predictions from a single
`predict()` call each, with no deepcopy and no transformer refit.

## Step Feature Alignment

When using the `"direct"` reduction strategy (which fits $H$ independent
estimators, one per forecast horizon), the `step_feature_alignment` parameter
controls which step columns each estimator sees. This parameter is available
on point, interval, and class-probability reduction forecasters.

| Mode | Estimator $h$ receives | Use case |
|---|---|---|
| `"all"` (default) | All step columns `*_step_1..H` | Maximum information, backward compatible |
| `"matched"` | Only `*_step_h` | Cleanest signal: each estimator sees only its horizon's forecast |
| `"cumulative"` | `*_step_1..h` | All information up to and including horizon $h$ |

For an electricity pricing use case, `step_feature_alignment="matched"` means
estimator $h$ trains on `(wind_step_h, price_step_h)`: the weather forecast
for time $T+h$ predicting the price at $T+h$. This avoids cross-horizon
information that could confuse simpler estimators.

### Why Only Direct

The parameter applies to `"direct"` alone. Setting it on another strategy
warns at fit and changes nothing, but the two exclusions are not the same kind
of thing.

`"multi-output"` *cannot* filter. One estimator predicts every horizon from a
single feature vector, and output $h$ reads `wind_step_h` from that same
vector, so every step column has to be present for some output. There is no
per-estimator view to narrow, because there is only one estimator. Since
`"multi-output"` is the default, this is the case most users are in.

`"dir-rec"` *could* filter. It fits $H$ estimators, one per step, exactly as
`"direct"` does, and a per-step column filter applied before each step's
feature augmentation would be well defined. It does not, and that is a
deliberate scope decision rather than a structural limit: combining a filtered
base with the accumulating augmentation columns raises design questions that
have not been worked through. Recording it here is what stops the exclusion
from being read as an oversight.

## Predict-Time Override (Column Swap)

When `predict(X_forecast=...)` or `predict(X_future=...)` is called with new
data, the framework temporarily replaces all step columns in `_X_t_observed`
with freshly derived values. The save-swap-restore flow:

1. Resolve effective raws: use the provided override, or fall back to the
   stored `_X_future_raw_` / `_X_forecast_raw_` from fit
2. Re-derive all step columns via `_derive_step_columns()`
3. Save the current step columns and raws from `_X_t_observed`
4. Swap the new raws and step columns into `_X_t_observed`
5. Call the underlying estimator's predict
6. Restore saved raws and step columns (in a `finally` block, so state is
   always restored even on error)

The forecaster's state is unchanged after the call. Five consecutive
`predict()` calls with five different `X_forecast` values return five
different results, all independent.

!!! warning "Thread Safety"
    The column-swap mechanism mutates and restores `_X_t_observed` in place.
    For parallel multi-vintage predictions, `copy.deepcopy(forecaster)` once
    per thread.

## Partial Coverage and Null Handling

Not every `X_forecast` vintage covers the full forecast horizon. If the weather
model issues a 12-step forecast but the model was trained with `H=24`, the
left join produces null step columns for steps 13 through 24. This is by
design: tree-based estimators (XGBoost, LightGBM, HistGradientBoosting)
handle null values natively. Linear models require imputation or complete
coverage.

Similarly, `X_forecast` may not cover all training observation times. Rows
without matching forecast data produce null step columns. This is common when
forecast archives start later than the target series.

Conversely, a vintage whose timestamps extend beyond the forecasting horizon
is clipped before pivoting. Each vintage keeps only timestamps in
$(T_v,\; T_v + H \cdot \Delta t]$ where $T_v$ is the vintage time. This
guarantees that step columns always span exactly `1..H` per value column,
preventing spurious `step_(H+1)` columns from appearing in the feature
matrix. If clipping leaves fewer than $H$ timestamps, the missing step
columns are padded with null and a `UserWarning` is emitted.

## The Observe-Predict Loop

The `observe()` method accepts `X_actual`, `X_future`, and `X_forecast`,
matching the `fit()` signature. When new data becomes available after fitting,
`observe()` extends the internal observation buffer (`_X_t_observed`) with
new `X_actual` data processed through `feature_transformer`, and re-derives
step columns from `X_future` and `X_forecast`.

A typical walk-forward evaluation alternates between `observe()` (to feed
new actual data) and `predict()` (to produce forecasts from the updated
state). The three-parameter separation ensures that `observe()` can update
the observation window with `X_actual` without any of that data leaking into
the prediction step, because `predict()` only accepts `X_future` and
`X_forecast`.

## Cross-Validation with Exogenous Data

In cross-validation, the three parameters receive different splitting
treatment:

- **`X_actual`** is split by time indices, same as the target `y`
- **`X_future`** is passed in full to both training and testing (deterministic
  data is available for all dates, so no filtering is needed)
- **`X_forecast`** is split by `vintage_time` range: training receives
  vintages where `vintage_time` $\leq T$, testing receives vintages where
  $T <$ `vintage_time` $\leq T_\text{test\_end}$, with $T$ as the fold's
  training cutoff

The `vintage_time` filter on `X_forecast` prevents future forecast vintages
from leaking into training folds. A forecast issued on Wednesday cannot train
a model whose observation point is Monday. The test fold only sees vintages
issued within the test window, matching what would be available in a real
walk-forward deployment.

The same splitting logic is available via
[`train_test_split`](/pages/api/generated/yohou.model_selection.split.train_test_split/)
for a single temporal split. Positional arrays (`y`, `X_actual`) are split by
row position, `X_future` is not split (pass it to both `fit` and `predict`
directly), and `X_forecast` is split by `vintage_time` range using the cutoff
inferred from the first array's `"time"` column.

## Composition Forecasters

All composition forecasters propagate the three parameters to their children:

- [**`ColumnForecaster`**](/pages/api/generated/yohou.compose.column_forecaster.ColumnForecaster/) routes `X_actual`, `X_future`, `X_forecast` to each
  child forecaster. Children that don't use exogenous features ignore the
  parameters via the `requires_exogenous` tag.

- [**`DecompositionPipeline`**](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) passes all three parameters to the residual
  forecaster after trend/seasonality removal.

- [**`ForecastedFeatureForecaster`**](/pages/api/generated/yohou.compose.forecasted_feature_forecaster.ForecastedFeatureForecaster/) uses `X_actual` as the target for the
  feature forecaster (training it to predict the exogenous series). At predict
  time the feature forecaster produces a forecast that is passed to the target
  forecaster as `X_forecast` (merged with any caller-supplied `X_forecast`), so
  the forecasted features become step columns the target consumes. `X_future`
  is forwarded to the target unchanged. Its `feature_stride` parameter sets how
  often the feature forecaster regenerates this forecast (every step by default),
  for cases where the feature model is too expensive to re-run on every step.

- [**`VotingPointForecaster`**](/pages/api/generated/yohou.ensemble.voting_point.VotingPointForecaster/),
  [**`VotingIntervalForecaster`**](/pages/api/generated/yohou.ensemble.voting_interval.VotingIntervalForecaster/),
  and [**`VotingClassProbaForecaster`**](/pages/api/generated/yohou.ensemble.voting_class_proba.VotingClassProbaForecaster/)
  each pass all three parameters to every ensemble member.

- [**`SplitConformalForecaster`**](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/) forwards all parameters to the wrapped point
  forecaster.

## Connections

- [Exogenous Features Tutorial](../tutorials/exogenous-features.md) provides a
  hands-on introduction with synthetic data
- [How to Use Exogenous Features](../how-to/exogenous-features.md) covers
  production workflow recipes
- [Forecaster Composition](forecaster-composition.md) describes
  [`ForecastedFeatureForecaster`](/pages/api/generated/yohou.compose.forecasted_feature_forecaster.ForecastedFeatureForecaster/),
  which automates the two-stage pattern of forecasting exogenous features
  before the target
- [Reduction Forecasting](reduction-forecasting.md) explains the direct
  reduction strategy and how `step_feature_alignment` fits in
- [`window_forecasts`](../api/utils.md) and [`window_futures`](../api/utils.md)
  handle vintage pivoting and known-future windowing respectively
