# How to Handle Long Series

This guide shows you how to limit look-back history, down-weight old
observations during evaluation, and resample when the data frequency does not
match the forecast requirement.

## Prerequisites

- Familiarity with core concepts such as `observation_horizon`
  ([Core Concepts](../explanation/core-concepts.md))
- Familiarity with time weighting ([Use Time Weighting](time-weighting.md))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Limit History with observation_horizon

Each stateful transformer exposes a read-only `observation_horizon` property
that reports how many past timesteps it retains. The value is derived from the
constructor parameters: for
[`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/)
it equals the maximum lag, and for
[`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/)
it equals the seasonality period. Older observations are dropped as new ones
arrive via `observe`.

To keep only recent history, choose parameter values that match the look-back
you need:

```python
from yohou.preprocessing import LagTransformer
from yohou.stationarity import SeasonalDifferencing

# observation_horizon = 7 (max lag)
lag_transformer = LagTransformer(lag=[1, 7])

# observation_horizon = 7 (seasonality period)
seasonal_diff = SeasonalDifferencing(seasonality=7)
```

In a [`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/), the pipeline's `observation_horizon` is the sum of
all its steps' horizons. Inspect it after fitting to verify the total
look-back is what you expect:

```python
from yohou.compose import FeaturePipeline

pipeline = FeaturePipeline([
    ("deseason", SeasonalDifferencing(seasonality=7)),
    ("lags", LagTransformer(lag=[1, 7])),
])
pipeline.fit(y)
pipeline.observation_horizon  # 14
```

## Weighting Recent Observations in Evaluation

Limiting history is a binary cutoff. Time weighting is a softer alternative
that keeps all data but gives more importance to recent errors during model
selection. Construct an
[`ExponentialDecayWeighter`](/pages/api/generated/yohou.utils.weighting.ExponentialDecayWeighter/)
or
[`LinearDecayWeighter`](/pages/api/generated/yohou.utils.weighting.LinearDecayWeighter/)
and pass it to the `time_weighter` slot of scorers and forecasters:

```python
from yohou.utils.weighting import ExponentialDecayWeighter

weighter = ExponentialDecayWeighter(half_life=365)
```

See [Use Time Weighting](time-weighting.md) for full recipes on configuring
weighters on scorers and forecasters and tuning them via search.

## Downsampling to a Lower Frequency

When data arrives at a higher frequency than the forecast requirement,
downsampling reduces computational cost and can improve model quality by
removing noise that is irrelevant to the forecasting horizon.

[`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/)
aggregates observations to a lower frequency. Place it at the start of the
pipeline, before any stateful transformers:

```python
from yohou.preprocessing.resampling import Downsampler

# Aggregate hourly data to daily, summing within each day
downsampler = Downsampler(interval="1d", aggregation="sum")
```

Available `aggregation` options: `"mean"` (default), `"sum"`, `"min"`,
`"max"`, `"first"`, `"last"`, and `"median"`. If the boundary alignment
matters (for example, whether a daily bin starts at midnight or noon), adjust
`closed` and `label`:

```python
downsampler = Downsampler(
    interval="1d",
    aggregation="sum",
    closed="right",
    label="right",
)
```

## Upsampling to a Higher Frequency

[`Upsampler`](/pages/api/generated/yohou.preprocessing.resampling.Upsampler/)
increases frequency by interpolating between existing observations. Use it
only when obtaining higher-frequency input data is not possible, because
interpolation creates artificial data points without genuine information
content:

```python
from yohou.preprocessing.resampling import Upsampler

upsampler = Upsampler(interval="1h", interpolation="linear")
```

Available `interpolation` options: `"linear"` (default), `"nearest"`,
`"forward"`, and `"backward"`.

## Combining Techniques

A common recipe for long series: downsample high-frequency data, limit
look-back via transformer parameters, and evaluate with time-weighted scoring.

```python
from yohou.preprocessing.resampling import Downsampler
from yohou.stationarity import SeasonalDifferencing
from yohou.preprocessing import LagTransformer
from yohou.compose import FeaturePipeline
from yohou.utils.weighting import ExponentialDecayWeighter
from yohou.metrics import MeanAbsoluteError

# 1. Downsample hourly data to daily
downsampler = Downsampler(interval="1d", aggregation="mean")

# 2. Build a feature pipeline with bounded look-back
pipeline = FeaturePipeline([
    ("deseason", SeasonalDifferencing(seasonality=7)),
    ("lags", LagTransformer(lag=[1, 7, 14])),
])

# 3. Evaluate with exponential decay (half-life of one year)
scorer = MeanAbsoluteError(time_weighter=ExponentialDecayWeighter(half_life=365))
scorer.fit(y_train)
score = scorer.score(y_test, y_pred)
```

## See Also

- [Handle Short Series](handle-short-series.md) for the opposite problem:
  when history is too limited for standard training or cross-validation.
- [Core Concepts](../explanation/core-concepts.md) for the `observation_horizon`
  mechanism and the fit/observe/predict lifecycle.
- [Use Time Weighting](time-weighting.md) for full recipes on applying decay
  weights to scorers, pipelines, and search objects.
