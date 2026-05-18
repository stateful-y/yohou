# How to Clean and Resample Time Series

This guide shows you how to prepare raw data for a forecasting pipeline:
validating dtypes and value ranges, then changing the series frequency with
[`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/)
and
[`Upsampler`](/pages/api/generated/yohou.preprocessing.resampling.Upsampler/).

## Prerequisites

- Familiarity with polars DataFrames and the `"time"` column convention ([Core Concepts](../explanation/core-concepts.md))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Validate Column Types

Yohou expects the `"time"` column to be `Datetime` or `Date` and all value
columns to be numeric. If your data arrives with string columns, cast them
before fitting:

```python
import polars as pl

df = df.with_columns(
    pl.col("time").str.to_datetime(),
    pl.col("value").cast(pl.Float64),
)
```

Run `df.dtypes` to verify the schema before proceeding.

## Clean Implausible Values

Sentinel values and out-of-range readings can bias model training. Replace them
with `null` so that downstream imputation or aggregation handles them
consistently:

```python
df = df.with_columns(
    pl.when(pl.col("value") < 0)
    .then(None)
    .otherwise(pl.col("value"))
    .alias("value")
)
```

If the valid range is known, `clip` is a simpler alternative:

```python
df = df.with_columns(pl.col("temperature").clip(-50, 60))
```

## Downsample with Downsampler

[`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/)
reduces frequency by aggregating within each period. Use it when the data
arrives at a higher frequency than you need for forecasting, or when
high-frequency noise obscures the signal:

```python
from yohou.preprocessing import Downsampler

downsampler = Downsampler(interval="1mo", aggregation="mean")
df_monthly = downsampler.fit_transform(df)
```

The `aggregation` parameter accepts `"mean"`, `"sum"`, `"min"`, `"max"`,
`"first"`, `"last"`, or `"median"`. Choose the one that matches the quantity:
`"sum"` for totals (revenue, rainfall), `"mean"` for rates and averages
(temperature, speed), `"last"` for snapshot values (inventory, account
balance).

For finer control over bin boundaries, pass `closed` and `label`:

```python
downsampler = Downsampler(
    interval="1w",
    aggregation="sum",
    closed="right",
    label="right",
)
```

## Upsample with Upsampler

[`Upsampler`](/pages/api/generated/yohou.preprocessing.resampling.Upsampler/)
increases frequency by creating new timestamps and filling in values. Use it
when your model needs a finer resolution than the source data provides:

```python
from yohou.preprocessing import Upsampler

upsampler = Upsampler(interval="1d", interpolation="linear")
df_daily = upsampler.fit_transform(df)
```

The `interpolation` parameter controls how gaps are filled:

| Method | Behaviour | Good for |
|--------|-----------|----------|
| `"linear"` | Interpolates between known points (default) | Smooth, continuous quantities (temperature, price) |
| `"forward"` | Carries the last known value forward | Step quantities that hold until updated (inventory, status) |
| `"backward"` | Fills from the next known value | Pre-announced values (scheduled rates, published targets) |
| `"nearest"` | Uses the closest known value in either direction | Sparse data where directionality does not matter |

## See Also

- [Handle Missing Data](handle-missing-data.md) for imputation after resampling
- [Handle Long Series](handle-long-series.md) for using downsampling to manage computational cost
- [Preprocessing API Reference](/pages/api/preprocessing/) for full parameter documentation
