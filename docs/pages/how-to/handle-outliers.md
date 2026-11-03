# How to Handle Outliers

This guide shows you how to clip or neutralise outliers before fitting a
forecasting model, and how to prevent outliers in the calibration set from
inflating conformal prediction intervals.

## Prerequisites

- Familiarity with the pipeline API ([Getting Started](../tutorials/getting-started.md))
- For the conformal interval section: familiarity with interval forecasting
  ([Interval Forecasting](../explanation/interval-forecasting.md))

<!-- COMPANION_NOTEBOOKS -->

## 1. Decide: Clip, Null, or Keep

Before choosing a handler, decide what the outlier represents:

- **Recording artefact** (sensor glitch, data entry error): clip to valid bounds
  or replace with null so downstream imputation handles it.
- **Genuine extreme event** (flash crash, once in a decade storm): keep the
  value so the model can learn from it, and optionally flag it as a binary
  feature. See [Preprocessing](../explanation/preprocessing.md) for how outlier
  handling interacts with stateful transformers.

## 2. Clip to Known Bounds

When you know the plausible range for your data, use
[`OutlierThresholdHandler`](/pages/api/generated/yohou.preprocessing.OutlierThresholdHandler/)
with fixed thresholds. Pass `None` to leave one side unbounded:

```python
from yohou.preprocessing import OutlierThresholdHandler

handler = OutlierThresholdHandler(low=0.0, high=1000.0)
handler.fit(y_train)
y_clipped = handler.transform(y_train)
```

To replace outliers with null instead of clipping, set `strategy="nan"`. This
is useful when you want a downstream imputer to fill the gaps:

```python
handler = OutlierThresholdHandler(low=0.0, high=1000.0, strategy="nan")
```

## 3. Clip to Data-Driven Bounds

When you do not have domain knowledge for fixed thresholds, use
[`OutlierPercentileHandler`](/pages/api/generated/yohou.preprocessing.OutlierPercentileHandler/)
to learn bounds from training data. During `fit`, the handler records the
requested percentiles; during `transform`, it clips (or nullifies) values
outside those bounds:

```python
from yohou.preprocessing import OutlierPercentileHandler

handler = OutlierPercentileHandler(low=1, high=99)
handler.fit(y_train)
y_clipped = handler.transform(y_train)
```

As with `OutlierThresholdHandler`, pass `strategy="nan"` to replace outliers
with null instead of clipping.

## 4. Handle Outliers by Time of Day

When your series has a daily cycle, a single threshold treats every hour alike.
On hourly electricity load where nights run near 60 and days near 300, a single
95th percentile sits near the daytime peak, so a night reading of 150 (more than
double the normal night level) is never caught. Give the handler a
[`CalendarSegmenter`](/pages/api/generated/yohou.segmenting.CalendarSegmenter/)
to work by time of day.

**Handle only part of the day.** With a `segmenter`, only rows the segmenter
selects are handled, and percentiles are learned from those rows only. Rows
outside the segment are returned unchanged:

```python
from yohou.preprocessing import OutlierPercentileHandler
from yohou.segmenting import CalendarSegmenter

business_hours = CalendarSegmenter("hour", values=range(8, 19))
handler = OutlierPercentileHandler(high=99, segmenter=business_hours)
```

**Learn one limit per hour.** With `by_segment=True`, the handler fits a
separate threshold pair for each hour, so 150 is clipped at 03:00 and kept at
12:00:

```python
handler = OutlierPercentileHandler(
    high=99,
    segmenter=CalendarSegmenter("hour"),
    by_segment=True,
)
handler.fit(y_train)
handler.segment_thresholds_["load"][3]  # (None, upper limit for 03:00)
```

Choose the coarsest segmentation that captures the cycle. Each hour gets
roughly a twenty-fourth of the training rows, and combining conditions (for
example weekday and hour with
[`IntersectionSegmenter`](/pages/api/generated/yohou.segmenting.IntersectionSegmenter/))
divides them further.

**Labels with too little data.** A label with fewer than `min_samples` finite
values (default 30) uses the pooled thresholds, which are fitted on all
in-segment rows together, and `fit` warns once, naming the affected (column,
label) pairs. A label that `fit` never saw, such as June when the training data
covered January to March with `CalendarSegmenter("month")`, also uses the pooled
thresholds, without a warning.

**Match the segment to the data frequency.** An hour segment on daily data
selects nothing (every daily timestamp sits at 00:00), so `fit` warns that the
column will be returned unchanged. For local hours on UTC data, pass
`time_zone`, for example `CalendarSegmenter("hour", values=range(8, 19), time_zone="Europe/Paris")`.

## 5. Place in a Pipeline

Both handlers should appear early in a
[`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/),
before lag features and stateful stationarity transforms (which would otherwise
propagate extreme values into derived columns):

```python
from yohou.compose import FeaturePipeline
from yohou.preprocessing import OutlierPercentileHandler, LagTransformer

pipeline = FeaturePipeline([
    ("outlier_clip", OutlierPercentileHandler(low=1, high=99)),
    ("lags", LagTransformer(lag=[1, 7])),
])
```

## 6. Prevent Outliers from Inflating Conformal Intervals

When using conformal prediction, a single extreme conformity score in the
calibration set raises the quantile used to set the interval margin, widening
every interval produced by the forecaster. Two complementary strategies help:

**1. Clip outliers before calibration.** Apply a handler to the calibration
period so extreme values do not enter the conformity score distribution.

**2. Use a normalised conformity score.**
[`GammaResidual`](/pages/api/generated/yohou.metrics.GammaResidual/)
divides each residual by the magnitude of the prediction, preventing
large magnitude outliers from dominating the calibration quantile:

```python
from yohou.interval import SplitConformalForecaster
from yohou.metrics.conformity import GammaResidual

forecaster = SplitConformalForecaster(
    point_forecaster=...,
    conformity_scorer=GammaResidual(),
)
forecaster.fit(y_train, forecasting_horizon=7, coverage_rates=[0.9])
```

`GammaResidual` is most effective when outliers tend to occur at high or low
predicted levels (multiplicative noise). For purely additive noise, the default
`Residual` scorer is usually sufficient.

## See Also

- [Handle Missing Data](handle-missing-data.md) for imputing gaps that outlier
  removal may create
- [Interval Forecasting](../explanation/interval-forecasting.md) for the
  conformal coverage guarantee and how calibration set composition affects
  interval validity
- [Clean and Resample](clean-and-resample.md) for removing or imputing missing values before modeling
- [Use Preprocessing Transformers](use-preprocessing-transformers.md) for individual transformer usage patterns
- [`yohou.segmenting`](/pages/api/segmenting/) for the segmenters that restrict handling to part of the calendar
