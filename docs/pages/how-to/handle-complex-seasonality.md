# How to Handle Complex Seasonality

This guide shows you four approaches for handling multiple overlapping seasonal
periods (e.g., weekly + annual in daily data) in yohou: Fourier terms, nested
decomposition, feature engineering, and calendar profiles for patterns tied to
local time or calendar dates.

## Prerequisites

- Familiarity with decomposition and stationarity
  ([Stationarity](../explanation/stationarity.md))
- Familiarity with the pipeline API ([Getting Started](../tutorials/getting-started.md))

<!-- COMPANION_NOTEBOOKS -->

## Choose an Approach

| Situation | Recommended approach |
|---|---|
| Periods are known; seasonal shape is smooth | Fourier terms |
| Shape is irregular; you have 3+ complete cycles per period | Nested decomposition |
| You have exogenous features or a tree-based model | Feature engineering |
| The pattern follows local time, calendar dates, or holidays | Calendar profiles |

These approaches can be combined: use decomposition for the dominant period and
Fourier terms for a secondary one, or pair calendar features with Fourier
harmonics.

## Use Fourier Terms for Smooth Patterns

[`FourierSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.FourierSeasonalityForecaster/)
represents a seasonal pattern as sine/cosine pairs at specified harmonics. Each
instance handles one period, so wrap multiple instances in a
[`DecompositionPipeline`](/pages/api/generated/yohou.compose.DecompositionPipeline/)
for overlapping periodicities:

```python
from yohou.compose import DecompositionPipeline
from yohou.stationarity.seasonality import FourierSeasonalityForecaster

pipeline = DecompositionPipeline(
    forecasters=[
        ("weekly", FourierSeasonalityForecaster(seasonality=7, harmonics=[1, 2, 3])),
        ("annual", FourierSeasonalityForecaster(seasonality=365.25, harmonics=[1, 2, 3, 4, 5])),
    ],
)
```

More harmonics capture more complex shapes but use more degrees of freedom.
Start with 2 to 4 and increase if the residuals still show seasonal structure.

## Nest Decomposition for Irregular Shapes

When the seasonal shape is irregular and you have enough data to estimate it
directly, use
[`PatternSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.PatternSeasonalityForecaster/)
inside a
[`DecompositionPipeline`](/pages/api/generated/yohou.compose.DecompositionPipeline/).
Each forecaster fits the residuals left by the previous one:

```python
from yohou.compose import DecompositionPipeline
from yohou.stationarity.seasonality import PatternSeasonalityForecaster

pipeline = DecompositionPipeline(
    forecasters=[
        ("weekly", PatternSeasonalityForecaster(seasonality=7)),
        ("annual", PatternSeasonalityForecaster(seasonality=365)),
    ],
)
```

Order by period length (shortest first) so the dominant short cycle is removed
before estimating the longer one.

## Add Calendar and Fourier Features for Tree-based Models

For tree-based regressors, encoding seasonal patterns as features often
outperforms explicit decomposition. Use
[`CalendarFeatureTransformer`](/pages/api/generated/yohou.preprocessing.CalendarFeatureTransformer/)
and
[`FourierFeatureTransformer`](/pages/api/generated/yohou.preprocessing.FourierFeatureTransformer/)
as part of a
[`FeatureUnion`](/pages/api/generated/yohou.compose.FeatureUnion/):

```python
from yohou.compose import FeatureUnion
from yohou.point import PointReductionForecaster
from yohou.preprocessing.calendar import CalendarFeatureTransformer
from yohou.preprocessing.time_features import FourierFeatureTransformer
from sklearn.ensemble import GradientBoostingRegressor

feature_pipeline = FeatureUnion([
    ("calendar", CalendarFeatureTransformer(features=["day_of_week", "month"])),
    ("fourier", FourierFeatureTransformer(seasonality=365.25, harmonics=[1, 2, 3])),
])

forecaster = PointReductionForecaster(
    estimator=GradientBoostingRegressor(),
    actual_transformer=feature_pipeline,
    target_as_feature="transformed",  # uses the transformed target as an additional feature
)
```

If you also need holiday indicators, add a
[`HolidayFeatureTransformer`](/pages/api/generated/yohou.preprocessing.HolidayFeatureTransformer/)
step to the pipeline.

## Use Calendar Profiles for Local-Time and Holiday Patterns

Row-count seasonality assumes every cycle has the same number of rows at the
same clock positions. That fails when the pattern follows local time and the
data is stored in UTC: an evening peak at 18:00 in Paris falls at 17:00 UTC in
winter and 16:00 UTC in summer, so a 24-row profile learned in winter forecasts
the summer peak one hour late, every day, for about seven months a year. It
also fails for calendar periods with no fixed row count, such as months, and
for holidays, which behave like Sundays but fall on weekdays.

[`SegmentProfileForecaster`](/pages/api/generated/yohou.point.SegmentProfileForecaster/)
labels each future timestamp with a segmenter and forecasts it from the values
that carried the same label. With a
[`CalendarSegmenter`](/pages/api/generated/yohou.segmenting.CalendarSegmenter/)
that reads the hour in local time, the profile stays aligned across
daylight-saving changes:

```python
from yohou.point import SegmentProfileForecaster
from yohou.segmenting import CalendarSegmenter

forecaster = SegmentProfileForecaster(CalendarSegmenter("hour", time_zone="Europe/Paris"))
forecaster.fit(y_train, forecasting_horizon=48)  # y_train["time"] is timezone-aware, e.g. UTC
```

Set `n_recent=1` for a calendar-aware seasonal naive forecast (the last value
with the same label), and `window_size` to cap how many recent rows the
profiles use.

For holidays, combine segments with a
[`UnionSegmenter`](/pages/api/generated/yohou.segmenting.UnionSegmenter/) and a
[`HolidaySegmenter`](/pages/api/generated/yohou.segmenting.HolidaySegmenter/).
Include an `hour` component that every row matches: without it, ordinary
weekdays match no component and get the overall average instead of their own
profile.

```python
from datetime import date

import polars as pl

from yohou.segmenting import HolidaySegmenter, UnionSegmenter

holidays = pl.DataFrame({"date": [date(2024, 12, 25), date(2025, 1, 1)]})
segmenter = UnionSegmenter([
    ("hour", CalendarSegmenter("hour", time_zone="Europe/Paris")),
    ("weekend", CalendarSegmenter("day_of_week", values=[6, 7], time_zone="Europe/Paris")),
    ("holiday", HolidaySegmenter(holidays=holidays, time_zone="Europe/Paris")),
])
forecaster = SegmentProfileForecaster(segmenter)
```

Each weekday holiday at a given hour then shares one profile, separate from
ordinary weekdays and from weekends. Holidays are rare, so their profile rests
on few values. A label with no data in memory, such as a holiday hour never
seen before, is forecast with the overall average of the column.

To remove a trend first, put the forecaster in a
[`DecompositionPipeline`](/pages/api/generated/yohou.compose.DecompositionPipeline/)
after a trend forecaster.

## See Also

- [Stationarity](../explanation/stationarity.md) for the conceptual background
  on seasonal estimation
- [How to Apply Stationarity Transforms](apply-stationarity-transforms.md) for
  single-period decomposition
- [`SegmentProfileForecaster`](/pages/api/generated/yohou.point.SegmentProfileForecaster/)
  and [`HolidaySegmenter`](/pages/api/generated/yohou.segmenting.HolidaySegmenter/)
  API reference for calendar profiles
