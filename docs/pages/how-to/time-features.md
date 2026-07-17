# How to Add Calendar and Time Features

This guide shows you how to derive calendar, holiday, Fourier, and trend features from the time column and pass them to a forecaster as deterministic inputs.

## Prerequisites

- yohou installed ([Installation](installation.md))
- Familiarity with transformers and `actual_transformer` ([How to Use Preprocessing Transformers](use-preprocessing-transformers.md))
- Background on time features ([Preprocessing](../explanation/preprocessing.md#time-features))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Extract Calendar Features

[`CalendarFeatureTransformer`](/pages/api/generated/yohou.preprocessing.calendar.CalendarFeatureTransformer/) extracts integer features from timestamps. Pass a list of feature names, or leave `features=None` to auto-detect all features applicable to your data's frequency:

```python
from yohou.preprocessing import CalendarFeatureTransformer
from yohou.datasets import fetch_electricity_demand

data = fetch_electricity_demand()
y = data.frame

transformer = CalendarFeatureTransformer(features=["month", "day_of_week", "hour"])
X = transformer.fit_transform(y)
# Output columns: cal_month, cal_day_of_week, cal_hour
```

Available features include `"year"`, `"month"`, `"quarter"`, `"week"`, `"day_of_week"`, `"day_of_month"`, `"day_of_year"`, `"hour"`, `"minute"`, and boolean indicators like `"is_weekend"`, `"is_month_start"`, `"is_month_end"`, `"is_quarter_start"`, `"is_quarter_end"`, `"is_year_start"`, `"is_year_end"`.

!!! info "Frequency aware"
    Sub-daily features (`"hour"`, `"minute"`) are only applicable to sub-daily data. When `features=None`, the transformer inspects the data interval and extracts only what makes sense. After fitting, check `transformer.applicable_features_` to see which features were selected.

## Mark Holidays

[`HolidayFeatureTransformer`](/pages/api/generated/yohou.preprocessing.calendar.HolidayFeatureTransformer/) creates a binary indicator column and optional proximity features showing distance to the nearest holiday:

```python
from yohou.preprocessing import HolidayFeatureTransformer
import polars as pl
from datetime import date

holidays = pl.DataFrame({
    "date": [date(2024, 1, 1), date(2024, 7, 4), date(2024, 12, 25)]
})

transformer = HolidayFeatureTransformer(
    holidays=holidays,
    days_to_next=True,
    days_since_last=True,
)
X = transformer.fit_transform(y)
# Output columns: holiday_indicator, holiday_days_to_next, holiday_days_since_last
```

The `holidays` DataFrame needs a `"date"` column of Date or Datetime type. Set `days_to_next=True` or `days_since_last=True` to add integer columns measuring the distance (in days) to surrounding holidays.

## Add Fourier Features

[`FourierFeatureTransformer`](/pages/api/generated/yohou.preprocessing.time_features.FourierFeatureTransformer/) captures smooth seasonal patterns with sine/cosine pairs at specified harmonics:

```python
from yohou.preprocessing import FourierFeatureTransformer

transformer = FourierFeatureTransformer(seasonality=24, harmonics=[1, 2, 3])
X = transformer.fit_transform(y)
# Output columns: fourier_24_sin_1, fourier_24_cos_1, fourier_24_sin_2, ...
```

Set `seasonality` to the period length in time steps (e.g., `24` for daily cycles in hourly data, `365.25` for yearly cycles in daily data). Use more harmonics for sharper seasonal shapes, fewer for smoother patterns. Harmonics must satisfy the Nyquist limit: each harmonic $k$ must be at most $S/2$ where $S$ is the seasonality.

## Create a Trend Index

[`TimeIndexTransformer`](/pages/api/generated/yohou.preprocessing.time_features.TimeIndexTransformer/) converts timestamps to a numeric index, useful for capturing linear or polynomial trends:

```python
from yohou.preprocessing import TimeIndexTransformer

transformer = TimeIndexTransformer(normalize=True, degree=2)
X = transformer.fit_transform(y)
# Output columns: time_index, time_index_2
```

With `degree=1` (the default), the transformer produces a single `time_index` column. Higher degrees add polynomial terms (`time_index_2`, `time_index_3`, etc.). Set `normalize=True` to scale the index to `[0, 1]` based on the training data length, which improves numerical stability for polynomial features.

## Combine Multiple Time Features

Use [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/) to run multiple time feature transformers in parallel and concatenate their outputs, then pass the result as `actual_transformer` to any forecaster:

```python
from yohou.compose import FeatureUnion
from yohou.preprocessing import (
    CalendarFeatureTransformer,
    FourierFeatureTransformer,
    TimeIndexTransformer,
)
from yohou.point import PointReductionForecaster
from sklearn.linear_model import Ridge

features = FeatureUnion([
    ("calendar", CalendarFeatureTransformer(features=["month", "day_of_week", "hour"])),
    ("fourier", FourierFeatureTransformer(seasonality=24, harmonics=[1, 2, 3])),
    ("trend", TimeIndexTransformer(normalize=True)),
])

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    actual_transformer=features,
)
forecaster.fit(y, forecasting_horizon=24)
predictions = forecaster.predict()
```

For more complex compositions (sequential chaining, nesting), see [How to Compose Feature Pipelines](compose-feature-pipelines.md).

!!! info "actual_transformer vs X_future"
    When time features are passed via `actual_transformer`, the forecaster generates them automatically at both fit and predict time. If you generate calendar or Fourier features externally (outside `actual_transformer`), pass them via `X_future` at both `fit()` and `predict()` time, since they are deterministic and known in advance. See [Exogenous Features](exogenous-features.md) for details.

## See Also

- [Preprocessing](../explanation/preprocessing.md#time-features): full details on each transformer
- [How to Compose Feature Pipelines](compose-feature-pipelines.md): sequential and parallel transformer composition
- [Exogenous Features](exogenous-features.md): when and how to use external features
- [API Reference: yohou.preprocessing](../api/preprocessing.md)
