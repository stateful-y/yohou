# How to Add Calendar and Time Features

This guide shows how to derive exogenous features from the time column using
the time feature transformers.

**Prerequisites**: Familiarity with transformers and the `feature_transformer`
parameter. See [Preprocessing](../explanation/preprocessing.md) for background.

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Calendar Features

Extract integer features from timestamps (month, day of week, hour, etc.):

```python
from yohou.preprocessing import CalendarFeatureTransformer

transformer = CalendarFeatureTransformer(features=["month", "day_of_week", "hour"])
X = transformer.fit_transform(y)
# Adds columns: month (1-12), day_of_week (0-6), hour (0-23)
```

## Holiday Features

Mark public holidays as binary indicators:

```python
from yohou.preprocessing import HolidayFeatureTransformer
import polars as pl
from datetime import date

holidays = pl.DataFrame({"date": [date(2024, 1, 1), date(2024, 7, 4), date(2024, 12, 25)]})
transformer = HolidayFeatureTransformer(holidays=holidays)
X = transformer.fit_transform(y)
# Adds a binary column: 1 on holidays, 0 otherwise
```

## Fourier Features

Capture smooth seasonal patterns with sine/cosine pairs:

```python
from yohou.preprocessing import FourierFeatureTransformer

transformer = FourierFeatureTransformer(seasonality=365.25, harmonics=[1, 2, 3, 4])
X = transformer.fit_transform(y)
# Adds 8 columns: fourier_365.25_sin_1, fourier_365.25_cos_1, ...
```

Use more harmonics for sharper seasonal shapes, fewer for smoother patterns.

## Time Index

Convert timestamps to a normalized numeric index:

```python
from yohou.preprocessing import TimeIndexTransformer

transformer = TimeIndexTransformer()
X = transformer.fit_transform(y)
```

## Combine in a Pipeline

Use [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/) to combine multiple time feature transformers, then pass as
`feature_transformer` to any forecaster:

```python
from yohou.compose import FeatureUnion
from yohou.point import PointReductionForecaster

features = FeatureUnion(
    transformer_list=[
        ("calendar", CalendarFeatureTransformer(features=["month", "day_of_week"])),
        ("fourier", FourierFeatureTransformer(seasonality=365.25, harmonics=[1, 2, 3])),
    ],
)

forecaster = PointReductionForecaster(feature_transformer=features)
forecaster.fit(y, forecasting_horizon=30)
```

### Which parameter to use

When passing time-derived features as exogenous data:

- **Calendar, holidays, Fourier features** use `X_future`: these are deterministic and known in advance for any future date.
- **Lag features** are handled automatically by `feature_transformer` (derived from `X_actual` internally during `fit`).

If you generate calendar features externally (outside `feature_transformer`), pass them via `X_future` at both `fit()` and `predict()` time.

### Related pages

- [Preprocessing](../explanation/preprocessing.md#time-features): full details on each transformer
- [Exogenous Features](../explanation/exogenous-features.md): when and how to use external features
- [API Reference: yohou.preprocessing](../api/preprocessing.md)
