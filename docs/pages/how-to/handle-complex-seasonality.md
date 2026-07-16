# How to Handle Complex Seasonality

This guide shows you three approaches for handling multiple overlapping seasonal
periods (e.g., weekly + annual in daily data) in yohou.

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
    feature_transformer=feature_pipeline,
    target_as_feature="transformed",  # uses the transformed target as an additional feature
)
```

If you also need holiday indicators, add a
[`HolidayFeatureTransformer`](/pages/api/generated/yohou.preprocessing.HolidayFeatureTransformer/)
step to the pipeline.

## See Also

- [Stationarity](../explanation/stationarity.md) for the conceptual background
  on seasonal estimation
- [How to Apply Stationarity Transforms](apply-stationarity-transforms.md) for
  single-period decomposition
