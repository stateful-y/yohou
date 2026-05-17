# How to Apply Stationarity Transforms

This guide shows you how to remove trends and seasonal patterns from time series data before modeling. Use this when your forecaster assumes stationary input or when you want to decompose a series into interpretable components.

## Prerequisites

- Familiarity with the fit/predict API ([Getting Started](../tutorials/getting-started.md))
- Understanding of stationarity concepts ([Stationarity](../explanation/stationarity.md))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Choose a Decomposition Strategy

| Situation | Recommended approach |
|---|---|
| Simple seasonal pattern, fixed period | [`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/) as `target_transformer` |
| Trend + seasonality, or component inspection | [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) with `store_residuals=True` |
| Multiple seasonal periods or non-integer periods | [`FourierSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.FourierSeasonalityForecaster/) with multiple harmonics |
| Multiplicative seasonality | `DecompositionPipeline` with [`LogTransformer`](/pages/api/generated/yohou.stationarity.transformers.LogTransformer/) as `target_transformer` |

## Remove a Seasonal Pattern with Differencing

If the series has a single, fixed-period seasonal pattern, use [`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/) as a `target_transformer`. It subtracts each value from its value one full cycle ago and automatically inverts the transform during prediction:

```python
from yohou.point import PointReductionForecaster
from yohou.stationarity import SeasonalDifferencing
from sklearn.linear_model import Ridge

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    target_transformer=SeasonalDifferencing(seasonality=12),
)
forecaster.fit(y_train, forecasting_horizon=forecast_horizon)
y_pred = forecaster.predict()
```

Set `seasonality` to the number of observations per period (12 for monthly data with yearly seasonality, 7 for daily data with weekly seasonality).

## Decompose Trend and Seasonality

If the series has both a trend and a seasonal component, use a [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) to model each explicitly. Each forecaster fits the residuals left by the previous ones:

```python
from yohou.compose import DecompositionPipeline
from yohou.stationarity import PatternSeasonalityForecaster, PolynomialTrendForecaster

pipeline = DecompositionPipeline(
    forecasters=[
        ("trend", PolynomialTrendForecaster(degree=1)),
        ("seasonality", PatternSeasonalityForecaster(seasonality=12)),
    ],
    store_residuals=True,
)
pipeline.fit(y_train, forecasting_horizon=forecast_horizon)
y_pred = pipeline.predict()
```

With `store_residuals=True`, inspect the residuals after each component via `pipeline.residuals_`, a dictionary keyed by forecaster name.

## Handle Multiplicative Seasonality

If the seasonal amplitude grows proportionally with the level of the series, apply a [`LogTransformer`](/pages/api/generated/yohou.stationarity.transformers.LogTransformer/) to convert multiplicative patterns into additive ones:

```python
from yohou.compose import DecompositionPipeline
from yohou.stationarity import (
    LogTransformer,
    PatternSeasonalityForecaster,
    PolynomialTrendForecaster,
)

pipeline = DecompositionPipeline(
    forecasters=[
        ("trend", PolynomialTrendForecaster(degree=1)),
        ("seasonality", PatternSeasonalityForecaster(seasonality=12)),
    ],
    target_transformer=LogTransformer(),
)
```

The log transform is applied before fitting and automatically inverted during prediction.

## Apply to Panel Data

Stationarity transforms work automatically with panel data. Each group receives the transform independently:

```python
from yohou.compose import LocalPanelForecaster
from yohou.point import PointReductionForecaster
from yohou.stationarity import SeasonalDifferencing
from sklearn.linear_model import Ridge

forecaster = LocalPanelForecaster(
    forecaster=PointReductionForecaster(
        estimator=Ridge(),
        target_transformer=SeasonalDifferencing(seasonality=12),
    ),
)
```

## See Also

- [Stationarity](../explanation/stationarity.md) for the conceptual background
- [Decomposition](../tutorials/decomposition.md) tutorial for a step-by-step walkthrough
