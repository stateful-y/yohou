# How to Choose a Forecasting Method

This guide walks through a practical process for selecting a forecasting method in yohou, from simple baselines to tuned models.

## Start with a Naive Baseline

Always begin with [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/). Set the seasonality to match the dominant cycle in your data (7 for daily data with weekly pattern, 12 for monthly data with yearly pattern, 1 for non-seasonal data):

```python
from yohou.point import SeasonalNaive

baseline = SeasonalNaive(seasonality=7)
baseline.fit(y_train, forecasting_horizon=14)
y_pred_baseline = baseline.predict(forecasting_horizon=14)
```

Evaluate with cross-validation to establish the score to beat. See [Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md) for details.

## Try a Linear Reduction Forecaster

If the series has learnable structure beyond seasonal repetition, a [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) with a linear model and simple lags often provides a strong second step:

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster

forecaster = PointReductionForecaster(estimator=Ridge())
forecaster.fit(y_train, X_actual=X_train, forecasting_horizon=14)
```

Compare against the baseline. If the linear model does not improve on `SeasonalNaive`, check whether the data has strong nonlinear patterns or whether the exogenous features carry useful signal.

## Add Stationarity Transformers

Non-stationary series (those with trend or changing variance) benefit from target transformers that stabilize the data before the regressor sees it:

```python
from yohou.stationarity import SeasonalDifferencing, LogTransformer

# Remove weekly seasonality before fitting
forecaster = PointReductionForecaster(
    estimator=Ridge(),
    target_transformer=SeasonalDifferencing(seasonality=7),
)

# Or stabilize multiplicative variance
forecaster = PointReductionForecaster(
    estimator=Ridge(),
    target_transformer=LogTransformer(),
)
```

The forecaster automatically applies `inverse_transform` when predicting. See [Stationarity](/pages/explanation/stationarity/) for guidance on choosing the right transform.

## Enrich Features

Feature transformers add derived signals that help the regressor capture temporal patterns:

```python
from yohou.preprocessing import LagTransformer, RollingStatisticsTransformer
from yohou.compose import FeatureUnion

feature_transformer = FeatureUnion(
    transformer_list=[
        ("lags", LagTransformer(lag=[1, 7, 14])),
        ("rolling_7", RollingStatisticsTransformer(window_size=7)),
        ("rolling_14", RollingStatisticsTransformer(window_size=14)),
    ]
)

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    feature_transformer=feature_transformer,
)
```

Add features one group at a time and check whether cross-validation scores improve. More features is not always better, especially with small training sets.

## Try Nonlinear Regressors

When the linear model plateaus, switch to a nonlinear estimator. Gradient boosting and random forests are common choices:

```python
from sklearn.ensemble import GradientBoostingRegressor

forecaster = PointReductionForecaster(
    estimator=GradientBoostingRegressor(n_estimators=200),
    target_transformer=SeasonalDifferencing(period=7),
    feature_transformer=feature_transformer,
)
```

Nonlinear regressors benefit more from rich feature sets. They also have more hyperparameters, so use [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/) rather than [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) to explore the space efficiently.

## Compare Reduction Strategies

The default multi-output strategy trains one model for all horizon steps. For longer horizons, consider direct or dir-rec strategies:

```python
# Direct: separate model per horizon step
forecaster = PointReductionForecaster(
    estimator=Ridge(),
    reduction_strategy="direct",
    n_jobs=-1,  # parallelize across horizon steps
)

# Dir-rec: each model sees predictions from earlier steps
forecaster = PointReductionForecaster(
    estimator=Ridge(),
    reduction_strategy="dir-rec",
)
```

Multi-output is fastest and a good default. Direct avoids error accumulation. Dir-rec combines per-step specialization with inter-step information flow.

## Use Decomposition for Complex Patterns

When the series has multiple seasonal components or a strong trend, a [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) separates the problem into simpler parts:

```python
from yohou.compose import DecompositionPipeline
from yohou.stationarity import PolynomialTrendForecaster, FourierSeasonalityForecaster

forecaster = DecompositionPipeline(
    forecasters=[
        ("trend", PolynomialTrendForecaster(degree=1)),
        ("seasonality", FourierSeasonalityForecaster(
            seasonality=365, harmonics=[1, 2, 3, 4, 5],
        )),
        ("residual", PointReductionForecaster(estimator=Ridge())),
    ]
)
```

## Add Prediction Intervals

Once you have a satisfactory point forecaster, wrap it with [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/) to quantify uncertainty:

```python
from yohou.interval import SplitConformalForecaster

interval_forecaster = SplitConformalForecaster(
    point_forecaster=forecaster,
    calibration_size=100,
)
interval_forecaster.fit(y_train, X_actual=X_train, forecasting_horizon=14, coverage_rates=[0.9])
intervals = interval_forecaster.predict_interval(coverage_rates=[0.9])
```

See [Interval Forecasting](/pages/explanation/interval-forecasting/) for details on conformal prediction and quantile regression approaches.

## Decision Summary

| Scenario | Recommended starting point |
|---|---|
| Quick benchmark, any series | [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/) |
| Few features, short horizon | [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) + `Ridge` |
| Strong trend or seasonality | [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) |
| Many features, nonlinear patterns | [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) + `GradientBoostingRegressor` |
| Multiple related series | Panel forecasting with `panel_strategy="global"` (default) |
| Uncertainty quantification needed | [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/) wrapping the chosen point forecaster |
| Only probabilistic forecasts needed | [`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/) with a quantile or interval regressor directly |
| Categorical targets | [`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.reduction.ClassProbaReductionForecaster/) with an sklearn classifier |
| Combining multiple forecasters | [`VotingPointForecaster`](/pages/api/generated/yohou.ensemble.voting_point.VotingPointForecaster/) or [`VotingIntervalForecaster`](/pages/api/generated/yohou.ensemble.voting_interval.VotingIntervalForecaster/) |
