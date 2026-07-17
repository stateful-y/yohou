# How to Choose a Forecasting Method

This guide walks you through selecting the right forecasting approach for your
data. Start with the quick reference table to find your scenario, then follow
the relevant section for setup details.

## Prerequisites

- Yohou installed ([Getting Started](../tutorials/getting-started.md))
- Training data prepared ([Getting Started](../tutorials/getting-started.md))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## Quick Reference

| Scenario | Recommended starting point |
|---|---|
| Quick benchmark, any series | [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/) |
| Few features, short horizon | [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) + [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) |
| Strong trend or seasonality | [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) |
| Many features, nonlinear patterns | [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) + [`GradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) |
| Multiple related series | Panel forecasting with `panel_strategy="global"` (default) |
| Uncertainty quantification needed | [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/) wrapping the chosen point forecaster |
| Only probabilistic forecasts needed | [`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/) with a quantile or interval regressor directly |
| Categorical targets | [`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.reduction.ClassProbaReductionForecaster/) with an sklearn classifier |
| Combining multiple forecasters | [`VotingPointForecaster`](/pages/api/generated/yohou.ensemble.voting_point.VotingPointForecaster/) or [`VotingIntervalForecaster`](/pages/api/generated/yohou.ensemble.voting_interval.VotingIntervalForecaster/) |

## 1. Establish a Naive Baseline

Always start here. [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/) gives you a score to beat with zero tuning. Set the seasonality to match the dominant cycle in your data (7 for daily with weekly pattern, 12 for monthly with yearly pattern, 1 for non-seasonal):

```python
from yohou.point import SeasonalNaive

baseline = SeasonalNaive(seasonality=7)
baseline.fit(y_train, forecasting_horizon=14)
y_pred_baseline = baseline.predict()
```

Evaluate with cross-validation to establish the score to beat. See [Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md) for details.

## 2. Try a Linear Reduction Model

If the baseline is not accurate enough, fit a [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) with a linear estimator:

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster

forecaster = PointReductionForecaster(estimator=Ridge())
forecaster.fit(y_train, X_actual=X_train, forecasting_horizon=14)
```

If this does not improve on the baseline, the data likely has nonlinear patterns or the exogenous features do not carry useful signal. Skip to [step 5](#5-switch-to-gradient-boosting).

## 3. Handle Non-stationarity with Target Transformers

If the series has trend or changing variance, add a target transformer such as [`SeasonalDifferencing`](/pages/api/generated/yohou.stationarity.transformers.SeasonalDifferencing/) or [`LogTransformer`](/pages/api/generated/yohou.stationarity.transformers.LogTransformer/):

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

See [Apply Stationarity Transforms](apply-stationarity-transforms.md) for the full procedure and [Stationarity](../explanation/stationarity.md) for guidance on choosing the right transform.

## 4. Enrich the Feature Set

Add feature transformers one group at a time. Combine [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/), [`RollingStatisticsTransformer`](/pages/api/generated/yohou.preprocessing.window.RollingStatisticsTransformer/), and other transformers using [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/). Check whether cross-validation scores improve after each addition:

```python
from yohou.preprocessing import LagTransformer, RollingStatisticsTransformer
from yohou.compose import FeatureUnion

actual_transformer = FeatureUnion(
    transformer_list=[
        ("lags", LagTransformer(lag=[1, 7, 14])),
        ("rolling_7", RollingStatisticsTransformer(window_size=7)),
        ("rolling_14", RollingStatisticsTransformer(window_size=14)),
    ]
)

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    actual_transformer=actual_transformer,
)
```

See [Build Reduction Forecasters](build-reduction-forecasters.md) for the full walkthrough of feature and target transformer composition.

## 5. Switch to Gradient Boosting

If the linear model plateaus, swap in a nonlinear estimator:

```python
from sklearn.ensemble import GradientBoostingRegressor

forecaster = PointReductionForecaster(
    estimator=GradientBoostingRegressor(n_estimators=200),
    target_transformer=SeasonalDifferencing(seasonality=7),
    actual_transformer=actual_transformer,
)
```

Use [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/) to explore the hyperparameter space efficiently. See [Tune Hyperparameters](tune-hyperparameters.md) for the full procedure.

## 6. Choose the Reduction Strategy

The default multi-output reduction strategy works well for short forecasting horizons. For longer horizons, switch to `"direct"` or `"dir-rec"`:

```python
# One model per horizon step (no error propagation)
forecaster = PointReductionForecaster(
    estimator=Ridge(),
    reduction_strategy="direct",
    n_jobs=-1,
)

# Direct with recursive features (balanced)
forecaster = PointReductionForecaster(
    estimator=Ridge(),
    reduction_strategy="dir-rec",
)
```

See [Reduction Forecasting](../explanation/reduction-forecasting.md) for background on how each strategy handles error propagation.

## 7. Use Decomposition for Complex Seasonality

If the series has multiple seasonal components or strong trend, a [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) separates the problem into simpler parts. Use [`PolynomialTrendForecaster`](/pages/api/generated/yohou.stationarity.trend.PolynomialTrendForecaster/) for trend and [`FourierSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.FourierSeasonalityForecaster/) for cyclical patterns:

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

## 8. Add Prediction Intervals

Once you have a satisfactory point forecaster, wrap it with [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/) to apply conformal prediction and quantify uncertainty:

```python
from yohou.interval import SplitConformalForecaster

interval_forecaster = SplitConformalForecaster(
    point_forecaster=forecaster,
    calibration_size=100,
)
interval_forecaster.fit(y_train, X_actual=X_train, forecasting_horizon=14, coverage_rates=[0.9])
intervals = interval_forecaster.predict_interval(coverage_rates=[0.9])
```

If you need quantile or interval regression directly (without a point forecaster), use [`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/) instead. See [Produce Prediction Intervals](interval-forecasting.md) for the full procedure.

## See Also

- [Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md): cross-validation setup and metric selection
- [Tune Hyperparameters](tune-hyperparameters.md): grid and randomized search over forecaster parameters
- [Ensemble Forecasting](ensemble-forecasting.md): combining multiple forecasters with voting
- [Reduction Forecasting](../explanation/reduction-forecasting.md): conceptual background on reduction strategies and model architecture
- [Stationarity](../explanation/stationarity.md): choosing the right target transformer
