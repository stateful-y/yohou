# Reduction Strategies

In this tutorial, we will compare the three reduction strategies available in [`PointReductionForecaster`](/pages/api/generated/yohou.point.PointReductionForecaster/): multi-output (the default), direct, and dir-rec. We will fit each strategy on the same dataset, compare per-step error, and see how `target_as_feature` affects the feature matrix.

## Prerequisites

- Completed [Getting Started](getting-started.md)

<!-- COMPANION_NOTEBOOKS -->

## 1. Load and Prepare Data

We resample the daily sunspot series to monthly frequency using [`Downsampler`](/pages/api/generated/yohou.preprocessing.Downsampler/), then split it into train and test sets:

```python
from yohou.datasets import fetch_sunspot
from yohou.model_selection import train_test_split
from yohou.preprocessing import Downsampler

bunch = fetch_sunspot()
y = Downsampler(interval="1mo", aggregation="mean").fit_transform(bunch.frame)

forecasting_horizon = 24
y_train, y_test = train_test_split(y, test_size=forecasting_horizon)
```

## 2. Multi-Output Strategy (Default)

The multi-output strategy trains a single model that predicts all `H` steps at once. This is the fastest approach and works well with sklearn's `MultiOutputRegressor` wrapper. We wrap a [`LagTransformer`](/pages/api/generated/yohou.preprocessing.LagTransformer/) in a [`FeaturePipeline`](/pages/api/generated/yohou.compose.FeaturePipeline/) to build the lag features each strategy shares:

```python
from sklearn.ensemble import RandomForestRegressor
from yohou.compose import FeaturePipeline
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer

fc_multi = PointReductionForecaster(
    estimator=RandomForestRegressor(n_estimators=50, random_state=42),
    feature_transformer=FeaturePipeline([
        ("lags", LagTransformer(lag=list(range(1, 13)))),
    ]),
    reduction_strategy="multi-output",
)
fc_multi.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_multi = fc_multi.predict(forecasting_horizon=forecasting_horizon)
```

## 3. Direct Strategy

The direct strategy trains `H` independent models, one per forecast step. Each model specializes in predicting a specific horizon:

```python
fc_direct = PointReductionForecaster(
    estimator=RandomForestRegressor(n_estimators=50, random_state=42),
    feature_transformer=FeaturePipeline([
        ("lags", LagTransformer(lag=list(range(1, 13)))),
    ]),
    reduction_strategy="direct",
)
fc_direct.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_direct = fc_direct.predict(forecasting_horizon=forecasting_horizon)
```

## 4. Dir-Rec Strategy

The dir-rec (direct-recursive) hybrid trains `H` sequential models. Each model receives the predictions of all previous steps as additional features:

```python
fc_dirrec = PointReductionForecaster(
    estimator=RandomForestRegressor(n_estimators=50, random_state=42),
    feature_transformer=FeaturePipeline([
        ("lags", LagTransformer(lag=list(range(1, 13)))),
    ]),
    reduction_strategy="dir-rec",
)
fc_dirrec.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_dirrec = fc_dirrec.predict(forecasting_horizon=forecasting_horizon)
```

## 5. Compare Per-Step Error

Score each strategy with [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.MeanAbsoluteError/) and look at how error varies across the forecast horizon:

```python
from yohou.metrics import MeanAbsoluteError

mae = MeanAbsoluteError()
mae.fit(y_train)

for name, y_pred in [
    ("Multi-output", y_pred_multi),
    ("Direct", y_pred_direct),
    ("Dir-rec", y_pred_dirrec),
]:
    score = mae.score(y_test, y_pred)
    print(f"{name:15s} MAE={score:.2f}")
```

Expected output:

```text
Multi-output    MAE=22.57
Direct          MAE=24.64
Dir-rec         MAE=3.39
```

The dir-rec MAE is dramatically lower on this single split. In practice, always cross-validate to confirm that this advantage generalises across folds.

Visualize per-step error with [`plot_score_per_step`](/pages/api/generated/yohou.plotting.plot_score_per_step/) to see where each strategy excels:

```python
from yohou.plotting import plot_score_per_step

preds = {
    "Multi-output": fc_multi.observe_predict(y=y_test, stride=1),
    "Direct": fc_direct.observe_predict(y=y_test, stride=1),
    "Dir-rec": fc_dirrec.observe_predict(y=y_test, stride=1),
}

fig = plot_score_per_step(mae, y_test, preds)
fig.show()
```

## 6. Using `target_as_feature`

The `target_as_feature` parameter controls what enters the feature matrix before `feature_transformer` generates lags. This is especially useful with the direct strategy, where lag features on the target series augment the feature matrix alongside exogenous inputs:

```python
fc_direct_taf = PointReductionForecaster(
    estimator=RandomForestRegressor(n_estimators=50, random_state=42),
    feature_transformer=FeaturePipeline([
        ("lags", LagTransformer(lag=list(range(1, 13)))),
    ]),
    reduction_strategy="direct",
    target_as_feature="transformed",
)
fc_direct_taf.fit(y_train, forecasting_horizon=forecasting_horizon)
y_pred_taf = fc_direct_taf.predict(forecasting_horizon=forecasting_horizon)

score_taf = mae.score(y_test, y_pred_taf)
print(f"Direct + target_as_feature MAE={score_taf:.2f}")
```

Expected output:

```text
Direct + target_as_feature MAE=24.64
```

This MAE matches the Direct strategy in section 3 exactly, because `target_as_feature="transformed"` is the default: both runs feed the transformed target into the feature matrix, so the feature columns are identical. The parameter becomes observable when you change it: `target_as_feature="raw"` lags the untransformed target instead (useful when `target_transformer` distorts the lag structure), while `target_as_feature=None` excludes the target entirely and relies solely on exogenous `X_actual` features (so it requires `X_actual` to be passed). The benefit of including the target depends on how much its own recent history helps the regressor beyond the other features already present.

## What You Built

We compared three reduction strategies on the same dataset, visualized per-step error with [`plot_score_per_step`](/pages/api/generated/yohou.plotting.plot_score_per_step/) to understand their tradeoffs, and explored how `target_as_feature` adds lagged target information to the feature matrix.

## Next Steps

- [Reduction Forecasting](../explanation/reduction-forecasting.md) for the conceptual background on reduction strategies
- [Forecasting Workflow](forecasting-workflow.md) for cross-validation and hyperparameter search
- [Forecast with CatBoost](../how-to/forecast-with-catboost.md) for using gradient-boosted trees as the reduction estimator
