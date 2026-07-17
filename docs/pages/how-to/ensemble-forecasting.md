# How to Combine Forecasters with Ensembles

This guide shows you how to combine multiple forecasters into a voting ensemble
for more stable predictions.

## Prerequisites

- yohou installed ([Installation](installation.md))
- Familiarity with fitting and predicting with point or interval forecasters ([Getting Started](../tutorials/getting-started.md))

<!-- COMPANION_NOTEBOOKS -->

## 1. Create a Point Ensemble

Pass named `(name, forecaster)` tuples to [`VotingPointForecaster`](/pages/api/generated/yohou.ensemble.VotingPointForecaster/):

```python
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from yohou.ensemble import VotingPointForecaster
from yohou.point import PointReductionForecaster
from yohou.datasets import fetch_electricity_demand

data = fetch_electricity_demand()
y = data.frame

ridge = PointReductionForecaster(estimator=Ridge())
rf = PointReductionForecaster(estimator=RandomForestRegressor(n_estimators=50))

ensemble = VotingPointForecaster(
    forecasters=[("ridge", ridge), ("rf", rf)],
    method="mean",
)
ensemble.fit(y, forecasting_horizon=24)
y_pred = ensemble.predict()
```

To favor one model over another, pass `weights`:

```python
weighted = VotingPointForecaster(
    forecasters=[("ridge", ridge), ("rf", rf)],
    method="mean",
    weights=[0.3, 0.7],  # favor random forest
)
```

Set `method="median"` for robustness against outlier predictions (weights
are ignored with median aggregation).

## 2. Ensemble Interval Forecasters

[`VotingIntervalForecaster`](/pages/api/generated/yohou.ensemble.VotingIntervalForecaster/) combines prediction intervals from
multiple interval forecasters such as [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.SplitConformalForecaster/):

```python
from yohou.ensemble import VotingIntervalForecaster
from yohou.interval import SplitConformalForecaster

interval_ensemble = VotingIntervalForecaster(
    forecasters=[
        ("conf_ridge", SplitConformalForecaster(
            point_forecaster=PointReductionForecaster(estimator=Ridge()),
        )),
        ("conf_rf", SplitConformalForecaster(
            point_forecaster=PointReductionForecaster(estimator=RandomForestRegressor()),
        )),
    ],
    method="envelope",  # most conservative: min of lowers, max of uppers
)
interval_ensemble.fit(y, forecasting_horizon=24, coverage_rates=[0.9])
y_interval = interval_ensemble.predict_interval()
```

Available `method` values: `"envelope"` (default, most conservative), `"mean"`, `"median"`.

## 3. Ensemble Classification Forecasters

[`VotingClassProbaForecaster`](/pages/api/generated/yohou.ensemble.VotingClassProbaForecaster/) combines class probability predictions:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from yohou.ensemble import VotingClassProbaForecaster
from yohou.class_proba import ClassProbaReductionForecaster
from yohou.datasets import fetch_air_quality_classification

class_data = fetch_air_quality_classification()
y_class = class_data.y

class_ensemble = VotingClassProbaForecaster(
    forecasters=[
        ("lr", ClassProbaReductionForecaster(estimator=LogisticRegression())),
        ("rf", ClassProbaReductionForecaster(estimator=RandomForestClassifier())),
    ],
    method="soft",  # weighted average of probabilities
)
class_ensemble.fit(y_class, forecasting_horizon=24)
y_proba = class_ensemble.predict_class_proba()
```

Use `method="hard"` for majority voting (argmax per base model, then mode).

## 4. Speed Up with Parallel Fitting

All voting forecasters accept `n_jobs` to fit base models in parallel:

```python
ensemble = VotingPointForecaster(
    forecasters=[("ridge", ridge), ("rf", rf)],
    n_jobs=-1,  # use all available cores
)
```

## 5. Use with Panel Data

All voting forecasters support panel data automatically. Pass a DataFrame
with `__` separated panel columns, and each base forecaster receives the full
panel. Aggregation happens per group:

```python
from yohou.datasets import fetch_tourism_monthly

bunch = fetch_tourism_monthly()
y_panel = bunch.frame.select(
    ["time", "T187__tourists", "T188__tourists", "T189__tourists"]
).drop_nulls()

ridge = PointReductionForecaster(estimator=Ridge())
rf = PointReductionForecaster(estimator=RandomForestRegressor(n_estimators=50))

panel_ensemble = VotingPointForecaster(
    forecasters=[("ridge", ridge), ("rf", rf)],
)
panel_ensemble.fit(y_panel, forecasting_horizon=12)
y_pred_panel = panel_ensemble.predict()
```

The output contains one column per group, each with the ensemble's aggregated
prediction.

See [Working with Panel Data](panel-data.md) for panel data preparation
and forecasting.

## 6. Evaluate with Rolling Predictions

Voting forecasters inherit the stateful lifecycle, so you can evaluate an
ensemble walk-forward with `observe_predict`. It steps through the test set in
batches of `stride`, emitting one vintage per batch without refitting:

```python
from yohou.model_selection import train_test_split

y_train, y_test = train_test_split(y, test_size=48)

ensemble.fit(y_train, forecasting_horizon=24)
y_pred = ensemble.observe_predict(y=y_test, stride=1, forecasting_horizon=24)
```

The resulting multi-vintage DataFrame feeds directly into per-step and
per-vintage scoring. See [How to Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md)
for the scoring workflow.

## See Also

- [Ensemble Forecasting](../explanation/ensemble-forecasting.md): theory and aggregation formulas
- [How to Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md)
- [API Reference: yohou.ensemble](../api/ensemble.md)
