# How to Combine Forecasters with Ensembles

This guide shows how to combine multiple forecasters using voting ensembles to
reduce prediction variance. Use ensembles when you have two or more forecasters
that perform reasonably well individually and you want more stable predictions.

**Prerequisites**: Familiarity with fitting and predicting with point or interval
forecasters. See [Your First Forecast](../tutorials/first-forecast.md) if needed.

## 1. Create a Point Ensemble

Pass named `(name, forecaster)` tuples to [`VotingPointForecaster`](/pages/api/generated/yohou.ensemble.voting_point.VotingPointForecaster/):

```python
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from yohou.ensemble import VotingPointForecaster
from yohou.point import PointReductionForecaster
from yohou.datasets import fetch_electricity_demand

data = fetch_electricity_demand()
y = data.frame

ensemble = VotingPointForecaster(
    forecasters=[
        ("ridge", PointReductionForecaster(estimator=Ridge())),
        ("rf", PointReductionForecaster(estimator=RandomForestRegressor(n_estimators=50))),
    ],
    method="mean",
)
ensemble.fit(y, forecasting_horizon=24)
y_pred = ensemble.predict()
```

## 2. Use Weighted Averaging

Assign higher weight to models you expect to perform better:

```python
ensemble = VotingPointForecaster(
    forecasters=[
        ("ridge", PointReductionForecaster(estimator=Ridge())),
        ("rf", PointReductionForecaster(estimator=RandomForestRegressor())),
    ],
    method="mean",
    weights=[0.3, 0.7],  # favor random forest
)
```

Set `method="median"` instead for robustness against outlier predictions (weights
are ignored with median aggregation).

## 3. Ensemble Interval Forecasters

[`VotingIntervalForecaster`](/pages/api/generated/yohou.ensemble.voting_interval.VotingIntervalForecaster/) combines prediction intervals:

```python
from yohou.ensemble import VotingIntervalForecaster
from yohou.interval import SplitConformalForecaster
from yohou.point import PointReductionForecaster

ensemble = VotingIntervalForecaster(
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
ensemble.fit(y, forecasting_horizon=24, coverage_rates=[0.9])
y_interval = ensemble.predict_interval()
```

Available methods: `"envelope"` (default, most conservative), `"mean"`, `"median"`.

## 4. Ensemble Classification Forecasters

[`VotingClassProbaForecaster`](/pages/api/generated/yohou.ensemble.voting_class_proba.VotingClassProbaForecaster/) combines class-probability predictions:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from yohou.ensemble import VotingClassProbaForecaster
from yohou.class_proba import ClassProbaReductionForecaster
from yohou.datasets import fetch_air_quality_classification

data = fetch_air_quality_classification()
y = data.y

ensemble = VotingClassProbaForecaster(
    forecasters=[
        ("lr", ClassProbaReductionForecaster(estimator=LogisticRegression())),
        ("rf", ClassProbaReductionForecaster(estimator=RandomForestClassifier())),
    ],
    method="soft",  # weighted average of probabilities
)
ensemble.fit(y, forecasting_horizon=24)
y_proba = ensemble.predict_class_proba()
```

Use `method="hard"` for majority voting (argmax of each base model, then mode).
Soft voting generally performs better because it preserves probability information.

## 5. Evaluate Ensemble vs. Individual Models

Compare the ensemble against its members:

```python
from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import ExpandingWindowSplitter

scorer = MeanAbsoluteError()
splitter = ExpandingWindowSplitter(n_splits=3)

# Evaluate each model and the ensemble
for name, model in [("ridge", ridge), ("rf", rf), ("ensemble", ensemble)]:
    scores = []
    for train_idx, test_idx in splitter.split(y):
        model.fit(y[train_idx], forecasting_horizon=len(test_idx))
        y_pred = model.predict()
        scores.append(scorer.fit(y[test_idx]).score(y[test_idx], y_pred))
```

### Related pages

- [Ensemble Forecasting](../explanation/ensemble-forecasting.md): theory and aggregation formulas
- [API Reference: yohou.ensemble](../api/ensemble.md)
- [Ensemble Examples](../examples/forecasting-models.md)
