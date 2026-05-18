# How to Produce Prediction Intervals

This guide shows you how to wrap a point forecaster with
[`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/)
to produce calibrated prediction intervals and evaluate their coverage.
Use this when you need uncertainty bounds around your forecasts, for
example to size safety stock or flag anomalous observations.

## Prerequisites

- yohou installed ([Installation](installation.md))
- A fitted or unfitted point forecaster ([Getting Started](../tutorials/getting-started.md))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## 1. Wrap a Point Forecaster

Pass any point forecaster to [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/).
It holds out a calibration set from the training data and uses it to
compute conformity scores that size the intervals:

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster
from yohou.interval import SplitConformalForecaster
from yohou.datasets import fetch_electricity_demand
from yohou.model_selection import train_test_split

data = fetch_electricity_demand()
y = data.frame

y_train, y_test = train_test_split(y, test_size=48)

point_forecaster = PointReductionForecaster(estimator=Ridge())

interval_forecaster = SplitConformalForecaster(
    point_forecaster=point_forecaster,
)
interval_forecaster.fit(y_train, forecasting_horizon=24, coverage_rates=[0.90])
```

`coverage_rates` is set at `fit()` time. Multiple rates can be requested at once
(`coverage_rates=[0.80, 0.90, 0.95]`).

If coverage is poor, increase `calibration_size` (default 100) to give the
conformal layer more residuals to learn from. Larger values improve calibration
but leave less data for fitting the point forecaster.

## 2. Predict Intervals

Call `predict_interval` to get a DataFrame with lower and upper
bound columns for each requested coverage rate:

```python
y_pred = interval_forecaster.predict_interval()
# columns include: nsw__demand_lower_0.9, nsw__demand_upper_0.9, ...
```

The column naming pattern is `{component}_lower_{rate}` and
`{component}_upper_{rate}`. For multiple components or coverage rates,
one pair of columns is produced per combination.

## 3. Score Coverage and Sharpness

Use [`EmpiricalCoverage`](/pages/api/generated/yohou.metrics.interval.EmpiricalCoverage/) to check whether the intervals contain the true
values at the claimed rate, and [`IntervalScore`](/pages/api/generated/yohou.metrics.interval.IntervalScore/) to penalize both
miscoverage and unnecessarily wide intervals:

```python
from yohou.metrics import EmpiricalCoverage, IntervalScore

coverage = EmpiricalCoverage()
coverage.fit(y_train)
print(coverage.score(y_test, y_pred))

sharpness = IntervalScore()
sharpness.fit(y_train)
print(sharpness.score(y_test, y_pred))
```

A well-calibrated forecaster has empirical coverage close to the
nominal rate (e.g., ~0.90 for a 90% interval). `IntervalScore` rewards
narrow intervals and penalizes observations that fall outside the bounds,
so lower is better.

## See Also

- [Combine Forecasters with Ensembles](ensemble-forecasting.md): average or envelope bounds from multiple interval forecasters with [`VotingIntervalForecaster`](/pages/api/generated/yohou.ensemble.voting_interval.VotingIntervalForecaster/)
- [Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md): point and interval metric overview
- [Visualize and Compare Model Scores](visualize-scores.md): plot coverage and interval width over time
- [`yohou.interval` API reference](/pages/api/interval/)
