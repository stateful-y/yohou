# How to Produce Prediction Intervals

This guide shows you how to wrap a point forecaster with
[`SplitConformalForecaster`](/pages/api/generated/yohou.interval.SplitConformalForecaster/)
to produce calibrated prediction intervals and evaluate their coverage.
Use this when you need uncertainty bounds around your forecasts, for
example to size safety stock or flag anomalous observations.

## Prerequisites

- yohou installed ([Installation](installation.md))
- A fitted or unfitted point forecaster ([Getting Started](../tutorials/getting-started.md))

<!-- COMPANION_NOTEBOOKS -->

## 1. Wrap a Point Forecaster

Pass any point forecaster to [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.SplitConformalForecaster/).
It applies conformal prediction: it holds out a calibration set from the
training data and uses conformity scores measured on that set to size the
intervals:

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
# columns include: nsw__demand, nsw__demand_lower_0.9, nsw__demand_upper_0.9, ...
```

The bound naming pattern is `{component}_lower_{rate}` and
`{component}_upper_{rate}`. For multiple components or coverage rates,
one pair of columns is produced per combination.

`SplitConformalForecaster` also returns a bare `{component}` column holding the
point forecast its bands are centred on. Read that column when you want the
model's own estimate. Do not re-derive it by averaging the bounds: that only
recovers the point forecast when the conformity scorer is symmetric, and returns
a different number for a signed scorer such as `Residual`.

An interval forecaster with no point forecaster inside it, such as
`IntervalReductionForecaster`, has no such value to report and returns bound
columns only.

### A note on `recursion_strategy`

`predict_interval` accepts a `recursion_strategy` argument that selects how a
recursive step derives its next observation from the previous step's bounds. It
applies to forecasters that recurse at the interval level.

`SplitConformalForecaster` does not: the wrapped point forecaster produces the
whole horizon in one call, and any recursion inside it runs on point values
rather than bound midpoints. `recursion_strategy` therefore defaults to
`"point"` on this class, which is what it has always done, and `"mean"` or
`"median"` raise rather than being silently ignored.

## 3. Score Coverage and Sharpness

Use [`EmpiricalCoverage`](/pages/api/generated/yohou.metrics.EmpiricalCoverage/) to check whether the intervals contain the true
values at the claimed rate, and [`IntervalScore`](/pages/api/generated/yohou.metrics.IntervalScore/) to penalize both
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

## 4. Forecast Intervals Directly with Reduction

When your estimator already produces quantiles or interval bounds (for
example a quantile-capable sklearn regressor or a CatBoost MultiQuantile
model), you can skip the conformal wrapper and use
[`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.IntervalReductionForecaster/)
to fit interval forecasts directly:

```python
from sklearn.linear_model import QuantileRegressor
from yohou.interval import IntervalReductionForecaster

interval_forecaster = IntervalReductionForecaster(
    estimator=QuantileRegressor(solver="highs"),
)
interval_forecaster.fit(y_train, forecasting_horizon=24, coverage_rates=[0.90])
y_pred = interval_forecaster.predict_interval()
```

The framework fits one quantile model per requested bound and assembles the
interval columns using the same `{component}_lower_{rate}` /
`{component}_upper_{rate}` naming as the conformal path. Use this route when
the estimator's own quantile loss is preferable to post-hoc calibration.

## 5. Adapt Coverage Online Under Drift

A static calibration set fixes the interval width once. When the series
drifts, realized coverage drifts with it and the intervals become too wide or
too narrow. Pass an
[`AdaptiveConformalInference`](/pages/api/generated/yohou.interval.AdaptiveConformalInference/)
adapter to nudge the effective miscoverage level up or down from the coverage
actually observed, restoring the target rate as you walk forward:

```python
from yohou.interval import AdaptiveConformalInference, SplitConformalForecaster
from yohou.point import SeasonalNaive

forecaster = SplitConformalForecaster(
    point_forecaster=SeasonalNaive(seasonality=7),
    calibration_size=100,
    adapter=AdaptiveConformalInference(step_size=0.05),
)
forecaster.fit(y_train, forecasting_horizon=3, coverage_rates=[0.9])

# Walk forward: each observe() updates the level, each predict reflects it.
y_pred = forecaster.observe_predict_interval(y_stream, stride=1, coverage_rates=[0.9])
```

The output columns keep their nominal labels (`..._lower_0.9`), so a "90%
interval" stays a 90% interval; only its width is adapted. `step_size` is the
learning rate: larger values react faster but track more noisily. Leave
`adapter=None` (the default) for the static behavior. The adapter composes
with `similarity`: the similarity sets which residuals count, the adapter sets
how far into their tail to reach. A coverage rate the adapter never tracked
falls back to the static level with a warning.

## See Also

- [About Interval Forecasting](../explanation/interval-forecasting.md): conformal theory, coverage guarantees, and when to prefer quantile regression over conformal wrapping
- [Combine Forecasters with Ensembles](ensemble-forecasting.md): average or envelope bounds from multiple interval forecasters with [`VotingIntervalForecaster`](/pages/api/generated/yohou.ensemble.VotingIntervalForecaster/)
- [Evaluate Forecast Accuracy](evaluate-forecast-accuracy.md): point and interval metric overview
- [Visualize and Compare Model Scores](visualize-scores.md): plot coverage and interval width over time
- [`yohou.interval` API reference](/pages/api/interval/)
