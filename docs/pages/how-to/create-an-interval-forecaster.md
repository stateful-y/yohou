# How to Create a Custom Interval Forecaster

This guide shows you how to implement an interval forecaster that produces
prediction intervals at specified coverage rates. Use this when you need a
custom approach to uncertainty quantification that the built-in
[`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/)
does not cover.

## Prerequisites

- Familiarity with the fit/predict API ([Getting Started](../tutorials/getting-started.md))
- Understanding of prediction intervals ([Produce Prediction Intervals](interval-forecasting.md))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## 1. Subclass `BaseIntervalForecaster`

Create a class that extends [`BaseIntervalForecaster`](/pages/api/generated/yohou.interval.base.BaseIntervalForecaster/)
and implement three things:

- **`_observation_horizon`** (property): how many recent observations the forecaster needs.
- **`_fit(self, y_t, X_t, forecasting_horizon)`**: learns whatever fitted state `_predict_one` later reads (here, per-column mean and standard deviation).
- **`_predict_one(groups, coverage_rates, **params)`**: produces a `pl.DataFrame` of interval predictions for exactly `self.fit_forecasting_horizon_` steps.

The base class `fit()` handles coverage rate validation, `_pre_fit()` setup,
and panel detection automatically. Your `_predict_one` method must return
columns following the naming convention `{target}_lower_{rate}` and
`{target}_upper_{rate}` for each coverage rate:

```python
import polars as pl
import scipy.stats as st
from yohou.interval.base import BaseIntervalForecaster
from yohou.utils.tags import Tags


class NaiveIntervalForecaster(BaseIntervalForecaster):
    """Produces intervals using historical mean and standard deviation."""

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.forecaster_tags.requires_exogenous = False
        tags.forecaster_tags.stateful = True
        return tags

    @property
    def _observation_horizon(self):
        return 10

    def _fit(self, y_t, X_t, forecasting_horizon):
        value_cols = [c for c in y_t.columns if c != "time"]
        self._stats = {}
        for col in value_cols:
            self._stats[col] = {
                "mean": y_t[col].mean(),
                "std": y_t[col].std(),
            }

    def _predict_one(self, groups, coverage_rates=None, **params):
        rates = coverage_rates or self.fit_coverage_rates_
        value_cols = list(self._stats.keys())
        h = self.fit_forecasting_horizon_

        data = {}
        for col in value_cols:
            mean = self._stats[col]["mean"]
            std = self._stats[col]["std"]
            for rate in rates:
                z = st.norm.ppf(0.5 + rate / 2)
                data[f"{col}_lower_{rate}"] = [mean - z * std] * h
                data[f"{col}_upper_{rate}"] = [mean + z * std] * h

        y_pred = pl.DataFrame(data)
        y_pred = self._add_time_columns(y_pred)
        return y_pred
```

`_predict_one` reads from `self._y_observed`, which updates when you call
`observe()`, so your forecaster stays current without refitting.

## 2. Add Constructor Parameters

If your forecaster accepts configuration, declare `_parameter_constraints` to
get automatic validation at `fit()` time:

```python
import numbers

from yohou.utils._compat import Interval


class QuantileIntervalForecaster(BaseIntervalForecaster):
    """Produces intervals from historical quantiles with a configurable window."""

    _parameter_constraints: dict = {
        **BaseIntervalForecaster._parameter_constraints,
        "window_size": [Interval(numbers.Integral, 2, None, closed="left")],
    }

    def __init__(self, window_size=30, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.forecaster_tags.requires_exogenous = False
        tags.forecaster_tags.stateful = True
        return tags

    @property
    def _observation_horizon(self):
        return self.window_size

    def _fit(self, y_t, X_t, forecasting_horizon):
        self.fitted_ = True  # At least one trailing _ attribute required

    def _predict_one(self, groups, coverage_rates=None, **params):
        rates = coverage_rates or self.fit_coverage_rates_
        y_obs = self._y_observed
        value_cols = [c for c in y_obs.columns if c != "time"]
        h = self.fit_forecasting_horizon_

        data = {}
        for col in value_cols:
            for rate in rates:
                alpha = (1 - rate) / 2
                data[f"{col}_lower_{rate}"] = [y_obs[col].quantile(alpha)] * h
                data[f"{col}_upper_{rate}"] = [y_obs[col].quantile(1 - alpha)] * h

        y_pred = pl.DataFrame(data)
        y_pred = self._add_time_columns(y_pred)
        return y_pred
```

Passing `window_size=1` now raises a validation error before any computation
starts.

## 3. Support Panel Data

If your forecaster should work with panel (multi-series) data, handle the
`groups` parameter in `_predict_one`. When panel data is present,
`self._y_observed` is a `dict[str, pl.DataFrame]` keyed by group name, and
each output column must be prefixed with `group_name__`:

```python
def _predict_one(self, groups, coverage_rates=None, **params):
    rates = coverage_rates or self.fit_coverage_rates_
    h = self.fit_forecasting_horizon_

    if self.groups_ is None:
        # Non-panel: self._y_observed is a single DataFrame
        y_obs = self._y_observed
        value_cols = [c for c in y_obs.columns if c != "time"]
        data = {}
        for col in value_cols:
            for rate in rates:
                alpha = (1 - rate) / 2
                data[f"{col}_lower_{rate}"] = [y_obs[col].quantile(alpha)] * h
                data[f"{col}_upper_{rate}"] = [y_obs[col].quantile(1 - alpha)] * h
    else:
        # Panel: self._y_observed is dict[str, pl.DataFrame]
        data = {}
        for name in groups:
            y_group = self._y_observed[name]
            value_cols = [c for c in y_group.columns if c != "time"]
            for col in value_cols:
                for rate in rates:
                    alpha = (1 - rate) / 2
                    data[f"{name}__{col}_lower_{rate}"] = [y_group[col].quantile(alpha)] * h
                    data[f"{name}__{col}_upper_{rate}"] = [y_group[col].quantile(1 - alpha)] * h

    y_pred = pl.DataFrame(data)
    y_pred = self._add_time_columns(y_pred)
    return y_pred
```

If your forecaster only makes sense for univariate, non-panel data, you can
skip the panel branch and set `supports_panel_data = False` in your tags.

## 4. Test Your Forecaster

Use the built-in check generator to validate API conformance. It runs the same
checks as point forecasters plus interval-specific ones
(`check_interval_prediction_columns`, `check_interval_bounds`,
`check_interval_prediction_types`):

```python
from conftest import run_checks
from yohou.testing import _yield_yohou_forecaster_checks


def test_naive_interval_forecaster(y_X_factory):
    from yohou.model_selection import train_test_split

    y, X = y_X_factory(length=100)
    y_train, y_test = train_test_split(y, test_size=20)

    forecaster = NaiveIntervalForecaster()
    forecaster.fit(y_train, forecasting_horizon=len(y_test), coverage_rates=[0.9, 0.95])

    run_checks(
        forecaster,
        _yield_yohou_forecaster_checks(forecaster, y_train, None, y_test, None),
    )
```

If any check fails, its name tells you exactly which contract is violated
(e.g., `check_interval_bounds`, `check_predict_time_columns`).

## See Also

- [Create a Point Forecaster](create-a-point-forecaster.md): simpler single-value forecasters
- [Create a Class-Probability Forecaster](create-a-class-proba-forecaster.md): categorical outcome forecasters
- [Create a Transformer](create-a-transformer.md): custom preprocessing and feature engineering
- [Produce Prediction Intervals](interval-forecasting.md): using the built-in `SplitConformalForecaster`
- [Create a Custom Scorer](create-a-scorer.md): implementing interval evaluation metrics with `BaseIntervalScorer`
- [Extending Yohou](../explanation/extending-yohou.md): when to extend vs compose, base class architecture
