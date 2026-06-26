# How to Create a Custom Point Forecaster

This guide shows you how to implement a point forecaster that plugs into
Yohou's fit/predict/observe lifecycle. Use this when the built-in forecasters
do not cover your modelling approach and you need full control over prediction
logic.

## Prerequisites

- Familiarity with the fit/predict API ([Getting Started](../tutorials/getting-started.md))
- Understanding of the observation horizon concept ([Core Concepts](../explanation/core-concepts.md))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## 1. Subclass `BasePointForecaster`

Create a class that extends [`BasePointForecaster`](/pages/api/generated/yohou.point.base.BasePointForecaster/) and implement
two things:

- **`_observation_horizon`** (property): how many recent observations the forecaster needs.
- **`_predict_one(groups, **params)`**: produces a `pl.DataFrame` of predictions for exactly `self.fit_forecasting_horizon_` steps.

```python
import polars as pl
import polars.selectors as cs
from yohou.point.base import BasePointForecaster
from yohou.utils.tags import Tags


class LastValueForecaster(BasePointForecaster):
    """Repeats the last observed value for every forecast step."""

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.forecaster_tags.requires_exogenous = False
        tags.forecaster_tags.stateful = True
        return tags

    @property
    def _observation_horizon(self):
        return 1

    def _predict_one(self, groups, **params):
        y_obs = self._y_observed  # pl.DataFrame (non-panel)
        last_row = y_obs.select(~cs.by_name("time")).row(-1)
        cols = [c for c in y_obs.columns if c != "time"]
        y_pred = pl.DataFrame(
            {col: [val] * self.fit_forecasting_horizon_ for col, val in zip(cols, last_row)}
        )
        y_pred = self._add_time_columns(y_pred)
        return y_pred
```

The base class `fit()` handles validation, transformer setup, and panel
detection automatically. `_predict_one` reads from `self._y_observed`, which
updates when you call `observe()`, so your forecaster stays current without
extra bookkeeping.

## 2. Add Constructor Parameters

If your forecaster accepts configuration, declare `_parameter_constraints` to
get automatic validation at `fit()` time:

```python
import numbers

from yohou.utils._compat import Interval


class WindowMeanForecaster(BasePointForecaster):
    """Predicts the mean of the last ``window_size`` observations."""

    _parameter_constraints: dict = {
        **BasePointForecaster._parameter_constraints,
        "window_size": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(self, window_size=7, **kwargs):
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

    def _predict_one(self, groups, **params):
        values = self._y_observed.select(~cs.by_name("time"))
        cols = values.columns
        mean_vals = values.select(pl.all().mean()).row(0)
        y_pred = pl.DataFrame(
            {col: [val] * self.fit_forecasting_horizon_ for col, val in zip(cols, mean_vals)}
        )
        y_pred = self._add_time_columns(y_pred)
        return y_pred
```

Passing `window_size=0` now raises a validation error before any computation
starts.

## 3. Support Panel Data

If your forecaster should work with panel (multi-series) data, handle the
`groups` parameter in `_predict_one`. When panel data is present,
`self._y_observed` is a `dict[str, pl.DataFrame]` keyed by group name, and
each output column must be prefixed with `group_name__`:

```python
def _predict_one(self, groups, **params):
    if self.groups_ is None:
        # Non-panel: self._y_observed is a single DataFrame
        values = self._y_observed.select(~cs.by_name("time"))
        mean_vals = values.select(pl.all().mean()).row(0)
        y_pred = pl.DataFrame(
            {col: [val] * self.fit_forecasting_horizon_ for col, val in zip(values.columns, mean_vals)}
        )
    else:
        # Panel: self._y_observed is dict[str, pl.DataFrame]
        parts = []
        for name in groups:
            y_group = self._y_observed[name].select(~cs.by_name("time"))
            mean_vals = y_group.select(pl.all().mean()).row(0)
            part = pl.DataFrame(
                {f"{name}__{col}": [val] * self.fit_forecasting_horizon_
                 for col, val in zip(y_group.columns, mean_vals)}
            )
            parts.append(part)
        y_pred = pl.concat(parts, how="horizontal")

    y_pred = self._add_time_columns(y_pred)
    return y_pred
```

If your forecaster only makes sense for univariate, non-panel data, you can
skip the panel branch and set `supports_panel_data = False` in your tags.

## 4. Test Your Forecaster

Use the built-in check generator to validate API conformance. It runs 27
checks covering fit/predict contracts, observation/rewind behaviour,
serialization, and panel data handling:

```python
from conftest import run_checks
from yohou.testing import _yield_yohou_forecaster_checks


def test_window_mean_forecaster(y_X_factory):
    from yohou.model_selection import train_test_split

    y, X = y_X_factory(length=100)
    y_train, y_test = train_test_split(y, test_size=20)

    forecaster = WindowMeanForecaster(window_size=5)
    forecaster.fit(y_train, forecasting_horizon=len(y_test))

    run_checks(
        forecaster,
        _yield_yohou_forecaster_checks(forecaster, y_train, None, y_test, None),
    )
```

!!! note
    `run_checks` is an internal utility from Yohou's own `conftest.py`. If you are building a forecaster in an external package, iterate over the checks directly:

    ```python
    for check in _yield_yohou_forecaster_checks(forecaster, y_train, None, y_test, None):
        check(forecaster)
    ```

If any check fails, its name tells you exactly which contract is violated
(e.g., `check_predict_time_columns`, `check_observe_extends_observations`).

## See Also

- [Create an Interval Forecaster](create-an-interval-forecaster.md): prediction interval forecasters
- [Create a Class-Probability Forecaster](create-a-class-proba-forecaster.md): categorical outcome forecasters
- [Create a Transformer](create-a-transformer.md): custom preprocessing and feature engineering
- [Create a Custom Scorer](create-a-scorer.md): custom evaluation metrics
- [Extending Yohou](../explanation/extending-yohou.md): when to extend vs compose, base class architecture
