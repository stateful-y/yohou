# How to Create a Custom Point Forecaster

This guide walks through implementing a single custom point forecaster. For
the full estimator API (all component types, anatomy, constraints, testing),
see the [Custom Estimator Reference](/pages/api/custom-estimators/). For the
design rationale behind extending vs composing, see
[Extending Yohou](/pages/explanation/extending-yohou/).

## Prerequisites

- Familiarity with the fit/predict API ([Getting Started](/pages/tutorials/getting-started/))
- Understanding of the observation horizon concept ([Core Concepts](/pages/explanation/core-concepts/))

## 1. Subclass the Base

Create a class that extends `BasePointForecaster` and implement
`_predict_one`. Override `_observation_horizon` as a property to declare
how many recent observations the forecaster needs:

```python
import polars as pl
import polars.selectors as cs
from yohou.point.base import BasePointForecaster


class LastValueForecaster(BasePointForecaster):
    """Repeats the last observed value."""

    _tags = {"ignores_exogenous": True, "stateful": True}

    @property
    def _observation_horizon(self):
        return 1

    def _predict_one(self, groups, **params):
        last_value = self._y_observed.select(~cs.by_name("time")).row(-1)[0]
        return pl.DataFrame({
            self._y_columns[0]: [last_value] * self.fit_forecasting_horizon_,
        })
```

The base `fit()` handles validation, transformer setup, panel detection, and
calls `_fit()` automatically. `_predict_one` produces raw predictions for one
forecast step, reading from `self._y_observed` so that `observe()` updates
carry through.

## 2. Add Parameters

If your forecaster has constructor parameters, declare constraints to get
automatic validation:

```python
import numbers
from yohou.utils._compat import Interval


class WindowMeanForecaster(BasePointForecaster):
    """Predicts the mean of the last `window_size` observations."""

    _tags = {"ignores_exogenous": True, "stateful": True}

    _parameter_constraints: dict = {
        "window_size": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(self, window_size=7, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size

    @property
    def _observation_horizon(self):
        return self.window_size

    def _predict_one(self, groups, **params):
        values = self._y_observed.select(~cs.by_name("time"))
        mean_val = values.select(pl.all().mean()).row(0)[0]
        return pl.DataFrame({
            self._y_columns[0]: [mean_val] * self.fit_forecasting_horizon_,
        })
```

## 3. Test Your Forecaster

Use the systematic check generator to validate API conformance:

```python
from conftest import run_checks
from yohou.testing import _yield_yohou_forecaster_checks


def test_my_forecaster(y_X_factory):
    y, X = y_X_factory(length=100)
    y_train, y_test = y[:80], y[80:]

    forecaster = WindowMeanForecaster(window_size=5)
    forecaster.fit(y_train, forecasting_horizon=len(y_test))

    run_checks(
        forecaster,
        _yield_yohou_forecaster_checks(forecaster, y_train, None, y_test),
    )
```

This runs 27 checks covering fit/predict contracts, observation/rewind behavior,
serialization, and more.

## See Also

- [Custom Estimator Reference](/pages/api/custom-estimators/): full API for all component types (transformers, scorers, interval forecasters, class-probability forecasters)
- [Create Custom Scorers](/pages/how-to/creating-a-scorer/): implementing custom evaluation metrics
- [Extending Yohou](/pages/explanation/extending-yohou/): when to extend vs compose, base class architecture
- [Extensions](/pages/reference/extensions/): official and community extensions
