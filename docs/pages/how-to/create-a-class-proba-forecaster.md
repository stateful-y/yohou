# How to Create a Custom Class-Probability Forecaster

This guide shows you how to implement a forecaster that predicts per-class
probabilities for categorical time series. Use this when you need a custom
classification approach that the built-in
[`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.ClassProbaReductionForecaster/)
does not cover.

## Prerequisites

- Familiarity with the fit/predict API ([Getting Started](../tutorials/getting-started.md))
- Understanding of class-probability forecasting ([Forecast with Class Probabilities](class-probability-forecasting.md))

<!-- COMPANION_NOTEBOOKS -->

## 1. Subclass `BaseClassProbaForecaster`

Create a class that extends [`BaseClassProbaForecaster`](/pages/api/generated/yohou.class_proba.BaseClassProbaForecaster/)
and implement three things:

- **`_observation_horizon`** (property): how many recent observations the forecaster needs.
- **`_fit(y_t, X_t, forecasting_horizon)`**: discover and store class metadata in `self.classes_`, `self.n_classes_`, and `self.label_to_code_`.
- **`_predict_class_proba_one(groups, **params)`**: produce a `pl.DataFrame` with columns named `{target}_proba_{class_label}` where probabilities sum to 1.0 per row.

```python
import polars as pl
from yohou.class_proba.base import BaseClassProbaForecaster
from yohou.utils.tags import Tags


class MajorityClassForecaster(BaseClassProbaForecaster):
    """Predicts the training-set class distribution at every step."""

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.forecaster_tags.requires_exogenous = False
        tags.forecaster_tags.stateful = True
        return tags

    @property
    def _observation_horizon(self):
        return 1

    def _fit(self, y_t, X_t, forecasting_horizon):
        value_cols = [c for c in y_t.columns if c != "time"]
        self.classes_ = {}
        self.n_classes_ = {}
        self.label_to_code_ = {}
        self._class_probs = {}

        for col in value_cols:
            labels = sorted(y_t[col].unique().to_list())
            self.classes_[col] = labels
            self.n_classes_[col] = len(labels)
            self.label_to_code_[col] = {label: i for i, label in enumerate(labels)}

            counts = y_t[col].value_counts()
            total = len(y_t)
            self._class_probs[col] = {
                row[col]: row["count"] / total for row in counts.iter_rows(named=True)
            }

    def _predict_class_proba_one(self, groups, **params):
        h = self.fit_forecasting_horizon_
        data = {}
        for col, probs in self._class_probs.items():
            for label in self.classes_[col]:
                prob = probs.get(label, 0.0)
                data[f"{col}_proba_{label}"] = [prob] * h

        y_pred = pl.DataFrame(data)
        y_pred = self._add_time_columns(y_pred)
        return y_pred
```

The base class `predict_class_proba()` handles validation, panel dispatch,
and calls `_predict_class_proba_one` automatically. The separate `predict()`
method returns the argmax class label.

!!! note "Required fitted attributes"
    The three dictionaries (`classes_`, `n_classes_`, `label_to_code_`) must
    be populated in `_fit`. They are validated by the automatic checks and
    used by `predict()` to map probabilities back to class labels.

## 2. Add Constructor Parameters

If your forecaster accepts configuration, declare `_parameter_constraints` to
get automatic validation at `fit()` time:

```python
import numbers

import polars as pl
from yohou.class_proba.base import BaseClassProbaForecaster
from yohou.utils._compat import Interval
from yohou.utils.tags import Tags


class SmoothedClassForecaster(BaseClassProbaForecaster):
    """Class forecaster with Laplace smoothing."""

    _parameter_constraints: dict = {
        **BaseClassProbaForecaster._parameter_constraints,
        "alpha": [Interval(numbers.Real, 0, None, closed="left")],
    }

    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.forecaster_tags.requires_exogenous = False
        tags.forecaster_tags.stateful = True
        return tags

    @property
    def _observation_horizon(self):
        return 1

    def _fit(self, y_t, X_t, forecasting_horizon):
        value_cols = [c for c in y_t.columns if c != "time"]
        self.classes_ = {}
        self.n_classes_ = {}
        self.label_to_code_ = {}
        self._class_probs = {}

        for col in value_cols:
            labels = sorted(y_t[col].unique().to_list())
            self.classes_[col] = labels
            self.n_classes_[col] = len(labels)
            self.label_to_code_[col] = {label: i for i, label in enumerate(labels)}

            counts = y_t[col].value_counts()
            count_map = {row[col]: row["count"] for row in counts.iter_rows(named=True)}
            total = len(y_t) + self.alpha * len(labels)
            self._class_probs[col] = {
                label: (count_map.get(label, 0) + self.alpha) / total
                for label in labels
            }

    def _predict_class_proba_one(self, groups, **params):
        h = self.fit_forecasting_horizon_
        data = {}
        for col, probs in self._class_probs.items():
            for label in self.classes_[col]:
                data[f"{col}_proba_{label}"] = [probs[label]] * h

        y_pred = pl.DataFrame(data)
        y_pred = self._add_time_columns(y_pred)
        return y_pred
```

Passing `alpha=-1` now raises a validation error before any computation
starts.

## 3. Support Panel Data

If your forecaster should work with panel (multi-series) data, handle the
`groups` parameter in `_predict_class_proba_one`. When panel data is present,
`self._y_observed` is a `dict[str, pl.DataFrame]` keyed by group name, and
each output column must be prefixed with `group_name__`:

```python
def _predict_class_proba_one(self, groups, **params):
    h = self.fit_forecasting_horizon_

    if self.groups_ is None:
        # Non-panel path
        data = {}
        for col, probs in self._class_probs.items():
            for label in self.classes_[col]:
                data[f"{col}_proba_{label}"] = [probs[label]] * h
        y_pred = pl.DataFrame(data)
    else:
        # Panel path: prefix each column with the group name
        parts = []
        for name in groups:
            group_data = {}
            for col, probs in self._class_probs.items():
                for label in self.classes_[col]:
                    group_data[f"{name}__{col}_proba_{label}"] = [probs[label]] * h
            parts.append(pl.DataFrame(group_data))
        y_pred = pl.concat(parts, how="horizontal")

    y_pred = self._add_time_columns(y_pred)
    return y_pred
```

If your forecaster only makes sense for non-panel data, you can skip the
panel branch and set `supports_panel_data = False` in your tags.

## 4. Test Your Forecaster

Use the built-in check generator to validate API conformance. It runs the
common forecaster checks plus six class-probability-specific checks covering
prediction structure, probability bounds and sums, fitted attributes, and
argmax label output:

```python
from conftest import run_checks
from yohou.testing import _yield_yohou_forecaster_checks


def test_majority_class_forecaster(class_proba_y_X_factory):
    from yohou.model_selection import train_test_split

    y, X_actual = class_proba_y_X_factory(length=100, n_targets=1, n_classes=3)
    y_train, y_test = train_test_split(y, test_size=20)

    forecaster = MajorityClassForecaster()
    forecaster.fit(y_train, forecasting_horizon=len(y_test))

    run_checks(
        forecaster,
        _yield_yohou_forecaster_checks(forecaster, y_train, None, y_test, None),
    )
```

If any check fails, its name tells you exactly which contract is violated
(e.g., `check_class_proba_prediction_bounds`, `check_class_proba_classes_attribute`).

## See Also

- [Forecast with Class Probabilities](class-probability-forecasting.md): using the built-in `ClassProbaReductionForecaster`
- [Create a Point Forecaster](create-a-point-forecaster.md): continuous value forecasters
- [Create an Interval Forecaster](create-an-interval-forecaster.md): prediction interval forecasters
- [Create a Transformer](create-a-transformer.md): custom preprocessing and feature engineering
- [Create a Custom Scorer](create-a-scorer.md): custom evaluation metrics
- [Extending Yohou](../explanation/extending-yohou.md): when to extend vs compose
