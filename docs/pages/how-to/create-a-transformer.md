# How to Create a Custom Transformer

This guide shows you how to implement a time series transformer that plugs
into Yohou's fit/transform/observe lifecycle. Use this when the built-in
preprocessing transformers do not cover your feature engineering or data
transformation needs.

## Prerequisites

- Yohou installed ([Getting Started](../tutorials/getting-started.md))
- Familiarity with Yohou transformers ([Use Preprocessing Transformers](use-preprocessing-transformers.md))
- Understanding of the observation horizon concept ([Core Concepts](../explanation/core-concepts.md))

!!! tip "Try it interactively"
    <!-- COMPANION_NOTEBOOKS -->

## 1. Subclass `BaseActualTransformer`

Create a class that extends [`BaseActualTransformer`](/pages/api/generated/yohou.base.transformer.BaseActualTransformer/)
and implement three methods:

- **`_fit(X, y=None)`**: store any state computed from the training data.
  Must set at least one fitted attribute (name ending in `_`).
- **`_transform(X)`**: return a transformed `pl.DataFrame`.
  Always preserve the `"time"` column.
- **`get_feature_names_out(input_features=None)`**: return the list of output
  column names (excluding `"time"`).

```python
import polars as pl
import polars.selectors as cs
from yohou.base import BaseActualTransformer


class ScaleTransformer(BaseActualTransformer):
    """Multiplies all numeric columns by a constant factor."""

    _parameter_constraints: dict = {
        "factor": [float, int],
    }

    def __init__(self, factor=2.0):
        self.factor = factor

    def _fit(self, X, y=None):
        self.columns_ = [c for c in X.columns if c != "time"]

    def _transform(self, X):
        return X.with_columns(cs.numeric() * self.factor)

    def get_feature_names_out(self, input_features=None):
        return self.columns_ if input_features is None else input_features
```

You only implement the underscore hooks (`_fit`, `_transform`). The base class
`fit()` handles input validation, sets `feature_names_in_`, `n_features_in_`,
`X_schema_`, and `interval_`, then delegates to your `_fit()`. Similarly,
`transform()` validates and delegates to `_transform()`.

`_parameter_constraints` declares validation rules for constructor parameters.
Constraints from parent classes are merged automatically, so you only need to
declare constraints for parameters you introduce.

## 2. Add Inverse Transform

If your transformation is reversible, implement `_inverse_transform` and
tag the transformer as invertible. This lets forecasters automatically
undo target transformations when producing predictions:

```python
class ScaleTransformer(BaseActualTransformer):
    """Multiplies all numeric columns by a constant factor."""

    _tags = {"invertible": True}

    _parameter_constraints: dict = {
        "factor": [float, int],
    }

    def __init__(self, factor=2.0):
        self.factor = factor

    def _fit(self, X, y=None):
        self.columns_ = [c for c in X.columns if c != "time"]

    def _transform(self, X):
        return X.with_columns(cs.numeric() * self.factor)

    def _inverse_transform(self, X_t, X_p=None):
        return X_t.with_columns(cs.numeric() / self.factor)

    def get_feature_names_out(self, input_features=None):
        return self.columns_ if input_features is None else input_features
```

`_inverse_transform` receives `X_t` (the transformed data) and optionally
`X_p` (past observations, useful for stateful transformers that need warmup
rows to reverse the operation).

If your transformation is not invertible, skip this step.

## 3. Make It Stateful

Stateful transformers maintain a memory buffer of recent observations. This is
needed when the transformation depends on a lookback window (lags, rolling
statistics, filters). Tag the transformer as stateful and override the
`observation_horizon` property to declare how many past rows are required:

```python
import numbers

import polars as pl
import polars.selectors as cs
from yohou.base import BaseActualTransformer
from yohou.utils._compat import Interval


class RollingMeanTransformer(BaseActualTransformer):
    """Replaces each value with the rolling mean over the last ``window`` steps."""

    _tags = {"stateful": True}

    _parameter_constraints: dict = {
        "window": [Interval(numbers.Integral, 2, None, closed="left")],
    }

    def __init__(self, window=5):
        self.window = window

    @property
    def observation_horizon(self):
        return self.window - 1

    def _fit(self, X, y=None):
        self.columns_ = [c for c in X.columns if c != "time"]

    def _transform(self, X):
        return X.with_columns(
            cs.numeric().rolling_mean(window_size=self.window)
        ).tail(len(X) - (self.window - 1))

    def get_feature_names_out(self, input_features=None):
        return self.columns_ if input_features is None else input_features
```

The `observation_horizon` value tells the framework how many warmup rows the
transform needs. A `rolling_mean(window_size=N)` produces `N - 1` leading
nulls, so the convention (matching `RollingStatisticsTransformer`) is
`observation_horizon = window - 1`. The first `observation_horizon` rows of
`_transform()` output are consumed by the rolling window and dropped via
`.tail()`.

The base class uses `observation_horizon` to manage the internal memory buffer
(`self._X_observed`). After fitting, `observe()` appends new data and trims the
buffer to `observation_horizon` rows, while `rewind()` replaces the buffer
entirely. This lifecycle is handled automatically when the transformer is used
inside a pipeline or forecaster.

For stateless transformers (no lookback needed), `observation_horizon` defaults
to 0 and you can skip this step.

## 4. Add Output Columns

If your transformer produces additional columns (e.g., lag features, calendar
features), reflect this in `get_feature_names_out`. The output must match the
actual columns produced by `_transform` (excluding `"time"`):

```python
class SimpleLagTransformer(BaseActualTransformer):
    """Creates a single lag-1 feature for each input column."""

    _tags = {"stateful": True}

    def __init__(self):
        pass

    @property
    def observation_horizon(self):
        return 1

    def _fit(self, X, y=None):
        self.columns_ = [c for c in X.columns if c != "time"]

    def _transform(self, X):
        lag_cols = {f"{col}_lag_1": X[col].shift(1) for col in self.columns_}
        result = X.with_columns(**lag_cols)
        return result.tail(len(result) - 1)  # drop first row (null from shift)

    def get_feature_names_out(self, input_features=None):
        cols = input_features if input_features is not None else self.columns_
        return list(cols) + [f"{c}_lag_1" for c in cols]
```

If your transformer preserves the same columns, `get_feature_names_out` can
simply return `input_features` or the stored `columns_` as shown in the earlier
examples.

## 5. Handle Panel Data

Transformers automatically support panel data (columns prefixed with
`group_name__`). The base class detects panel structure and your `_transform`
receives the full DataFrame with prefixed columns.

If your transformation logic is column-agnostic (operating on all numeric
columns via `cs.numeric()`), panel data works without additional code. If you
reference specific column names, make sure to handle the prefixed variants.

If your transformer only makes sense for non-panel data, override
`__sklearn_tags__` to declare this:

```python
from yohou.utils.tags import Tags


class MyUnivariateTransformer(BaseActualTransformer):
    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.transformer_tags.supports_panel_data = False
        return tags
```

## 6. Test Your Transformer

Use `_yield_yohou_transformer_checks` to validate API conformance. It runs
26 checks covering fit/transform contracts, observation/rewind behaviour,
inverse transforms, feature names, and panel data:

```python
from sklearn.base import clone

from conftest import run_checks
from yohou.testing import _yield_yohou_transformer_checks


class TestScaleTransformer:
    def test_systematic_checks(self, time_series_train_test_factory):
        X_train, X_test = time_series_train_test_factory(
            train_length=100, test_length=30
        )

        transformer = ScaleTransformer(factor=3.0)
        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(
                transformer_fitted, X_train, None, X_test
            ),
        )
```

If a check does not apply (e.g., `check_inverse_transform_identity` for a
non-invertible transformer), pass it as an expected failure:

```python
run_checks(
    transformer_fitted,
    _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
    expected_failures={"check_inverse_transform_identity"},
)
```

If any check fails, its name tells you exactly which contract is violated.
Add your own tests for numerical correctness alongside the generated checks.

## See Also

- [Use Preprocessing Transformers](use-preprocessing-transformers.md): using the built-in transformers
- [Compose Feature Pipelines](compose-feature-pipelines.md): combining transformers into pipelines
- [Create a Point Forecaster](create-a-point-forecaster.md): using transformers inside forecasters via `target_transformer` and `feature_transformer`
- [Create an Interval Forecaster](create-an-interval-forecaster.md): prediction interval forecasters
- [Create a Class-Probability Forecaster](create-a-class-proba-forecaster.md): categorical outcome forecasters
- [Create a Custom Scorer](create-a-scorer.md): custom evaluation metrics
- [Extending Yohou](../explanation/extending-yohou.md): when to extend vs compose
