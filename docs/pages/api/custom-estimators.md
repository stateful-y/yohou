# Custom Estimator Reference

Complete API reference for implementing custom Yohou components. Each component
type has a base class, methods to implement, and systematic checks.

## Component Types

| Type | Base class | Methods to implement | Use case |
|------|-----------|---------------------|----------|
| Point forecaster | `BasePointForecaster` | `_predict_one`, (`_fit`, `_observation_horizon`) | Single-value continuous or categorical predictions |
| Interval forecaster | `BaseIntervalForecaster` | `_predict_one`, (`_fit`, `_observation_horizon`) | Prediction intervals with coverage rates |
| Class-probability forecaster | `BaseClassProbaForecaster` | `_predict_one`, (`_fit`, `_observation_horizon`) | Categorical outcome probabilities |
| Transformer | `BaseTransformer` | `_fit`, `_transform`, `get_feature_names_out`, (`_inverse_transform`) | Feature engineering, preprocessing |
| Scorer | `BasePointScorer` / `BaseIntervalScorer` | `_compute_raw_errors` | Evaluation metrics |

## Point Forecaster

Tier 1 (simple) forecasters override `_observation_horizon` and `_predict_one`.
Override `_fit()` when custom fitting logic is needed. The base `fit()` handles
validation, transformer setup, panel detection, and calls `_fit()` automatically.

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

- `_observation_horizon` (property): declares the lookback window size. Return a
  value computed from constructor params (e.g. `self.seasonality`).
- `_fit(y_t, X_t, forecasting_horizon)` (optional): receives transformed data
  after `_pre_fit()` runs. Override this for model-specific fitting logic.
- `_predict_one(groups, **params)`: produces raw predictions for one
  forecast step. Read from `self._y_observed` so that `observe()` updates carry
  through. Return a DataFrame without a `"time"` column.
- Tier 2 (complex) forecasters override `fit()` directly for full control.

## Interval Forecaster

```python
from yohou.interval import BaseIntervalForecaster


class MyIntervalForecaster(BaseIntervalForecaster):
    _tags = {"ignores_exogenous": True}

    def _predict_one(self, groups, **params):
        # Fit your interval model
        # Column naming: {target}_lower_{rate}, {target}_upper_{rate}
        ...
```

The base class handles `coverage_rates` validation, storage, and calls `_fit()`
automatically. Override `_fit()` for model-specific logic. Test with
`_yield_yohou_forecaster_checks` (interval-specific checks are included
automatically).

## Class-Probability Forecaster

```python
from yohou.class_proba import BaseClassProbaForecaster


class MyClassProbaForecaster(BaseClassProbaForecaster):
    def _predict_one(self, groups, **params):
        # Return DataFrame with {target}_proba_{class} columns
        # Probabilities should sum to 1 per row
        ...
```

## Transformer

```python
import polars as pl
import polars.selectors as cs
from yohou.base import BaseTransformer


class ScaleTransformer(BaseTransformer):
    """Multiplies all numeric values by a constant."""

    _tags = {"stateful": False, "invertible": True}

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

- `_fit(X, y)` is called at the end of `fit()`. The base class handles
  validation and fitted attribute setup (`feature_names_in_`, `interval_`, etc.)
  before calling `_fit`.
- `_transform(X)` receives already-validated data with the `"time"` column.
  Return a DataFrame with `"time"` preserved.
- `_inverse_transform(X_t, X_p)` reverses the transformation. `X_p` provides
  warmup rows for stateful transformers. Set `_tags = {"invertible": True}` when
  implementing this.

For stateful transformers that need a lookback window:

```python
class MyWindowTransformer(BaseTransformer):
    _tags = {"stateful": True}

    _parameter_constraints: dict = {
        "window_size": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(self, window_size=5):
        self.window_size = window_size

    @property
    def observation_horizon(self):
        return self.window_size
```

## Scorer

Scorers implement `_compute_raw_errors()` which receives aligned DataFrames
(time column already removed) and returns per-timestep per-component raw
scores. The base class `score()` method handles validation, time weighting,
aggregation, post-aggregate transforms, and column renaming automatically:

```python
import polars as pl
from yohou.metrics.base import BasePointScorer


class MaxAbsoluteError(BasePointScorer):
    """Maximum absolute error across the forecast horizon."""

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
    }

    _metric_name = "max_ae"

    def __init__(
        self,
        aggregation_method="all",
        groups=None,
        components=None,
    ):
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
        )

    def _compute_raw_errors(self, y_truth, y_pred):
        return (y_truth - y_pred).select(pl.all().abs())
```

- Raw scores must be a DataFrame without a `"time"` column.
- `_aggregate_scores` applies the `aggregation_method` strategy (stepwise,
  vintagewise, componentwise, groupwise, all).
- Override `_post_aggregate` for transforms after aggregation (e.g. sqrt for
  RMSE).

For interval scorers, extend `BaseIntervalScorer` instead. It adds
`coverage_rates` and `"coveragewise"` aggregation.

For the full walkthrough, see
[How to Create a Custom Scorer](/pages/how-to/create-a-scorer/).

## Anatomy of an Estimator

### `__init__`

Follow scikit-learn conventions: every parameter must be stored as an attribute
with the same name. Forward shared parameters (`target_transformer`,
`feature_transformer`, `panel_strategy`) to the parent:

```python
def __init__(self, my_param=1, **kwargs):
    super().__init__(**kwargs)
    self.my_param = my_param
```

### `_tags`

Declare tags as a class-level dictionary. The base class merges `_tags` from
all classes in the MRO (most-derived wins):

```python
class MyForecaster(BasePointForecaster):
    _tags = {"ignores_exogenous": True, "stateful": True}
```

Common tag keys for forecasters: `ignores_exogenous`, `stateful`,
`forecaster_type`, `uses_reduction`, `supports_panel_data`.

Common tag keys for transformers: `stateful`, `invertible`.

For advanced cases (dynamic tags based on constructor params or child
estimators), override `__sklearn_tags__()` instead:

```python
def __sklearn_tags__(self):
    tags = super().__sklearn_tags__()
    tags.forecaster_tags.ignores_exogenous = self.my_param > 0
    return tags
```

### `_parameter_constraints`

Define only your own constraints. Parent constraints are merged automatically
via `__init_subclass__`:

```python
_parameter_constraints: dict = {
    "my_param": [Interval(numbers.Integral, 1, None, closed="left")],
    "mode": [StrOptions({"fast", "accurate"})],
}
```

Constraint types: `Interval`, `StrOptions`, `HasMethods`, type classes
(`int`, `float`, `str`), `None`, `"boolean"`, `callable`, `"no_validation"`.

### `observation_horizon`

Declare as a `@property` that computes from constructor params.
Returns 0 for stateless estimators (the default):

```python
@property
def observation_horizon(self):
    return self.seasonality
```

The base class uses this to manage internal memory buffers for `observe()`
and `rewind()` operations.

## Testing

Yohou provides systematic check generators that validate API conformance:

```python
from conftest import run_checks
from yohou.testing import _yield_yohou_forecaster_checks


def test_my_forecaster(y_X_factory):
    from yohou.model_selection import train_test_split

    y, X = y_X_factory(length=100)
    y_train, y_test = train_test_split(y, test_size=20)

    forecaster = MyForecaster()
    forecaster.fit(y_train, forecasting_horizon=len(y_test))

    run_checks(
        forecaster,
        _yield_yohou_forecaster_checks(forecaster, y_train, None, y_test),
    )
```

### Check Generators

| Generator | Checks | Component |
|-----------|--------|-----------|
| `_yield_yohou_forecaster_checks` | 27 | Forecasters (point, interval, class-probability) |
| `_yield_yohou_transformer_checks` | 26 | Transformers |
| `_yield_yohou_scorer_checks` | 11 | Scorers |
| `_yield_yohou_splitter_checks` | 8 | Splitters |

### Skipping Checks

Skip checks that don't apply with `expected_failures`:

```python
run_checks(
    transformer,
    _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
    expected_failures={"check_inverse_transform_identity"},
)
```

## Panel Data

The base class handles panel dispatch automatically when columns use the
`group__column` naming convention. Your `_predict_one` receives
`groups` indicating which groups to predict for:

```python
def _predict_one(self, groups, **params):
    if self.groups_ is None:
        # Non-panel: self._y_observed is a pl.DataFrame
        ...
    else:
        # Panel: self._y_observed is a dict[str, pl.DataFrame]
        for group in groups:
            group_data = self._y_observed[group]
            ...
```

## Using `_pre_fit()` Directly

For complex estimators that need full control over the fit process (reduction
forecasters, ensemble methods, pipeline composites), call `_pre_fit()` instead
of `super().fit()`:

```python
class MyComplexForecaster(BasePointForecaster):
    def fit(self, y, X_actual=None, forecasting_horizon=1, **params):
        self._pre_fit(y, X_actual, forecasting_horizon)
        # Custom fitting logic...
        return self
```

Call `self._pre_fit(y, X_actual, forecasting_horizon)` to run validation, transformer
setup, and panel detection.

## See Also

- [How to Create a Custom Point Forecaster](/pages/how-to/create-a-point-forecaster/) for a focused walkthrough
- [Extending Yohou](/pages/explanation/extending-yohou/) for design rationale
- [Extensions](/pages/reference/extensions/) for official and community extensions
